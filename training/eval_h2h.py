"""
Head-to-head evaluation between two checkpoints.
Runs the game engine directly in-process (no API servers needed).

Usage:
    python training/eval_h2h.py training/data/checkpoints/3M_iters/strategy_cpp.bin training/data/checkpoints/8M_iters/strategy_cpp.bin --matches 5
"""

import argparse
import os
import sys
import struct
import pickle
import random
import logging
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "submission"))

from gym_env import PokerEnv
from abstractions.discard_oracle import choose_discard
from abstractions.action_abs import get_valid_abstract_actions, abstract_to_concrete, action_to_short, get_action_context
from abstractions.card_utils import card_rank, canonicalize_suits, ACE_RANK_IDX, get_evaluator, int_to_treys, ALL_CARDS
from abstractions.board_texture import board_bucket_for_street
from abstractions.hand_bucket import hand_bucket_for_street
from abstractions.opp_discard_bucket import opp_discard_bucket
from abstractions.public_state import line_bucket, pressure_bucket
from abstractions.discard_oracle import estimate_opp_keep_weights

_FOLD = 0
_RAISE = 1
_CHECK = 2
_CALL = 3
_DISCARD = 4
_CTX_MAP = {"no_bet": 0, "facing_bet": 1, "high_pressure": 2}


def _fnv_hash(data):
    h = 14695981039346656037
    for b in data:
        h ^= b
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def convert_bin(bin_path):
    """Convert binary to numpy arrays in memory. Returns (key_to_idx, probs, act_types, action_lists)."""
    with open(bin_path, 'rb') as f:
        data = f.read()
    offset = 0
    iters, nodes = struct.unpack_from('<II', data, offset); offset += 8
    per_node = (len(data) - 8) / nodes if nodes > 0 else 42
    has_conf = (per_node >= 49.5)

    ACTION_LISTS = [('FOLD','CALL','JAM'),('FOLD','CALL'),('FOLD','CALL','RAISE_SMALL','RAISE_LARGE'),
                    ('CHECK','BET_SMALL','BET_LARGE'),('CHECK',)]
    MAX_ACTIONS = 4

    keys = np.zeros(nodes, dtype=np.uint64)
    act_types = np.zeros(nodes, dtype=np.uint8)
    probs = np.zeros((nodes, MAX_ACTIONS), dtype=np.uint8)

    confidence = np.zeros(nodes, dtype=np.float32) if has_conf else None

    for i in range(nodes):
        key = struct.unpack_from('<Q', data, offset)[0]; offset += 8
        atype = struct.unpack_from('<B', data, offset)[0]; offset += 1
        nact = struct.unpack_from('<B', data, offset)[0]; offset += 1
        avg = struct.unpack_from(f'<{MAX_ACTIONS}d', data, offset); offset += MAX_ACTIONS * 8
        if has_conf:
            conf_val = struct.unpack_from('<d', data, offset)[0]; offset += 8
            confidence[i] = conf_val
        keys[i] = key; act_types[i] = atype; nact = min(nact, MAX_ACTIONS)
        if nact == 0: continue
        raw = list(avg[:nact]); total = sum(raw)
        if total > 0: raw = [p / total for p in raw]
        else: raw = [1.0 / nact] * nact
        for j in range(nact):
            probs[i, j] = max(0, min(255, int(round(raw[j] * 255))))
        rs = sum(probs[i, :nact])
        if rs > 0 and rs != 255:
            mx = np.argmax(probs[i, :nact]); probs[i, mx] = max(0, min(255, probs[i, mx] + (255 - rs)))

    key_to_idx = {int(keys[i]): i for i in range(nodes)}
    return key_to_idx, probs, act_types, ACTION_LISTS, confidence, iters, nodes


CONFIDENCE_THRESHOLD = 300000000.0


def _mc_choose_discard(hand_5, board_3, opp_discards=None, top_k=3, mc_sims=150):
    """Old MC-based discard for comparison."""
    from abstractions.discard_oracle import _fast_score, KEEP_PAIRS
    dead = [c for c in (opp_discards or []) if c >= 0]

    candidates = []
    for i, j in KEEP_PAIRS:
        keep = [hand_5[i], hand_5[j]]
        discarded = [hand_5[k] for k in range(5) if k != i and k != j]
        fast = _fast_score(keep, board_3)
        candidates.append((fast, i, j, keep, discarded))
    candidates.sort(key=lambda x: -x[0])

    ev = get_evaluator()
    best_score = -1.0
    best_keep = (candidates[0][1], candidates[0][2])

    for fast, i, j, keep, discarded in candidates[:top_k]:
        all_dead = dead + discarded
        used = set(keep) | set(board_3) | set(all_dead)
        remaining = [c for c in ALL_CARDS if c not in used]
        if len(remaining) < 4:
            continue
        my_h = [int_to_treys(c) for c in keep]
        wins = ties = total = 0
        for _ in range(mc_sims):
            sample = random.sample(remaining, 4)
            b = [int_to_treys(c) for c in board_3 + sample[:2]]
            oh = [int_to_treys(c) for c in sample[2:4]]
            mr = ev.evaluate(my_h, b)
            opr = ev.evaluate(oh, b)
            if mr < opr: wins += 1
            elif mr == opr: ties += 1
            total += 1
        eq = (wins + 0.5*ties) / total if total > 0 else 0.5
        score = 0.7 * (eq * 10) + 0.3 * fast
        if score > best_score:
            best_score = score
            best_keep = (i, j)
    return best_keep


class SimpleAgent:
    """Lightweight agent that plays using a specific strategy table."""

    def __init__(self, name, key_to_idx, probs, act_types, action_lists, confidence=None, use_exact_discard=True, use_subgame_sizing=False):
        self.name = name
        self.key_to_idx = key_to_idx
        self.probs = probs
        self.act_types = act_types
        self.action_lists = action_lists
        self.confidence = confidence
        self.use_exact_discard = use_exact_discard
        self.use_subgame_sizing = use_subgame_sizing
        self.reset()

    def reset(self):
        self.street_history = ""
        self.last_street = -1
        self.hero_last_raiser = False
        self.villain_last_raiser = False
        self.my_hand_5 = []
        self.my_hand_2 = []
        self.my_discards = []
        self.opp_discards = []

    def _check_street(self, street):
        if street != self.last_street:
            self.street_history = ""
            self.last_street = street

    def _make_preflop_key(self, hand5, is_bb):
        canon = canonicalize_suits(tuple(sorted(hand5)))
        buf = bytearray()
        buf.append(0)
        buf.append(1 if is_bb else 0)
        buf.append(line_bucket(self.street_history))
        for c in canon: buf.append(c)
        return _fnv_hash(bytes(buf))

    def _make_postdiscard_key(self, street, hand2, community, opp_disc, is_bb, my_bet, opp_bet, n):
        action_ctx = get_action_context([1,1,1,1,0], my_bet, opp_bet, 100 - max(my_bet, opp_bet))
        dead = list(self.my_discards) + list(self.opp_discards)
        buf = bytearray()
        buf.append(street)
        buf.append(1 if is_bb else 0)
        buf.append(1 if self.hero_last_raiser else (2 if self.villain_last_raiser else 0))
        buf.append(line_bucket(self.street_history))
        buf.append(pressure_bucket(my_bet, opp_bet))
        buf.extend(struct.pack('<H', board_bucket_for_street(community, street)))
        board3 = community[:3]
        buf.append(opp_discard_bucket(opp_disc, board3) & 0xFF)
        buf.append(hand_bucket_for_street(hand2, community, street, dead) & 0xFF)
        buf.append(_CTX_MAP.get(action_ctx, 0))
        buf.append(n)
        return _fnv_hash(bytes(buf))

    def _cfr_action(self, obs):
        """Try CFR lookup, return (concrete_action, confidence) or (None, 0)."""
        my_cards = [c for c in obs["my_cards"] if c != -1]
        community = [c for c in obs["community_cards"] if c != -1]
        is_bb = obs.get("blind_position", 0) == 1
        street = obs["street"]
        valid = obs["valid_actions"]
        my_bet, opp_bet = obs["my_bet"], obs["opp_bet"]
        min_raise, max_raise = obs["min_raise"], obs["max_raise"]

        valid_abs = get_valid_abstract_actions(valid, my_bet, opp_bet, min_raise, max_raise)
        n = len(valid_abs)

        if street == 0:
            h = self._make_preflop_key(self.my_hand_5 or my_cards, is_bb)
        else:
            hand2 = self.my_hand_2 or my_cards[:2]
            opp_disc = self.opp_discards if self.opp_discards else [-1,-1,-1]
            h = self._make_postdiscard_key(street, hand2, community, opp_disc, is_bb, my_bet, opp_bet, n)

        idx = self.key_to_idx.get(h)
        if idx is None:
            return None, 0.0

        stored_type = self.act_types[idx]
        if stored_type >= len(self.action_lists):
            return None, 0.0
        stored_actions = list(self.action_lists[stored_type])
        if len(stored_actions) != n or stored_actions != valid_abs:
            return None, 0.0

        # Confidence
        conf = 0.0
        if self.confidence is not None and idx < len(self.confidence):
            conf = min(float(self.confidence[idx]) / CONFIDENCE_THRESHOLD, 1.0)

        raw_probs = self.probs[idx, :n].astype(np.float32)
        total = raw_probs.sum()
        probs = raw_probs / total if total > 0 else np.ones(n) / n

        chosen_idx = random.choices(range(n), weights=probs.tolist(), k=1)[0]
        chosen_abs = valid_abs[chosen_idx]

        concrete = abstract_to_concrete(chosen_abs, int(min_raise), int(max_raise), int(my_bet), int(opp_bet))
        return concrete, conf, chosen_abs

    def _fallback_action(self, obs):
        """GTO-balanced equity fallback."""
        valid = obs["valid_actions"]
        my_cards = [c for c in obs["my_cards"] if c != -1]
        community = [c for c in obs["community_cards"] if c != -1]
        my_bet, opp_bet = obs["my_bet"], obs["opp_bet"]
        min_raise, max_raise = int(obs["min_raise"]), int(obs["max_raise"])
        hand = self.my_hand_2 if self.my_hand_2 else my_cards[:2]

        to_call = opp_bet - my_bet
        pot = my_bet + opp_bet
        pot_odds = to_call / (to_call + pot) if to_call > 0 and pot > 0 else 0

        # MC equity (discard-aware)
        equity = 0.5
        if len(hand) == 2 and community:
            ev = get_evaluator()
            dead_set = set(hand) | set(community)
            for c in self.my_discards:
                if c >= 0: dead_set.add(c)
            for c in self.opp_discards:
                if c >= 0: dead_set.add(c)
            remaining = [c for c in ALL_CARDS if c not in dead_set]
            board_need = 5 - len(community)
            if len(remaining) >= board_need + 2:
                from itertools import combinations
                opp_combos = list(combinations(remaining, 2))
                board3 = community[:3]
                opp_weights = None
                if self.opp_discards and len(self.opp_discards) == 3 and all(c >= 0 for c in self.opp_discards):
                    w_dict = estimate_opp_keep_weights(self.opp_discards, board3, remaining)
                    if w_dict:
                        opp_weights = [w_dict.get((c1,c2), 0.05) for c1,c2 in opp_combos]
                        wt = sum(opp_weights)
                        if wt > 0: opp_weights = [w/wt for w in opp_weights]
                        else: opp_weights = None

                my_h = [int_to_treys(c) for c in hand]
                wins = ties = total = 0
                for _ in range(200):
                    if opp_weights:
                        oi = random.choices(range(len(opp_combos)), weights=opp_weights, k=1)[0]
                        opp_pair = list(opp_combos[oi])
                    else:
                        opp_pair = list(random.sample(remaining, 2))
                    if board_need > 0:
                        br = [c for c in remaining if c not in opp_pair]
                        if len(br) < board_need: continue
                        extra = random.sample(br, board_need)
                        fb = community + extra
                    else:
                        fb = community
                    b = [int_to_treys(c) for c in fb]
                    oh = [int_to_treys(c) for c in opp_pair]
                    mr = ev.evaluate(my_h, b); opr = ev.evaluate(oh, b)
                    if mr < opr: wins += 1
                    elif mr == opr: ties += 1
                    total += 1
                equity = (wins + 0.5*ties) / total if total > 0 else 0.5
        elif len(hand) == 2:
            r0, r1 = card_rank(hand[0]), card_rank(hand[1])
            pp = (r0 == r1); ha = (r0 == ACE_RANK_IDX or r1 == ACE_RANK_IDX)
            equity = 0.3
            if pp: equity = 0.55 + max(r0,r1)*0.02
            elif ha: equity = 0.45 + min(r0,r1)*0.02
            else: equity = 0.25 + max(r0,r1)*0.03
        else:
            ranks = sorted([card_rank(c) for c in my_cards], reverse=True)
            pp = len(set(ranks)) < len(ranks); ha = ACE_RANK_IDX in ranks
            equity = 0.3
            if pp: equity = 0.55
            elif ha: equity = 0.45
            elif max(ranks) >= 6: equity = 0.35

        # GTO-balanced decision
        can_raise = valid[_RAISE] and max_raise > 0
        facing_bet = to_call > 0

        if facing_bet:
            if can_raise and equity > 0.75:
                r = random.random()
                if r < 0.7:
                    amt = max(min_raise, min(int(pot*0.75), max_raise))
                    self.street_history += "R"; self.hero_last_raiser = True; self.villain_last_raiser = False
                    return (_RAISE, amt, 0, 0)
                else:
                    self.street_history += "C"; return (_CALL, 0, 0, 0)
            elif equity > pot_odds + 0.1:
                if can_raise and random.random() < 0.15:
                    amt = max(min_raise, min(int(pot*0.4), max_raise))
                    self.street_history += "r"; self.hero_last_raiser = True; return (_RAISE, amt, 0, 0)
                self.street_history += "C"; return (_CALL, 0, 0, 0)
            elif equity >= pot_odds:
                self.street_history += "C"; return (_CALL, 0, 0, 0)
            else:
                self.street_history += "F"; return (_FOLD, 0, 0, 0)
        else:
            if can_raise:
                bet_l = max(min_raise, min(int(pot*0.75), max_raise))
                bet_s = max(min_raise, min(int(pot*0.33), max_raise))
                bluff_ratio = bet_s / (pot + bet_s) if pot > 0 else 0.25

                if equity > 0.70:
                    if random.random() < 0.75:
                        self.street_history += "B"; self.hero_last_raiser = True; return (_RAISE, bet_l, 0, 0)
                    self.street_history += "K"; return (_CHECK, 0, 0, 0)
                elif equity > 0.55:
                    if random.random() < 0.5:
                        self.street_history += "b"; self.hero_last_raiser = True; return (_RAISE, bet_s, 0, 0)
                    self.street_history += "K"; return (_CHECK, 0, 0, 0)
                elif equity < 0.25:
                    if random.random() < bluff_ratio * 0.5:
                        self.street_history += "b"; self.hero_last_raiser = True; return (_RAISE, bet_s, 0, 0)
                    self.street_history += "K"; return (_CHECK, 0, 0, 0)
                else:
                    self.street_history += "K"; return (_CHECK, 0, 0, 0)
            else:
                self.street_history += "K"; return (_CHECK, 0, 0, 0) if valid[_CHECK] else (_FOLD, 0, 0, 0)

    def _refine_size(self, chosen_abs, obs):
        mn = int(obs["min_raise"]); mx = int(obs["max_raise"])
        if mx <= 0 or mx < mn: return mn
        pot = int(obs["my_bet"]) + int(obs["opp_bet"])
        spread = mx - mn
        if spread <= 0: return mn
        if chosen_abs in ("BET_LARGE","RAISE_LARGE"): fracs = [0.55,0.70,0.85,1.0]
        else: fracs = [0.10,0.20,0.30,0.40]
        cands = sorted(set(max(mn, min(mn+int(spread*f), mx)) for f in fracs))
        if len(cands) <= 1: return cands[0] if cands else mn

        # Quick equity (50 sims)
        hand = self.my_hand_2 if self.my_hand_2 else []
        community = [c for c in obs.get("community_cards", []) if c != -1]
        eq = 0.5
        if len(hand) == 2 and community:
            evl = get_evaluator()
            dead_set = set(hand) | set(community)
            for c in self.my_discards:
                if c >= 0: dead_set.add(c)
            for c in self.opp_discards:
                if c >= 0: dead_set.add(c)
            rem = [c for c in ALL_CARDS if c not in dead_set]
            bn = 5 - len(community)
            if len(rem) >= bn + 2:
                my_h = [int_to_treys(c) for c in hand]
                w = t = 0
                for _ in range(50):
                    s = random.sample(rem, bn + 2)
                    b = [int_to_treys(c) for c in community + s[:bn]]
                    oh = [int_to_treys(c) for c in s[bn:]]
                    if evl.evaluate(my_h, b) < evl.evaluate(oh, b): w += 1
                    t += 1
                eq = w / t if t > 0 else 0.5

        sd = 2.0 * eq - 1.0
        stake = min(int(obs["my_bet"]), int(obs["opp_bet"]))
        best_ev, best = float('-inf'), cands[len(cands)//2]
        for sz in cands:
            ff = sz / (sz + pot) if (sz+pot) > 0 else 0.3
            ev = ff * int(obs["opp_bet"]) + (1-ff) * sd * (stake+sz)
            if ev > best_ev: best_ev = ev; best = sz
        return best

    def _clamp(self, action, obs):
        at, amt, k1, k2 = action
        if at == _RAISE:
            mn, mx = int(obs["min_raise"]), int(obs["max_raise"])
            amt = max(mn, min(int(amt), mx))
            if mx <= 0:
                return (_CALL, 0, 0, 0) if obs["valid_actions"][_CALL] else (_CHECK, 0, 0, 0)
        return (at, amt, k1, k2)

    def act(self, obs):
        self._check_street(obs["street"])
        my_cards = [c for c in obs["my_cards"] if c != -1]
        if len(my_cards) == 5 and not self.my_hand_5: self.my_hand_5 = list(my_cards)
        if len(my_cards) == 2 and not self.my_hand_2: self.my_hand_2 = list(my_cards)
        od = [c for c in obs.get("opp_discarded_cards", []) if c != -1]
        if od and not self.opp_discards: self.opp_discards = list(od)
        md = [c for c in obs.get("my_discarded_cards", []) if c != -1]
        if md and not self.my_discards: self.my_discards = list(md)

        if obs["valid_actions"][_DISCARD]:
            community = [c for c in obs["community_cards"] if c != -1]
            if self.use_exact_discard:
                ki, kj = choose_discard(my_cards, community, od)
            else:
                ki, kj = _mc_choose_discard(my_cards, community, od)
            self.my_hand_2 = [my_cards[ki], my_cards[kj]]
            self.my_discards = [my_cards[k] for k in range(5) if k != ki and k != kj]
            return (_DISCARD, 0, ki, kj)

        cfr_result = self._cfr_action(obs)
        cfr_action, conf = cfr_result[0], cfr_result[1]

        if cfr_action is not None:
            use_cfr = False
            if conf >= 0.8:
                use_cfr = True
            elif random.random() < conf:
                use_cfr = True

            if use_cfr:
                chosen_abs = cfr_result[2]
                # Sizing subgame refinement
                at, amt, k1, k2 = cfr_action
                if self.use_subgame_sizing and at == _RAISE and chosen_abs in ("BET_SMALL","BET_LARGE","RAISE_SMALL","RAISE_LARGE"):
                    amt = self._refine_size(chosen_abs, obs)
                    cfr_action = (at, amt, k1, k2)
                self.street_history += action_to_short(chosen_abs)
                if chosen_abs in ("BET_SMALL","BET_LARGE","RAISE_SMALL","RAISE_LARGE","JAM"):
                    self.hero_last_raiser = True
                    self.villain_last_raiser = False
                return self._clamp(cfr_action, obs)

        return self._clamp(self._fallback_action(obs), obs)

    def observe_opp(self, opp_action):
        if opp_action == "RAISE":
            self.street_history += "R"
            self.villain_last_raiser = True
            self.hero_last_raiser = False
        elif opp_action == "CHECK": self.street_history += "K"
        elif opp_action == "CALL": self.street_history += "C"


def run_match(agent0, agent1, num_hands=1000):
    """Run a match between two SimpleAgents using gym_env directly."""
    env = PokerEnv(logger=logging.getLogger("h2h"))
    total_reward = 0

    for hand_num in range(num_hands):
        sb_player = hand_num % 2
        (obs0, obs1), info = env.reset(options={"small_blind_player": sb_player})
        agent0.reset(); agent1.reset()
        terminated = False

        while not terminated:
            acting = obs0["acting_agent"]
            if acting == 0:
                action = agent0.act(obs0)
            else:
                action = agent1.act(obs1)

            (obs0, obs1), (r0, r1), terminated, truncated, info = env.step(action)

            # Track opponent actions
            if not terminated:
                if acting == 0:
                    at = action[0]
                    name = {0:"FOLD",1:"RAISE",2:"CHECK",3:"CALL",4:"DISCARD"}.get(at,"")
                    if name and name != "DISCARD": agent1.observe_opp(name)
                else:
                    at = action[0]
                    name = {0:"FOLD",1:"RAISE",2:"CHECK",3:"CALL",4:"DISCARD"}.get(at,"")
                    if name and name != "DISCARD": agent0.observe_opp(name)

        total_reward += r0

    return total_reward


def main():
    parser = argparse.ArgumentParser(description="Head-to-head checkpoint evaluation")
    parser.add_argument("bin_a", help="First checkpoint binary")
    parser.add_argument("bin_b", nargs="?", default=None, help="Second checkpoint (omit for self-compare)")
    parser.add_argument("--matches", type=int, default=5)
    parser.add_argument("--self-compare", action="store_true", help="Compare exact vs MC discard on same checkpoint")
    parser.add_argument("--subgame-compare", action="store_true", help="Compare with vs without sizing subgame")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    print(f"Loading {args.bin_a}...")
    kA, pA, atA, alA, cA, iA, nA = convert_bin(args.bin_a)
    print(f"  → {iA:,} iters, {nA:,} nodes, conf={'YES' if cA is not None else 'NO'}")

    if args.subgame_compare:
        nameA = "subgame"
        nameB = "fixed"
        print(f"\n=== Sizing subgame vs fixed sizing ({args.matches} matches) ===")
        winsA = winsB = 0
        for m in range(args.matches):
            agentA = SimpleAgent(nameA, kA, pA, atA, alA, cA, use_subgame_sizing=True)
            agentB = SimpleAgent(nameB, kA, pA, atA, alA, cA, use_subgame_sizing=False)
            reward = run_match(agentA, agentB, num_hands=1000)
            winner = nameA if reward > 0 else (nameB if reward < 0 else "TIE")
            if reward > 0: winsA += 1
            elif reward < 0: winsB += 1
            print(f"  Match {m+1}: {winner} wins ({reward:+d})")
        print(f"\nResult: subgame={winsA}W  fixed={winsB}W  TIE={args.matches-winsA-winsB}")
        return

    if args.self_compare:
        # Same checkpoint, exact discard vs MC discard
        nameA = "exact"
        nameB = "mc"
        print(f"\n=== Exact discard vs MC discard ({args.matches} matches) ===")

        winsA = winsB = 0
        for m in range(args.matches):
            agentA = SimpleAgent(nameA, kA, pA, atA, alA, cA, use_exact_discard=True)
            agentB = SimpleAgent(nameB, kA, pA, atA, alA, cA, use_exact_discard=False)
            reward = run_match(agentA, agentB, num_hands=1000)
            winner = nameA if reward > 0 else (nameB if reward < 0 else "TIE")
            if reward > 0: winsA += 1
            elif reward < 0: winsB += 1
            print(f"  Match {m+1}: {winner} wins ({reward:+d})")
        print(f"\nResult: exact={winsA}W  mc={winsB}W  TIE={args.matches-winsA-winsB}")
        return

    if args.bin_b is None:
        print("Error: need bin_b or --self-compare")
        return

    print(f"Loading {args.bin_b}...")
    kB, pB, atB, alB, cB, iB, nB = convert_bin(args.bin_b)
    print(f"  → {iB:,} iters, {nB:,} nodes, conf={'YES' if cB is not None else 'NO'}")

    nameA = f"{iA//1000}k"
    nameB = f"{iB//1000}k"

    winsA = winsB = 0
    print(f"\n=== {nameA} vs {nameB} ({args.matches} matches) ===")

    for m in range(args.matches):
        agentA = SimpleAgent(nameA, kA, pA, atA, alA, cA)
        agentB = SimpleAgent(nameB, kB, pB, atB, alB, cB)

        reward = run_match(agentA, agentB, num_hands=1000)
        winner = nameA if reward > 0 else (nameB if reward < 0 else "TIE")
        if reward > 0: winsA += 1
        elif reward < 0: winsB += 1
        print(f"  Match {m+1}: {winner} wins ({reward:+d})")

    print(f"\nResult: {nameA}={winsA}W  {nameB}={winsB}W  TIE={args.matches-winsA-winsB}")


if __name__ == "__main__":
    main()

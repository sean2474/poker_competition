import os
import sys
import pickle
import random
import struct

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.agent import Agent
from gym_env import PokerEnv

from abstractions.discard_oracle import choose_discard
from abstractions.action_abs import (
    get_valid_abstract_actions, abstract_to_concrete,
    action_to_short, get_action_context,
)
from abstractions.card_utils import card_rank, canonicalize_suits, ACE_RANK_IDX
from abstractions.board_texture import board_bucket_for_street
from abstractions.hand_bucket import hand_bucket_for_street
from abstractions.opp_discard_bucket import opp_discard_bucket
from abstractions.public_state import line_bucket, pressure_bucket

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
STRATEGY_KEYS_PATH = os.path.join(DATA_DIR, "strategy_keys.npy")
STRATEGY_PROBS_PATH = os.path.join(DATA_DIR, "strategy_probs.npy")
STRATEGY_ACTTYPE_PATH = os.path.join(DATA_DIR, "strategy_acttype.npy")
STRATEGY_CONFIDENCE_PATH = os.path.join(DATA_DIR, "strategy_confidence.npy")
STRATEGY_META_PATH = os.path.join(DATA_DIR, "strategy_meta.pkl")

CONFIDENCE_THRESHOLD = 50.0

_CTX_MAP = {"no_bet": 0, "facing_bet": 1, "high_pressure": 2}


# ─── FNV-1a hash (matches C++ cfr_engine.h exactly) ───
def _fnv_hash(data: bytes) -> int:
    h = 14695981039346656037
    for b in data:
        h ^= b
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def _make_preflop_key(hand5: list, is_bb: bool, street_history: str) -> int:
    """Matches C++ make_preflop_key byte layout exactly."""
    canon = canonicalize_suits(tuple(sorted(hand5)))
    pos = 1 if is_bb else 0
    line = line_bucket(street_history)
    buf = bytearray()
    buf.append(0)  # PF marker
    buf.append(pos)
    buf.append(line)
    for c in canon:
        buf.append(c)
    return _fnv_hash(bytes(buf))


def _make_postdiscard_key(street: int, hand2: list, community: list,
                           opp_discards: list, is_bb: bool,
                           hero_agg: bool, villain_agg: bool,
                           street_history: str, my_bet: int, opp_bet: int,
                           dead: list, action_ctx: str, n_actions: int) -> int:
    """Matches C++ make_postdiscard_key byte layout exactly."""
    pos = 1 if is_bb else 0
    init = 1 if hero_agg else (2 if villain_agg else 0)
    line = line_bucket(street_history)
    press = pressure_bucket(my_bet, opp_bet)
    board_bkt = board_bucket_for_street(community, street)
    board3 = community[:3] if len(community) >= 3 else community
    opp_disc_bkt = opp_discard_bucket(opp_discards, board3)
    hand_bkt = hand_bucket_for_street(hand2, community, street, dead)
    actx = _CTX_MAP.get(action_ctx, 0)

    buf = bytearray()
    buf.append(street)
    buf.append(pos)
    buf.append(init)
    buf.append(line)
    buf.append(press)
    buf.extend(struct.pack('<H', board_bkt))  # uint16 little-endian
    buf.append(opp_disc_bkt & 0xFF)
    buf.append(hand_bkt & 0xFF)
    buf.append(actx)
    buf.append(n_actions)
    return _fnv_hash(bytes(buf))

# ActionType enum values
_FOLD = 0
_RAISE = 1
_CHECK = 2
_CALL = 3
_DISCARD = 4


class PlayerAgent(Agent):
    def __init__(self, stream: bool = True):
        super().__init__(stream)

        # Strategy lookup tables
        self.strategy_loaded = False
        self.key_to_idx = None
        self.probs = None
        self.act_types = None
        self.action_lists = None
        self.confidence = None

        # Per-hand tracking
        self.current_hand = -1
        self.street_history = ""
        self.last_street = -1
        self.hero_last_raiser = False
        self.villain_last_raiser = False
        self.my_hand_5 = []
        self.my_hand_2 = []
        self.my_discards = []
        self.opp_discards = []

        self._load_strategy()

    def _load_strategy(self):
        """Load compressed numpy strategy."""
        if not os.path.exists(STRATEGY_KEYS_PATH):
            return
        try:
            keys = np.load(STRATEGY_KEYS_PATH)
            self.probs = np.load(STRATEGY_PROBS_PATH)
            self.act_types = np.load(STRATEGY_ACTTYPE_PATH)
            if os.path.exists(STRATEGY_CONFIDENCE_PATH):
                self.confidence = np.load(STRATEGY_CONFIDENCE_PATH)
            with open(STRATEGY_META_PATH, 'rb') as f:
                meta = pickle.load(f)
            self.action_lists = meta['action_lists']
            self.key_to_idx = {int(keys[i]): i for i in range(len(keys))}
            self.strategy_loaded = True
            self.logger.info(
                f"Loaded strategy: {meta['iterations']} iters, {meta['num_nodes']} nodes"
            )
        except Exception as e:
            self.logger.warning(f"Failed to load strategy: {e}")

    def __name__(self):
        return "PlayerAgent"

    def _reset_hand(self, hand_number):
        if hand_number != self.current_hand:
            self.current_hand = hand_number
            self.street_history = ""
            self.last_street = -1
            self.hero_last_raiser = False
            self.villain_last_raiser = False
            self.my_hand_5 = []
            self.my_hand_2 = []
            self.my_discards = []
            self.opp_discards = []

    def _check_street_change(self, street):
        if street != self.last_street:
            self.street_history = ""
            if self.last_street >= 0:
                pass  # initiative carries over via hero/villain_last_raiser
            self.last_street = street

    def _is_bb(self, observation):
        return observation.get("blind_position", 0) == 1

    def _cfr_lookup(self, observation) -> tuple:
        """
        CFR strategy lookup with confidence-based blending.
        Returns (concrete_action, confidence) or (None, 0).
        confidence in [0, 1]: how much to trust CFR vs fallback.
        """
        if not self.strategy_loaded:
            return None, 0.0

        my_cards = [c for c in observation["my_cards"] if c != -1]
        community = [c for c in observation["community_cards"] if c != -1]
        is_bb = self._is_bb(observation)
        street = observation["street"]

        valid = observation["valid_actions"]
        min_raise = observation["min_raise"]
        max_raise = observation["max_raise"]
        my_bet = observation["my_bet"]
        opp_bet = observation["opp_bet"]
        valid_abs = get_valid_abstract_actions(valid, my_bet, opp_bet, min_raise, max_raise)
        n = len(valid_abs)
        action_ctx = get_action_context(valid, my_bet, opp_bet, max_raise)

        if street == 0:
            hand5 = self.my_hand_5 if self.my_hand_5 else my_cards
            h = _make_preflop_key(hand5, is_bb, self.street_history)
        else:
            hand2 = self.my_hand_2 if self.my_hand_2 else my_cards[:2]
            dead = list(self.my_discards) + list(self.opp_discards)
            opp_disc = self.opp_discards if self.opp_discards else [-1, -1, -1]
            h = _make_postdiscard_key(
                street, hand2, community, opp_disc, is_bb,
                self.hero_last_raiser, self.villain_last_raiser,
                self.street_history, my_bet, opp_bet, dead,
                action_ctx, n
            )

        idx = self.key_to_idx.get(h)
        if idx is None:
            return None, 0.0

        stored_type = self.act_types[idx]
        if stored_type >= len(self.action_lists):
            return None, 0.0
        stored_actions = list(self.action_lists[stored_type])
        if len(stored_actions) != n or stored_actions != valid_abs:
            return None, 0.0

        # Confidence from strategy_sum total
        conf = 0.0
        if self.confidence is not None and idx < len(self.confidence):
            raw_conf = float(self.confidence[idx])
            conf = min(raw_conf / CONFIDENCE_THRESHOLD, 1.0)

        # Read quantized probs and dequantize
        raw_probs = self.probs[idx, :n].astype(np.float32)
        total = raw_probs.sum()
        if total > 0:
            probs = raw_probs / total
        else:
            probs = np.ones(n, dtype=np.float32) / n

        chosen_idx = random.choices(range(n), weights=probs.tolist(), k=1)[0]
        chosen_abs = valid_abs[chosen_idx]

        concrete = abstract_to_concrete(chosen_abs, min_raise, max_raise, my_bet, opp_bet)

        self.street_history += action_to_short(chosen_abs)
        if chosen_abs in ("BET_SMALL", "BET_LARGE", "RAISE_SMALL", "RAISE_LARGE", "JAM"):
            self.hero_last_raiser = True
            self.villain_last_raiser = False

        return concrete, conf

    def _fallback_action(self, observation) -> tuple:
        """MC equity-based fallback when CFR has no entry."""
        valid = observation["valid_actions"]
        my_cards = [c for c in observation["my_cards"] if c != -1]
        community = [c for c in observation["community_cards"] if c != -1]
        my_bet = observation["my_bet"]
        opp_bet = observation["opp_bet"]
        min_raise = observation["min_raise"]
        max_raise = observation["max_raise"]
        street = observation["street"]

        hand = self.my_hand_2 if self.my_hand_2 else my_cards[:2]

        to_call = opp_bet - my_bet
        pot = my_bet + opp_bet
        pot_odds = to_call / (to_call + pot) if to_call > 0 and pot > 0 else 0

        # ─── Compute MC equity (discard-aware) ───
        if len(hand) == 2 and community:
            from abstractions.card_utils import get_evaluator, int_to_treys, ALL_CARDS
            from abstractions.discard_oracle import estimate_opp_keep_weights
            from itertools import combinations
            ev = get_evaluator()
            dead_set = set(hand) | set(community)
            for c in self.my_discards:
                if c >= 0: dead_set.add(c)
            for c in self.opp_discards:
                if c >= 0: dead_set.add(c)
            remaining = [c for c in ALL_CARDS if c not in dead_set]
            board_need = 5 - len(community)
            total_need = board_need + 2

            if len(remaining) >= total_need:
                my_h = [int_to_treys(c) for c in hand]

                # Build weighted opponent range from their discards
                opp_combos = list(combinations(remaining, 2))
                board3 = community[:3] if len(community) >= 3 else community
                opp_weights = None
                if self.opp_discards and len(self.opp_discards) == 3 and all(c >= 0 for c in self.opp_discards):
                    w_dict = estimate_opp_keep_weights(self.opp_discards, board3, remaining)
                    if w_dict:
                        opp_weights = [w_dict.get((c1, c2), 0.05) for c1, c2 in opp_combos]
                        w_total = sum(opp_weights)
                        if w_total > 0:
                            opp_weights = [w / w_total for w in opp_weights]
                        else:
                            opp_weights = None

                wins = ties = total = 0
                import random as _rng
                for _ in range(200):
                    # Pick opponent hand (weighted or uniform)
                    if opp_weights:
                        opp_idx = _rng.choices(range(len(opp_combos)), weights=opp_weights, k=1)[0]
                        opp_pair = list(opp_combos[opp_idx])
                    else:
                        opp_pair = list(_rng.sample(remaining, 2))

                    # Sample remaining board cards (excluding opp hand)
                    if board_need > 0:
                        board_remaining = [c for c in remaining if c not in opp_pair]
                        if len(board_remaining) < board_need:
                            continue
                        extra = _rng.sample(board_remaining, board_need)
                        full_board = community + extra
                    else:
                        full_board = community

                    b = [int_to_treys(c) for c in full_board]
                    opp_h = [int_to_treys(c) for c in opp_pair]
                    mr = ev.evaluate(my_h, b)
                    opr = ev.evaluate(opp_h, b)
                    if mr < opr: wins += 1
                    elif mr == opr: ties += 1
                    total += 1
                equity = (wins + 0.5 * ties) / total if total > 0 else 0.5
            else:
                equity = 0.5
        elif len(hand) == 2:
            # Preflop with 2 cards: rough equity from card features
            from abstractions.card_utils import ACE_RANK_IDX
            r0, r1 = card_rank(hand[0]), card_rank(hand[1])
            pp = (r0 == r1)
            has_a = (r0 == ACE_RANK_IDX or r1 == ACE_RANK_IDX)
            high = max(r0, r1)
            equity = 0.3  # base
            if pp: equity = 0.55 + high * 0.02
            elif has_a: equity = 0.45 + min(r0, r1) * 0.02
            else: equity = 0.25 + high * 0.03
        else:
            # Preflop 5 cards: rough score
            from abstractions.card_utils import ACE_RANK_IDX
            ranks = sorted([card_rank(c) for c in my_cards], reverse=True)
            pp = len(set(ranks)) < len(ranks)
            has_a = ACE_RANK_IDX in ranks
            equity = 0.3
            if pp: equity = 0.55
            elif has_a: equity = 0.45
            elif max(ranks) >= 6: equity = 0.35

        # ─── Equity-based decision ───
        def _act(action_str, action_type, raise_amt=0):
            self.street_history += action_to_short(action_str)
            if action_str in ("BET_SMALL", "BET_LARGE", "RAISE_SMALL", "RAISE_LARGE", "JAM"):
                self.hero_last_raiser = True
                self.villain_last_raiser = False
            return (action_type, raise_amt, 0, 0)

        if equity > 0.75 and valid[_RAISE]:
            amount = max(min_raise, min(int(pot * 0.75), max_raise))
            return _act("BET_LARGE", _RAISE, amount)
        elif equity > 0.55 and valid[_RAISE] and to_call <= 0:
            amount = max(min_raise, min(int(pot * 0.5), max_raise))
            return _act("BET_SMALL", _RAISE, amount)
        elif equity >= pot_odds and equity > 0.35 and valid[_CALL]:
            return _act("CALL", _CALL)
        elif valid[_CHECK]:
            return _act("CHECK", _CHECK)
        elif equity >= pot_odds and valid[_CALL]:
            return _act("CALL", _CALL)
        else:
            return _act("FOLD", _FOLD)

    def act(self, observation, reward, terminated, truncated, info):
        hand_number = info.get('hand_number', -1)
        self._reset_hand(hand_number)
        self._check_street_change(observation["street"])

        valid_actions = observation["valid_actions"]

        # Track our cards
        my_cards = [c for c in observation["my_cards"] if c != -1]
        if len(my_cards) == 5 and not self.my_hand_5:
            self.my_hand_5 = list(my_cards)
        if len(my_cards) == 2 and not self.my_hand_2:
            self.my_hand_2 = list(my_cards)

        # Track discards
        opp_disc = [c for c in observation.get("opp_discarded_cards", []) if c != -1]
        if opp_disc and not self.opp_discards:
            self.opp_discards = list(opp_disc)
        my_disc = [c for c in observation.get("my_discarded_cards", []) if c != -1]
        if my_disc and not self.my_discards:
            self.my_discards = list(my_disc)

        # ─── Discard phase ───
        if valid_actions[_DISCARD]:
            community = [c for c in observation["community_cards"] if c != -1]
            ki, kj = choose_discard(my_cards, community, opp_disc, top_k=3, mc_sims=150)
            self.my_hand_2 = [my_cards[ki], my_cards[kj]]
            self.my_discards = [my_cards[k] for k in range(5) if k != ki and k != kj]
            return (_DISCARD, 0, ki, kj)

        # ─── Betting phase: confidence-based CFR/fallback blending ───
        cfr_action, confidence = self._cfr_lookup(observation)

        if cfr_action is None:
            # CFR miss → pure fallback
            return self._fallback_action(observation)

        if confidence >= 0.8:
            # High confidence → trust CFR
            return cfr_action

        # Low/medium confidence → coin flip weighted by confidence
        if random.random() < confidence:
            return cfr_action
        return self._fallback_action(observation)

    def observe(self, observation, reward, terminated, truncated, info):
        """Track opponent actions to maintain history."""
        opp_action = observation.get("opp_last_action", "None")
        if opp_action and opp_action != "None":
            # Map engine action names to our short codes
            if opp_action == "FOLD":
                self.street_history += "F"
            elif opp_action == "CHECK":
                self.street_history += "K"
            elif opp_action == "CALL":
                self.street_history += "C"
            elif opp_action == "RAISE":
                self.street_history += "R"
                self.villain_last_raiser = True
                self.hero_last_raiser = False


"""
Analyze CFR lookup hit rates and chip flow across 1000 hands vs ProbabilityAgent.
Runs a full match with instrumented PlayerAgent and collects stats.
"""

import os
import sys
import pickle
import struct
import random
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gym_env import PokerEnv
from abstractions.discard_oracle import choose_discard
from abstractions.action_abs import (
    get_valid_abstract_actions, abstract_to_concrete,
    action_to_short, get_action_context,
)
from abstractions.hand_bucket import _made_tier_from_structure, _draw_tier, hand_bucket_for_street
from abstractions.card_utils import card_rank, card_suit, canonicalize_suits, ACE_RANK_IDX
from abstractions.board_texture import board_bucket_for_street
from abstractions.opp_discard_bucket import opp_discard_bucket
from abstractions.public_state import line_bucket, pressure_bucket

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_CTX_MAP = {"no_bet": 0, "facing_bet": 1, "high_pressure": 2}

def _fnv_hash(data: bytes) -> int:
    h = 14695981039346656037
    for b in data:
        h ^= b
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h

def _make_preflop_key(hand5, is_bb, street_history):
    canon = canonicalize_suits(tuple(sorted(hand5)))
    buf = bytearray()
    buf.append(0)
    buf.append(1 if is_bb else 0)
    buf.append(line_bucket(street_history))
    for c in canon:
        buf.append(c)
    return _fnv_hash(bytes(buf))

def _make_postdiscard_key(street, hand2, community, opp_discards, is_bb,
                           hero_agg, villain_agg, street_history, my_bet, opp_bet,
                           dead, action_ctx, n_actions):
    buf = bytearray()
    buf.append(street)
    buf.append(1 if is_bb else 0)
    buf.append(1 if hero_agg else (2 if villain_agg else 0))
    buf.append(line_bucket(street_history))
    buf.append(pressure_bucket(my_bet, opp_bet))
    buf.extend(struct.pack('<H', board_bucket_for_street(community, street)))
    board3 = community[:3] if len(community) >= 3 else community
    buf.append(opp_discard_bucket(opp_discards, board3) & 0xFF)
    buf.append(hand_bucket_for_street(hand2, community, street, dead) & 0xFF)
    buf.append(_CTX_MAP.get(action_ctx, 0))
    buf.append(n_actions)
    return _fnv_hash(bytes(buf))

def load_strategy():
    keys_path = os.path.join(DATA_DIR, "strategy_keys.npy")
    if not os.path.exists(keys_path):
        print("No compressed strategy found")
        return None, None, None, None
    keys = np.load(keys_path)
    probs = np.load(os.path.join(DATA_DIR, "strategy_probs.npy"))
    act_types = np.load(os.path.join(DATA_DIR, "strategy_acttype.npy"))
    with open(os.path.join(DATA_DIR, "strategy_meta.pkl"), 'rb') as f:
        meta = pickle.load(f)
    key_to_idx = {int(keys[i]): i for i in range(len(keys))}
    return key_to_idx, probs, act_types, meta

def analyze():
    key_to_idx, probs, act_types, meta = load_strategy()
    if key_to_idx is None:
        return
    action_lists = meta['action_lists']
    print(f"Strategy: {meta['iterations']} iters, {meta['num_nodes']} nodes")

    # Stats
    stats = {
        'total_decisions': 0,
        'cfr_hits': 0,
        'cfr_misses': 0,
        'hits_by_street': defaultdict(int),
        'misses_by_street': defaultdict(int),
        'hits_by_ctx': defaultdict(int),
        'misses_by_ctx': defaultdict(int),
        'chip_flow_hit': 0,   # chips won/lost when CFR was used
        'chip_flow_miss': 0,  # chips won/lost when fallback was used
        'hands_hit': 0,
        'hands_miss': 0,
        'hand_buckets_seen': defaultdict(int),
        'board_buckets_seen': defaultdict(int),
    }

    import logging
    env = PokerEnv(logger=logging.getLogger())

    num_hands = 1000
    total_reward = 0

    for hand_num in range(num_hands):
        sb_player = hand_num % 2
        (obs0, obs1), info = env.reset(options={"small_blind_player": sb_player})

        # Track per-hand
        my_hand_5 = [c for c in obs0["my_cards"] if c != -1]
        my_hand_2 = []
        my_discards = []
        opp_discards_list = []
        street_history = ""
        last_street = -1
        hero_agg = False
        villain_agg = False
        hand_had_hit = False
        hand_had_miss = False

        terminated = False
        reward = 0

        while not terminated:
            acting = obs0["acting_agent"]
            street = obs0["street"]

            # Street change
            if street != last_street:
                street_history = ""
                last_street = street

            if acting == 0:
                valid = obs0["valid_actions"]
                my_cards = [c for c in obs0["my_cards"] if c != -1]
                community = [c for c in obs0["community_cards"] if c != -1]
                my_bet = obs0["my_bet"]
                opp_bet = obs0["opp_bet"]

                # Track cards
                if len(my_cards) == 5 and not my_hand_5:
                    my_hand_5 = list(my_cards)
                if len(my_cards) == 2 and not my_hand_2:
                    my_hand_2 = list(my_cards)
                od = [c for c in obs0.get("opp_discarded_cards", []) if c != -1]
                if od and not opp_discards_list:
                    opp_discards_list = list(od)

                # Discard
                if valid[4]:  # DISCARD
                    ki, kj = choose_discard(my_cards, community, od, top_k=3, mc_sims=100)
                    my_hand_2 = [my_cards[ki], my_cards[kj]]
                    my_discards = [my_cards[k] for k in range(5) if k != ki and k != kj]
                    action = (4, 0, ki, kj)
                else:
                    # CFR lookup attempt
                    min_raise = obs0["min_raise"]
                    max_raise = obs0["max_raise"]
                    valid_abs = get_valid_abstract_actions(valid, my_bet, opp_bet, min_raise, max_raise)
                    n = len(valid_abs)
                    action_ctx = get_action_context(valid, my_bet, opp_bet, max_raise)
                    is_bb = obs0.get("blind_position", 0) == 1

                    # Build key
                    if street == 0:
                        h = _make_preflop_key(my_hand_5 or my_cards, is_bb, street_history)
                    else:
                        hand2 = my_hand_2 or my_cards[:2]
                        dead = list(my_discards) + list(opp_discards_list)
                        opp_disc = opp_discards_list if opp_discards_list else [-1,-1,-1]
                        h = _make_postdiscard_key(
                            street, hand2, community, opp_disc, is_bb,
                            hero_agg, villain_agg, street_history,
                            my_bet, opp_bet, dead, action_ctx, n
                        )

                    stats['total_decisions'] += 1

                    # Track buckets
                    if community:
                        stats['board_buckets_seen'][board_bucket_for_street(community, street)] += 1
                        if my_hand_2:
                            stats['hand_buckets_seen'][hand_bucket_for_street(my_hand_2, community, street)] += 1

                    idx = key_to_idx.get(h)
                    if idx is not None:
                        stats['cfr_hits'] += 1
                        stats['hits_by_street'][street] += 1
                        stats['hits_by_ctx'][action_ctx] += 1
                        hand_had_hit = True

                        # Use CFR strategy
                        stored_type = act_types[idx]
                        stored_actions = list(action_lists[stored_type])
                        raw = probs[idx, :n].astype(np.float32)
                        total = raw.sum()
                        if total > 0:
                            p = raw / total
                        else:
                            p = np.ones(n) / n
                        ci = random.choices(range(n), weights=p.tolist(), k=1)[0]
                        chosen = valid_abs[ci]
                        action = abstract_to_concrete(chosen, min_raise, max_raise, my_bet, opp_bet)
                        street_history += action_to_short(chosen)
                        if chosen in ("BET_SMALL","BET_LARGE","RAISE_SMALL","RAISE_LARGE","JAM"):
                            hero_agg = True; villain_agg = False
                    else:
                        stats['cfr_misses'] += 1
                        stats['misses_by_street'][street] += 1
                        stats['misses_by_ctx'][action_ctx] += 1
                        hand_had_miss = True

                        # Fallback: simple check/fold
                        if valid[2]: action = (2, 0, 0, 0); street_history += "K"
                        elif valid[3]: action = (3, 0, 0, 0); street_history += "C"
                        else: action = (0, 0, 0, 0); street_history += "F"
            else:
                # Opponent acts - use simple strategy for analysis
                if obs1["valid_actions"][4]:
                    cards1 = [c for c in obs1["my_cards"] if c != -1]
                    comm1 = [c for c in obs1["community_cards"] if c != -1]
                    ki, kj = choose_discard(cards1, comm1, [], top_k=2, mc_sims=50)
                    action = (4, 0, ki, kj)
                elif obs1["valid_actions"][2]:
                    action = (2, 0, 0, 0)
                elif obs1["valid_actions"][3]:
                    action = (3, 0, 0, 0)
                else:
                    action = (0, 0, 0, 0)

            (obs0, obs1), (r0, r1), terminated, truncated, info = env.step(action)

            if terminated:
                reward = r0
                total_reward += r0

        if hand_had_hit and not hand_had_miss:
            stats['chip_flow_hit'] += reward
            stats['hands_hit'] += 1
        elif hand_had_miss:
            stats['chip_flow_miss'] += reward
            stats['hands_miss'] += 1

    # Print results
    print(f"\n{'='*60}")
    print(f"ANALYSIS: {num_hands} hands, total reward: {total_reward}")
    print(f"{'='*60}")

    total_d = stats['total_decisions']
    hits = stats['cfr_hits']
    misses = stats['cfr_misses']
    print(f"\nOverall hit rate: {hits}/{total_d} = {hits/total_d*100:.1f}%")

    print(f"\nBy street:")
    for st in range(4):
        h = stats['hits_by_street'].get(st, 0)
        m = stats['misses_by_street'].get(st, 0)
        t = h + m
        pct = h/t*100 if t > 0 else 0
        name = ['Preflop','Flop','Turn','River'][st]
        print(f"  {name}: {h}/{t} = {pct:.1f}% hit")

    print(f"\nBy action context:")
    for ctx in ['no_bet', 'facing_bet', 'high_pressure']:
        h = stats['hits_by_ctx'].get(ctx, 0)
        m = stats['misses_by_ctx'].get(ctx, 0)
        t = h + m
        pct = h/t*100 if t > 0 else 0
        print(f"  {ctx}: {h}/{t} = {pct:.1f}% hit")

    print(f"\nChip flow:")
    if stats['hands_hit'] > 0:
        print(f"  CFR-only hands: {stats['hands_hit']} hands, {stats['chip_flow_hit']:+d} chips ({stats['chip_flow_hit']/stats['hands_hit']:+.1f}/hand)")
    if stats['hands_miss'] > 0:
        print(f"  Fallback hands: {stats['hands_miss']} hands, {stats['chip_flow_miss']:+d} chips ({stats['chip_flow_miss']/stats['hands_miss']:+.1f}/hand)")

    print(f"\nUnique board buckets seen: {len(stats['board_buckets_seen'])}")
    print(f"Unique hand buckets seen: {len(stats['hand_buckets_seen'])}")

    # Top missed board/hand buckets
    print(f"\nTop 10 board buckets by frequency:")
    for bkt, cnt in sorted(stats['board_buckets_seen'].items(), key=lambda x: -x[1])[:10]:
        print(f"  bucket {bkt}: {cnt}")

    print(f"\nTop 10 hand buckets by frequency:")
    for bkt, cnt in sorted(stats['hand_buckets_seen'].items(), key=lambda x: -x[1])[:10]:
        print(f"  bucket {bkt}: {cnt}")


if __name__ == "__main__":
    analyze()

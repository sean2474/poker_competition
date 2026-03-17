"""
Diagnose CFR lookup hit rate: key found, atype match, conf usage.
"""
import os, sys, struct, random, logging
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "submission_mccfr"))

from gym_env import PokerEnv
from mccfr_training.eval_h2h import convert_bin, compute_confidence_threshold, SimpleAgent, run_match
from abstractions.action_abs import get_valid_abstract_actions, abstract_to_concrete
from abstractions.card_utils import card_rank, canonicalize_suits, ACE_RANK_IDX, get_evaluator, int_to_treys, ALL_CARDS
from abstractions.board_texture import board_bucket_for_street
from abstractions.hand_bucket import hand_bucket_for_street
from abstractions.opp_discard_bucket import opp_discard_bucket
from abstractions.public_state import line_bucket, pressure_bucket

_FOLD = 0; _RAISE = 1; _CHECK = 2; _CALL = 3; _DISCARD = 4
_CTX_MAP = {"no_bet": 0, "facing_bet": 1}

def _fnv_hash(data):
    h = 14695981039346656037
    for b in data:
        h ^= b
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h

def diagnose(bin_path, num_hands=500):
    print(f"Loading {bin_path}...")
    key_to_idx, probs, act_types, action_lists, confidence, iters, nodes = convert_bin(bin_path)
    threshold = compute_confidence_threshold(confidence)
    print(f"  {iters:,} iters, {nodes:,} nodes, threshold={threshold:.0f}")

    # Counters
    total_decisions = 0
    key_found = 0
    atype_match = 0
    cfr_used = 0
    atype_mismatch_details = {}

    env = PokerEnv(logger=logging.getLogger("diag"))

    for hand_num in range(num_hands):
        sb_player = hand_num % 2
        (obs0, obs1), info = env.reset(options={"small_blind_player": sb_player})

        # Track per-hand state
        state = {
            0: {"hand5": [], "hand2": [], "my_disc": [], "opp_disc": [], "street_hist": "", "last_street": -1, "hero_agg": False, "villain_agg": False},
            1: {"hand5": [], "hand2": [], "my_disc": [], "opp_disc": [], "street_hist": "", "last_street": -1, "hero_agg": False, "villain_agg": False},
        }
        terminated = False

        while not terminated:
            obs_list = [obs0, obs1]
            acting = obs0["acting_agent"]
            obs = obs_list[acting]
            s = state[acting]
            opp_s = state[1 - acting]

            # Update street history
            street = obs["street"]
            if street != s["last_street"]:
                s["street_hist"] = ""
                s["last_street"] = street

            my_cards = [c for c in obs["my_cards"] if c != -1]
            if len(my_cards) == 5 and not s["hand5"]: s["hand5"] = list(my_cards)
            if len(my_cards) == 2 and not s["hand2"]: s["hand2"] = list(my_cards)
            od = [c for c in obs.get("opp_discarded_cards", []) if c != -1]
            if od and not s["opp_disc"]: s["opp_disc"] = list(od)
            md = [c for c in obs.get("my_discarded_cards", []) if c != -1]
            if md and not s["my_disc"]: s["my_disc"] = list(md)

            if obs["valid_actions"][_DISCARD]:
                action = (_DISCARD, 0, 0, 1)  # dummy discard
                (obs0, obs1), _, terminated, _, _ = env.step(action)
                continue

            # CFR lookup diagnostic
            community = [c for c in obs["community_cards"] if c != -1]
            is_bb = obs.get("blind_position", 0) == 1
            my_bet, opp_bet = obs["my_bet"], obs["opp_bet"]
            min_raise, max_raise = obs["min_raise"], obs["max_raise"]
            valid = obs["valid_actions"]
            valid_abs = get_valid_abstract_actions(valid, my_bet, opp_bet, min_raise, max_raise)
            n = len(valid_abs)

            if street == 0:
                hand5 = s["hand5"] or my_cards
                canon = canonicalize_suits(tuple(sorted(hand5)))
                buf = bytearray([0, 1 if is_bb else 0, line_bucket(s["street_hist"])])
                for c in canon: buf.append(c)
                h = _fnv_hash(bytes(buf))
            else:
                hand2 = s["hand2"] or my_cards[:2]
                opp_disc = s["opp_disc"] if s["opp_disc"] else [-1, -1, -1]
                dead = list(s["my_disc"]) + list(s["opp_disc"])
                from abstractions.action_abs import get_action_context
                actx = get_action_context(valid, my_bet, opp_bet, max_raise)
                buf = bytearray()
                buf.append(street)
                buf.append(1 if is_bb else 0)
                buf.append(1 if s["hero_agg"] else (2 if s["villain_agg"] else 0))
                buf.append(line_bucket(s["street_hist"]))
                buf.append(pressure_bucket(my_bet, opp_bet))
                buf.extend(struct.pack('<H', board_bucket_for_street(community, street)))
                board3 = community[:3]
                buf.append(opp_discard_bucket(opp_disc, board3) & 0xFF)
                buf.append(hand_bucket_for_street(hand2, community, street, dead) & 0xFF)
                buf.append(_CTX_MAP.get(actx, 0))
                buf.append(n)
                h = _fnv_hash(bytes(buf))

            total_decisions += 1
            idx = key_to_idx.get(h)
            if idx is not None:
                key_found += 1
                stored_type = act_types[idx]
                if stored_type < len(action_lists):
                    stored_actions = list(action_lists[stored_type])
                    if len(stored_actions) == n and stored_actions == valid_abs:
                        atype_match += 1
                        # Check confidence
                        conf = 0.0
                        if confidence is not None and idx < len(confidence):
                            conf = min(float(confidence[idx]) / threshold, 1.0)
                        use_cfr = conf >= 0.8 or random.random() < conf
                        if use_cfr:
                            cfr_used += 1
                    else:
                        k = (tuple(stored_actions), tuple(valid_abs))
                        atype_mismatch_details[k] = atype_mismatch_details.get(k, 0) + 1

            # Simple action for progression
            if valid[_CHECK]:
                action = (_CHECK, 0, 0, 0)
            elif valid[_CALL]:
                action = (_CALL, 0, 0, 0)
            else:
                action = (_FOLD, 0, 0, 0)

            (obs0, obs1), _, terminated, _, _ = env.step(action)

    print(f"\n=== CFR Diagnostic ({num_hands} hands) ===")
    print(f"Total decisions:  {total_decisions}")
    print(f"Key found:        {key_found} ({100*key_found/total_decisions:.1f}%)")
    if key_found > 0:
        print(f"Atype match:      {atype_match} ({100*atype_match/key_found:.1f}% of found)")
        print(f"CFR actually used:{cfr_used} ({100*cfr_used/key_found:.1f}% of found)")
    print(f"CFR overall rate: {100*cfr_used/total_decisions:.1f}%")
    if atype_mismatch_details:
        print(f"\nAtype mismatch cases (stored→valid_abs):")
        for (stored, valid), cnt in sorted(atype_mismatch_details.items(), key=lambda x: -x[1])[:5]:
            print(f"  {list(stored)} → {list(valid)}: {cnt}x")

if __name__ == "__main__":
    import sys
    bin_path = sys.argv[1] if len(sys.argv) > 1 else "mccfr_training/data/checkpoints/merged_143M/strategy_cpp.bin"
    diagnose(bin_path, num_hands=200)

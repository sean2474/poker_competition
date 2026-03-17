"""
Blueprint-based opponent range tracking.

Uses CFR blueprint strategies to compute P(opp_hand | action_history).
Each time opponent takes an action, multiply their range by the blueprint's
probability of that action for each possible hand.

This is the Libratus approach to range estimation.
"""

import numpy as np
from itertools import combinations
from abstractions.card_utils import (
    ALL_CARDS, card_rank, card_suit, canonicalize_suits,
    get_evaluator, int_to_treys,
)
from abstractions.action_abs import get_valid_abstract_actions, get_action_context
from abstractions.board_texture import board_bucket_for_street
from abstractions.hand_bucket import hand_bucket_for_street
from abstractions.opp_discard_bucket import opp_discard_bucket
from abstractions.public_state import line_bucket, pressure_bucket
import struct


_CTX_MAP = {"no_bet": 0, "facing_bet": 1}


def _fnv_hash(data):
    h = 14695981039346656037
    for b in data:
        h ^= b
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def _make_postdiscard_key(street, hand2, community, opp_disc, is_bb,
                           hero_agg, villain_agg, street_history,
                           my_bet, opp_bet, dead, action_ctx, n_actions):
    """Make infoset key from opponent's perspective."""
    pos = 1 if is_bb else 0
    init = 1 if hero_agg else (2 if villain_agg else 0)
    line = line_bucket(street_history)
    press = pressure_bucket(my_bet, opp_bet)
    board_bkt = board_bucket_for_street(community, street)
    board3 = community[:3] if len(community) >= 3 else community
    opp_disc_bkt = opp_discard_bucket(opp_disc, board3)
    hand_bkt = hand_bucket_for_street(hand2, community, street, dead)
    actx = _CTX_MAP.get(action_ctx, 0)

    buf = bytearray()
    buf.append(street)
    buf.append(pos)
    buf.append(init)
    buf.append(line)
    buf.append(press)
    buf.extend(struct.pack('<H', board_bkt))
    buf.append(opp_disc_bkt & 0xFF)
    buf.append(hand_bkt & 0xFF)
    buf.append(actx)
    buf.append(n_actions)
    return _fnv_hash(bytes(buf))


class BlueprintRange:
    """
    Track opponent's hand range using blueprint CFR strategies.
    
    After each opponent action, update the range by multiplying each
    possible hand's weight by the blueprint probability of that action.
    """

    def __init__(self, key_to_idx, probs, act_types, action_lists):
        self.key_to_idx = key_to_idx
        self.probs = probs
        self.act_types = act_types
        self.action_lists = action_lists

        # Range: dict of (c1, c2) -> weight
        self.range_weights = {}
        self.initialized = False

    def initialize(self, my_hand, community, my_discards, opp_discards):
        """Set initial uniform range over all possible opponent hands."""
        dead = set(my_hand) | set(community)
        for c in my_discards:
            if c >= 0: dead.add(c)
        for c in opp_discards:
            if c >= 0: dead.add(c)

        remaining = [c for c in ALL_CARDS if c not in dead]
        self.range_weights = {}
        for c1, c2 in combinations(remaining, 2):
            self.range_weights[(c1, c2)] = 1.0

        self.initialized = True

    def update_for_action(self, opp_action, observation,
                           my_hand, community, my_discards, opp_discards,
                           street_history, hero_agg, villain_agg):
        """
        Update range after opponent takes an action.
        
        For each possible opp hand, look up blueprint probability of 
        the observed action, and multiply the weight.
        
        opp_action: "CHECK", "CALL", "RAISE", "FOLD"
        """
        if not self.initialized or not self.range_weights:
            return

        street = observation.get("street", 1)
        # From opponent's perspective: my_bet/opp_bet are swapped
        hero_bet = observation.get("my_bet", 0)
        opp_bet = observation.get("opp_bet", 0)
        min_raise = observation.get("min_raise", 2)
        max_raise = observation.get("max_raise", 0)

        # Map action name to abstract action index
        action_map = {
            "CHECK": "CHECK", "CALL": "CALL", "FOLD": "FOLD",
            "RAISE": None,  # need to determine RAISE_SMALL or RAISE_LARGE
        }

        # Determine valid abstract actions from opp's perspective
        # From opp's view: their bet = our opp_bet, facing our bet = hero_bet
        valid = observation.get("valid_actions", [1, 1, 1, 1, 0])
        opp_valid_abs = get_valid_abstract_actions(valid, opp_bet, hero_bet, min_raise, max_raise)
        opp_ctx = get_action_context(valid, opp_bet, hero_bet, max_raise)

        # Find which abstract action matches the observed action
        if opp_action == "RAISE":
            # Could be RAISE_SMALL, RAISE_LARGE, BET_SMALL, or BET_LARGE
            # Default: assume it maps to the larger raise
            if "RAISE_LARGE" in opp_valid_abs:
                target_action = "RAISE_LARGE"
            elif "RAISE_SMALL" in opp_valid_abs:
                target_action = "RAISE_SMALL"
            elif "BET_LARGE" in opp_valid_abs:
                target_action = "BET_LARGE"
            elif "BET_SMALL" in opp_valid_abs:
                target_action = "BET_SMALL"
            else:
                return
        elif opp_action == "CHECK":
            target_action = "CHECK"
        elif opp_action == "CALL":
            target_action = "CALL"
        elif opp_action == "FOLD":
            # Fold removes from range entirely
            self.range_weights = {}
            return
        else:
            return

        if target_action not in opp_valid_abs:
            return

        target_idx = opp_valid_abs.index(target_action)
        n_actions = len(opp_valid_abs)

        # Dead cards from opponent's perspective
        dead = list(my_discards) + list(opp_discards)
        dead = [c for c in dead if c >= 0]

        # For each possible opp hand, look up blueprint probability
        new_weights = {}
        for (c1, c2), w in self.range_weights.items():
            if w <= 0:
                continue

            # Opponent's infoset key (from their perspective)
            opp_hand = [c1, c2]
            # Note: from opp's view, is_bb is flipped, hero/villain_agg are flipped
            is_opp_bb = not (observation.get("blind_position", 0) == 1)

            try:
                h = _make_postdiscard_key(
                    street, opp_hand, community, my_discards,
                    is_opp_bb, villain_agg, hero_agg,  # flipped
                    street_history, opp_bet, hero_bet,  # flipped
                    dead, opp_ctx, n_actions
                )

                idx = self.key_to_idx.get(h)
                if idx is not None:
                    stored_type = self.act_types[idx]
                    if stored_type < len(self.action_lists):
                        stored_actions = list(self.action_lists[stored_type])
                        if len(stored_actions) == n_actions:
                            raw = self.probs[idx, :n_actions].astype(np.float32)
                            total = raw.sum()
                            if total > 0:
                                action_prob = raw[target_idx] / total
                            else:
                                action_prob = 1.0 / n_actions
                            new_weights[(c1, c2)] = w * max(action_prob, 0.01)
                            continue

                # Blueprint miss: keep uniform probability for this action
                new_weights[(c1, c2)] = w * (1.0 / n_actions)
            except Exception:
                new_weights[(c1, c2)] = w * (1.0 / n_actions)

        # Normalize
        total = sum(new_weights.values())
        if total > 0:
            self.range_weights = {k: v / total for k, v in new_weights.items()}
        else:
            self.range_weights = new_weights

    def get_range(self, top_n=30):
        """Return top_n most likely hands with weights."""
        if not self.range_weights:
            return [], []

        sorted_hands = sorted(self.range_weights.items(), key=lambda x: -x[1])[:top_n]
        hands = [list(k) for k, _ in sorted_hands]
        weights = [v for _, v in sorted_hands]

        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]

        return hands, weights

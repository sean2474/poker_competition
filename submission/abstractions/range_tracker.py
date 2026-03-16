"""
Opponent range tracking via Bayesian updates.

Combines:
  1. Discard-aware range (what they likely kept based on what they threw away)
  2. Action-based narrowing (betting = stronger range, checking = weaker)

Each opponent hand candidate gets a weight that's updated based on observed actions.
"""

import numpy as np
from itertools import combinations
from abstractions.card_utils import (
    ALL_CARDS, get_evaluator, int_to_treys, card_rank, card_suit,
)
from abstractions.discard_oracle import _fast_score, estimate_opp_keep_weights, KEEP_PAIRS


def build_initial_range(my_hand, community, my_discards, opp_discards):
    """
    Build initial opponent range from all possible 2-card hands,
    weighted by discard likelihood.
    Returns: (opp_combos, weights) as lists
    """
    dead = set(my_hand) | set(community)
    for c in my_discards:
        if c >= 0: dead.add(c)
    for c in opp_discards:
        if c >= 0: dead.add(c)
    remaining = [c for c in ALL_CARDS if c not in dead]

    opp_combos = list(combinations(remaining, 2))

    # Get discard-aware weights
    board3 = community[:3] if len(community) >= 3 else community
    valid_opp_disc = [c for c in opp_discards if c >= 0]

    if len(valid_opp_disc) == 3:
        w_dict = estimate_opp_keep_weights(valid_opp_disc, board3, remaining)
        weights = [w_dict.get((c1, c2), 0.05) for c1, c2 in opp_combos]
    else:
        weights = [1.0] * len(opp_combos)

    # Normalize
    total = sum(weights)
    if total > 0:
        weights = [w / total for w in weights]
    else:
        weights = [1.0 / len(opp_combos)] * len(opp_combos)

    return opp_combos, weights


def compute_hand_equities(my_hand, community, opp_combos, num_sims=50):
    """
    Compute equity of each opp hand vs hero.
    Returns: array of opp equities in [0,1] (from opp's perspective).
    """
    ev = get_evaluator()
    my_h = [int_to_treys(c) for c in my_hand]
    board_need = 5 - len(community)
    dead_set = set(my_hand) | set(community)

    opp_equities = np.zeros(len(opp_combos))
    for oi, opp in enumerate(opp_combos):
        opp_h = [int_to_treys(c) for c in opp]
        used = dead_set | set(opp)
        rem = [c for c in ALL_CARDS if c not in used]

        if board_need == 0:
            b = [int_to_treys(c) for c in community[:5]]
            mr = ev.evaluate(my_h, b)
            opr = ev.evaluate(opp_h, b)
            opp_equities[oi] = 0.0 if mr < opr else (0.5 if mr == opr else 1.0)
        elif len(rem) >= board_need:
            import random
            wins = 0
            sims = min(num_sims, 50)
            for _ in range(sims):
                extra = random.sample(rem, board_need)
                b = [int_to_treys(c) for c in community + extra]
                mr = ev.evaluate(my_h, b)
                opr = ev.evaluate(opp_h, b)
                if mr > opr: wins += 1
                elif mr == opr: wins += 0.5
            opp_equities[oi] = wins / sims
        else:
            opp_equities[oi] = 0.5

    return opp_equities


def update_range_for_action(weights, opp_equities, action, pot_size, to_call):
    """
    Bayesian update: given opponent took 'action', adjust weights.

    Logic:
      - BET/RAISE: increase weight of strong hands, decrease weak
      - CHECK/CALL: slightly increase medium hands
      - FOLD: would remove from range, but we don't see this (hand ends)

    action: 'BET', 'RAISE', 'CHECK', 'CALL'
    """
    new_weights = list(weights)
    n = len(weights)

    if action in ('BET', 'RAISE'):
        # Opponents who bet/raise have stronger hands
        # P(bet | hand) ∝ equity^2 (strong hands bet more)
        for i in range(n):
            eq = opp_equities[i]
            # Strong hands: high likelihood of betting
            # Weak hands: low likelihood (but some bluff)
            bet_likelihood = eq * eq * 0.8 + 0.05  # min 5% bluff
            new_weights[i] *= bet_likelihood

    elif action == 'CALL':
        # Callers have medium-strong hands
        # Very weak hands fold, very strong hands raise
        for i in range(n):
            eq = opp_equities[i]
            # Bell curve around medium equity
            call_likelihood = max(0.05, 1.0 - abs(eq - 0.5) * 2)
            # But also strong hands sometimes slow-play
            if eq > 0.7:
                call_likelihood = max(call_likelihood, 0.4)
            new_weights[i] *= call_likelihood

    elif action == 'CHECK':
        # Checkers have weaker hands (but sometimes trap with strong)
        for i in range(n):
            eq = opp_equities[i]
            # Weak hands check most
            check_likelihood = (1.0 - eq) * 0.7 + 0.1
            # Strong hands sometimes check (trap)
            if eq > 0.7:
                check_likelihood = max(check_likelihood, 0.3)
            new_weights[i] *= check_likelihood

    # Normalize
    total = sum(new_weights)
    if total > 0:
        new_weights = [w / total for w in new_weights]
    else:
        new_weights = [1.0 / n] * n

    return new_weights


class OpponentRangeTracker:
    """
    Tracks opponent's likely hand range throughout a hand.
    Updated after each observed opponent action.
    """

    def __init__(self):
        self.opp_combos = []
        self.weights = []
        self.opp_equities = None
        self.initialized = False

    def initialize(self, my_hand, community, my_discards, opp_discards):
        """Set up initial range after discard phase."""
        self.opp_combos, self.weights = build_initial_range(
            my_hand, community, my_discards, opp_discards
        )
        if community and len(my_hand) == 2:
            self.opp_equities = compute_hand_equities(
                my_hand, community, self.opp_combos, num_sims=50
            )
        self.initialized = True

    def update(self, action, my_hand, community, pot_size=0, to_call=0):
        """Update range based on observed opponent action."""
        if not self.initialized or not self.opp_combos:
            return

        # Recompute equities if community cards changed
        if community and len(my_hand) == 2:
            self.opp_equities = compute_hand_equities(
                my_hand, community, self.opp_combos, num_sims=30
            )

        if self.opp_equities is not None:
            self.weights = update_range_for_action(
                self.weights, self.opp_equities, action, pot_size, to_call
            )

    def get_range(self, top_n=20):
        """
        Return top_n most likely opponent hands with weights.
        For use in MC equity or subgame solving.
        """
        if not self.initialized:
            return [], []

        # Sort by weight descending, take top_n
        indexed = sorted(enumerate(self.weights), key=lambda x: -x[1])[:top_n]
        combos = [list(self.opp_combos[i]) for i, _ in indexed]
        weights = [w for _, w in indexed]

        # Renormalize
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]

        return combos, weights

"""
Multi-street post-discard CFR solver.

Recursively solves from current street through river.
Each iteration samples a board runout and traverses the complete
game tree with proper terminal payoffs.

Key differences from 1-street solver:
  - Bet on flop → future streets have value (not just showdown)
  - Building pot early = more to win later
  - This is why strong hands should bet: they build the pot for future streets

State space: ~105 opp hands × 3 streets × ~8 nodes/street = manageable
Time budget: ~200ms per decision
"""

import random
import numpy as np
from itertools import combinations
from abstractions.card_utils import get_evaluator, int_to_treys, ALL_CARDS

_FOLD = 0
_RAISE = 1
_CHECK = 2
_CALL = 3
MAX_BET = 100
BIG_BLIND = 2


def _enumerate_opp_hands(my_hand, community, my_discards, opp_discards):
    dead = set(my_hand) | set(community)
    for c in my_discards:
        if c >= 0: dead.add(c)
    for c in opp_discards:
        if c >= 0: dead.add(c)
    remaining = [c for c in ALL_CARDS if c not in dead]
    return list(combinations(remaining, 2)), remaining


def _showdown_value(my_hand, opp, community, evaluator):
    """Returns hero payoff multiplier: +1 win, -1 lose, 0 tie."""
    my_h = [int_to_treys(c) for c in my_hand]
    opp_h = [int_to_treys(c) for c in opp]
    b = [int_to_treys(c) for c in community]
    mr = evaluator.evaluate(my_h, b)
    opr = evaluator.evaluate(opp_h, b)
    if mr < opr: return 1.0
    if mr > opr: return -1.0
    return 0.0


class MultiStreetNode:
    """A decision node in the multi-street game tree."""
    __slots__ = ['regret', 'strat_sum', 'n_actions']

    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.regret = np.zeros(n_actions)
        self.strat_sum = np.zeros(n_actions)

    def get_strategy(self, t):
        pos = np.maximum(self.regret, 0)
        total = pos.sum()
        strat = pos / total if total > 0 else np.ones(self.n_actions) / self.n_actions
        self.strat_sum += strat * max(t, 1)
        return strat

    def get_average(self):
        total = self.strat_sum.sum()
        return self.strat_sum / total if total > 0 else np.ones(self.n_actions) / self.n_actions


class MultiStreetSolver:
    """
    Multi-street CFR solver. Traverses flop→turn→river betting tree.
    """

    def __init__(self, my_hand, community, my_discards, opp_discards):
        self.my_hand = my_hand
        self.community = list(community)
        self.evaluator = get_evaluator()

        # Enumerate opp hands
        self.opp_hands, self.remaining = _enumerate_opp_hands(
            my_hand, community, my_discards or [], opp_discards or []
        )
        self.n_opp = len(self.opp_hands)

        # CFR nodes: keyed by (player, street, bets_tuple, opp_idx)
        self.nodes = {}

    def _get_node(self, key, n_actions):
        if key not in self.nodes:
            self.nodes[key] = MultiStreetNode(n_actions)
        return self.nodes[key]

    def _bet_sizes(self, pot, max_raise, min_raise):
        if max_raise <= 0 or max_raise < min_raise:
            return []
        sizes = []
        for frac in [0.5, 1.0]:
            amt = max(min_raise, min(int(pot * frac), max_raise))
            if amt not in sizes:
                sizes.append(amt)
        return sizes

    def solve(self, my_bet, opp_bet, min_raise, street, num_iters=100):
        """Run multi-street CFR. Returns (action_type, amount)."""
        for t in range(num_iters):
            # Sample board runout for remaining streets
            for oi in range(self.n_opp):
                opp = self.opp_hands[oi]
                opp_set = set(opp)
                deck = [c for c in self.remaining if c not in opp_set]

                # Deal remaining board cards
                board_need = 5 - len(self.community)
                if board_need > 0 and len(deck) >= board_need:
                    extra = random.sample(deck, board_need)
                else:
                    extra = []
                full_board = self.community + extra

                # Traverse from current street
                self._cfr_traverse(
                    oi, opp, full_board, my_bet, opp_bet, min_raise,
                    street, is_hero_turn=True, t=t
                )

        # Get root strategy
        pot = my_bet + opp_bet
        max_raise = MAX_BET - max(my_bet, opp_bet)
        bet_sizes = self._bet_sizes(pot, max_raise, min_raise)

        to_call = opp_bet - my_bet
        if to_call > 0:
            actions = ["FOLD", "CALL"] + [f"RAISE_{s}" for s in bet_sizes]
            amounts = [0, 0] + list(bet_sizes)
        else:
            actions = ["CHECK"] + [f"BET_{s}" for s in bet_sizes]
            amounts = [0] + list(bet_sizes)

        # Average over all opp hands
        avg_strat = np.zeros(len(actions))
        for oi in range(self.n_opp):
            key = ("hero", street, (my_bet, opp_bet), oi)
            node = self._get_node(key, len(actions))
            avg_strat += node.get_average()
        avg_strat /= self.n_opp

        total = avg_strat.sum()
        if total > 0:
            avg_strat /= total

        chosen = random.choices(range(len(actions)), weights=avg_strat.tolist(), k=1)[0]
        amt = amounts[chosen]

        if actions[chosen] == "FOLD":
            return (_FOLD, 0)
        elif actions[chosen] == "CHECK":
            return (_CHECK, 0)
        elif actions[chosen] == "CALL":
            return (_CALL, 0)
        else:
            return (_RAISE, amt)

    def _cfr_traverse(self, oi, opp, full_board, my_bet, opp_bet,
                       min_raise, street, is_hero_turn, t):
        """
        Recursive CFR traversal through multi-street game tree.
        Returns hero EV for this subtree.
        """
        pot = my_bet + opp_bet
        max_raise = MAX_BET - max(my_bet, opp_bet)
        to_call = (opp_bet - my_bet) if is_hero_turn else (my_bet - opp_bet)

        # Terminal: street > 3 = showdown
        if street > 3:
            board5 = full_board[:5]
            sd = _showdown_value(self.my_hand, opp, board5, self.evaluator)
            stake = min(my_bet, opp_bet)
            return sd * stake

        bet_sizes = self._bet_sizes(pot, max_raise, min_raise)

        if is_hero_turn:
            if to_call > 0:
                actions_n = 2 + len(bet_sizes)  # FOLD, CALL, RAISE...
            else:
                actions_n = 1 + len(bet_sizes)  # CHECK, BET...

            key = ("hero", street, (my_bet, opp_bet), oi)
            node = self._get_node(key, actions_n)
            strat = node.get_strategy(t)

            action_evs = np.zeros(actions_n)

            if to_call > 0:
                # FOLD
                action_evs[0] = -my_bet

                # CALL → advance street (or showdown if river)
                new_my = opp_bet
                if street == 3:
                    # River call → showdown
                    sd = _showdown_value(self.my_hand, opp, full_board[:5], self.evaluator)
                    action_evs[1] = sd * new_my
                else:
                    action_evs[1] = self._cfr_traverse(
                        oi, opp, full_board, new_my, opp_bet,
                        BIG_BLIND, street + 1, False, t  # opp acts first next street
                    )

                # RAISE sizes
                for si, sz in enumerate(bet_sizes):
                    new_my = opp_bet + sz
                    action_evs[2 + si] = self._cfr_traverse(
                        oi, opp, full_board, new_my, opp_bet,
                        min(sz, max_raise), street, False, t  # opp responds
                    )
            else:
                # CHECK → opp turn
                action_evs[0] = self._cfr_traverse(
                    oi, opp, full_board, my_bet, opp_bet,
                    min_raise, street, False, t  # opp acts
                )

                # BET sizes
                for si, sz in enumerate(bet_sizes):
                    new_my = my_bet + sz
                    action_evs[1 + si] = self._cfr_traverse(
                        oi, opp, full_board, new_my, opp_bet,
                        min(sz, max_raise), street, False, t  # opp responds
                    )

            ev = np.dot(strat, action_evs)
            node.regret = np.maximum(node.regret + (action_evs - ev), 0)
            return ev

        else:
            # Opponent's turn
            if to_call > 0:
                actions_n = 2 + len(bet_sizes)
            else:
                actions_n = 1 + len(bet_sizes)

            key = ("opp", street, (my_bet, opp_bet), oi)
            node = self._get_node(key, actions_n)
            strat = node.get_strategy(t)

            action_evs = np.zeros(actions_n)

            if to_call > 0:
                # OPP FOLD
                action_evs[0] = opp_bet  # hero wins opp's bet

                # OPP CALL → advance street
                new_opp = my_bet
                if street == 3:
                    sd = _showdown_value(self.my_hand, opp, full_board[:5], self.evaluator)
                    action_evs[1] = sd * my_bet
                else:
                    action_evs[1] = self._cfr_traverse(
                        oi, opp, full_board, my_bet, new_opp,
                        BIG_BLIND, street + 1, True, t
                    )

                # OPP RAISE
                for si, sz in enumerate(bet_sizes):
                    new_opp = my_bet + sz
                    action_evs[2 + si] = self._cfr_traverse(
                        oi, opp, full_board, my_bet, new_opp,
                        min(sz, max_raise), street, True, t  # hero responds
                    )
            else:
                # OPP CHECK → advance street (both checked)
                if street == 3:
                    sd = _showdown_value(self.my_hand, opp, full_board[:5], self.evaluator)
                    action_evs[0] = sd * min(my_bet, opp_bet)
                else:
                    action_evs[0] = self._cfr_traverse(
                        oi, opp, full_board, my_bet, opp_bet,
                        BIG_BLIND, street + 1, True, t  # hero first next street
                    )

                # OPP BET
                for si, sz in enumerate(bet_sizes):
                    new_opp = opp_bet + sz
                    action_evs[1 + si] = self._cfr_traverse(
                        oi, opp, full_board, my_bet, new_opp,
                        min(sz, max_raise), street, True, t  # hero responds
                    )

            # Opp minimizes hero EV (negate for opp regret)
            opp_evs = -action_evs
            opp_ev = np.dot(strat, opp_evs)
            node.regret = np.maximum(node.regret + (opp_evs - opp_ev), 0)
            return np.dot(strat, action_evs)  # return hero EV


def solve_subgame(my_hand, community, my_bet, opp_bet, min_raise, max_raise,
                   valid_actions, opp_hands=None, opp_weights=None,
                   num_iters=100, my_discards=None, opp_discards=None):
    """
    Multi-street solver entry point.
    Returns (action_type, raise_amount).
    """
    street = len([c for c in community if c >= 0])
    if street < 3:
        street_num = 1  # flop
    elif street < 4:
        street_num = 1
    elif street < 5:
        street_num = 2  # turn
    else:
        street_num = 3  # river

    solver = MultiStreetSolver(my_hand, community, my_discards, opp_discards)
    if solver.n_opp == 0:
        return None, None

    return solver.solve(my_bet, opp_bet, min_raise, street_num, num_iters)

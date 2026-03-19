"""
heuristic/prob_agent.py — Python HeuristicAgent wrapping the C++ prob_agent.

Load order:
  1. Try compiled shared library (cpp/prob_agent.so) via ctypes  → fast
  2. Fall back to pure Python implementation                      → portable

Interface mirrors agents/prob_agent.py but without the FastAPI/Agent base class
so it can be called directly from training loops.
"""

import ctypes
import os
import random
import sys

import numpy as np

# ── Try loading compiled C++ library ─────────────────────────────────────────

_LIB = None
_CPP_DIR = os.path.join(os.path.dirname(__file__), '..', 'cpp')

for _ext in ('so', 'dylib', 'dll'):
    _path = os.path.join(_CPP_DIR, f'prob_agent.{_ext}')
    if os.path.exists(_path):
        try:
            _LIB = ctypes.CDLL(_path)
            # mc_equity(int*, int, int*, int, int*, int, int, uint) -> float
            _LIB.mc_equity.restype  = ctypes.c_float
            _LIB.mc_equity.argtypes = [
                ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                ctypes.c_int, ctypes.c_uint,
            ]
            # best_discard(int*, int*, int, int*, int, int, uint, int*, int*)
            _LIB.best_discard.restype  = None
            _LIB.best_discard.argtypes = [
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                ctypes.c_int, ctypes.c_uint,
                ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
            ]
            break
        except Exception:
            _LIB = None

_USE_CPP = _LIB is not None


def _arr(lst):
    """Convert list to ctypes int array."""
    n = len(lst)
    return (ctypes.c_int * n)(*lst), n


# ── Shared evaluator (lazy-loaded once) ─────────────────────────────────────

_EVALUATOR   = None
_INT_TO_CARD = None

def _get_eval():
    global _EVALUATOR, _INT_TO_CARD
    if _EVALUATOR is None:
        from gym_env import PokerEnv, WrappedEval
        _EVALUATOR   = WrappedEval()
        _INT_TO_CARD = PokerEnv.int_to_card
    return _EVALUATOR, _INT_TO_CARD


# ── Pure Python MC equity — identical algorithm to agents/prob_agent.py ───────

def _py_compute_equity(my_cards, community_cards, opp_discards, num_sims=400):
    """MC equity: win fraction for my_cards vs random opponent.
    Uses WrappedEval (treys) — same evaluator as ProbabilityAgent."""
    evaluator, int_to_card = _get_eval()
    DECK_SIZE = 27

    shown = set(c for c in my_cards if c >= 0)
    shown.update(c for c in community_cards if c >= 0)
    shown.update(c for c in opp_discards    if c >= 0)

    non_shown    = [i for i in range(DECK_SIZE) if i not in shown]
    board_needed = 5 - len(community_cards)
    sample_size  = 2 + board_needed

    if sample_size > len(non_shown):
        return 0.5

    wins = valid = 0
    for _ in range(num_sims):
        sample     = random.sample(non_shown, sample_size)
        opp_pair   = sample[:2]
        full_board = list(community_cards) + sample[2:]
        if len(full_board) != 5:
            continue
        my_h  = [int_to_card(c) for c in my_cards]
        opp_h = [int_to_card(c) for c in opp_pair]
        brd   = [int_to_card(c) for c in full_board]
        ms  = evaluator.evaluate(my_h, brd)
        ops = evaluator.evaluate(opp_h, brd)
        if ms < ops:
            wins += 1
        valid += 1

    return wins / valid if valid > 0 else 0.0


def _cards_seed(my_cards, community_cards, opp_discards):
    """Deterministic seed from card values — same input always gives same seed."""
    all_cards = sorted(c for c in list(my_cards) + list(community_cards) + list(opp_discards) if c >= 0)
    val = 0
    for i, c in enumerate(all_cards):
        val ^= (c + 1) * (1000003 ** i)
    return val & 0xFFFFFFFF


# ── HeuristicAgent ────────────────────────────────────────────────────────────

class HeuristicAgent:
    """
    Stateless MC-equity heuristic agent.

    action(obs) → (action_type, amount, card_i, card_j)  — matches PokerEnv format
    compute_equity(my_cards, community, opp_discards, n_sims) → float
    best_discard(hand5, board, opp_discards, n_sims)           → (ki, kj)
    """

    _seed_ctr = 0  # monotonically increasing seed for C++ calls

    def _next_seed(self):
        HeuristicAgent._seed_ctr += 1
        return HeuristicAgent._seed_ctr

    # ── Public API ────────────────────────────────────────────────────────────

    def compute_equity(self, my_cards, community_cards,
                       opp_discards=None, num_sims=400):
        """Deterministic: same cards → same seed → same MC samples → same equity."""
        opp_discards = opp_discards or []
        random.seed(_cards_seed(my_cards, community_cards, opp_discards))
        return _py_compute_equity(my_cards, community_cards, opp_discards, num_sims)

    def best_discard(self, hand5, board, opp_discards=None, num_sims=200,
                     temperature: float = 0.05):
        """Slightly probabilistic: softmax over equity values with low temperature.
        Best pair still has the highest probability but is not always chosen.
        temperature=0 → greedy (always pick best), higher → more random.
        Returns (ki, kj): indices 0-4 of the 2 cards to keep."""
        import math
        opp_discards = opp_discards or []

        combos = [(i, j) for i in range(5) for j in range(i + 1, 5)]
        equities = [
            self.compute_equity([hand5[i], hand5[j]], board, opp_discards, num_sims)
            for i, j in combos
        ]

        if temperature <= 0:
            best = max(range(len(equities)), key=lambda k: equities[k])
            return combos[best]

        # Softmax with temperature
        max_eq = max(equities)
        weights = [math.exp((eq - max_eq) / temperature) for eq in equities]
        chosen = random.choices(combos, weights=weights)[0]
        return chosen

    def action(self, observation, num_sims_discard=200, num_sims_bet=400):
        """
        Produce an action given a PokerEnv observation dict.
        Returns (action_type_int, amount, card_i, card_j).
        """
        try:
            from gym_env import PokerEnv
            _FOLD, _RAISE, _CHECK, _CALL, _DISCARD = 0, 1, 2, 3, 4
            action_types = PokerEnv.ActionType
        except ImportError:
            _FOLD, _RAISE, _CHECK, _CALL, _DISCARD = 0, 1, 2, 3, 4

        my_cards_raw     = observation['my_cards']
        my_cards         = [c for c in my_cards_raw if c != -1]
        community_cards  = [c for c in observation.get('community_cards', []) if c != -1]
        opp_discards     = [c for c in observation.get('opp_discarded_cards', []) if c != -1]
        valid_actions    = observation['valid_actions']
        my_bet           = observation.get('my_bet', 0)
        opp_bet          = observation.get('opp_bet', 0)
        min_raise        = observation.get('min_raise', 2)
        max_raise        = observation.get('max_raise', 100)

        # ── Discard phase ─────────────────────────────────────────────────────
        if valid_actions[_DISCARD]:
            assert len(my_cards) == 5
            ki, kj = self.best_discard(my_cards, community_cards,
                                        opp_discards, num_sims_discard)
            return (_DISCARD, 0, ki, kj)

        # ── Betting phase ─────────────────────────────────────────────────────
        hand2 = my_cards[:2]
        equity = self.compute_equity(hand2, community_cards,
                                     opp_discards, num_sims_bet)

        to_call  = opp_bet - my_bet
        pot      = my_bet + opp_bet
        pot_odds = to_call / (to_call + pot) if to_call > 0 and pot > 0 else 0.

        if equity > 0.75 and valid_actions[_RAISE]:
            amount = int(pot * 0.75)
            amount = max(amount, min_raise)
            amount = min(amount, max_raise)
            return (_RAISE, amount, 0, 0)
        elif equity >= pot_odds and equity > 0.35 and valid_actions[_CALL]:
            return (_CALL, 0, 0, 0)
        elif valid_actions[_CHECK]:
            return (_CHECK, 0, 0, 0)
        elif equity >= pot_odds and valid_actions[_CALL]:
            return (_CALL, 0, 0, 0)
        else:
            return (_FOLD, 0, 0, 0)


# Singleton for training loops
_default_agent = None

def get_heuristic_agent() -> HeuristicAgent:
    global _default_agent
    if _default_agent is None:
        _default_agent = HeuristicAgent()
        backend = 'C++ (compiled)' if _USE_CPP else 'Python (fallback)'
        print(f'[HeuristicAgent] backend={backend}')
    return _default_agent

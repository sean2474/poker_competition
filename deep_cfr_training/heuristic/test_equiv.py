"""
heuristic/test_equiv.py — Verify that HeuristicAgent produces the same decisions
as agents/prob_agent.py (ProbabilityAgent).

Run from project root:
  python -m deep_cfr_training.heuristic.test_equiv
"""

import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np

from heuristic.prob_agent import HeuristicAgent, _USE_CPP

# ── Helpers ───────────────────────────────────────────────────────────────────

DECK_SIZE = 27

def random_hand(n, exclude=()):
    deck = [c for c in range(DECK_SIZE) if c not in exclude]
    return random.sample(deck, n)


def make_obs(hand2, community, opp_discards, my_bet=0, opp_bet=2,
             is_discard=False, hand5=None, street=1):
    cards = hand5 if (is_discard and hand5) else hand2
    valid = [0, 0, 0, 0, 0]
    if is_discard:
        valid[4] = 1  # DISCARD
    else:
        valid[0] = 1  # FOLD
        valid[3] = 1  # CALL
        valid[2] = 1  # CHECK
        valid[1] = 1  # RAISE
    return {
        'street':              street,
        'my_cards':            (cards + [-1] * 5)[:5],
        'community_cards':     (community + [-1] * 5)[:5],
        'opp_discarded_cards': (opp_discards + [-1] * 3)[:3],
        'valid_actions':       valid,
        'my_bet':              my_bet,
        'opp_bet':             opp_bet,
        'min_raise':           2,
        'max_raise':           100,
    }


# ── Test 1: HeuristicAgent determinism ───────────────────────────────────────

def test_determinism(n_cases=30, n_sims=200):
    """Same input must produce exactly identical output every call."""
    ours = HeuristicAgent()
    passed = True
    for _ in range(n_cases):
        hand5 = random_hand(5)
        board = random_hand(3, exclude=set(hand5))
        opp_d = random_hand(3, exclude=set(hand5) | set(board))
        hand2 = hand5[:2]

        # equity: call twice, must match exactly
        eq1 = ours.compute_equity(hand2, board, opp_d, num_sims=n_sims)
        eq2 = ours.compute_equity(hand2, board, opp_d, num_sims=n_sims)
        if eq1 != eq2:
            print(f'  equity mismatch: {eq1} != {eq2}')
            passed = False

        # discard: call twice, must match exactly
        k1 = ours.best_discard(hand5, board, opp_d, num_sims=n_sims)
        k2 = ours.best_discard(hand5, board, opp_d, num_sims=n_sims)
        if k1 != k2:
            print(f'  discard mismatch: {k1} != {k2}')
            passed = False

    print(f'[determinism]         {"PASS" if passed else "FAIL"}')
    return passed


# ── Test 2: same output as ProbabilityAgent (with same seed) ─────────────────

def test_equity_match(n_cases=30, n_sims=400, tol=0.02):
    """
    HeuristicAgent uses same algorithm + same seed as ProbabilityAgent.
    With same random.seed() before each call, outputs must be identical.
    """
    from agents.prob_agent import ProbabilityAgent
    ref  = ProbabilityAgent(stream=False)
    ours = HeuristicAgent()

    errors = []
    for _ in range(n_cases):
        hand2 = random_hand(2)
        board = random_hand(3, exclude=set(hand2))
        opp_d = random_hand(3, exclude=set(hand2) | set(board))

        # HeuristicAgent seeds internally; ref agent uses same seed externally
        from heuristic.prob_agent import _cards_seed
        seed = _cards_seed(hand2, board, opp_d)

        our_eq = ours.compute_equity(hand2, board, opp_d, num_sims=n_sims)
        random.seed(seed)   # set same seed before ref call
        ref_eq = ref._compute_equity(hand2, board, opp_d, num_simulations=n_sims)

        errors.append(abs(our_eq - ref_eq))

    all_zero = all(e == 0.0 for e in errors)
    mae = float(np.mean(errors))
    passed = all_zero
    print(f'[equity exact match]  all_zero={all_zero}  MAE={mae:.6f}  {"PASS" if passed else "FAIL"}')
    return passed


# ── Test 3: discard exact match ───────────────────────────────────────────────

def test_discard_exact(n_cases=50, n_sims=200):
    """
    With same random seed, HeuristicAgent and ProbabilityAgent must keep
    the exact same 2 cards.
    """
    from agents.prob_agent import ProbabilityAgent
    from heuristic.prob_agent import _cards_seed
    ref  = ProbabilityAgent(stream=False)
    ours = HeuristicAgent()

    mismatches = 0
    for _ in range(n_cases):
        hand5 = random_hand(5)
        board = random_hand(3, exclude=set(hand5))
        opp_d = random_hand(3, exclude=set(hand5) | set(board))

        # Our agent — internally deterministic
        ki, kj = ours.best_discard(hand5, board, opp_d, num_sims=n_sims)

        # Reference — set same seed before each pair evaluation
        best_ref = (0, 1); best_eq = -1.0
        for i in range(5):
            for j in range(i + 1, 5):
                seed = _cards_seed([hand5[i], hand5[j]], board, opp_d)
                random.seed(seed)
                eq = ref._compute_equity([hand5[i], hand5[j]], board, opp_d,
                                         num_simulations=n_sims)
                if eq > best_eq:
                    best_eq = eq; best_ref = (i, j)

        if (ki, kj) != best_ref:
            mismatches += 1

    passed = mismatches == 0
    print(f'[discard exact match] mismatches={mismatches}/{n_cases}  {"PASS" if passed else "FAIL"}')
    return passed


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print(f'Backend: {"C++ (compiled)" if _USE_CPP else "Python (fallback)"}')
    print('=' * 55)

    results = [
        test_determinism(),
        test_equity_match(),
        test_discard_exact(),
    ]

    print('=' * 55)
    all_passed = all(results)
    print(f'Overall: {"ALL PASS" if all_passed else "SOME TESTS FAILED"}')
    sys.exit(0 if all_passed else 1)

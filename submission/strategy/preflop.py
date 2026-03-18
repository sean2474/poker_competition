"""
strategy/preflop.py — Preflop tabular CFR strategy.

Loads preflop_chart.pkl (dict: key → np.ndarray[3]) and looks up
the strategy for the current hand/history.  Falls back to a simple
hand-strength heuristic when the key is not in the chart.
"""

import numpy as np
from features import card_rank, card_suit, NUM_RANKS
from action import BIG_BLIND, FOLD, RAISE, CALL, CHECK, bet_frac

# ── Preflop key helpers ───────────────────────────────────────────────────────

_SZ_T = [(2.6, 's'), (3.2, 'm'), (4.1, 'l')]
_A_CH = {0: 'f', 1: 'c', 2: 'k', 3: 'b', 4: 'B', 5: 'r', 6: 'R', 7: 'p'}


def canonicalize(hand5: list) -> tuple:
    """Suit-normalized sorted 5-card tuple (matches training preflop key)."""
    cnt = [0, 0, 0]
    for c in hand5: cnt[c // NUM_RANKS] += 1
    sm = {s: i for i, s in enumerate(sorted(range(3), key=lambda s: (-cnt[s], s)))}
    return tuple(sorted((c % NUM_RANKS) + sm[c // NUM_RANKS] * NUM_RANKS for c in hand5))


def size_bucket(max_bet: int) -> str:
    bb = max_bet / BIG_BLIND
    for t, l in _SZ_T:
        if bb <= t: return l
    return 'L'


def preflop_key(hand5: list, max_bet: int, action_history: list) -> tuple:
    """Infoset key: (canonical_hand5, size_bucket, history_string)."""
    hist = ''.join(_A_CH.get(a, '?') for a in action_history)
    return (canonicalize(hand5), size_bucket(max_bet), hist)


# ── Preflop strategy lookup ───────────────────────────────────────────────────

def preflop_action(obs: dict, chart: dict, action_history: list) -> tuple:
    """
    Returns (action_type, raise_amount, 0, 0) for preflop street.

    chart: dict loaded from deep_cfr_preflop_chart.pkl
    action_history: list of training action indices taken so far this hand
    """
    v = obs['valid_actions']
    hand5 = [c for c in obs['my_cards'] if c >= 0]

    assert chart, 'preflop chart must be loaded'
    assert len(hand5) == 5, f'Expected 5-card hand at preflop, got {len(hand5)}: {hand5}'

    max_bet = max(obs['my_bet'], obs['opp_bet'])

    key   = preflop_key(hand5, max_bet, action_history)
    strat = chart.get(key)
    assert strat is not None, f'preflop chart missing key: {key}'

    total = float(strat.sum())
    assert total > 0, f'preflop strategy sums to zero for key: {key}'

    probs = strat / total   # 3-slot: [fold, call/check, raise]
    slot  = int(np.random.choice(3, p=probs.astype(np.float64)))
    if slot == 0 and v[FOLD]:  return (FOLD, 0, 0, 0)
    if slot == 2 and v[RAISE]:
        pot = obs['my_bet'] + obs['opp_bet']
        amt = min(obs['max_raise'], max(obs['min_raise'], pot // 2 + 1))
        return (RAISE, amt, 0, 0)
    return (CALL, 0, 0, 0) if v[CALL] else (CHECK, 0, 0, 0)

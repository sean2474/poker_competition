"""
strategy/preflop.py — Preflop tabular CFR strategy.

Loads preflop_chart.pkl (dict: key → np.ndarray[3]) and looks up
the strategy for the current hand/history.  Falls back to a
hand-strength heuristic (hand_role + apply_size_adjustment) when the
key is not in the chart (e.g. off-size opens from AllIn/Random agents).
"""

import numpy as np
from features import NUM_RANKS
from action import BIG_BLIND, FOLD, RAISE, CALL, CHECK

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


# ── Fallback call-frequency tracking ────────────────────────────────────────

_stats = {'total': 0, 'hit': 0, 'size_adj': 0, 'blind_call': 0}


def get_fallback_stats() -> dict:
    """Return fallback frequency stats (reset with reset_fallback_stats)."""
    s = _stats
    miss = s['size_adj'] + s['blind_call']
    pct  = 100.0 * miss / s['total'] if s['total'] else 0.0
    return {
        'total':      s['total'],
        'chart_hit':  s['hit'],
        'size_adj':   s['size_adj'],
        'blind_call': s['blind_call'],
        'fallback_%': round(pct, 2),
    }


def reset_fallback_stats() -> None:
    _stats.update({'total': 0, 'hit': 0, 'size_adj': 0, 'blind_call': 0})


# ── Off-size fallback: look up 2.5bb ('s') key and MDF-adjust ────────────────

_TRAIN_BET = int(2.5 * BIG_BLIND)   # 5 chips — the fixed open used in training
_DEAD      = BIG_BLIND * 1.5        # SB(1)+BB(2) = 3 chips


def _adjust_strat_for_size(strat: np.ndarray, actual_bet: int) -> np.ndarray:
    """Scale fold% using MDF ratio: training 2.5bb → actual bet size."""
    mdf_train  = _DEAD / (_DEAD + _TRAIN_BET)
    mdf_actual = _DEAD / (_DEAD + max(actual_bet, 1))
    p = strat / strat.sum()
    cont_new  = min(1.0, (1.0 - p[0]) * (mdf_actual / mdf_train))
    fold_new  = 1.0 - cont_new
    cr_sum = p[1] + p[2]
    if cr_sum > 0:
        call_new  = p[1] / cr_sum * cont_new
        raise_new = p[2] / cr_sum * cont_new
    else:
        call_new, raise_new = cont_new, 0.0
    return np.array([fold_new, call_new, raise_new])


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

    _stats['total'] += 1
    if strat is None or strat.sum() <= 0:
        # Off-size open: look up training 2.5bb key and MDF-adjust
        other_bucket = 'L' if key[1] == 's' else 's'
        train_key  = (key[0], other_bucket, key[2])
        strat_base = chart.get(train_key)
        if strat_base is not None and strat_base.sum() > 0:
            _stats['size_adj'] += 1
            strat = _adjust_strat_for_size(strat_base, max_bet)
        else:
            _stats['blind_call'] += 1
            return (CALL, 0, 0, 0) if v[CALL] else (CHECK, 0, 0, 0)
    else:
        _stats['hit'] += 1

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

"""
strategy/preflop.py — Preflop tabular CFR strategy.

Loads preflop_chart.pkl (dict: key → np.ndarray[3]) and looks up
the strategy for the current hand/history.  Falls back to a simple
hand-strength heuristic when the key is not in the chart.
"""

import numpy as np
from collections import Counter
from features import card_rank, card_suit, NUM_RANKS
from action import BIG_BLIND, FOLD, RAISE, CALL, CHECK, bet_frac

# ── Preflop key helpers ───────────────────────────────────────────────────────

_SZ_T    = [(2.6, 's'), (3.2, 'm'), (4.1, 'l')]
_A_CH    = {0: 'f', 1: 'c', 2: 'k', 3: 'b', 4: 'B', 5: 'r', 6: 'R', 7: 'p'}
_SZ_ALL  = ['s', 'm', 'l', 'L']   # fallback search order
_BASE_BB = 2.5                     # training baseline open size


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


# ── Hand strength score ────────────────────────────────────────────────────

def _hand_score(hand5: list) -> float:
    """0–1 preflop strength: trips+=0.75, pair=0.45-0.7, high card=0.0-0.44."""
    ranks = sorted([card_rank(c) for c in hand5], reverse=True)
    suits = [card_suit(c) for c in hand5]
    rc    = Counter(ranks)
    max_c = max(rc.values())
    if max_c >= 3:
        tr = max(r for r, c in rc.items() if c >= 3)
        return min(0.75 + tr / 32.0, 1.0)
    pairs = sorted([r for r, c in rc.items() if c >= 2], reverse=True)
    if len(pairs) >= 2:
        return min(0.65 + pairs[0] / 32.0, 0.90)
    if pairs:
        return min(0.45 + pairs[0] / 20.0, 0.75)
    top2  = (ranks[0] + ranks[1]) / (2.0 * (NUM_RANKS - 1))
    flush = 0.07 if len(set(suits)) <= 2 else 0.0
    return min(top2 * 0.65 + flush, 0.44)


# ── Open-size adjustment (logit-based) ─────────────────────────────────────

def _adjust_for_sizing(strat: np.ndarray, size_ratio: float,
                       hand_score: float) -> np.ndarray:
    """
    Adjust [fold, call, raise] probs for oversized opens via logit penalization.
    Weak hands: call penalized first, bluff-raise second, value-raise least.
    """
    if size_ratio <= 1.1:
        return strat
    strat  = np.maximum(strat, 1e-9)
    logits = np.log(strat)
    alpha  = min(size_ratio - 1.0, 2.5)          # cap at size_ratio=3.5
    weak   = 1.0 - hand_score                    # 1=very weak, 0=very strong
    logits[1] -= alpha * 2.5 * weak              # call penalty
    logits[2] -= alpha * 1.8 * (weak * 0.85)    # bluff-raise penalty (slightly less)
    logits    -= logits.max()
    probs      = np.exp(logits)
    return probs / probs.sum()


# ── Preflop strategy lookup ────────────────────────────────────────────────

def preflop_action(obs: dict, chart: dict, action_history: list) -> tuple:
    """
    Returns (action_type, raise_amount, 0, 0) for preflop street.

    chart: dict loaded from deep_cfr_preflop_chart.pkl
    action_history: list of training action indices taken so far this hand
    """
    v     = obs['valid_actions']
    hand5 = [c for c in obs['my_cards'] if c >= 0]

    assert chart,          'preflop chart must be loaded'
    assert len(hand5) == 5, f'Expected 5-card hand at preflop, got {len(hand5)}'

    max_bet    = max(obs['my_bet'], obs['opp_bet'])
    open_bb    = max_bet / BIG_BLIND
    size_ratio = open_bb / _BASE_BB          # ratio vs training baseline (2.5bb)

    # ── Chart lookup with size-bucket fallback ────────────────────────────
    hand_key  = canonicalize(hand5)
    hist_str  = ''.join(_A_CH.get(a, '?') for a in action_history)
    strat     = chart.get((hand_key, size_bucket(max_bet), hist_str))
    if strat is None:                        # try other size buckets
        for sz in _SZ_ALL:
            strat = chart.get((hand_key, sz, hist_str))
            if strat is not None:
                break
    if strat is None:                        # pure heuristic fallback
        sc    = _hand_score(hand5)
        fold  = max(0.05, 0.55 - sc)
        raise_ = sc * 0.45
        call  = max(0.0, 1.0 - fold - raise_)
        strat = np.array([fold, call, raise_], dtype=np.float32)

    total = float(strat.sum())
    if total <= 0:
        strat = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        strat = strat / total

    # ── Adjust for open size vs 2.5bb baseline ────────────────────────────
    score = _hand_score(hand5)
    strat = _adjust_for_sizing(strat, size_ratio, score)

    slot = int(np.random.choice(3, p=strat.astype(np.float64)))
    if slot == 0 and v[FOLD]:  return (FOLD, 0, 0, 0)
    if slot == 2 and v[RAISE]:
        # 3-bet sizing: proportional to open (IP ~3x, OOP ~3.5x)
        # Use is_bb as OOP proxy (BB is OOP preflop)
        is_oop  = obs.get('acting_agent', 0) == 1   # BB = player 1 = OOP
        mult    = 3.5 if is_oop else 3.0
        amt     = int(max_bet * mult)
        amt     = min(obs['max_raise'], max(obs['min_raise'], amt))
        return (RAISE, amt, 0, 0)
    return (CALL, 0, 0, 0) if v[CALL] else (CHECK, 0, 0, 0)

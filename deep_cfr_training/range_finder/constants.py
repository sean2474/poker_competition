"""Shared constants, pair-index helpers, and NNUE card encoders."""

import numpy as np

N_HANDS = 351   # C(27,2)

N_CATS = 17
CATEGORY_NAMES = [
    'straight_flush',                         # 0
    'full_house',                             # 1
    'flush',                                  # 2
    'straight',                               # 3
    'top_set', 'middle_set', 'bottom_set',    # 4-6
    'two_pair', 'bottom_two_pair',            # 7-8
    'overpair',                               # 9
    'top_pair', 'middle_pair', 'bottom_pair', # 10-12
    'sf_draw', 'flush_draw', 'straight_draw', # 13-15
    'high_card',                              # 16
]

# ── Pair index ────────────────────────────────────────────────────────────────

_PAIR_IDX: dict = {}
_idx = 0
for _a in range(27):
    for _b in range(_a + 1, 27):
        _PAIR_IDX[(_a, _b)] = _idx
        _idx += 1


def pidx(c0: int, c1: int) -> int:
    """Canonical index for a 2-card pair (order-independent, 0-350)."""
    return _PAIR_IDX[(c0, c1)] if c0 < c1 else _PAIR_IDX[(c1, c0)]


# ── NNUE card encoding (mirrors features.h card_features) ────────────────────

_NUM_RANKS = 9


def card_feat4(c: int) -> np.ndarray:
    """4-dim card encoding: [rank/8, suit0, suit1, suit2].  c=-1 → zeros."""
    f = np.zeros(4, dtype=np.float32)
    if c >= 0:
        f[0] = (c % _NUM_RANKS) / 8.
        f[1 + c // _NUM_RANKS] = 1.
    return f


def hand2_feat20(c0: int, c1: int) -> np.ndarray:
    """20-dim hero_hand slice (dims 0-19 of 119-dim feature vector).
    Sorted low→high to match C++ state_to_features ordering."""
    f = np.zeros(20, dtype=np.float32)
    lo, hi = (c0, c1) if c0 < c1 else (c1, c0)
    f[0:4] = card_feat4(lo)
    f[4:8] = card_feat4(hi)
    return f

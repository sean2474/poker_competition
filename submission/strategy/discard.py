"""
strategy/discard.py — Discard decision (Phase 2+).

Uses DiscardNet if loaded; falls back to fast_score heuristic.
Feature assembly mirrors discard_cfr/features.py (44-dim NNUE).
"""

import numpy as np
from features import (
    classify_hand, compute_blocker_flags,
    KEEP_PAIRS, opp_range_cats as _opp_range_cats,
)
from action import DISCARD

# ── Feature builders (mirrors discard_cfr/features.py) ───────────────────────

_NUM_RANKS = 9


def _pair_features(hand5: list, board3: list) -> np.ndarray:
    """(10, 23) = my_cat(17) + hi_rank(1) + lo_rank(1) + blocker(4) per keep pair."""
    b5 = (board3 + [-1]*5)[:5]
    n  = sum(1 for c in board3 if c >= 0)
    out = np.zeros((10, 23), dtype=np.float32)
    for k, (ai, aj) in enumerate(KEEP_PAIRS):
        c0, c1 = hand5[ai], hand5[aj]
        cat_oh = np.zeros(17, dtype=np.float32)
        cat_oh[classify_hand(c0, c1, b5, n)] = 1.
        hi = max(c0 % _NUM_RANKS, c1 % _NUM_RANKS) / 8.
        lo = min(c0 % _NUM_RANKS, c1 % _NUM_RANKS) / 8.
        blk = compute_blocker_flags(c0, c1, b5, n)
        out[k] = np.concatenate([cat_oh, [hi, lo], blk])
    return out   # (10, 23)


def _ctx_features(board3: list, opp_cats: np.ndarray, is_bb: bool) -> np.ndarray:
    """(21,) = board_ranks(3) + opp_cats(17) + is_bb(1)."""
    brs = np.array([(c % _NUM_RANKS) / 8. for c in board3[:3]], dtype=np.float32)
    return np.concatenate([brs, opp_cats, [1. if is_bb else 0.]])


def build_discard_feats(hand5: list, board3: list,
                        opp_cats: np.ndarray, is_bb: bool) -> np.ndarray:
    """(10, 44) feature matrix for all keep pairs."""
    pair = _pair_features(hand5, board3)          # (10, 23)
    ctx  = np.tile(_ctx_features(board3, opp_cats, is_bb), (10, 1))  # (10, 21)
    return np.concatenate([pair, ctx], axis=1)    # (10, 44)


def _opp_cats_from_obs(hand5: list, board3: list, obs: dict) -> np.ndarray:
    """17-dim opp range from observed discards."""
    opp_disc = [c for c in obs.get('opp_discarded_cards', [-1]*3) if c >= 0]
    n        = sum(1 for c in board3 if c >= 0)
    board    = board3 + [-1] * (5 - len(board3))
    return _opp_range_cats(hand5, opp_disc, board, n)


# ── Discard decision ──────────────────────────────────────────────────────────

def decide_discard(obs: dict, discard_net=None) -> tuple:
    """
    Returns (DISCARD, 0, keep_idx_i, keep_idx_j).

    discard_net: loaded DiscardNet or None (falls back to fast_score heuristic).
    is_bb: BB discards second (knows opp discards).
    """
    hand5  = [c for c in obs['my_cards'] if c >= 0]
    comm   = obs.get('community_cards', [-1]*5)
    board3 = [c for c in comm if c >= 0][:3]
    is_bb  = obs.get('acting_agent', 0) == 1   # BB is player 1

    assert len(hand5) == 5, f'Expected 5-card hand at discard, got {len(hand5)}: {hand5}'
    assert discard_net is not None, 'discard_net must be loaded'

    opp_cats = _opp_cats_from_obs(hand5, board3, obs)
    feats    = build_discard_feats(hand5, board3, opp_cats, is_bb)
    strat    = discard_net.get_strategy(feats.astype(np.float32))
    strat    = strat.astype(np.float64); strat /= strat.sum()
    ka       = int(np.random.choice(10, p=strat))
    ki, kj   = KEEP_PAIRS[ka]
    return (DISCARD, 0, ki, kj)

"""
Feature extraction for discard CFR.

Design: category-based NNUE split.
  pair  (19-dim): my_cat[17] + my_hi_rank[1] + my_lo_rank[1]  — per keep pair
  ctx   (20-dim): board_ranks[3] + opp_cats[17]               — shared across 10 pairs

Full feature per sample = pair(19) + ctx(20) = FEAT_DIM(39)

Single network for both Player A and B — only opp_cats differs:
  A: opp_cats = uniform range (no opp info)
  B: opp_cats = oracle-narrowed range (knows opp_disc3)
"""

import numpy as np

N_KEEP_PAIRS = 10
KEEP_PAIRS   = [(i, j) for i in range(5) for j in range(i + 1, 5)]

N_CATS    = 17   # must match range_finder.N_CATS
NUM_RANKS = 9

PAIR_DIM = N_CATS + 2 + 4    # my_cat(17) + hi_rank(1) + lo_rank(1) + blocker(4) = 23
CTX_DIM  = 3 + N_CATS + 1   # board_ranks(3) + opp_cats(17) + is_bb(1) = 21
FEAT_DIM = PAIR_DIM + CTX_DIM                                       # 44


# ── Range helpers ─────────────────────────────────────────────────────────────

def opp_cats_uniform(hand5, board3) -> np.ndarray:
    """17-dim opp range probs for Player A (uniform prior, dead = hand5)."""
    from range_finder import RangeFinder
    rf = RangeFinder()
    rf.init(dead_cards=list(hand5))
    rf.remove_cards(list(board3))
    return rf.category_probs(list(board3), threshold=0.0)


def opp_cats_narrowed(hand5, board3, opp_disc3) -> np.ndarray:
    """17-dim opp range probs for Player B (oracle-narrowed by opp_disc3)."""
    from range_finder import RangeFinder
    rf = RangeFinder()
    rf.init(dead_cards=list(hand5))
    rf.remove_cards(list(board3))
    rf.update_discard(list(opp_disc3), list(board3))
    return rf.category_probs(list(board3), threshold=0.0)


# ── Per-pair classification ───────────────────────────────────────────────────

def classify_all_pairs(hand5, board3) -> np.ndarray:
    """
    Classify all 10 keep pairs of hand5 on board3.
    Returns float32(10, 17) — one-hot category per pair.
    Uses c_classify_hand directly (3x faster than RangeFinder approach).
    """
    import ctypes
    from game.features import _c_lib
    n     = sum(1 for c in board3 if c >= 0)
    board = (ctypes.c_int * 5)(*[int(board3[i]) if i < len(board3) and board3[i] >= 0
                                   else -1 for i in range(5)])
    cats  = np.zeros((N_KEEP_PAIRS, N_CATS), dtype=np.float32)
    for k, (ai, aj) in enumerate(KEEP_PAIRS):
        cat = _c_lib.c_classify_hand(int(hand5[ai]), int(hand5[aj]), board, n)
        cats[k, cat] = 1.
    return cats   # (10, 17)


def pair_rank_feats(hand5) -> np.ndarray:
    """(10, 2) normalized [hi_rank, lo_rank] for each keep pair."""
    ranks = np.zeros((N_KEEP_PAIRS, 2), dtype=np.float32)
    for k, (ai, aj) in enumerate(KEEP_PAIRS):
        hi = max(hand5[ai] % NUM_RANKS, hand5[aj] % NUM_RANKS)
        lo = min(hand5[ai] % NUM_RANKS, hand5[aj] % NUM_RANKS)
        ranks[k] = [hi / 8., lo / 8.]
    return ranks   # (10, 2)


def pair_blocker_feats(hand5, board3) -> np.ndarray:
    """
    (10, 4) blocker flags per keep pair using shared c_blocker_flags.
    Mirrors compute_blocker_flags in cpp/hand_eval.h.
    """
    import ctypes
    from game.features import _c_lib
    board_n = sum(1 for c in board3 if c >= 0)
    board_c = (ctypes.c_int * 5)(*[board3[i] if i < len(board3) and board3[i] >= 0
                                    else -1 for i in range(5)])
    out = np.zeros((N_KEEP_PAIRS, 4), dtype=np.float32)
    tmp = (ctypes.c_float * 4)()
    for k, (ai, aj) in enumerate(KEEP_PAIRS):
        c0, c1 = int(hand5[ai]), int(hand5[aj])
        _c_lib.c_blocker_flags(c0, c1, board_c, ctypes.c_int(board_n), tmp)
        out[k] = list(tmp)
    return out   # (10, 4)


# ── Feature assembly ──────────────────────────────────────────────────────────

def build_all_feats(pair_cats: np.ndarray,
                    pair_ranks: np.ndarray,
                    pair_blockers: np.ndarray,
                    board3,
                    opp_cats: np.ndarray,
                    is_bb: bool = False) -> np.ndarray:
    """
    Assemble (10, 44) feature matrix from pre-computed components.

    pair_cats     : (10, 17)  — from classify_all_pairs
    pair_ranks    : (10,  2)  — from pair_rank_feats
    pair_blockers : (10,  4)  — from pair_blocker_feats
    board3        : list[3]
    opp_cats      : (17,)     — from opp_cats_uniform or opp_cats_narrowed
    is_bb         : True if this player discards SECOND (knows opp discards)

    Returns float32(10, 44): [pair_cat|pair_ranks|pair_blockers | board_ranks|opp_cats|is_bb]
    """
    pair_part = np.concatenate([pair_cats, pair_ranks, pair_blockers], axis=1)  # (10, 23)
    board_r   = np.array([c % NUM_RANKS / 8. for c in board3],
                         dtype=np.float32)                                       # (3,)
    ctx       = np.concatenate([board_r, opp_cats,
                                 np.array([1. if is_bb else 0.],
                                          dtype=np.float32)])                    # (21,)
    ctx_tiled = np.tile(ctx, (N_KEEP_PAIRS, 1))                                 # (10, 21)
    return np.concatenate([pair_part, ctx_tiled], axis=1)                       # (10, 44)

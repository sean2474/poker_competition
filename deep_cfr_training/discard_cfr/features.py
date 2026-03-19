"""
Discard CFR feature extraction — 44-dim NNUE per keep-pair sample.

PAIR features (23-dim per pair):
  [0:17]  hand category one-hot (17 classes)
  [17]    hi rank / 8
  [18]    lo rank / 8
  [19:23] blocker flags

CTX features (21-dim, shared across all 10 pairs of same game):
  [0:3]   board ranks / 8
  [3:20]  opp_cats[17]  — opponent range category distribution
  [20]    is_bb

FEAT_DIM = 23 + 21 = 44

opp_cats computation:
  Player A (SB): uniform prior over B's possible hands given A's dead cards + board
                 = category distribution weighted by range probability
  Player B (BB): narrowed by A's actual 3 discarded cards (known at B's turn)
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Re-use submission's Python classify_hand and range utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'submission'))
from features import (classify_hand as _classify_hand,
                      compute_blocker_flags as _blocker_flags,
                      _ALL_PAIRS, _range_uniform, _range_update_discard,
                      _range_to_cats)

KEEP_PAIRS   = [(i, j) for i in range(5) for j in range(i + 1, 5)]
N_KEEP_PAIRS = 10
PAIR_DIM     = 23
CTX_DIM      = 21
FEAT_DIM     = PAIR_DIM + CTX_DIM   # 44
NUM_RANKS    = 9
NUM_SUITS    = 3
N_CATS       = 17


# ── Single-game Python feature builders ───────────────────────────────────────

def classify_all_pairs(hand5: list, board3: list) -> np.ndarray:
    """(10, 17) one-hot hand category per keep pair."""
    b5  = (list(board3) + [-1, -1])[:5]
    n   = sum(1 for c in board3 if c >= 0)
    out = np.zeros((N_KEEP_PAIRS, N_CATS), dtype=np.float32)
    for k, (ai, aj) in enumerate(KEEP_PAIRS):
        out[k, _classify_hand(hand5[ai], hand5[aj], b5, n)] = 1.
    return out


def pair_rank_feats(hand5: list) -> np.ndarray:
    """(10, 2) = [hi_rank/8, lo_rank/8] per pair."""
    out = np.zeros((N_KEEP_PAIRS, 2), dtype=np.float32)
    for k, (ai, aj) in enumerate(KEEP_PAIRS):
        r0, r1 = hand5[ai] % NUM_RANKS, hand5[aj] % NUM_RANKS
        out[k, 0] = max(r0, r1) / 8.
        out[k, 1] = min(r0, r1) / 8.
    return out


def pair_blocker_feats(hand5: list, board3: list) -> np.ndarray:
    """(10, 4) blocker flags per pair."""
    b5  = (list(board3) + [-1, -1])[:5]
    n   = sum(1 for c in board3 if c >= 0)
    out = np.zeros((N_KEEP_PAIRS, 4), dtype=np.float32)
    for k, (ai, aj) in enumerate(KEEP_PAIRS):
        out[k] = _blocker_flags(hand5[ai], hand5[aj], b5, n)
    return out


def build_all_feats(pcats: np.ndarray, pranks: np.ndarray,
                    pblk: np.ndarray, board3: list,
                    opp_cats: np.ndarray, is_bb: bool) -> np.ndarray:
    """(10, 44) full feature matrix."""
    pair = np.concatenate([pcats, pranks, pblk], axis=1)   # (10, 23)
    brs  = np.array([(c % NUM_RANKS) / 8. for c in board3[:3]], dtype=np.float32)
    ctx  = np.concatenate([brs, opp_cats, [1. if is_bb else 0.]])   # (21,)
    return np.concatenate([pair, np.tile(ctx, (N_KEEP_PAIRS, 1))], axis=1)  # (10,44)


def opp_cats_uniform(hand5: list, board3: list) -> np.ndarray:
    """
    17-dim category distribution of B's possible hands given A's dead cards.
    Dead = hand5 (5 cards) + board3 (3 cards). Range weighted by category.
    """
    dead  = [c for c in hand5 if c >= 0] + [c for c in board3 if c >= 0]
    probs = _range_uniform(dead)
    b5    = (list(board3) + [-1, -1])[:5]
    n     = sum(1 for c in board3 if c >= 0)
    return _range_to_cats(probs, b5, n)


def opp_cats_narrowed(hand5_B: list, board3: list, a_disc: list) -> np.ndarray:
    """
    17-dim category distribution for B's range knowing A discarded a_disc.
    Dead = hand5_B + board3. Range = uniform then remove a_disc overlap.
    """
    dead  = [c for c in hand5_B if c >= 0] + [c for c in board3 if c >= 0]
    probs = _range_uniform(dead)
    disc  = [c for c in a_disc if c >= 0]
    if disc:
        probs = _range_update_discard(probs, disc)
    b5 = (list(board3) + [-1, -1])[:5]
    n  = sum(1 for c in board3 if c >= 0)
    return _range_to_cats(probs, b5, n)


# ── Batch helpers for traversal.py ────────────────────────────────────────────

def opp_cats_uniform_batch(h5A: np.ndarray, b3: np.ndarray) -> np.ndarray:
    """
    [N, 17] — batch opp_cats_uniform for Player A.
    h5A: (N, 5), b3: (N, 3).
    Weights B's possible hands by range probability given A's dead cards.
    """
    N   = len(h5A)
    out = np.zeros((N, N_CATS), dtype=np.float32)
    for n in range(N):
        out[n] = opp_cats_uniform(h5A[n].tolist(), b3[n].tolist())
    return out

"""
Discard CFR traversal — EV-based hybrid strategy.

For each game (hand5_A, hand5_B, board3):
  1. Compute EV matrix[10×10] via C++ MC runouts
  2. Player A: net forward → EV predictions
               candidates = {k: ev_pred[k] >= max - EV_THRESHOLD}
               strat_A = softmax(ev_pred, candidates)
  3. Player B: same (using narrowed opp_cats from A's actual discard)
  4. Ground truth ev_A = einsum(strat_B, ev_mats)
     target_A = ev_A - mean(ev_A)   ← mean-centered EV (MSE target)
  5. Sample ka_actual from strat_A, compute target_B similarly

Returns (2*N*10, 44) features and (2*N*10,) EV targets.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import ctypes
import numpy as np
import torch

from game.features import _c_lib
from .features import (KEEP_PAIRS, N_KEEP_PAIRS, FEAT_DIM, PAIR_DIM, CTX_DIM,
                       opp_cats_uniform_batch)
from .model import EV_THRESHOLD, SOFTMAX_T

_ALL_CARDS = list(range(27))

# Precomputed discard index lookup: for keep pair (i,j), which 3 indices are discarded
_KP_IDX   = np.array([[i, j] for i in range(5) for j in range(i+1, 5)], dtype=np.int32)  # [10,2]
_DISC_IDX = np.array([[k for k in range(5) if k not in (i, j)]
                       for i, j in [(a, b) for a in range(5) for b in range(a+1, 5)]],
                      dtype=np.int32)   # [10,3]


def _softmax_candidates(ev_out: np.ndarray, threshold: float, T: float) -> np.ndarray:
    """
    ev_out: [N, 10] net EV predictions
    Returns strat: [N, 10] — softmax within candidates, 0 outside.
    """
    best  = ev_out.max(axis=1, keepdims=True)            # [N, 1]
    mask  = ev_out >= best - threshold                    # [N, 10]
    ev_c  = ev_out - best                                 # [N, 10] shifted
    exp_v = np.exp(np.clip(ev_c * T, -20., 0.))          # [N, 10]
    exp_v[~mask] = 0.
    s     = exp_v.sum(axis=1, keepdims=True)
    # fallback: uniform over candidates when all exp underflow
    fallback = mask.astype(np.float32) / np.maximum(mask.sum(axis=1, keepdims=True), 1)
    return np.where(s > 1e-9, exp_v / np.where(s > 1e-9, s, 1.), fallback)


def run_batch(hand5_As, hand5_Bs, boards5, net,
              iteration: float, n_mc: int = 8,
              threshold: float = EV_THRESHOLD):
    """
    Vectorized traversal for N games.
    Returns (2*N*10, 44) features and (2*N*10,) mean-centered EV targets.
    """
    N    = len(hand5_As)
    PDIM = PAIR_DIM
    FDIM = FEAT_DIM

    h5A = np.ascontiguousarray(hand5_As, dtype=np.int32)   # [N,5]
    h5B = np.ascontiguousarray(hand5_Bs, dtype=np.int32)   # [N,5]
    b3  = np.ascontiguousarray(boards5[:, :3], dtype=np.int32)  # [N,3]

    # ── Step 1: EV matrices (C++ batch) ───────────────────────────────────────
    ev_flat = np.zeros(N * 100, dtype=np.float32)
    _c_lib.c_compute_discard_ev_matrix_batch(
        ctypes.c_int(N),
        h5A.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        h5B.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        b3.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ev_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(n_mc),
        ctypes.c_uint(int(np.random.randint(0, 2**31))),
    )
    ev_mats = ev_flat.reshape(N, 10, 10)   # ev_mats[n, ka, kb] = P(A wins)

    # ── Step 2: Pair features A, B (C++ batch) ────────────────────────────────
    pair_A = np.zeros((N * 10, PDIM), dtype=np.float32)
    pair_B = np.zeros((N * 10, PDIM), dtype=np.float32)
    _c_lib.c_build_discard_pair_features_batch(
        ctypes.c_int(N),
        h5A.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        h5B.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        b3.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        pair_A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        pair_B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    brs    = np.where(b3 >= 0, b3 % 9 / 8., 0.).astype(np.float32)   # [N,3]
    brs_10 = np.repeat(brs, 10, axis=0)   # [N*10, 3]

    # ── Step 3: feats_A (uniform opp_cats) + net forward A ───────────────────
    feats_A = np.empty((N * 10, FDIM), dtype=np.float32)
    # opp_cats_A: category distribution of B's possible hands given A's dead cards
    # Dead = A's 5 cards + board 3. Weighted by range probability — NOT flat 1/17.
    oc_A     = opp_cats_uniform_batch(h5A, b3)           # [N, 17]
    oc_A_10  = np.repeat(oc_A, 10, axis=0)              # [N*10, 17]

    feats_A[:, :PDIM]           = pair_A
    feats_A[:, PDIM:PDIM+3]    = brs_10
    feats_A[:, PDIM+3:PDIM+20] = oc_A_10
    feats_A[:, PDIM+20]         = 0.         # is_bb=False for SB

    _device = next(net.parameters()).device
    with torch.no_grad():
        ev_out_A = net(torch.from_numpy(feats_A).to(_device)).cpu().numpy().reshape(N, 10)

    strat_A = _softmax_candidates(ev_out_A, threshold, SOFTMAX_T)  # [N,10]

    # ── Step 4: A's discard cards for all N×10 ────────────────────────────────
    a_disc = h5A[:, _DISC_IDX].reshape(N * 10, 3)   # [N*10, 3]

    # ── Step 5: opp_cats_B narrowed by A's discard (C++ batch) ───────────────
    opp_cats_B = np.empty((N * 10, 17), dtype=np.float32)
    h5B_rep    = np.repeat(h5B, 10, axis=0)
    b3_rep     = np.repeat(b3,  10, axis=0)
    _c_lib.c_opp_cats_narrowed_batch(
        ctypes.c_int(N * 10),
        np.ascontiguousarray(h5B_rep).ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        np.ascontiguousarray(b3_rep).ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        np.ascontiguousarray(a_disc).ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        opp_cats_B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )

    # ── Step 6: feats_B for N×100 + net forward B ────────────────────────────
    pair_B_100 = np.repeat(pair_B, 10, axis=0)      # [N*100, PDIM]
    brs_100    = np.repeat(brs_10, 10, axis=0)       # [N*100, 3]
    oc_B_100   = np.repeat(opp_cats_B, 10, axis=0)  # [N*100, 17]

    feats_B_all = np.empty((N * 100, FDIM), dtype=np.float32)
    feats_B_all[:, :PDIM]           = pair_B_100
    feats_B_all[:, PDIM:PDIM+3]    = brs_100
    feats_B_all[:, PDIM+3:PDIM+20] = oc_B_100
    feats_B_all[:, PDIM+20]         = 1.   # is_bb=True for BB

    with torch.no_grad():
        ev_out_B = net(torch.from_numpy(feats_B_all).to(_device)).cpu().numpy()

    ev_out_B_r = ev_out_B.reshape(N * 10, 10)
    strat_B    = _softmax_candidates(ev_out_B_r, threshold, SOFTMAX_T)  # [N*10,10]
    strat_B_ka = strat_B.reshape(N, 10, 10)   # [N, ka, kb]

    # ── Step 7: ground-truth ev_A, target_A ──────────────────────────────────
    ev_A    = np.einsum('ijk,ijk->ij', strat_B_ka, ev_mats)    # [N, 10]
    tgt_A   = (ev_A - ev_A.mean(axis=1, keepdims=True)).astype(np.float32)   # [N,10]

    # ── Step 8: sample ka_actual, compute target_B ───────────────────────────
    rng       = np.random.default_rng()
    cumA      = np.cumsum(strat_A.astype(np.float64), axis=1)
    ka_actual = (rng.random((N,))[:, None] > cumA).sum(axis=1).clip(0, 9)

    idx       = np.arange(N)
    strat_B_a = strat_B_ka[idx, ka_actual]       # [N, 10]
    ev_B      = 1.0 - ev_mats[idx, ka_actual]    # [N, 10]
    tgt_B     = (ev_B - ev_B.mean(axis=1, keepdims=True)).astype(np.float32)

    # ── Step 9: select feats_B for ka_actual ─────────────────────────────────
    row_starts     = (idx * 100 + ka_actual * 10)[:, None] + np.arange(10)[None, :]
    feats_B_actual = feats_B_all[row_starts.flatten()]   # [N*10, 44]

    all_feats  = np.vstack([feats_A,          feats_B_actual])
    all_targets = np.concatenate([tgt_A.flatten(), tgt_B.flatten()])
    return all_feats, all_targets

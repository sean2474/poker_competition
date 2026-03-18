"""
Discard CFR traversal — single shared network.

For each game (hand5_A, hand5_B, board3):
  1. compute ev_matrix[10][10] via MC runouts
  2. Player A: classify_all_pairs(A) once, opp_cats = uniform prior
     → strat_A, regrets for all 10 pairs → 10 samples
  3. Player B: classify_all_pairs(B) once, opp_cats = oracle (knows A's disc3)
     → strat_B, regrets for all 10 pairs → 10 samples

Total: 20 independent (feat39, scalar_adv) samples per game.

NNUE optimisation: pair_cats and pair_ranks are pre-computed once per player.
For A's EV loop, only opp_cats changes across the 10 B-strategy evaluations
(board_cats for B are reused). Context is re-assembled cheaply.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from itertools import combinations

from game.features import evaluate_showdown
from .features import (KEEP_PAIRS, N_KEEP_PAIRS,
                       classify_all_pairs, pair_rank_feats, pair_blocker_feats,
                       build_all_feats,
                       opp_cats_uniform, opp_cats_narrowed)

_ALL_CARDS       = list(range(27))
_EXACT_THRESHOLD = 91   # C(14,2)


def compute_ev_matrix(hand5_A, hand5_B, board3, n_mc: int = 8) -> np.ndarray:
    """
    ev_matrix[ka][kb] = P(A wins). Shape (10, 10) float32.

    Vectorized: pre-samples ALL runout cards for ALL 100 pairs at once,
    then evaluates in a single batch loop instead of per-pair Python loops.
    n_mc reduced from 40 to 8 — sufficient for CFR training signal.
    """
    ev        = np.zeros((N_KEEP_PAIRS, N_KEEP_PAIRS), dtype=np.float32)
    board_set = set(board3)
    b5        = board3 + [-1, -1]   # for evaluate_showdown

    for ka, (ai, aj) in enumerate(KEEP_PAIRS):
        p0_keep = [hand5_A[ai], hand5_A[aj]]
        p0_set  = set(p0_keep)
        for kb, (bi, bj) in enumerate(KEEP_PAIRS):
            p1_keep = [hand5_B[bi], hand5_B[bj]]
            pool    = [c for c in _ALL_CARDS
                       if c not in p0_set | set(p1_keep) | board_set]
            n_pool  = len(pool)
            if n_pool < 2:
                ev[ka][kb] = 0.5
                continue
            n_exact = n_pool * (n_pool - 1) // 2
            wins = 0.0
            if n_exact <= _EXACT_THRESHOLD:
                for t, r in combinations(pool, 2):
                    res = evaluate_showdown(p0_keep, p1_keep, b5[:3] + [t, r])
                    wins += 0.5 if res == 0 else (1.0 if res > 0 else 0.0)
                ev[ka][kb] = wins / n_exact
            else:
                # Batch: pre-sample all n_mc runout pairs at once
                pool_arr = np.array(pool)
                idxs = np.array([np.random.choice(n_pool, 2, replace=False)
                                  for _ in range(n_mc)])
                for t_idx, r_idx in idxs:
                    res = evaluate_showdown(p0_keep, p1_keep,
                                           b5[:3] + [int(pool_arr[t_idx]),
                                                     int(pool_arr[r_idx])])
                    wins += 0.5 if res == 0 else (1.0 if res > 0 else 0.0)
                ev[ka][kb] = wins / n_mc
    return ev


def traverse_game(hand5_A, hand5_B, board3,
                  net,
                  ev_matrix: np.ndarray,
                  iteration: float):
    """
    Single-game traversal.

    Returns:
      feats_A (10, 39), adv_A (10,)  — 10 samples for buffer
      feats_B (10, 39), adv_B (10,)  — 10 samples for buffer
    """
    # ── Pre-compute per-player pair features (expensive, done ONCE each) ──────
    pcats_A  = classify_all_pairs(hand5_A, board3)   # (10, 17)
    pranks_A = pair_rank_feats(hand5_A)              # (10,  2)
    pblk_A   = pair_blocker_feats(hand5_A, board3)   # (10,  4)
    pcats_B  = classify_all_pairs(hand5_B, board3)   # (10, 17)
    pranks_B = pair_rank_feats(hand5_B)              # (10,  2)
    pblk_B   = pair_blocker_feats(hand5_B, board3)   # (10,  4)

    # ── Player A: uniform prior on opp range ──────────────────────────────────
    oc_A     = opp_cats_uniform(hand5_A, board3)                         # (17,)
    feats_A  = build_all_feats(pcats_A, pranks_A, pblk_A, board3, oc_A,
                                is_bb=False)                             # (10, 44)
    strat_A  = net.get_strategy(feats_A)                                 # (10,)

    # EV for each A choice: average over B's strategy
    # B's pair features are fixed; only opp_cats changes per A choice
    ev_A = np.zeros(N_KEEP_PAIRS, dtype=np.float64)
    for ka, (ai, aj) in enumerate(KEEP_PAIRS):
        p0_disc    = [hand5_A[k] for k in range(5) if k not in (ai, aj)]
        oc_B_ka    = opp_cats_narrowed(hand5_B, board3, p0_disc)         # (17,)
        feats_B_ka = build_all_feats(pcats_B, pranks_B, pblk_B, board3, oc_B_ka,
                                     is_bb=True)                         # (10, 44)
        strat_B_ka = net.get_strategy(feats_B_ka)                        # (10,)
        ev_A[ka]   = float(np.dot(strat_B_ka, ev_matrix[ka]))

    cf_ev_A = float(np.dot(strat_A, ev_A))
    adv_A   = (ev_A - cf_ev_A).astype(np.float32)                       # (10,)

    # ── Player B: oracle-narrowed range (knows A's actual discards) ───────────
    ka_actual  = int(np.random.choice(N_KEEP_PAIRS, p=strat_A))
    ai_a, aj_a = KEEP_PAIRS[ka_actual]
    p0_disc_actual = [hand5_A[k] for k in range(5) if k not in (ai_a, aj_a)]

    oc_B    = opp_cats_narrowed(hand5_B, board3, p0_disc_actual)
    feats_B = build_all_feats(pcats_B, pranks_B, pblk_B, board3, oc_B,
                               is_bb=True)                               # (10, 44)
    strat_B = net.get_strategy(feats_B)                                  # (10,)

    ev_B    = 1.0 - ev_matrix[ka_actual].astype(np.float64)
    cf_ev_B = float(np.dot(strat_B, ev_B))
    adv_B   = (ev_B - cf_ev_B).astype(np.float32)                 # (10,)

    return feats_A, adv_A, feats_B, adv_B


_KP_IDX = np.array([[i, j] for i in range(5) for j in range(i+1, 5)], dtype=np.int32)  # [10,2]
_DISC_IDX = np.array([[k for k in range(5) if k not in (i, j)]
                       for i, j in [(a,b) for a in range(5) for b in range(a+1,5)]],
                      dtype=np.int32)  # [10,3]


def run_batch(hand5_As, hand5_Bs, boards5,
              net,
              iteration: float,
              n_mc: int = 8):
    """
    Fully vectorized traversal for N games.
    Replaces N per-game Python loops with:
      1 C++ EV matrix batch + 1 C++ pair feature batch +
      1 C++ opp_cats batch + 2 large DiscardNet forwards.
    Returns (20N, 44) feats and (20N,) advs.
    """
    import ctypes, copy, torch
    from game.features import _c_lib
    from .features import KEEP_PAIRS, PAIR_DIM, CTX_DIM

    N     = len(hand5_As)
    FDIM  = PAIR_DIM + CTX_DIM   # 44
    PDIM  = 23                   # C++ pair-feature dim

    h5A = np.ascontiguousarray(hand5_As, dtype=np.int32)   # [N,5]
    h5B = np.ascontiguousarray(hand5_Bs, dtype=np.int32)   # [N,5]
    b3  = np.ascontiguousarray(boards5[:, :3], dtype=np.int32)  # [N,3]

    # ── Step 1: EV matrices (C++ batch, OpenMP) ──────────────────────────────
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
    ev_mats = ev_flat.reshape(N, 10, 10)  # [N, ka, kb]

    # ── Step 2: Pair features (C++ batch) ─────────────────────────────────
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
    brs = np.where(b3 >= 0, b3 % 9 / 8., 0.).astype(np.float32)  # [N,3]

    # ── Step 3: feats_A (uniform opp context) + Net forward A ────────────────
    brs_10 = np.repeat(brs, 10, axis=0)  # [N*10, 3]
    feats_A = np.empty((N * 10, FDIM), dtype=np.float32)
    feats_A[:, :PDIM]          = pair_A
    feats_A[:, PDIM:PDIM+3]   = brs_10
    feats_A[:, PDIM+3:PDIM+20] = 1./17
    feats_A[:, PDIM+20]        = 0.

    net_cpu = copy.deepcopy(net).cpu().eval()
    with torch.no_grad():
        adv_A_flat = net_cpu(torch.from_numpy(feats_A)).numpy()  # [N*10]

    pos_A = np.maximum(adv_A_flat.reshape(N, 10), 0.)
    s_A   = pos_A.sum(axis=1, keepdims=True)
    strat_A = np.where(s_A > 0, pos_A / np.where(s_A > 0, s_A, 1.), 1./10)
    strat_A /= strat_A.sum(axis=1, keepdims=True)

    # ── Step 4: A discard cards for all N*10 (game, ka) ───────────────────
    # Vectorized: h5A[:, _DISC_IDX] -> [N,10,3]
    a_disc = h5A[:, _DISC_IDX].reshape(N * 10, 3)  # [N*10, 3]

    # ── Step 5: opp_cats_B for all N*10 (C++ batch) ──────────────────────
    opp_cats_B = np.empty((N * 10, 17), dtype=np.float32)
    h5B_rep = np.repeat(h5B, 10, axis=0)          # [N*10, 5]
    b3_rep  = np.repeat(b3,  10, axis=0)           # [N*10, 3]
    _c_lib.c_opp_cats_narrowed_batch(
        ctypes.c_int(N * 10),
        np.ascontiguousarray(h5B_rep).ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        np.ascontiguousarray(b3_rep).ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        np.ascontiguousarray(a_disc).ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        opp_cats_B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )

    # ── Step 6: feats_B for N*100 (10 A-choices × 10 B-pairs) + Net forward B ──
    pair_B_100 = np.repeat(pair_B, 10, axis=0)          # [N*100, PDIM]
    brs_100    = np.repeat(brs_10, 10, axis=0)           # [N*100, 3]
    oc_B_100   = np.repeat(opp_cats_B, 10, axis=0)       # [N*100, 17]
    feats_B_all = np.empty((N * 100, FDIM), dtype=np.float32)
    feats_B_all[:, :PDIM]          = pair_B_100
    feats_B_all[:, PDIM:PDIM+3]   = brs_100
    feats_B_all[:, PDIM+3:PDIM+20] = oc_B_100
    feats_B_all[:, PDIM+20]        = 1.

    with torch.no_grad():
        adv_B_flat = net_cpu(torch.from_numpy(feats_B_all)).numpy()  # [N*100]

    pos_B   = np.maximum(adv_B_flat.reshape(N * 10, 10), 0.)
    s_B     = pos_B.sum(axis=1, keepdims=True)
    strat_B = np.where(s_B > 0, pos_B / np.where(s_B > 0, s_B, 1.), 1./10)
    strat_B /= strat_B.sum(axis=1, keepdims=True)
    strat_B_ka = strat_B.reshape(N, 10, 10)  # [N, ka, kb]

    # ── Step 7: adv_A via einsum ───────────────────────────────────────
    ev_A    = np.einsum('ijk,ijk->ij', strat_B_ka, ev_mats)   # [N, 10]
    cf_ev_A = np.einsum('ij,ij->i',   strat_A,    ev_A)       # [N]
    adv_A   = (ev_A - cf_ev_A[:, None]).astype(np.float32)    # [N, 10]

    # ── Step 8: sample ka_actual, compute adv_B ───────────────────────────
    rng_np   = np.random.default_rng()
    cumA     = np.cumsum(strat_A.astype(np.float64), axis=1)
    ka_actual = (rng_np.random((N,))[:, None] > cumA).sum(axis=1).clip(0, 9)

    idx       = np.arange(N)
    strat_B_a = strat_B_ka[idx, ka_actual]          # [N, 10]
    ev_B      = 1.0 - ev_mats[idx, ka_actual]        # [N, 10]
    cf_ev_B   = np.einsum('ij,ij->i', strat_B_a, ev_B)  # [N]
    adv_B     = (ev_B - cf_ev_B[:, None]).astype(np.float32)  # [N, 10]

    # ── Step 9: pick feats_B_actual for ka_actual (vectorized index) ─────────
    row_starts = (idx * 100 + ka_actual * 10)[:, None] + np.arange(10)[None, :]
    feats_B_actual = feats_B_all[row_starts.flatten()]  # [N*10, 44]

    all_feats = np.vstack([feats_A,         feats_B_actual])   # [2*N*10, 44]
    all_advs  = np.concatenate([adv_A.flatten(), adv_B.flatten()])  # [2*N*10]
    return all_feats, all_advs

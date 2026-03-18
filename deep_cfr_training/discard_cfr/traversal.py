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


def run_batch(hand5_As, hand5_Bs, boards5,
              net,
              iteration: float,
              n_mc: int = 8):
    """
    Traverse N games. Returns flat (20N, 39) feats and (20N,) advs.
    Each game contributes 10 A-samples + 10 B-samples = 20 total.
    """
    all_feats, all_advs = [], []

    for i in range(len(hand5_As)):
        board3 = list(boards5[i][:3])
        h5A    = list(hand5_As[i])
        h5B    = list(hand5_Bs[i])

        ev_mat = compute_ev_matrix(h5A, h5B, board3, n_mc)
        fA, aA, fB, aB = traverse_game(h5A, h5B, board3, net, ev_mat, iteration)

        all_feats.append(fA); all_advs.append(aA)
        all_feats.append(fB); all_advs.append(aB)

    return np.concatenate(all_feats), np.concatenate(all_advs)  # (20N,44), (20N,)

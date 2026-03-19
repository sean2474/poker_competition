"""
test_discard_parallel.py

두 가지 _recompute_discards_with_cfr 구현 비교:
  A) 현재: A forward → sample A → opp_cats_B narrowed → B forward (sequential, 2 GPU calls)
  B) 제안: A와 B 모두 uniform opp cats로 동시 계산 (1 GPU call, 2N*10 배치)

테스트 항목:
  1. 두 방식의 discard 전략 분포가 얼마나 다른가 (KL divergence)
  2. 속도 비교 (N=5000 기준)
  3. pair_layer 공유로 NNUE 분리 효과 확인

Usage:
    python -m tests.test_discard_parallel
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import torch

from game import batch_deal_discard
from game.features import _c_lib
from discard_cfr.features import KEEP_PAIRS as DK_PAIRS, PAIR_DIM, CTX_DIM
from discard_cfr import DiscardNet
import ctypes

FDIM = PAIR_DIM + CTX_DIM   # 44
PDIM = 23
N    = 5000

_KP_DISC = [[k for k in range(5) if k not in (ai, aj)]
             for ai, aj in [(i,j) for i in range(5) for j in range(i+1,5)]]
_rng = np.random.default_rng(42)


def _build_pair_and_board(p0h5, p1h5, comms):
    """공통: C++ pair feature 빌드 (A/B 동시)."""
    pair_A = np.zeros((N * 10, PDIM), dtype=np.float32)
    pair_B = np.zeros((N * 10, PDIM), dtype=np.float32)
    boards3 = np.ascontiguousarray(comms[:, :3], dtype=np.int32)
    _c_lib.c_build_discard_pair_features_batch(
        ctypes.c_int(N),
        np.ascontiguousarray(p0h5, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        np.ascontiguousarray(p1h5, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        boards3.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        pair_A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        pair_B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    brs     = np.where(boards3 >= 0, boards3 % 9 / 8., 0.).astype(np.float32)
    brs_rep = np.repeat(brs, 10, axis=0)
    return pair_A, pair_B, boards3, brs_rep


def _sample_from_adv(adv_flat, N):
    adv = adv_flat.reshape(N, 10)
    pos  = np.maximum(adv, 0.)
    sums = pos.sum(axis=1, keepdims=True)
    safe = np.where(sums > 0, sums, 1.0)
    strat = np.where(sums > 0, pos / safe, 1./10)
    strat /= strat.sum(axis=1, keepdims=True)
    r   = _rng.random((N,))
    cum = np.cumsum(strat.astype(np.float64), axis=1)
    k   = (r[:, None] > cum).sum(axis=1).clip(0, 9)
    return strat, k.astype(np.int32)


# ── 공통: opp_cats_B 계산 (A discard 필요) ──────────────────────────────────

def _compute_opp_cats_B(p0h5, p1h5, boards3, ka_arr):
    p0h = np.zeros((N, 2), dtype=np.int32)
    p0d = np.full((N, 3), -1, dtype=np.int32)
    for i in range(N):
        ai, aj = DK_PAIRS[ka_arr[i]]
        h5 = p0h5[i]
        p0h[i] = [h5[ai], h5[aj]]
        p0d[i] = [h5[k] for k in _KP_DISC[ka_arr[i]]]
    opp_cats_B = np.empty((N, 17), dtype=np.float32)
    _c_lib.c_opp_cats_narrowed_batch(
        ctypes.c_int(N),
        np.ascontiguousarray(p1h5, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        boards3.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        np.ascontiguousarray(p0d, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        opp_cats_B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    return p0h, p0d, opp_cats_B


# ── A) 원본: net(feats_A) → sample A → net(feats_B)  (2번 full forward) ─────

def recompute_original(p0h5, p1h5, comms, net, dev):
    pair_A, pair_B, boards3, brs_rep = _build_pair_and_board(p0h5, p1h5, comms)

    feats_A = np.empty((N * 10, FDIM), dtype=np.float32)
    feats_A[:, :PDIM]          = pair_A
    feats_A[:, PDIM:PDIM+3]    = brs_rep
    feats_A[:, PDIM+3:PDIM+20] = 1.0 / 17
    feats_A[:, PDIM+20]        = 0.

    net.eval()
    with torch.no_grad():
        adv_A = net(torch.from_numpy(feats_A).to(dev)).cpu().numpy()

    _, ka = _sample_from_adv(adv_A, N)
    _, p0d, opp_cats_B = _compute_opp_cats_B(p0h5, p1h5, boards3, ka)
    opp_cats_B_rep = np.repeat(opp_cats_B, 10, axis=0)

    feats_B = np.empty((N * 10, FDIM), dtype=np.float32)
    feats_B[:, :PDIM]          = pair_B
    feats_B[:, PDIM:PDIM+3]    = brs_rep
    feats_B[:, PDIM+3:PDIM+20] = opp_cats_B_rep
    feats_B[:, PDIM+20]        = 1.

    with torch.no_grad():
        adv_B = net(torch.from_numpy(feats_B).to(dev)).cpu().numpy()

    return adv_A, adv_B, ka


# ── B) NNUE 분리: pair_layer(concat A,B) 1번 → ctx/output 분리  (동일 결과) ─

def recompute_nnue(p0h5, p1h5, comms, net, dev, ka_override=None):
    """pair_layer는 2N*10 배치로 1번, ctx/output은 분리. 결과는 원본과 동일."""
    pair_A, pair_B, boards3, brs_rep = _build_pair_and_board(p0h5, p1h5, comms)

    ctx_A = np.empty((N * 10, CTX_DIM), dtype=np.float32)
    ctx_A[:, :3]   = brs_rep
    ctx_A[:, 3:20] = 1.0 / 17
    ctx_A[:, 20]   = 0.

    pair_AB = np.concatenate([pair_A, pair_B], axis=0)   # (2*N*10, PDIM)
    net.eval()
    with torch.no_grad():
        pair_AB_t   = torch.from_numpy(pair_AB).to(dev)
        pair_emb_AB = net.pair_layer(pair_AB_t)
        pair_emb_A  = pair_emb_AB[:N * 10]
        pair_emb_B  = pair_emb_AB[N * 10:]

        ctx_emb_A = net.ctx_layer(torch.from_numpy(ctx_A).to(dev))
        adv_A = net.output(pair_emb_A + ctx_emb_A).squeeze(-1).cpu().numpy()

    if ka_override is not None:
        ka = ka_override
    else:
        _, ka = _sample_from_adv(adv_A, N)

    _, p0d, opp_cats_B = _compute_opp_cats_B(p0h5, p1h5, boards3, ka)
    opp_cats_B_rep = np.repeat(opp_cats_B, 10, axis=0)

    ctx_B = np.empty((N * 10, CTX_DIM), dtype=np.float32)
    ctx_B[:, :3]   = brs_rep
    ctx_B[:, 3:20] = opp_cats_B_rep
    ctx_B[:, 20]   = 1.

    with torch.no_grad():
        ctx_emb_B = net.ctx_layer(torch.from_numpy(ctx_B).to(dev))
        adv_B = net.output(pair_emb_B + ctx_emb_B).squeeze(-1).cpu().numpy()

    return adv_A, adv_B, ka


# ── KL divergence ─────────────────────────────────────────────────────────────

def kl_div(p, q, eps=1e-9):
    p = np.clip(p, eps, None); p /= p.sum(axis=1, keepdims=True)
    q = np.clip(q, eps, None); q /= q.sum(axis=1, keepdims=True)
    return (p * np.log(p / q)).sum(axis=1).mean()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = DiscardNet(hidden_dim=128).to(dev)
    net.eval()
    print(f'Device: {dev}')

    r = batch_deal_discard(N)
    _, _, _, _, comms, p0h5, p1h5 = r

    # ── 수학적 동일성 검증 ────────────────────────────────────────────────────
    # 동일한 ka (A의 discard 선택) 사용 → opp_cats_B 동일 → adv_B 동일해야 함
    adv_A_orig, adv_B_orig, ka = recompute_original(p0h5, p1h5, comms, net, dev)
    adv_A_nnue, adv_B_nnue, _  = recompute_nnue(p0h5, p1h5, comms, net, dev,
                                                  ka_override=ka)

    diff_A = np.abs(adv_A_orig - adv_A_nnue).max()
    diff_B = np.abs(adv_B_orig - adv_B_nnue).max()
    print(f'\n[수학적 동일성 — max absolute diff (0이어야 함)]')
    print(f'  adv_A: {diff_A:.2e}  {"✓ 동일" if diff_A < 1e-4 else "✗ 다름"}')
    print(f'  adv_B: {diff_B:.2e}  {"✓ 동일" if diff_B < 1e-4 else "✗ 다름"}')
    print(f'  (float32 machine epsilon: ~1.2e-7, 오차는 floating point 누적)')

    # ── 속도 비교 ─────────────────────────────────────────────────────────────
    WARMUP = 3
    for _ in range(WARMUP):
        recompute_original(p0h5, p1h5, comms, net, dev)
        recompute_nnue(p0h5, p1h5, comms, net, dev)

    BENCH = 20
    t0 = time.perf_counter()
    for _ in range(BENCH):
        recompute_original(p0h5, p1h5, comms, net, dev)
    t_orig = (time.perf_counter() - t0) / BENCH

    t0 = time.perf_counter()
    for _ in range(BENCH):
        recompute_nnue(p0h5, p1h5, comms, net, dev)
    t_nnue = (time.perf_counter() - t0) / BENCH

    print(f'\n[속도 비교 — N={N}, {BENCH} runs]')
    print(f'  Original (2x full net()):  {t_orig*1000:.1f} ms/call')
    print(f'  NNUE    (pair_AB 1x):      {t_nnue*1000:.1f} ms/call')
    print(f'  Speedup: {t_orig/t_nnue:.2f}x')
    print(f'\n  → _recompute_discards_with_cfr 총 시간: ~{t_orig*1000:.0f}ms/iter')
    print(f'  → Phase3 90s 중 {t_orig/90*100:.1f}% — 병목인가? {"NO" if t_orig < 1.0 else "YES"}')


if __name__ == '__main__':
    main()

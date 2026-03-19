"""
strategy/discard-2.py — EV-based probabilistic discard.

DiscardNet 대신 직접 MC equity 계산:
  1. 10가지 keep pair 각각에 대해 opp range 기반 기대 equity 계산
  2. EV에 softmax 씌워서 확률 분포 생성 (mixed strategy)
  3. 분포에서 샘플링 (probabilistic)

OppRangeTracker range 활용 시 preflop action 반영된 더 정확한 range 사용.
"""

import numpy as np
from itertools import combinations

from features import classify_hand, KEEP_PAIRS, _ALL_PAIRS
from action import DISCARD

_ALL_CARDS   = list(range(27))
_SOFTMAX_T   = 10.0     # temperature: 클수록 uniform, 작을수록 argmax에 가까움
_N_MC        = 20       # MC runouts per (our_pair, opp_pair), pool 크면 샘플링
_EXACT_LIMIT = 120      # pool C(n,2) ≤ 이 값이면 exact enumeration


def _showdown(c0, c1, oc0, oc1, board5: list) -> float:
    """Return equity of (c0,c1) vs (oc0,oc1) on board5. 1=win, 0.5=tie, 0=lose."""
    my  = classify_hand(c0,  c1,  board5, 5)
    opp = classify_hand(oc0, oc1, board5, 5)
    if my < opp: return 1.0
    if my > opp: return 0.0
    # Same category: compare high rank (tiebreak approx)
    my_hi  = max(c0 % 9, c1 % 9)
    opp_hi = max(oc0 % 9, oc1 % 9)
    if my_hi > opp_hi: return 1.0
    if my_hi < opp_hi: return 0.0
    return 0.5


def _compute_evs(hand5: list, board3: list, opp_probs: np.ndarray) -> np.ndarray:
    """
    EV[10] for each keep pair via MC/exact enumeration over opp range.
    Returns float64 array in [0,1].
    """
    comm_dead = set(c for c in board3 if c >= 0)
    board_base = board3 + [-1, -1]  # 5-slot board for classify_hand
    evs = np.zeros(10, dtype=np.float64)

    for ka, (ai, aj) in enumerate(KEEP_PAIRS):
        c0, c1    = hand5[ai], hand5[aj]
        our_dead  = {c0, c1} | comm_dead
        total_w   = 0.0
        total_ev  = 0.0

        for oi, (oc0, oc1) in enumerate(_ALL_PAIRS):
            w = float(opp_probs[oi])
            if w < 1e-9:
                continue
            if oc0 in our_dead or oc1 in our_dead:
                continue

            pool   = [c for c in _ALL_CARDS if c not in our_dead and c != oc0 and c != oc1]
            n_pool = len(pool)
            n_exact = n_pool * (n_pool - 1) // 2

            if n_pool < 2:
                ev_pair = 0.5
            elif n_exact <= _EXACT_LIMIT:
                wins = 0.0; cnt = 0
                for t, r in combinations(pool, 2):
                    wins += _showdown(c0, c1, oc0, oc1, board3 + [t, r])
                    cnt  += 1
                ev_pair = wins / cnt if cnt > 0 else 0.5
            else:
                pool_arr = np.array(pool, dtype=np.int32)
                wins = 0.0
                for _ in range(_N_MC):
                    t, r = pool_arr[np.random.choice(n_pool, 2, replace=False)]
                    wins += _showdown(int(c0), int(c1), int(oc0), int(oc1),
                                      board3 + [int(t), int(r)])
                ev_pair = wins / _N_MC

            total_w  += w
            total_ev += w * ev_pair

        evs[ka] = total_ev / total_w if total_w > 1e-9 else 0.5

    return evs


def decide_discard_ev(obs: dict, opp_probs: np.ndarray = None) -> tuple:
    """
    EV-based probabilistic discard.
    opp_probs: 351-dim range distribution from player.py's get_probs_for_discard().
               If None, uses uniform over valid hands.
    Returns (DISCARD, 0, keep_idx_i, keep_idx_j).
    """
    hand5  = [c for c in obs['my_cards'] if c >= 0]
    board3 = [c for c in obs.get('community_cards', [-1]*5) if c >= 0][:3]

    assert len(hand5) == 5, f'Expected 5-card hand, got {hand5}'
    assert len(board3) == 3, f'Expected 3 board cards at discard, got {board3}'

    # ── Build opp distribution ────────────────────────────────────────────────
    if opp_probs is not None:
        opp_probs = opp_probs.astype(np.float64).copy()
    else:
        opp_probs = np.ones(len(_ALL_PAIRS), dtype=np.float64)

    # Zero out impossible hands (contain our cards or board)
    dead = set(hand5) | set(board3)
    for i, (a, b) in enumerate(_ALL_PAIRS):
        if a in dead or b in dead:
            opp_probs[i] = 0.
    s = opp_probs.sum()
    if s > 1e-9:
        opp_probs /= s

    # ── Compute EVs ───────────────────────────────────────────────────────────
    evs = _compute_evs(hand5, board3, opp_probs)

    # ── Softmax with temperature → probability distribution ───────────────────
    #   P(keep pair) ∝ exp(EV * T)  — preserves mixed strategy
    logits = (evs - evs.mean()) * _SOFTMAX_T
    logits -= logits.max()
    probs   = np.exp(logits)
    probs  /= probs.sum()

    ka     = int(np.random.choice(10, p=probs))
    ki, kj = KEEP_PAIRS[ka]
    return (DISCARD, 0, ki, kj)

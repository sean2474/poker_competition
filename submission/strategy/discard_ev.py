"""
strategy/discard_ev.py — Hybrid EV + DiscardNet probabilistic discard.

Algorithm:
  1. MC equity 계산: 10가지 keep pair 각각 EV 측정 (opp range로 가중치)
  2. Candidate set: EV >= best_EV - EV_THRESHOLD 인 pair들
     (EV_THRESHOLD = max(0.01, 1.25 × MC stderr) ≈ 0.025)
  3. Candidate가 1개면 바로 선택
     여러 개면 DiscardNet이 candidate 중 softmax로 선택
     → EV 명확한 패는 항상 선택, 비슷한 패들끼리 range balance mixing

opp_probs: preflop chart 기반 range update (OppRangeTracker.get_probs_for_discard)
"""

import numpy as np
from itertools import combinations

from features import classify_hand, KEEP_PAIRS, _ALL_PAIRS
from action import DISCARD

_PAIRS_ARR   = np.array(_ALL_PAIRS, dtype=np.int32)   # (351, 2) precomputed
EV_THRESHOLD = 0.025     # candidate set: best_EV - 0.025
SOFTMAX_T    = 8.0       # temperature within candidate set
_W_SKIP      = 1e-9      # skip only zero-weight pairs
_ALL_CARDS   = list(range(27))


def _showdown(c0: int, c1: int, oc0: int, oc1: int, board5: list) -> float:
    """P(c0,c1 wins) vs (oc0,oc1) on full 5-card board. 1=win, 0.5=tie, 0=lose."""
    my  = classify_hand(c0,  c1,  board5, 5)
    opp = classify_hand(oc0, oc1, board5, 5)
    if my < opp: return 1.0
    if my > opp: return 0.0
    return 1.0 if max(c0 % 9, c1 % 9) > max(oc0 % 9, oc1 % 9) else (
           0.0 if max(c0 % 9, c1 % 9) < max(oc0 % 9, oc1 % 9) else 0.5)


def _compute_evs(hand5: list, board3: list,
                 opp_probs: np.ndarray) -> np.ndarray:
    """
    EV[10] for each keep pair via turn+river MC runouts weighted by opp range.

    For each (our keep pair, opp kept pair):
      pool = remaining cards after dead cards
      sample _N_MC turn+river combos → evaluate showdown → win rate
      EV contribution = opp_range[opp_pair] × win_rate

    Total EV(our pair) = sum over all opp pairs: opp_range × win_rate_vs_opp
    """
    PA        = _PAIRS_ARR
    comm_dead = set(c for c in board3 if c >= 0)
    evs       = np.zeros(10, dtype=np.float64)

    for ka, (ai, aj) in enumerate(KEEP_PAIRS):
        c0, c1   = hand5[ai], hand5[aj]
        our_dead = {c0, c1} | comm_dead
        total_w  = 0.0
        total_ev = 0.0

        for oi in range(351):
            w = float(opp_probs[oi])
            if w < _W_SKIP:
                continue
            oc0, oc1 = int(PA[oi, 0]), int(PA[oi, 1])
            if oc0 in our_dead or oc1 in our_dead:
                continue

            # Pool for turn+river: exclude all held cards
            pool = [c for c in _ALL_CARDS
                    if c not in our_dead and c != oc0 and c != oc1]
            n_pool = len(pool)
            if n_pool < 2:
                ev_pair = 0.5
            else:
                wins = 0.0; cnt = 0
                for t, r in combinations(pool, 2):
                    wins += _showdown(c0, c1, oc0, oc1, board3 + [t, r])
                    cnt  += 1
                ev_pair = wins / cnt if cnt > 0 else 0.5

            total_w  += w
            total_ev += w * ev_pair

        evs[ka] = total_ev / total_w if total_w > 1e-9 else 0.5

    return evs


def decide_discard_ev(obs: dict, opp_probs: np.ndarray = None,
                      discard_net=None) -> tuple:
    """
    Hybrid EV + DiscardNet discard decision.

    opp_probs:   351-dim range from OppRangeTracker.get_probs_for_discard()
    discard_net: DiscardNet for tie-breaking within candidates (optional)
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

    dead = set(hand5) | set(board3)
    for i, (a, b) in enumerate(_ALL_PAIRS):
        if a in dead or b in dead:
            opp_probs[i] = 0.
    s = opp_probs.sum()
    opp_probs = opp_probs / s if s > 1e-9 else opp_probs

    # ── Compute EVs (weighted by opp range) ───────────────────────────────────
    evs     = _compute_evs(hand5, board3, opp_probs)
    best_ev = evs.max()
    mask    = evs >= best_ev - EV_THRESHOLD   # candidate set

    n_cands = mask.sum()

    # ── Single clear winner ───────────────────────────────────────────────────
    if n_cands == 1:
        ka = int(np.argmax(evs))
        ki, kj = KEEP_PAIRS[ka]
        return (DISCARD, 0, ki, kj)

    # ── Multiple candidates: DiscardNet tie-breaks ────────────────────────────
    if discard_net is not None:
        try:
            from strategy.discard import build_discard_feats, _opp_cats_from_obs
            opp_cats = _opp_cats_from_obs(hand5, board3, obs)
            is_bb    = obs.get('acting_agent', 0) == 1
            feats    = build_discard_feats(hand5, board3, opp_cats, is_bb)
            net_strat = discard_net.get_strategy(feats.astype(np.float32))
            # Mask to EV candidates only
            net_strat[~mask] = 0.
            s2 = net_strat.sum()
            if s2 > 1e-9:
                net_strat /= s2
                ka = int(np.random.choice(10, p=net_strat.astype(np.float64)))
                ki, kj = KEEP_PAIRS[ka]
                return (DISCARD, 0, ki, kj)
        except Exception:
            pass

    # ── Fallback: softmax over candidate EVs ─────────────────────────────────
    ev_c    = evs - best_ev
    exp_v   = np.exp(np.clip(ev_c * SOFTMAX_T, -20., 0.))
    exp_v[~mask] = 0.
    probs   = exp_v / exp_v.sum()
    ka      = int(np.random.choice(10, p=probs))
    ki, kj  = KEEP_PAIRS[ka]
    return (DISCARD, 0, ki, kj)

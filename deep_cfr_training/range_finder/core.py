"""
range_finder/core.py — Bayesian range updating utilities.

calculate_range() takes prior ranges + observed information (board, discards)
and returns updated (hero_range, opp_range) distributions.

Each range is a np.ndarray of length 351 = C(27,2),
indexed by _ALL_PAIR from discard_cfr/utils.py.

Phase logic:
  'preflop': dead-card filter + weight by P(call | hand) from Preflop chart
  'discard': dead-card filter + soft-weight opp_range by discard model EV
"""

import numpy as np
from stretegy.discard_cfr.utils import _ALL_PAIR, calculate_ev
from stretegy.preflop_cfr.core import Preflop
from stretegy.discard_cfr.core import Discard
from stretegy.postflop_cfr.core import Postflop

_N      = len(_ALL_PAIR)           # 351
_UNIFORM = np.full(_N, 1. / _N, dtype=np.float32)


# ── Module-level helpers ──────────────────────────────────────────────────────

def _dead_filter(w: np.ndarray, dead: set) -> np.ndarray:
    w = w.copy()
    for idx, (c0, c1) in enumerate(_ALL_PAIR):
        if c0 in dead or c1 in dead:
            w[idx] = 0.
    return w


def _normalize(w: np.ndarray) -> np.ndarray:
    s = float(w.sum())
    return (w / s).astype(np.float32) if s > 1e-9 else _UNIFORM.copy()


# ── Range class ───────────────────────────────────────────────────────────────

class Range:
    def __init__(self):
        self.hero_range: np.ndarray = _UNIFORM.copy()
        self.opp_range:  np.ndarray = _UNIFORM.copy()

    def reset(self):
        self.hero_range = _UNIFORM.copy()
        self.opp_range  = _UNIFORM.copy()

    def update(self, board, hero_discard, opp_discard,
               phase: str, model: Preflop | Discard | Postflop,
               action: str = None, hero_action: str = None,
               history: str = ''):
        """
        Bayesian in-place update of hero_range and opp_range.

        Args:
            board        : list[int]  — community cards (-1 = unknown)
            hero_discard : list[int]  — 3 cards hero discarded (or [])
            opp_discard  : list[int]  — 3 cards opponent discarded (or [])
            phase        : 'preflop' | 'discard'
            model        : Preflop | Discard
            action       : observed opponent action ('f','k','c','r') or None
            hero_action  : hero's own action taken (updates hero_range) or None
            history      : betting history string up to this action
        """
        hero_w = self.hero_range.copy()
        opp_w  = self.opp_range.copy()

        # Dead-card filter
        known = (
            [c for c in (board        or []) if c >= 0] +
            [c for c in (hero_discard or []) if c >= 0] +
            [c for c in (opp_discard  or []) if c >= 0]
        )
        dead_all = set(known)
        hero_w = _dead_filter(hero_w, dead_all)
        opp_w  = _dead_filter(opp_w,  dead_all)

        if phase == 'preflop' and isinstance(model, Preflop):
            def _weight(w, obs):
                if obs is None:
                    return w
                _FILLERS = [0, 9, 18]
                for idx, (c0, c1) in enumerate(_ALL_PAIR):
                    if w[idx] < 1e-9:
                        continue
                    fillers = [f for f in _FILLERS if f != c0 and f != c1][:3]
                    if len(fillers) < 3:
                        fillers = [f for f in range(27) if f != c0 and f != c1][:3]
                    _, probs = model.action([c0, c1] + fillers, history)
                    w[idx] *= probs.get(obs, 0.)
                return w

            opp_w  = _weight(opp_w,  action)
            hero_w = _weight(hero_w, hero_action)

        elif phase == 'discard' and isinstance(model, Discard):
            hero_norm = _normalize(hero_w)
            dead_base = set(c for c in (list(board or []) + list(hero_discard or [])) if c >= 0)
            evs = np.zeros(_N, dtype=np.float32)
            for idx, (c0, c1) in enumerate(_ALL_PAIR):
                if opp_w[idx] < 1e-9:
                    continue
                evs[idx] = calculate_ev(
                    list(board or []), (c0, c1), hero_norm,
                    dead=dead_base | {c0, c1},
                )
            nonzero = opp_w > 1e-9
            if nonzero.any():
                ev_sub = evs[nonzero] - evs[nonzero].max()
                opp_w[nonzero] *= np.exp(ev_sub / 0.1)
                
        elif phase == 'postflop' and isinstance(model, Postflop):
            # TODO: Implement postflop range update
            pass

        self.hero_range = _normalize(hero_w)
        self.opp_range  = _normalize(opp_w)
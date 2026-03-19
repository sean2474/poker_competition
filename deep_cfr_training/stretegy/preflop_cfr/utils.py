from typing import List
import numpy as np
from game.game import canonicalize
from .state import _State

_SLOT = {'f': 0, 'k': 1, 'c': 1, 'r': 2}


def _payoff(state: _State, traverser: int) -> float:
    if state.folded_by >= 0:
        winner = 1 - state.folded_by
        gain = float(min(state.bets))
        return gain if traverser == winner else -gain
    return 0.


def _match(regrets: np.ndarray, valid: List[str]) -> dict:
    n = len(valid)
    if n == 3:
        r0 = regrets[0]; r0 = r0 if r0 > 0. else 0.
        r1 = regrets[1]; r1 = r1 if r1 > 0. else 0.
        r2 = regrets[2]; r2 = r2 if r2 > 0. else 0.
        tot = r0 + r1 + r2
        if tot > 0.:
            return {'f': r0/tot, 'c': r1/tot, 'r': r2/tot}
        return {'f': 1./3., 'c': 1./3., 'r': 1./3.}
    a0 = valid[0]
    if n == 1:
        return {a0: 1.0}
    a1 = valid[1]
    s0, s1 = _SLOT[a0], _SLOT[a1]
    r0 = regrets[s0]; r0 = r0 if r0 > 0. else 0.
    r1 = regrets[s1]; r1 = r1 if r1 > 0. else 0.
    tot = r0 + r1
    if tot > 0.:
        return {a0: r0/tot, a1: r1/tot}
    return {a0: 0.5, a1: 0.5}


def _cfr(h0, h1, state: _State, tp: int,
         regrets: dict, strat_sum: dict, t: int, rng) -> float:
    if state.done:
        return _payoff(state, tp)
    cp    = state.acting
    hand  = h0 if cp == 0 else h1
    key   = (canonicalize(hand), state.hist)
    valid = state.valid()
    regs  = regrets.setdefault(key, np.zeros(3))
    strat = _match(regs, valid)

    ss = strat_sum.setdefault(key, np.zeros(3))
    for a in valid:
        ss[_SLOT[a]] += t * strat[a]

    if cp == tp:
        vals = {a: _cfr(h0, h1, state.apply(a), tp, regrets, strat_sum, t, rng)
                for a in valid}
        ev = sum(strat[a] * vals[a] for a in valid)
        for a in valid:
            regs[_SLOT[a]] = max(0., regs[_SLOT[a]] + vals[a] - ev)
        return ev
    else:
        a_sample = rng.choices(valid, weights=[strat[a] for a in valid])[0]
        return _cfr(h0, h1, state.apply(a_sample), tp, regrets, strat_sum, t, rng)

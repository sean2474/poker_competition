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
    slot_acts: dict = {}
    for a in valid:
        slot_acts.setdefault(_SLOT[a], []).append(a)
    pos   = [max(float(regrets[s]), 0.) for s in range(3)]
    total = sum(pos[s] for s in slot_acts)
    p     = ({s: pos[s] / total for s in slot_acts} if total > 0
             else {s: 1. / len(slot_acts) for s in slot_acts})
    return {a: p[_SLOT[a]] / len(slot_acts[_SLOT[a]]) for a in valid}


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

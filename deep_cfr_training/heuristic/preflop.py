"""
heuristic/preflop.py — Equity-based heuristic preflop model.
"""

import math
import random

from interface.model import PreflopModel


_FOLD_THRESH  = 0.38
_RAISE_THRESH = 0.58


class HeuristicPreflop(PreflopModel):
    def __init__(self, num_sims: int = 400):
        self._num_sims = num_sims
        self._agent    = None  # lazy

    def _get_agent(self):
        if self._agent is None:
            from heuristic.prob_agent import HeuristicAgent
            self._agent = HeuristicAgent()
        return self._agent

    def action(self, hand: list, history: str) -> tuple[str, dict]:
        """
        Equity-based preflop decision.
        hand    : 5-card preflop hand
        history : betting history string
        Returns (action, probs)
        """
        from stretegy.preflop_cfr.state import _State

        state = _State()
        for ch in history:
            state = state.apply(ch)
        valid = state.valid()

        agent  = self._get_agent()
        equity = agent.compute_equity(hand[:2], [], [], self._num_sims)

        if 'r' in valid and equity >= _RAISE_THRESH:
            best = 'r'
        elif 'f' in valid and equity < _FOLD_THRESH:
            best = 'f'
        elif 'c' in valid:
            best = 'c'
        elif 'k' in valid:
            best = 'k'
        else:
            best = valid[0]

        # Soft probs: best gets 0.7, rest share 0.3
        n = len(valid)
        probs = {a: (0.7 if a == best else (0.3 / (n - 1) if n > 1 else 0.)) for a in valid}
        action = random.choices(list(probs.keys()), weights=list(probs.values()))[0]
        return action, probs

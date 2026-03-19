"""
heuristic/postflop.py — Equity-based heuristic postflop model.
"""

import numpy as np

from interface.model import PostflopModel

_FOLD_THRESH  = 0.38
_RAISE_THRESH = 0.60


class HeuristicPostflop(PostflopModel):
    def __init__(self, num_sims: int = 400):
        self._num_sims = num_sims
        self._agent    = None  # lazy

    def _get_agent(self):
        if self._agent is None:
            from heuristic.prob_agent import HeuristicAgent
            self._agent = HeuristicAgent()
        return self._agent

    def action(self, hand: list, board: list, history: str,
               hero_range: np.ndarray, opp_range: np.ndarray) -> tuple[str, dict]:
        """
        Returns (action, probs).
        hand : 2-card kept hand (post-discard)
        """
        agent  = self._get_agent()
        equity = agent.compute_equity(hand, board, [], self._num_sims)

        if equity >= _RAISE_THRESH:
            action = 'r'
            probs  = {'r': 0.7, 'c': 0.25, 'f': 0.05}
        elif equity < _FOLD_THRESH:
            action = 'f'
            probs  = {'f': 0.7, 'c': 0.25, 'r': 0.05}
        else:
            action = 'c'
            probs  = {'c': 0.6, 'r': 0.3, 'f': 0.1}

        return action, probs

"""
heuristic/discard.py — Equity-based heuristic discard model.
"""

import itertools
import math
import random
from typing import Optional

import numpy as np

from interface.model import DiscardModel

_KEEP_COMBOS = list(itertools.combinations(range(5), 2))


class HeuristicDiscard(DiscardModel):
    def __init__(self, num_sims: int = 200, temperature: float = 0.05):
        self._num_sims    = num_sims
        self._temperature = temperature
        self._agent       = None  # lazy

    def _get_agent(self):
        if self._agent is None:
            from heuristic.prob_agent import HeuristicAgent
            self._agent = HeuristicAgent()
        return self._agent

    def action(self, board: list, hand: list, history: str,
               hero_range: np.ndarray, opp_range: np.ndarray,
               opp_discard_card: Optional[list] = None,
               temperature: float = None) -> tuple[tuple, np.ndarray]:
        """
        Returns (keep_idx, probs).
          keep_idx : tuple[int, int] — indices into hand of 2 kept cards
          probs    : np.ndarray(10)  — softmax probability over 10 keep combos
        """
        temp  = temperature if temperature is not None else self._temperature
        agent = self._get_agent()
        opp_discards = opp_discard_card or []

        equities = [
            agent.compute_equity([hand[i], hand[j]], board, opp_discards, self._num_sims)
            for i, j in _KEEP_COMBOS
        ]

        eq_arr = np.array(equities, dtype=np.float32)

        if temp <= 0:
            best = int(np.argmax(eq_arr))
            probs = np.zeros(len(_KEEP_COMBOS), dtype=np.float32)
            probs[best] = 1.
            return _KEEP_COMBOS[best], probs

        shifted = (eq_arr - eq_arr.max()) / temp
        probs   = np.exp(shifted).astype(np.float32)
        probs  /= probs.sum()

        combo_idx = random.choices(range(len(_KEEP_COMBOS)), weights=probs)[0]
        return _KEEP_COMBOS[combo_idx], probs

"""
interface/model.py — Abstract base classes for phase models.

Each phase model must implement action() and optionally train()/save()/load().
Heuristic models inherit these and leave train() as a no-op.
CFR/NN strategy models override train().
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class PreflopModel(ABC):
    @abstractmethod
    def action(self, hand: list, history: str) -> tuple[str, dict]:
        """
        Returns (action, probs).
          action : 'f' | 'c' | 'r'
          probs  : dict[str, float] — probability for each valid action
        """
        ...

    def train(self, **kwargs):
        """Train this model. No-op for heuristic models."""
        pass

    def save(self, path: str):
        pass

    def load(self, path: str):
        pass


class DiscardModel(ABC):
    @abstractmethod
    def action(self, board: list, hand: list, history: str,
               hero_range: np.ndarray, opp_range: np.ndarray,
               opp_discard_card: Optional[list] = None,
               temperature: float = 1.0) -> tuple[tuple, np.ndarray]:
        """
        Returns (keep_idx, probs).
          keep_idx : tuple[int, int] — indices into hand of 2 kept cards
          probs    : np.ndarray(10)  — probability for each of 10 keep combos
        """
        ...

    def train(self, **kwargs):
        pass

    def save(self, path: str):
        pass

    def load(self, path: str):
        pass


class PostflopModel(ABC):
    @abstractmethod
    def action(self, hand: list, board: list, history: str,
               hero_range: np.ndarray, opp_range: np.ndarray) -> tuple[str, dict]:
        """
        Returns (action, probs).
          action : 'f' | 'c' | 'r'
          probs  : dict[str, float]
        """
        ...

    def train(self, **kwargs):
        pass

    def save(self, path: str):
        pass

    def load(self, path: str):
        pass

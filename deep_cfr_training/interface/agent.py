"""
interface/agent.py — Abstract base class for all poker agents.

Concrete agents (heuristic, strategy) inherit from AgentBase.
train() delegates to each model's train() — heuristic models skip silently.
"""

from abc import ABC, abstractmethod
import numpy as np

from interface.model import PreflopModel, DiscardModel, PostflopModel


class AgentBase(ABC):
    def __init__(self, preflop: PreflopModel, discard: DiscardModel,
                 postflop: PostflopModel = None):
        self.preflop  = preflop
        self.discard  = discard
        self.postflop = postflop

    @abstractmethod
    def reset(self):
        """Call at the start of each new hand."""
        ...

    @abstractmethod
    def act_preflop(self, hand: list, history: str) -> tuple[str, dict]:
        """Returns (action, probs)."""
        ...

    @abstractmethod
    def act_discard(self, hand: list, board: list,
                    opp_discard: list = None) -> tuple[tuple, np.ndarray]:
        """Returns (keep_idx, probs)."""
        ...

    @abstractmethod
    def act_postflop(self, hand: list, board: list,
                     history: str) -> tuple[str, dict]:
        """Returns (action, probs)."""
        ...

    def observe_opp_preflop(self, opp_action: str, history: str):
        pass

    def observe_opp_discard(self, opp_discard: list, board: list):
        pass

    def observe_opp_postflop(self, opp_action: str, history: str, board: list):
        pass

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, phase: str = 'all',
              preflop_kwargs: dict = None,
              discard_kwargs: dict = None,
              postflop_kwargs: dict = None):
        """
        Train all (or specified) models.
        Heuristic models implement train() as no-op so they're silently skipped.
        """
        if phase in ('preflop', 'all'):
            self.preflop.train(**(preflop_kwargs or {}))
        if phase in ('discard', 'all'):
            self.discard.train(**(discard_kwargs or {}))
        if phase in ('postflop', 'all') and self.postflop is not None:
            self.postflop.train(**(postflop_kwargs or {}))

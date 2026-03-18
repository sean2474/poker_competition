"""
Abstract interfaces for all decision models and trainers.

All three CFR components (preflop, discard, postflop) expose the same
interface as their heuristic counterpart, allowing the PhaseRunner to
swap them in/out without knowing implementation details.
"""

from abc import ABC, abstractmethod
import numpy as np


# ── Discard ───────────────────────────────────────────────────────────────────

class IDiscardModel(ABC):
    """
    Decides which 2 cards to keep from a 5-card hand.
    Input:  hand5, board3, opp_cats (17-dim range probs), is_bb (bool)
    Output: float32[10] probability over KEEP_PAIRS (index matches KEEP_PAIRS list)
    """

    @abstractmethod
    def get_strategy(self,
                     hand5,
                     board3,
                     opp_cats: np.ndarray = None,
                     is_bb: bool = False) -> np.ndarray:
        ...

    def sample_keep(self, hand5, board3, opp_cats=None, is_bb=False):
        """Sample a single (ki, kj) keep pair index using get_strategy."""
        from discard_cfr.features import KEEP_PAIRS
        strat = self.get_strategy(hand5, board3, opp_cats, is_bb)
        ka    = int(np.random.choice(len(KEEP_PAIRS), p=strat))
        return KEEP_PAIRS[ka]


class IDiscardTrainer(IDiscardModel):
    """IDiscardModel that also supports CFR training."""

    @abstractmethod
    def run_iter(self, hand5_As, hand5_Bs, boards5) -> None:
        """Accumulate regret samples from a batch of games."""
        ...

    @abstractmethod
    def train(self) -> float:
        """Train network on accumulated buffer. Returns mean loss."""
        ...

    @abstractmethod
    def is_converged(self) -> bool:
        """True when loss has plateaued (Phase 2 → 3 transition signal)."""
        ...


# ── Postflop ──────────────────────────────────────────────────────────────────

class IPostflopModel(ABC):
    """
    Evaluates / plays postflop nodes.
    Used by the traversal coroutine: either returns a heuristic EV (C++ equity)
    or performs neural net batch inference for CFR regret collection.
    """

    @abstractmethod
    def is_ready(self) -> bool:
        """True once the model has enough data to replace the heuristic."""
        ...


class IPostflopTrainer(IPostflopModel):
    """IPostflopModel that also supports CFR training."""

    @abstractmethod
    def train(self, trainer_state) -> list:
        """Train advantage nets. Returns [loss_p0, loss_p1]."""
        ...


# ── Preflop ───────────────────────────────────────────────────────────────────

class IPreflopModel(ABC):
    """
    Handles preflop action decisions via tabular CFR or heuristic.
    The traversal accesses regret tables directly, so this interface
    provides a thin wrapper for phase-tracking purposes.
    """

    @abstractmethod
    def is_ready(self) -> bool:
        """True when preflop CFR has visited enough infosets (Phase 1 → 2)."""
        ...


# ── Generic Trainable ─────────────────────────────────────────────────────────

class ITrainable(ABC):
    """Generic mixin for any component that supports iteration + training."""

    @abstractmethod
    def run_iter(self, *args, **kwargs) -> None: ...

    @abstractmethod
    def train(self) -> float: ...

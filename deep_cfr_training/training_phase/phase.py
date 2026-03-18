"""
Phase ABC — base class for all training phases.

Each phase only answers ONE question: is_complete(stats) → bool
All training logic lives in the CFR modules and heuristics.
The PhaseRunner reads this signal and transitions accordingly.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PhaseStats:
    """Stats snapshot passed to is_complete() each iteration."""
    iteration:            int   = 0
    pf_infosets:          int   = 0      # preflop infosets visited
    pf_visits:            float = 0.0    # total strategy-sum weight
    discard_loss:         float = 0.0    # last discard training loss
    discard_loss_history: list  = field(default_factory=list)
    postflop_buf_min:     int   = 0      # min samples across (player, street)


class Phase(ABC):
    name: str = 'phase'

    @abstractmethod
    def is_complete(self, stats: PhaseStats) -> bool:
        """Return True when this phase should hand over to the next."""
        ...

    def summary(self, stats: PhaseStats) -> str:
        return (f'[{self.name}] iter={stats.iteration}'
                f' pf_infosets={stats.pf_infosets}'
                f' discard_loss={stats.discard_loss:.4f}'
                f' pf_buf_min={stats.postflop_buf_min}')

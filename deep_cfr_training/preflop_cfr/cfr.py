"""
PreflopCFR — tabular CFR+ for preflop streets.

Training happens inline during traversal (no separate train() step).
State is stored on the trainer_state object as:
  trainer_state.preflop_regrets       : dict[key → np.ndarray(3)]
  trainer_state.preflop_strategy_sum  : dict[key → np.ndarray(3)]

is_ready() delegates to PreflopCFRGate.
"""

from interfaces import IPreflopModel
from heuristic.preflop import PreflopCFRGate

_gate = PreflopCFRGate()


class PreflopCFR(IPreflopModel):
    """
    Tabular CFR+ wrapper implementing IPreflopModel.

    There is no explicit train() — regrets are updated inline during
    traversal in postflop_cfr/traversal.py::traverse_coro().
    is_ready() checks visit accumulation via PreflopCFRGate.
    """

    def __init__(self, trainer_state=None):
        self._state = trainer_state

    def bind(self, trainer_state):
        """Bind to a trainer state object (call once before training starts)."""
        self._state = trainer_state

    def is_ready(self) -> bool:
        return _gate.is_ready(self._state)

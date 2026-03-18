"""
PreflopHeuristic — phase-transition bookkeeping for preflop.

The traversal always uses tabular CFR from iter 0.
PreflopCFRGate signals when Phase 1 → 2 transition can happen.
"""

from interfaces import IPreflopModel

_MIN_INFOSETS = 20
_MIN_VISITS   = 5_000


class PreflopHeuristic(IPreflopModel):
    """Always returns is_ready=False. Keeps Phase 1 running."""

    def is_ready(self) -> bool:
        return False


class PreflopCFRGate(IPreflopModel):
    """Returns True once preflop CFR has accumulated enough infoset visits."""

    def is_ready(self, trainer_state=None) -> bool:
        if trainer_state is None:
            return False
        ss = trainer_state.preflop_strategy_sum
        if len(ss) < _MIN_INFOSETS:
            return False
        visits = sum(float(s.sum()) for s in ss.values())
        return visits >= _MIN_VISITS

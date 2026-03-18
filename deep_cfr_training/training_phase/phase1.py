"""
Phase 1 — preflop warmup.

  preflop : tabular CFR warms up
  discard : DiscardHeuristic (fast_discard, no training)
  postflop: PostflopHeuristic (C++ equity, no training)

Transition to Phase 2 when preflop CFR has visited enough infosets.
"""

from .phase import Phase, PhaseStats

MIN_PF_INFOSETS = 20
MIN_PF_VISITS   = 5_000


class Phase1(Phase):
    name = 'phase1_preflop_warmup'

    def is_complete(self, stats: PhaseStats) -> bool:
        return (stats.pf_infosets >= MIN_PF_INFOSETS and
                stats.pf_visits  >= MIN_PF_VISITS)

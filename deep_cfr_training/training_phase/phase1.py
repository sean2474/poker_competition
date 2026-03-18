"""
Phase 1 — preflop warmup.

  preflop : tabular CFR warms up
  discard : DiscardHeuristic (fast_discard, no training)
  postflop: PostflopHeuristic (C++ equity, no training)

Transition to Phase 2 when preflop CFR has visited enough infosets.
"""

from .phase import Phase, PhaseStats

MIN_PF_ITERS    = 50        # minimum iterations before even checking
MIN_PF_INFOSETS = 10_000    # unique preflop infosets visited
MIN_PF_VISITS   = 200_000   # total strategy-sum weight accumulated


class Phase1(Phase):
    name = 'phase1_preflop_warmup'

    def is_complete(self, stats: PhaseStats) -> bool:
        if stats.iteration < MIN_PF_ITERS:
            return False
        return (stats.pf_infosets >= MIN_PF_INFOSETS and
                stats.pf_visits  >= MIN_PF_VISITS)

"""
Phase 3 — joint training.

  preflop : tabular CFR continues
  discard : DiscardCFR trains jointly each iteration
  postflop: PostflopCFR trains (switches from C++ equity once buffer fills)

Transition: Phase 3 is the final phase — no further transitions.
Training logic lives in postflop_cfr/cfr.py and discard_cfr/cfr.py.
"""

from .phase import Phase, PhaseStats


class Phase3(Phase):
    name = 'phase3_joint_training'

    def is_complete(self, stats: PhaseStats) -> bool:
        return False   # Final phase — runs until num_iterations reached

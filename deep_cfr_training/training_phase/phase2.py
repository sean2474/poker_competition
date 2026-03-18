"""
Phase 2 — discard CFR pre-training.

  preflop : tabular CFR continues
  discard : DiscardCFR trains standalone (run_iter + train each iteration)
  postflop: PostflopHeuristic (C++ equity, no training)

Transition to Phase 3 when discard loss has plateaued.
Training logic lives in discard_cfr/cfr.py::DiscardCFR.
"""

from .phase import Phase, PhaseStats

MIN_ITERS = 30    # minimum iterations before checking convergence
MAX_ITERS = 100   # force transition after this many iterations regardless
PATIENCE  = 10    # rolling window for plateau detection
DELTA     = 0.02  # 2% relative range = plateau


class Phase2(Phase):
    name = 'phase2_discard_pretrain'

    def is_complete(self, stats: PhaseStats) -> bool:
        if stats.iteration < MIN_ITERS:
            return False
        if stats.iteration >= MAX_ITERS:
            return True   # force exit even without plateau
        hist = stats.discard_loss_history
        if len(hist) < PATIENCE:
            return False
        recent = hist[-PATIENCE:]
        lo, hi = min(recent), max(recent)
        return hi > 0 and (hi - lo) / hi < DELTA

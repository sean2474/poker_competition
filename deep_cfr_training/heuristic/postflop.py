"""
PostflopHeuristic — C++ equity warmup gate as IPostflopModel.

PostflopCFRGate signals when buffer is full enough to switch to neural net.
"""

from interfaces import IPostflopModel
from postflop_cfr.traversal import _postflop_ready


class PostflopHeuristic(IPostflopModel):
    """Always returns is_ready=False. Forces C++ equity warmup."""

    def is_ready(self) -> bool:
        return False


class PostflopCFRGate(IPostflopModel):
    """Returns True once postflop adv_buffers have MIN_WARMUP_SAMPLES."""

    def is_ready(self, trainer_state=None) -> bool:
        if trainer_state is None:
            return False
        return _postflop_ready(trainer_state)

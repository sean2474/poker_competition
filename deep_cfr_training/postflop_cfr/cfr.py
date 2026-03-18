"""
PostflopCFR — Deep CFR for postflop streets (flop / turn / river).

Implements IPostflopTrainer:
  is_ready()      : True when adv_buffers have MIN_WARMUP_SAMPLES
  train()         : retrain advantage nets from scratch on buffer

The traversal logic (coroutine-based, C++ OpenMP) lives in traversal.py.
This class is the interface wrapper that the PhaseRunner interacts with.
"""

from interfaces import IPostflopTrainer
from .traversal import _postflop_ready, run_traversals_batched
from .training  import train_adv_networks, train_strategy_nets


class PostflopCFR(IPostflopTrainer):
    """
    Wraps the postflop neural net training pipeline.
    PhaseRunner calls run_traversals() and train() each iteration.
    """

    def __init__(self, trainer_state):
        self._state = trainer_state

    def is_ready(self) -> bool:
        """True when every (player, street) buffer has MIN_WARMUP_SAMPLES."""
        return _postflop_ready(self._state)

    def run_traversals(self, traversals_per_iter: int,
                       traversing_player: int,
                       discard_trainer=None,
                       discard_n_games: int = 50,
                       phase: int = 1,
                       adv_bufs=None, str_buf=None) -> None:
        """Run one round of game traversals (both warmup and neural modes)."""
        run_traversals_batched(self._state, traversals_per_iter, traversing_player,
                               discard_trainer=discard_trainer,
                               discard_n_games=discard_n_games,
                               phase=phase,
                               adv_bufs=adv_bufs, str_buf=str_buf)

    def train(self, trainer_state=None) -> list:
        """Retrain advantage nets. Returns [loss_p0, loss_p1]."""
        return train_adv_networks(self._state)

    def train_strategy(self, num_batches: int = None) -> None:
        """Train the final average strategy net (called once at end of training)."""
        train_strategy_nets(self._state, num_batches=num_batches)

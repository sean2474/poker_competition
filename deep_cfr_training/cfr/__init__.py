"""
Deep CFR trainer — composes traversal + training + runner.
"""

import torch

from models import (
    PreflopAdvantageNet,  PostflopAdvantageNet,
    PreflopStrategyNet,   PostflopStrategyNet,
    ReservoirBuffer,
)
from .runner   import run, save_checkpoint, load_checkpoint, export


# Device selection
if torch.cuda.is_available():
    _DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    _DEVICE = torch.device('mps')
else:
    _DEVICE = torch.device('cpu')

DEVICE = _DEVICE
print(f'Training device: {DEVICE}')


class DeepCFR:
    """
    Deep CFR trainer with separate preflop / postflop networks.

    Networks:
      pf_adv_nets[2]    : preflop advantage nets (street 0 samples)
      adv_nets[2]       : postflop advantage nets (streets 1-3 samples)
      pf_strategy_net   : preflop average strategy
      strategy_net      : postflop average strategy

    Buffers (stratified by street):
      adv_buffers[2]    : advantage memories per player
      strategy_buffer   : strategy memories
    """

    def __init__(self, lr: float = 1e-3, buffer_size: int = 2_000_000):
        self.lr     = lr
        self.device = DEVICE

        # Advantage nets (stay on CPU during traversal, move to GPU for training)
        self.pf_adv_nets = [PreflopAdvantageNet()  for _ in range(2)]
        self.adv_nets    = [PostflopAdvantageNet() for _ in range(2)]

        # Strategy nets
        self.pf_strategy_net = PreflopStrategyNet()
        self.strategy_net    = PostflopStrategyNet()

        # Buffers
        self.adv_buffers     = [ReservoirBuffer(buffer_size) for _ in range(2)]
        self.strategy_buffer = ReservoirBuffer(buffer_size)

        # Training state
        self.iteration        = 0
        self.total_iterations = 1
        self.batch_size       = 2048
        self.num_batches      = 100

    # ── Delegate to modules ──────────────────────────────────────────────────

    def run(self, num_iterations=500, traversals_per_iter=1000,
            train_interval=1, batch_size=2048, num_batches=100,
            checkpoint_interval=50, checkpoint_dir='model'):
        run(self, num_iterations=num_iterations,
            traversals_per_iter=traversals_per_iter,
            train_interval=train_interval,
            batch_size=batch_size, num_batches=num_batches,
            checkpoint_interval=checkpoint_interval,
            checkpoint_dir=checkpoint_dir)

    def export(self, path_prefix: str):
        export(self, path_prefix)

    def save_checkpoint(self, path: str, iteration: int):
        save_checkpoint(self, path, iteration)

    def load_checkpoint(self, path: str) -> int:
        return load_checkpoint(self, path)

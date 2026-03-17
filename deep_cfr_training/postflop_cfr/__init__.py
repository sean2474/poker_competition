"""
Deep CFR trainer — composes traversal + training + runner.
"""

import torch

from models import (
    PostflopAdvantageNet,
    PostflopStrategyNet,
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
    Hybrid CFR trainer.

    Preflop (street=0): tabular CFR — preflop_regrets / preflop_strategy_sum
      - Infoset key: (canonical_hand_tuple, history_string)
      - CFR+ regret updates, linear-weighted strategy sum
      - No neural network needed for preflop

    Postflop (streets 1-3): neural network
      - adv_nets[2]     : PostflopAdvantageNet per player
      - strategy_net    : PostflopStrategyNet (average strategy)

    Buffers (postflop only, stratified by street):
      adv_buffers[2]    : advantage memories
      strategy_buffer   : strategy memories
    """

    def __init__(self, lr: float = 3e-4, buffer_size: int = 2_000_000):
        self.lr     = lr
        self.device = DEVICE

        # Postflop advantage nets (CPU during traversal, GPU during training)
        self.adv_nets    = [PostflopAdvantageNet() for _ in range(2)]

        # Postflop average strategy net
        self.strategy_net = PostflopStrategyNet()

        # Preflop tabular CFR tables
        # key: (canonical_hand_tuple, history_str)  value: np.array(NUM_ACTIONS)
        self.preflop_regrets      = {}
        self.preflop_strategy_sum = {}

        # Postflop buffers
        self.adv_buffers     = [ReservoirBuffer(buffer_size) for _ in range(2)]
        self.strategy_buffer = ReservoirBuffer(buffer_size)

        # Training state
        self.iteration        = 0
        self.total_iterations = 1
        self.batch_size       = 2048
        self.num_batches      = 100
        self.warmup_iters     = 50   # use equity EV until postflop net converges

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

"""
DeepCFR — top-level trainer state object.

Owns all three CFR components and their associated nets/buffers.
Does NOT contain any training logic — delegates to PhaseRunner.
"""

import torch

from postflop_cfr import PostflopAdvantageNet, PostflopStrategyNet, ReservoirBuffer
from postflop_cfr.checkpoint import save_checkpoint, load_checkpoint, export
from discard_cfr.cfr import DiscardCFR
from training_phase import PhaseRunner


if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')

print(f'[deep_cfr] device: {DEVICE}')


class DeepCFR:
    """
    Trainer state: nets, buffers, and tabular tables for all 3 CFR components.

    Layout:
      preflop  : tabular CFR  (preflop_regrets, preflop_strategy_sum)
      discard  : DiscardCFR   (discard_trainer)
      postflop : neural CFR   (adv_nets, strategy_net, adv_buffers, strategy_buffer)
    """

    def __init__(self, lr: float = 3e-4, buffer_size: int = 2_000_000):
        self.lr     = lr
        self.device = DEVICE

        # ── Postflop nets + buffers ────────────────────────────────────────
        self.adv_nets     = [PostflopAdvantageNet().to(DEVICE) for _ in range(2)]
        self.strategy_net = PostflopStrategyNet().to(DEVICE)
        self.adv_buffers  = [ReservoirBuffer(buffer_size) for _ in range(2)]
        self.strategy_buffer = ReservoirBuffer(buffer_size)

        # ── Preflop tabular CFR ────────────────────────────────────────────
        self.preflop_regrets      = {}
        self.preflop_strategy_sum = {}

        # ── Discard CFR ────────────────────────────────────────────────────
        self.discard_trainer = DiscardCFR()

        # ── Training state ─────────────────────────────────────────────────
        self.iteration        = 0
        self.total_iterations = 1
        self.batch_size       = 2048
        self.num_batches      = 100

    # ── High-level API ────────────────────────────────────────────────────────

    def run(self,
            num_iterations:      int = 500,
            traversals_per_iter: int = 1000,
            train_interval:      int = 1,
            batch_size:          int = 2048,
            num_batches:         int = 100,
            checkpoint_interval: int = 50,
            checkpoint_dir:      str = 'model',
            discard_n_games:     int = 50):
        runner = PhaseRunner(
            self,
            num_iterations      = num_iterations,
            traversals_per_iter = traversals_per_iter,
            train_interval      = train_interval,
            batch_size          = batch_size,
            num_batches         = num_batches,
            checkpoint_interval = checkpoint_interval,
            checkpoint_dir      = checkpoint_dir,
            discard_n_games     = discard_n_games,
        )
        runner.run()

    def export(self, path_prefix: str):
        export(self, path_prefix)

    def save_checkpoint(self, path: str, iteration: int, save_buffers=False):
        save_checkpoint(self, path, iteration, save_buffers)

    def load_checkpoint(self, path: str) -> int:
        return load_checkpoint(self, path)

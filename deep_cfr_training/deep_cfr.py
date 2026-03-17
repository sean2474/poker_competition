"""
Deep CFR — entry point.

Package layout:
  game/    : GameState, features, constants
  models/  : AdvantageNet, StrategyNet, ReservoirBuffer
  cfr/     : DeepCFR, traversal, training, runner

Usage:
    python deep_cfr.py --iterations 500 --traversals 2000
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cfr import DeepCFR, DEVICE  # noqa: F401 (DEVICE exported for benchmark.py)


def main():
    parser = argparse.ArgumentParser(description="Deep CFR Training")
    parser.add_argument("--iterations",     type=int,   default=500)
    parser.add_argument("--traversals",     type=int,   default=1000)
    parser.add_argument("--train-interval", type=int,   default=1)
    parser.add_argument("--batch-size",     type=int,   default=65536)
    parser.add_argument("--train-batches",  type=int,   default=50)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--buffer-size",    type=int,   default=2_000_000)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    _default = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'deep_cfr')
    parser.add_argument("--output",         type=str,   default=_default)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    trainer = DeepCFR(lr=args.lr, buffer_size=args.buffer_size)
    trainer.run(
        num_iterations      = args.iterations,
        traversals_per_iter = args.traversals,
        train_interval      = args.train_interval,
        batch_size          = args.batch_size,
        num_batches         = args.train_batches,
        checkpoint_interval = args.checkpoint_every,
        checkpoint_dir      = os.path.dirname(args.output),
    )
    trainer.export(args.output)


if __name__ == "__main__":
    main()

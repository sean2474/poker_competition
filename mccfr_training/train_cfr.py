"""
Offline CFR training script.
Usage:
    python training/train_cfr.py --iterations 5000
    python training/train_cfr.py --iterations 5000 --resume
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "submission"))

from cfr_trainer import CFRTrainer

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
STRATEGY_PATH = os.path.join(DATA_DIR, "strategy.pkl")
CHECKPOINT_PATH = os.path.join(DATA_DIR, "checkpoint.pkl")


def main():
    parser = argparse.ArgumentParser(description="Train CFR strategy")
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    trainer = CFRTrainer()

    if args.resume and os.path.exists(CHECKPOINT_PATH):
        trainer.load_checkpoint(CHECKPOINT_PATH)
        print(f"Resuming from {trainer.iterations} iterations")

    trainer.train(num_iterations=args.iterations)

    # Save both: play-only strategy + full checkpoint for resuming
    trainer.save(STRATEGY_PATH)
    trainer.save_checkpoint(CHECKPOINT_PATH)
    print(f"\nDone. Total iterations: {trainer.iterations}")
    print(f"  Play strategy: {STRATEGY_PATH}")
    print(f"  Checkpoint:    {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()

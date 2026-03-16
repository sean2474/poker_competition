"""
Offline CFR training script.
Usage:
    python -m submission.train_cfr --iterations 5000
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from submission.cfr_trainer import CFRTrainer

DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "data", "strategy.pkl")


def main():
    parser = argparse.ArgumentParser(description="Train CFR strategy")
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    trainer = CFRTrainer()
    trainer.train(num_iterations=args.iterations)
    trainer.save(args.output)
    print(f"\nDone. Strategy saved to: {args.output}")


if __name__ == "__main__":
    main()

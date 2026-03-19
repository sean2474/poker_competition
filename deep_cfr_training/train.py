"""
train.py — Training entry point.

Usage:
  python train.py preflop
  python train.py discard
  python train.py postflop
  python train.py all
  python train.py preflop --iters 300000 --workers 8
  python train.py discard --rounds 5 --hands 1000 --epochs 20
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(__file__))

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

PREFLOP_PATH = os.path.join(MODELS_DIR, 'preflop.pkl')
PREFLOP_CKPT = os.path.join(MODELS_DIR, 'preflop.ckpt')
DISCARD_PATH = os.path.join(MODELS_DIR, 'discard.pt')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase', nargs='?', default='all',
                   choices=['preflop', 'discard', 'postflop', 'all'])
    p.add_argument('--iters',   type=int, default=200_000)
    p.add_argument('--rounds',  type=int, default=5)
    p.add_argument('--hands',   type=int, default=5_000)
    p.add_argument('--epochs',  type=int, default=100)
    p.add_argument('--workers', type=int, default=1)
    args = p.parse_args()

    from stretegy import Preflop, Discard
    from agent import Agent

    preflop = Preflop()
    discard = Discard()

    if os.path.exists(PREFLOP_CKPT):
        preflop.load_checkpoint(PREFLOP_CKPT)  # resume training state
    elif os.path.exists(PREFLOP_PATH):
        preflop.load(PREFLOP_PATH)             # inference-only chart
    if os.path.exists(DISCARD_PATH):
        discard.load(DISCARD_PATH)

    agent = Agent(preflop, discard)

    agent.train(
        phase=args.phase,
        preflop_kwargs={
            'n_iters':    args.iters,
            'n_workers':  args.workers,
            'save_path':  PREFLOP_PATH,
            'checkpoint_path': PREFLOP_CKPT,
        },
        discard_kwargs={
            'n_rounds':        args.rounds,
            'n_preflop_iters': args.iters,
            'n_episodes':      args.hands,
            'n_epochs':        args.epochs,
            'n_workers':       args.workers,
            'preflop_save':    PREFLOP_PATH,
            'discard_save':    DISCARD_PATH,
        },
    )
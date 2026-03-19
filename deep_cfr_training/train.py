"""
train.py — Training pipeline for all three phases.

Usage:
  python train.py preflop              # tabular MCCFR (~10 sec)
  python train.py discard              # DiscardNet via MC EV targets
  python train.py postflop             # (placeholder)
  python train.py all                  # all three in sequence
  python train.py preflop --iters 300000
"""

import os
import sys
import random

sys.path.insert(0, os.path.dirname(__file__))

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)


# ── Preflop ───────────────────────────────────────────────────────────────────

def train_preflop(n_iters: int = 200_000, save_path: str = None,
                  discard_sims: int = 20):
    from preflop_cfr.train import train as _train_preflop
    save_path = save_path or os.path.join(MODELS_DIR, 'preflop.pkl')
    return _train_preflop(n_iters=n_iters, save_path=save_path,
                          discard_sims=discard_sims)

def train_discard():
    """Placeholder — postflop Deep CFR training."""
    print('[train_postflop] not yet implemented')

# ── Postflop ──────────────────────────────────────────────────────────────────

def train_postflop():
    """Placeholder — postflop Deep CFR training."""
    print('[train_postflop] not yet implemented')


# ── Dispatcher ────────────────────────────────────────────────────────────────

def train(phase: str = 'all'):
    if phase in ('preflop', 'all'):
        train_preflop()
    if phase in ('discard', 'all'):
        train_discard()
    if phase in ('postflop', 'all'):
        train_postflop()


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Deep CFR Training')
    p.add_argument('phase', nargs='?', default='all',
                   choices=['preflop', 'discard', 'postflop', 'all'])
    p.add_argument('--iters',   type=int, default=200_000, help='preflop iterations')
    p.add_argument('--hands',   type=int, default=5_000,   help='discard training hands')
    p.add_argument('--epochs',  type=int, default=100,     help='discard training epochs')
    p.add_argument('--sims',    type=int, default=50,      help='MC sims per EV estimate')
    args = p.parse_args()

    if args.phase in ('preflop', 'all'):
        train_preflop(n_iters=args.iters)
    if args.phase in ('discard', 'all'):
        train_discard()
    if args.phase in ('postflop', 'all'):
        train_postflop()
"""
training_phase/preflop.py — Preflop-only training orchestration.
"""


def train(preflop_model, n_iters: int = 200_000,
          save_path: str = None, n_workers: int = 1,
          discard_sims: int = 20):
    """Train preflop CFR standalone."""
    preflop_model.train(
        n_iters=n_iters,
        save_path=save_path,
        n_workers=n_workers,
        discard_sims=discard_sims,
    )

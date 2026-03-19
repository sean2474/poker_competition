"""
training_phase/discard.py — Joint preflop + discard alternating training.

Each round:
  1. Train preflop CFR for n_preflop_iters
  2. Generate discard episodes using updated preflop range
  3. Train DiscardNet on generated data

Alternating rounds allow each model to improve with the other's latest weights.
"""

from tqdm import tqdm


def train(preflop_model, discard_model,
          n_rounds:         int   = 5,
          n_preflop_iters:  int   = 40_000,
          n_episodes:       int   = 1_000,
          n_epochs:         int   = 20,
          batch_size:       int   = 256,
          lr:               float = 1e-3,
          preflop_save:     str   = None,
          discard_save:     str   = None,
          n_workers:        int   = 1):
    """
    Jointly train preflop CFR and DiscardNet in alternating rounds.

    Args:
        preflop_model    : Preflop instance
        discard_model    : Discard instance
        n_rounds         : number of alternating rounds
        n_preflop_iters  : CFR iterations per round
        n_episodes       : discard training episodes per round
        n_epochs         : NN training epochs per round
        batch_size       : mini-batch size
        lr               : Adam lr for DiscardNet
        preflop_save     : path to save preflop chart after each round
        discard_save     : path to save discard weights after each round
        n_workers        : parallel workers for preflop CFR
    """
    from stretegy.discard_cfr.train import generate_episodes

    for r in tqdm(range(1, n_rounds + 1), desc='joint rounds', ncols=80):
        tqdm.write(f'\n── Round {r}/{n_rounds} ──────────────────────────')

        # ── Step 1: preflop CFR ───────────────────────────────────────────────
        tqdm.write(f'[preflop] {n_preflop_iters:,} CFR iters ...')
        preflop_model.train(
            n_iters=n_preflop_iters,
            save_path=preflop_save,
            n_workers=n_workers,
        )

        # ── Step 2: generate discard episodes with updated preflop range ──────
        tqdm.write(f'[discard] generating {n_episodes} episodes ...')
        X, Y = generate_episodes(preflop_model, n_episodes)

        # ── Step 3: train DiscardNet ──────────────────────────────────────────
        tqdm.write(f'[discard] training {n_epochs} epochs ...')
        discard_model.train(
            X, Y,
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr,
            save_path=discard_save,
        )

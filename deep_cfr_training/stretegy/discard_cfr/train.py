"""
discard_cfr/train.py — DiscardNet training utilities.

Two public functions:
  generate_episodes(preflop_model, n, rng) → (X, Y) numpy arrays
    Called by Agent.train() which owns the range propagation logic.

  train_net(net, X, Y, ...) → DiscardNet
    Called by Discard.train() with pre-computed data.
"""

import os
import random
import itertools

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .model import DiscardNet

_KEEP_COMBOS = list(itertools.combinations(range(5), 2))

# ── NN training (owned by Discard) ───────────────────────────────────────────

def train_net(net: DiscardNet,
              X: np.ndarray, Y: np.ndarray,
              n_epochs:   int   = 20,
              batch_size: int   = 256,
              lr:         float = 1e-3,
              save_path:  str   = None) -> DiscardNet:
    """
    Train DiscardNet on pre-computed (X, Y) data.

    Args:
        net        : DiscardNet to train (modified in-place)
        X          : features  (N, FEAT_DIM)
        Y          : EV targets (N,)
        n_epochs   : training epochs
        batch_size : mini-batch size
        lr         : Adam learning rate
        save_path  : optional path to save weights (.pt)
    """
    net.train()
    Xt = torch.tensor(X, dtype=torch.float32)
    Yt = torch.tensor(Y, dtype=torch.float32)

    loader  = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xt, Yt),
        batch_size=batch_size, shuffle=True)

    opt     = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in tqdm(range(n_epochs), desc='train epochs', ncols=80, leave=False):
        total = 0.
        for xb, yb in loader:
            pred  = net(xb).squeeze(-1)
            loss  = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(xb)
        if (epoch + 1) % max(1, n_epochs // 5) == 0:
            tqdm.write(f'  epoch {epoch+1}/{n_epochs}  loss={total/len(X):.5f}')

    net.eval()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        torch.save(net.state_dict(), save_path)
        print(f'[discard] saved → {save_path}')

    return net

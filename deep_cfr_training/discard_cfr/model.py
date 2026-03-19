"""
DiscardNet — NNUE EV predictor for discard decisions.

Architecture: pair_layer(23→H) + ctx_layer(21→H) → ReLU → H/2 → scalar EV
Strategy:
  1. Forward pass → EV estimates per keep pair
  2. candidate set = {k : EV(k) >= EV_max - EV_THRESHOLD}
     (EV_THRESHOLD = max(1 chip, 1.25 × MC stderr) ≈ 0.025 in equity units)
  3. softmax(EV * T) within candidates, zero outside
  → Clear EV winners always chosen; near-ties get strategic mixing.
"""

import numpy as np
import torch
import torch.nn as nn

from .features import FEAT_DIM, PAIR_DIM, CTX_DIM, N_KEEP_PAIRS

EV_THRESHOLD = 0.025   # max(0.01 chip/100, 1.25 × stderr≈0.02) in [0,1] equity
SOFTMAX_T    = 8.0     # temperature within candidate set


class DiscardNet(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.pair_layer = nn.Linear(PAIR_DIM, hidden_dim)
        self.ctx_layer  = nn.Linear(CTX_DIM,  hidden_dim)
        self.output     = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 44) → (B,) scalar EV estimates."""
        return self.output(
            self.pair_layer(x[:, :PAIR_DIM]) + self.ctx_layer(x[:, PAIR_DIM:])
        ).squeeze(-1)

    def get_strategy(self, all_pair_feats: np.ndarray,
                     threshold: float = EV_THRESHOLD) -> np.ndarray:
        """
        all_pair_feats: (10, 44) from build_all_feats()
        Returns strategy[10]:
          - candidates (EV within threshold of best) get softmax weight
          - non-candidates get 0
        """
        device = next(self.parameters()).device
        with torch.no_grad():
            ev = self.forward(
                torch.from_numpy(all_pair_feats).float().to(device)
            ).cpu().numpy()   # (10,)

        best  = ev.max()
        mask  = ev >= best - threshold          # candidate set
        ev_c  = ev - best                       # shift for numerical stability
        exp_v = np.exp(np.clip(ev_c * SOFTMAX_T, -20., 0.))
        exp_v[~mask] = 0.
        total = exp_v.sum()
        if total > 1e-9:
            return exp_v / total
        # fallback: uniform over candidates
        return mask.astype(np.float32) / mask.sum()


def make_net(hidden_dim: int = 128) -> DiscardNet:
    return DiscardNet(hidden_dim)

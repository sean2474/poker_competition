"""
Discard advantage network — NNUE additive architecture.

Input per keep-pair sample: feat(44) = pair(23) + ctx(21)
  pair: my_cat[17] + hi_rank[1] + lo_rank[1] + blocker[4]
  ctx:  board_ranks[3] + opp_cats[17] + is_bb[1]

NNUE trick: pair and ctx are projected independently into a shared hidden
space and ADDED before the output head. At inference the ctx embedding is
computed once and reused for all 10 keep-pair samples (board+opp are fixed).

Output: scalar advantage per sample.
Strategy: regret_matching(advantages) over the 10 keep pairs.

Single shared network for Player A and B — only opp_cats changes.
"""

import numpy as np
import torch
import torch.nn as nn

from .features import FEAT_DIM, PAIR_DIM, CTX_DIM, N_KEEP_PAIRS


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
        """x: (B, FEAT_DIM=44) → (B,) scalar advantages.
        Internally split: x[:,:PAIR_DIM=23] → pair_layer, x[:,PAIR_DIM:] → ctx_layer."""
        pair_h = self.pair_layer(x[:, :PAIR_DIM])   # (B, H)
        ctx_h  = self.ctx_layer(x[:, PAIR_DIM:])    # (B, H)
        return self.output(pair_h + ctx_h).squeeze(-1)   # (B,)

    def get_strategy(self, all_pair_feats: np.ndarray) -> np.ndarray:
        """
        all_pair_feats: (10, FEAT_DIM=44) — from build_all_feats()
        Returns: strategy[10] via regret matching on predicted advantages.
        """
        device = next(self.parameters()).device
        with torch.no_grad():
            adv = self.forward(
                torch.from_numpy(all_pair_feats).float().to(device)
            ).cpu().numpy()   # (10,)
        pos   = np.maximum(adv, 0.0)
        total = pos.sum()
        return pos / total if total > 0 else np.ones(N_KEEP_PAIRS) / N_KEEP_PAIRS


def make_net(hidden_dim: int = 128) -> DiscardNet:
    return DiscardNet(hidden_dim)

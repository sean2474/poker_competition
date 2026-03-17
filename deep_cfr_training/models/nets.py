"""
Neural networks for Deep CFR.

Two separate sets of networks:
  - Preflop nets: trained only on street-0 samples
      Input: 93-dim features (community/discard slots are zeros preflop)
      Action space: FOLD, CALL, RAISE_SMALL only
  - Postflop nets: trained on streets 1-3
      Input: 93-dim features (full board/discard info)
      Action space: FOLD, CALL, CHECK, RAISE_SMALL, RAISE_LARGE,
                    BET_SMALL, BET_LARGE, BET_POT

Both use the same architecture (street one-hot disambiguates).
"""

import torch
import torch.nn as nn

from game.constants import FEATURE_DIM, NUM_ACTIONS


class _ResBlock(nn.Module):
    """Single residual block: Linear → LayerNorm → ReLU → Linear → LayerNorm + skip."""
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        h = torch.relu(self.ln1(self.fc1(x)))
        return torch.relu(self.ln2(self.fc2(h)) + x)


class AdvantageNet(nn.Module):
    """Predicts per-action advantage values A(s, a)."""

    def __init__(self, input_dim: int = FEATURE_DIM,
                 hidden_dim: int = 512, output_dim: int = NUM_ACTIONS):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.res   = nn.Sequential(
            _ResBlock(hidden_dim),
            _ResBlock(hidden_dim),
            _ResBlock(hidden_dim),
        )
        self.head  = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.res(self.embed(x)))

    def get_strategy(self, features, valid_actions: list) -> dict:
        """Regret matching on predicted advantages → strategy dict."""
        with torch.no_grad():
            x    = torch.from_numpy(features).float().unsqueeze(0)
            advs = self.forward(x).squeeze(0).numpy()

        total = 0.0
        best_a, best_v = valid_actions[0], -1e9
        for a in valid_actions:
            v = float(advs[a])
            if v > 0:
                total += v
            if v > best_v:
                best_v, best_a = v, a

        if total > 0:
            inv = 1.0 / total
            return {a: max(float(advs[a]), 0) * inv for a in valid_actions}
        return {a: (1.0 if a == best_a else 0.0) for a in valid_actions}


# Preflop and postflop advantage nets share the same architecture
PreflopAdvantageNet  = AdvantageNet
PostflopAdvantageNet = AdvantageNet


class StrategyNet(nn.Module):
    """
    Average strategy network — trained after all CFR iterations.
    Output: action logits (apply softmax externally).
    This is the network used during actual play.
    """

    def __init__(self, input_dim: int = FEATURE_DIM,
                 hidden_dim: int = 512, output_dim: int = NUM_ACTIONS):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.res   = nn.Sequential(
            _ResBlock(hidden_dim),
            _ResBlock(hidden_dim),
            _ResBlock(hidden_dim),
        )
        self.head  = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.res(self.embed(x)))

    def get_action_probs(self, features, valid_actions: list) -> dict:
        with torch.no_grad():
            x      = torch.from_numpy(features).float().unsqueeze(0)
            logits = self.forward(x).squeeze(0)
            mask   = torch.full((NUM_ACTIONS,), float('-inf'))
            for a in valid_actions:
                mask[a] = logits[a]
            probs = torch.softmax(mask, dim=0).numpy()
        return {a: float(probs[a]) for a in valid_actions}


# Preflop and postflop strategy nets share the same architecture
PreflopStrategyNet  = StrategyNet
PostflopStrategyNet = StrategyNet

"""
action.py — Action constants, net architectures, and action mapping helpers.
Shared across strategy/preflop.py, strategy/discard.py, strategy/postflop.py.
"""

import numpy as np
import torch
import torch.nn as nn

from features import FEATURE_DIM

# ── Tournament action type values ─────────────────────────────────────────────
FOLD    = 0
RAISE   = 1
CHECK   = 2
CALL    = 3
DISCARD = 4

# ── Training action IDs ───────────────────────────────────────────────────────
# 0=FOLD  1=CALL  2=CHECK  3=BET_SMALL(33%)  4=BET_LARGE(75%)
# 5=RAISE_SMALL(33%)  6=RAISE_LARGE(75%)  7=BET_POT(100%)
NUM_ACTIONS = 8
BIG_BLIND   = 2

# ── Net architectures (must match training) ───────────────────────────────────

class _ResBlock(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.fc1 = nn.Linear(d, d); self.ln1 = nn.LayerNorm(d)
        self.fc2 = nn.Linear(d, d); self.ln2 = nn.LayerNorm(d)
    def forward(self, x):
        return torch.relu(self.ln2(self.fc2(torch.relu(self.ln1(self.fc1(x)))) + x))


class StrategyNet(nn.Module):
    """Postflop average-strategy network: 77-dim → 8 action logits."""
    def __init__(self, inp: int = FEATURE_DIM, h: int = 256):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(inp, h), nn.ReLU())
        self.res   = nn.Sequential(_ResBlock(h), _ResBlock(h))
        self.head  = nn.Linear(h, NUM_ACTIONS)
    def forward(self, x):
        return self.head(self.res(self.embed(x)))


class DiscardNet(nn.Module):
    """Discard advantage network — NNUE additive: pair(23) + ctx(21) → 1 scalar."""
    PAIR_DIM = 23
    CTX_DIM  = 21

    def __init__(self, h: int = 128):
        super().__init__()
        self.pair_layer = nn.Linear(self.PAIR_DIM, h)
        self.ctx_layer  = nn.Linear(self.CTX_DIM,  h)
        self.output = nn.Sequential(
            nn.ReLU(), nn.Linear(h, h // 2), nn.ReLU(), nn.Linear(h // 2, 1))

    def forward(self, x):
        return self.output(
            self.pair_layer(x[:, :self.PAIR_DIM]) +
            self.ctx_layer(x[:, self.PAIR_DIM:])
        ).squeeze(-1)

    def get_strategy(self, feats: np.ndarray) -> np.ndarray:
        """feats: (10, 44) → strategy[10] via regret matching."""
        with torch.no_grad():
            adv = self.forward(torch.from_numpy(feats).float()).numpy()
        pos = np.maximum(adv, 0.)
        s = pos.sum()
        return pos / s if s > 0 else np.ones(10) / 10


# ── Bet sizing helpers ────────────────────────────────────────────────────────

def bet_frac(obs: dict, frac: float) -> tuple:
    pot = obs['my_bet'] + obs['opp_bet']
    amt = max(obs['min_raise'], min(obs['max_raise'], max(int(pot * frac), 1)))
    return (RAISE, amt, 0, 0)


def map_training_action(action_idx: int, obs: dict) -> tuple:
    """Map 0-7 training action → (action_type, raise_amount, 0, 0)."""
    v = obs['valid_actions']
    if   action_idx == 0: return (FOLD,  0, 0, 0)
    elif action_idx == 1: return (CALL,  0, 0, 0) if v[CALL]  else (CHECK, 0, 0, 0)
    elif action_idx == 2: return (CHECK, 0, 0, 0) if v[CHECK] else (CALL,  0, 0, 0)
    elif action_idx in (3, 5): return bet_frac(obs, 0.33) if v[RAISE] else (CHECK, 0, 0, 0)
    elif action_idx in (4, 6): return bet_frac(obs, 0.75) if v[RAISE] else (CHECK, 0, 0, 0)
    elif action_idx == 7:
        return (RAISE, obs['max_raise'], 0, 0) if v[RAISE] else (CALL, 0, 0, 0)
    return (FOLD, 0, 0, 0)


def valid_training_actions(obs: dict) -> list:
    v = obs['valid_actions']
    out = []
    if v[FOLD]:  out.append(0)
    if v[CALL]:  out.append(1)
    if v[CHECK]: out.append(2)
    if v[RAISE]: out.extend([3, 4, 5, 6, 7])
    return out or [0]

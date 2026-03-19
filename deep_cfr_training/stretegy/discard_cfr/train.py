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

# ── Regret matching helper ────────────────────────────────────────────────────

def _regret_match(advantages: np.ndarray) -> np.ndarray:
    """Regret matching: strategy proportional to positive advantages."""
    pos = np.maximum(advantages, 0.)
    s   = pos.sum()
    return pos / s if s > 1e-9 else np.full(len(advantages), 1. / len(advantages))


# ── Episode generation (owned by Agent, called externally) ────────────────────

def generate_episodes(preflop_model, advantage_net,
                      n_episodes: int,
                      rng: random.Random = None) -> tuple:
    """
    Generate Deep CFR training data for one iteration.

    For each hand:
      1. Compute MC EV for all 10 keep combos
      2. Get current strategy via regret matching on advantage_net predictions
         (uniform if advantage_net is None — cold start)
      3. Compute advantage targets: adv[a] = EV[a] - E_pi[EV]
      4. Collect (features, adv_target) for AdvantageNet
      5. Collect (features, strategy_prob) for StrategyNet

    Returns:
        adv_X : (N, FEAT_DIM)  advantage net features
        adv_Y : (N,)           advantage targets
        str_X : (N, FEAT_DIM)  strategy net features
        str_Y : (N,)           strategy probability targets
    """
    from range_finder.core import Range

    if rng is None:
        rng = random.Random()

    adv_feats, adv_targets = [], []
    str_feats, str_probs   = [], []

    for _ in tqdm(range(n_episodes), desc='episodes', ncols=80, leave=False):
        deck = list(range(DECK_SIZE))
        rng.shuffle(deck)
        h0, h1, board = deck[:5], deck[5:10], deck[10:13]

        range_obj = Range()
        if preflop_model is not None:
            from stretegy.preflop_cfr.state import _State
            state = _State()
            hands = [h0, h1]
            while not state.done:
                cp, hist = state.acting, state.hist
                act, _   = preflop_model.action(hands[cp], hist)
                if cp == 0:
                    range_obj.update([], [], [], 'preflop', preflop_model,
                                     hero_action=act, history=hist)
                else:
                    range_obj.update([], [], [], 'preflop', preflop_model,
                                     action=act, history=hist)
                state = state.apply(act)

        opp_range  = range_obj.opp_range
        hero_range = range_obj.hero_range
        dead_base  = set(h0) | set(h1) | set(board)

        features = np.stack([
            build_features(h0[i], h0[j], board, opp_range, hero_range)
            for i, j in _KEEP_COMBOS
        ])                                              # (10, FEAT_DIM)

        ev = np.array([
            calculate_ev(board, (h0[i], h0[j]), opp_range,
                         dead=dead_base | {h0[i], h0[j]})
            for i, j in _KEEP_COMBOS
        ], dtype=np.float32)                            # (10,)

        # Current strategy via regret matching on advantage net
        if advantage_net is not None:
            with torch.no_grad():
                adv_pred = advantage_net(
                    torch.tensor(features, dtype=torch.float32)
                ).numpy()                              # (10,)
            pi = _regret_match(adv_pred)
        else:
            pi = np.full(len(_KEEP_COMBOS), 1. / len(_KEEP_COMBOS))

        # Advantage targets: EV[a] - E_pi[EV]  (for AdvantageNet)
        ev_pi      = float(pi @ ev)
        adv_target = ev - ev_pi                        # (10,)

        # StrategyNet target: logit adjustment = log(pi) - EV
        # So that softmax(EV + net_output) = pi at inference
        log_pi     = np.log(pi + 1e-9)                 # (10,)
        str_target = log_pi - ev                       # (10,)

        adv_feats.append(features)
        adv_targets.append(adv_target)
        str_feats.append(features)
        str_probs.append(str_target)

    adv_X = np.concatenate(adv_feats,   axis=0).astype(np.float32)
    adv_Y = np.concatenate(adv_targets, axis=0).astype(np.float32)
    str_X = np.concatenate(str_feats,   axis=0).astype(np.float32)
    str_Y = np.concatenate(str_probs,   axis=0).astype(np.float32)
    return adv_X, adv_Y, str_X, str_Y


# ── NN training (owned by Discard) ───────────────────────────────────────────

def _fit(net, X: np.ndarray, Y: np.ndarray,
         n_epochs: int, batch_size: int, lr: float, desc: str) -> None:
    net.train()
    Xt = torch.tensor(X, dtype=torch.float32)
    Yt = torch.tensor(Y, dtype=torch.float32)
    loader  = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xt, Yt),
        batch_size=batch_size, shuffle=True)
    opt     = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for epoch in tqdm(range(n_epochs), desc=desc, ncols=80, leave=False):
        total = 0.
        for xb, yb in loader:
            pred = net(xb); loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(xb)
        if (epoch + 1) % max(1, n_epochs // 5) == 0:
            tqdm.write(f'  [{desc}] epoch {epoch+1}/{n_epochs}  loss={total/len(X):.5f}')
    net.eval()


def train_net(adv_net, str_net,
              adv_X: np.ndarray, adv_Y: np.ndarray,
              str_X: np.ndarray, str_Y: np.ndarray,
              n_epochs:   int   = 20,
              batch_size: int   = 256,
              lr:         float = 1e-3,
              adv_save:   str   = None,
              str_save:   str   = None):
    """Train AdvantageNet on (adv_X, adv_Y) and StrategyNet on (str_X, str_Y)."""
    _fit(adv_net, adv_X, adv_Y, n_epochs, batch_size, lr, 'advantage')
    _fit(str_net, str_X, str_Y, n_epochs, batch_size, lr, 'strategy')

    for net, path in [(adv_net, adv_save), (str_net, str_save)]:
        if path:
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            torch.save(net.state_dict(), path)

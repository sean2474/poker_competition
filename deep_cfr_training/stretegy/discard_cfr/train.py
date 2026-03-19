"""
discard_cfr/train.py — Range-aware DiscardNet training.

Each episode:
  1. Deal 5+5+3 cards (hero hand, opp hand, board)
  2. Simulate preflop using preflop_model (if available) → build opp range
     via Range.update(phase='preflop'), else uniform range
  3. Compute exact EV targets for all C(5,2)=10 keep combos
  4. Collect (features, targets) as supervised training data
  5. Train DiscardNet on collected data via MSE loss
"""

import os
import random
import itertools

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .model import DiscardNet, FEAT_DIM
from .utils import build_features, calculate_ev, _ALL_PAIR, DECK_SIZE

_KEEP_COMBOS = list(itertools.combinations(range(5), 2))


# ── Preflop simulation helper ─────────────────────────────────────────────────

def _simulate_preflop(preflop_model, h0: list, h1: list, rng: random.Random):
    """
    Simulate one preflop street between two players using preflop_model.
    Returns (hero_action, opp_action, history_at_hero_act, history_at_opp_act)
    so that Range can be updated correctly.
    """
    from stretegy.preflop_cfr.state import _State

    state = _State()
    actions = []

    hands = [h0, h1]   # player 0 = h0, player 1 = h1

    while not state.done:
        cp    = state.acting
        hand  = hands[cp]
        valid = state.valid()
        hist  = state.hist

        if preflop_model is not None:
            act, _ = preflop_model.action(hand, hist)
        else:
            act = rng.choice(valid)

        actions.append((cp, act, hist))
        state = state.apply(act)

    return actions   # [(player_id, action_char, history_before_action), ...]


# ── Single training episode ───────────────────────────────────────────────────

def _episode(preflop_model, rng: random.Random):
    """
    Returns list of (features_10xFEAT_DIM, ev_targets_10) for hero (player 0).
    hero is always player 0 for simplicity; both directions sampled over episodes.
    """
    # Deal cards
    deck = list(range(DECK_SIZE))
    rng.shuffle(deck)
    h0    = deck[:5]
    h1    = deck[5:10]
    board = deck[10:13]

    # Build range from preflop history
    from range_finder.core import Range
    range_obj = Range()

    if preflop_model is not None:
        actions = _simulate_preflop(preflop_model, h0, h1, rng)
        for (cp, act, hist) in actions:
            if cp == 0:
                range_obj.update(board=[], hero_discard=[], opp_discard=[],
                                 phase='preflop', model=preflop_model,
                                 hero_action=act, history=hist)
            else:
                range_obj.update(board=[], hero_discard=[], opp_discard=[],
                                 phase='preflop', model=preflop_model,
                                 action=act, history=hist)

    opp_range = range_obj.opp_range

    # Dead cards for EV computation
    dead_base = set(h0) | set(h1) | set(board)

    # Compute exact EV targets for all 10 keep combos
    ev_targets = np.array([
        calculate_ev(board, (h0[i], h0[j]), opp_range,
                     dead=dead_base | {h0[i], h0[j]})
        for i, j in _KEEP_COMBOS
    ], dtype=np.float32)

    # Build features for each keep combo
    features = np.stack([
        build_features(h0[i], h0[j], board, opp_range)
        for i, j in _KEEP_COMBOS
    ])  # (10, FEAT_DIM)

    return features, ev_targets


# ── Main training function ────────────────────────────────────────────────────

def train(discard_model: DiscardNet = None,
          preflop_model=None,
          n_episodes: int = 5_000,
          n_epochs:   int = 20,
          batch_size: int = 256,
          lr:         float = 1e-3,
          save_path:  str = None) -> DiscardNet:
    """
    Train (or continue training) a DiscardNet.

    Args:
        discard_model : existing DiscardNet to continue training (None = new)
        preflop_model : trained Preflop model for range generation
                        (None = uniform range)
        n_episodes    : number of training hands to generate
        n_epochs      : SGD epochs over collected data
        batch_size    : mini-batch size
        lr            : Adam learning rate
        save_path     : where to save trained weights (.pt)
    """
    if discard_model is None:
        discard_model = DiscardNet()

    net = discard_model
    net.train()

    rng = random.Random()
    all_feat   = []
    all_target = []

    print(f'[discard train] generating {n_episodes} episodes ...')
    for _ in tqdm(range(n_episodes), desc='episodes', ncols=80):
        feat, tgt = _episode(preflop_model, rng)
        all_feat.append(feat)
        all_target.append(tgt)

    X = torch.tensor(np.concatenate(all_feat,   axis=0), dtype=torch.float32)
    Y = torch.tensor(np.concatenate(all_target, axis=0), dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(X, Y)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                          shuffle=True)

    opt      = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn  = nn.MSELoss()

    print(f'[discard train] training {n_epochs} epochs on {len(X):,} samples ...')
    for epoch in tqdm(range(n_epochs), desc='epochs', ncols=80):
        total_loss = 0.
        for xb, yb in loader:
            pred = net(xb).squeeze(-1)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(xb)
        if (epoch + 1) % max(1, n_epochs // 5) == 0:
            tqdm.write(f'  epoch {epoch+1}/{n_epochs}  loss={total_loss/len(X):.5f}')

    net.eval()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        torch.save(net.state_dict(), save_path)
        print(f'[discard train] saved → {save_path}')

    return net

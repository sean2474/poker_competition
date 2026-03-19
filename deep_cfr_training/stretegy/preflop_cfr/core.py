"""
preflop_cfr/core.py — Tabular External-Sampling MCCFR for preflop.

Infoset key : (canonicalize(hand5), history_string)
Action slots: 0=fold  1=call/check  2=raise

action() : inference — sample from average strategy
train()  : run MCCFR for N iterations
save()   : export chart to .pkl
"""

import os, pickle, random
import numpy as np

from game.game import (
    DECK_SIZE,
    canonicalize,
)
from .state import _State
from .utils import _cfr
from interface.model import PreflopModel

_SLOT     = {'f': 0, 'k': 1, 'c': 1, 'r': 2}
_ALL_PAIR = [(i, j) for i in range(DECK_SIZE) for j in range(i + 1, DECK_SIZE)]

class Preflop(PreflopModel):
    def __init__(self):
        self._regrets:   dict = {}     # key → np.ndarray(3) — CFR+ regrets
        self._strat_sum: dict = {}     # key → np.ndarray(3) — linear avg strategy
        self._chart:     dict = {}     # key → np.ndarray(3) normalized — cached after train

    # ── Inference ─────────────────────────────────────────────────────────────

    def action(self, hand: list, history: str) -> tuple[str, dict[str, float]]:
        """
        Sample action from average strategy for (hand, history).
        Returns (action: str, probs: dict[str, float])
          action — sampled action char: 'f' | 'k' | 'c' | 'r'
          probs  — {action: probability} for all valid actions
        """
        s = _State()
        for ch in history:
            s = s.apply(ch)
        valid = s.valid()

        key = (canonicalize(hand), history)
        ss  = self._chart.get(key)

        if ss is None or ss.sum() == 0:
            uni = 1. / len(valid)
            probs = {a: uni for a in valid}
            fallback = 'r' if 'r' in valid else ('c' if 'c' in valid else 'k')
            return fallback, probs

        slot_to_acts: dict = {}
        for a in valid:
            slot_to_acts.setdefault(_SLOT[a], []).append(a)
        pos   = [max(float(ss[sl]), 0.) for sl in range(3)]
        total = sum(pos[sl] for sl in slot_to_acts)
        if total > 0:
            slot_p = {sl: pos[sl] / total for sl in slot_to_acts}
        else:
            slot_p = {sl: 1. / len(slot_to_acts) for sl in slot_to_acts}

        weights = [slot_p[_SLOT[a]] / len(slot_to_acts[_SLOT[a]]) for a in valid]
        probs   = {a: w for a, w in zip(valid, weights)}
        action  = random.choices(valid, weights=weights)[0]
        return action, probs

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, n_iters: int = 3000, log_every: int = 500):
        rng = random.Random()
        for t in range(1, n_iters + 1):
            # Deal 5 cards to each player
            deck  = list(range(DECK_SIZE))
            rng.shuffle(deck)
            h0, h1 = deck[:5], deck[5:10]

            for tp in [0, 1]:
                _cfr(tuple(h0), tuple(h1), _State(), tp,
                     self._regrets, self._strat_sum, t, rng)

            if t % log_every == 0:
                print(f'iter {t}/{n_iters}  infosets={len(self._strat_sum)}')

        self._build_chart()

    def _build_chart(self):
        """Normalize strategy_sum → average strategy chart."""
        self._chart = {}
        for key, ss in self._strat_sum.items():
            s = ss.sum()
            self._chart[key] = ss / s if s > 0 else np.full(3, 1. / 3)

    # ── Training (CFR) ────────────────────────────────────────────────────────

    def train(self, n_iters: int = 200_000, save_path: str = None,
              discard_sims: int = 20, n_workers: int = 1,
              terminal_fn=None, **kwargs):
        from .train import train as _train_fn
        result = _train_fn(
            n_iters=n_iters, save_path=save_path,
            discard_sims=discard_sims, n_workers=n_workers,
            terminal_fn=terminal_fn,
            init_regrets=self._regrets   if self._regrets   else None,
            init_strat_sum=self._strat_sum if self._strat_sum else None,
        )
        self._chart     = result._chart
        self._regrets   = result._regrets
        self._strat_sum = result._strat_sum

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self._chart, f)
        print(f'saved {len(self._chart)} infosets → {path}')

    def load(self, path: str):
        with open(path, 'rb') as f:
            self._chart = pickle.load(f)
        print(f'loaded {len(self._chart)} infosets from {path}')
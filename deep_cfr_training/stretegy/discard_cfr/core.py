import itertools
import os
import random
from typing import Optional

import numpy as np
import torch

from .model import AdvantageNet, StrategyNet, FEAT_DIM
from .utils import (
    build_features, calculate_ev,
    _ALL_PAIR, DECK_SIZE,
)
from interface.model import DiscardModel

# All C(5,2)=10 ways to choose 2 indices from a 5-card hand
_KEEP_COMBOS = list(itertools.combinations(range(5), 2))


class Discard(DiscardModel):
    def __init__(self):
        self._adv_net: AdvantageNet = None
        self._str_net: StrategyNet  = None

    @property
    def _net(self):
        """Backward compat: expose _adv_net as _net."""
        return self._adv_net

    def load(self, path: str):
        """Load StrategyNet weights (inference model)."""
        self._str_net = StrategyNet()
        self._str_net.load_state_dict(torch.load(path, map_location='cpu'))
        self._str_net.eval()
        print(f'StrategyNet loaded from {path}')

    def save(self, path: str):
        """Save StrategyNet weights."""
        if self._str_net is None:
            raise RuntimeError('StrategyNet not trained yet')
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save(self._str_net.state_dict(), path)
        print(f'StrategyNet saved to {path}')

    def save_adv(self, path: str):
        """Save AdvantageNet weights for training resumption."""
        if self._adv_net is None:
            raise RuntimeError('AdvantageNet not trained yet')
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save(self._adv_net.state_dict(), path)

    def load_adv(self, path: str):
        """Load AdvantageNet weights for training resumption."""
        self._adv_net = AdvantageNet()
        self._adv_net.load_state_dict(torch.load(path, map_location='cpu'))
        self._adv_net.eval()

    def train(self, adv_X: np.ndarray, adv_Y: np.ndarray,
              str_X: np.ndarray, str_Y: np.ndarray,
              n_epochs: int = 20, batch_size: int = 256,
              lr: float = 1e-3,
              adv_save: str = None, str_save: str = None, **kwargs):
        """Train AdvantageNet + StrategyNet on Deep CFR data."""
        from .train import train_net
        if self._adv_net is None:
            self._adv_net = AdvantageNet()
        if self._str_net is None:
            self._str_net = StrategyNet()
        train_net(
            self._adv_net, self._str_net,
            adv_X, adv_Y, str_X, str_Y,
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr,
            adv_save=adv_save,
            str_save=str_save,
        )

    def action(self, board: list, hand: list, history: str,
               hero_range: list, opp_range: list,
               opp_discard_card: Optional[list] = None,
               temperature: float = 1.0) -> tuple[tuple[int, int], np.ndarray]:
        """
        Choose which 2 cards to keep from 5-card hand.

        Returns:
            keep_idx : tuple[int, int]  — indices (0-4) into hand of kept cards
            probs    : np.ndarray(10)   — probability for each of 10 keep combos
        """
        # 공개카드로 range 에 있는거 제거
        dead = set(hand) | {b for b in board if b >= 0}
        if opp_discard_card:
            dead |= {c for c in opp_discard_card if c >= 0}

        opp_w = np.asarray(opp_range, dtype=np.float32).copy()
        for idx, (c0, c1) in enumerate(_ALL_PAIR):
            if c0 in dead or c1 in dead:
                opp_w[idx] = 0.
        s = opp_w.sum()
        if s > 1e-9:
            opp_w /= s

        hero_w = np.asarray(hero_range, dtype=np.float32).copy()
        hs = hero_w.sum()
        if hs > 1e-9:
            hero_w /= hs

        # EV for all 10 keep combos (hand strength baseline)
        ev = self._mc_evs(hand, board, opp_w, dead)

        # Network logit adjustment (strategic deviation from EV)
        if self._str_net is not None:
            adj = self._net_output(self._str_net, hand, board, opp_w, hero_w)
        elif self._adv_net is not None:
            adj = self._net_output(self._adv_net, hand, board, opp_w, hero_w)
        else:
            adj = np.zeros(len(_KEEP_COMBOS), dtype=np.float32)

        # score = EV + adjustment → softmax → strategy
        score = (ev + adj).astype(np.float64) / temperature
        score -= score.max()
        probs = np.exp(score).astype(np.float32)
        probs /= probs.sum()

        combo_idx = random.choices(range(len(_KEEP_COMBOS)), weights=probs)[0]
        return _KEEP_COMBOS[combo_idx], probs

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _net_output(self, net, hand: list, board: list,
                    opp_range: np.ndarray,
                    hero_range: np.ndarray) -> np.ndarray:
        """Batch forward pass for all 10 keep combos. Returns logit adjustments."""
        feats = np.stack([
            build_features(hand[i], hand[j], board, opp_range, hero_range)
            for i, j in _KEEP_COMBOS
        ])                                          # (10, FEAT_DIM)
        with torch.no_grad():
            out = net(torch.tensor(feats, dtype=torch.float32)).numpy()
        return out                                  # (10,)

    def _mc_evs(self, hand: list, board: list, opp_range: np.ndarray,
                dead: set, n_samples: int = 50) -> np.ndarray:
        """MC EV for all 10 keep combos (fallback when model not loaded)."""
        return np.array([
            calculate_ev(board, (hand[i], hand[j]), opp_range, dead,
                         n_samples=n_samples)
            for i, j in _KEEP_COMBOS
        ])

    def compute_all_evs(self, board: list, hand: list,
                        opp_range: np.ndarray, dead: set = None,
                        n_samples: int = None) -> np.ndarray:
        """
        Exact (or MC) EV for all 10 keep combos. Used to generate training targets.
        Returns np.ndarray(10) of equity values in [0, 1].
        """
        if dead is None:
            dead = set(hand) | set(board)
        opp_w = np.asarray(opp_range, dtype=np.float32)
        return np.array([
            calculate_ev(board, (hand[i], hand[j]), opp_w, dead,
                         n_samples=n_samples)
            for i, j in _KEEP_COMBOS
        ])
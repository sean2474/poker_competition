import itertools
import os
import random
from typing import Optional

import numpy as np
import torch

from .model import DiscardNet, FEAT_DIM
from .utils import (
    build_features, calculate_ev,
    _ALL_PAIR, DECK_SIZE,
)
from interface.model import DiscardModel

# All C(5,2)=10 ways to choose 2 indices from a 5-card hand
_KEEP_COMBOS = list(itertools.combinations(range(5), 2))


class Discard(DiscardModel):
    def __init__(self):
        self._net: DiscardNet = None

    def load(self, path: str):
        self._net = DiscardNet()
        self._net.load_state_dict(torch.load(path, map_location='cpu'))
        self._net.eval()
        print(f'DiscardNet loaded from {path}')

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save(self._net.state_dict(), path)
        print(f'DiscardNet saved to {path}')

    def train(self, preflop_model=None, n_episodes: int = 5_000,
              n_epochs: int = 20, batch_size: int = 256,
              lr: float = 1e-3, save_path: str = None, **kwargs):
        from .train import train as _train_fn
        self._net = _train_fn(
            discard_model=self._net,
            preflop_model=preflop_model,
            n_episodes=n_episodes,
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr,
            save_path=save_path,
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

        # 각 discard의 ev계산
        if self._net is not None:
            evs = self._net_evs(hand, board, opp_w)
        else:
            evs = self._mc_evs(hand, board, opp_w, dead)

        # range를 feature화, 내 실제 핸드들의 현재 strength 추출
        # → build_features: hand_category(17) + blockers(4) + board_texture(4) + opp(13)

        # 추출한 features, board texture 등등, 모델에 전달
        # → _net_evs: batch forward pass through DiscardNet

        # 모델에서 받은 각각의 출력을 바탕으로 확률 계산해서 리턴
        ev_t = np.array(evs, dtype=np.float64) / temperature
        ev_t -= ev_t.max()
        probs = np.exp(ev_t).astype(np.float32)
        probs /= probs.sum()

        combo_idx = random.choices(range(len(_KEEP_COMBOS)), weights=probs)[0]
        return _KEEP_COMBOS[combo_idx], probs

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _net_evs(self, hand: list, board: list,
                 opp_range: np.ndarray) -> np.ndarray:
        """Batch forward pass for all 10 keep combos."""
        feats = np.stack([
            build_features(hand[i], hand[j], board, opp_range)
            for i, j in _KEEP_COMBOS
        ])                                          # (10, FEAT_DIM)
        with torch.no_grad():
            evs = self._net(torch.tensor(feats, dtype=torch.float32)).numpy()
        return evs                                  # (10,)

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
"""
DiscardCFR — Deep CFR for discard decisions.

Implements IDiscardTrainer:
  get_strategy()  : neural net inference (pair feature → scalar advantage)
  run_iter()      : accumulate regret samples into buffer
  train()         : retrain net from scratch on buffer
  is_converged()  : loss plateau detection (Phase 2 → 3 signal)

Single shared NNUE network for both Player A (SB) and Player B (BB).
"""

import os
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from interfaces import IDiscardTrainer
from .features import (KEEP_PAIRS, N_KEEP_PAIRS, FEAT_DIM,
                       classify_all_pairs, pair_rank_feats, pair_blocker_feats,
                       build_all_feats, opp_cats_uniform, opp_cats_narrowed)
from .model   import DiscardNet, make_net
from .buffer  import DiscardBuffer
from .traversal import run_batch


class DiscardCFR(IDiscardTrainer):
    """
    Deep CFR trainer for discard decisions.
    Uses a single NNUE network shared between Player A (SB) and B (BB).
    """

    def __init__(self,
                 lr:          float = 1e-3,
                 buffer_size: int   = 500_000,
                 batch_size:  int   = 32_768,
                 num_batches: int   = 30,
                 hidden_dim:  int   = 128):
        self.lr          = lr
        self.batch_size  = batch_size
        self.num_batches = num_batches
        self.hidden_dim  = hidden_dim
        self.iteration   = 0

        dev = ('cuda' if torch.cuda.is_available()
               else 'mps' if torch.backends.mps.is_available()
               else 'cpu')
        self.device = torch.device(dev)

        self.net = make_net(hidden_dim).to(self.device)
        self.buf = DiscardBuffer(buffer_size)

        # Loss plateau tracking for is_converged()
        self._recent_losses: list = []
        self._patience    = 10
        self._min_delta   = 0.02

    # ── IDiscardModel ─────────────────────────────────────────────────────────

    def get_strategy(self,
                     hand5,
                     board3,
                     opp_cats: np.ndarray = None,
                     is_bb: bool = False) -> np.ndarray:
        """
        Returns float32[10] strategy over KEEP_PAIRS via regret matching.
        If opp_cats is None, computes uniform range prior.
        """
        if opp_cats is None:
            opp_cats = opp_cats_uniform(hand5, board3)

        pcats   = classify_all_pairs(hand5, board3)
        pranks  = pair_rank_feats(hand5)
        pblk    = pair_blocker_feats(hand5, board3)
        feats   = build_all_feats(pcats, pranks, pblk, board3, opp_cats, is_bb)
        return self.net.get_strategy(feats)

    # ── IDiscardTrainer ───────────────────────────────────────────────────────

    def run_iter(self, hand5_As, hand5_Bs, boards5) -> None:
        """Traverse a batch of games and add 20×N samples to buffer.
        Called from a single thread (main training loop); no locking needed."""
        self.iteration += 1
        self.net.eval()
        feats, advs = run_batch(hand5_As, hand5_Bs, boards5,
                                self.net, float(self.iteration))
        self.buf.add_batch(feats, advs, float(self.iteration))
        self.net.train()  # restore train mode for subsequent training

    def train(self) -> float:
        """Retrain net from scratch on buffer. Returns mean loss."""
        assert len(self.buf) >= self.batch_size, (
            f'discard buffer too small to train: {len(self.buf)} < {self.batch_size}'
        )
        net = DiscardNet(self.hidden_dim).to(self.device)
        net.train()
        opt        = optim.Adam(net.parameters(), lr=self.lr)
        T          = float(self.iteration)
        total_loss = 0.0

        for _ in range(self.num_batches):
            sample = self.buf.sample(self.batch_size)
            assert sample is not None, 'discard buffer returned None during training'
            feats, advs, iters = sample
            w   = torch.from_numpy(
                    (2.0 * iters / (T * (T + 1.0))).astype(np.float32)
                  ).to(self.device)
            x   = torch.from_numpy(feats).to(self.device)
            tgt = torch.from_numpy(advs).to(self.device)
            pred = net(x)
            loss = (w * (pred - tgt) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += float(loss.item())

        net.eval()
        self.net = net
        mean_loss = total_loss / max(self.num_batches, 1)

        # Track for convergence
        if mean_loss > 0:
            self._recent_losses.append(mean_loss)
            if len(self._recent_losses) > self._patience:
                self._recent_losses.pop(0)

        return mean_loss

    def is_converged(self) -> bool:
        """True when loss range over recent window < 2% relative."""
        if len(self._recent_losses) < self._patience:
            return False
        lo, hi = min(self._recent_losses), max(self._recent_losses)
        return hi > 0 and (hi - lo) / hi < self._min_delta

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def export(self, path: str):
        p = path if path.endswith('.pt') else path + '.pt'
        os.makedirs(os.path.dirname(p) or '.', exist_ok=True)
        torch.save({'net': self.net.state_dict(),
                    'iteration': self.iteration,
                    'hidden_dim': self.hidden_dim}, p)

    def load(self, path: str):
        p    = path if path.endswith('.pt') else path + '.pt'
        ckpt = torch.load(p, map_location=self.device)
        self.hidden_dim = ckpt.get('hidden_dim', self.hidden_dim)
        self.net = make_net(self.hidden_dim).to(self.device)
        self.net.load_state_dict(ckpt['net'])
        self.iteration = ckpt.get('iteration', 0)
        self.net.eval()

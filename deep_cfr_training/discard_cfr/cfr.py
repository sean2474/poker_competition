"""
DiscardCFR — trainer for EV-based hybrid discard strategy.

run_iter(): run_batch → add to buffer
train():    MSE(pred, target) weighted by iteration (linear CFR weighting)
"""

import os
import numpy as np
import torch
import torch.optim as optim

from interfaces import IDiscardTrainer
from .features import (KEEP_PAIRS, N_KEEP_PAIRS,
                       classify_all_pairs, pair_rank_feats, pair_blocker_feats,
                       build_all_feats, opp_cats_uniform, opp_cats_narrowed)
from .model   import DiscardNet, make_net, EV_THRESHOLD
from .buffer  import DiscardBuffer
from .traversal import run_batch


class DiscardCFR(IDiscardTrainer):
    def __init__(self,
                 lr:           float = 1e-3,
                 buffer_size:  int   = 500_000,
                 batch_size:   int   = 32_768,
                 num_batches:  int   = 30,
                 hidden_dim:   int   = 128,
                 ev_threshold: float = EV_THRESHOLD):
        self.lr           = lr
        self.batch_size   = batch_size
        self.num_batches  = num_batches
        self.hidden_dim   = hidden_dim
        self.ev_threshold = ev_threshold
        self.iteration    = 0

        dev = ('cuda' if torch.cuda.is_available()
               else 'mps' if torch.backends.mps.is_available()
               else 'cpu')
        self.device = torch.device(dev)

        self.net = make_net(hidden_dim).to(self.device)
        self.buf = DiscardBuffer(buffer_size)

    # ── IDiscardModel ──────────────────────────────────────────────────────────

    def get_strategy(self, hand5, board3, opp_cats=None, is_bb=False):
        """Returns float32[10] strategy via candidate-masked softmax."""
        if opp_cats is None:
            opp_cats = opp_cats_uniform(hand5, board3)
        pcats  = classify_all_pairs(hand5, board3)
        pranks = pair_rank_feats(hand5)
        pblk   = pair_blocker_feats(hand5, board3)
        feats  = build_all_feats(pcats, pranks, pblk, board3, opp_cats, is_bb)
        return self.net.get_strategy(feats, threshold=self.ev_threshold)

    # ── IDiscardTrainer ────────────────────────────────────────────────────────

    def run_iter(self, hand5_As, hand5_Bs, boards5) -> None:
        """Traverse batch, store mean-centered EV targets in buffer."""
        self.iteration += 1
        self.net.eval()
        feats, targets = run_batch(
            hand5_As, hand5_Bs, boards5,
            self.net, float(self.iteration),
            threshold=self.ev_threshold,
        )
        self.buf.add_batch(feats, targets, float(self.iteration))
        self.net.train()

    def train(self) -> float:
        """Retrain net from scratch on buffer. Returns mean loss."""
        assert len(self.buf) >= self.batch_size
        net = DiscardNet(self.hidden_dim).to(self.device)
        net.train()
        opt = optim.Adam(net.parameters(), lr=self.lr)
        T   = float(self.iteration)
        total_loss = 0.0

        for _ in range(self.num_batches):
            sample = self.buf.sample(self.batch_size)
            assert sample is not None
            feats, targets, iters = sample
            w    = torch.from_numpy(
                       (2.0 * iters / (T * (T + 1.0))).astype(np.float32)
                   ).to(self.device)
            x    = torch.from_numpy(feats).to(self.device)
            tgt  = torch.from_numpy(targets).to(self.device)
            pred = net(x)
            loss = (w * (pred - tgt) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += float(loss.item())

        net.eval()
        self.net = net
        return total_loss / max(self.num_batches, 1)

    def is_converged(self) -> bool:
        return False   # let phase runner decide convergence

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def export(self, path: str):
        p = path if path.endswith('.pt') else path + '.pt'
        os.makedirs(os.path.dirname(p) or '.', exist_ok=True)
        torch.save({
            'net':          self.net.state_dict(),
            'iteration':    self.iteration,
            'hidden_dim':   self.hidden_dim,
            'ev_threshold': self.ev_threshold,
        }, p)

    def load(self, path: str):
        p    = path if path.endswith('.pt') else path + '.pt'
        ckpt = torch.load(p, map_location=self.device)
        self.hidden_dim   = ckpt.get('hidden_dim',   self.hidden_dim)
        self.ev_threshold = ckpt.get('ev_threshold', self.ev_threshold)
        self.net = make_net(self.hidden_dim).to(self.device)
        self.net.load_state_dict(ckpt['net'])
        self.iteration = ckpt.get('iteration', 0)
        self.net.eval()

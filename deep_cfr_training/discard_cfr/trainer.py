"""
DiscardCFR trainer.

Single shared network for Player A and B.
  net: DiscardNet  (NNUE, 39-dim → scalar advantage)
  buf: DiscardBuffer (scalar targets, 20 samples/game)

Joint training usage (called from postflop_cfr/runner.py):
  trainer = DiscardCFR()
  trainer.run_one_iter(p0h5, p1h5, comms)   # fill buffer
  trainer.train()                            # update net

Standalone usage:
  trainer.run(n_iters=200, n_games=300)
  trainer.export('model/discard.pt')
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from game.features import batch_deal_discard
from .model     import make_net, DiscardNet
from .buffer    import DiscardBuffer
from .traversal import run_batch


class DiscardCFR:
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

    # ── One iteration ─────────────────────────────────────────────────────────

    def run_one_iter(self, hand5_As, hand5_Bs, boards5):
        """Traverse a batch of games and add 20×N samples to buffer."""
        self.iteration += 1
        net_cpu = self.net.cpu().eval()

        feats, advs = run_batch(
            hand5_As, hand5_Bs, boards5,
            net_cpu, float(self.iteration),
        )

        self.buf.add_batch(feats, advs, float(self.iteration))
        self.net.to(self.device)

    # ── Training step ─────────────────────────────────────────────────────────

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

            # Linear CFR weighting: w_t = 2t / T(T+1)
            w   = torch.from_numpy(
                    (2.0 * iters / (T * (T + 1.0))).astype(np.float32)
                  ).to(self.device)
            x   = torch.from_numpy(feats).to(self.device)
            tgt = torch.from_numpy(advs).to(self.device)

            pred = net(x)                                   # (B,)
            loss = (w * (pred - tgt) ** 2).mean()

            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += float(loss.item())

        net.eval()
        self.net = net
        return total_loss / max(self.num_batches, 1)

    # ── Full standalone training loop ─────────────────────────────────────────

    def run(self, n_iters: int = 200, n_games: int = 300,
            n_mc: int = 40, train_every: int = 1,
            checkpoint_every: int = 50, checkpoint_dir: str = 'model'):
        """
        Standalone Phase 2 training loop.

        n_games  : games per iteration (more → slower but lower variance)
        n_mc     : MC runout samples per (A_keep, B_keep) pair
        """
        os.makedirs(checkpoint_dir, exist_ok=True)

        for t in tqdm(range(n_iters), desc='DiscardCFR'):
            r = batch_deal_discard(n_games)
            _, _, _, _, comms, p0h5, p1h5 = r

            self.run_one_iter(p0h5, p1h5, comms)

            if (t + 1) % train_every == 0:
                loss = self.train()
                tqdm.write(f'  iter {t+1:4d}  buf={len(self.buf):>7d}'
                           f'  loss={loss:.4f}')

            if (t + 1) % checkpoint_every == 0:
                ckpt = os.path.join(checkpoint_dir, f'discard_ckpt_{t+1:04d}.pt')
                self.export(ckpt)
                tqdm.write(f'  [ckpt] {ckpt}')

    # ── Export / load ─────────────────────────────────────────────────────────

    def export(self, path: str):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        p = path if path.endswith('.pt') else path + '.pt'
        torch.save({
            'net':        self.net.state_dict(),
            'iteration':  self.iteration,
            'hidden_dim': self.hidden_dim,
        }, p)

    def load(self, path: str):
        p    = path if path.endswith('.pt') else path + '.pt'
        ckpt = torch.load(p, map_location=self.device)
        self.hidden_dim = ckpt.get('hidden_dim', self.hidden_dim)
        self.net = make_net(self.hidden_dim).to(self.device)
        self.net.load_state_dict(ckpt['net'])
        self.iteration = ckpt.get('iteration', 0)
        self.net.eval()

"""
Reservoir buffer for discard CFR training samples.

Each sample: (feat[39], scalar_advantage, iteration).
One game generates 20 independent samples (10 pairs × 2 players).
"""

import random
import numpy as np

from .features import FEAT_DIM


class DiscardBuffer:
    def __init__(self, max_size: int = 500_000):
        self.max_size = max_size
        self.feats = np.zeros((max_size, FEAT_DIM), dtype=np.float32)
        self.advs  = np.zeros(max_size,             dtype=np.float32)
        self.iters = np.zeros(max_size,             dtype=np.float32)
        self.size  = 0
        self.total = 0

    def add_batch(self, feats: np.ndarray, advs: np.ndarray, iteration: float):
        """feats: (N, 39)  advs: (N,)  — reservoir sampling."""
        n = len(feats)
        for i in range(n):
            self.total += 1
            if self.size < self.max_size:
                idx = self.size
                self.size += 1
            else:
                idx = random.randint(0, self.total - 1)
                if idx >= self.max_size:
                    continue
            self.feats[idx] = feats[i]
            self.advs[idx]  = advs[i]
            self.iters[idx] = float(iteration)

    def sample(self, batch_size: int):
        """Returns (feats(B,39), advs(B,), iters(B,)) or None."""
        if self.size == 0:
            return None
        idx = np.random.choice(self.size, min(batch_size, self.size), replace=False)
        return self.feats[idx], self.advs[idx], self.iters[idx]

    def __len__(self) -> int:
        return self.size

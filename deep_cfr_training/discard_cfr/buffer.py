"""
Reservoir buffer for discard CFR samples.
Stores (features: N×44, ev_targets: N, iterations: N).
"""

import random
import numpy as np

from .features import FEAT_DIM


class DiscardBuffer:
    def __init__(self, capacity: int = 500_000):
        self.capacity  = capacity
        self._feats    = None   # (capacity, 44)
        self._targets  = None   # (capacity,)
        self._iters    = None   # (capacity,)
        self._size     = 0
        self._total    = 0      # total samples seen (for reservoir)

    def _init(self):
        if self._feats is None:
            self._feats   = np.empty((self.capacity, FEAT_DIM), dtype=np.float32)
            self._targets = np.empty(self.capacity,             dtype=np.float32)
            self._iters   = np.empty(self.capacity,             dtype=np.float32)

    def add_batch(self, feats: np.ndarray, targets: np.ndarray, iteration: float):
        """Add N samples. feats: (N, 44), targets: (N,)."""
        self._init()
        n = len(feats)
        for k in range(n):
            self._total += 1
            if self._size < self.capacity:
                idx = self._size
                self._size += 1
            else:
                idx = random.randint(0, self._total - 1)
                if idx >= self.capacity:
                    continue
            self._feats[idx]   = feats[k]
            self._targets[idx] = targets[k]
            self._iters[idx]   = iteration

    def sample(self, batch_size: int):
        """Returns (feats, targets, iters) or None if buffer too small."""
        if self._size < batch_size:
            return None
        idx = np.random.choice(self._size, batch_size, replace=False)
        return (
            self._feats[:self._size][idx],
            self._targets[:self._size][idx],
            self._iters[:self._size][idx],
        )

    def __len__(self):
        return self._size

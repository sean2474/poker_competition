"""
ReservoirBuffer with stratified street sampling.

Preflop samples dominate without stratification (~4x more than river).
Each street gets its own sub-buffer (equal capacity).

Optimized: pre-allocated NumPy arrays instead of Python lists of tuples.
Sampling uses np.random.choice + fancy indexing → 7-10x faster than
Python random.sample + list-comprehension + np.array().
"""

import random
import numpy as np


_FEAT_DIM  = 119
_VAL_DIM   = 8


class ReservoirBuffer:
    def __init__(self, capacity: int = 2_000_000):
        self.capacity      = capacity
        self._sub_cap      = capacity // 4
        self.count         = 0
        self.street_counts = [0] * 4

        # Pre-allocated contiguous NumPy arrays per street (lazy init)
        self._feats  = [None] * 4   # (sub_cap, FEAT_DIM)  float32
        self._values = [None] * 4   # (sub_cap, VAL_DIM)   float32
        self._iters  = [None] * 4   # (sub_cap,)            float32
        self._masks  = [None] * 4   # (sub_cap, VAL_DIM)   float32
        self._size   = [0] * 4      # current fill level

    # ── Lazy initializer ─────────────────────────────────────────────────────

    def _init(self, s: int):
        if self._feats[s] is None:
            self._feats[s]  = np.empty((self._sub_cap, _FEAT_DIM), dtype=np.float32)
            self._values[s] = np.empty((self._sub_cap, _VAL_DIM),  dtype=np.float32)
            self._iters[s]  = np.empty(self._sub_cap,               dtype=np.float32)
            self._masks[s]  = np.ones( (self._sub_cap, _VAL_DIM),  dtype=np.float32)

    # ── Single-sample insertion ───────────────────────────────────────────────

    def add(self, features, values, iteration, valid_mask=None, street: int = 0):
        s = max(0, min(int(street), 3))
        self._init(s)
        self.count += 1
        self.street_counts[s] += 1

        if self._size[s] < self._sub_cap:
            idx = self._size[s]
            self._size[s] += 1
        else:
            idx = random.randint(0, self.street_counts[s] - 1)
            if idx >= self._sub_cap:
                return

        self._feats[s][idx]  = features
        self._values[s][idx] = values
        self._iters[s][idx]  = float(iteration)
        if valid_mask is not None:
            self._masks[s][idx] = valid_mask

    # ── Batch insertion (C++ output) ──────────────────────────────────────────

    def add_batch(self, features_arr, values_arr, iterations_arr, masks_arr, streets_arr):
        """Bulk-add N samples from numpy arrays; fully vectorized per street."""
        for s in range(4):
            mask = streets_arr == s
            n_new = int(mask.sum())
            if n_new == 0:
                continue
            self._init(s)
            self.count += n_new
            self.street_counts[s] += n_new

            f = features_arr[mask]
            v = values_arr[mask]
            it = iterations_arr[mask]
            m = masks_arr[mask]

            cur = self._size[s]
            space = self._sub_cap - cur

            if space >= n_new:
                # Buffer has room — direct write
                self._feats[s][cur:cur + n_new]  = f
                self._values[s][cur:cur + n_new] = v
                self._iters[s][cur:cur + n_new]  = it
                self._masks[s][cur:cur + n_new]  = m
                self._size[s] += n_new
            else:
                # Fill remaining space first
                if space > 0:
                    self._feats[s][cur:]  = f[:space]
                    self._values[s][cur:] = v[:space]
                    self._iters[s][cur:]  = it[:space]
                    self._masks[s][cur:]  = m[:space]
                    self._size[s] = self._sub_cap

                # Reservoir replacement for overflow items
                for k in range(space, n_new):
                    cnt = self.street_counts[s] - n_new + k + 1
                    idx = random.randint(0, cnt - 1)
                    if idx < self._sub_cap:
                        self._feats[s][idx]  = f[k]
                        self._values[s][idx] = v[k]
                        self._iters[s][idx]  = it[k]
                        self._masks[s][idx]  = m[k]

    # ── Fast NumPy sampling ───────────────────────────────────────────────────

    def _sample_one_street(self, s: int, k: int):
        n = self._size[s]
        k = min(k, n)
        idx = np.random.choice(n, k, replace=False)
        return (self._feats[s][idx],
                self._values[s][idx],
                self._iters[s][idx],
                self._masks[s][idx])

    def sample_streets(self, streets: list, batch_size: int):
        """Sample equally from each active street; returns (feats, vals, iters, masks)."""
        non_empty = [s for s in streets if self._size[s] > 0]
        if not non_empty:
            return None
        per = max(1, batch_size // len(non_empty))
        parts = [self._sample_one_street(s, per) for s in non_empty]
        return (np.concatenate([p[0] for p in parts], axis=0),
                np.concatenate([p[1] for p in parts], axis=0),
                np.concatenate([p[2] for p in parts], axis=0),
                np.concatenate([p[3] for p in parts], axis=0))

    def sample_street(self, street: int, batch_size: int):
        if self._size[street] == 0:
            return None
        f, v, it, m = self._sample_one_street(street, batch_size)
        return f, v, it, m

    def sample(self, batch_size: int):
        return self.sample_streets(list(range(4)), batch_size)

    # ── Compatibility shim: street_bufs[s] truthiness ────────────────────────

    @property
    def street_bufs(self):
        """Read-only shim so old code `if buf.street_bufs[s]` still works."""
        return _StreetBufProxy(self._size)

    def __len__(self):
        return sum(self._size)


class _SizedInt(int):
    """int subclass that also supports len() — makes street_bufs[s] work with len()."""
    def __len__(self): return int(self)


class _StreetBufProxy:
    """Lightweight proxy: index returns truthy iff street has data, len() returns size."""
    __slots__ = ('_sizes',)
    def __init__(self, sizes): self._sizes = sizes
    def __getitem__(self, i):  return _SizedInt(self._sizes[i])

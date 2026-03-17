"""
ReservoirBuffer with stratified street sampling.

Preflop samples dominate without stratification (~4x more than river).
Each street gets its own sub-buffer (equal capacity).
Sampling draws equally from each non-empty street.
"""

import random
import numpy as np


class ReservoirBuffer:
    def __init__(self, capacity: int = 2_000_000):
        self.capacity = capacity
        self.count = 0
        self.street_bufs   = [[] for _ in range(4)]   # one per street
        self.street_counts = [0]  * 4
        self._sub_cap = capacity // 4

    def add(self, features, values, iteration, valid_mask=None, street: int = 0):
        s = max(0, min(int(street), 3))
        self.count += 1
        self.street_counts[s] += 1
        item = (features, values, iteration, valid_mask)
        buf  = self.street_bufs[s]
        if len(buf) < self._sub_cap:
            buf.append(item)
        else:
            idx = random.randint(0, self.street_counts[s] - 1)
            if idx < self._sub_cap:
                buf[idx] = item

    def add_batch(self, features_arr, values_arr, iterations_arr, masks_arr, streets_arr):
        """Bulk-add N samples from numpy arrays (fast path for C++ buffer output)."""
        n = len(streets_arr)
        for k in range(n):
            s = max(0, min(int(streets_arr[k]), 3))
            self.count += 1
            self.street_counts[s] += 1
            item = (features_arr[k], values_arr[k], float(iterations_arr[k]), masks_arr[k])
            buf  = self.street_bufs[s]
            if len(buf) < self._sub_cap:
                buf.append(item)
            else:
                idx = random.randint(0, self.street_counts[s] - 1)
                if idx < self._sub_cap:
                    buf[idx] = item

    def sample(self, batch_size: int):
        non_empty = [s for s in range(4) if self.street_bufs[s]]
        if not non_empty:
            dummy = np.zeros((1, 1), dtype=np.float32)
            return dummy, dummy, np.ones(1, dtype=np.float32), dummy

        per_street = max(1, batch_size // len(non_empty))
        batch = []
        for s in non_empty:
            k = min(per_street, len(self.street_bufs[s]))
            batch.extend(random.sample(self.street_bufs[s], k))

        if len(batch) > batch_size:
            batch = random.sample(batch, batch_size)

        features   = np.array([b[0] for b in batch])
        values     = np.array([b[1] for b in batch])
        iterations = np.array([b[2] for b in batch], dtype=np.float32)
        masks      = np.array(
            [b[3] if b[3] is not None else np.ones(values.shape[1]) for b in batch]
        )
        return features, values, iterations, masks

    def sample_street(self, street: int, batch_size: int):
        """Sample only from a specific street's sub-buffer."""
        buf = self.street_bufs[street]
        if not buf:
            return None
        batch = random.sample(buf, min(batch_size, len(buf)))
        features   = np.array([b[0] for b in batch])
        values     = np.array([b[1] for b in batch])
        iterations = np.array([b[2] for b in batch], dtype=np.float32)
        masks      = np.array(
            [b[3] if b[3] is not None else np.ones(values.shape[1]) for b in batch]
        )
        return features, values, iterations, masks

    def sample_streets(self, streets: list, batch_size: int):
        """Sample from a list of streets (e.g., [1,2,3] for postflop)."""
        non_empty = [s for s in streets if self.street_bufs[s]]
        if not non_empty:
            return None
        per_street = max(1, batch_size // len(non_empty))
        batch = []
        for s in non_empty:
            k = min(per_street, len(self.street_bufs[s]))
            batch.extend(random.sample(self.street_bufs[s], k))
        if len(batch) > batch_size:
            batch = random.sample(batch, batch_size)
        features   = np.array([b[0] for b in batch])
        values     = np.array([b[1] for b in batch])
        iterations = np.array([b[2] for b in batch], dtype=np.float32)
        masks      = np.array(
            [b[3] if b[3] is not None else np.ones(values.shape[1]) for b in batch]
        )
        return features, values, iterations, masks

    def __len__(self):
        return sum(len(b) for b in self.street_bufs)

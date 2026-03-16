"""
Compress strategy.pkl into compact numpy arrays for submission.

Output files:
  data/strategy_keys.npy    - uint64 hashed infoset keys
  data/strategy_acttype.npy - uint8 action list type per node (0-4)
  data/strategy_probs.npy   - uint8 quantized probabilities (0-255 = 0.0-1.0)
  data/strategy_meta.pkl    - small metadata (action list mapping, iterations)

Saves ~80-90% space vs raw pickle.
"""

import pickle
import numpy as np
import hashlib
import os
import sys
import struct

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

INPUT_PATH = os.path.join(os.path.dirname(__file__), "data", "strategy.pkl")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")

# The 5 known action list types (from training analysis)
ACTION_LISTS = [
    ("FOLD", "CALL", "JAM"),                       # 0
    ("FOLD", "CALL"),                               # 1
    ("FOLD", "CALL", "RAISE_SMALL", "RAISE_LARGE"), # 2
    ("CHECK", "BET_SMALL", "BET_LARGE"),            # 3
    ("CHECK",),                                     # 4
]
ACTLIST_TO_ID = {al: i for i, al in enumerate(ACTION_LISTS)}
MAX_ACTIONS = 4  # max number of actions across all types


def hash_key(key: tuple) -> int:
    """Hash a tuple key to uint64."""
    raw = pickle.dumps(key)
    h = hashlib.md5(raw).digest()[:8]
    return struct.unpack('<Q', h)[0]


def quantize_prob(p: float) -> int:
    """Quantize probability [0,1] to uint8 [0,255]."""
    return max(0, min(255, int(round(p * 255))))


def dequantize_prob(q: int) -> float:
    """Dequantize uint8 back to float."""
    return q / 255.0


def compress():
    print(f"Loading {INPUT_PATH}...")
    with open(INPUT_PATH, 'rb') as f:
        data = pickle.load(f)

    strats = data['strategies']
    n = len(strats)
    print(f"Nodes: {n}, Iterations: {data['iterations']}")

    # Allocate arrays
    keys = np.zeros(n, dtype=np.uint64)
    act_types = np.zeros(n, dtype=np.uint8)
    probs = np.zeros((n, MAX_ACTIONS), dtype=np.uint8)

    # Also build reverse lookup: hash -> index
    hash_to_idx = {}

    pruned = 0
    for i, (key, node) in enumerate(strats.items()):
        h = hash_key(key)
        keys[i] = h
        hash_to_idx[h] = i

        al = tuple(node['actions'])
        act_types[i] = ACTLIST_TO_ID.get(al, 255)

        strategy = node['strategy']
        num_actions = len(strategy)

        # Prune: if any prob < 0.001, set to 0 and renormalize
        cleaned = [p if p >= 0.001 else 0.0 for p in strategy]
        total = sum(cleaned)
        if total > 0:
            cleaned = [p / total for p in cleaned]
        else:
            cleaned = [1.0 / num_actions] * num_actions

        for j in range(num_actions):
            probs[i, j] = quantize_prob(cleaned[j])
            if strategy[j] >= 0.001 and cleaned[j] == 0:
                pruned += 1

        # Ensure quantized probs sum to ~255
        # Adjust the largest to absorb rounding error
        row_sum = sum(probs[i, :num_actions])
        if row_sum > 0 and row_sum != 255:
            max_idx = np.argmax(probs[i, :num_actions])
            diff = 255 - row_sum
            probs[i, max_idx] = max(0, min(255, probs[i, max_idx] + diff))

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    keys_path = os.path.join(OUTPUT_DIR, "strategy_keys.npy")
    act_path = os.path.join(OUTPUT_DIR, "strategy_acttype.npy")
    probs_path = os.path.join(OUTPUT_DIR, "strategy_probs.npy")
    meta_path = os.path.join(OUTPUT_DIR, "strategy_meta.pkl")

    np.save(keys_path, keys)
    np.save(act_path, act_types)
    np.save(probs_path, probs)

    meta = {
        'iterations': data['iterations'],
        'num_nodes': n,
        'action_lists': ACTION_LISTS,
        'max_actions': MAX_ACTIONS,
    }
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)

    # Report sizes
    total_compressed = 0
    for p in [keys_path, act_path, probs_path, meta_path]:
        sz = os.path.getsize(p)
        total_compressed += sz
        print(f"  {os.path.basename(p)}: {sz / 1024:.1f} KB")

    orig_size = os.path.getsize(INPUT_PATH)
    print(f"\nOriginal:   {orig_size / 1024 / 1024:.2f} MB")
    print(f"Compressed: {total_compressed / 1024 / 1024:.2f} MB")
    print(f"Ratio:      {total_compressed / orig_size * 100:.1f}%")
    print(f"Pruned entries: {pruned}")


if __name__ == "__main__":
    compress()

"""
Convert C++ binary strategy output to Python-compatible formats:
  1. strategy.pkl (pickle, for backward compat)
  2. Compressed numpy arrays (for fast loading in PlayerAgent)

Binary format from C++:
  Header: iterations(uint32), num_nodes(uint32)
  Per node: key(uint64), action_type(uint8), num_actions(uint8), avg_strategy[4](float64*4)
"""

import struct
import pickle
import numpy as np
import os
import sys
import hashlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ACTION_LISTS = [
    ("FOLD", "CALL", "JAM"),
    ("FOLD", "CALL"),
    ("FOLD", "CALL", "RAISE_SMALL", "RAISE_LARGE"),
    ("CHECK", "BET_SMALL", "BET_LARGE"),
    ("CHECK",),
]
MAX_ACTIONS = 4


def convert(bin_path, out_dir):
    with open(bin_path, 'rb') as f:
        data = f.read()

    offset = 0
    iterations, num_nodes = struct.unpack_from('<II', data, offset)
    offset += 8
    print(f"Binary: {iterations} iters, {num_nodes} nodes")

    # Parse nodes
    keys = np.zeros(num_nodes, dtype=np.uint64)
    act_types = np.zeros(num_nodes, dtype=np.uint8)
    probs = np.zeros((num_nodes, MAX_ACTIONS), dtype=np.uint8)

    node_size = 8 + 1 + 1 + MAX_ACTIONS * 8  # 42 bytes per node
    for i in range(num_nodes):
        key = struct.unpack_from('<Q', data, offset)[0]; offset += 8
        atype = struct.unpack_from('<B', data, offset)[0]; offset += 1
        nact = struct.unpack_from('<B', data, offset)[0]; offset += 1
        avg = struct.unpack_from(f'<{MAX_ACTIONS}d', data, offset); offset += MAX_ACTIONS * 8

        keys[i] = key
        act_types[i] = atype

        # Quantize and normalize
        raw = list(avg[:nact])
        total = sum(raw)
        if total > 0:
            raw = [p / total for p in raw]
        else:
            raw = [1.0 / nact] * nact

        for j in range(nact):
            probs[i, j] = max(0, min(255, int(round(raw[j] * 255))))

        # Fix rounding to sum to 255
        row_sum = sum(probs[i, :nact])
        if row_sum > 0 and row_sum != 255 and nact > 0:
            mx = np.argmax(probs[i, :nact])
            probs[i, mx] = max(0, min(255, probs[i, mx] + (255 - row_sum)))

    os.makedirs(out_dir, exist_ok=True)

    # Save compressed numpy
    np.save(os.path.join(out_dir, "strategy_keys.npy"), keys)
    np.save(os.path.join(out_dir, "strategy_acttype.npy"), act_types)
    np.save(os.path.join(out_dir, "strategy_probs.npy"), probs)

    meta = {
        'iterations': iterations,
        'num_nodes': num_nodes,
        'action_lists': ACTION_LISTS,
        'max_actions': MAX_ACTIONS,
    }
    with open(os.path.join(out_dir, "strategy_meta.pkl"), 'wb') as f:
        pickle.dump(meta, f)

    # Report sizes
    total_size = 0
    for fname in ["strategy_keys.npy", "strategy_acttype.npy", "strategy_probs.npy", "strategy_meta.pkl"]:
        p = os.path.join(out_dir, fname)
        sz = os.path.getsize(p)
        total_size += sz
        print(f"  {fname}: {sz / 1024:.1f} KB")
    print(f"  Total compressed: {total_size / 1024 / 1024:.2f} MB")
    print(f"  Original binary: {os.path.getsize(bin_path) / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_to_python.py <strategy.bin> [output_dir]")
        sys.exit(1)
    bin_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    convert(bin_path, out_dir)

"""
Merge two MCCFR checkpoints by summing regret_sum and strategy_sum.

CFR theory: regret and strategy sums are additive.
Two independent runs with N1 and N2 iterations → merge = N1+N2 effective iterations.

Usage:
    python training/cpp/merge_checkpoints.py checkpoint_a.bin checkpoint_b.bin merged.bin
"""

import struct
import sys
import os

def load_checkpoint(path):
    """Load checkpoint: returns (iterations, nodes_dict)
    nodes_dict: key -> (num_actions, action_type, regret_sum[4], strategy_sum[4])
    """
    with open(path, 'rb') as f:
        data = f.read()
    
    offset = 0
    iters, num_nodes = struct.unpack_from('<II', data, offset); offset += 8
    
    nodes = {}
    for i in range(num_nodes):
        key = struct.unpack_from('<Q', data, offset)[0]; offset += 8
        atype = struct.unpack_from('<B', data, offset)[0]; offset += 1
        nact = struct.unpack_from('<B', data, offset)[0]; offset += 1
        regret = list(struct.unpack_from('<4d', data, offset)); offset += 32
        strat = list(struct.unpack_from('<4d', data, offset)); offset += 32
        nodes[key] = {
            'nact': nact,
            'atype': atype,
            'regret': regret,
            'strat': strat,
        }
    
    return iters, nodes


def merge_nodes(nodes_a, nodes_b):
    """Merge two node dicts by summing regret and strategy sums."""
    merged = {}
    
    all_keys = set(nodes_a.keys()) | set(nodes_b.keys())
    
    for key in all_keys:
        a = nodes_a.get(key)
        b = nodes_b.get(key)
        
        if a and b:
            # Both have this node: sum regret and strategy
            merged[key] = {
                'nact': a['nact'],
                'atype': a['atype'],
                'regret': [max(a['regret'][i] + b['regret'][i], 0) for i in range(4)],
                'strat': [a['strat'][i] + b['strat'][i] for i in range(4)],
            }
        elif a:
            merged[key] = a
        else:
            merged[key] = b
    
    return merged


def save_checkpoint(path, iterations, nodes):
    """Save merged checkpoint."""
    with open(path, 'wb') as f:
        f.write(struct.pack('<II', iterations, len(nodes)))
        for key, node in nodes.items():
            f.write(struct.pack('<Q', key))
            f.write(struct.pack('<B', node['atype']))
            f.write(struct.pack('<B', node['nact']))
            f.write(struct.pack('<4d', *node['regret']))
            f.write(struct.pack('<4d', *node['strat']))


def save_strategy_bin(path, iterations, nodes):
    """Save as strategy binary (for convert_to_python.py)."""
    with open(path, 'wb') as f:
        f.write(struct.pack('<II', iterations, len(nodes)))
        for key, node in nodes.items():
            f.write(struct.pack('<Q', key))
            f.write(struct.pack('<B', node['atype']))
            f.write(struct.pack('<B', node['nact']))
            # Average strategy
            nact = node['nact']
            strat = node['strat'][:nact]
            total = sum(strat)
            if total > 0:
                avg = [s / total for s in strat] + [0.0] * (4 - nact)
            else:
                avg = [1.0 / nact] * nact + [0.0] * (4 - nact)
            f.write(struct.pack('<4d', *avg))
            # Confidence = sum of strategy_sum
            conf = sum(node['strat'][:nact])
            f.write(struct.pack('<d', conf))


def main():
    if len(sys.argv) < 4:
        print("Usage: python merge_checkpoints.py checkpoint_a.bin checkpoint_b.bin output.bin")
        print("  Merges two MCCFR checkpoints by summing regret/strategy sums.")
        print("  Output: merged checkpoint + strategy binary")
        sys.exit(1)
    
    path_a = sys.argv[1]
    path_b = sys.argv[2]
    output = sys.argv[3]
    
    print(f"Loading {path_a}...")
    iters_a, nodes_a = load_checkpoint(path_a)
    print(f"  → {iters_a:,} iters, {len(nodes_a):,} nodes")
    
    print(f"Loading {path_b}...")
    iters_b, nodes_b = load_checkpoint(path_b)
    print(f"  → {iters_b:,} iters, {len(nodes_b):,} nodes")
    
    print(f"Merging...")
    merged = merge_nodes(nodes_a, nodes_b)
    total_iters = iters_a + iters_b
    
    common = len(set(nodes_a.keys()) & set(nodes_b.keys()))
    only_a = len(nodes_a) - common
    only_b = len(nodes_b) - common
    
    print(f"  Common nodes: {common:,}")
    print(f"  Only in A: {only_a:,}")
    print(f"  Only in B: {only_b:,}")
    print(f"  Merged total: {len(merged):,} nodes, {total_iters:,} effective iters")
    
    # Save checkpoint
    ckpt_path = output
    save_checkpoint(ckpt_path, total_iters, merged)
    print(f"  Saved checkpoint: {ckpt_path}")
    
    # Save strategy binary
    strat_path = output.replace('checkpoint', 'strategy').replace('.bin', '_cpp.bin')
    if strat_path == output:
        strat_path = output.rsplit('.', 1)[0] + '_strategy.bin'
    save_strategy_bin(strat_path, total_iters, merged)
    print(f"  Saved strategy: {strat_path}")
    
    print(f"\nDone! Merged {iters_a:,} + {iters_b:,} = {total_iters:,} iters")


if __name__ == "__main__":
    main()

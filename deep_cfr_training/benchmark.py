"""
Deep CFR Benchmark Script
Tests training throughput across different devices and batch sizes.

Usage:
    python benchmark.py              # Quick benchmark
    python benchmark.py --full       # Full benchmark with all batch sizes
    python benchmark.py --traversal  # Benchmark traversal speed
"""

import argparse
import time
import numpy as np
import torch

from game_env import FEATURE_DIM, NUM_ACTIONS, GameState, batch_deal_discard, state_to_features
from deep_cfr import DeepCFR, DEVICE
import deep_cfr as dc_mod


def get_available_devices():
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.insert(0, 'cuda')
    elif torch.backends.mps.is_available():
        devices.insert(0, 'mps')
    return devices


def bench_training(device_str, batch_size, n_batches=20, n_samples=None):
    """Benchmark training throughput on given device."""
    dc_mod.DEVICE = torch.device(device_str)

    trainer = DeepCFR()
    trainer.iteration = 1
    trainer.total_iterations = 100

    features = np.random.randn(FEATURE_DIM).astype(np.float32)
    advantages = np.random.randn(NUM_ACTIONS).astype(np.float32)
    mask = np.ones(NUM_ACTIONS, dtype=np.float32)

    n = n_samples or max(batch_size * 4, 50000)
    for _ in range(n):
        trainer.adv_buffers[0].add(features, advantages, 1, mask)
    for _ in range(n // 2):
        trainer.adv_buffers[1].add(features, advantages, 1, mask)
        trainer.strategy_buffer.add(features, advantages * 0 + 1/NUM_ACTIONS, 1, mask)

    # Warmup
    trainer.train_networks(batch_size=batch_size, num_batches=3)
    trainer.train_strategy_net(batch_size=batch_size, num_batches=3)

    # Benchmark
    t0 = time.time()
    l = trainer.train_networks(batch_size=batch_size, num_batches=n_batches)
    trainer.train_strategy_net(batch_size=batch_size, num_batches=n_batches)
    elapsed = time.time() - t0

    total_samples = batch_size * n_batches * 3  # 2 adv nets + 1 strat net
    throughput = total_samples / elapsed / 1000  # K samples/sec
    ms_per_batch = elapsed / (n_batches * 3) * 1000

    return ms_per_batch, throughput


def bench_traversal(n_traversals=200):
    """Benchmark C++ traversal speed."""
    trainer = DeepCFR()
    trainer.iteration = 1
    trainer.total_iterations = 100

    # Warmup
    for p in range(2):
        trainer.run_traversals_batched(20, p)

    t0 = time.time()
    for p in range(2):
        trainer.run_traversals_batched(n_traversals, p)
    elapsed = time.time() - t0

    ms_per_trav = elapsed / n_traversals * 1000
    return ms_per_trav


def main():
    parser = argparse.ArgumentParser(description='Deep CFR Benchmark')
    parser.add_argument('--full', action='store_true', help='Full benchmark all batch sizes')
    parser.add_argument('--traversal', action='store_true', help='Benchmark traversal speed')
    parser.add_argument('--device', type=str, default=None, help='Force specific device')
    args = parser.parse_args()

    devices = [args.device] if args.device else get_available_devices()

    print("=" * 60)
    print("Deep CFR Benchmark")
    print("=" * 60)
    print(f"Available devices: {devices}")
    print(f"FEATURE_DIM: {FEATURE_DIM}, NUM_ACTIONS: {NUM_ACTIONS}")
    print()

    # ── Traversal Benchmark ──────────────────────────────────
    if args.traversal or not args.full:
        print("── Traversal Speed (C++ + batch inference) ──")
        n = 200
        ms = bench_traversal(n)
        print(f"  {n} traversals: {ms:.2f}ms/trav")
        print(f"  Estimated 500 × 1000: {ms * 500 * 1000 / 1000 / 3600:.2f}h")
        print()

    # ── Training Throughput ──────────────────────────────────
    if args.full:
        batch_sizes = [4096, 8192, 16384, 32768, 65536, 131072, 131072*2]
    else:
        batch_sizes = [8192, 32768, 65536]

    print("── Training Throughput (samples/sec) ──")
    header = f"{'batch_size':>10}"
    for dev in devices:
        header += f"  {dev:>8} ms/b  {dev:>8} K/s"
    print(header)
    print("-" * len(header))

    best = {}
    for bs in batch_sizes:
        row = f"{bs:>10}"
        for dev in devices:
            try:
                ms, ks = bench_training(dev, bs, n_batches=15)
                row += f"  {ms:>10.1f}  {ks:>8.0f}K"
                if dev not in best or ks > best[dev][1]:
                    best[dev] = (bs, ks)
            except Exception as e:
                row += f"  {'ERROR':>10}  {'---':>8}"
        print(row)

    print()
    print("── Optimal batch sizes ──")
    for dev, (bs, ks) in best.items():
        print(f"  {dev}: batch_size={bs:,} → {ks:.0f}K samples/sec")

    print()
    print("── Recommendation ──")
    if 'cuda' in best:
        bs = best['cuda'][0]
        ks = best['cuda'][1]
        print(f"  CUDA optimal: --batch-size {bs} ({ks:.0f}K samples/sec)")
    elif 'mps' in best:
        bs = best['mps'][0]
        ks = best['mps'][1]
        print(f"  MPS optimal:  --batch-size {bs} ({ks:.0f}K samples/sec)")
    if 'cpu' in best:
        bs = best['cpu'][0]
        ks = best['cpu'][1]
        print(f"  CPU optimal:  --batch-size {bs} ({ks:.0f}K samples/sec)")

    # Restore default device
    dc_mod.DEVICE = DEVICE


if __name__ == '__main__':
    main()

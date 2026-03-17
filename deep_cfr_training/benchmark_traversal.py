"""
Traversal bottleneck profiler.
Measures time spent in each phase of the training loop.

Usage: python benchmark_traversal.py
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from postflop_cfr import DeepCFR
from game import batch_deal_discard, GameState
from postflop_cfr.traversal import _preflop_key, _tabular_strategy, _PF_SLOTS, _PF_SLOT
from preflop_cfr.equity import warmup_ev

N = 200   # traversals to benchmark

def time_gamestate_ops():
    """Measure Python GameState overhead."""
    t0 = time.perf_counter()
    deals = batch_deal_discard(N)
    p0h, p1h, p0d, p1d, comms, p0h5, p1h5 = deals

    apply_calls = 0
    copy_calls  = 0
    t_apply = 0.
    t_copy  = 0.

    for i in range(N):
        state = GameState()
        # Simulate a preflop+postflop sequence
        for _ in range(3):   # a few preflop actions
            va = state.get_valid_actions()
            if not va or state.is_terminal: break
            ta = time.perf_counter()
            ns = state.apply(va[0])
            t_apply += time.perf_counter() - ta
            apply_calls += 1
            state = ns

    dt = time.perf_counter() - t0
    print(f"GameState ops ({N} games):")
    print(f"  Total:         {dt*1000:.1f} ms")
    print(f"  apply() calls: {apply_calls}")
    print(f"  Per apply:     {t_apply/max(apply_calls,1)*1e6:.2f} μs")


def time_warmup_ev():
    """Measure warmup equity computation per postflop decision."""
    deals = batch_deal_discard(N)
    p0h5 = deals[5]; p1h5 = deals[6]

    from postflop_cfr.traversal import traverse_coro
    trainer = type('T', (), {
        'preflop_regrets': {}, 'preflop_strategy_sum': {},
        'iteration': 1, 'warmup_iters': 999,
        'adv_nets': [type('N', (), {'__call__': lambda s,x: torch.zeros(x.shape[0],8)})() for _ in range(2)],
        'adv_buffers': [type('B', (), {'street_bufs': [[] for _ in range(4)], 'add': lambda *a: None})() for _ in range(2)],
        'strategy_buffer': type('B', (), {'add': lambda *a: None})(),
    })()

    t0 = time.perf_counter()
    ev_calls = 0
    for i in range(N):
        state = GameState()
        hand5_0 = list(p0h5[i]); hand5_1 = list(p1h5[i])
        # Advance to flop
        va = state.get_valid_actions()
        if va and not state.is_terminal:
            state = state.apply(va[-1])  # raise
            va2 = state.get_valid_actions()
            if va2 and not state.is_terminal:
                state = state.apply(va2[1] if len(va2) > 1 else va2[0])  # call
        if state.street > 0:
            t1 = time.perf_counter()
            ev = warmup_ev(hand5_0, hand5_1, state, 0)
            ev_calls += 1
    dt = time.perf_counter() - t0
    print(f"\nwarmup_ev ({ev_calls} calls):")
    print(f"  Total:    {dt*1000:.1f} ms")
    if ev_calls > 0:
        print(f"  Per call: {dt/ev_calls*1000:.2f} ms")


def time_feature_extraction():
    """Measure feature extraction per postflop node."""
    from game import state_to_features
    deals = batch_deal_discard(N)
    p0h, p1h, p0d, p1d, comms = deals[:5]

    t0 = time.perf_counter()
    for i in range(N):
        hand = list(p0h[i]); comm = list(comms[i][:3])
        state = GameState()
        state = state.apply(state.get_valid_actions()[-1])  # raise
        state = state.apply(state.get_valid_actions()[1])    # call → flop
        f = state_to_features(
            hand, comm, 10, 10, 1, False,
            list(p0d[i]), list(p1d[i]),
            street_bets=state.street_bets,
            history=state.history,
            num_actions_this_street=state.num_actions_this_street,
            street_last_ratios=state.street_last_ratios,
            street_bet_counts=state.street_bet_counts,
        )
    dt = time.perf_counter() - t0
    print(f"\nFeature extraction ({N} calls):")
    print(f"  Total:    {dt*1000:.1f} ms")
    print(f"  Per call: {dt/N*1e6:.1f} μs")


def time_batch_inference():
    """Measure GPU batch inference vs individual."""
    net = torch.nn.Sequential(
        torch.nn.Linear(119, 256), torch.nn.ReLU(),
        torch.nn.Linear(256, 256), torch.nn.ReLU(),
        torch.nn.Linear(256, 256), torch.nn.ReLU(),
        torch.nn.Linear(256, 8),
    )
    net.eval()
    feats = torch.randn(N, 119)

    # Batch
    t0 = time.perf_counter()
    for _ in range(10):
        with torch.no_grad():
            out = net(feats)
    dt = time.perf_counter() - t0
    print(f"\nBatch inference ({N} samples, 10 repeats):")
    print(f"  Total:     {dt*1000:.1f} ms")
    print(f"  Per sample: {dt/N/10*1e6:.2f} μs")

    # Individual
    t0 = time.perf_counter()
    for i in range(min(N, 50)):
        with torch.no_grad():
            out = net(feats[i:i+1])
    dt2 = time.perf_counter() - t0
    print(f"  Individual 50 calls: {dt2*1000:.1f} ms ({dt2/50*1e6:.1f} μs each)")
    print(f"  Batch speedup: {(dt2/50)/(dt/N/10):.1f}x")


def time_full_traversal():
    """End-to-end traversal timing for N games (warmup mode)."""
    from postflop_cfr.traversal import run_traversals_batched

    trainer = DeepCFR()
    trainer.warmup_iters = 9999  # stay in warmup
    trainer.iteration    = 1

    t0 = time.perf_counter()
    run_traversals_batched(trainer, N, 0)
    dt = time.perf_counter() - t0
    print(f"\nFull traversal warmup ({N} games, player 0):")
    print(f"  Total:   {dt*1000:.1f} ms")
    print(f"  Per game: {dt/N*1000:.2f} ms")

    # After warmup (neural net mode, but with random net)
    trainer.warmup_iters = 0
    trainer.iteration    = 1
    t0 = time.perf_counter()
    run_traversals_batched(trainer, N, 0)
    dt2 = time.perf_counter() - t0
    print(f"\nFull traversal neural ({N} games, player 0):")
    print(f"  Total:   {dt2*1000:.1f} ms")
    print(f"  Per game: {dt2/N*1000:.2f} ms")
    print(f"  Overhead vs warmup: {dt2/dt:.1f}x")


if __name__ == '__main__':
    print("=" * 55)
    print("Traversal Bottleneck Profile")
    print("=" * 55)
    time_gamestate_ops()
    time_warmup_ev()
    time_feature_extraction()
    time_batch_inference()
    time_full_traversal()
    print()
    print("Done.")

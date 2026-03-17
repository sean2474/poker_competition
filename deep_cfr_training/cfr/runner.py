"""
Main Deep CFR training loop, checkpointing, and model export.
"""

import os
import time
import torch
from tqdm import tqdm

from .traversal import run_traversals_batched
from .training  import train_adv_networks, train_strategy_nets


def run(trainer, num_iterations=500, traversals_per_iter=1000,
        train_interval=1, batch_size=2048, num_batches=100,
        checkpoint_interval=50, checkpoint_dir='model'):

    trainer.batch_size       = batch_size
    trainer.num_batches      = num_batches
    trainer.total_iterations = num_iterations
    os.makedirs(checkpoint_dir, exist_ok=True)

    ckpt_path  = os.path.join(checkpoint_dir, 'checkpoint_latest.pt')
    start_iter = load_checkpoint(trainer, ckpt_path)

    print(f"Deep CFR: {num_iterations} iters × {traversals_per_iter} traversals")
    print(f"Device: {trainer.device}  |  Preflop net + Postflop net (separate)")
    if start_iter > 0:
        print(f"Resuming from iter {start_iter}")
    print()

    t0     = time.time()
    losses = [0.0, 0.0]

    _bar_fmt = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    pbar  = tqdm(range(start_iter, num_iterations), desc='CFR iters',
                 initial=start_iter, total=num_iterations, position=0, leave=True)
    inner = tqdm(total=1, position=1, leave=False, bar_format=_bar_fmt)

    for t in pbar:
        trainer.iteration = t + 1

        for traversing in range(2):
            inner.reset(total=traversals_per_iter)
            inner.set_description(f'Trav P{traversing}')
            run_traversals_batched(trainer, traversals_per_iter, traversing)
            inner.refresh()

        if (t + 1) % train_interval == 0:
            inner.reset(total=num_batches * 4)   # 2 players × 2 nets
            inner.set_description('Training ')
            losses = train_adv_networks(trainer)
            inner.refresh()

        elapsed = time.time() - t0
        done    = t - start_iter + 1
        ips     = done / elapsed if elapsed > 0 else 0
        buf     = [len(b) for b in trainer.adv_buffers]
        pbar.set_postfix({
            'it/s': f'{ips:.1f}',
            'loss': f'{losses[0]:.3f}/{losses[1]:.3f}',
            'buf':  f'{buf[0]//1000}K/{buf[1]//1000}K',
        }, refresh=False)

        if (t + 1) % checkpoint_interval == 0:
            save_checkpoint(trainer, ckpt_path, t + 1)
            tagged = os.path.join(checkpoint_dir, f'checkpoint_{t+1:04d}.pt')
            save_checkpoint(trainer, tagged, t + 1)
            tqdm.write(f'  [ckpt] iter {t+1} → {tagged}')

    inner.close()

    tqdm.write("\nTraining strategy networks...")
    train_strategy_nets(trainer, num_batches=num_batches * 3)

    elapsed = time.time() - t0
    tqdm.write(f"\nDone: {num_iterations} iters in {elapsed:.0f}s  ({elapsed/num_iterations:.1f}s/iter)")
    tqdm.write(f"Adv buffers: {[len(b) for b in trainer.adv_buffers]}")
    tqdm.write(f"Strategy buffer: {len(trainer.strategy_buffer)}")


def save_checkpoint(trainer, path, iteration):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torch.save({
        'iteration':         iteration,
        'total_iterations':  trainer.total_iterations,
        'pf_adv_net_0':      trainer.pf_adv_nets[0].state_dict(),
        'pf_adv_net_1':      trainer.pf_adv_nets[1].state_dict(),
        'adv_net_0':         trainer.adv_nets[0].state_dict(),
        'adv_net_1':         trainer.adv_nets[1].state_dict(),
        'pf_strategy_net':   trainer.pf_strategy_net.state_dict(),
        'strategy_net':      trainer.strategy_net.state_dict(),
    }, path)


def load_checkpoint(trainer, path) -> int:
    if not os.path.exists(path):
        return 0
    ckpt = torch.load(path, map_location='cpu')
    trainer.pf_adv_nets[0].load_state_dict(ckpt['pf_adv_net_0'])
    trainer.pf_adv_nets[1].load_state_dict(ckpt['pf_adv_net_1'])
    trainer.adv_nets[0].load_state_dict(ckpt['adv_net_0'])
    trainer.adv_nets[1].load_state_dict(ckpt['adv_net_1'])
    trainer.pf_strategy_net.load_state_dict(ckpt['pf_strategy_net'])
    trainer.strategy_net.load_state_dict(ckpt['strategy_net'])
    trainer.total_iterations = ckpt.get('total_iterations', trainer.total_iterations)
    it = ckpt.get('iteration', 0)
    print(f"Resumed checkpoint: iter {it}/{trainer.total_iterations}")
    return it


def export(trainer, path_prefix):
    """Export strategy nets for submission inference."""
    os.makedirs(os.path.dirname(path_prefix) or '.', exist_ok=True)

    # Postflop strategy (most valuable — majority of decisions)
    torch.save(trainer.strategy_net.state_dict(),    path_prefix + '_strategy.pt')
    # Preflop strategy
    torch.save(trainer.pf_strategy_net.state_dict(), path_prefix + '_pf_strategy.pt')
    # Full checkpoint
    torch.save({
        'strategy_net':    trainer.strategy_net.state_dict(),
        'pf_strategy_net': trainer.pf_strategy_net.state_dict(),
        'adv_net_0':       trainer.adv_nets[0].state_dict(),
        'adv_net_1':       trainer.adv_nets[1].state_dict(),
        'iteration':       trainer.iteration,
    }, path_prefix + '_full.pt')

    n_pf = sum(p.numel() for p in trainer.pf_strategy_net.parameters())
    n_pf_sz = os.path.getsize(path_prefix + '_pf_strategy.pt') / 1024
    n_po = sum(p.numel() for p in trainer.strategy_net.parameters())
    n_po_sz = os.path.getsize(path_prefix + '_strategy.pt') / 1024
    print(f"Exported:")
    print(f"  Preflop  strategy: {n_pf:,} params, {n_pf_sz:.0f} KB  → {path_prefix}_pf_strategy.pt")
    print(f"  Postflop strategy: {n_po:,} params, {n_po_sz:.0f} KB  → {path_prefix}_strategy.pt")

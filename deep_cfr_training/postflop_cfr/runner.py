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

    # ── Graceful Ctrl+C: save checkpoint then exit ──────────────────────────
    import signal
    _interrupted = [False]

    def _sigint_handler(sig, frame):
        if _interrupted[0]:          # second Ctrl+C → force quit
            raise KeyboardInterrupt
        _interrupted[0] = True
        tqdm.write("\n[!] Ctrl+C received — saving checkpoint before exit...")

    signal.signal(signal.SIGINT, _sigint_handler)
    # ────────────────────────────────────────────────────────────────────────

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
            save_checkpoint(trainer, ckpt_path, t + 1, save_buffers=True)
            tagged = os.path.join(checkpoint_dir, f'checkpoint_{t+1:04d}.pt')
            save_checkpoint(trainer, tagged, t + 1, save_buffers=False)
            tqdm.write(f'  [ckpt] iter {t+1} → {tagged}')

        if _interrupted[0]:
            save_checkpoint(trainer, ckpt_path, t + 1, save_buffers=True)
            tqdm.write(f'  [ckpt] interrupted at iter {t+1} → {ckpt_path}  (buffers saved)')
            inner.close()
            return

    inner.close()

    tqdm.write("\nTraining strategy networks...")
    train_strategy_nets(trainer, num_batches=num_batches * 3)

    elapsed = time.time() - t0
    tqdm.write(f"\nDone: {num_iterations} iters in {elapsed:.0f}s  ({elapsed/num_iterations:.1f}s/iter)")
    tqdm.write(f"Adv buffers: {[len(b) for b in trainer.adv_buffers]}")
    tqdm.write(f"Strategy buffer: {len(trainer.strategy_buffer)}")


def _save_buffer(buf, path):
    """Save ReservoirBuffer NumPy arrays to a .npz file."""
    import numpy as np
    arrays = {}
    for s in range(4):
        n = buf._size[s]
        if n == 0 or buf._feats[s] is None:
            continue
        arrays[f's{s}_feats']  = buf._feats[s][:n]
        arrays[f's{s}_values'] = buf._values[s][:n]
        arrays[f's{s}_iters']  = buf._iters[s][:n]
        arrays[f's{s}_masks']  = buf._masks[s][:n]
        arrays[f's{s}_meta']   = np.array([n, buf.street_counts[s]], dtype=np.int64)
    arrays['count'] = np.array([buf.count], dtype=np.int64)
    np.savez(path, **arrays)


def _load_buffer(buf, path):
    """Restore ReservoirBuffer from a .npz file."""
    import numpy as np
    if not os.path.exists(path + '.npz'):
        return
    data = np.load(path + '.npz')
    buf.count = int(data['count'][0])
    for s in range(4):
        key = f's{s}_feats'
        if key not in data:
            continue
        f = data[f's{s}_feats']
        n = f.shape[0]
        buf._init(s)
        buf._feats[s][:n]  = f
        buf._values[s][:n] = data[f's{s}_values']
        buf._iters[s][:n]  = data[f's{s}_iters']
        buf._masks[s][:n]  = data[f's{s}_masks']
        meta = data[f's{s}_meta']
        buf._size[s]          = int(meta[0])
        buf.street_counts[s]  = int(meta[1])


def save_checkpoint(trainer, path, iteration, save_buffers=False):
    import pickle
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torch.save({
        'iteration':        iteration,
        'total_iterations': trainer.total_iterations,
        'adv_net_0':        trainer.adv_nets[0].state_dict(),
        'adv_net_1':        trainer.adv_nets[1].state_dict(),
        'strategy_net':     trainer.strategy_net.state_dict(),
    }, path)
    # Preflop tables saved alongside (pickle — dicts can be huge)
    pf_path = path + '.preflop.pkl'
    with open(pf_path, 'wb') as f:
        pickle.dump({
            'regrets':      trainer.preflop_regrets,
            'strategy_sum': trainer.preflop_strategy_sum,
        }, f)
    # Buffers: only for latest checkpoint (tagged checkpoints skip to save disk)
    if save_buffers:
        _save_buffer(trainer.adv_buffers[0],  path + '.buf_adv0')
        _save_buffer(trainer.adv_buffers[1],  path + '.buf_adv1')
        _save_buffer(trainer.strategy_buffer, path + '.buf_str')


def load_checkpoint(trainer, path) -> int:
    import pickle
    if not os.path.exists(path):
        return 0
    ckpt = torch.load(path, map_location='cpu')
    trainer.adv_nets[0].load_state_dict(ckpt['adv_net_0'])
    trainer.adv_nets[1].load_state_dict(ckpt['adv_net_1'])
    trainer.strategy_net.load_state_dict(ckpt['strategy_net'])
    trainer.total_iterations = ckpt.get('total_iterations', trainer.total_iterations)
    # Load preflop tables if available
    pf_path = path + '.preflop.pkl'
    if os.path.exists(pf_path):
        with open(pf_path, 'rb') as f:
            pf = pickle.load(f)
        trainer.preflop_regrets      = pf.get('regrets', {})
        trainer.preflop_strategy_sum = pf.get('strategy_sum', {})
    # Restore buffers if saved
    _load_buffer(trainer.adv_buffers[0],  path + '.buf_adv0')
    _load_buffer(trainer.adv_buffers[1],  path + '.buf_adv1')
    _load_buffer(trainer.strategy_buffer, path + '.buf_str')
    it = ckpt.get('iteration', 0)
    buf_sizes = [len(trainer.adv_buffers[0]), len(trainer.adv_buffers[1]),
                 len(trainer.strategy_buffer)]
    print(f"Resumed checkpoint: iter {it}/{trainer.total_iterations}  "
          f"(preflop: {len(trainer.preflop_regrets):,} infosets, "
          f"bufs: {buf_sizes[0]//1000}K/{buf_sizes[1]//1000}K/{buf_sizes[2]//1000}K)")
    return it


def export(trainer, path_prefix):
    """Export strategy nets + preflop chart for submission inference."""
    import pickle, numpy as np
    os.makedirs(os.path.dirname(path_prefix) or '.', exist_ok=True)

    # Postflop strategy net
    torch.save(trainer.strategy_net.state_dict(), path_prefix + '_strategy.pt')

    # Preflop chart: normalize strategy_sum → probabilities
    preflop_chart = {}
    for key, s_sum in trainer.preflop_strategy_sum.items():
        total = s_sum.sum()
        preflop_chart[key] = s_sum / total if total > 0 else s_sum.copy()

    pf_chart_path = path_prefix + '_preflop_chart.pkl'
    with open(pf_chart_path, 'wb') as f:
        pickle.dump(preflop_chart, f)

    # Full checkpoint
    torch.save({
        'strategy_net': trainer.strategy_net.state_dict(),
        'adv_net_0':    trainer.adv_nets[0].state_dict(),
        'adv_net_1':    trainer.adv_nets[1].state_dict(),
        'iteration':    trainer.iteration,
    }, path_prefix + '_full.pt')

    n_po    = sum(p.numel() for p in trainer.strategy_net.parameters())
    n_po_sz = os.path.getsize(path_prefix + '_strategy.pt') / 1024
    n_pf_keys = len(preflop_chart)
    pf_sz   = os.path.getsize(pf_chart_path) / 1024
    print(f"Exported:")
    print(f"  Postflop strategy: {n_po:,} params, {n_po_sz:.0f} KB  → {path_prefix}_strategy.pt")
    print(f"  Preflop chart:     {n_pf_keys:,} infosets, {pf_sz:.0f} KB  → {pf_chart_path}")

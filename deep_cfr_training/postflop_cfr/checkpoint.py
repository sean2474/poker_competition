"""Checkpoint save / load / export for DeepCFR trainer."""

import os
import pickle
import numpy as np
import torch


# ── Buffer serialization ──────────────────────────────────────────────────────

def _save_buffer(buf, path):
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
    if not os.path.exists(path + '.npz'):
        return
    data = np.load(path + '.npz')
    buf.count = int(data['count'][0])
    for s in range(4):
        if f's{s}_feats' not in data:
            continue
        f = data[f's{s}_feats']
        n = f.shape[0]
        buf._init(s)
        buf._feats[s][:n]  = f
        buf._values[s][:n] = data[f's{s}_values']
        buf._iters[s][:n]  = data[f's{s}_iters']
        buf._masks[s][:n]  = data[f's{s}_masks']
        meta = data[f's{s}_meta']
        buf._size[s]         = int(meta[0])
        buf.street_counts[s] = int(meta[1])


# ── Public API ────────────────────────────────────────────────────────────────

def save_checkpoint(trainer, path, iteration, save_buffers=False):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    ckpt = {
        'iteration':        iteration,
        'total_iterations': trainer.total_iterations,
        'adv_net_0':        trainer.adv_nets[0].state_dict(),
        'adv_net_1':        trainer.adv_nets[1].state_dict(),
        'strategy_net':     trainer.strategy_net.state_dict(),
    }
    dt = getattr(trainer, 'discard_trainer', None)
    if dt is not None:
        ckpt['discard_net']       = dt.net.state_dict()
        ckpt['discard_iteration'] = dt.iteration
        ckpt['discard_hidden']    = dt.hidden_dim
    torch.save(ckpt, path)
    with open(path + '.preflop.pkl', 'wb') as f:
        pickle.dump({'regrets': trainer.preflop_regrets,
                     'strategy_sum': trainer.preflop_strategy_sum}, f)
    if save_buffers:
        _save_buffer(trainer.adv_buffers[0],  path + '.buf_adv0')
        _save_buffer(trainer.adv_buffers[1],  path + '.buf_adv1')
        _save_buffer(trainer.strategy_buffer, path + '.buf_str')


def load_checkpoint(trainer, path) -> int:
    if not os.path.exists(path):
        return 0
    try:
        ckpt = torch.load(path, map_location='cpu')
    except (EOFError, RuntimeError, pickle.UnpicklingError) as e:
        print(f'[ckpt] Corrupted checkpoint {path} ({e}), starting fresh.')
        return 0
    trainer.adv_nets[0].load_state_dict(ckpt['adv_net_0'])
    trainer.adv_nets[1].load_state_dict(ckpt['adv_net_1'])
    trainer.strategy_net.load_state_dict(ckpt['strategy_net'])
    dt = getattr(trainer, 'discard_trainer', None)
    if dt is not None and 'discard_net' in ckpt:
        from discard_cfr.model import make_net
        dt.hidden_dim = ckpt.get('discard_hidden', dt.hidden_dim)
        dt.net = make_net(dt.hidden_dim).to(dt.device)
        dt.net.load_state_dict(ckpt['discard_net'])
        dt.iteration = ckpt.get('discard_iteration', 0)
        dt.net.eval()
    trainer.total_iterations = ckpt.get('total_iterations', trainer.total_iterations)
    pf_path = path + '.preflop.pkl'
    if os.path.exists(pf_path):
        try:
            with open(pf_path, 'rb') as f:
                pf = pickle.load(f)
            trainer.preflop_regrets      = pf.get('regrets', {})
            trainer.preflop_strategy_sum = pf.get('strategy_sum', {})
        except (EOFError, pickle.UnpicklingError) as e:
            print(f'[ckpt] Corrupted preflop pkl ({e}), resetting preflop.')
    device = trainer.device
    for net in trainer.adv_nets:
        net.to(device)
    trainer.strategy_net.to(device)
    _load_buffer(trainer.adv_buffers[0],  path + '.buf_adv0')
    _load_buffer(trainer.adv_buffers[1],  path + '.buf_adv1')
    _load_buffer(trainer.strategy_buffer, path + '.buf_str')
    it = ckpt.get('iteration', 0)
    bufs = [len(trainer.adv_buffers[0]), len(trainer.adv_buffers[1]),
            len(trainer.strategy_buffer)]
    print(f"Resumed iter {it}/{trainer.total_iterations}  "
          f"pf:{len(trainer.preflop_regrets):,} infosets  "
          f"bufs:{bufs[0]//1000}K/{bufs[1]//1000}K/{bufs[2]//1000}K")
    return it


def export(trainer, path_prefix):
    """Export nets + preflop chart for submission."""
    os.makedirs(os.path.dirname(path_prefix) or '.', exist_ok=True)
    torch.save(trainer.strategy_net.state_dict(), path_prefix + '_strategy.pt')

    preflop_chart = {k: (v / v.sum() if v.sum() > 0 else v.copy())
                     for k, v in trainer.preflop_strategy_sum.items()}
    pf_path = path_prefix + '_preflop_chart.pkl'
    with open(pf_path, 'wb') as f:
        pickle.dump(preflop_chart, f)

    full = {
        'strategy_net': trainer.strategy_net.state_dict(),
        'adv_net_0':    trainer.adv_nets[0].state_dict(),
        'adv_net_1':    trainer.adv_nets[1].state_dict(),
        'iteration':    trainer.iteration,
    }
    dt = getattr(trainer, 'discard_trainer', None)
    if dt is not None:
        full['discard_net']       = dt.net.state_dict()
        full['discard_iteration'] = dt.iteration
        full['discard_hidden']    = dt.hidden_dim
    torch.save(full, path_prefix + '_full.pt')

    n_params = sum(p.numel() for p in trainer.strategy_net.parameters())
    print(f"Exported:")
    print(f"  strategy: {n_params:,} params → {path_prefix}_strategy.pt")
    print(f"  preflop:  {len(preflop_chart):,} infosets → {pf_path}")
    print(f"  full:     → {path_prefix}_full.pt")

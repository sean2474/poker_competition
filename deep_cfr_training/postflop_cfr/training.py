"""
Network training routines.

train_adv_networks: trains preflop and postflop advantage nets separately.
train_strategy_nets: trains preflop and postflop strategy nets.

Weighting: 2t/T (linear CFR weighting — recent iterations trusted more).
"""

import torch
import torch.optim as optim

from models import (
    PostflopAdvantageNet,
    PostflopStrategyNet,
)


def _to_device(arr, device, dtype=torch.float32):
    """Zero-copy numpy→tensor with async GPU transfer."""
    import numpy as _np
    t = torch.from_numpy(_np.ascontiguousarray(arr)).to(dtype)
    if device.type == 'cpu':
        return t
    if device.type == 'cuda':
        return t.pin_memory().to(device, non_blocking=True)
    return t.to(device)


def train_adv_networks(trainer) -> list:
    """
    Train postflop advantage nets FROM SCRATCH (paper Section 5.2).
    Preflop is handled by tabular CFR in traversal — no net needed.
    Returns [loss_p0, loss_p1].
    """
    device  = trainer.device
    bs      = trainer.batch_size
    n_batch = trainer.num_batches
    losses  = [0.0, 0.0]

    for p in range(2):
        buf = trainer.adv_buffers[p]
        postflop_has_data = any(buf.street_bufs[s] for s in [1, 2, 3])
        if postflop_has_data:
            trainer.adv_nets[p] = PostflopAdvantageNet().to(device)
            opt = optim.Adam(trainer.adv_nets[p].parameters(), lr=trainer.lr)
            loss = _train_adv_net(trainer.adv_nets[p], opt, buf, [1, 2, 3],
                                  bs, n_batch, trainer.total_iterations, device)
            losses[p] = loss  # keep on GPU for traversal inference

    return losses


_GPU_LOAD_CAP = 1_048_576  # 1M — balances coverage (33% of 3M buf) vs randperm speed
_PROFILE      = False      # flip True to print per-phase ms once for debugging


def _sync(device):
    if device.type == 'cuda': torch.cuda.synchronize()
    elif device.type == 'mps':
        try: torch.mps.synchronize()
        except Exception: pass


def _make_scaler(device):
    use_amp = (device.type == 'cuda')
    scaler  = torch.amp.GradScaler('cuda') if use_amp else None
    return use_amp, scaler


def _train_adv_net(net, opt, buf, streets, batch_size, num_batches, total_iters, device):
    import torch.optim.lr_scheduler as sched
    import time as _t

    t0 = _t.perf_counter()
    n_load = min(max(batch_size, len(buf)), _GPU_LOAD_CAP)
    data = buf.sample_streets(streets, n_load)
    if data is None:
        return 0.0
    features, values, iterations, masks = data
    weights = (2.0 * iterations / max(total_iters, 1)).astype('float32')
    t1 = _t.perf_counter()

    x_all = _to_device(features, device)
    y_all = _to_device(values,   device)
    w_all = _to_device(weights,  device)
    m_all = _to_device(masks,    device)
    _sync(device)
    t2 = _t.perf_counter()

    N  = x_all.shape[0]
    bs = min(batch_size, N)
    use_amp, scaler = _make_scaler(device)
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    scheduler    = sched.CosineAnnealingLR(opt, T_max=num_batches, eta_min=1e-5)
    running_loss = torch.zeros(1, device=device)  # no per-batch CPU sync
    batch_count  = 0

    # Pre-epoch shuffle: one randperm per epoch instead of per batch
    while batch_count < num_batches:
        perm = torch.randperm(N, device=device)
        for start in range(0, N - bs + 1, bs):
            if batch_count >= num_batches:
                break
            idx = perm[start : start + bs]
            x = x_all[idx]; y = y_all[idx]; w = w_all[idx]; m = m_all[idx]

            with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                pred = net(x)
                loss = ((pred - y) ** 2 * w.unsqueeze(1) * m).sum() / (m.sum() + 1e-8)

            opt.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                opt.step()

            scheduler.step()
            running_loss.add_(loss.detach())
            batch_count += 1

    _sync(device)
    t3 = _t.perf_counter()
    del x_all, y_all, w_all, m_all

    if _PROFILE:
        print(f"    [adv] sample={1000*(t1-t0):.0f}ms  xfer={1000*(t2-t1):.0f}ms"
              f"  train={1000*(t3-t2):.0f}ms  N={N}  bs={bs}")
    return running_loss.item() / num_batches


def train_strategy_nets(trainer, num_batches: int = None):
    """Train postflop strategy net. Preflop = tabular chart (no net needed)."""
    device  = trainer.device
    bs      = trainer.batch_size
    n_batch = num_batches or trainer.num_batches
    buf     = trainer.strategy_buffer

    postflop_has_data = any(buf.street_bufs[s] for s in [1, 2, 3])
    if postflop_has_data:
        trainer.strategy_net = PostflopStrategyNet().to(device)
        opt = optim.Adam(trainer.strategy_net.parameters(), lr=trainer.lr)
        _train_strategy_net(trainer.strategy_net, opt, buf, [1, 2, 3],
                            bs, n_batch, trainer.total_iterations, device)


def _train_strategy_net(net, opt, buf, streets, batch_size, num_batches, total_iters, device):
    import torch.optim.lr_scheduler as sched
    import time as _t

    t0 = _t.perf_counter()
    n_load = min(max(batch_size, len(buf)), _GPU_LOAD_CAP)
    data = buf.sample_streets(streets, n_load)
    if data is None:
        return 0.0
    features, strategies, iterations, masks = data
    weights = (2.0 * iterations / max(total_iters, 1)).astype('float32')
    t1 = _t.perf_counter()

    x_all = _to_device(features,   device)
    y_all = _to_device(strategies, device)
    w_all = _to_device(weights,    device)
    m_all = _to_device(masks,      device)
    _sync(device)
    t2 = _t.perf_counter()

    N  = x_all.shape[0]
    bs = min(batch_size, N)
    use_amp, scaler = _make_scaler(device)
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    scheduler    = sched.CosineAnnealingLR(opt, T_max=num_batches, eta_min=1e-5)
    running_loss = torch.zeros(1, device=device)
    batch_count  = 0

    while batch_count < num_batches:
        perm = torch.randperm(N, device=device)
        for start in range(0, N - bs + 1, bs):
            if batch_count >= num_batches:
                break
            idx = perm[start : start + bs]
            x = x_all[idx]; y = y_all[idx]; w = w_all[idx]; m = m_all[idx]

            with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                logits    = net(x)
                log_probs = torch.log_softmax(logits, dim=1)
                loss      = -(y * log_probs * m * w.unsqueeze(1)).sum() / (m.sum() + 1e-8)

            opt.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                opt.step()

            scheduler.step()
            running_loss.add_(loss.detach())
            batch_count += 1

    _sync(device)
    t3 = _t.perf_counter()
    del x_all, y_all, w_all, m_all

    if _PROFILE:
        print(f"    [str] sample={1000*(t1-t0):.0f}ms  xfer={1000*(t2-t1):.0f}ms"
              f"  train={1000*(t3-t2):.0f}ms  N={N}  bs={bs}")
    return running_loss.item() / num_batches

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
            trainer.adv_nets[p] = trainer.adv_nets[p].cpu()
            losses[p] = loss

    return losses


def _train_adv_net(net, opt, buf, streets, batch_size, num_batches, total_iters, device):
    import torch.optim.lr_scheduler as sched
    # Load entire buffer to GPU once — eliminates per-batch CPU sampling overhead
    n_load = max(batch_size * num_batches, len(buf))
    data = buf.sample_streets(streets, n_load)
    if data is None:
        return 0.0
    features, values, iterations, masks = data
    weights = (2.0 * iterations / max(total_iters, 1)).astype('float32')

    import numpy as _np
    x_all = _to_device(features, device)
    y_all = _to_device(values,   device)
    w_all = _to_device(weights,  device)
    m_all = _to_device(masks,    device)
    N = x_all.shape[0]

    scheduler = sched.CosineAnnealingLR(opt, T_max=num_batches, eta_min=1e-5)
    total_loss = 0.0
    bs = min(batch_size, N)
    for _ in range(num_batches):
        idx = torch.randperm(N, device=device)[:bs]
        x = x_all[idx]; y = y_all[idx]; w = w_all[idx]; m = m_all[idx]
        pred  = net(x)
        loss  = ((pred - y) ** 2 * w.unsqueeze(1) * m).sum() / (m.sum() + 1e-8)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        opt.step(); scheduler.step()
        total_loss += loss.item()

    del x_all, y_all, w_all, m_all
    return total_loss / num_batches


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
        trainer.strategy_net = trainer.strategy_net.cpu()


def _train_strategy_net(net, opt, buf, streets, batch_size, num_batches, total_iters, device):
    import torch.optim.lr_scheduler as sched
    # Load entire buffer to GPU once
    n_load = max(batch_size * num_batches, len(buf))
    data = buf.sample_streets(streets, n_load)
    if data is None:
        return 0.0
    features, strategies, iterations, masks = data
    weights = (2.0 * iterations / max(total_iters, 1)).astype('float32')

    x_all = _to_device(features,   device)
    y_all = _to_device(strategies, device)
    w_all = _to_device(weights,    device)
    m_all = _to_device(masks,      device)
    N = x_all.shape[0]

    scheduler = sched.CosineAnnealingLR(opt, T_max=num_batches, eta_min=1e-5)
    total_loss = 0.0
    bs = min(batch_size, N)
    for b in range(num_batches):
        idx = torch.randperm(N, device=device)[:bs]
        x = x_all[idx]; y = y_all[idx]; w = w_all[idx]; m = m_all[idx]
        logits    = net(x)
        log_probs = torch.log_softmax(logits, dim=1)
        loss      = -(y * log_probs * m * w.unsqueeze(1)).sum() / (m.sum() + 1e-8)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        opt.step(); scheduler.step()
        total_loss += loss.item()

    del x_all, y_all, w_all, m_all
    return total_loss / num_batches

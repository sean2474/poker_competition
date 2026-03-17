"""
Network training routines.

train_adv_networks: trains preflop and postflop advantage nets separately.
train_strategy_nets: trains preflop and postflop strategy nets.

Weighting: 2t/T (linear CFR weighting — recent iterations trusted more).
"""

import torch
import torch.optim as optim

from models import (
    PreflopAdvantageNet, PostflopAdvantageNet,
    PreflopStrategyNet,  PostflopStrategyNet,
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
    Train preflop and postflop advantage nets FROM SCRATCH (paper Section 5.2).
    Returns [loss_p0, loss_p1].
    """
    device  = trainer.device
    bs      = trainer.batch_size
    n_batch = trainer.num_batches
    losses  = [0.0, 0.0]

    for p in range(2):
        buf = trainer.adv_buffers[p]

        # ── Preflop net (street 0) ──────────────────────────────────────────
        if buf.street_bufs[0]:
            trainer.pf_adv_nets[p] = PreflopAdvantageNet().to(device)
            opt = optim.Adam(trainer.pf_adv_nets[p].parameters(), lr=trainer.lr)
            _train_adv_net(trainer.pf_adv_nets[p], opt, buf, [0],
                           bs, n_batch, trainer.total_iterations, device)
            trainer.pf_adv_nets[p] = trainer.pf_adv_nets[p].cpu()

        # ── Postflop net (streets 1-3) ──────────────────────────────────────
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
    total_loss = 0.0
    for _ in range(num_batches):
        data = buf.sample_streets(streets, batch_size)
        if data is None:
            continue
        features, values, iterations, masks = data
        weights = 2.0 * iterations / max(total_iters, 1)

        x = _to_device(features, device)
        y = _to_device(values,   device)
        w = _to_device(weights,  device)
        m = _to_device(masks,    device)

        pred  = net(x)
        loss  = ((pred - y) ** 2 * w.unsqueeze(1) * m).sum() / (m.sum() + 1e-8)
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item()

    return total_loss / num_batches


def train_strategy_nets(trainer, num_batches: int = None):
    """Train preflop and postflop strategy nets from strategy buffer."""
    device  = trainer.device
    bs      = trainer.batch_size
    n_batch = num_batches or trainer.num_batches * 3
    buf     = trainer.strategy_buffer

    if buf.street_bufs[0]:
        trainer.pf_strategy_net = PreflopStrategyNet().to(device)
        opt = optim.Adam(trainer.pf_strategy_net.parameters(), lr=trainer.lr)
        _train_strategy_net(trainer.pf_strategy_net, opt, buf, [0],
                            bs, n_batch, trainer.total_iterations, device)
        trainer.pf_strategy_net = trainer.pf_strategy_net.cpu()

    postflop_has_data = any(buf.street_bufs[s] for s in [1, 2, 3])
    if postflop_has_data:
        trainer.strategy_net = PostflopStrategyNet().to(device)
        opt = optim.Adam(trainer.strategy_net.parameters(), lr=trainer.lr)
        _train_strategy_net(trainer.strategy_net, opt, buf, [1, 2, 3],
                            bs, n_batch, trainer.total_iterations, device)
        trainer.strategy_net = trainer.strategy_net.cpu()


def _train_strategy_net(net, opt, buf, streets, batch_size, num_batches, total_iters, device):
    total_loss = 0.0
    for b in range(num_batches):
        data = buf.sample_streets(streets, batch_size)
        if data is None:
            continue
        features, strategies, iterations, masks = data
        x = _to_device(features,   device)
        y = _to_device(strategies, device)
        w = _to_device(2.0 * iterations / max(total_iters, 1), device)
        m = _to_device(masks,      device)

        logits    = net(x)
        log_probs = torch.log_softmax(logits, dim=1)
        loss      = -(y * log_probs * m * w.unsqueeze(1)).sum() / (m.sum() + 1e-8)
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item()

        if (b + 1) % 100 == 0:
            print(f"  strategy [{streets}] batch {b+1}/{num_batches} loss={total_loss/(b+1):.4f}")

    return total_loss / num_batches

"""
Deep CFR Training Speed Benchmark
Measures each stage separately to identify bottlenecks.

Usage:
    python benchmark.py              # Full benchmark
    python benchmark.py --quick      # Quick (fewer reps)
    python benchmark.py --traversal  # Include traversal benchmark

Output includes system info + per-stage ms + projected training time.
Compare M3 Max vs RTX 3090 by running on both machines.
"""

import argparse
import os
import sys
import time
import platform

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FEAT_DIM   = 119
ACT_DIM    = 8
ITERS      = 1500
TRAVERSALS = 2000
N_NETS     = 3          # adv_p0, adv_p1, strategy
N_BATCHES  = 150
BS         = 131_072    # 128K


# ── System Info ──────────────────────────────────────────────────────────────

def system_info():
    lines = []
    lines.append(f"  OS        : {platform.system()} {platform.release()}")
    lines.append(f"  Python    : {platform.python_version()}")
    lines.append(f"  PyTorch   : {torch.__version__}")

    # CPU
    try:
        import subprocess
        if platform.system() == "Darwin":
            cpu = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        else:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        cpu = line.split(":")[1].strip(); break
                else:
                    cpu = "unknown"
    except Exception:
        cpu = platform.processor()
    cores = os.cpu_count()
    lines.append(f"  CPU       : {cpu} ({cores} cores)")

    # RAM
    try:
        import psutil
        ram = psutil.virtual_memory().total / 1024**3
        lines.append(f"  RAM       : {ram:.0f} GB")
    except ImportError:
        pass

    # GPU
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        lines.append(f"  GPU       : {name} ({vram:.0f} GB VRAM)  [CUDA]")
    elif torch.backends.mps.is_available():
        lines.append(f"  GPU       : Apple Silicon MPS")
    else:
        lines.append(f"  GPU       : CPU only")

    return "\n".join(lines)


# ── Device helpers ───────────────────────────────────────────────────────────

def best_device():
    if torch.cuda.is_available():    return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def sync(device):
    if device.type == "cuda": torch.cuda.synchronize()
    elif device.type == "mps":
        try: torch.mps.synchronize()
        except Exception: pass


def to_dev(arr, device):
    t = torch.from_numpy(np.ascontiguousarray(arr)).float()
    if device.type == "cuda":
        return t.pin_memory().to(device, non_blocking=True)
    return t.to(device)


# ── Minimal network matching production architecture ─────────────────────────

def make_net(device):
    class _R(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.fc1=nn.Linear(d,d); self.ln1=nn.LayerNorm(d)
            self.fc2=nn.Linear(d,d); self.ln2=nn.LayerNorm(d)
        def forward(self, x):
            h = torch.relu(self.ln1(self.fc1(x)))
            return torch.relu(self.ln2(self.fc2(h)) + x)
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Sequential(nn.Linear(FEAT_DIM, 256), nn.ReLU())
            self.res   = nn.Sequential(_R(256), _R(256))
            self.head  = nn.Linear(256, ACT_DIM)
        def forward(self, x): return self.head(self.res(self.embed(x)))
    return Net().to(device)


# ── Buffer (matches production ReservoirBuffer) ───────────────────────────────

def make_buffer(n_per_street=400_000):
    """Pre-filled NumPy buffer matching production layout."""
    from models.buffers import ReservoirBuffer
    buf = ReservoirBuffer(4_000_000)
    for s in [1, 2, 3]:
        buf.add_batch(
            np.random.randn(n_per_street, FEAT_DIM).astype(np.float32),
            np.random.randn(n_per_street, ACT_DIM).astype(np.float32),
            np.ones(n_per_street, np.float32),
            np.ones((n_per_street, ACT_DIM), np.float32),
            np.full(n_per_street, s, dtype=np.int32),
        )
    return buf


# ── Stage benchmarks ─────────────────────────────────────────────────────────

def bench_sampling(buf, reps=50):
    t0 = time.perf_counter()
    for _ in range(reps):
        buf.sample_streets([1, 2, 3], BS)
    return (time.perf_counter() - t0) / reps * 1000   # ms


def bench_transfer(buf, device, reps=30):
    f, v, it, m = buf.sample_streets([1, 2, 3], BS)
    sync(device)
    t0 = time.perf_counter()
    for _ in range(reps):
        xd = to_dev(f, device)
        yd = to_dev(v, device)
        wd = to_dev(it, device)
        md = to_dev(m, device)
        sync(device)
    return (time.perf_counter() - t0) / reps * 1000


def bench_forward(net, device, reps=30):
    x = torch.randn(BS, FEAT_DIM, device=device)
    sync(device)
    with torch.no_grad():
        for _ in range(3): net(x); sync(device)   # warmup
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(reps): net(x); sync(device)
    return (time.perf_counter() - t0) / reps * 1000


def bench_train_step(net, opt, device, reps=20):
    x = torch.randn(BS, FEAT_DIM, device=device)
    y = torch.randn(BS, ACT_DIM,  device=device)
    sync(device)
    for _ in range(3):                              # warmup
        pred = net(x); loss = ((pred-y)**2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sync(device)
    t0 = time.perf_counter()
    for _ in range(reps):
        pred = net(x); loss = ((pred-y)**2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sync(device)
    return (time.perf_counter() - t0) / reps * 1000


def bench_full_train(buf, device, reps=5):
    """End-to-end: sample → transfer → forward → backward → step (GPU preload style)."""
    net = make_net(device)
    opt = optim.Adam(net.parameters(), lr=3e-4)

    # Warmup
    f,v,it,m = buf.sample_streets([1,2,3], BS)
    x=to_dev(f,device); y=to_dev(v,device)
    for _ in range(3):
        pred=net(x); loss=((pred-y)**2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sync(device)

    t0 = time.perf_counter()
    for _ in range(reps):
        # --- GPU preload (production code path) ---
        f,v,it,m = buf.sample_streets([1,2,3], len(buf))
        x_all = to_dev(f,device); y_all = to_dev(v,device)
        w_all = to_dev(it,device); m_all = to_dev(m,device)
        N = x_all.shape[0]
        for b in range(N_BATCHES):
            idx = torch.randperm(N, device=device)[:min(BS,N)]
            x=x_all[idx]; y=y_all[idx]
            pred=net(x); loss=((pred-y)**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        del x_all, y_all, w_all, m_all
        sync(device)
    elapsed = (time.perf_counter() - t0) / reps

    return elapsed  # seconds per training step (1 network)


def bench_traversal(n=500):
    """C++ traversal + Python overhead per game."""
    try:
        from game import batch_deal_discard
    except ImportError:
        return None
    t0 = time.perf_counter()
    batch_deal_discard(n)
    return (time.perf_counter() - t0) / n * 1000   # ms per game


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick",     action="store_true", help="Fewer repetitions")
    parser.add_argument("--traversal", action="store_true", help="Include traversal bench")
    args = parser.parse_args()

    reps = 10 if args.quick else 30
    device = best_device()

    W = 62
    print("=" * W)
    print("  Deep CFR Benchmark")
    print("=" * W)
    print(system_info())
    print(f"  Device    : {device}")
    print("=" * W)
    print()

    # Fill buffer
    print("Filling buffer (400K × 3 streets)...", end=" ", flush=True)
    buf = make_buffer(400_000)
    print(f"{len(buf):,} samples")

    net = make_net(device)
    opt = optim.Adam(net.parameters(), lr=3e-4)
    params = sum(p.numel() for p in net.parameters())
    print(f"Network: {params:,} params  (256-dim, 2 ResBlocks)\n")

    # ── Stage 1: Sampling ────────────────────────────────────
    print("Benchmarking stages...")
    s_ms = bench_sampling(buf, reps=max(reps, 20))
    print(f"  [1] Buffer sampling (128K from 1.2M):  {s_ms:6.1f} ms")

    # ── Stage 2: CPU→GPU Transfer ────────────────────────────
    t_ms = bench_transfer(buf, device, reps=max(reps//2, 10))
    print(f"  [2] CPU → {device.type.upper()} transfer (128K×119):  {t_ms:6.1f} ms")

    # ── Stage 3: Forward pass ─────────────────────────────────
    f_ms = bench_forward(net, device, reps=reps)
    print(f"  [3] Forward pass (BS={BS//1024}K):          {f_ms:6.1f} ms")

    # ── Stage 4: Full train step (fwd+bwd+opt) ────────────────
    tr_ms = bench_train_step(net, opt, device, reps=reps)
    print(f"  [4] fwd+bwd+optimizer step:            {tr_ms:6.1f} ms")

    # ── Stage 5: Traversal ────────────────────────────────────
    trav_ms = None
    if args.traversal:
        trav_ms = bench_traversal(500)
        if trav_ms is not None:
            print(f"  [5] C++ traversal:                     {trav_ms:6.2f} ms/game")

    print()

    # ── Projection ───────────────────────────────────────────
    print("=" * W)
    print("  Per-iteration breakdown (2000 traversals, 150 batches × 3 nets)")
    print("=" * W)

    # GPU preload: sampling done ONCE per net (not 150×)
    sample_s   = (s_ms * N_NETS) / 1000          # 3× sampling per iter
    transfer_s = (t_ms * N_NETS) / 1000          # 3× bulk transfer
    train_s    = (tr_ms * N_BATCHES * N_NETS) / 1000  # 450 gradient steps
    trav_s     = (trav_ms * TRAVERSALS * 2 / 1000) if trav_ms else 2.0

    total_s = trav_s + sample_s + transfer_s + train_s

    print(f"  Traversal  ({TRAVERSALS}×2 games)     : {trav_s:6.1f} s")
    print(f"  Sampling   (3× full buffer load):  {sample_s:6.2f} s")
    print(f"  Transfer   (3× bulk to {device.type.upper()}):       {transfer_s:6.2f} s")
    print(f"  GPU train  ({N_BATCHES}×{N_NETS} steps):          {train_s:6.1f} s")
    print(f"  ─────────────────────────────────────────")
    print(f"  Total per iter                  : {total_s:6.1f} s")
    print()

    total_h = total_s * ITERS / 3600
    print(f"  Projected {ITERS} iterations:")
    print(f"    {total_s * ITERS:,.0f} s  =  {total_h:.1f} h  =  {total_h/24:.1f} days")
    print()

    if total_h < 6:
        verdict = "✓ Excellent — completes in < 6 hours"
    elif total_h < 12:
        verdict = "✓ Good — completes within 12 hours"
    elif total_h < 24:
        verdict = "△ Acceptable — completes within 1 day"
    else:
        verdict = "✗ Slow — consider reducing iters or batches"
    print(f"  {verdict}")
    print("=" * W)


if __name__ == "__main__":
    main()

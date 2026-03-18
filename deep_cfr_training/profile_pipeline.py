"""
Real-pipeline bottleneck profiler — runs actual training iterations with
fine-grained timing instrumentation.

Unlike benchmark_phases.py which uses isolated warm mocks, this script runs
the exact same code path as trainer.py so timing matches production.

Usage:
    python profile_pipeline.py --iterations 30 --traversals 1000

Output: per-iteration CSV with all component timings + trend analysis.
"""

import sys, os, time, csv, argparse, threading
os.environ.setdefault('OMP_NUM_THREADS', '96')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

try:
    import psutil as _psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


class _UtilMonitor:
    """Background thread that samples GPU + CPU utilization at 0.5s intervals."""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self._gpu   = []   # utilization %
        self._cpu   = []   # percent across all cores
        self._stop  = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._has_cuda = torch.cuda.is_available()
        self._has_psutil = _HAS_PSUTIL

    def start(self):
        self._gpu.clear(); self._cpu.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)

    def _loop(self):
        while not self._stop.is_set():
            if self._has_cuda:
                try:
                    self._gpu.append(torch.cuda.utilization())
                except Exception:
                    pass
            if self._has_psutil:
                self._cpu.append(_psutil.cpu_percent(interval=None))
            self._stop.wait(self.interval)

    def stats(self):
        """Returns (gpu_avg, gpu_peak, cpu_avg, cpu_peak) or None for missing."""
        gpu_avg  = float(np.mean(self._gpu))  if self._gpu  else 0.
        gpu_peak = float(np.max(self._gpu))   if self._gpu  else 0.
        cpu_avg  = float(np.mean(self._cpu))  if self._cpu  else 0.
        cpu_peak = float(np.max(self._cpu))   if self._cpu  else 0.
        return gpu_avg, gpu_peak, cpu_avg, cpu_peak

from deep_cfr import DeepCFR
from training_phase.runner import PhaseRunner
from postflop_cfr.checkpoint import load_checkpoint, save_checkpoint
from game.features import batch_deal_discard


# ── Timing hooks injected into the pipeline ───────────────────────────────────

class PipelineProfiler:
    """Wraps the actual training loop and records per-iteration timings."""

    def __init__(self, n_iter: int = 30, traversals: int = 1000,
                 discard_n_games: int = 50, batch_size: int = 65536,
                 num_batches: int = 50, force_phase: int = 1,
                 n_trav_threads: int = 1):
        self.trainer = DeepCFR()
        self.records = []        # list of dicts, one per iteration
        self.n_iter  = n_iter
        self.traversals = traversals
        self.discard_n_games = discard_n_games
        self.batch_size  = batch_size
        self.num_batches = num_batches
        self.force_phase = force_phase
        self.n_trav_threads = n_trav_threads

    def run(self):
        import threading
        from postflop_cfr.traversal import run_traversals_batched
        from postflop_cfr.training  import train_adv_networks, train_strategy_nets
        from postflop_cfr.buffers   import ReservoirBuffer
        from postflop_cfr.traversal import _postflop_ready
        from discard_cfr.cfr        import DiscardCFR
        from game.features          import batch_deal_discard
        from training_phase         import Phase1, Phase2, Phase3, PhaseStats

        trainer = self.trainer
        dc      = DiscardCFR()

        # If force_phase > 1, skip directly to that phase
        # (bypasses phase transition checks — for profiling Phase 2/3 directly)
        phase_idx     = max(0, self.force_phase - 1)
        phases        = [Phase1(), Phase2(), Phase3()]
        discard_model = dc if self.force_phase >= 2 else None
        pf_history    = []
        print(f"  Starting at Phase {phase_idx + 1}")
        if self.force_phase >= 2:
            print(f"  Force-seeding preflop regrets to skip Phase 1 check...")
            trainer.preflop_strategy_sum['FORCE'] = np.ones(3) * 9999.

        print(f"{'Iter':>4}  {'Phase':>5}  {'trav0':>7} {'trav1':>7}"
              f"  {'disc_mc':>7}  {'adv_tr':>7}  {'disc_tr':>6}"
              f"  {'merge':>6}  {'total':>7}  {'buf_min':>8}  {'is_ready':>8}")
        print("-" * 95)

        trainer = self.trainer
        for it in range(1, self.n_iter + 1):
            trainer.iteration = it  # enables linear strategy sum weighting
            t_iter_start = time.time()
            rec = {'iter': it, 'phase': phase_idx + 1}

            # ── Phase transition check ───────────────────────────────────────
            pf_ss = trainer.preflop_strategy_sum
            pf_v  = sum(float(s.sum()) for s in pf_ss.values()) if pf_ss else 0.
            buf_min = min(
                len(trainer.adv_buffers[p].street_bufs[s])
                for p in range(2) for s in [1, 2, 3]
            )
            stats = PhaseStats(
                iteration=it, pf_infosets=len(pf_ss), pf_visits=pf_v,
                postflop_buf_min=buf_min,
                discard_loss_history=pf_history,
            )
            if phase_idx < 2 and phases[phase_idx].is_complete(stats):
                phase_idx += 1
                if phase_idx == 1:
                    discard_model = dc
                    print(f"  [→ Phase 2] iter={it}")
                elif phase_idx == 2:
                    print(f"  [→ Phase 3] iter={it}")

            dt = discard_model if phase_idx == 2 else None

            # ── Traversal (n_trav_threads per player) ─────────────────────
            n_t      = self.n_trav_threads
            per_th   = max(1, self.traversals // n_t)
            tot_th   = n_t * 2
            buf_cap  = trainer.adv_buffers[0].capacity
            tmp_adv  = [[ReservoirBuffer(buf_cap), ReservoirBuffer(buf_cap)]
                        for _ in range(tot_th)]
            tmp_str  = [ReservoirBuffer(buf_cap) for _ in range(tot_th)]

            # ── Phase 3: monkey-patch sub-component timers ────────────────
            _p3_timers = {'recompute': [0.], 'run_iter': [0.],
                          'range_feat': [0.], 'pure_cpp': [0.]}
            if phase_idx == 2:
                from postflop_cfr import traversal as _t
                _orig_recompute = _t._recompute_discards_with_cfr
                _orig_apply     = None
                import postflop_cfr.range_tracker as _rt
                _orig_apply = _rt.apply_range_features
                def _timed_apply(feats, infos, gstates):
                    t = time.time(); r = _orig_apply(feats, infos, gstates)
                    _p3_timers['range_feat'][0] += time.time()-t; return r
                _rt.apply_range_features = _timed_apply

                def _timed_recompute(p0h5, p1h5, comms, dt):
                    t = time.time(); r = _orig_recompute(p0h5, p1h5, comms, dt)
                    _p3_timers['recompute'][0] += time.time()-t; return r
                _t._recompute_discards_with_cfr = _timed_recompute

                _orig_run_iter = dc.run_iter
                def _timed_ri(h5A, h5B, b):
                    t = time.time(); _orig_run_iter(h5A, h5B, b)
                    _p3_timers['run_iter'][0] += time.time()-t
                dc.run_iter = _timed_ri

            def _run_tp(thread_idx, tp):
                run_traversals_batched(
                    trainer, per_th, tp,
                    discard_trainer=dt, discard_n_games=self.discard_n_games,
                    phase=phase_idx + 1,
                    adv_bufs=tmp_adv[thread_idx], str_buf=tmp_str[thread_idx],
                )

            _mon = _UtilMonitor(interval=0.5)
            _mon.start()
            t0 = time.time()
            threads = []
            for tp in range(2):
                for j in range(n_t):
                    threads.append(threading.Thread(target=_run_tp, args=(tp*n_t+j, tp)))
            for th in threads: th.start()
            for th in threads: th.join()
            t_trav = time.time() - t0
            _mon.stop()
            gpu_avg, gpu_peak, cpu_avg, cpu_peak = _mon.stats()

            t0 = time.time()
            for idx in range(tot_th):
                for p in range(2): trainer.adv_buffers[p].merge_from(tmp_adv[idx][p])
                trainer.strategy_buffer.merge_from(tmp_str[idx])
            t_merge = time.time() - t0

            # ── Restore Phase 3 monkey-patches ────────────────────────────────
            if phase_idx == 2:
                from postflop_cfr import traversal as _t
                _t._recompute_discards_with_cfr = _orig_recompute
                dc.run_iter = _orig_run_iter
                if _orig_apply is not None:
                    import postflop_cfr.range_tracker as _rt
                    _rt.apply_range_features = _orig_apply

            # Record per-player timing separately
            rec['t_trav_parallel'] = t_trav
            rec['t_merge']         = t_merge
            rec['p3_recompute']    = _p3_timers['recompute'][0]
            rec['p3_run_iter']     = _p3_timers['run_iter'][0]
            rec['p3_range_feat']   = _p3_timers['range_feat'][0]
            rec['p3_pure_cpp']     = max(0., t_trav - _p3_timers['recompute'][0]
                                         - _p3_timers['run_iter'][0]
                                         - _p3_timers['range_feat'][0])

            # ── Discard MC (Phase 2 standalone) ─────────────────────────────
            t_disc_mc = 0.
            if phase_idx == 1:
                r = batch_deal_discard(self.discard_n_games)
                t0 = time.time()
                dc.run_iter(r[5], r[6], r[4])
                t_disc_mc = time.time() - t0

            # ── Postflop net training ────────────────────────────────────────
            t0 = time.time()
            losses = train_adv_networks(trainer)
            t_adv_train = time.time() - t0

            # ── Discard net training ─────────────────────────────────────────
            t_disc_tr = 0.
            if phase_idx >= 1:
                if len(dc.buf) >= dc.batch_size:
                    t0 = time.time()
                    dloss = dc.train()
                    t_disc_tr = time.time() - t0
                    pf_history.append(float(dloss))
                else:
                    dloss = 0.
            else:
                dloss = 0.

            # ── Record ───────────────────────────────────────────────────────
            t_total = time.time() - t_iter_start
            buf_min_new = min(
                len(trainer.adv_buffers[p].street_bufs[s])
                for p in range(2) for s in [1, 2, 3]
            )
            rec.update({
                't_disc_mc':   t_disc_mc,
                't_adv_train': t_adv_train,
                't_disc_tr':   t_disc_tr,
                't_total':     t_total,
                'buf_min':     buf_min_new,
                'is_ready':    _postflop_ready(trainer),
                'dloss':       dloss,
                'pf_infosets': len(pf_ss),
                'gpu_avg':     gpu_avg,
                'gpu_peak':    gpu_peak,
                'cpu_avg':     cpu_avg,
                'cpu_peak':    cpu_peak,
            })
            self.records.append(rec)

            # Print live row
            print(f"{it:>4}  {'P'+str(phase_idx+1):>5}"
                  f"  {t_trav/2:>7.2f} {t_trav/2:>7.2f}"
                  f"  {t_disc_mc:>7.3f}"
                  f"  {t_adv_train:>7.3f}"
                  f"  {t_disc_tr:>6.3f}"
                  f"  {t_merge:>6.3f}"
                  f"  {t_total:>7.2f}"
                  f"  {buf_min_new:>8d}"
                  f"  {str(_postflop_ready(trainer)):>8}")
            print(f"       GPU: avg={gpu_avg:4.0f}%  peak={gpu_peak:4.0f}%"
                  f"   CPU: avg={cpu_avg:5.1f}%  peak={cpu_peak:5.1f}%")
            if phase_idx == 2 and any(v[0] > 0.001 for v in _p3_timers.values()):
                rc = _p3_timers['recompute'][0]; ri = _p3_timers['run_iter'][0]
                rf = _p3_timers['range_feat'][0]
                pc = max(0., t_trav - rc - ri - rf)
                print(f"       P3-breakdown:  recompute={rc:.2f}s  run_iter={ri:.2f}s"
                      f"  range_feat={rf:.2f}s  C++/GPU={pc:.2f}s")

        self._summarise()

    def _summarise(self):
        if not self.records:
            return

        print("\n" + "=" * 95)
        print("SUMMARY BY PHASE")
        print("=" * 95)

        from collections import defaultdict
        phase_rec = defaultdict(list)
        for r in self.records:
            phase_rec[r['phase']].append(r)

        cols = ['t_trav_parallel', 't_disc_mc', 't_adv_train', 't_disc_tr', 't_total']
        labels = {
            't_trav_parallel': 'Traversal(‖)',
            't_disc_mc':       'Discard MC',
            't_adv_train':     'Net Train',
            't_disc_tr':       'Disc Train',
            't_total':         'TOTAL',
        }

        for ph, recs in sorted(phase_rec.items()):
            print(f"\n  Phase {ph}  ({len(recs)} iterations)")
            print(f"  {'Component':>18}  {'mean':>8}  {'min':>8}  {'max':>8}  {'%total':>7}")
            totals = [r['t_total'] for r in recs]
            avg_total = np.mean(totals)
            for c in cols:
                vals = [r[c] for r in recs]
                mu = np.mean(vals); mn = np.min(vals); mx = np.max(vals)
                pct = mu / avg_total * 100 if avg_total > 0 else 0
                print(f"  {labels[c]:>18}  {mu:>8.3f}  {mn:>8.3f}  {mx:>8.3f}  {pct:>6.1f}%")

        # Trend: how does t_total change over iterations?
        print(f"\n  Iteration trend (t_total):")
        n = len(self.records)
        q1 = self.records[:n//3]
        q2 = self.records[n//3:2*n//3]
        q3 = self.records[2*n//3:]
        for label, group in [("early", q1), ("mid", q2), ("late", q3)]:
            if group:
                avg = np.mean([r['t_total'] for r in group])
                print(f"    {label:>6}: {avg:.2f}s/iter")

        # Projection
        last10 = self.records[-10:] if len(self.records) >= 10 else self.records
        avg_last = np.mean([r['t_total'] for r in last10])
        print(f"\n  Steady-state iter time (last {len(last10)}): {avg_last:.2f}s")
        print(f"  Projected 1500 iters: {avg_last*1500:.0f}s = {avg_last*1500/3600:.1f}h")

        # GPU/CPU utilization summary
        all_gpu_avg  = [r.get('gpu_avg',  0) for r in self.records]
        all_gpu_peak = [r.get('gpu_peak', 0) for r in self.records]
        all_cpu_avg  = [r.get('cpu_avg',  0) for r in self.records]
        all_cpu_peak = [r.get('cpu_peak', 0) for r in self.records]
        if any(v > 0 for v in all_gpu_avg) or any(v > 0 for v in all_cpu_avg):
            print(f"\n  Utilization (during traversal):")
            print(f"    GPU  avg={np.mean(all_gpu_avg):5.1f}%  peak={np.max(all_gpu_peak):5.1f}%")
            print(f"    CPU  avg={np.mean(all_cpu_avg):5.1f}%  peak={np.max(all_cpu_peak):5.1f}%")

        # Save CSV
        csv_path = os.path.join(os.path.dirname(__file__), 'profile_results.csv')
        with open(csv_path, 'w', newline='') as f:
            if self.records:
                writer = csv.DictWriter(f, fieldnames=self.records[0].keys())
                writer.writeheader()
                writer.writerows(self.records)
        print(f"\n  Results saved to: {csv_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Real-pipeline bottleneck profiler')
    parser.add_argument('--iterations',  type=int, default=30)
    parser.add_argument('--traversals',  type=int, default=1000)
    parser.add_argument('--disc-games',  type=int, default=50)
    parser.add_argument('--batch-size',  type=int, default=65536)
    parser.add_argument('--num-batches', type=int, default=50)
    parser.add_argument('--force-phase',    type=int, default=1,
                        help='Start at Phase N directly (skip earlier phases)')
    parser.add_argument('--n-trav-threads', type=int, default=1,
                        help='parallel traversal threads per player (matches runner.py n_trav_threads)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    total_threads = args.n_trav_threads * 2
    per_thread    = max(1, args.traversals // args.n_trav_threads)
    print(f"\nReal-pipeline profiler: {args.iterations} iters × {args.traversals} traversals")
    print(f"Device: {device}  |  threads: {total_threads} ({args.n_trav_threads}/player)  |  {per_thread} games/thread\n")

    prof = PipelineProfiler(
        n_iter=args.iterations,
        traversals=args.traversals,
        discard_n_games=args.disc_games,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        force_phase=args.force_phase,
        n_trav_threads=args.n_trav_threads,
    )
    prof.run()

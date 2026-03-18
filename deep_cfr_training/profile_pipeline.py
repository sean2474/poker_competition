"""
Real-pipeline bottleneck profiler — runs actual training iterations with
fine-grained timing instrumentation.

Unlike benchmark_phases.py which uses isolated warm mocks, this script runs
the exact same code path as trainer.py so timing matches production.

Usage:
    python profile_pipeline.py --iterations 30 --traversals 1000

Output: per-iteration CSV with all component timings + trend analysis.
"""

import sys, os, time, csv, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

from deep_cfr import DeepCFR
from training_phase.runner import PhaseRunner
from postflop_cfr.checkpoint import load_checkpoint, save_checkpoint
from game.features import batch_deal_discard


# ── Timing hooks injected into the pipeline ───────────────────────────────────

class PipelineProfiler:
    """Wraps the actual training loop and records per-iteration timings."""

    def __init__(self, n_iter: int = 30, traversals: int = 1000,
                 discard_n_games: int = 50, batch_size: int = 65536,
                 num_batches: int = 50):
        self.trainer = DeepCFR()
        self.records = []        # list of dicts, one per iteration
        self.n_iter  = n_iter
        self.traversals = traversals
        self.discard_n_games = discard_n_games
        self.batch_size  = batch_size
        self.num_batches = num_batches

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

        phase_idx  = 0
        phases     = [Phase1(), Phase2(), Phase3()]
        discard_model = None    # Phase1: heuristic
        pf_history = []

        print(f"{'Iter':>4}  {'Phase':>5}  {'trav0':>7} {'trav1':>7}"
              f"  {'disc_mc':>7}  {'adv_tr':>7}  {'disc_tr':>6}"
              f"  {'merge':>6}  {'total':>7}  {'buf_min':>8}  {'is_ready':>8}")
        print("-" * 95)

        for it in range(1, self.n_iter + 1):
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

            # ── Traversal (parallel: player 0 + 1) ──────────────────────────
            buf_cap = trainer.adv_buffers[0].capacity
            tmp_adv = [[ReservoirBuffer(buf_cap), ReservoirBuffer(buf_cap)]
                       for _ in range(2)]
            tmp_str = [ReservoirBuffer(buf_cap) for _ in range(2)]

            def _run_tp(tp):
                run_traversals_batched(
                    trainer, self.traversals, tp,
                    discard_trainer=dt, discard_n_games=self.discard_n_games,
                    phase=phase_idx + 1,
                    adv_bufs=tmp_adv[tp], str_buf=tmp_str[tp],
                )

            t0 = time.time()
            threads = [threading.Thread(target=_run_tp, args=(tp,)) for tp in range(2)]
            for th in threads: th.start()
            for th in threads: th.join()
            t_trav = time.time() - t0

            t0 = time.time()
            for tp in range(2):
                for p in range(2): trainer.adv_buffers[p].merge_from(tmp_adv[tp][p])
                trainer.strategy_buffer.merge_from(tmp_str[tp])
            t_merge = time.time() - t0

            # Record per-player timing separately (Phase 2 runs sequentially)
            rec['t_trav_parallel'] = t_trav
            rec['t_merge']         = t_merge

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
                t0 = time.time()
                dloss = dc.train()
                t_disc_tr = time.time() - t0
                pf_history.append(float(dloss))
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
            })
            self.records.append(rec)

            # Print live row
            print(f"{it:>4}  {'P'+str(phase_idx+1):>5}"
                  f"  {t_trav/2:>7.2f} {t_trav/2:>7.2f}"   # approx per-player
                  f"  {t_disc_mc:>7.3f}"
                  f"  {t_adv_train:>7.3f}"
                  f"  {t_disc_tr:>6.3f}"
                  f"  {t_merge:>6.3f}"
                  f"  {t_total:>7.2f}"
                  f"  {buf_min_new:>8d}"
                  f"  {str(_postflop_ready(trainer)):>8}")

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
    args = parser.parse_args()

    print(f"\nReal-pipeline profiler: {args.iterations} iters × {args.traversals} traversals")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'}\n")

    prof = PipelineProfiler(
        n_iter=args.iterations,
        traversals=args.traversals,
        discard_n_games=args.disc_games,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
    )
    prof.run()

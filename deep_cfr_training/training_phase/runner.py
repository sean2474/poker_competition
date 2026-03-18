"""
PhaseRunner — orchestrates the 3-phase Deep CFR training pipeline.

Responsibilities:
  1. Run one iteration (deal → traverse → train) using currently active models
  2. Collect PhaseStats after each iteration
  3. Ask the current Phase if it's complete
  4. When complete, swap in the next Phase's models and advance

Model assignments per phase:
  Phase 1: preflop=PreflopCFR,  discard=DiscardHeuristic,  postflop=PostflopHeuristic
  Phase 2: preflop=PreflopCFR,  discard=DiscardCFR (training), postflop=PostflopHeuristic
  Phase 3: preflop=PreflopCFR,  discard=DiscardCFR (joint),    postflop=PostflopCFR

All training logic stays inside the CFR modules.
"""

import os
import time
import signal
from tqdm import tqdm

from .phase  import PhaseStats
from .phase1 import Phase1
from .phase2 import Phase2
from .phase3 import Phase3
from postflop_cfr.checkpoint import save_checkpoint, load_checkpoint
from postflop_cfr.cfr        import PostflopCFR
from discard_cfr.cfr         import DiscardCFR as DiscardCFRModel
from heuristic.discard import DiscardHeuristic


class PhaseRunner:
    """
    Top-level training orchestrator.

    Usage:
        runner = PhaseRunner(trainer_state, ...)
        runner.run()
    """

    def __init__(self,
                 trainer_state,
                 num_iterations:      int = 500,
                 traversals_per_iter: int = 1000,
                 train_interval:      int = 1,
                 batch_size:          int = 2048,
                 num_batches:         int = 100,
                 checkpoint_interval: int = 50,
                 checkpoint_dir:      str = 'model',
                 discard_n_games:     int = 50,
                 n_trav_threads:      int = 1):

        self.state               = trainer_state
        self.num_iterations      = num_iterations
        self.traversals_per_iter = traversals_per_iter
        self.train_interval      = train_interval
        self.batch_size          = batch_size
        self.num_batches         = num_batches
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir      = checkpoint_dir
        self.discard_n_games     = discard_n_games
        self.n_trav_threads      = n_trav_threads

        # ── Model instances ───────────────────────────────────────────────
        self.preflop_cfr  = None   # always PreflopCFR (bound after init)
        self.discard_model: DiscardHeuristic | DiscardCFRModel = DiscardHeuristic()
        self.postflop_cfr = PostflopCFR(trainer_state)

        # ── Phase state ───────────────────────────────────────────────────
        self._phases      = [Phase1(), Phase2(), Phase3()]
        self._phase_idx   = 0
        self._phase2_iters = 0   # Phase 2 local iteration counter

        # ── Loss history for phase stats ──────────────────────────────────
        self._discard_loss_history: list = []

    @property
    def _current_phase(self):
        return self._phases[self._phase_idx]

    def _collect_stats(self, iteration: int, discard_loss: float) -> PhaseStats:
        ss = self.state.preflop_strategy_sum
        pf_visits = sum(float(s.sum()) for s in ss.values()) if ss else 0.0
        buf_min = min(
            len(self.state.adv_buffers[p].street_bufs[s])
            for p in range(2) for s in [1, 2, 3]
        )
        if discard_loss > 0:
            self._discard_loss_history.append(discard_loss)
        return PhaseStats(
            iteration            = iteration,
            pf_infosets          = len(ss),
            pf_visits            = pf_visits,
            discard_loss         = discard_loss,
            discard_loss_history = list(self._discard_loss_history),
            postflop_buf_min     = buf_min,
        )

    def _advance_phase(self):
        if self._phase_idx >= len(self._phases) - 1:
            return   # already at last phase
        self._phase_idx += 1
        tqdm.write(f'\n[phase → {self._current_phase.name}]')

        # Swap discard model when entering Phase 2 or 3
        if self._phase_idx == 1:
            tqdm.write('  Swapping discard: Heuristic → DiscardCFR (standalone training)')
            self.discard_model = self.state.discard_trainer
        elif self._phase_idx == 2:
            tqdm.write('  Entering joint training (discard + postflop)')
            # discard_model already DiscardCFR from phase 2

    def run(self):
        self.state.batch_size       = self.batch_size
        self.state.num_batches      = self.num_batches
        self.state.total_iterations = self.num_iterations
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        ckpt_path  = os.path.join(self.checkpoint_dir, 'checkpoint_latest.pt')
        start_iter = load_checkpoint(self.state, ckpt_path)

        print(f'Training: {self.num_iterations} iters × {self.traversals_per_iter} traversals')
        print(f'Device: {self.state.device}')
        if start_iter > 0:
            print(f'Resuming from iter {start_iter}')
        print()

        # ── Graceful interrupt ─────────────────────────────────────────────
        _interrupted = [False]
        def _sigint(sig, frame):
            if _interrupted[0]: raise KeyboardInterrupt
            _interrupted[0] = True
            tqdm.write('\n[!] Ctrl+C — saving...')
        signal.signal(signal.SIGINT, _sigint)

        t0     = time.time()
        losses = [0.0, 0.0]
        discard_loss = 0.0
        phase2_local = 0   # iteration counter within Phase 2

        # ── Phase timing ───────────────────────────────────────────────────
        _phase_t0    = time.time()
        _phase_iters = 0

        _bar = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        pbar  = tqdm(range(start_iter, self.num_iterations), desc='CFR',
                     initial=start_iter, total=self.num_iterations,
                     position=0, leave=True)
        inner = tqdm(total=1, position=1, leave=False, bar_format=_bar)

        import threading
        from postflop_cfr.buffers import ReservoirBuffer
        from game.features import batch_deal_discard

        for t in pbar:
            self.state.iteration = t + 1

            # ── Phase transition check ─────────────────────────────────────
            stats = self._collect_stats(
                iteration    = phase2_local if self._phase_idx == 1 else t + 1,
                discard_loss = discard_loss,
            )
            if self._current_phase.is_complete(stats):
                _phase_dt = time.time() - _phase_t0
                _s_per_it = _phase_dt / max(_phase_iters, 1)
                tqdm.write(f'\n  {self._current_phase.summary(stats)}')
                tqdm.write(
                    f'  [timing] {self._current_phase.name}: '
                    f'{_phase_iters} iters  {_phase_dt:.1f}s total  '
                    f'{_s_per_it:.2f}s/iter'
                )
                self._advance_phase()
                _phase_t0    = time.time()
                _phase_iters = 0
            _phase_iters += 1

            # ── Determine active models ────────────────────────────────────
            # Phase 1: no discard training
            # Phase 2: discard trains standalone, postflop = warmup
            # Phase 3: discard trains jointly with postflop
            in_phase2 = (self._phase_idx == 1)
            in_phase3 = (self._phase_idx == 2)
            # Always None: use heuristic discard during DFS traversal.
            # Discard CFR trains separately via run_iter (below), avoiding
            # GPU inference inside each DFS round which causes growing slowdown.
            dt_for_traversal = None

            if in_phase2:
                phase2_local += 1

            # ── Traversal (parallel: n_trav_threads per player) ────────────
            buf_cap   = self.state.adv_buffers[0].capacity
            n_threads = self.n_trav_threads
            # Each thread handles traversals_per_iter // n_threads games
            per_thread = max(1, self.traversals_per_iter // n_threads)

            # Temp buffers: n_threads per player × 2 players
            total_threads = n_threads * 2  # 0..n_threads-1 = tp0, n_threads..2n-1 = tp1
            tmp_adv = [[ReservoirBuffer(buf_cap), ReservoirBuffer(buf_cap)]
                       for _ in range(total_threads)]
            tmp_str = [ReservoirBuffer(buf_cap) for _ in range(total_threads)]

            def _run_trav(thread_idx, traversing):
                self.postflop_cfr.run_traversals(
                    per_thread, traversing,
                    discard_trainer  = dt_for_traversal,
                    discard_n_games  = self.discard_n_games,
                    phase            = self._phase_idx + 1,
                    adv_bufs         = tmp_adv[thread_idx],
                    str_buf          = tmp_str[thread_idx],
                )

            inner.reset(total=self.traversals_per_iter * 2)
            inner.set_description(f'P{self._phase_idx+1} Trav×{n_threads*2}')
            threads = []
            for tp in range(2):
                for j in range(n_threads):
                    idx = tp * n_threads + j
                    threads.append(threading.Thread(target=_run_trav, args=(idx, tp)))
            for th in threads: th.start()
            for th in threads: th.join()

            # Merge all temp buffers → real trainer buffers (sequential, safe)
            for idx in range(total_threads):
                for p in range(2):
                    self.state.adv_buffers[p].merge_from(tmp_adv[idx][p])
                self.state.strategy_buffer.merge_from(tmp_str[idx])
            inner.refresh()

            # ── Discard sample collection (Phase 2 + Phase 3) ────────────────
            # Single call after all traversal threads: avoids n_trav_threads×2 concurrent
            # CPU run_iters. Uses a fresh batch for efficient GPU inference.
            if in_phase2 or in_phase3:
                n_disc = self.discard_n_games * max(1, self.n_trav_threads)
                _, _, _, _, comms_d, p0h5_d, p1h5_d = batch_deal_discard(n_disc)
                self.state.discard_trainer.run_iter(p0h5_d, p1h5_d, comms_d)

            # ── Training ────────────────────────────────────────────────
            if (t + 1) % self.train_interval == 0:
                inner.set_description('Train')
                losses = self.postflop_cfr.train()

                if in_phase2 or in_phase3:
                    dt = self.state.discard_trainer
                    if len(dt.buf) >= dt.batch_size:
                        discard_loss = dt.train()
                        losses.append(discard_loss)

                inner.refresh()

            # ── Progress bar ───────────────────────────────────────────────
            elapsed = time.time() - t0
            done    = t - start_iter + 1
            buf     = [len(b) for b in self.state.adv_buffers]
            d_buf   = len(self.state.discard_trainer.buf)
            pbar.set_postfix({
                'ph':    self._phase_idx + 1,
                'it/s':  f'{done / elapsed:.1f}',
                'loss':  f'{losses[0]:.3f}/{losses[1]:.3f}',
                'dloss': f'{discard_loss:.4f}',
                'buf':   f'{buf[0]//1000}K/{buf[1]//1000}K',
                'dbuf':  f'{d_buf//1000}K',
            }, refresh=False)

            # ── Checkpoint ─────────────────────────────────────────────────
            if (t + 1) % self.checkpoint_interval == 0:
                save_checkpoint(self.state, ckpt_path, t + 1, save_buffers=True)
                tagged = os.path.join(self.checkpoint_dir,
                                      f'checkpoint_{t+1:04d}.pt')
                save_checkpoint(self.state, tagged, t + 1, save_buffers=False)
                tqdm.write(f'  [ckpt] iter {t+1} → {tagged}')
                from range_finder.eval import eval_rangefinder_mse
                rf = eval_rangefinder_mse(self.state, n_games=200)
                tqdm.write(
                    f'  [rf]  p_true {rf["p_true_uniform"]:.4f}'
                    f'→{rf["p_true_post_discard"]:.4f}'
                )

            if _interrupted[0]:
                save_checkpoint(self.state, ckpt_path, t + 1, save_buffers=True)
                tqdm.write(f'  [ckpt] interrupted at iter {t+1}')
                inner.close()
                return

        inner.close()
        _phase_dt = time.time() - _phase_t0
        _s_per_it = _phase_dt / max(_phase_iters, 1)
        tqdm.write(
            f'\n  [timing] {self._current_phase.name}: '
            f'{_phase_iters} iters  {_phase_dt:.1f}s total  '
            f'{_s_per_it:.2f}s/iter'
        )
        tqdm.write('\nTraining strategy networks...')
        self.postflop_cfr.train_strategy(
            num_batches=self.num_batches * 3)
        elapsed = time.time() - t0
        tqdm.write(
            f'Done: {self.num_iterations} iters in {elapsed:.0f}s'
            f'  ({elapsed / self.num_iterations:.1f}s/iter)')

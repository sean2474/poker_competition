"""
preflop_cfr/train.py — External-sampling MCCFR for preflop with self-play terminals.

Terminal values come from prob-agent self-play:
  - Fold  → fold payoff (chips in pot)
  - Call  → both players discard (prob agent) → showdown → actual winner
"""

import ctypes
import random
import time
import numpy as np
from tqdm import tqdm

from .core   import Preflop
from .state  import _State
from .utils  import _SLOT, _match
from game.game import DECK_SIZE, canonicalize


# ── Shared progress counter (set by worker initializer) ─────────────────────

_g_progress = None
_g_lock      = None

def _init_worker_progress(shared_val, lock):
    global _g_progress, _g_lock
    _g_progress = shared_val
    _g_lock     = lock


# ── Evaluator (lazy, shared) ──────────────────────────────────────────────────

_ev_cache = {}

def _get_eval():
    if not _ev_cache:
        from gym_env import PokerEnv, WrappedEval
        _ev_cache['ev']  = WrappedEval()
        _ev_cache['itc'] = PokerEnv.int_to_card
    return _ev_cache['ev'], _ev_cache['itc']


# ── Terminal value ────────────────────────────────────────────────────────────

def _terminal(h0, h1, state, traverser, rng, heuristic, discard_sims):
    """Simulate discard+showdown between two prob agents → value to traverser."""
    if state.folded_by >= 0:
        winner = 1 - state.folded_by
        gain   = float(min(state.bets))
        return gain if traverser == winner else -gain

    # Deal 3-card flop
    dead      = set(h0) | set(h1)
    remaining = [c for c in range(DECK_SIZE) if c not in dead]
    flop      = rng.sample(remaining, 3)

    # Both players keep best 2 cards (prob agent) based on flop
    ki0, kj0 = heuristic.best_discard(list(h0), flop, num_sims=discard_sims)
    ki1, kj1 = heuristic.best_discard(list(h1), flop, num_sims=discard_sims)
    kept0 = [h0[ki0], h0[kj0]]
    kept1 = [h1[ki1], h1[kj1]]

    # Deal turn + river from remaining deck (exclude all known cards)
    dead_full  = dead | set(flop)
    remaining2 = [c for c in range(DECK_SIZE) if c not in dead_full]
    turn, river = rng.sample(remaining2, 2)
    board = flop + [turn, river]

    # Showdown with full 5-card board (no postflop betting)
    ev_obj, itc = _get_eval()
    brd = [itc(c) for c in board]
    s0  = ev_obj.evaluate([itc(c) for c in kept0], brd)
    s1  = ev_obj.evaluate([itc(c) for c in kept1], brd)

    bet = float(min(state.bets))
    if   s0 < s1: winner = 0   # lower = better in treys
    elif s1 < s0: winner = 1
    else:         winner = -1  # tie

    if winner == traverser:     return  bet
    if winner == 1 - traverser: return -bet
    return 0.


# ── CFR step ──────────────────────────────────────────────────────────────────

def _cfr(h0, h1, state, traverser, regrets, strat_sum, t, rng,
         heuristic, discard_sims, terminal_fn=None):
    """External-sampling MCCFR step.
    terminal_fn(h0, h1, state, traverser, rng) overrides default _terminal.
    """
    if state.done:
        if terminal_fn is not None:
            return terminal_fn(h0, h1, state, traverser, rng)
        return _terminal(h0, h1, state, traverser, rng, heuristic, discard_sims)

    cp    = state.acting
    hand  = h0 if cp == 0 else h1
    key   = (canonicalize(hand), state.hist)
    valid = state.valid()
    regs  = regrets.setdefault(key, np.zeros(3))
    strat = _match(regs, valid)

    ss = strat_sum.setdefault(key, np.zeros(3))
    for a in valid:
        ss[_SLOT[a]] += t * strat[a]

    if cp == traverser:
        vals = {a: _cfr(h0, h1, state.apply(a), traverser,
                        regrets, strat_sum, t, rng, heuristic, discard_sims,
                        terminal_fn)
                for a in valid}
        ev = sum(strat[a] * vals[a] for a in valid)
        for a in valid:
            regs[_SLOT[a]] = max(0., regs[_SLOT[a]] + vals[a] - ev)
        return ev
    else:
        a_s = rng.choices(valid, weights=[strat[a] for a in valid])[0]
        return _cfr(h0, h1, state.apply(a_s), traverser,
                    regrets, strat_sum, t, rng, heuristic, discard_sims,
                    terminal_fn)


# ── Parallel worker (module-level for multiprocessing spawn) ──────────────────

def _worker_fn(args):
    """Run N iters of MCCFR independently. Returns strat_sum dict."""
    import os, sys
    _this   = os.path.abspath(__file__)
    _cfr_dir  = os.path.dirname(os.path.dirname(_this))          # deep_cfr_training/
    _proj_dir = os.path.dirname(_cfr_dir)                        # project root (gym_env.py)
    for p in (_cfr_dir, _proj_dir):
        if p not in sys.path:
            sys.path.insert(0, p)

    worker_id, n_iters, discard_sims = args

    from preflop_cfr.state  import _State
    from preflop_cfr.utils  import _SLOT, _match
    from game.game          import DECK_SIZE, canonicalize
    from heuristic.prob_agent import HeuristicAgent

    heuristic  = HeuristicAgent()
    regrets    = {}
    strat_sum  = {}
    rng        = random.Random(worker_id * 97651 + 42)

    _report_every = max(50, n_iters // 200)   # ~200 flushes per worker
    _local = 0

    for t in range(1, n_iters + 1):
        deck = list(range(DECK_SIZE))
        rng.shuffle(deck)
        h0, h1 = tuple(deck[:5]), tuple(deck[5:10])
        for tp in [0, 1]:
            _cfr(h0, h1, _State(), tp,
                 regrets, strat_sum, t, rng,
                 heuristic, discard_sims)
        _local += 1
        if _local >= _report_every and _g_progress is not None:
            with _g_lock:
                _g_progress.value += _local
            _local = 0

    if _local > 0 and _g_progress is not None:
        with _g_lock:
            _g_progress.value += _local

    return strat_sum


def _merge_strat_sums(results: list) -> dict:
    """Sum strat_sum arrays from all workers → correct average strategy."""
    merged = {}
    for ss in results:
        for key, arr in ss.items():
            if key in merged:
                merged[key] = merged[key] + arr
            else:
                merged[key] = arr.copy()
    return merged


# ── Main training function ────────────────────────────────────────────────────

def train(n_iters: int = 200_000, save_path: str = None,
          discard_sims: int = 20, log_every: int = None,
          n_workers: int = 1, terminal_fn=None,
          init_regrets: dict = None, init_strat_sum: dict = None) -> Preflop:
    """
    Run external-sampling MCCFR for preflop.

    Args:
        n_iters:      total MCCFR iterations (split across workers)
        save_path:    where to save the trained chart (.pkl)
        discard_sims: MC sims per keep-pair in discard simulation
        log_every:    print progress every N iters (single-worker only)
        n_workers:    number of parallel processes (default 1 = single-threaded)
    """
    if n_workers > 1:
        return _train_parallel(n_iters, save_path, discard_sims, n_workers)

    from heuristic.prob_agent import get_heuristic_agent

    heuristic = get_heuristic_agent()
    rng = random.Random()
    p   = Preflop()
    if init_regrets   is not None: p._regrets   = init_regrets
    if init_strat_sum is not None: p._strat_sum = init_strat_sum

    with tqdm(range(1, n_iters + 1), desc='preflop MCCFR', ncols=80) as bar:
        for t in bar:
            deck = list(range(DECK_SIZE))
            rng.shuffle(deck)
            h0, h1 = tuple(deck[:5]), tuple(deck[5:10])

            for tp in [0, 1]:
                _cfr(h0, h1, _State(), tp,
                     p._regrets, p._strat_sum, t, rng,
                     heuristic, discard_sims, terminal_fn)

            if t % max(1, n_iters // 20) == 0:
                bar.set_postfix(infosets=len(p._strat_sum))

    p._build_chart()
    if save_path:
        p.save(save_path)
    return p


def _train_parallel(n_iters: int, save_path: str, discard_sims: int,
                    n_workers: int) -> Preflop:
    from multiprocessing import Pool, Value, Lock

    iters_per = n_iters // n_workers
    remainder = n_iters - iters_per * n_workers

    # Last worker gets the remainder iterations
    work = [(i, iters_per + (remainder if i == n_workers - 1 else 0), discard_sims)
            for i in range(n_workers)]

    print(f'[parallel] {n_workers} workers × ~{iters_per} iters = {n_iters} total')

    # Pre-warm evaluator in parent before forking so workers inherit the cached
    # module and gym's deprecation warning fires only once.
    _get_eval()

    progress  = Value(ctypes.c_int64, 0)
    prog_lock = Lock()

    with Pool(processes=n_workers,
              initializer=_init_worker_progress,
              initargs=(progress, prog_lock)) as pool:
        async_results = [pool.apply_async(_worker_fn, (w,)) for w in work]

        with tqdm(total=n_iters, desc='preflop MCCFR', ncols=80,
                  unit='iter', unit_scale=True) as bar:
            last = 0
            while not all(r.ready() for r in async_results):
                time.sleep(0.25)
                curr = progress.value
                if curr > last:
                    bar.update(curr - last)
                    last = curr
            curr = progress.value
            if curr > last:
                bar.update(curr - last)

        results = [r.get() for r in async_results]

    print(f'[parallel] merging {n_workers} strat_sum dicts ...')
    merged = _merge_strat_sums(results)

    p = Preflop()
    p._strat_sum = merged
    p._build_chart()
    print(f'[parallel] total infosets={len(p._chart)}')
    if save_path:
        p.save(save_path)
    return p

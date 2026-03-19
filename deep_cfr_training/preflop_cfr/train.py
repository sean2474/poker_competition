"""
preflop_cfr/train.py — External-sampling MCCFR for preflop with self-play terminals.

Terminal values come from prob-agent self-play:
  - Fold  → fold payoff (chips in pot)
  - Call  → both players discard (prob agent) → showdown → actual winner
"""

import random
import numpy as np
from tqdm import tqdm

from preflop_cfr.core  import Preflop
from preflop_cfr.state import _State
from preflop_cfr.utils import _SLOT, _match
from game.game         import DECK_SIZE, canonicalize


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

    # Deal 3-card board
    dead      = set(h0) | set(h1)
    remaining = [c for c in range(DECK_SIZE) if c not in dead]
    board     = rng.sample(remaining, 3)

    # Both players keep best 2 cards (prob agent)
    ki0, kj0 = heuristic.best_discard(list(h0), board, num_sims=discard_sims)
    ki1, kj1 = heuristic.best_discard(list(h1), board, num_sims=discard_sims)
    kept0 = [h0[ki0], h0[kj0]]
    kept1 = [h1[ki1], h1[kj1]]

    # Showdown (no postflop betting — clean approximation for preflop training)
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
         heuristic, discard_sims):
    """External-sampling MCCFR step with self-play terminal values."""
    if state.done:
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
                        regrets, strat_sum, t, rng, heuristic, discard_sims)
                for a in valid}
        ev = sum(strat[a] * vals[a] for a in valid)
        for a in valid:
            regs[_SLOT[a]] = max(0., regs[_SLOT[a]] + vals[a] - ev)
        return ev
    else:
        a_s = rng.choices(valid, weights=[strat[a] for a in valid])[0]
        return _cfr(h0, h1, state.apply(a_s), traverser,
                    regrets, strat_sum, t, rng, heuristic, discard_sims)


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

    for t in tqdm(range(1, n_iters + 1), desc=f'worker {worker_id}',
                  position=worker_id, leave=True, ncols=80):
        deck = list(range(DECK_SIZE))
        rng.shuffle(deck)
        h0, h1 = tuple(deck[:5]), tuple(deck[5:10])
        for tp in [0, 1]:
            _cfr(h0, h1, _State(), tp,
                 regrets, strat_sum, t, rng,
                 heuristic, discard_sims)

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
          n_workers: int = 1) -> Preflop:
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

    with tqdm(range(1, n_iters + 1), desc='preflop MCCFR', ncols=80) as bar:
        for t in bar:
            deck = list(range(DECK_SIZE))
            rng.shuffle(deck)
            h0, h1 = tuple(deck[:5]), tuple(deck[5:10])

            for tp in [0, 1]:
                _cfr(h0, h1, _State(), tp,
                     p._regrets, p._strat_sum, t, rng,
                     heuristic, discard_sims)

            if t % max(1, n_iters // 20) == 0:
                bar.set_postfix(infosets=len(p._strat_sum))

    p._build_chart()
    if save_path:
        p.save(save_path)
    return p


def _train_parallel(n_iters: int, save_path: str, discard_sims: int,
                    n_workers: int) -> Preflop:
    from multiprocessing import Pool

    iters_per = n_iters // n_workers
    remainder = n_iters - iters_per * n_workers

    # Last worker gets the remainder iterations
    work = [(i, iters_per + (remainder if i == n_workers - 1 else 0), discard_sims)
            for i in range(n_workers)]

    print(f'[parallel] {n_workers} workers × ~{iters_per} iters = {n_iters} total')

    with Pool(processes=n_workers) as pool:
        results = pool.map(_worker_fn, work)

    print(f'[parallel] merging {n_workers} strat_sum dicts ...')
    merged = _merge_strat_sums(results)

    p = Preflop()
    p._strat_sum = merged
    p._build_chart()
    print(f'[parallel] total infosets={len(p._chart)}')
    if save_path:
        p.save(save_path)
    return p

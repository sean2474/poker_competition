"""
test_preflop_cfr.py — Compare ProbAgent vs CFR-Preflop + ProbAgent.

Uses Agent class from deep_cfr_training/agent.py.
Runs rounds in parallel via multiprocessing.

Run:
  python test_preflop_cfr.py --hands 300 --rounds 5
"""

import os, sys, pickle, random, argparse
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

_ROOT  = os.path.dirname(os.path.abspath(__file__))
_TRAIN = os.path.join(_ROOT, 'deep_cfr_training')
for _p in (_ROOT, _TRAIN):
    if _p not in sys.path:
        sys.path.insert(0, _p)

FOLD, RAISE, CHECK, CALL, DISCARD = 0, 1, 2, 3, 4


# ── PokerEnv ↔ Agent bridge ───────────────────────────────────────────────────

def _char_to_env(ch: str, obs: dict) -> tuple:
    """Agent action char → PokerEnv action tuple."""
    valid = obs['valid_actions']
    if ch == 'f':
        return (FOLD, 0, 0, 0)
    if ch == 'r' and valid[RAISE]:
        return (RAISE, max(obs['min_raise'], 1), 0, 0)
    if valid[CALL]:
        return (CALL, 0, 0, 0)
    return (CHECK, 0, 0, 0)


def _env_to_char(at: int) -> str:
    return {FOLD: 'f', CALL: 'c', CHECK: 'k', RAISE: 'r'}.get(at, 'c')


class _EnvPlayer:
    """Wraps Agent + HeuristicAgent for direct PokerEnv play."""

    def __init__(self, agent):
        self.agent = agent
        self._pf_hist   = ''   # preflop history for Agent
        self._post_hist = ''   # postflop history for Agent
        self._prev_street = -1

    def reset(self):
        self.agent.reset()
        self._pf_hist   = ''
        self._post_hist = ''
        self._prev_street = -1

    def act(self, obs: dict) -> tuple:
        street = obs['street']
        valid  = obs['valid_actions']
        hand   = [c for c in obs['my_cards']        if c >= 0]
        board  = [c for c in obs['community_cards'] if c >= 0]
        opp_d  = [c for c in obs['opp_discarded_cards'] if c >= 0]

        if valid[DISCARD]:                         # discard phase
            keep_idx, _ = self.agent.act_discard(hand, board,
                                                  opp_d if opp_d else None)
            return (DISCARD, 0, keep_idx[0], keep_idx[1])

        if street == 0:                            # preflop betting
            ch, _ = self.agent.act_preflop(hand, self._pf_hist)
            env_a  = _char_to_env(ch, obs)
            self._pf_hist += _env_to_char(env_a[0])
            return env_a

        # postflop betting
        ch, _ = self.agent.act_postflop(hand, board, self._post_hist)
        env_a  = _char_to_env(ch, obs)
        self._post_hist += _env_to_char(env_a[0])
        return env_a

    def observe_opp(self, obs: dict, opp_action_tuple: tuple):
        """Update range with opponent's observed action."""
        street = obs['street']
        at     = opp_action_tuple[0]
        board  = [c for c in obs['community_cards'] if c >= 0]
        opp_d  = [c for c in obs['opp_discarded_cards'] if c >= 0]

        if at == DISCARD:
            if opp_d:
                self.agent.observe_opp_discard(opp_d, board)
        elif street == 0:
            self.agent.observe_opp_preflop(_env_to_char(at), self._pf_hist)
        else:
            self.agent.observe_opp_postflop(_env_to_char(at),
                                            self._post_hist, board)


# ── Worker (module-level for multiprocessing) ─────────────────────────────────

def _build_heuristic_agent():
    from heuristic.prob_agent  import HeuristicAgent
    from heuristic.discard     import HeuristicDiscard
    from heuristic.postflop    import HeuristicPostflop
    from agent import Agent
    h = HeuristicAgent()
    class _HeuristicPreflop:
        """Thin wrapper: uses HeuristicAgent's equity for preflop."""
        def action(self, hand, history):
            from stretegy.preflop_cfr.state import _State
            state = _State()
            for ch in history: state = state.apply(ch)
            valid = state.valid()
            eq = h.compute_equity(hand[:2], [], [], 200)
            if   'r' in valid and eq >= 0.58: best = 'r'
            elif 'f' in valid and eq < 0.38:  best = 'f'
            elif 'c' in valid:                best = 'c'
            elif 'k' in valid:                best = 'k'
            else:                             best = valid[0]
            return best, {a: (0.7 if a == best else 0.15) for a in valid}
        def train(self, **kw): pass
        def save(self, *a): pass
        def load(self, *a): pass
    return Agent(_HeuristicPreflop(), HeuristicDiscard(), HeuristicPostflop())


def _build_cfr_agent(chart: dict):
    from heuristic.discard     import HeuristicDiscard
    from heuristic.postflop    import HeuristicPostflop
    from stretegy.preflop_cfr.core import Preflop
    from agent import Agent
    preflop = Preflop()
    preflop._chart = chart
    return Agent(preflop, HeuristicDiscard(), HeuristicPostflop())


def _run_match_worker(args):
    """Worker: (chart_or_None, n_hands, seed) → (chips_p0, chips_p1, wins_p0, wins_p1)"""
    chart, n_hands, seed = args

    # gym passes numpy integers to random.seed() which fails on Python 3.12
    import random as _r
    _orig = _r.seed
    def _safe_seed(a=None, version=2):
        if a is not None and not isinstance(a, (int, float, str, bytes, bytearray)):
            a = int(a)
        return _orig(a, version=version)
    _r.seed = _safe_seed

    _r.seed(seed)
    np.random.seed(seed % (2**32))

    from gym_env import PokerEnv

    if chart is None:
        agents = [_build_heuristic_agent(), _build_heuristic_agent()]
    else:
        agents = [_build_cfr_agent(chart), _build_heuristic_agent()]

    players = [_EnvPlayer(a) for a in agents]
    chips   = [0., 0.]
    wins    = [0,  0]

    for _ in range(n_hands):
        env = PokerEnv()
        obs, _ = env.reset()
        for p in players: p.reset()

        terminated = False
        rew = [0., 0.]
        last_action = [None, None]

        while not terminated:
            acting = env.acting_agent
            action = players[acting].act(obs[acting])
            last_action[acting] = action

            obs, reward, terminated, truncated, info = env.step(action)
            rew = list(reward)

            # Notify opponent about action just taken
            opp = 1 - acting
            players[opp].observe_opp(obs[opp], action)

        chips[0] += rew[0]; chips[1] += rew[1]
        if rew[0] > 0: wins[0] += 1
        elif rew[1] > 0: wins[1] += 1

    return chips[0], chips[1], wins[0], wins[1]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hands',   type=int, default=300)
    parser.add_argument('--rounds',  type=int, default=5)
    parser.add_argument('--workers', type=int, default=min(cpu_count(), 10))
    parser.add_argument('--pkl', default='deep_cfr_training/models/preflop.pkl')
    args = parser.parse_args()

    with open(args.pkl, 'rb') as f:
        chart = pickle.load(f)
    print(f"Chart: {len(chart):,} infosets | "
          f"{args.rounds} rounds × {args.hands} hands | "
          f"{args.workers} workers\n")

    n, R = args.hands, args.rounds

    # Build work list: all baseline + all CFR rounds in parallel
    work = []
    for r in range(R):
        work.append((None,  n, 2000 + r))   # baseline
        work.append((chart, n, 3000 + r))   # CFR

    print('Running...', flush=True)
    with Pool(processes=args.workers) as pool:
        results = list(tqdm(
            pool.imap(_run_match_worker, work),
            total=len(work), desc='matches'
        ))

    base_rows = results[0::2]
    cfr_rows  = results[1::2]

    print()
    print(f"  {'Round':>5}  {'Baseline P0':>12}  {'CFR P0':>10}  {'CFR win%':>9}")
    print(f"  {'-'*5}  {'-'*12}  {'-'*10}  {'-'*9}")

    base_chips, cfr_chips, cfr_wr = [], [], []
    for i, (br, cr) in enumerate(zip(base_rows, cfr_rows)):
        bc, _, bw0, bw1 = br
        cc, _, cw0, cw1 = cr
        wr = cw0 / max(cw0 + cw1, 1) * 100
        base_chips.append(bc); cfr_chips.append(cc); cfr_wr.append(wr)
        print(f"  {i+1:>5}  {bc:>+12.0f}  {cc:>+10.0f}  {wr:>8.1f}%")

    print()
    delta = np.mean(cfr_chips) - np.mean(base_chips)
    print(f"  Baseline avg : {np.mean(base_chips):+.1f} chips/round")
    print(f"  CFR      avg : {np.mean(cfr_chips):+.1f} chips/round  "
          f"(Δ {delta:+.1f},  {delta/n:+.2f}/hand)")
    print(f"  CFR win-rate : {np.mean(cfr_wr):.1f}%")


if __name__ == '__main__':
    main()

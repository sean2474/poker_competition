"""
Hybrid CFR traversal.

Preflop nodes:  tabular CFR (regrets dict on trainer)
Postflop nodes: neural network (batch inference via generator yield)

traverse_coro: generator — yields (features, valid_actions, cp, street) at
               postflop nodes; handles preflop internally via tabular lookup.

run_traversals_batched: N coroutines simultaneously, batch-inferring only
                        postflop nodes (preflop uses tabular — no network call).
"""

import numpy as np
import torch

from game import (GameState, evaluate_showdown, batch_deal_discard,
                  batch_warmup_ev, PostflopBatch)
from game.constants import NUM_ACTIONS, BIG_BLIND, FEATURE_DIM
from preflop_cfr.canonical import canonicalize

WARMUP_ITERS       = 50    # hard fallback (if buffer never fills for some reason)
MIN_WARMUP_SAMPLES = 2000  # per street per player before switching to neural net


def _postflop_ready(trainer) -> bool:
    """Switch from warmup equity to neural net when buffer is sufficiently warm."""
    for p in range(2):
        for s in [1, 2, 3]:
            if len(trainer.adv_buffers[p].street_bufs[s]) < MIN_WARMUP_SAMPLES:
                return False
    return True


# Preflop 3-slot abstraction ─────────────────────────────────────────────
# slot 0 = FOLD
# slot 1 = CALL / CHECK  (never both valid at same node → safe to merge)
# slot 2 = RAISE         (BET_SMALL or RAISE_SMALL, always fixed size)
_PF_SLOTS = 3
_PF_SLOT  = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2}
#             F     C     K     bS    bL    rS    rL    bP


def _tabular_strategy(regrets: np.ndarray, valid_actions: list) -> dict:
    """CFR+ regret matching on 3-slot preflop abstraction."""
    # Map each concrete action to its abstract slot
    slot_to_actions: dict = {}
    for a in valid_actions:
        s = _PF_SLOT.get(a, 2)
        slot_to_actions.setdefault(s, []).append(a)

    pos   = [max(float(regrets[s]), 0.0) for s in range(_PF_SLOTS)]
    total = sum(pos[s] for s in slot_to_actions)
    if total > 0:
        probs = {s: pos[s] / total for s in slot_to_actions}
    else:
        n = len(slot_to_actions)
        probs = {s: 1.0 / n for s in slot_to_actions}

    strategy = {}
    for s, actions in slot_to_actions.items():
        per = probs[s] / len(actions)
        for a in actions:
            strategy[a] = per
    return strategy


_SZ_THRESHOLDS = [
    (2.6, 's'),   # ≤2.6bb: small open / limp (our 2.5bb open falls here)
    (3.2, 'm'),   # 2.6–3.2bb: medium (3bb open)
    (4.1, 'l'),   # 3.2–4.1bb: large (3.5–4bb open)
]                 # >4.1bb → 'L'


def _size_bucket(state) -> str:
    """Bucket max(bets) by BB multiples to handle opponent off-size opens."""
    bb_mult = max(state.bets) / BIG_BLIND
    for thresh, label in _SZ_THRESHOLDS:
        if bb_mult <= thresh:
            return label
    return 'L'


def _preflop_key(hand5, state) -> tuple:
    """Tabular infoset key: (canonical_hand, size_bucket, history_string)."""
    canon = canonicalize(hand5)
    hist  = ''.join(_ACTION_CHAR.get(a, '?') for _, a in state.history)
    return (canon, _size_bucket(state), hist)


_ACTION_CHAR = {0: 'f', 1: 'c', 2: 'k', 3: 'b', 4: 'B', 5: 'r', 6: 'R', 7: 'p'}


def traverse_coro(trainer, state, p0_hand, p1_hand, p0_hand5, p1_hand5,
                  community, p0_disc, p1_disc, traversing_player):
    """
    Hybrid generator traversal.
    - Preflop (street=0): tabular CFR regrets — handled inline, no yield.
    - Postflop (street>0): neural net — yields for batch inference.
    """
    if state.is_terminal:
        if state.folded_player >= 0:
            if state.folded_player == traversing_player:
                return -float(state.bets[traversing_player])
            else:
                return float(state.bets[1 - traversing_player])
        pot = min(state.bets[0], state.bets[1])
        sd  = evaluate_showdown(p0_hand, p1_hand, community)
        return float(sd * pot if traversing_player == 0 else -sd * pot)

    cp = state.current_player
    valid_actions = state.get_valid_actions()
    if not valid_actions:
        return 0.0

    if cp == 0:
        hand, hand5, is_bb = p0_hand, p0_hand5, False
        my_disc, opp_disc  = p0_disc, p1_disc
    else:
        hand, hand5, is_bb = p1_hand, p1_hand5, True
        my_disc, opp_disc  = p1_disc, p0_disc

    # ── PREFLOP: tabular CFR (no network inference) ───────────────────────
    if state.street == 0:
        key  = _preflop_key(hand5, state)
        regs = trainer.preflop_regrets.get(key, np.zeros(_PF_SLOTS))
        strategy = _tabular_strategy(regs, valid_actions)

        # Explore ALL actions (complete tree for preflop)
        action_values = {}
        for a in valid_actions:
            ns  = state.apply(a)
            sub = traverse_coro(trainer, ns, p0_hand, p1_hand, p0_hand5, p1_hand5,
                                 community, p0_disc, p1_disc, traversing_player)
            try:
                req = next(sub)
                while True:
                    resp = yield req   # forward postflop inference requests up
                    req  = sub.send(resp)
            except StopIteration as e:
                action_values[a] = e.value

        ev = sum(strategy.get(a, 0) * action_values[a] for a in valid_actions)

        if cp == traversing_player:
            # CFR+ regret update on 3 abstract slots
            r = trainer.preflop_regrets.setdefault(key, np.zeros(_PF_SLOTS))
            for a in valid_actions:
                slot = _PF_SLOT.get(a, 2)
                r[slot] = max(0.0, r[slot] + action_values[a] - ev)

        # Linear-weighted strategy accumulation (3 abstract slots)
        s = trainer.preflop_strategy_sum.setdefault(key, np.zeros(_PF_SLOTS))
        t = float(trainer.iteration)
        for a in valid_actions:
            slot = _PF_SLOT.get(a, 2)
            s[slot] += t * strategy.get(a, 0.0)

        return ev

    # ── POSTFLOP: yield ('CPP_POSTFLOP', ...) → runner handles via C++ ────────
    # Runner will call C++ batch warmup_ev OR C++ PostflopBatch state machine
    # and send back the postflop EV as a single float.
    postflop_ev = yield ('CPP_POSTFLOP',
                         state, p0_hand, p1_hand,
                         p0_hand5, p1_hand5, community, p0_disc, p1_disc)
    return float(postflop_ev)


def run_traversals_batched(trainer, traversals_per_iter: int, traversing_player: int):
    """
    C++ traversal: preflop Python tabular CFR → postflop C++ state machine.

    Each coroutine yields ('CPP_POSTFLOP', state, ...) once per postflop
    entry (one per preflop branch). Sequential rounds process each batch.

    Warmup: c_batch_warmup_ev (C++ OpenMP, no GPU)
    Neural: C++ PostflopBatch state machine + GPU batch inference
    """
    r = batch_deal_discard(traversals_per_iter)
    p0h, p1h, p0d, p1d, comms, p0h5, p1h5 = r

    is_warmup = (not _postflop_ready(trainer) and
                 getattr(trainer, 'iteration', 1) <= getattr(trainer, 'warmup_iters', WARMUP_ITERS))

    # Start N coroutines; they run preflop and yield CPP_POSTFLOP at first postflop state
    gens    = {}   # key → generator
    pending = {}   # key → (gen, (state, p0h, p1h, p0h5, p1h5, comm, p0d, p1d))

    for i in range(traversals_per_iter):
        g = traverse_coro(
            trainer, GameState(),
            list(p0h[i]),  list(p1h[i]),
            list(p0h5[i]), list(p1h5[i]),
            list(comms[i]), list(p0d[i]), list(p1d[i]),
            traversing_player,
        )
        gens[i] = g
        try:
            req = next(g)
            if req[0] == 'CPP_POSTFLOP':
                pending[i] = (g, req[1:])
        except StopIteration:
            del gens[i]

    # Process rounds: each round sends EVs to all pending coroutines,
    # which advance to the next preflop branch's postflop entry.
    if is_warmup:
        while pending:
            _warmup_round(trainer, gens, pending, traversing_player)
    else:
        while pending:
            _neural_round(trainer, gens, pending, traversing_player)


# ── Warmup round (C++ OpenMP equity) ─────────────────────────────────────────

def _warmup_round(trainer, gens, pending, tp):
    """One round: C++ batch equity for all pending CPP_POSTFLOP requests."""
    keys = list(pending.keys())
    n    = len(keys)
    if n == 0: return

    p0h5_arr = np.zeros((n, 5), dtype=np.int32)
    p1h5_arr = np.zeros((n, 5), dtype=np.int32)
    my_bets  = np.zeros(n, dtype=np.int32)
    opp_bets = np.zeros(n, dtype=np.int32)

    for j, k in enumerate(keys):
        g, (state, p0h, p1h, p0h5, p1h5, comm, p0d, p1d) = pending[k]
        p0h5_arr[j] = p0h5[:5]
        p1h5_arr[j] = p1h5[:5]
        my_bets[j]  = state.bets[tp]
        opp_bets[j] = state.bets[1 - tp]

    evs = batch_warmup_ev(p0h5_arr, p1h5_arr, my_bets, opp_bets,
                          np.full(n, tp, dtype=np.int32))

    pending.clear()
    for j, k in enumerate(keys):
        g = gens.get(k)
        if g is None: continue
        try:
            req = g.send(float(evs[j]))
            if req[0] == 'CPP_POSTFLOP': pending[k] = (g, req[1:])
        except StopIteration:
            del gens[k]


# ── Neural round (C++ PostflopBatch state machine + GPU) ─────────────────────

def _neural_round(trainer, gens, pending, tp):
    """One round: C++ PostflopBatch + GPU inference for all pending requests."""
    keys = list(pending.keys())
    n    = len(keys)
    if n == 0: return

    batch = PostflopBatch(n)
    for j, k in enumerate(keys):
        g, (state, p0h, p1h, p0h5, p1h5, comm, p0d, p1d) = pending[k]
        batch.init_one(j, state, p0h, p1h, p0h5, p1h5, comm, p0d, p1d, tp)

    # GPU inference loop (all n games advance together)
    while batch.n_pending() > 0:
        cnt, feats, valid, n_valid, players, game_idxs = batch.collect_pending()
        if cnt == 0: break
        strategies = np.zeros((cnt, NUM_ACTIONS), dtype=np.float32)
        for p in [0, 1]:
            pidx = np.where(players[:cnt] == p)[0]
            if len(pidx) == 0: continue
            x = torch.tensor(feats[pidx], dtype=torch.float32)
            with torch.no_grad():
                adv = trainer.adv_nets[p](x).numpy()
            for k2, gi in enumerate(pidx):
                strategies[gi] = adv[k2]
        batch.resume(game_idxs[:cnt].copy(), strategies[:cnt])

    evs = batch.get_evs()

    # Collect + bulk-add buffer samples
    (adv_f, adv_v, adv_m, adv_s, adv_p, adv_i,
     str_f, str_v, str_m, str_s, str_i) = batch.collect_samples(
         float(trainer.iteration), tp)
    if len(adv_s) > 0:
        for p in [0, 1]:
            pm = adv_p == p
            if pm.any():
                trainer.adv_buffers[p].add_batch(
                    adv_f[pm], adv_v[pm], adv_i[pm], adv_m[pm], adv_s[pm])
    if len(str_s) > 0:
        trainer.strategy_buffer.add_batch(str_f, str_v, str_i, str_m, str_s)

    batch.free()
    pending.clear()
    for j, k in enumerate(keys):
        g = gens.get(k)
        if g is None: continue
        try:
            req = g.send(float(evs[j]))
            if req[0] == 'CPP_POSTFLOP': pending[k] = (g, req[1:])
        except StopIteration:
            del gens[k]

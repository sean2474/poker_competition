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
from game.constants import NUM_ACTIONS, BIG_BLIND
from preflop_cfr.canonical import canonicalize

MIN_WARMUP_SAMPLES = 2000  # per street per player before switching to neural net

_rng = np.random.default_rng()


def _postflop_ready(trainer) -> bool:
    """
    Dynamically switch from C++ equity warmup to neural net.
    Ends warmup when every (player, street) slot has MIN_WARMUP_SAMPLES.
    No hard iteration cap — warmup length adapts to traversal speed.
    """
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

        if cp == traversing_player:
            # Traversing player: explore ALL branches for regret update
            action_values = {}
            for a in valid_actions:
                ns  = state.apply(a)
                sub = traverse_coro(trainer, ns, p0_hand, p1_hand, p0_hand5, p1_hand5,
                                     community, p0_disc, p1_disc, traversing_player)
                try:
                    req = next(sub)
                    while True:
                        resp = yield req
                        req  = sub.send(resp)
                except StopIteration as e:
                    action_values[a] = e.value

            ev = sum(strategy.get(a, 0) * action_values[a] for a in valid_actions)

            # CFR+ regret update on 3 abstract slots
            r = trainer.preflop_regrets.setdefault(key, np.zeros(_PF_SLOTS))
            for a in valid_actions:
                slot = _PF_SLOT.get(a, 2)
                r[slot] = max(0.0, r[slot] + action_values[a] - ev)
        else:
            # Opponent (external sampling): sample ONE action to cut branching 3-9x
            weights = [max(strategy.get(a, 0.0), 0.0) for a in valid_actions]
            total_w = sum(weights)
            if total_w > 0:
                weights = [w / total_w for w in weights]
            else:
                n = len(valid_actions)
                weights = [1.0 / n] * n
            a_sampled = _rng.choice(len(valid_actions), p=np.array(weights, dtype=np.float64))
            a_sampled = valid_actions[a_sampled]
            ns  = state.apply(a_sampled)
            sub = traverse_coro(trainer, ns, p0_hand, p1_hand, p0_hand5, p1_hand5,
                                 community, p0_disc, p1_disc, traversing_player)
            try:
                req = next(sub)
                while True:
                    resp = yield req
                    req  = sub.send(resp)
            except StopIteration as e:
                ev = e.value

        # Linear-weighted strategy accumulation for current player
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


# Keep-pair index lookup: for chosen (ai,aj), which cards are discarded?
_KP_DISC = [[k for k in range(5) if k not in (ai, aj)]
             for ai, aj in [(i,j) for i in range(5) for j in range(i+1,5)]]


def _recompute_discards_with_cfr(p0h5, p1h5, comms, discard_trainer):
    """
    Phase 2/3: replace fast_discard with DiscardCFR choices.

    Fully batched: pair features built by C++ (OpenMP), context built by numpy.
    Two DiscardNet forwards on contiguous arrays. No Python-level game loop.
    """
    import torch
    from discard_cfr.features import KEEP_PAIRS as DK_PAIRS, PAIR_DIM, CTX_DIM
    from game.features import _c_lib
    import ctypes

    N     = len(p0h5)
    FDIM  = PAIR_DIM + CTX_DIM   # 44
    PDIM  = 23                   # C++ pair-feature dim (matches _DISCARD_PAIR_DIM)

    p0h   = np.zeros((N, 2), dtype=np.int32)
    p1h   = np.zeros((N, 2), dtype=np.int32)
    p0d   = np.full((N, 3), -1, dtype=np.int32)
    p1d   = np.full((N, 3), -1, dtype=np.int32)

    # ── Step 1: batch pair features via single C++ call ───────────────────────
    pair_A = np.zeros((N * 10, PDIM), dtype=np.float32)
    pair_B = np.zeros((N * 10, PDIM), dtype=np.float32)
    boards3 = np.ascontiguousarray(comms[:, :3], dtype=np.int32)
    _c_lib.c_build_discard_pair_features_batch(
        ctypes.c_int(N),
        np.ascontiguousarray(p0h5, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        np.ascontiguousarray(p1h5, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        boards3.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        pair_A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        pair_B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )

    # ── Step 2: board rank context (vectorized numpy) ─────────────────────────
    brs = np.where(boards3 >= 0, boards3 % 9 / 8., 0.).astype(np.float32)  # [N,3]
    brs_rep = np.repeat(brs, 10, axis=0)   # [N*10, 3]

    # ── Step 3: build feats_A with uniform opp context ────────────────────────
    feats_A = np.empty((N * 10, FDIM), dtype=np.float32)
    feats_A[:, :PDIM] = pair_A
    feats_A[:, PDIM:PDIM+3]  = brs_rep
    feats_A[:, PDIM+3:PDIM+20] = 1.0 / 17  # uniform opp cats
    feats_A[:, PDIM+20] = 0.              # is_bb=False for player A

    # ── Step 4: DiscardNet forward for Player A (CPU, thread-safe copy) ───────
    import copy
    net = copy.deepcopy(discard_trainer.net).cpu().eval()
    with torch.no_grad():
        adv_A = net(torch.from_numpy(feats_A)).numpy()   # (N*10,)

    # ── Step 5: sample Player A discards (vectorized) ─────────────────────────
    adv_A_reshaped = adv_A.reshape(N, 10)
    pos_A = np.maximum(adv_A_reshaped, 0.)
    sums_A = pos_A.sum(axis=1, keepdims=True)
    safe_A = np.where(sums_A > 0, sums_A, 1.0)   # avoid 0/0 when all advantages negative
    strat_A = np.where(sums_A > 0, pos_A / safe_A, 1./10)
    strat_A /= strat_A.sum(axis=1, keepdims=True)  # renorm for floating point
    # Sample using cumsum trick (vectorized, avoids Python loop)
    r_A = _rng.random((N,))
    cumA = np.cumsum(strat_A.astype(np.float64), axis=1)
    ka_arr = (r_A[:, None] > cumA).sum(axis=1).clip(0, 9)

    for i in range(N):
        ka = int(ka_arr[i])
        ai, aj = DK_PAIRS[ka]
        h5A = p0h5[i]
        p0h[i] = [h5A[ai], h5A[aj]]
        p0d[i] = [h5A[k] for k in _KP_DISC[ka]]

    # ── Step 6: Player B context — opp cats from C++ batch ────────────────────
    opp_cats_B = np.empty((N, 17), dtype=np.float32)
    _c_lib.c_opp_cats_narrowed_batch(
        ctypes.c_int(N),
        np.ascontiguousarray(p1h5, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        boards3.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        np.ascontiguousarray(p0d, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        opp_cats_B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    opp_cats_B_rep = np.repeat(opp_cats_B, 10, axis=0)  # [N*10, 17]

    # ── Step 7: build feats_B with actual opp context ─────────────────────────
    feats_B = np.empty((N * 10, FDIM), dtype=np.float32)
    feats_B[:, :PDIM] = pair_B
    feats_B[:, PDIM:PDIM+3]   = brs_rep
    feats_B[:, PDIM+3:PDIM+20] = opp_cats_B_rep
    feats_B[:, PDIM+20] = 1.              # is_bb=True for player B

    # ── Step 8: DiscardNet forward for Player B ────────────────────────────────
    with torch.no_grad():
        adv_B = net(torch.from_numpy(feats_B)).numpy()   # (N*10,)

    # ── Step 9: sample Player B discards (vectorized) ─────────────────────────
    adv_B_reshaped = adv_B.reshape(N, 10)
    pos_B = np.maximum(adv_B_reshaped, 0.)
    sums_B = pos_B.sum(axis=1, keepdims=True)
    safe_B = np.where(sums_B > 0, sums_B, 1.0)   # avoid 0/0 when all advantages negative
    strat_B = np.where(sums_B > 0, pos_B / safe_B, 1./10)
    strat_B /= strat_B.sum(axis=1, keepdims=True)
    r_B = _rng.random((N,))
    cumB = np.cumsum(strat_B.astype(np.float64), axis=1)
    kb_arr = (r_B[:, None] > cumB).sum(axis=1).clip(0, 9)

    for i in range(N):
        kb = int(kb_arr[i])
        bi, bj = DK_PAIRS[kb]
        h5B = p1h5[i]
        p1h[i] = [h5B[bi], h5B[bj]]
        p1d[i] = [h5B[k] for k in _KP_DISC[kb]]

    return p0h, p1h, p0d, p1d


def run_traversals_batched(trainer, traversals_per_iter: int, traversing_player: int,
                           discard_trainer=None, discard_n_games: int = 50,
                           phase: int = 1,
                           adv_bufs=None, str_buf=None):
    """
    C++ traversal: preflop Python tabular CFR → postflop C++ state machine.

    Phase 1: fast_discard choices (heuristic) — range uses fast_discard distribution
    Phase 2/3: DiscardCFR choices — range features match actual discard strategy

    discard_trainer: optional DiscardCFR — if set, also accumulates discard samples.
    discard_n_games: max games forwarded to discard_trainer per call.
    """
    r = batch_deal_discard(traversals_per_iter)
    p0h, p1h, p0d, p1d, comms, p0h5, p1h5 = r

    if discard_trainer is not None:
        # Phase 2/3: recompute discards with DiscardCFR so range features match
        p0h, p1h, p0d, p1d = _recompute_discards_with_cfr(
            p0h5, p1h5, comms, discard_trainer)
        n_d = min(discard_n_games, traversals_per_iter)
        discard_trainer.run_iter(p0h5[:n_d], p1h5[:n_d], comms[:n_d])

    is_warmup = not _postflop_ready(trainer)

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
    # Phase 3 ALWAYS uses neural mode (bootstraps from empty buffers).
    # Phase 1/2 use warmup unless buffers are pre-filled.
    is_warmup = is_warmup and (phase < 3)

    # Compute per-game ranges from the discard phase output.
    # These carry over the discard-updated range to postflop (no re-init from scratch).
    init_ranges = None
    if not is_warmup:
        opp_ranges, my_ranges = PostflopBatch.compute_ranges_batch(
            traversals_per_iter, traversing_player,
            np.ascontiguousarray(p0h, dtype=np.int32),
            np.ascontiguousarray(p1h, dtype=np.int32),
            np.ascontiguousarray(p0d, dtype=np.int32),
            np.ascontiguousarray(p1d, dtype=np.int32),
            np.ascontiguousarray(comms, dtype=np.int32),
        )
        init_ranges = (opp_ranges, my_ranges)  # indexed by original game key

    if is_warmup:
        while pending:
            _warmup_round(trainer, gens, pending, traversing_player)
    else:
        while pending:
            _neural_round(trainer, gens, pending, traversing_player,
                          init_ranges=init_ranges,
                          adv_bufs=adv_bufs, str_buf=str_buf)


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

def _neural_round(trainer, gens, pending, tp, game_states=None,
                   init_ranges=None, adv_bufs=None, str_buf=None):
    """One round: C++ PostflopBatch + GPU inference for all pending requests."""
    keys = list(pending.keys())
    n    = len(keys)
    if n == 0: return

    assert init_ranges is not None, \
        '_neural_round called without init_ranges — ranges must be passed from discard phase'
    opp_ranges, my_ranges = init_ranges
    assert opp_ranges.shape[1] == 351 and my_ranges.shape[1] == 351, \
        f'init_ranges wrong shape: opp={opp_ranges.shape} my={my_ranges.shape}'

    batch = PostflopBatch(n)
    for j, k in enumerate(keys):
        g, (state, p0h, p1h, p0h5, p1h5, comm, p0d, p1d) = pending[k]
        assert 0 <= k < opp_ranges.shape[0], \
            f'game key {k} out of range for init_ranges (size={opp_ranges.shape[0]})'
        batch.init_one(j, state, p0h, p1h, p0h5, p1h5, comm, p0d, p1d, tp,
                       opp_range=opp_ranges[k], my_range=my_ranges[k])

    # GPU inference loop (all n games advance together)
    device = trainer.device
    nets   = trainer.adv_nets
    while batch.n_pending() > 0:
        cnt, feats, valid, n_valid, players, game_idxs = batch.collect_pending()
        if cnt == 0: break
        # Python range_tracker disabled (game_states=None); C++ handles range features.
        if game_states is not None:
            from postflop_cfr.range_tracker import apply_range_features
            info = batch.get_pending_game_info(cnt)
            feats = apply_range_features(feats[:cnt].copy(), info, game_states)
        strategies = np.zeros((cnt, NUM_ACTIONS), dtype=np.float32)
        f_t = torch.from_numpy(feats[:cnt]).to(device, non_blocking=True)
        for p in [0, 1]:
            pidx = np.where(players[:cnt] == p)[0]
            if len(pidx) == 0: continue
            with torch.no_grad():
                adv = nets[p](f_t[pidx]).cpu().numpy()
            strategies[pidx] = adv   # vectorized: no Python loop
        batch.resume(game_idxs[:cnt].copy(), strategies[:cnt])

    evs = batch.get_evs()

    # Collect + bulk-add buffer samples (C++ outputs 75-dim directly)
    (adv_f, adv_v, adv_m, adv_s, adv_p, adv_i,
     str_f, str_v, str_m, str_s, str_i) = batch.collect_samples(
         float(trainer.iteration), tp)
    _adv_bufs = adv_bufs if adv_bufs is not None else trainer.adv_buffers
    _str_buf  = str_buf  if str_buf  is not None else trainer.strategy_buffer
    if len(adv_s) > 0:
        for p in [0, 1]:
            pm = adv_p == p
            if pm.any():
                _adv_bufs[p].add_batch(
                    adv_f[pm], adv_v[pm], adv_i[pm], adv_m[pm], adv_s[pm])
    if len(str_s) > 0:
        _str_buf.add_batch(str_f, str_v, str_i, str_m, str_s)

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

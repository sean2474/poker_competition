"""
Hybrid CFR traversal.

Preflop nodes:  tabular CFR (regrets dict on trainer)
Postflop nodes: neural network (batch inference via generator yield)

traverse_coro: generator — yields (features, valid_actions, cp, street) at
               postflop nodes; handles preflop internally via tabular lookup.

run_traversals_batched: N coroutines simultaneously, batch-inferring only
                        postflop nodes (preflop uses tabular — no network call).
"""

import random
import numpy as np
import torch

from game import GameState, state_to_features, evaluate_showdown, batch_deal_discard
from game.constants import NUM_ACTIONS, BIG_BLIND
from preflop_cfr.canonical import canonicalize
from preflop_cfr.equity   import warmup_ev

WARMUP_ITERS       = 50    # hard fallback (if buffer never fills for some reason)
MIN_WARMUP_SAMPLES = 2000  # per street per player before switching to neural net


def _postflop_ready(trainer) -> bool:
    """Switch from warmup equity to neural net when buffer is sufficiently warm."""
    for p in range(2):
        for s in [1, 2, 3]:
            if len(trainer.adv_buffers[p].street_bufs[s]) < MIN_WARMUP_SAMPLES:
                return False
    return True


# ── Helpers ──────────────────────────────────────────────────────────────────

def _regret_matching(adv_arr, valid_actions: list) -> dict:
    """Regret matching for postflop neural network output."""
    total = 0.0
    best_a, best_v = valid_actions[0], -1e9
    for a in valid_actions:
        v = float(adv_arr[a])
        if v > 0:
            total += v
        if v > best_v:
            best_v, best_a = v, a
    if total > 0:
        inv = 1.0 / total
        return {a: max(float(adv_arr[a]), 0) * inv for a in valid_actions}
    return {a: (1.0 if a == best_a else 0.0) for a in valid_actions}


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

    # ── POSTFLOP: warmup OR neural network ───────────────────────────────
    # Use equity approximation until buffer is warm enough OR hard iter limit
    _iter_limit = getattr(trainer, 'warmup_iters', WARMUP_ITERS)
    if not _postflop_ready(trainer) and getattr(trainer, 'iteration', 1) <= _iter_limit:
        return warmup_ev(p0_hand5, p1_hand5, state, traversing_player)

    vis_comm = ([], community[:3], community[:4], community[:5])[min(state.street, 3)]
    features = state_to_features(
        hand, vis_comm, state.bets[cp], state.bets[1 - cp],
        state.street, is_bb, my_disc, opp_disc,
        hero_hand5=None,
        street_bets=state.street_bets,
        history=state.history,
        num_actions_this_street=state.num_actions_this_street,
        street_last_ratios=state.street_last_ratios,
        street_bet_counts=state.street_bet_counts,
    )

    strategy = yield (features, valid_actions, cp, state.street)

    if cp == traversing_player:
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

        ev         = sum(strategy.get(a, 0) * action_values[a] for a in valid_actions)
        advantages = np.zeros(NUM_ACTIONS)
        valid_mask = np.zeros(NUM_ACTIONS)
        for a in valid_actions:
            advantages[a] = action_values[a] - ev
            valid_mask[a] = 1.0
        trainer.adv_buffers[cp].add(
            features, advantages, trainer.iteration, valid_mask, street=state.street
        )
        return ev

    else:
        strat_target = np.zeros(NUM_ACTIONS)
        valid_mask   = np.zeros(NUM_ACTIONS)
        for a in valid_actions:
            strat_target[a] = strategy.get(a, 0)
            valid_mask[a]   = 1.0
        trainer.strategy_buffer.add(
            features, strat_target, trainer.iteration, valid_mask, street=state.street
        )
        chosen = random.choices(list(strategy.keys()),
                                weights=list(strategy.values()), k=1)[0]
        ns  = state.apply(chosen)
        sub = traverse_coro(trainer, ns, p0_hand, p1_hand, p0_hand5, p1_hand5,
                             community, p0_disc, p1_disc, traversing_player)
        try:
            req = next(sub)
            while True:
                resp = yield req
                req  = sub.send(resp)
        except StopIteration as e:
            return e.value


def run_traversals_batched(trainer, traversals_per_iter: int, traversing_player: int):
    """
    Run N traversals simultaneously.
    Preflop nodes are handled inline (tabular CFR, no yield).
    Only postflop nodes appear in pending → batch by player only.
    """
    r = batch_deal_discard(traversals_per_iter)
    p0h, p1h, p0d, p1d, comms, p0h5, p1h5 = r

    gens    = {}
    pending = {}   # slot → (features, valid_actions, cp, street)

    for i in range(traversals_per_iter):
        g = traverse_coro(
            trainer,
            GameState(),
            list(p0h[i]),  list(p1h[i]),
            list(p0h5[i]), list(p1h5[i]),
            list(comms[i]), list(p0d[i]), list(p1d[i]),
            traversing_player,
        )
        gens[i] = g
        try:
            pending[i] = next(g)
        except StopIteration:
            del gens[i]

    while pending:
        # All pending nodes are postflop → batch by player only
        for p in [0, 1]:
            idxs = [i for i in list(pending.keys()) if pending[i][2] == p]
            if not idxs:
                continue

            feats = np.stack([pending[i][0] for i in idxs])
            x     = torch.tensor(feats, dtype=torch.float32)
            with torch.no_grad():
                adv_batch = trainer.adv_nets[p](x).numpy()

            for j, i in enumerate(idxs):
                _, valid_actions, _, _ = pending.pop(i)
                strategy = _regret_matching(adv_batch[j], valid_actions)
                try:
                    pending[i] = gens[i].send(strategy)
                except StopIteration:
                    del gens[i]

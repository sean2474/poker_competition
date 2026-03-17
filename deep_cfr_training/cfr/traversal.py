"""
External sampling CFR traversal.

traverse_coro: generator that yields (features, valid_actions, cp, street)
               and receives strategy dict via .send().

run_traversals_batched: runs N coroutines simultaneously, routing inference
                        requests to preflop or postflop advantage networks.
"""

import random
import numpy as np
import torch

from game import GameState, state_to_features, evaluate_showdown, batch_deal_discard
from game.constants import NUM_ACTIONS


def _regret_matching(adv_arr, valid_actions: list) -> dict:
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


def traverse_coro(trainer, state, p0_hand, p1_hand, p0_hand5, p1_hand5,
                  community, p0_disc, p1_disc, traversing_player):
    """
    Generator traversal.
    Yields (features, valid_actions, cp, street) for batch inference.
    Receives strategy dict via .send(strategy).
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

    vis_comm = ([], community[:3], community[:4], community[:5])[min(state.street, 3)]
    features = state_to_features(
        hand, vis_comm, state.bets[cp], state.bets[1 - cp],
        state.street, is_bb, my_disc, opp_disc,
        hero_hand5=hand5 if state.street == 0 else None,
        street_bets=state.street_bets,
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
    Run N traversals simultaneously with batch inference.
    Routes to preflop_adv_nets (street=0) or adv_nets (street>0).
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
        # Batch by (player, is_preflop) — 4 possible groups
        for p in [0, 1]:
            for is_pf in [True, False]:
                idxs = [i for i in list(pending.keys())
                        if pending[i][2] == p and (pending[i][3] == 0) == is_pf]
                if not idxs:
                    continue

                feats = np.stack([pending[i][0] for i in idxs])
                x     = torch.tensor(feats, dtype=torch.float32)
                net   = trainer.pf_adv_nets[p] if is_pf else trainer.adv_nets[p]

                with torch.no_grad():
                    adv_batch = net(x).numpy()

                for j, i in enumerate(idxs):
                    _, valid_actions, _, _ = pending.pop(i)
                    strategy = _regret_matching(adv_batch[j], valid_actions)
                    try:
                        pending[i] = gens[i].send(strategy)
                    except StopIteration:
                        del gens[i]

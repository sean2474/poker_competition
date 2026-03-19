"""
strategy/postflop.py — Postflop strategy net inference (Phase 3+).

Uses StrategyNet (77-dim → 8 actions) for flop/turn/river decisions.
State tracking: aggressor, bet counts, to_call needed for 77-dim features.
"""

import numpy as np
import torch
from features import state_to_features
from action import NUM_ACTIONS, map_training_action, valid_training_actions


def postflop_action(obs: dict, strategy_net,
                    my_id: int,
                    my_disc: list, opp_disc: list,
                    aggressor_me: bool, aggressor_opp: bool,
                    n_bets_me: int, n_bets_opp: int,
                    is_bb: bool = False) -> tuple:
    """
    Returns (action_type, raise_amount, 0, 0) for postflop street.

    strategy_net: loaded StrategyNet or None (fallback: call/check).
    my_id: 0 or 1 (which player I am).
    my_disc / opp_disc: [3] discard cards (-1 = unknown).
    aggressor_me/opp: who was last aggressor.
    n_bets_me/opp: bet counts this street.
    is_bb: True if this agent is BB this hand (computed per-hand in player.py).
    """
    assert strategy_net is not None, 'strategy_net must be loaded'

    hand2  = [c for c in obs['my_cards'] if c >= 0][:2]
    comm   = list(obs.get('community_cards', [-1]*5))
    street = int(obs.get('street', 1))
    my_bet = int(obs.get('my_bet', 0))
    opp_bet= int(obs.get('opp_bet', 0))
    to_call = max(opp_bet - my_bet, 0)

    assert len(hand2) == 2, f'Expected 2-card hand, got {len(hand2)}: {hand2}'
    comm5 = (comm + [-1]*5)[:5]

    feat = state_to_features(
        hand2=hand2, board=comm5,
        my_bet=my_bet, opp_bet=opp_bet,
        street=street, is_bb=is_bb,
        my_disc=(my_disc + [-1]*3)[:3],
        opp_disc=(opp_disc + [-1]*3)[:3],
        to_call=to_call,
        n_bets_me=n_bets_me, n_bets_opp=n_bets_opp,
        aggressor_me=aggressor_me, aggressor_opp=aggressor_opp,
    )

    valid_t = valid_training_actions(obs)
    with torch.no_grad():
        logits = strategy_net(
            torch.from_numpy(feat).float().unsqueeze(0)
        ).squeeze(0).numpy()

    mask = np.full(NUM_ACTIONS, -1e9)
    for a in valid_t: mask[a] = logits[a]
    probs = np.exp(mask - mask.max())
    probs = probs / probs.sum()

    chosen = int(np.random.choice(NUM_ACTIONS, p=probs.astype(np.float64)))
    return map_training_action(chosen, obs)

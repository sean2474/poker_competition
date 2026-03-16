"""
CFR+ trainer for the 27-card poker variant.

Architecture:
  - Preflop: canonical 5-card exact hand + action abstraction
  - Discard: oracle solver OUTSIDE CFR (not a game tree node)
  - Post-discard (flop/turn/river): multi-feature bucketed CFR

Produces a strategy table (pickle) for the PlayerAgent.
"""

import random
import pickle
import numpy as np
from typing import Dict, List, Tuple
import time
import os
import sys
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from abstractions.card_utils import (
    DECK_SIZE, get_evaluator, int_to_treys,
)
from abstractions.action_abs import action_to_short
from abstractions.discard_oracle import choose_discard
from abstractions.infoset import preflop_infoset, postdiscard_infoset

MAX_BET = 100
SMALL_BLIND = 1
BIG_BLIND = 2


class GameState:
    """Abstract game state for CFR traversal."""

    __slots__ = [
        'street', 'bets', 'is_terminal', 'winner', 'folded_player',
        'current_player', 'min_raise', 'last_street_bet',
        'street_history', 'full_history', 'last_aggressor',
    ]

    def __init__(self):
        self.street = 0
        self.bets = [SMALL_BLIND, BIG_BLIND]
        self.is_terminal = False
        self.winner = None
        self.folded_player = None
        self.current_player = 0
        self.min_raise = BIG_BLIND
        self.last_street_bet = 0
        self.street_history = ""
        self.full_history = ""
        self.last_aggressor = -1

    def copy(self):
        s = GameState.__new__(GameState)
        s.street = self.street
        s.bets = self.bets[:]
        s.is_terminal = self.is_terminal
        s.winner = self.winner
        s.folded_player = self.folded_player
        s.current_player = self.current_player
        s.min_raise = self.min_raise
        s.last_street_bet = self.last_street_bet
        s.street_history = self.street_history
        s.full_history = self.full_history
        s.last_aggressor = self.last_aggressor
        return s

    def get_action_ctx(self) -> str:
        """Return action context: 'no_bet', 'facing_bet', or 'high_pressure'."""
        cp = self.current_player
        opp = 1 - cp
        to_call = self.bets[opp] - self.bets[cp]
        max_raise = MAX_BET - max(self.bets)
        can_raise = max_raise > 0 and self.min_raise <= max_raise
        if to_call <= 0:
            return "no_bet"
        remaining_cap = MAX_BET - max(self.bets)
        if (remaining_cap > 0 and to_call / remaining_cap > 0.6) or not can_raise:
            return "high_pressure"
        return "facing_bet"

    def get_valid_actions(self) -> List[str]:
        cp = self.current_player
        opp = 1 - cp
        to_call = self.bets[opp] - self.bets[cp]
        max_raise = MAX_BET - max(self.bets)
        can_raise = max_raise > 0 and self.min_raise <= max_raise

        if to_call <= 0:
            actions = ["CHECK"]
            if can_raise:
                actions.extend(["BET_SMALL", "BET_LARGE"])
            return actions
        else:
            remaining_cap = MAX_BET - max(self.bets)
            high_pressure = (remaining_cap > 0 and to_call / remaining_cap > 0.6) or not can_raise
            if high_pressure:
                actions = ["FOLD", "CALL"]
                if can_raise:
                    actions.append("JAM")
                return actions
            else:
                actions = ["FOLD", "CALL"]
                if can_raise:
                    actions.extend(["RAISE_SMALL", "RAISE_LARGE"])
                return actions

    def apply_action(self, action: str) -> 'GameState':
        s = self.copy()
        pot = s.bets[0] + s.bets[1]
        cp = s.current_player
        opp = 1 - cp
        max_raise = MAX_BET - max(s.bets)

        short = action_to_short(action)
        s.street_history += short
        s.full_history += short

        if action == "FOLD":
            s.is_terminal = True
            s.winner = opp
            s.folded_player = cp
            return s

        if action == "CHECK":
            end_street = False
            if s.street == 0 and cp == 1:
                end_street = True
            elif s.street >= 1 and cp == 0:
                end_street = True
            if end_street:
                s._advance_street()
            else:
                s.current_player = opp
            return s

        if action == "CALL":
            s.bets[cp] = s.bets[opp]
            if not (s.street == 0 and cp == 0 and s.bets[cp] == BIG_BLIND):
                s._advance_street()
            else:
                s.current_player = opp
            return s

        # Raise/bet types
        if action == "JAM":
            raise_amount = max_raise
        elif action in ("BET_LARGE", "RAISE_LARGE"):
            spread = max_raise - s.min_raise
            raise_amount = s.min_raise + int(spread * 0.70)
        else:
            spread = max_raise - s.min_raise
            raise_amount = s.min_raise + int(spread * 0.25)

        raise_amount = max(s.min_raise, min(raise_amount, max_raise))
        s.bets[cp] = s.bets[opp] + raise_amount
        raise_so_far = s.bets[opp] - s.last_street_bet
        min_raise_nl = raise_so_far + raise_amount
        s.min_raise = min(min_raise_nl, MAX_BET - max(s.bets))
        s.last_aggressor = cp
        s.current_player = opp
        return s

    def _advance_street(self):
        self.street += 1
        if self.street > 3:
            self.is_terminal = True
            self.winner = -1
        else:
            self.min_raise = BIG_BLIND
            self.last_street_bet = self.bets[0]
            self.current_player = 1
            self.street_history = ""
            self.last_aggressor = -1


class CFRNode:
    __slots__ = ['regret_sum', 'strategy_sum', 'num_actions', 'actions']

    def __init__(self, num_actions: int, actions: list):
        self.num_actions = num_actions
        self.actions = actions
        self.regret_sum = np.zeros(num_actions, dtype=np.float64)
        self.strategy_sum = np.zeros(num_actions, dtype=np.float64)

    def get_strategy(self, reach_weight: float) -> np.ndarray:
        pos_regret = np.maximum(self.regret_sum, 0)
        total = pos_regret.sum()
        if total > 0:
            strat = pos_regret / total
        else:
            strat = np.ones(self.num_actions) / self.num_actions
        self.strategy_sum += reach_weight * strat
        return strat

    def get_average_strategy(self) -> np.ndarray:
        total = self.strategy_sum.sum()
        if total > 0:
            return self.strategy_sum / total
        return np.ones(self.num_actions) / self.num_actions


class CFRTrainer:
    def __init__(self):
        self.nodes: Dict[tuple, CFRNode] = {}
        self.iterations = 0

    def _get_node(self, key: tuple, actions: list) -> CFRNode:
        if key not in self.nodes:
            self.nodes[key] = CFRNode(len(actions), list(actions))
        return self.nodes[key]

    def _deal(self):
        deck = list(range(DECK_SIZE))
        random.shuffle(deck)
        return deck[:5], deck[5:10], deck[10:15]

    def _do_discard(self, hand_5, board_3, opp_disc):
        ki, kj = choose_discard(hand_5, board_3, opp_disc, top_k=3, mc_sims=60)
        kept = [hand_5[ki], hand_5[kj]]
        discarded = [hand_5[k] for k in range(5) if k != ki and k != kj]
        return kept, discarded

    def _make_key(self, state, cp, p_hand, p_hand_5, community, opp_disc, p_disc, ctx_key):
        is_bb = (cp == 1)
        if state.street == 0:
            base = preflop_infoset(p_hand_5, is_bb, state.street_history)
        else:
            hero_agg = (state.last_aggressor == cp)
            villain_agg = (state.last_aggressor == (1 - cp))
            dead = list(p_disc) + list(opp_disc)
            base = postdiscard_infoset(
                state.street, p_hand, community, opp_disc,
                is_bb, hero_agg, villain_agg,
                state.street_history,
                state.bets[cp], state.bets[1 - cp], dead
            )
        return base + (ctx_key,)

    def cfr(self, state, p0_hand, p1_hand, p0_hand_5, p1_hand_5,
            community, p0_disc, p1_disc, reach_0, reach_1):
        """Vanilla CFR+ traversal. Returns (util_p0, util_p1)."""

        if state.is_terminal:
            pot = min(state.bets[0], state.bets[1])
            if state.folded_player is not None:
                return (-pot, pot) if state.folded_player == 0 else (pot, -pot)
            ev = get_evaluator()
            b = [int_to_treys(c) for c in community]
            h0 = [int_to_treys(c) for c in p0_hand]
            h1 = [int_to_treys(c) for c in p1_hand]
            r0, r1 = ev.evaluate(h0, b), ev.evaluate(h1, b)
            if r0 < r1:
                return (pot, -pot)
            elif r1 < r0:
                return (-pot, pot)
            return (0, 0)

        cp = state.current_player
        actions = state.get_valid_actions()
        n = len(actions)

        action_ctx = state.get_action_ctx()
        ctx_key = (action_ctx, n)
        if cp == 0:
            key = self._make_key(state, cp, p0_hand, p0_hand_5, community, p1_disc, p0_disc, ctx_key)
        else:
            key = self._make_key(state, cp, p1_hand, p1_hand_5, community, p0_disc, p1_disc, ctx_key)

        node = self._get_node(key, actions)
        reach = reach_0 if cp == 0 else reach_1
        strategy = node.get_strategy(reach)

        action_utils = np.zeros(n)
        node_util = np.zeros(2)

        for i, action in enumerate(actions):
            ns = state.apply_action(action)
            if cp == 0:
                u0, u1 = self.cfr(ns, p0_hand, p1_hand, p0_hand_5, p1_hand_5,
                                   community, p0_disc, p1_disc,
                                   reach_0 * strategy[i], reach_1)
            else:
                u0, u1 = self.cfr(ns, p0_hand, p1_hand, p0_hand_5, p1_hand_5,
                                   community, p0_disc, p1_disc,
                                   reach_0, reach_1 * strategy[i])
            action_utils[i] = u0 if cp == 0 else u1
            node_util[0] += strategy[i] * u0
            node_util[1] += strategy[i] * u1

        my_util = node_util[cp]
        opp_reach = reach_1 if cp == 0 else reach_0
        for i in range(n):
            node.regret_sum[i] = max(
                node.regret_sum[i] + opp_reach * (action_utils[i] - my_util), 0
            )

        return (node_util[0], node_util[1])

    def train(self, num_iterations: int = 5000):
        pbar = tqdm(range(num_iterations), desc="CFR+", unit="it")
        for i in pbar:
            p0_5, p1_5, community = self._deal()
            board_3 = community[:3]

            p0_hand, p0_disc = self._do_discard(p0_5, board_3, [])
            p1_hand, p1_disc = self._do_discard(p1_5, board_3, p0_disc)

            state = GameState()
            self.cfr(state, p0_hand, p1_hand, p0_5, p1_5,
                     community, p0_disc, p1_disc, 1.0, 1.0)
            self.iterations += 1

            if (i + 1) % 100 == 0:
                pbar.set_postfix(nodes=len(self.nodes))

        print(f"Done: {self.iterations} iters, {len(self.nodes)} nodes")

    def save(self, path: str):
        """Save play-only strategy (average strategy + actions). Used by PlayerAgent."""
        data = {}
        for key, node in self.nodes.items():
            data[key] = {
                'strategy': node.get_average_strategy().tolist(),
                'actions': node.actions,
            }
        payload = {'strategies': data, 'iterations': self.iterations}
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(payload, f)
        sz = os.path.getsize(path) / (1024 * 1024)
        print(f"Saved strategy to {path} ({sz:.2f} MB, {len(data)} nodes)")

    def save_checkpoint(self, path: str):
        """Save full training state (regret_sum + strategy_sum) for resuming."""
        data = {}
        for key, node in self.nodes.items():
            data[key] = {
                'regret_sum': node.regret_sum.tolist(),
                'strategy_sum': node.strategy_sum.tolist(),
                'actions': node.actions,
            }
        payload = {'nodes': data, 'iterations': self.iterations}
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(payload, f)
        sz = os.path.getsize(path) / (1024 * 1024)
        print(f"Saved checkpoint to {path} ({sz:.2f} MB, {len(data)} nodes, {self.iterations} iters)")

    def load_checkpoint(self, path: str):
        """Load training state from checkpoint to resume training."""
        with open(path, 'rb') as f:
            payload = pickle.load(f)
        self.iterations = payload['iterations']
        self.nodes = {}
        for key, nd in payload['nodes'].items():
            actions = nd['actions']
            node = CFRNode(len(actions), actions)
            node.regret_sum = np.array(nd['regret_sum'], dtype=np.float64)
            node.strategy_sum = np.array(nd['strategy_sum'], dtype=np.float64)
            self.nodes[key] = node
        print(f"Loaded checkpoint: {self.iterations} iters, {len(self.nodes)} nodes")

    @staticmethod
    def load(path: str) -> dict:
        """Load play-only strategy (used by PlayerAgent)."""
        with open(path, 'rb') as f:
            return pickle.load(f)

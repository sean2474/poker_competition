"""
Fallback-only version of PlayerAgent — no CFR lookup, pure MC equity + discard-aware.
Used for A/B testing CFR value.
"""
import os
import sys
import random
import struct

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "submission"))

from agents.agent import Agent
from gym_env import PokerEnv

from abstractions.discard_oracle import choose_discard, estimate_opp_keep_weights
from abstractions.action_abs import action_to_short, get_action_context
from abstractions.card_utils import card_rank, ACE_RANK_IDX, get_evaluator, int_to_treys, ALL_CARDS
from itertools import combinations

_FOLD = 0
_RAISE = 1
_CHECK = 2
_CALL = 3
_DISCARD = 4


class FallbackOnlyAgent(Agent):
    def __init__(self, stream: bool = True):
        super().__init__(stream)
        self.current_hand = -1
        self.my_hand_5 = []
        self.my_hand_2 = []
        self.my_discards = []
        self.opp_discards = []

    def __name__(self):
        return "FallbackOnlyAgent"

    def _reset_hand(self, hand_number):
        if hand_number != self.current_hand:
            self.current_hand = hand_number
            self.my_hand_5 = []
            self.my_hand_2 = []
            self.my_discards = []
            self.opp_discards = []

    def _fallback_action(self, observation):
        valid = observation["valid_actions"]
        my_cards = [c for c in observation["my_cards"] if c != -1]
        community = [c for c in observation["community_cards"] if c != -1]
        my_bet = observation["my_bet"]
        opp_bet = observation["opp_bet"]
        min_raise = observation["min_raise"]
        max_raise = observation["max_raise"]

        hand = self.my_hand_2 if self.my_hand_2 else my_cards[:2]
        to_call = opp_bet - my_bet
        pot = my_bet + opp_bet
        pot_odds = to_call / (to_call + pot) if to_call > 0 and pot > 0 else 0

        # MC equity (discard-aware)
        if len(hand) == 2 and community:
            ev = get_evaluator()
            dead_set = set(hand) | set(community)
            for c in self.my_discards:
                if c >= 0: dead_set.add(c)
            for c in self.opp_discards:
                if c >= 0: dead_set.add(c)
            remaining = [c for c in ALL_CARDS if c not in dead_set]
            board_need = 5 - len(community)

            if len(remaining) >= board_need + 2:
                my_h = [int_to_treys(c) for c in hand]
                opp_combos = list(combinations(remaining, 2))
                board3 = community[:3] if len(community) >= 3 else community
                opp_weights = None
                if self.opp_discards and len(self.opp_discards) == 3 and all(c >= 0 for c in self.opp_discards):
                    w_dict = estimate_opp_keep_weights(self.opp_discards, board3, remaining)
                    if w_dict:
                        opp_weights = [w_dict.get((c1, c2), 0.05) for c1, c2 in opp_combos]
                        w_total = sum(opp_weights)
                        if w_total > 0:
                            opp_weights = [w / w_total for w in opp_weights]
                        else:
                            opp_weights = None

                wins = ties = total = 0
                for _ in range(200):
                    if opp_weights:
                        opp_idx = random.choices(range(len(opp_combos)), weights=opp_weights, k=1)[0]
                        opp_pair = list(opp_combos[opp_idx])
                    else:
                        opp_pair = list(random.sample(remaining, 2))
                    if board_need > 0:
                        board_remaining = [c for c in remaining if c not in opp_pair]
                        if len(board_remaining) < board_need: continue
                        extra = random.sample(board_remaining, board_need)
                        full_board = community + extra
                    else:
                        full_board = community
                    b = [int_to_treys(c) for c in full_board]
                    opp_h = [int_to_treys(c) for c in opp_pair]
                    mr = ev.evaluate(my_h, b)
                    opr = ev.evaluate(opp_h, b)
                    if mr < opr: wins += 1
                    elif mr == opr: ties += 1
                    total += 1
                equity = (wins + 0.5 * ties) / total if total > 0 else 0.5
            else:
                equity = 0.5
        elif len(hand) == 2:
            r0, r1 = card_rank(hand[0]), card_rank(hand[1])
            pp = (r0 == r1)
            has_a = (r0 == ACE_RANK_IDX or r1 == ACE_RANK_IDX)
            high = max(r0, r1)
            equity = 0.3
            if pp: equity = 0.55 + high * 0.02
            elif has_a: equity = 0.45 + min(r0, r1) * 0.02
            else: equity = 0.25 + high * 0.03
        else:
            ranks = sorted([card_rank(c) for c in my_cards], reverse=True)
            pp = len(set(ranks)) < len(ranks)
            has_a = ACE_RANK_IDX in ranks
            equity = 0.3
            if pp: equity = 0.55
            elif has_a: equity = 0.45
            elif max(ranks) >= 6: equity = 0.35

        if equity > 0.75 and valid[_RAISE]:
            amount = max(min_raise, min(int(pot * 0.75), max_raise))
            return (_RAISE, amount, 0, 0)
        elif equity > 0.55 and valid[_RAISE] and to_call <= 0:
            amount = max(min_raise, min(int(pot * 0.5), max_raise))
            return (_RAISE, amount, 0, 0)
        elif equity >= pot_odds and equity > 0.35 and valid[_CALL]:
            return (_CALL, 0, 0, 0)
        elif valid[_CHECK]:
            return (_CHECK, 0, 0, 0)
        elif equity >= pot_odds and valid[_CALL]:
            return (_CALL, 0, 0, 0)
        else:
            return (_FOLD, 0, 0, 0)

    def act(self, observation, reward, terminated, truncated, info):
        hand_number = info.get('hand_number', -1)
        self._reset_hand(hand_number)

        my_cards = [c for c in observation["my_cards"] if c != -1]
        if len(my_cards) == 5 and not self.my_hand_5:
            self.my_hand_5 = list(my_cards)
        if len(my_cards) == 2 and not self.my_hand_2:
            self.my_hand_2 = list(my_cards)

        opp_disc = [c for c in observation.get("opp_discarded_cards", []) if c != -1]
        if opp_disc and not self.opp_discards:
            self.opp_discards = list(opp_disc)
        my_disc = [c for c in observation.get("my_discarded_cards", []) if c != -1]
        if my_disc and not self.my_discards:
            self.my_discards = list(my_disc)

        if observation["valid_actions"][_DISCARD]:
            community = [c for c in observation["community_cards"] if c != -1]
            ki, kj = choose_discard(my_cards, community, opp_disc, top_k=3, mc_sims=150)
            self.my_hand_2 = [my_cards[ki], my_cards[kj]]
            self.my_discards = [my_cards[k] for k in range(5) if k != ki and k != kj]
            return (_DISCARD, 0, ki, kj)

        return self._fallback_action(observation)

    def observe(self, observation, reward, terminated, truncated, info):
        pass

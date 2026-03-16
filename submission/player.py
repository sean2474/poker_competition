import os
import pickle
import random

from agents.agent import Agent
from gym_env import PokerEnv

from submission.abstractions.discard_oracle import choose_discard
from submission.abstractions.infoset import build_infoset_key
from submission.abstractions.action_abs import (
    get_valid_abstract_actions, abstract_to_concrete,
    concrete_to_abstract, action_to_short,
)
from submission.abstractions.hand_bucket import _made_tier_from_structure, _draw_tier


STRATEGY_PATH = os.path.join(os.path.dirname(__file__), "data", "strategy.pkl")

# ActionType enum values
_FOLD = 0
_RAISE = 1
_CHECK = 2
_CALL = 3
_DISCARD = 4


class PlayerAgent(Agent):
    def __init__(self, stream: bool = True):
        super().__init__(stream)
        self.action_types = PokerEnv.ActionType

        # Load CFR strategy table
        self.strategy_data = None
        if os.path.exists(STRATEGY_PATH):
            try:
                with open(STRATEGY_PATH, 'rb') as f:
                    self.strategy_data = pickle.load(f)
                self.logger.info(
                    f"Loaded CFR strategy: {self.strategy_data['iterations']} iters, "
                    f"{len(self.strategy_data['strategies'])} nodes"
                )
            except Exception as e:
                self.logger.warning(f"Failed to load strategy: {e}")

        # Per-hand tracking
        self.current_hand = -1
        self.street_history = ""
        self.last_street = -1
        self.hero_last_raiser = False
        self.villain_last_raiser = False
        self.my_hand_5 = []
        self.my_hand_2 = []
        self.my_discards = []
        self.opp_discards = []

    def __name__(self):
        return "PlayerAgent"

    def _reset_hand(self, hand_number):
        if hand_number != self.current_hand:
            self.current_hand = hand_number
            self.street_history = ""
            self.last_street = -1
            self.hero_last_raiser = False
            self.villain_last_raiser = False
            self.my_hand_5 = []
            self.my_hand_2 = []
            self.my_discards = []
            self.opp_discards = []

    def _check_street_change(self, street):
        if street != self.last_street:
            self.street_history = ""
            if self.last_street >= 0:
                pass  # initiative carries over via hero/villain_last_raiser
            self.last_street = street

    def _is_bb(self, observation):
        return observation.get("blind_position", 0) == 1

    def _cfr_lookup(self, observation) -> tuple:
        """Try CFR strategy lookup. Returns concrete action or None."""
        if not self.strategy_data:
            return None

        my_cards = [c for c in observation["my_cards"] if c != -1]
        community = [c for c in observation["community_cards"] if c != -1]
        is_bb = self._is_bb(observation)
        street = observation["street"]

        # Use stored 5-card hand for preflop, 2-card for post-discard
        if street == 0:
            hand_for_key = self.my_hand_5 if self.my_hand_5 else my_cards
        else:
            hand_for_key = self.my_hand_2 if self.my_hand_2 else my_cards[:2]

        # Get valid abstract actions
        valid = observation["valid_actions"]
        min_raise = observation["min_raise"]
        max_raise = observation["max_raise"]
        my_bet = observation["my_bet"]
        opp_bet = observation["opp_bet"]
        valid_abs = get_valid_abstract_actions(valid, my_bet, opp_bet, min_raise, max_raise)
        n = len(valid_abs)

        # Build infoset key
        key = build_infoset_key(
            observation, hand_for_key, is_bb,
            self.hero_last_raiser, self.villain_last_raiser,
            self.street_history,
            self.my_discards, self.opp_discards,
        )
        # Append num_actions to match trainer key format
        key = key + (n,)

        if key not in self.strategy_data['strategies']:
            return None

        node = self.strategy_data['strategies'][key]
        stored_actions = node['actions']
        strategy = node['strategy']

        # Verify action list matches
        if len(stored_actions) != n or stored_actions != valid_abs:
            return None

        # Sample from strategy
        total = sum(strategy)
        if total > 0:
            probs = [p / total for p in strategy]
        else:
            probs = [1.0 / n] * n

        chosen_idx = random.choices(range(n), weights=probs, k=1)[0]
        chosen_abs = valid_abs[chosen_idx]

        # Convert to concrete
        concrete = abstract_to_concrete(chosen_abs, min_raise, max_raise, my_bet, opp_bet)

        # Update history
        self.street_history += action_to_short(chosen_abs)
        if chosen_abs in ("BET_SMALL", "BET_LARGE", "RAISE_SMALL", "RAISE_LARGE", "JAM"):
            self.hero_last_raiser = True
            self.villain_last_raiser = False

        return concrete

    def _fallback_action(self, observation) -> tuple:
        """Deterministic fallback when CFR has no entry."""
        valid = observation["valid_actions"]
        my_cards = [c for c in observation["my_cards"] if c != -1]
        community = [c for c in observation["community_cards"] if c != -1]
        my_bet = observation["my_bet"]
        opp_bet = observation["opp_bet"]
        min_raise = observation["min_raise"]
        max_raise = observation["max_raise"]
        street = observation["street"]

        hand = self.my_hand_2 if self.my_hand_2 else my_cards[:2]

        to_call = opp_bet - my_bet
        pot = my_bet + opp_bet

        # ─── Preflop (no community cards): use raw card features ───
        if not community or street == 0:
            from submission.abstractions.card_utils import card_rank, card_suit, ACE_RANK_IDX
            ranks = sorted([card_rank(c) for c in my_cards], reverse=True)
            has_pocket_pair = len(set(ranks)) < len(ranks)
            has_ace = ACE_RANK_IDX in ranks
            high = max(ranks)
            # Preflop strength: rough score 0-4
            pf_str = 0
            if has_pocket_pair:
                pf_str = 2
                if high >= 5:   # pair of 7+
                    pf_str = 3
            elif has_ace:
                pf_str = 2
                if ranks[1] >= 5:  # Ace + 7+
                    pf_str = 3
            elif high >= 6:  # 8+
                pf_str = 1

            if pf_str >= 3 and valid[_RAISE]:
                amount = max(min_raise, min(pot, max_raise))
                self.street_history += action_to_short("BET_LARGE")
                self.hero_last_raiser = True
                self.villain_last_raiser = False
                return (_RAISE, amount, 0, 0)
            if pf_str >= 2:
                if valid[_CALL]:
                    self.street_history += action_to_short("CALL")
                    return (_CALL, 0, 0, 0)
                if valid[_RAISE]:
                    amount = max(min_raise, min(pot // 2, max_raise))
                    self.street_history += action_to_short("BET_SMALL")
                    self.hero_last_raiser = True
                    return (_RAISE, amount, 0, 0)
            if pf_str >= 1:
                if valid[_CHECK]:
                    self.street_history += action_to_short("CHECK")
                    return (_CHECK, 0, 0, 0)
                if to_call <= 4 and valid[_CALL]:
                    self.street_history += action_to_short("CALL")
                    return (_CALL, 0, 0, 0)
            if valid[_CHECK]:
                self.street_history += action_to_short("CHECK")
                return (_CHECK, 0, 0, 0)
            self.street_history += action_to_short("FOLD")
            return (_FOLD, 0, 0, 0)

        # ─── Post-flop: deterministic hand strength ───
        made = _made_tier_from_structure(hand, community)
        draw = _draw_tier(hand, community)

        # Strong hand (two pair+): raise or call anything
        if made >= 3:
            if valid[_RAISE]:
                amount = max(min_raise, min(pot, max_raise))
                self.street_history += action_to_short("BET_LARGE")
                self.hero_last_raiser = True
                self.villain_last_raiser = False
                return (_RAISE, amount, 0, 0)
            if valid[_CALL]:
                self.street_history += action_to_short("CALL")
                return (_CALL, 0, 0, 0)

        # Top pair / overpair: bet, or call even large bets
        if made >= 2:
            if to_call <= 0 and valid[_RAISE]:
                amount = max(min_raise, min(pot // 2, max_raise))
                self.street_history += action_to_short("BET_SMALL")
                self.hero_last_raiser = True
                self.villain_last_raiser = False
                return (_RAISE, amount, 0, 0)
            if valid[_CALL]:
                self.street_history += action_to_short("CALL")
                return (_CALL, 0, 0, 0)
            if valid[_CHECK]:
                self.street_history += action_to_short("CHECK")
                return (_CHECK, 0, 0, 0)

        # Draw: call if pot odds decent, check otherwise
        if draw >= 2:
            pot_odds = to_call / (to_call + pot) if to_call > 0 and pot > 0 else 0
            if pot_odds < 0.35 and valid[_CALL]:
                self.street_history += action_to_short("CALL")
                return (_CALL, 0, 0, 0)
            if valid[_CHECK]:
                self.street_history += action_to_short("CHECK")
                return (_CHECK, 0, 0, 0)

        # Weak pair: check, or call small-to-medium bets
        if made >= 1:
            if valid[_CHECK]:
                self.street_history += action_to_short("CHECK")
                return (_CHECK, 0, 0, 0)
            if to_call <= pot * 0.35 and valid[_CALL]:
                self.street_history += action_to_short("CALL")
                return (_CALL, 0, 0, 0)

        # Air: check or fold
        if valid[_CHECK]:
            self.street_history += action_to_short("CHECK")
            return (_CHECK, 0, 0, 0)

        self.street_history += action_to_short("FOLD")
        return (_FOLD, 0, 0, 0)

    def act(self, observation, reward, terminated, truncated, info):
        hand_number = info.get('hand_number', -1)
        self._reset_hand(hand_number)
        self._check_street_change(observation["street"])

        valid_actions = observation["valid_actions"]

        # Track our cards
        my_cards = [c for c in observation["my_cards"] if c != -1]
        if len(my_cards) == 5 and not self.my_hand_5:
            self.my_hand_5 = list(my_cards)
        if len(my_cards) == 2 and not self.my_hand_2:
            self.my_hand_2 = list(my_cards)

        # Track discards
        opp_disc = [c for c in observation.get("opp_discarded_cards", []) if c != -1]
        if opp_disc and not self.opp_discards:
            self.opp_discards = list(opp_disc)
        my_disc = [c for c in observation.get("my_discarded_cards", []) if c != -1]
        if my_disc and not self.my_discards:
            self.my_discards = list(my_disc)

        # ─── Discard phase ───
        if valid_actions[_DISCARD]:
            community = [c for c in observation["community_cards"] if c != -1]
            ki, kj = choose_discard(my_cards, community, opp_disc, top_k=3, mc_sims=150)
            self.my_hand_2 = [my_cards[ki], my_cards[kj]]
            self.my_discards = [my_cards[k] for k in range(5) if k != ki and k != kj]
            return (_DISCARD, 0, ki, kj)

        # ─── Betting phase: try CFR, fallback to heuristic ───
        result = self._cfr_lookup(observation)
        if result is not None:
            return result
        return self._fallback_action(observation)

    def observe(self, observation, reward, terminated, truncated, info):
        """Track opponent actions to maintain history."""
        opp_action = observation.get("opp_last_action", "None")
        if opp_action and opp_action != "None":
            # Map engine action names to our short codes
            if opp_action == "FOLD":
                self.street_history += "F"
            elif opp_action == "CHECK":
                self.street_history += "K"
            elif opp_action == "CALL":
                self.street_history += "C"
            elif opp_action == "RAISE":
                self.street_history += "R"
                self.villain_last_raiser = True
                self.hero_last_raiser = False


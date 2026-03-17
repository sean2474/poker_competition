"""
Game environment wrapper for Deep CFR training.

Features are raw card values + hand strength + draw potential + 
opponent range estimation from discards.
No abstraction needed — the neural network learns its own.
"""

import sys
import os
import random
import math
import numpy as np
from itertools import combinations

# Try both possible submission folder names
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ["submission", "submission_mccfr"]:
    _p = os.path.join(_parent, _sub)
    if os.path.isdir(os.path.join(_p, "abstractions")):
        sys.path.insert(0, _p)
        break

from abstractions.card_utils import (
    card_rank, card_suit, NUM_RANKS, NUM_SUITS, DECK_SIZE, ALL_CARDS,
    get_evaluator, int_to_treys, KEEP_PAIRS,
)

MAX_BET = 100
SMALL_BLIND = 1
BIG_BLIND = 2

# Actions
A_FOLD = 0
A_CALL = 1
A_CHECK = 2
A_BET_SMALL = 3
A_BET_LARGE = 4
A_RAISE_SMALL = 5
A_RAISE_LARGE = 6
NUM_ACTIONS = 7

_evaluator = None
def _get_eval():
    global _evaluator
    if _evaluator is None:
        _evaluator = get_evaluator()
    return _evaluator


# ─── Feature Extraction ───

def card_to_features(card):
    """Card → [rank_norm, suit_onehot×3]. Returns 4 floats."""
    if card < 0:
        return [0.0, 0.0, 0.0, 0.0]
    r = card_rank(card) / (NUM_RANKS - 1)
    s = card_suit(card)
    return [r, float(s == 0), float(s == 1), float(s == 2)]


def hand_strength_features(hero_hand, community):
    """
    Fast deterministic hand strength features (no MC — network learns equity).
    Returns 6 features.
    """
    if len(hero_hand) < 2:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    r0, r1 = card_rank(hero_hand[0]), card_rank(hero_hand[1])
    s0, s1 = card_suit(hero_hand[0]), card_suit(hero_hand[1])
    
    # 1. High card value
    high = max(r0, r1) / (NUM_RANKS - 1)
    
    # 2. Pocket pair
    is_pair = 1.0 if r0 == r1 else 0.0
    
    # 3. Suited
    is_suited = 1.0 if s0 == s1 else 0.0
    
    # 4. Flush draw potential
    flush_draw = 0.0
    if community:
        suit_counts = [0, 0, 0]
        for c in list(hero_hand) + list(community):
            if c >= 0:
                suit_counts[card_suit(c)] += 1
        max_suit = max(suit_counts)
        flush_draw = 1.0 if max_suit >= 4 else (0.5 if max_suit >= 3 else 0.0)
    
    # 5. Connectedness
    gap = abs(r0 - r1)
    connected = 1.0 if gap <= 1 else (0.5 if gap <= 3 else 0.0)
    
    # 6. Board pairing (do we hit the board?)
    board_hit = 0.0
    if community:
        board_ranks = [card_rank(c) for c in community if c >= 0]
        if r0 in board_ranks or r1 in board_ranks:
            board_hit = 0.5
            if r0 in board_ranks and r1 in board_ranks:
                board_hit = 1.0
            elif max(r0, r1) == max(board_ranks, default=-1):
                board_hit = 0.75  # top pair
    
    return [high, is_pair, is_suited, flush_draw, connected, board_hit]


def opp_range_features(opp_discards, community):
    """
    Estimate opponent's hand range from their discards using gaussian-like weighting.
    
    Idea: opponent discarded 3 cards. From this we can infer:
      - What ranks they kept (likely high cards, pairs)
      - What suits they kept (likely suited cards matching board)
      - Their approximate hand strength distribution
    
    Returns ~6 features representing estimated opponent range.
    """
    if not opp_discards or all(c < 0 for c in opp_discards):
        return [0.5, 0.5, 0.5, 0.0, 0.0, 0.0]
    
    disc = [c for c in opp_discards if c >= 0]
    if not disc:
        return [0.5, 0.5, 0.5, 0.0, 0.0, 0.0]
    
    feats = []
    
    disc_ranks = [card_rank(c) for c in disc]
    disc_suits = [card_suit(c) for c in disc]
    
    # 1. Average discarded rank (high = they threw away high cards = weak hand likely)
    avg_disc_rank = sum(disc_ranks) / len(disc_ranks) / (NUM_RANKS - 1)
    feats.append(avg_disc_rank)
    
    # 2. Max discarded rank (threw away ace? probably not a strong ace hand)
    max_disc_rank = max(disc_ranks) / (NUM_RANKS - 1)
    feats.append(max_disc_rank)
    
    # 3. Discarded pair? (if they threw a pair, they probably have something better)
    disc_has_pair = len(disc_ranks) != len(set(disc_ranks))
    feats.append(1.0 if disc_has_pair else 0.0)
    
    # 4. Suit concentration in discards (if all same suit discarded, opp doesn't have that suit)
    suit_counts = [0, 0, 0]
    for s in disc_suits:
        suit_counts[s] += 1
    max_suit_disc = max(suit_counts)
    feats.append(max_suit_disc / 3.0)
    
    # 5. Board suit match — did opp discard cards matching dominant board suit?
    if community:
        board_suits = [card_suit(c) for c in community if c >= 0]
        if board_suits:
            dominant_suit = max(set(board_suits), key=board_suits.count)
            disc_matching = sum(1 for s in disc_suits if s == dominant_suit)
            feats.append(disc_matching / 3.0)
        else:
            feats.append(0.0)
    else:
        feats.append(0.0)
    
    # 6. Gaussian estimate: opp likely kept cards that are "distant" from discards
    # Higher rank distance from discards = more likely kept
    avg_kept_rank_est = 1.0 - avg_disc_rank  # inverse: if discarded low, kept high
    feats.append(avg_kept_rank_est)
    
    return feats


def state_to_features(hero_hand, community, my_bet, opp_bet, street, is_bb,
                       my_discards=None, opp_discards=None, pot=None,
                       hero_hand5=None):
    """
    Convert game state to feature vector for neural network.
    
    Features (total = 85):
      - hero hand: 2 cards × 4 = 8 (or 5 cards × 4 = 20 for preflop)
      - community: 5 cards × 4 = 20
      - my_discards: 3 × 4 = 12
      - opp_discards: 3 × 4 = 12
      - street one-hot: 4
      - position: 1
      - bet info: 4
      - hand strength features: 6
      - opp range features: 6
    Total: 20 + 20 + 12 + 12 + 4 + 1 + 4 + 6 + 6 = 85
    """
    feats = []
    
    # Hero hand — preflop uses 5 cards, post-discard uses 2
    if street == 0 and hero_hand5 is not None and len(hero_hand5) == 5:
        for i in range(5):
            feats.extend(card_to_features(hero_hand5[i]))
    else:
        for i in range(2):
            if i < len(hero_hand):
                feats.extend(card_to_features(hero_hand[i]))
            else:
                feats.extend([0.0] * 4)
        feats.extend([0.0] * 12)  # pad to 20
    
    # Community cards (5 slots)
    for i in range(5):
        if community and i < len(community) and community[i] >= 0:
            feats.extend(card_to_features(community[i]))
        else:
            feats.extend([0.0] * 4)
    
    # My discards (3 slots)
    if my_discards:
        for i in range(3):
            if i < len(my_discards) and my_discards[i] >= 0:
                feats.extend(card_to_features(my_discards[i]))
            else:
                feats.extend([0.0] * 4)
    else:
        feats.extend([0.0] * 12)
    
    # Opp discards (3 slots)
    if opp_discards:
        for i in range(3):
            if i < len(opp_discards) and opp_discards[i] >= 0:
                feats.extend(card_to_features(opp_discards[i]))
            else:
                feats.extend([0.0] * 4)
    else:
        feats.extend([0.0] * 12)
    
    # Street one-hot (4)
    for s in range(4):
        feats.append(1.0 if street == s else 0.0)
    
    # Position
    feats.append(1.0 if is_bb else 0.0)
    
    # Bet info (4)
    if pot is None:
        pot = my_bet + opp_bet
    feats.append(my_bet / MAX_BET)
    feats.append(opp_bet / MAX_BET)
    feats.append(pot / (2 * MAX_BET))
    feats.append(max(opp_bet - my_bet, 0) / MAX_BET)
    
    # Hand strength features (6)
    visible_community = [c for c in (community or []) if c >= 0]
    if street > 0 and len(hero_hand) >= 2:
        feats.extend(hand_strength_features(hero_hand, visible_community))
    else:
        # Preflop: basic hand features
        if hero_hand5 and len(hero_hand5) == 5:
            ranks = sorted([card_rank(c) for c in hero_hand5], reverse=True)
            feats.append(max(ranks) / (NUM_RANKS - 1))  # high card
            feats.append(1.0 if len(set(ranks)) < len(ranks) else 0.0)  # has pair
            suits = [card_suit(c) for c in hero_hand5]
            feats.append(max(suits.count(s) for s in set(suits)) / 5.0)  # suit concentration
            feats.extend([0.0, 0.0, 0.0])  # pad
        else:
            feats.extend([0.5, 0.5, 0.0, 0.0, 0.0, 0.0])
    
    # Opp range features (6)
    feats.extend(opp_range_features(opp_discards, visible_community))
    
    return np.array(feats, dtype=np.float32)


FEATURE_DIM = 20 + 20 + 12 + 12 + 4 + 1 + 4 + 6 + 6  # = 85


# ─── Game State ───

class GameState:
    """Game state for Deep CFR training. Includes preflop."""
    
    def __init__(self):
        self.street = 0
        self.bets = [SMALL_BLIND, BIG_BLIND]  # SB=1, BB=2
        self.current_player = 0  # SB acts first preflop
        self.is_terminal = False
        self.folded_player = -1
        self.min_raise = BIG_BLIND
        self.history = []
        self.last_street_bet = 0
        self.num_actions_this_street = 0
    
    def copy(self):
        s = GameState()
        s.street = self.street
        s.bets = list(self.bets)
        s.current_player = self.current_player
        s.is_terminal = self.is_terminal
        s.folded_player = self.folded_player
        s.min_raise = self.min_raise
        s.history = list(self.history)
        s.last_street_bet = self.last_street_bet
        s.num_actions_this_street = self.num_actions_this_street
        return s
    
    def get_valid_actions(self):
        cp = self.current_player
        opp = 1 - cp
        to_call = self.bets[opp] - self.bets[cp]
        max_raise = MAX_BET - max(self.bets)
        can_raise = max_raise > 0 and self.min_raise <= max_raise
        
        actions = []
        if to_call > 0:
            actions.append(A_FOLD)
            actions.append(A_CALL)
            if can_raise:
                actions.append(A_RAISE_SMALL)
                actions.append(A_RAISE_LARGE)
        else:
            actions.append(A_CHECK)
            if can_raise:
                actions.append(A_BET_SMALL)
                actions.append(A_BET_LARGE)
        return actions
    
    def apply(self, action):
        s = self.copy()
        cp = s.current_player
        opp = 1 - cp
        max_raise = MAX_BET - max(s.bets)
        
        s.history.append((cp, action))
        
        if action == A_FOLD:
            s.is_terminal = True
            s.folded_player = cp
            return s
        
        if action == A_CHECK:
            s.num_actions_this_street += 1
            # Both players checked → advance street
            # This handles: BB check after SB limp, and check-check on any street
            if s.num_actions_this_street >= 2 and s.bets[0] == s.bets[1]:
                s._advance_street()
            else:
                s.current_player = opp
            return s
        
        if action == A_CALL:
            s.bets[cp] = s.bets[opp]
            s.num_actions_this_street += 1
            if not (s.street == 0 and cp == 0 and s.bets[cp] == BIG_BLIND):
                s._advance_street()
            else:
                s.current_player = opp
            return s
        
        # Raise/bet
        s.num_actions_this_street += 1
        spread = max_raise - s.min_raise
        if action in (A_BET_SMALL, A_RAISE_SMALL):
            raise_amt = s.min_raise + int(spread * 0.25)
        else:
            raise_amt = s.min_raise + int(spread * 0.70)
        
        raise_amt = max(s.min_raise, min(raise_amt, max_raise))
        s.bets[cp] = s.bets[opp] + raise_amt
        s.min_raise = max(raise_amt, s.min_raise)
        s.current_player = opp
        return s
    
    def _advance_street(self):
        if self.street >= 3:
            self.is_terminal = True
        else:
            self.street += 1
            # Post-flop: BB (player 1) acts first. Preflop: SB (player 0) acts first.
            self.current_player = 1 if self.street >= 1 else 0
            self.last_street_bet = max(self.bets)
            self.min_raise = BIG_BLIND
            self.num_actions_this_street = 0


# ─── Utilities ───

def fast_discard(hand5, board3, temperature=0.05):
    """
    Softmax temperature discard selection.
    
    Low temperature → mostly optimal, slight randomness for:
      1. Training diversity (CFR sees varied hand distributions)
      2. Information hiding (opponent can't assume we always pick max EV)
    
    temperature=0 → deterministic max (pure exploitation)
    temperature=0.05 → nearly deterministic, rare suboptimal picks
    """
    from abstractions.discard_oracle import _fast_score
    scores = []
    pairs = []
    for i, j in KEEP_PAIRS:
        keep = [hand5[i], hand5[j]]
        score = _fast_score(keep, board3)
        scores.append(score)
        pairs.append((i, j))
    
    if temperature <= 0:
        best_idx = max(range(len(scores)), key=lambda k: scores[k])
        return pairs[best_idx]
    
    scores = np.array(scores)
    # Normalize to prevent overflow
    scores = scores - scores.max()
    probs = np.exp(scores / temperature)
    probs /= probs.sum()
    
    chosen_idx = np.random.choice(len(pairs), p=probs)
    return pairs[chosen_idx]


def deal_game():
    """Deal a complete game: return (p0_hand5, p1_hand5, community5)."""
    deck = list(ALL_CARDS)
    random.shuffle(deck)
    return deck[:5], deck[5:10], deck[10:15]


def evaluate_showdown(p0_hand, p1_hand, community):
    """Return +1 if p0 wins, -1 if p1 wins, 0 if tie."""
    ev = _get_eval()
    h0 = [int_to_treys(c) for c in p0_hand]
    h1 = [int_to_treys(c) for c in p1_hand]
    b = [int_to_treys(c) for c in community]
    r0 = ev.evaluate(h0, b)
    r1 = ev.evaluate(h1, b)
    if r0 < r1: return 1
    if r0 > r1: return -1
    return 0

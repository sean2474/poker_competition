import os
import sys
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.agent import Agent
from gym_env import PokerEnv

# ─── Constants (must match deep_cfr_training exactly) ─────────────────────────
NUM_RANKS  = 9      # 2–9, A
NUM_SUITS  = 3      # d, h, s
MAX_BET    = 100
SMALL_BLIND = 1
BIG_BLIND   = 2
FEATURE_DIM = 93
NUM_ACTIONS = 8

# Deep CFR action IDs
A_FOLD       = 0
A_CALL       = 1
A_CHECK      = 2
A_BET_SMALL  = 3
A_BET_LARGE  = 4
A_RAISE_SMALL = 5
A_RAISE_LARGE = 6
A_BET_POT    = 7

# Game action IDs
_FOLD    = 0
_RAISE   = 1
_CHECK   = 2
_CALL    = 3
_DISCARD = 4

DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")
MODEL_PATH = os.path.join(DATA_DIR, "deep_cfr_strategy.pt")


# ─── Pure Python feature extraction (matches game_state.h exactly) ────────────

def card_rank(c: int) -> int:
    return c % NUM_RANKS

def card_suit(c: int) -> int:
    return c // NUM_RANKS

def card_features(card: int) -> list:
    if card < 0:
        return [0.0, 0.0, 0.0, 0.0]
    r = card_rank(card)
    s = card_suit(card)
    return [r / (NUM_RANKS - 1), float(s == 0), float(s == 1), float(s == 2)]

def hand_strength_features(hand2: list, community: list) -> list:
    if len(hand2) < 2 or hand2[0] < 0 or hand2[1] < 0:
        return [0.0] * 6
    r0, r1 = card_rank(hand2[0]), card_rank(hand2[1])
    s0, s1 = card_suit(hand2[0]), card_suit(hand2[1])
    out = [0.0] * 6

    out[0] = max(r0, r1) / (NUM_RANKS - 1)     # high card
    out[1] = float(r0 == r1)                    # pocket pair
    out[2] = float(s0 == s1)                    # suited

    n_comm = len([c for c in community if c >= 0])
    if n_comm > 0:
        suit_counts = [0, 0, 0]
        suit_counts[s0] += 1; suit_counts[s1] += 1
        for c in community:
            if c >= 0:
                suit_counts[card_suit(c)] += 1
        mx = max(suit_counts)
        out[3] = 1.0 if mx >= 4 else (0.5 if mx >= 3 else 0.0)  # flush draw
    else:
        out[3] = 0.0

    gap = abs(r0 - r1)
    out[4] = 1.0 if gap <= 1 else (0.5 if gap <= 3 else 0.0)    # connectedness

    if n_comm > 0:
        comm_ranks = [card_rank(c) for c in community if c >= 0]
        max_board = max(comm_ranks) if comm_ranks else -1
        hit0 = r0 in comm_ranks
        hit1 = r1 in comm_ranks
        if hit0 and hit1:
            out[5] = 1.0
        elif hit0 or hit1:
            out[5] = 0.75 if max(r0, r1) == max_board else 0.5
        else:
            out[5] = 0.0
    else:
        out[5] = 0.0

    return out

def opp_range_features(opp_disc: list, community: list) -> list:
    valid = [c for c in opp_disc if c >= 0]
    if not valid:
        return [0.5, 0.5, 0.5, 0.0, 0.0, 0.5]

    disc_ranks = [card_rank(c) for c in valid]
    disc_suits = [card_suit(c) for c in valid]
    nd = len(valid)
    out = [0.0] * 6

    avg = sum(disc_ranks) / nd / (NUM_RANKS - 1)
    out[0] = avg                                          # avg discarded rank
    out[1] = max(disc_ranks) / (NUM_RANKS - 1)           # max discarded rank
    # discarded pair
    out[2] = float(len(disc_ranks) != len(set(disc_ranks)))
    sc = [0, 0, 0]
    for s in disc_suits:
        sc[s] += 1
    out[3] = max(sc) / 3.0                               # suit concentration

    n_comm = len([c for c in community if c >= 0])
    if n_comm > 0:
        board_suits = [0, 0, 0]
        for c in community:
            if c >= 0:
                board_suits[card_suit(c)] += 1
        dom = board_suits.index(max(board_suits))
        out[4] = sum(1 for s in disc_suits if s == dom) / 3.0  # board suit match
    out[5] = 1.0 - avg                                    # kept rank estimate

    return out

def build_features(
    hero_hand: list,       # 2 cards (post-discard) or 5 (preflop)
    community: list,       # 0–5 visible community cards
    my_bet: int,
    opp_bet: int,
    street: int,
    is_bb: bool,
    my_disc: list,         # 3 discarded cards (or empty)
    opp_disc: list,        # 3 opp discarded cards (or empty)
    street_bets: list,     # [[p0,p1],[p0,p1],[p0,p1],[p0,p1]]
) -> np.ndarray:
    feats = []

    # Hero hand (20 floats) — sorted for permutation invariance
    if street == 0 and len(hero_hand) == 5:
        sorted_hand = sorted([c for c in hero_hand if c >= 0])
        for c in sorted_hand:
            feats.extend(card_features(c))
        for _ in range(5 - len(sorted_hand)):
            feats.extend([0.0, 0.0, 0.0, 0.0])
    else:
        h2 = sorted([c for c in hero_hand[:2] if c >= 0])
        for c in h2:
            feats.extend(card_features(c))
        for _ in range(2 - len(h2)):
            feats.extend([0.0, 0.0, 0.0, 0.0])
        feats.extend([0.0] * 12)  # padding to 20

    # Community (20 floats) — flop sorted
    comm_valid = [c for c in community if c >= 0]
    if len(comm_valid) >= 3:
        flop = sorted(comm_valid[:3])
        rest = comm_valid[3:]
        sorted_comm = flop + rest
    else:
        sorted_comm = comm_valid
    for i in range(5):
        if i < len(sorted_comm):
            feats.extend(card_features(sorted_comm[i]))
        else:
            feats.extend([0.0, 0.0, 0.0, 0.0])

    # My discards (12 floats)
    md = [c for c in my_disc if c >= 0][:3]
    for i in range(3):
        feats.extend(card_features(md[i] if i < len(md) else -1))

    # Opp discards (12 floats)
    od = [c for c in opp_disc if c >= 0][:3]
    for i in range(3):
        feats.extend(card_features(od[i] if i < len(od) else -1))

    # Street one-hot (4)
    feats.extend([float(street == s) for s in range(4)])

    # Position (1)
    feats.append(1.0 if is_bb else 0.0)

    # Bet info (4)
    pot = my_bet + opp_bet
    feats.append(my_bet / MAX_BET)
    feats.append(opp_bet / MAX_BET)
    feats.append(pot / (2 * MAX_BET))
    feats.append(max(opp_bet - my_bet, 0) / MAX_BET)

    # Hand strength (6)
    vis_comm = comm_valid[:max(0, [0, 3, 4, 5][min(street, 3)])]
    h2_feats = hero_hand if len(hero_hand) == 2 else hero_hand[:2]
    feats.extend(hand_strength_features(h2_feats, vis_comm))

    # Betting history (8): per-street bet counts, normalized
    player_idx = 1 if is_bb else 0
    opp_idx = 1 - player_idx
    for s in range(4):
        my_bets = min(street_bets[s][player_idx] / 3.0, 1.0)
        op_bets = min(street_bets[s][opp_idx] / 3.0, 1.0)
        feats.append(my_bets)
        feats.append(op_bets)

    # Opp range (6)
    feats.extend(opp_range_features(od, vis_comm))

    assert len(feats) == FEATURE_DIM, f"Feature dim mismatch: {len(feats)} != {FEATURE_DIM}"
    return np.array(feats, dtype=np.float32)


# ─── Inference model ──────────────────────────────────────────────────────────

class StrategyNet:
    """Lightweight torch inference wrapper — no training code."""
    def __init__(self, state_dict: dict):
        import torch
        import torch.nn as nn
        hidden_dim = 256

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(FEATURE_DIM, hidden_dim), nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                    nn.Linear(hidden_dim, NUM_ACTIONS),
                )
            def forward(self, x):
                return self.net(x)

        self._net = _Net()
        self._net.load_state_dict(state_dict)
        self._net.eval()
        self._torch = torch

    def get_probs(self, features: np.ndarray, valid_cfr_actions: list) -> dict:
        import torch
        with torch.no_grad():
            x = torch.from_numpy(features).unsqueeze(0)
            logits = self._net(x).squeeze(0)
            mask = torch.full((NUM_ACTIONS,), float('-inf'))
            for a in valid_cfr_actions:
                mask[a] = logits[a]
            probs = torch.softmax(mask, dim=0).numpy()
        return {a: float(probs[a]) for a in valid_cfr_actions}


# ─── Discard heuristic ────────────────────────────────────────────────────────

def choose_discard(hand5: list, board3: list) -> tuple:
    """Keep 2 cards that maximize hand strength. Returns (keep_i, keep_j)."""
    best_score = -1
    best = (0, 1)
    board_ranks = [card_rank(c) for c in board3 if c >= 0]
    board_suits = [card_suit(c) for c in board3 if c >= 0]

    for i in range(5):
        for j in range(i + 1, 5):
            c0, c1 = hand5[i], hand5[j]
            r0, r1 = card_rank(c0), card_rank(c1)
            s0, s1 = card_suit(c0), card_suit(c1)
            score = max(r0, r1)
            if r0 == r1:                     score += 20
            if s0 == s1:                     score += 5
            if r0 in board_ranks:            score += 15
            if r1 in board_ranks:            score += 15
            if abs(r0 - r1) <= 2:            score += 3
            # flush draw bonus
            sc = [0, 0, 0]
            sc[s0] += 1; sc[s1] += 1
            for bs in board_suits:
                sc[bs] += 1
            if max(sc) >= 4:                 score += 10
            if score > best_score:
                best_score = score
                best = (i, j)
    return best


# ─── Main PlayerAgent ─────────────────────────────────────────────────────────

class PlayerAgent(Agent):

    def __init__(self, stream: bool = True):
        super().__init__(stream)
        self.model = None
        self._load_model()

        # Per-hand state
        self.current_hand = -1
        self.last_street  = -1
        self.my_hand_5    = []
        self.my_hand_2    = []
        self.my_disc      = []
        self.opp_disc     = []
        self.street_bets  = [[0, 0], [0, 0], [0, 0], [0, 0]]

    def __name__(self):
        return "DeepCFRAgent"

    def _load_model(self):
        if not os.path.exists(MODEL_PATH):
            self.logger.warning(f"Model not found: {MODEL_PATH}")
            return
        try:
            import torch
            data = torch.load(MODEL_PATH, map_location='cpu')
            state_dict = data.get('state_dict', data)
            self.model = StrategyNet(state_dict)
            self.logger.info(f"Loaded Deep CFR model from {MODEL_PATH}")
        except Exception as e:
            self.logger.warning(f"Failed to load model: {e}")

    def _reset_hand(self, hand_number: int):
        if hand_number != self.current_hand:
            self.current_hand = hand_number
            self.last_street  = -1
            self.my_hand_5    = []
            self.my_hand_2    = []
            self.my_disc      = []
            self.opp_disc     = []
            self.street_bets  = [[0, 0], [0, 0], [0, 0], [0, 0]]

    def _is_bb(self, obs) -> bool:
        return obs.get("blind_position", 0) == 1

    def _get_cfr_valid_actions(self, obs) -> list:
        """Map game valid_actions to Deep CFR action IDs."""
        valid = obs["valid_actions"]
        my_bet  = obs["my_bet"]
        opp_bet = obs["opp_bet"]
        min_r   = int(obs["min_raise"])
        max_r   = int(obs["max_raise"])

        to_call = opp_bet - my_bet
        can_raise = valid[_RAISE] and max_r > 0

        cfr_valid = []
        if to_call > 0:
            if valid[_FOLD]:  cfr_valid.append(A_FOLD)
            if valid[_CALL]:  cfr_valid.append(A_CALL)
            if can_raise:
                cfr_valid.append(A_RAISE_SMALL)
                cfr_valid.append(A_RAISE_LARGE)
        else:
            if valid[_CHECK]: cfr_valid.append(A_CHECK)
            if can_raise:
                cfr_valid.append(A_BET_SMALL)
                cfr_valid.append(A_BET_LARGE)
                cfr_valid.append(A_BET_POT)

        return cfr_valid if cfr_valid else [A_CHECK if valid[_CHECK] else A_CALL]

    def _cfr_to_game_action(self, cfr_action: int, obs) -> tuple:
        """Convert Deep CFR action to game (action_type, amount, k1, k2)."""
        min_r = int(obs["min_raise"])
        max_r = int(obs["max_raise"])
        spread = max(max_r - min_r, 0)

        if cfr_action == A_FOLD:
            return (_FOLD, 0, 0, 0)
        elif cfr_action == A_CALL:
            return (_CALL, 0, 0, 0)
        elif cfr_action == A_CHECK:
            return (_CHECK, 0, 0, 0)
        elif cfr_action in (A_BET_SMALL, A_RAISE_SMALL):
            amt = max(min_r, min(min_r + int(spread * 0.25), max_r))
            return (_RAISE, amt, 0, 0)
        elif cfr_action == A_BET_POT:
            pot = obs["my_bet"] + obs["opp_bet"]
            amt = max(min_r, min(pot, max_r))
            return (_RAISE, amt, 0, 0)
        else:  # BET_LARGE / RAISE_LARGE
            amt = max(min_r, min(min_r + int(spread * 0.70), max_r))
            return (_RAISE, amt, 0, 0)

    def _fallback_action(self, obs) -> tuple:
        """Simple equity-based fallback when model is unavailable."""
        valid  = obs["valid_actions"]
        my_bet = obs["my_bet"]
        opp_bet = obs["opp_bet"]
        min_r  = int(obs["min_raise"])
        max_r  = int(obs["max_raise"])
        to_call = opp_bet - my_bet

        hand = self.my_hand_2 or [c for c in obs["my_cards"] if c >= 0][:2]
        if not hand:
            return (_CHECK, 0, 0, 0) if valid[_CHECK] else (_CALL, 0, 0, 0)

        # Rough equity from card ranks
        ranks = sorted([card_rank(c) for c in hand], reverse=True)
        equity = 0.35 + ranks[0] * 0.03 + (0.15 if ranks[0] == ranks[1] else 0)
        equity = min(equity, 0.85)

        pot = my_bet + opp_bet
        pot_odds = to_call / (to_call + pot) if to_call > 0 and pot else 0

        if to_call > 0:
            if equity >= pot_odds + 0.15 and valid[_RAISE] and max_r > 0:
                return (_RAISE, max(min_r, min(int(pot * 0.7), max_r)), 0, 0)
            elif equity >= pot_odds and valid[_CALL]:
                return (_CALL, 0, 0, 0)
            else:
                return (_FOLD, 0, 0, 0) if valid[_FOLD] else (_CALL, 0, 0, 0)
        else:
            if equity > 0.6 and valid[_RAISE] and max_r > 0:
                return (_RAISE, max(min_r, min(int(pot * 0.65), max_r)), 0, 0)
            return (_CHECK, 0, 0, 0) if valid[_CHECK] else (_CALL, 0, 0, 0)

    def act(self, observation, reward, terminated, truncated, info):
        hand_number = info.get("hand_number", -1)
        self._reset_hand(hand_number)

        street = observation["street"]
        valid  = observation["valid_actions"]
        my_cards = [c for c in observation["my_cards"] if c >= 0]

        # Track cards
        if len(my_cards) == 5 and not self.my_hand_5:
            self.my_hand_5 = list(my_cards)
        if len(my_cards) == 2 and not self.my_hand_2:
            self.my_hand_2 = list(my_cards)

        # Track discards
        od = [c for c in observation.get("opp_discarded_cards", []) if c >= 0]
        if od and not self.opp_disc:
            self.opp_disc = od
        md = [c for c in observation.get("my_discarded_cards", []) if c >= 0]
        if md and not self.my_disc:
            self.my_disc = md

        # ── Discard phase ──────────────────────────────────────────────────
        if valid[_DISCARD]:
            community = [c for c in observation["community_cards"] if c >= 0]
            ki, kj = choose_discard(my_cards, community[:3])
            self.my_hand_2 = [my_cards[ki], my_cards[kj]]
            self.my_disc   = [my_cards[k] for k in range(5) if k != ki and k != kj]
            return (_DISCARD, 0, ki, kj)

        # ── Betting phase ──────────────────────────────────────────────────
        if self.model is None:
            return self._fallback_action(observation)

        is_bb    = self._is_bb(observation)
        my_bet   = observation["my_bet"]
        opp_bet  = observation["opp_bet"]
        community = [c for c in observation["community_cards"] if c >= 0]
        hero_hand = self.my_hand_2 if self.my_hand_2 else my_cards
        if street == 0 and self.my_hand_5:
            hero_hand = self.my_hand_5

        try:
            features = build_features(
                hero_hand  = hero_hand,
                community  = community,
                my_bet     = my_bet,
                opp_bet    = opp_bet,
                street     = street,
                is_bb      = is_bb,
                my_disc    = self.my_disc,
                opp_disc   = self.opp_disc,
                street_bets = self.street_bets,
            )
        except Exception as e:
            self.logger.warning(f"Feature extraction failed: {e}")
            return self._fallback_action(observation)

        cfr_valid = self._get_cfr_valid_actions(observation)
        probs     = self.model.get_probs(features, cfr_valid)

        # Sample from strategy
        actions = list(probs.keys())
        weights = [probs[a] for a in actions]
        chosen_cfr = random.choices(actions, weights=weights, k=1)[0]

        # Track hero bet/raise for street_bets
        if chosen_cfr in (A_BET_SMALL, A_BET_LARGE, A_RAISE_SMALL, A_RAISE_LARGE):
            player_idx = 1 if is_bb else 0
            self.street_bets[street][player_idx] += 1

        return self._cfr_to_game_action(chosen_cfr, observation)

    def observe(self, observation, reward, terminated, truncated, info):
        """Track opponent bets for street_bets history."""
        opp_action = observation.get("opp_last_action", "")
        if opp_action == "RAISE":
            street = observation["street"]
            is_bb = self._is_bb(observation)
            opp_idx = 0 if is_bb else 1  # opponent is opposite position
            if 0 <= street < 4:
                self.street_bets[street][opp_idx] += 1

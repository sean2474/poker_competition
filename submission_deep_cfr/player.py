"""
Deep CFR Submission Agent.

Loads:
  data/deep_cfr_strategy.pt        — postflop StrategyNet weights
  data/deep_cfr_preflop_chart.pkl  — tabular preflop strategy (CFR+)

Preflop  : tabular chart lookup → off-size adjustment (preflop.py)
Discard  : heuristic oracle (utils.py)
Postflop : StrategyNet inference with 119-dim features (features.py)
"""

import os
import sys
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.agent import Agent

from utils    import card_rank, choose_discard
from features import build_features, FEATURE_DIM
from preflop  import (
    load_chart, lookup_chart, apply_size_adjustment, resolve_action,
    A_FOLD, A_CALL, A_CHECK, A_BET_SMALL, A_BET_LARGE,
    A_RAISE_SMALL, A_RAISE_LARGE, A_BET_POT,
)

# ─── Constants ────────────────────────────────────────────────────────────────

NUM_ACTIONS = 8
MAX_BET     = 100

_FOLD    = 0   # game action indices
_RAISE   = 1
_CHECK   = 2
_CALL    = 3
_DISCARD = 4

_AGGRESSIVE = {3, 4, 5, 6, 7}
_ACT_CHAR   = {0:'f', 1:'c', 2:'k', 3:'b', 4:'B', 5:'r', 6:'R', 7:'p'}

DATA_DIR   = os.path.join(os.path.dirname(__file__), 'data')
MODEL_PATH = os.path.join(DATA_DIR, 'deep_cfr_strategy.pt')


# ─── Postflop StrategyNet ─────────────────────────────────────────────────────

class _StrategyNet:
    """Minimal inference-only wrapper around the trained strategy network."""

    def __init__(self, state_dict: dict):
        import torch
        import torch.nn as nn
        H = 512

        class _ResBlock(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.fc1 = nn.Linear(dim, dim); self.ln1 = nn.LayerNorm(dim)
                self.fc2 = nn.Linear(dim, dim); self.ln2 = nn.LayerNorm(dim)
            def forward(self, x):
                h = torch.relu(self.ln1(self.fc1(x)))
                return torch.relu(self.ln2(self.fc2(h)) + x)

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Sequential(nn.Linear(FEATURE_DIM, H), nn.ReLU())
                self.res   = nn.Sequential(_ResBlock(H), _ResBlock(H), _ResBlock(H))
                self.head  = nn.Linear(H, NUM_ACTIONS)
            def forward(self, x):
                return self.head(self.res(self.embed(x)))

        self._net = _Net()
        self._net.load_state_dict(state_dict)
        self._net.eval()
        self._torch = torch

    def get_probs(self, feats: np.ndarray, valid: list,
                  deterministic_threshold: float = 0.75) -> dict:
        with self._torch.no_grad():
            x      = self._torch.from_numpy(feats).unsqueeze(0)
            logits = self._net(x).squeeze(0)
            mask   = self._torch.full((NUM_ACTIONS,), float('-inf'))
            for a in valid: mask[a] = logits[a]
            probs  = self._torch.softmax(mask, 0).numpy()
        result = {a: float(probs[a]) for a in valid}
        # Variance reduction: if one action dominates, play it deterministically
        best_a = max(result, key=result.get)
        if result[best_a] >= deterministic_threshold:
            return {a: (1.0 if a == best_a else 0.0) for a in valid}
        return result


# ─── PlayerAgent ─────────────────────────────────────────────────────────────

class PlayerAgent(Agent):

    def __init__(self, stream: bool = True):
        super().__init__(stream)
        self._net   = None
        self._chart = {}
        self._load()
        self._reset_state(-1)

    def __name__(self): return "DeepCFRAgent"

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load(self):
        try:
            import torch
            sd = torch.load(MODEL_PATH, map_location='cpu')
            self._net = _StrategyNet(sd.get('state_dict', sd))
            self.logger.info("Deep CFR: postflop net loaded")
        except Exception as e:
            self.logger.warning(f"Deep CFR: postflop net failed — {e}")

        self._chart = load_chart()
        if self._chart:
            self.logger.info(f"Deep CFR: preflop chart loaded ({len(self._chart):,} infosets)")
        else:
            self.logger.warning("Deep CFR: preflop chart not found")

    # ── Per-hand state ────────────────────────────────────────────────────────

    def _reset_state(self, hand_num: int):
        self._hand_num          = hand_num
        self._hand5             = []    # 5 cards before discard
        self._hand2             = []    # 2 kept cards
        self._my_disc           = []
        self._opp_disc          = []
        self._history           = []    # [(player_id, cfr_action), ...]
        self._last_ratios       = [[0., 0.]] * 4  # last bet/pot per street/player
        self._bet_counts        = [[0,  0 ]] * 4  # raise count per street/player
        self._num_acts          = 0     # actions taken this street
        self._street            = -1
        self._prev_opp_bet      = 0

    def _check_reset(self, info):
        n = info.get('hand_number', -1)
        if n != self._hand_num:
            self._reset_state(n)

    # ── Street boundary ───────────────────────────────────────────────────────

    def _on_street(self, street: int):
        if street != self._street:
            self._num_acts = 0
            self._street   = street

    # ── Bet tracking ──────────────────────────────────────────────────────────

    def _record_raise(self, player_id: int, raise_amt: int,
                      pot_before: int, street: int):
        if raise_amt <= 0: return
        ratio = raise_amt / max(float(pot_before), 1.)
        row = list(self._last_ratios[street])
        row[player_id] = ratio
        self._last_ratios = (self._last_ratios[:street]
                             + [row]
                             + self._last_ratios[street + 1:])
        row2 = list(self._bet_counts[street])
        row2[player_id] += 1
        self._bet_counts = (self._bet_counts[:street]
                            + [row2]
                            + self._bet_counts[street + 1:])

    # ── CFR valid actions ─────────────────────────────────────────────────────

    def _cfr_valid(self, obs) -> list:
        valid   = obs['valid_actions']
        to_call = obs['opp_bet'] - obs['my_bet']
        max_r   = int(obs['max_raise'])
        can_r   = valid[_RAISE] and max_r > 0
        if to_call > 0:
            acts = ([A_FOLD] if valid[_FOLD] else []) + ([A_CALL] if valid[_CALL] else [])
            if can_r: acts += [A_RAISE_SMALL, A_RAISE_LARGE]
        else:
            acts = ([A_CHECK] if valid[_CHECK] else [])
            if can_r: acts += [A_BET_SMALL, A_BET_LARGE, A_BET_POT]
        return acts or ([A_CHECK] if valid[_CHECK] else [A_CALL])

    # ── CFR action → game action ──────────────────────────────────────────────

    def _to_game(self, cfr_a: int, obs) -> tuple:
        min_r = int(obs['min_raise'])
        max_r = int(obs['max_raise'])   # = MAX_PLAYER_BET - max(bets)  (increment)
        pot   = obs['my_bet'] + obs['opp_bet']
        if cfr_a == A_FOLD:  return (_FOLD,  0, 0, 0)
        if cfr_a == A_CALL:  return (_CALL,  0, 0, 0)
        if cfr_a == A_CHECK: return (_CHECK, 0, 0, 0)
        # pot-relative sizing (raise_amount is INCREMENT above opp bet)
        if cfr_a in (A_BET_SMALL, A_RAISE_SMALL):
            amt = max(min_r, min(int(pot * 0.33), max_r))
        elif cfr_a == A_BET_POT:
            amt = max(min_r, min(pot, max_r))
        else:  # BET_LARGE / RAISE_LARGE
            amt = max(min_r, min(int(pot * 0.75), max_r))
        return (_RAISE, amt, 0, 0)

    # ── Preflop decision ──────────────────────────────────────────────────────

    def _preflop_act(self, obs, is_bb: bool) -> tuple:
        cfr_valid = self._cfr_valid(obs)
        slot_probs = lookup_chart(self._chart, self._hand5, self._history,
                                  obs['opp_bet'])

        if slot_probs is None:
            # chart miss → uniform
            probs = {a: 1. / len(cfr_valid) for a in cfr_valid}
        else:
            slot_probs = apply_size_adjustment(slot_probs, tuple(self._hand5),
                                               obs['opp_bet'])
            probs = resolve_action(slot_probs, cfr_valid)
            if not probs:
                probs = {a: 1. / len(cfr_valid) for a in cfr_valid}

        valid_w = [(a, probs.get(a, 0.)) for a in cfr_valid if probs.get(a, 0.) > 0]
        if not valid_w:
            valid_w = [(a, 1.) for a in cfr_valid]
        chosen = random.choices([a for a,_ in valid_w],
                                [w for _,w in valid_w], k=1)[0]

        pid = 1 if is_bb else 0
        self._history.append((pid, chosen))
        self._num_acts += 1
        if chosen in _AGGRESSIVE:
            pot_before = obs['my_bet'] + obs['opp_bet']
            self._record_raise(pid, int(obs['min_raise']), pot_before, obs['street'])

        return self._to_game(chosen, obs)

    # ── Postflop decision ─────────────────────────────────────────────────────

    def _postflop_act(self, obs, is_bb: bool) -> tuple:
        cfr_valid = self._cfr_valid(obs)

        if self._net is None:
            return self._fallback(obs)

        street    = obs['street']
        my_bet    = obs['my_bet']; opp_bet = obs['opp_bet']
        community = [c for c in obs['community_cards'] if c >= 0]
        hero      = self._hand2 or [c for c in obs['my_cards'] if c >= 0][:2]

        try:
            feats = build_features(
                hero_hand            = hero,
                community            = community,
                my_bet               = my_bet,
                opp_bet              = opp_bet,
                street               = street,
                is_bb                = is_bb,
                my_disc              = self._my_disc,
                opp_disc             = self._opp_disc,
                street_last_ratios   = self._last_ratios,
                street_bet_counts    = self._bet_counts,
                history              = self._history,
                num_acts_this_street = self._num_acts,
            )
        except Exception as e:
            self.logger.warning(f"feature error: {e}")
            return self._fallback(obs)

        probs  = self._net.get_probs(feats, cfr_valid)
        chosen = random.choices(list(probs), list(probs.values()), k=1)[0]

        pid = 1 if is_bb else 0
        self._history.append((pid, chosen))
        self._num_acts += 1
        if chosen in _AGGRESSIVE:
            pot_before = obs['my_bet'] + obs['opp_bet']
            self._record_raise(pid, int(obs['min_raise']), pot_before, street)

        return self._to_game(chosen, obs)

    # ── Fallback (no model) ───────────────────────────────────────────────────

    def _fallback(self, obs) -> tuple:
        valid   = obs['valid_actions']
        my_bet  = obs['my_bet']; opp_bet = obs['opp_bet']
        to_call = opp_bet - my_bet
        min_r   = int(obs['min_raise']); max_r = int(obs['max_raise'])
        hand    = self._hand2 or [c for c in obs['my_cards'] if c >= 0][:2]
        if not hand:
            return (_CHECK, 0, 0, 0) if valid[_CHECK] else (_CALL, 0, 0, 0)
        ranks  = sorted([card_rank(c) for c in hand], reverse=True)
        equity = min(0.35 + ranks[0] * 0.03 + (0.15 if ranks[0] == ranks[1] else 0), 0.85)
        pot    = my_bet + opp_bet
        po     = to_call / (to_call + pot) if to_call > 0 and pot else 0
        if to_call > 0:
            if equity >= po + 0.15 and valid[_RAISE] and max_r > 0:
                return (_RAISE, max(min_r, min(int(pot * 0.7), max_r)), 0, 0)
            return (_CALL, 0, 0, 0) if equity >= po and valid[_CALL] else (_FOLD, 0, 0, 0)
        if equity > 0.6 and valid[_RAISE] and max_r > 0:
            return (_RAISE, max(min_r, min(int(pot * 0.65), max_r)), 0, 0)
        return (_CHECK, 0, 0, 0) if valid[_CHECK] else (_CALL, 0, 0, 0)

    # ── act ───────────────────────────────────────────────────────────────────

    def act(self, observation, reward, terminated, truncated, info):
        self._check_reset(info)

        street = observation['street']
        valid  = observation['valid_actions']
        cards  = [c for c in observation['my_cards'] if c >= 0]
        is_bb  = observation.get('blind_position', 0) == 1

        self._on_street(street)

        # card tracking
        if len(cards) == 5 and not self._hand5: self._hand5 = list(cards)
        if len(cards) == 2 and not self._hand2: self._hand2 = list(cards)

        od = [c for c in observation.get('opp_discarded_cards', []) if c >= 0]
        if od and not self._opp_disc: self._opp_disc = od
        md = [c for c in observation.get('my_discarded_cards', []) if c >= 0]
        if md and not self._my_disc:  self._my_disc  = md

        # discard
        if valid[_DISCARD]:
            comm  = [c for c in observation['community_cards'] if c >= 0]
            ki, kj = choose_discard(cards, comm[:3])
            self._hand2   = [cards[ki], cards[kj]]
            self._my_disc = [cards[k] for k in range(5) if k not in (ki, kj)]
            return (_DISCARD, 0, ki, kj)

        if street == 0:
            return self._preflop_act(observation, is_bb)
        return self._postflop_act(observation, is_bb)

    # ── observe ───────────────────────────────────────────────────────────────

    def observe(self, observation, reward, terminated, truncated, info):
        self._check_reset(info)

        street   = observation.get('street', 0)
        self._on_street(street)

        opp_last = observation.get('opp_last_action', '')
        my_bet   = observation.get('my_bet', 0)
        opp_bet  = observation.get('opp_bet', 0)
        is_bb    = observation.get('blind_position', 0) == 1
        opp_pid  = 0 if is_bb else 1

        _map = {'FOLD': A_FOLD, 'CALL': A_CALL, 'CHECK': A_CHECK,
                'INVALID': A_FOLD}   # invalid treated as fold by engine
        if opp_last in _map:
            self._history.append((opp_pid, _map[opp_last]))
            self._num_acts += 1
        elif opp_last == 'RAISE':
            self._history.append((opp_pid, A_RAISE_SMALL))
            self._num_acts += 1
            raise_amt  = opp_bet - self._prev_opp_bet
            pot_before = self._prev_opp_bet + my_bet
            self._record_raise(opp_pid, raise_amt, max(pot_before, 1), street)

        self._prev_opp_bet = opp_bet

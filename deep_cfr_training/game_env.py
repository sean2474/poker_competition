"""
Game environment for Deep CFR training.
C++ acceleration required (libtraversal.so / .dylib).
"""

import sys
import os
import random
import numpy as np
import ctypes

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
A_BET_POT = 7
NUM_ACTIONS = 8
FEATURE_DIM = 93  # 85 + 8 betting history


# ─── C++ Library (required) ───

_cpp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cpp')
_c_lib = None
for _ext in ['libtraversal.so', 'libtraversal.dylib']:
    _p = os.path.join(_cpp_dir, _ext)
    if os.path.exists(_p):
        _c_lib = ctypes.CDLL(_p)
        break

assert _c_lib is not None, (
    f"C++ library not found in {_cpp_dir}. "
    "Build it: cd cpp && bash build.sh"
)

# Function signatures
_c_lib.c_state_features.argtypes = [
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int), ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
    ctypes.c_int, ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_int)  # street_bets_flat (8 ints) or null
]
_c_lib.c_evaluate_showdown.argtypes = [
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)
]
_c_lib.c_evaluate_showdown.restype = ctypes.c_int
_c_lib.c_batch_deal_discard.argtypes = [
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
    ctypes.c_uint, ctypes.c_float
]

print(f"[game_env] C++ loaded: {_p}")
HAS_CPP = True


# ─── C++ wrapped functions ───

def state_to_features(hero_hand, community, my_bet, opp_bet, street, is_bb,
                       my_discards=None, opp_discards=None, pot=None, hero_hand5=None,
                       street_bets=None):
    h2 = (ctypes.c_int * 2)(*(list(hero_hand)[:2] + [-1, -1])[:2])
    h5 = (ctypes.c_int * 5)(*(list(hero_hand5) if hero_hand5 else [-1]*5)[:5])
    comm = (ctypes.c_int * 5)(*([c for c in (community or [])] + [-1]*5)[:5])
    n_comm = len([c for c in (community or []) if c >= 0])
    md = (ctypes.c_int * 3)(*([c for c in (my_discards or [])] + [-1]*3)[:3])
    od = (ctypes.c_int * 3)(*([c for c in (opp_discards or [])] + [-1]*3)[:3])
    feat = (ctypes.c_float * FEATURE_DIM)()
    use5 = 1 if (hero_hand5 is not None and street == 0) else 0
    # street_bets: [[p0,p1],[p0,p1],[p0,p1],[p0,p1]] or None
    if street_bets is not None:
        sb_flat = (ctypes.c_int * 8)(*[street_bets[s][p] for s in range(4) for p in range(2)])
        _c_lib.c_state_features(h2, h5, comm, n_comm, int(my_bet), int(opp_bet),
                                 street, 1 if is_bb else 0, md, od, use5, feat, sb_flat)
    else:
        _c_lib.c_state_features(h2, h5, comm, n_comm, int(my_bet), int(opp_bet),
                                 street, 1 if is_bb else 0, md, od, use5, feat, None)
    return np.array(feat, dtype=np.float32)


def evaluate_showdown(p0_hand, p1_hand, community):
    h0 = (ctypes.c_int * 2)(*list(p0_hand)[:2])
    h1 = (ctypes.c_int * 2)(*list(p1_hand)[:2])
    comm = (ctypes.c_int * 5)(*list(community)[:5])
    return _c_lib.c_evaluate_showdown(h0, h1, comm)


def deal_game():
    """Deal using C++ batch of 1."""
    r = batch_deal_discard(1)
    return list(r[5][0]), list(r[6][0]), list(r[4][0])


def fast_discard(hand5, board3, temperature=0.05):
    """Single discard using C++ batch of 1 — returns (ki, kj)."""
    # Use batch_deal_discard internally is overkill for 1 game
    # Fall back to simple C++ call
    ki = ctypes.c_int()
    kj = ctypes.c_int()
    h5 = (ctypes.c_int * 5)(*list(hand5)[:5])
    b3 = (ctypes.c_int * 3)(*list(board3)[:3])
    seed = random.randint(0, 2**31)
    _c_lib.c_fast_discard(h5, b3, ctypes.byref(ki), ctypes.byref(kj),
                           ctypes.c_uint(seed), ctypes.c_float(temperature))
    return ki.value, kj.value


def batch_deal_discard(n, temperature=0.05):
    """Deal and discard N games at once using C++ multi-threading."""
    p0h = (ctypes.c_int * (n*2))()
    p1h = (ctypes.c_int * (n*2))()
    p0d = (ctypes.c_int * (n*3))()
    p1d = (ctypes.c_int * (n*3))()
    comms = (ctypes.c_int * (n*5))()
    p0h5 = (ctypes.c_int * (n*5))()
    p1h5 = (ctypes.c_int * (n*5))()
    seed = random.randint(0, 2**31)
    _c_lib.c_batch_deal_discard(n, p0h, p1h, p0d, p1d, comms, p0h5, p1h5,
                                 ctypes.c_uint(seed), ctypes.c_float(temperature))
    return (np.frombuffer(p0h, dtype=np.int32).reshape(n, 2).copy(),
            np.frombuffer(p1h, dtype=np.int32).reshape(n, 2).copy(),
            np.frombuffer(p0d, dtype=np.int32).reshape(n, 3).copy(),
            np.frombuffer(p1d, dtype=np.int32).reshape(n, 3).copy(),
            np.frombuffer(comms, dtype=np.int32).reshape(n, 5).copy(),
            np.frombuffer(p0h5, dtype=np.int32).reshape(n, 5).copy(),
            np.frombuffer(p1h5, dtype=np.int32).reshape(n, 5).copy())


# ─── Game State (Python — needed for recursive traversal) ───

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
        self.street_bets = [[0, 0], [0, 0], [0, 0], [0, 0]]  # [street][player] max raise amount
    
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
        s.street_bets = [list(b) for b in self.street_bets]
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
                pot = self.bets[0] + self.bets[1]
                mn = self.min_raise
                # Pot-relative sizing: 33% / 75% / 100% of pot
                small_amt = max(mn, min(int(pot * 0.33), max_raise))
                large_amt = max(mn, min(int(pot * 0.75), max_raise))
                pot_amt   = max(mn, min(pot,             max_raise))
                thresh = max(2, pot // 20)  # 5% of pot, min 2
                actions.append(A_BET_SMALL)
                if abs(pot_amt - large_amt) > thresh:
                    actions.append(A_BET_POT)
                if abs(large_amt - small_amt) > thresh:
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
        pot = s.bets[0] + s.bets[1]
        mn = s.min_raise
        if action in (A_BET_SMALL, A_RAISE_SMALL):
            raise_amt = max(mn, min(int(pot * 0.33), max_raise))
        elif action == A_BET_POT:
            raise_amt = max(mn, min(pot, max_raise))
        else:  # BET_LARGE / RAISE_LARGE
            raise_amt = max(mn, min(int(pot * 0.75), max_raise))
        
        raise_amt = max(s.min_raise, min(raise_amt, max_raise))
        s.street_bets[s.street][cp] = max(s.street_bets[s.street][cp], raise_amt)
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
            # Note: street_bets carry over (history preserved)

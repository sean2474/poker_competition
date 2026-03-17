"""C++ acceleration wrapper — feature extraction + game primitives."""

import os
import ctypes
import random
import numpy as np

from .constants import FEATURE_DIM, CPP_FEATURE_DIM, MAX_BET

_AGGRESSIVE = {3, 4, 5, 6, 7}  # BET_SMALL, BET_LARGE, RAISE_SMALL, RAISE_LARGE, BET_POT


def _extra_features(history, community, my_bet, opp_bet,
                    num_actions_this_street, hero_player,
                    street_bet_counts=None) -> np.ndarray:
    """26 additional Python-computed features appended after C++ base (93).
    [0-17]  initiative, action context, line class, board texture, bet ratios
    [18-25] bet counts per player per street (re-raise tracking)
    """
    feat = np.zeros(26, dtype=np.float32)

    # [0-1] Initiative: last aggressor
    for p, a in reversed(history):
        if a in _AGGRESSIVE:
            feat[0] = float(p == hero_player)   # hero was last aggressor
            feat[1] = float(p != hero_player)   # villain was last aggressor
            break

    # [2-3] Action context
    to_call   = max(opp_bet - my_bet, 0)
    feat[2]   = float(to_call > 0)   # facing_bet
    feat[3]   = float(to_call == 0)  # can_check

    # [4-7] Line class on current street
    curr = history[-num_actions_this_street:] if num_actions_this_street > 0 else []
    bets = sum(1 for _, a in curr if a in _AGGRESSIVE)
    if   to_call == 0 and bets == 0: feat[4] = 1.0  # checked_to / street start
    elif to_call > 0  and bets == 1: feat[5] = 1.0  # facing_lead
    elif to_call > 0  and bets >= 2: feat[6] = 1.0  # facing_raise
    elif to_call == 0 and bets >= 1: feat[7] = 1.0  # raised_pot (opp just called)

    # [8-12] Board texture
    board = [c for c in community if c is not None and c >= 0]
    if board:
        ranks       = [c % 9 for c in board]
        suits       = [c // 9 for c in board]
        suit_counts = [suits.count(s) for s in range(3)]

        feat[8]  = float(len(ranks) != len(set(ranks)))            # paired board
        feat[9]  = float(len(board) >= 3 and len(set(suits)) == 1) # monotone
        feat[10] = float(max(suit_counts) >= 2)                    # two-suited (flush draw)
        if len(board) >= 3:
            span    = max(ranks) - min(ranks)
            feat[11] = float(span <= 4)                            # connected
        if len(board) >= 4:                                        # scare card
            prev_suits = [c // 9 for c in board[:-1]]
            feat[12] = float(prev_suits.count(suits[-1]) >= 2)

    # [13-17] Bet ratios
    pot      = my_bet + opp_bet
    sp       = max(float(pot), 1.0)
    max_r    = max(MAX_BET - max(my_bet, opp_bet), 0)
    feat[13] = min(to_call  / sp, 4.0)   # to_call / pot
    feat[14] = min(opp_bet  / sp, 4.0)   # opp_bet / pot
    feat[15] = min(my_bet   / sp, 4.0)   # my_bet / pot
    feat[16] = min(max_r    / sp, 4.0)   # max_raise / pot
    feat[17] = max_r / 100.0             # remaining raise room

    # [18-25] Bet counts per player per street (normalized to /4; >4 is capped)
    # Layout: [hero_pf, opp_pf, hero_flop, opp_flop, hero_turn, opp_turn, hero_river, opp_river]
    if street_bet_counts is not None:
        for s in range(4):
            feat[18 + s*2]     = min(street_bet_counts[s][hero_player]  / 4.0, 1.0)
            feat[18 + s*2 + 1] = min(street_bet_counts[s][1-hero_player] / 4.0, 1.0)

    return feat

# ── C++ library loading ─────────────────────────────────────────────────────
_cpp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cpp')
_c_lib = None
for _ext in ['libtraversal.so', 'libtraversal.dylib']:
    _p = os.path.join(_cpp_dir, _ext)
    if os.path.exists(_p):
        _c_lib = ctypes.CDLL(_p)
        break

assert _c_lib is not None, (
    f"C++ library not found in {_cpp_dir}.\n"
    "Build: cd cpp && bash build.sh"
)

# ── Function signatures ─────────────────────────────────────────────────────
_c_lib.c_state_features.argtypes = [
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int), ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
    ctypes.c_int, ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_int),
]
_c_lib.c_evaluate_showdown.argtypes = [
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
]
_c_lib.c_evaluate_showdown.restype = ctypes.c_int
_c_lib.c_fast_discard.argtypes = [
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
    ctypes.c_uint, ctypes.c_float,
]
_c_lib.c_batch_deal_discard.argtypes = [
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
    ctypes.c_uint, ctypes.c_float,
]

print(f"[game] C++ loaded: {_p}")


# ── Public API ──────────────────────────────────────────────────────────────

_BET_HIST_START = 79   # C++ betting history dims start here (8 dims)


def state_to_features(hero_hand, community, my_bet, opp_bet, street, is_bb,
                      my_discards=None, opp_discards=None,
                      hero_hand5=None, street_bets=None,
                      history=None, num_actions_this_street=0,
                      street_last_ratios=None, street_bet_counts=None) -> np.ndarray:
    h2   = (ctypes.c_int * 2)(*(list(hero_hand)[:2] + [-1, -1])[:2])
    h5   = (ctypes.c_int * 5)(*(list(hero_hand5) if hero_hand5 else [-1] * 5)[:5])
    comm = (ctypes.c_int * 5)(*([c for c in (community or [])] + [-1] * 5)[:5])
    n_c  = len([c for c in (community or []) if c >= 0])
    md   = (ctypes.c_int * 3)(*([c for c in (my_discards  or [])] + [-1] * 3)[:3])
    od   = (ctypes.c_int * 3)(*([c for c in (opp_discards or [])] + [-1] * 3)[:3])
    feat = (ctypes.c_float * CPP_FEATURE_DIM)()
    use5 = 1 if (hero_hand5 is not None and street == 0) else 0

    if street_bets is not None:
        sb = (ctypes.c_int * 8)(*[street_bets[s][p] for s in range(4) for p in range(2)])
        _c_lib.c_state_features(h2, h5, comm, n_c, int(my_bet), int(opp_bet),
                                 street, 1 if is_bb else 0, md, od, use5, feat, sb)
    else:
        _c_lib.c_state_features(h2, h5, comm, n_c, int(my_bet), int(opp_bet),
                                 street, 1 if is_bb else 0, md, od, use5, feat, None)
    base = np.array(feat, dtype=np.float32)

    # Override C++ betting history (dims 79-86) with last-bet pot-relative ratios
    if street_last_ratios is not None:
        hero_player = 1 if is_bb else 0
        for s in range(4):
            base[_BET_HIST_START + s*2]     = min(float(street_last_ratios[s][hero_player]),     4.0)
            base[_BET_HIST_START + s*2 + 1] = min(float(street_last_ratios[s][1-hero_player]), 4.0)

    hero_player = 1 if is_bb else 0
    extra = _extra_features(
        history or [], community or [], my_bet, opp_bet,
        num_actions_this_street, hero_player,
        street_bet_counts=street_bet_counts,
    )
    return np.concatenate([base, extra])


def evaluate_showdown(p0_hand, p1_hand, community) -> int:
    h0   = (ctypes.c_int * 2)(*list(p0_hand)[:2])
    h1   = (ctypes.c_int * 2)(*list(p1_hand)[:2])
    comm = (ctypes.c_int * 5)(*list(community)[:5])
    return _c_lib.c_evaluate_showdown(h0, h1, comm)


def fast_discard(hand5, board3, temperature=0.05):
    ki = ctypes.c_int(); kj = ctypes.c_int()
    h5 = (ctypes.c_int * 5)(*list(hand5)[:5])
    b3 = (ctypes.c_int * 3)(*list(board3)[:3])
    _c_lib.c_fast_discard(h5, b3, ctypes.byref(ki), ctypes.byref(kj),
                           ctypes.c_uint(random.randint(0, 2**31)),
                           ctypes.c_float(temperature))
    return ki.value, kj.value


def batch_deal_discard(n: int, temperature: float = 0.05):
    """Deal and discard N games at once using C++ OpenMP."""
    p0h  = (ctypes.c_int * (n * 2))()
    p1h  = (ctypes.c_int * (n * 2))()
    p0d  = (ctypes.c_int * (n * 3))()
    p1d  = (ctypes.c_int * (n * 3))()
    comm = (ctypes.c_int * (n * 5))()
    p0h5 = (ctypes.c_int * (n * 5))()
    p1h5 = (ctypes.c_int * (n * 5))()
    _c_lib.c_batch_deal_discard(n, p0h, p1h, p0d, p1d, comm, p0h5, p1h5,
                                 ctypes.c_uint(random.randint(0, 2**31)),
                                 ctypes.c_float(temperature))
    return (
        np.frombuffer(p0h,  dtype=np.int32).reshape(n, 2).copy(),
        np.frombuffer(p1h,  dtype=np.int32).reshape(n, 2).copy(),
        np.frombuffer(p0d,  dtype=np.int32).reshape(n, 3).copy(),
        np.frombuffer(p1d,  dtype=np.int32).reshape(n, 3).copy(),
        np.frombuffer(comm, dtype=np.int32).reshape(n, 5).copy(),
        np.frombuffer(p0h5, dtype=np.int32).reshape(n, 5).copy(),
        np.frombuffer(p1h5, dtype=np.int32).reshape(n, 5).copy(),
    )

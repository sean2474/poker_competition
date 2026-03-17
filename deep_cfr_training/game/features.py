"""C++ acceleration wrapper — feature extraction + game primitives."""

import os
import ctypes
import random
import numpy as np

from .constants import FEATURE_DIM

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
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),  # hand2, hand5
    ctypes.POINTER(ctypes.c_int), ctypes.c_int,                  # community, n_comm
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,      # my_bet,opp_bet,street,is_bb
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),  # my_disc, opp_disc
    ctypes.c_int, ctypes.POINTER(ctypes.c_float),                 # use_hand5, features_out
    ctypes.POINTER(ctypes.c_int),                                 # street_bets_flat
    ctypes.POINTER(ctypes.c_float),                               # street_last_ratios_flat
    ctypes.POINTER(ctypes.c_int),                                 # street_bet_counts_flat
    ctypes.POINTER(ctypes.c_int),                                 # history_players
    ctypes.POINTER(ctypes.c_int),                                 # history_actions
    ctypes.c_int,                                                 # history_len
    ctypes.c_int,                                                 # num_acts_this_street
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

_NULL_INT   = ctypes.POINTER(ctypes.c_int)()
_NULL_FLOAT = ctypes.POINTER(ctypes.c_float)()


def state_to_features(hero_hand, community, my_bet, opp_bet, street, is_bb,
                      my_discards=None, opp_discards=None,
                      hero_hand5=None, street_bets=None,
                      history=None, num_actions_this_street=0,
                      street_last_ratios=None, street_bet_counts=None) -> np.ndarray:
    """Single C++ call outputs full 119-dim feature vector."""
    h2   = (ctypes.c_int * 2)(*(list(hero_hand)[:2] + [-1, -1])[:2])
    h5   = (ctypes.c_int * 5)(*(list(hero_hand5) if hero_hand5 else [-1] * 5)[:5])
    comm = (ctypes.c_int * 5)(*([c for c in (community or [])] + [-1] * 5)[:5])
    n_c  = len([c for c in (community or []) if c >= 0])
    md   = (ctypes.c_int * 3)(*([c for c in (my_discards  or [])] + [-1] * 3)[:3])
    od   = (ctypes.c_int * 3)(*([c for c in (opp_discards or [])] + [-1] * 3)[:3])
    feat = (ctypes.c_float * FEATURE_DIM)()
    use5 = 1 if (hero_hand5 is not None and street == 0) else 0

    # street_bets (chip amounts, fallback)
    sb = ((ctypes.c_int * 8)(*[street_bets[s][p] for s in range(4) for p in range(2)])
          if street_bets is not None else _NULL_INT)

    # street_last_ratios (pot-relative, overrides [79:87])
    slr = ((ctypes.c_float * 8)(*[float(street_last_ratios[s][p])
                                   for s in range(4) for p in range(2)])
           if street_last_ratios is not None else _NULL_FLOAT)

    # street_bet_counts (re-raise tracking)
    sbc = ((ctypes.c_int * 8)(*[int(street_bet_counts[s][p])
                                  for s in range(4) for p in range(2)])
           if street_bet_counts is not None else _NULL_INT)

    # history as two flat int arrays
    hist = history or []
    hlen = len(hist)
    if hlen > 0:
        hp_arr = (ctypes.c_int * hlen)(*[p for p, _ in hist])
        ha_arr = (ctypes.c_int * hlen)(*[a for _, a in hist])
    else:
        hp_arr = ha_arr = _NULL_INT

    _c_lib.c_state_features(
        h2, h5, comm, n_c, int(my_bet), int(opp_bet),
        street, 1 if is_bb else 0, md, od, use5, feat,
        sb, slr, sbc, hp_arr, ha_arr,
        ctypes.c_int(hlen), ctypes.c_int(num_actions_this_street),
    )
    return np.frombuffer(feat, dtype=np.float32).copy()


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

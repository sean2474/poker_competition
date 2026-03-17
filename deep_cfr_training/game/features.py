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

def state_to_features(hero_hand, community, my_bet, opp_bet, street, is_bb,
                      my_discards=None, opp_discards=None,
                      hero_hand5=None, street_bets=None) -> np.ndarray:
    h2   = (ctypes.c_int * 2)(*(list(hero_hand)[:2] + [-1, -1])[:2])
    h5   = (ctypes.c_int * 5)(*(list(hero_hand5) if hero_hand5 else [-1] * 5)[:5])
    comm = (ctypes.c_int * 5)(*([c for c in (community or [])] + [-1] * 5)[:5])
    n_c  = len([c for c in (community or []) if c >= 0])
    md   = (ctypes.c_int * 3)(*([c for c in (my_discards  or [])] + [-1] * 3)[:3])
    od   = (ctypes.c_int * 3)(*([c for c in (opp_discards or [])] + [-1] * 3)[:3])
    feat = (ctypes.c_float * FEATURE_DIM)()
    use5 = 1 if (hero_hand5 is not None and street == 0) else 0

    if street_bets is not None:
        sb = (ctypes.c_int * 8)(*[street_bets[s][p] for s in range(4) for p in range(2)])
        _c_lib.c_state_features(h2, h5, comm, n_c, int(my_bet), int(opp_bet),
                                 street, 1 if is_bb else 0, md, od, use5, feat, sb)
    else:
        _c_lib.c_state_features(h2, h5, comm, n_c, int(my_bet), int(opp_bet),
                                 street, 1 if is_bb else 0, md, od, use5, feat, None)
    return np.array(feat, dtype=np.float32)


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

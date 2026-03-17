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
_c_lib.c_batch_warmup_ev.argtypes = [
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),  # p0_hand5s, p1_hand5s
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),  # my_bets, opp_bets
    ctypes.POINTER(ctypes.c_int),                                 # traversing_players
    ctypes.POINTER(ctypes.c_float),                               # evs_out
    ctypes.c_uint, ctypes.c_int,                                  # seed, n_boards
]
# Postflop state machine
_c_lib.c_postflop_alloc.argtypes       = [ctypes.c_int]
_c_lib.c_postflop_alloc.restype        = ctypes.c_void_p
_c_lib.c_postflop_free.argtypes        = [ctypes.c_void_p]
_c_lib.c_postflop_init_one.argtypes    = [
    ctypes.c_void_p, ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),                          # state_flat
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),  # p0/p1 hand
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),  # p0/p1 hand5
    ctypes.POINTER(ctypes.c_int),                          # community
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),  # p0/p1 disc
    ctypes.c_int,                                          # traversing_player
]
_c_lib.c_postflop_collect_pending.argtypes = [
    ctypes.c_void_p, ctypes.c_int,
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),   ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
]
_c_lib.c_postflop_collect_pending.restype = ctypes.c_int
_c_lib.c_postflop_resume_batch.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_uint,
]
_c_lib.c_postflop_collect_samples.argtypes = [
    ctypes.c_void_p, ctypes.c_int,
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_int),   ctypes.POINTER(ctypes.c_int),   ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_int),   ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_float, ctypes.c_int, ctypes.c_int,
]
_c_lib.c_postflop_get_evs.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
_c_lib.c_postflop_n_pending.argtypes  = [ctypes.c_void_p, ctypes.c_int]
_c_lib.c_postflop_n_pending.restype   = ctypes.c_int

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


def batch_warmup_ev(p0_hand5s: np.ndarray, p1_hand5s: np.ndarray,
                    my_bets: np.ndarray, opp_bets: np.ndarray,
                    traversing_players: np.ndarray,
                    n_boards: int = 15) -> np.ndarray:
    """C++ OpenMP batch equity EV for N postflop states (warmup phase)."""
    n = len(traversing_players)
    evs = (ctypes.c_float * n)()
    _c_lib.c_batch_warmup_ev(
        n,
        p0_hand5s.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        p1_hand5s.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        my_bets.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        opp_bets.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        traversing_players.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        evs,
        ctypes.c_uint(random.randint(0, 2**31)),
        ctypes.c_int(n_boards),
    )
    return np.frombuffer(evs, dtype=np.float32).copy()


# ── Postflop C++ state machine ───────────────────────────────────────────────

def serialize_gamestate(state) -> np.ndarray:
    """Serialize a Python GameState into a flat int32 array for C++ init."""
    parts = [
        state.street, state.bets[0], state.bets[1],
        state.current_player, int(state.is_terminal), state.folded_player,
        state.min_raise, state.last_street_bet, state.num_actions_this_street,
        state.preflop_open_override if state.preflop_open_override is not None else -1,
        # street_bets[4][2]
        *[state.street_bets[s][p] for s in range(4) for p in range(2)],
    ]
    int_arr = np.array(parts, dtype=np.int32)
    # street_last_ratios[4][2] as float32
    ratios = np.array([state.street_last_ratios[s][p]
                       for s in range(4) for p in range(2)], dtype=np.float32)
    # street_bet_counts[4][2]
    counts = np.array([state.street_bet_counts[s][p]
                       for s in range(4) for p in range(2)], dtype=np.int32)
    hist = state.history or []
    hlen = np.array([len(hist)], dtype=np.int32)
    hplayers = np.array([p for p, _ in hist], dtype=np.int32) if hist else np.array([], dtype=np.int32)
    hactions = np.array([a for _, a in hist], dtype=np.int32) if hist else np.array([], dtype=np.int32)
    # Concatenate: ints, then floats (reinterpreted as ints), then ints again
    return np.concatenate([
        int_arr,
        ratios.view(np.int32),    # cast float bits as int for uniform buffer
        counts,
        hlen, hplayers, hactions,
    ])


class PostflopBatch:
    """Python wrapper for C++ PostflopGame batch state machine."""

    def __init__(self, n: int):
        self.n      = n
        self._ptr   = _c_lib.c_postflop_alloc(n)
        # Pre-allocate output arrays
        self._feats     = np.zeros((n, FEATURE_DIM), dtype=np.float32)
        self._valid     = np.zeros((n, 8), dtype=np.int32)
        self._n_valid   = np.zeros(n, dtype=np.int32)
        self._players   = np.zeros(n, dtype=np.int32)
        self._idx       = np.zeros(n, dtype=np.int32)

    def init_one(self, i: int, state, p0_hand, p1_hand,
                 p0_hand5, p1_hand5, community, p0_disc, p1_disc,
                 traversing_player: int):
        flat = serialize_gamestate(state)
        flat_c = flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        _c_lib.c_postflop_init_one(
            self._ptr, i, flat_c,
            (ctypes.c_int*2)(*list(p0_hand)[:2]),
            (ctypes.c_int*2)(*list(p1_hand)[:2]),
            (ctypes.c_int*5)(*list(p0_hand5)[:5]),
            (ctypes.c_int*5)(*list(p1_hand5)[:5]),
            (ctypes.c_int*5)(*list(community)[:5]),
            (ctypes.c_int*3)(*list(p0_disc)[:3]),
            (ctypes.c_int*3)(*list(p1_disc)[:3]),
            ctypes.c_int(traversing_player),
        )

    def collect_pending(self):
        cnt = _c_lib.c_postflop_collect_pending(
            self._ptr, self.n,
            self._feats.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._valid.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            self._n_valid.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            self._players.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            self._idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        )
        return (cnt,
                self._feats[:cnt],
                self._valid[:cnt],
                self._n_valid[:cnt],
                self._players[:cnt],
                self._idx[:cnt])

    def resume(self, game_idxs: np.ndarray, net_advs: np.ndarray):
        n_p = len(game_idxs)
        if n_p == 0: return
        _c_lib.c_postflop_resume_batch(
            self._ptr,
            game_idxs.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            np.ascontiguousarray(net_advs, dtype=np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(n_p),
            ctypes.c_uint(random.randint(0, 2**31)),
        )

    def n_pending(self) -> int:
        return _c_lib.c_postflop_n_pending(self._ptr, self.n)

    def get_evs(self) -> np.ndarray:
        evs = np.zeros(self.n, dtype=np.float32)
        _c_lib.c_postflop_get_evs(
            self._ptr, self.n,
            evs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        return evs

    def collect_samples(self, iteration: float, traversing_player: int,
                        max_per_buf: int = 200_000):
        adv_f = np.zeros((max_per_buf, FEATURE_DIM), dtype=np.float32)
        adv_v = np.zeros((max_per_buf, 8),           dtype=np.float32)
        adv_m = np.zeros((max_per_buf, 8),           dtype=np.float32)
        adv_s = np.zeros(max_per_buf,                dtype=np.int32)
        adv_p = np.zeros(max_per_buf,                dtype=np.int32)
        adv_i = np.zeros(max_per_buf,                dtype=np.float32)
        str_f = np.zeros((max_per_buf, FEATURE_DIM), dtype=np.float32)
        str_v = np.zeros((max_per_buf, 8),           dtype=np.float32)
        str_m = np.zeros((max_per_buf, 8),           dtype=np.float32)
        str_s = np.zeros(max_per_buf,                dtype=np.int32)
        str_i = np.zeros(max_per_buf,                dtype=np.float32)
        na_out = ctypes.c_int(0); ns_out = ctypes.c_int(0)
        _c_lib.c_postflop_collect_samples(
            self._ptr, self.n,
            adv_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            adv_v.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            adv_m.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            adv_s.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            adv_p.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            adv_i.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(na_out),
            str_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            str_v.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            str_m.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            str_s.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            str_i.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(ns_out),
            ctypes.c_float(iteration),
            ctypes.c_int(traversing_player),
            ctypes.c_int(max_per_buf),
        )
        na, ns = na_out.value, ns_out.value
        return (adv_f[:na], adv_v[:na], adv_m[:na], adv_s[:na], adv_p[:na], adv_i[:na],
                str_f[:ns], str_v[:ns], str_m[:ns], str_s[:ns], str_i[:ns])

    def free(self):
        if self._ptr:
            _c_lib.c_postflop_free(self._ptr)
            self._ptr = None

    def __del__(self): self.free()


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

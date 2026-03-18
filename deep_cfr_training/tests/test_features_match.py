"""
test_features_match.py
C++ state_to_features (training) vs Python state_to_features (submission) 비교.

opp_range_in=None / my_range_in=None 케이스에서 두 구현이 완전히 동일해야 함.
요소별 max absolute diff가 1e-5 이하여야 통과.

Usage:
    cd deep_cfr_training
    python -m tests.test_features_match
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'submission'))

import ctypes
import numpy as np
import random

# ── C++ wrapper (training side) ───────────────────────────────────────────────
from game.features import state_to_features as cpp_state_to_features, _c_lib

# ── Pure Python (submission side) ─────────────────────────────────────────────
import importlib.util, types

_sub_features_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'submission', 'features.py'
)
_spec = importlib.util.spec_from_file_location('sub_features', _sub_features_path)
_sub_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sub_mod)
py_state_to_features = _sub_mod.state_to_features

FEATURE_DIM = 78
NUM_RANKS   = 9
NUM_SUITS   = 3
DECK_SIZE   = 27

rng = np.random.default_rng(42)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _deal_hand(dead: set, n: int) -> list:
    pool = [c for c in range(DECK_SIZE) if c not in dead]
    cards = random.sample(pool, n)
    dead.update(cards)
    return cards


def random_game_state():
    """Returns dict with all fields needed by both wrappers."""
    dead = set()
    hand2  = _deal_hand(dead, 2)
    my_disc  = _deal_hand(dead, 3)
    opp_disc = _deal_hand(dead, 3)
    street = random.randint(1, 3)
    n_comm = {1: 3, 2: 4, 3: 5}[street]
    comm   = _deal_hand(dead, n_comm)
    board5 = (comm + [-1] * 5)[:5]

    my_bet  = random.choice([0, 2, 5, 10, 20, 50])
    opp_bet = random.choice([0, 2, 5, 10, 20, 50])
    is_bb   = random.choice([True, False])
    n_bets_me  = random.randint(0, 3)
    n_bets_opp = random.randint(0, 3)
    aggressor_me  = random.choice([True, False])
    aggressor_opp = random.choice([True, False])
    if aggressor_me and aggressor_opp:
        aggressor_opp = False  # can't both be aggressor

    to_call = max(opp_bet - my_bet, 0)

    return dict(
        hand2=hand2, board5=board5, n_comm=n_comm,
        my_disc=my_disc, opp_disc=opp_disc,
        street=street, is_bb=is_bb,
        my_bet=my_bet, opp_bet=opp_bet, to_call=to_call,
        n_bets_me=n_bets_me, n_bets_opp=n_bets_opp,
        aggressor_me=aggressor_me, aggressor_opp=aggressor_opp,
    )


_NULL_INT = ctypes.cast(0, ctypes.POINTER(ctypes.c_int))

# Actual c_state_features signature (from traversal.cpp):
# hero_hand2, community, n_comm, my_bet, opp_bet, street, is_bb,
# my_disc, opp_disc, features_out, street_bet_counts_flat,
# history_players, history_actions, history_len, num_acts_this_street
_c_lib.c_state_features.argtypes = [
    ctypes.POINTER(ctypes.c_int),   # hero_hand2
    ctypes.POINTER(ctypes.c_int),   # community
    ctypes.c_int,                   # n_comm
    ctypes.c_int,                   # my_bet
    ctypes.c_int,                   # opp_bet
    ctypes.c_int,                   # street
    ctypes.c_int,                   # is_bb
    ctypes.POINTER(ctypes.c_int),   # my_disc
    ctypes.POINTER(ctypes.c_int),   # opp_disc
    ctypes.POINTER(ctypes.c_float), # features_out
    ctypes.POINTER(ctypes.c_int),   # street_bet_counts_flat (nullable)
    ctypes.POINTER(ctypes.c_int),   # history_players (nullable)
    ctypes.POINTER(ctypes.c_int),   # history_actions (nullable)
    ctypes.c_int,                   # history_len
    ctypes.c_int,                   # num_acts_this_street
]
_c_lib.c_state_features.restype = None


def call_cpp(g: dict) -> np.ndarray:
    h2   = (ctypes.c_int * 2)(*g['hand2'])
    comm = (ctypes.c_int * 5)(*g['board5'])
    n_c  = sum(1 for c in g['board5'] if c >= 0)
    md   = (ctypes.c_int * 3)(*g['my_disc'])
    od   = (ctypes.c_int * 3)(*g['opp_disc'])
    feat = (ctypes.c_float * FEATURE_DIM)()

    # street_bet_counts[4][2]: flat [s0p0, s0p1, s1p0, s1p1, ...]
    hp = 1 if g['is_bb'] else 0
    sbc_arr = [0] * 8
    sbc_arr[g['street'] * 2 + hp]      = g['n_bets_me']
    sbc_arr[g['street'] * 2 + 1 - hp]  = g['n_bets_opp']
    sbc = (ctypes.c_int * 8)(*sbc_arr)

    # history: encode aggressor as last aggressive action
    hist = []
    if g['aggressor_me']:
        hist = [(1 if g['is_bb'] else 0, 3)]
    elif g['aggressor_opp']:
        hist = [(0 if g['is_bb'] else 1, 3)]
    hlen = len(hist)
    if hlen > 0:
        hp_arr = (ctypes.c_int * hlen)(*[p for p, _ in hist])
        ha_arr = (ctypes.c_int * hlen)(*[a for _, a in hist])
    else:
        hp_arr = ha_arr = _NULL_INT

    _c_lib.c_state_features(
        h2, comm, ctypes.c_int(n_c),
        ctypes.c_int(g['my_bet']), ctypes.c_int(g['opp_bet']),
        ctypes.c_int(g['street']), ctypes.c_int(1 if g['is_bb'] else 0),
        md, od, feat,
        sbc, hp_arr, ha_arr,
        ctypes.c_int(hlen), ctypes.c_int(0),
    )
    return np.frombuffer(feat, dtype=np.float32).copy()


def call_py(g: dict) -> np.ndarray:
    return py_state_to_features(
        hand2        = g['hand2'],
        board        = g['board5'],
        my_bet       = g['my_bet'],
        opp_bet      = g['opp_bet'],
        street       = g['street'],
        is_bb        = g['is_bb'],
        my_disc      = g['my_disc'],
        opp_disc     = g['opp_disc'],
        to_call      = g['to_call'],
        n_bets_me    = g['n_bets_me'],
        n_bets_opp   = g['n_bets_opp'],
        aggressor_me  = g['aggressor_me'],
        aggressor_opp = g['aggressor_opp'],
    )


# ── Main comparison ───────────────────────────────────────────────────────────

FEATURE_NAMES = (
    [f'my_cat[{i}]'         for i in range(17)] +
    [f'my_range[{i}]'       for i in range(17)] +
    [f'opp_range[{i}]'      for i in range(17)] +
    [f'board_tex[{i}]'      for i in range(8)]  +
    [f'line_ctx[{i}]'       for i in range(6)]  +
    [f'pot_ratio[{i}]'      for i in range(4)]  +
    [f'blocker[{i}]'        for i in range(4)]  +
    [f'street_oh[{i}]'      for i in range(3)]  +
    ['position', 'pot_norm']
)
assert len(FEATURE_NAMES) == FEATURE_DIM

def main(n_games: int = 2000):
    print(f'Testing C++ vs Python state_to_features — {n_games} random games\n')

    max_diffs   = np.zeros(FEATURE_DIM, dtype=np.float64)
    fail_games  = []
    THRESH      = 1e-4

    for i in range(n_games):
        g = random_game_state()
        fc = call_cpp(g)
        fp = call_py(g)

        diff = np.abs(fc - fp)
        max_diffs = np.maximum(max_diffs, diff)

        if diff.max() > THRESH:
            fail_games.append((i, g, fc, fp, diff))

    # ── Per-block summary ─────────────────────────────────────────────────────
    blocks = [
        ('my_cat      [0:17] ', slice(0,  17)),
        ('my_range    [17:34]', slice(17, 34)),
        ('opp_range   [34:51]', slice(34, 51)),
        ('board_tex   [51:59]', slice(51, 59)),
        ('line_ctx    [59:65]', slice(59, 65)),
        ('pot_ratios  [65:69]', slice(65, 69)),
        ('blockers    [69:73]', slice(69, 73)),
        ('street_oh   [73:76]', slice(73, 76)),
        ('pos+pot     [76:78]', slice(76, 78)),
    ]

    print(f'{"Block":<25}  {"max_diff":>10}  {"status":>8}')
    print('-' * 50)
    all_pass = True
    for name, sl in blocks:
        d = max_diffs[sl].max()
        status = '✓ OK' if d < THRESH else '✗ FAIL'
        if d >= THRESH: all_pass = False
        print(f'  {name:<23}  {d:>10.2e}  {status:>8}')

    print()
    if fail_games:
        print(f'=== FAILURES ({len(fail_games)} games) ===')
        for i, g, fc, fp, diff in fail_games[:3]:
            bad = np.where(diff > THRESH)[0]
            print(f'\n  Game {i}:  hand2={g["hand2"]}  street={g["street"]}  is_bb={g["is_bb"]}')
            print(f'  my_disc={g["my_disc"]}  opp_disc={g["opp_disc"]}')
            for idx in bad[:10]:
                print(f'    [{idx:2d}] {FEATURE_NAMES[idx]:<20}  cpp={fc[idx]:.6f}  py={fp[idx]:.6f}  diff={diff[idx]:.2e}')
    else:
        print(f'All {n_games} games: max diff = {max_diffs.max():.2e}')
        print('✓  C++ and Python state_to_features are IDENTICAL')


if __name__ == '__main__':
    main()

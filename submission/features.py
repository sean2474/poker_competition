"""
Pure Python 77-dim feature extraction — mirrors deep_cfr_training/cpp/cfr/features.h.
No C++ dependencies. Used in submission/ only.

Card encoding: card = suit * 9 + rank  (rank 0-8, suit 0-2)
"""

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────
NUM_RANKS  = 9
NUM_SUITS  = 3
MAX_BET    = 100
N_CATS     = 17
N_HANDS    = 351   # C(27,2)
FEATURE_DIM = 78

def card_rank(c): return c % NUM_RANKS
def card_suit(c): return c // NUM_RANKS

# ── All 351 pairs (precomputed) ───────────────────────────────────────────────
_ALL_PAIRS  = [(a, b) for a in range(27) for b in range(a + 1, 27)]
_PAIR_IDX   = {(a, b): i for i, (a, b) in enumerate(_ALL_PAIRS)}
def pidx(c0, c1): return _PAIR_IDX[(c0,c1)] if c0<c1 else _PAIR_IDX[(c1,c0)]

# ── KEEP_PAIRS (discard) ──────────────────────────────────────────────────────
KEEP_PAIRS = [(i, j) for i in range(5) for j in range(i+1, 5)]

# ── Hand category classifier (mirrors core/hand_eval.h::classify_hand) ───────

def classify_hand(c0: int, c1: int, board: list, n_board: int) -> int:
    """Returns HandCat index 0-16 for (c0,c1) given board cards."""
    use    = min(n_board, 5)
    n      = 2 + use
    cards  = [c0, c1] + [board[i] for i in range(use)]
    ranks  = [card_rank(c) for c in cards]
    suits  = [card_suit(c) for c in cards]
    brank  = sorted([ranks[i] for i in range(2, n)], reverse=True)
    nb     = len(brank)

    # Rank/suit counts
    rcnt = {}
    for r in ranks:
        rcnt[r] = rcnt.get(r, 0) + 1
    scnt = [0, 0, 0]
    for s in suits:
        scnt[s] += 1
    max_suit = max(scnt)

    # ── Made hands (n >= 5) ───────────────────────────────────────────────────
    if n >= 5:
        is_str = False
        for lo in range(NUM_RANKS - 4):
            if sum(1 for r in ranks if lo <= r < lo + 5) >= 5:
                is_str = True; break
        if max_suit >= 5 and is_str: return 0   # SF
        if max_suit >= 5:            return 2   # flush
        if is_str:                   return 3   # straight

    trip_rank = next((r for r in sorted(rcnt, reverse=True) if rcnt[r] >= 3), -1)
    n_trips   = sum(1 for v in rcnt.values() if v >= 3)
    pair_ranks = sorted([r for r, v in rcnt.items() if v == 2], reverse=True)
    n_pairs   = len(pair_ranks)

    if n_trips >= 1 and n_pairs >= 1: return 1   # full house
    if n_trips >= 1:
        if nb == 0:                   return 4   # top_set
        if trip_rank == brank[0]:     return 4   # top_set
        if trip_rank == brank[-1]:    return 6   # bot_set
        return 5                                  # mid_set
    if n_pairs >= 2:
        if nb > 0 and len(pair_ranks) >= 2 and pair_ranks[0] < brank[0] and pair_ranks[1] < brank[0]:
            return 8                              # bottom_two
        return 7                                  # two_pair

    r0, r1 = card_rank(c0), card_rank(c1)
    if n_pairs == 1:
        pr = pair_ranks[0]
        if r0 == r1:
            over = all(pr >= b for b in brank) if brank else True
            if over: return 9                     # overpair
        if nb == 0:               return 11       # mid_pair
        if pr == brank[0]:        return 10       # top_pair
        if pr == brank[-1]:       return 12       # bot_pair
        return 11                                 # mid_pair

    # ── Draw detection ────────────────────────────────────────────────────────
    if max_suit >= 4 and n >= 4:
        for s in range(NUM_SUITS):
            sr2 = sorted([ranks[i] for i in range(n) if suits[i] == s])
            if len(sr2) < 4: continue
            for lo in range(NUM_RANKS - 3):
                if sum(1 for r in sr2 if lo <= r < lo + 4) >= 4: return 13  # SF_draw
    if max_suit >= 4: return 14                   # flush_draw
    if n >= 4:
        present = set(ranks)
        for lo in range(NUM_RANKS - 3):
            if sum(1 for k in range(lo, lo+4) if k in present) >= 4: return 15  # str_draw
    return 16                                     # high_card


# ── Blocker flags (mirrors core/hand_eval.h::compute_blocker_flags) ──────────

def compute_blocker_flags(c0: int, c1: int, board: list, n_board: int) -> np.ndarray:
    out = np.zeros(4, dtype=np.float32)
    if c0 < 0 or n_board == 0:
        return out
    hr = [card_rank(c0), card_rank(c1)]
    hs = [card_suit(c0), card_suit(c1)]
    b_comm = [board[i] for i in range(n_board) if board[i] >= 0]
    if not b_comm:
        return out
    bsc = [0, 0, 0]
    b_set = set()
    top_r, sec_r = -1, -1
    for c in b_comm:
        r, s = card_rank(c), card_suit(c)
        bsc[s] += 1
        b_set.add(r)
        if r > top_r: sec_r, top_r = top_r, r
        elif r > sec_r: sec_r = r
    dom = bsc.index(max(bsc))
    out[0] = 1. if (hr[0]==top_r or hr[1]==top_r) else 0.
    out[1] = 1. if (top_r>=0 and sec_r>=0 and
                    (hr[0]==top_r or hr[1]==top_r) and
                    (hr[0]==sec_r or hr[1]==sec_r)) else 0.
    out[2] = 1. if (hs[0]==dom or hs[1]==dom) else 0.
    for r in hr:
        for ws in range(max(0, r-4), min(5, r+1)):
            cnt = sum(1 for k in range(ws, ws+5) if k < NUM_RANKS and k in b_set)
            if cnt >= 3 and ws <= r < ws + 5:
                out[3] = 1.; break
        if out[3]: break
    return out


# ── Fast discard score (mirrors heuristic/discard.h::fast_score) ─────────────

def fast_score(c0: int, c1: int, board3: list) -> float:
    r0, r1 = card_rank(c0), card_rank(c1)
    s0, s1 = card_suit(c0), card_suit(c1)
    sc = 0.
    if r0 == r1: sc += 10.
    sc += max(r0, r1) * 0.5
    if s0 == s1: sc += 3.
    for b in board3:
        if b < 0: continue
        br, bs = card_rank(b), card_suit(b)
        if br == r0 or br == r1: sc += 5.
        if abs(br - r0) <= 1 or abs(br - r1) <= 1: sc += 1.
        if bs == s0 and s0 == s1: sc += 2.
    return sc


def best_discard_keep(hand5: list, board3: list) -> tuple:
    """Return (ki, kj) indices 0-4 of best 2 cards to keep."""
    best, best_s = (0, 1), -1e9
    for ki, kj in KEEP_PAIRS:
        s = fast_score(hand5[ki], hand5[kj], board3)
        if s > best_s:
            best_s, best = s, (ki, kj)
    return best


# ── Range estimation (pure Python Bayesian, mirrors c_range_* functions) ──────

def _range_uniform(dead_cards: list) -> np.ndarray:
    dead = set(c for c in dead_cards if c >= 0)
    r = np.array([0. if (a in dead or b in dead) else 1.
                  for a, b in _ALL_PAIRS], dtype=np.float32)
    s = r.sum()
    return r / s if s > 0 else r


def _range_update_discard(probs: np.ndarray, disc3: list) -> np.ndarray:
    """
    Remove hands that overlap with observed discards (uniform weight among survivors).
    Proper DiscardCFR-weighted update is done separately in Python when net is available.
    Mirrors the corrected c_range_update_discard in range/rangefinder.cpp.
    """
    disc_set = set(c for c in disc3 if c >= 0)
    if not disc_set:
        return probs
    new = probs.copy()
    for i, (a, b) in enumerate(_ALL_PAIRS):
        if a in disc_set or b in disc_set:
            new[i] = 0.
    s = new.sum()
    return new / s if s > 0 else probs


def _range_to_cats(probs: np.ndarray, board: list, n_board: int) -> np.ndarray:
    cats = np.zeros(N_CATS, dtype=np.float32)
    b5   = (list(board[:n_board]) + [-1]*5)[:5]
    for i, (c0, c1) in enumerate(_ALL_PAIRS):
        if probs[i] > 1e-9:
            cats[classify_hand(c0, c1, b5, n_board)] += probs[i]
    s = cats.sum()
    return cats / s if s > 0 else np.ones(N_CATS, dtype=np.float32) / N_CATS


def my_range_cats(my_disc: list, board: list, n_board: int) -> np.ndarray:
    """17-dim: opp's view of my range (dead=community, update=my_disc)."""
    dead = [b for b in board[:n_board] if b >= 0]
    probs = _range_uniform(dead)
    if any(c >= 0 for c in my_disc):
        probs = _range_update_discard(probs, my_disc)
    return _range_to_cats(probs, board, n_board)


def opp_range_cats(hand2: list, opp_disc: list, board: list, n_board: int) -> np.ndarray:
    """17-dim: my view of opp range (dead=hand2+community, update=opp_disc)."""
    dead = [c for c in hand2 if c >= 0] + [b for b in board[:n_board] if b >= 0]
    probs = _range_uniform(dead)
    if any(c >= 0 for c in opp_disc):
        probs = _range_update_discard(probs, opp_disc)
    return _range_to_cats(probs, board, n_board)


def my_cat_onehot(c0: int, c1: int, board: list, n_board: int) -> np.ndarray:
    """17-dim one-hot for my actual hand category."""
    cat = classify_hand(c0, c1, (list(board[:n_board]) + [-1]*5)[:5], n_board)
    out = np.zeros(N_CATS, dtype=np.float32)
    out[cat] = 1.
    return out


# ── Board texture (mirrors cfr/features.h board_texture block) ───────────────

def board_texture(board: list, n_board: int) -> np.ndarray:
    """8-dim board texture features."""
    b_comm = [board[i] for i in range(n_board) if board[i] >= 0]
    f = np.zeros(8, dtype=np.float32)
    if not b_comm: return f
    ranks = [card_rank(c) for c in b_comm]
    suits = [card_suit(c) for c in b_comm]
    bsc   = [0, 0, 0]
    for s in suits: bsc[s] += 1
    msc   = max(bsc)
    nb    = len(b_comm)
    min_r, max_r = min(ranks), max(ranks)
    seen, paired = {}, False
    for r in ranks:
        if r in seen: paired = True
        seen[r] = True
    n_suits = sum(1 for s in bsc if s > 0)
    f[0] = 1. if paired else 0.                               # paired
    f[1] = 1. if (nb > 0 and msc == nb) else 0.              # flush_complete
    f[2] = 1. if msc >= 2 else 0.                             # fd_present
    if nb >= 3: f[3] = 1. if (max_r - min_r) <= 4 else 0.    # connected
    f[4] = max_r / 8.                                          # high_rank
    f[5] = 1. if (nb > 0 and n_suits == 3) else 0.            # rainbow (3 different suits)
    f[6] = 1. if (nb > 0 and n_suits == 2) else 0.           # two_suited
    f[7] = 1. if (paired and (msc >= 2 or (nb >= 3 and (max_r-min_r) <= 4))) else 0.  # coord
    return f


# ── Full 77-dim state_to_features ────────────────────────────────────────────

def state_to_features(
    hand2:   list,        # [c0, c1] kept 2-card hand
    board:   list,        # community cards, -1 = not yet dealt (5 elements)
    my_bet:  int,
    opp_bet: int,
    street:  int,         # 1=flop, 2=turn, 3=river
    is_bb:   bool,
    my_disc:  list,       # [3] my discarded cards
    opp_disc: list,       # [3] opp's discarded cards
    to_call:  int,
    n_bets_me:  int = 0,
    n_bets_opp: int = 0,
    aggressor_me:  bool = False,
    aggressor_opp: bool = False,
) -> np.ndarray:
    f   = np.zeros(FEATURE_DIM, dtype=np.float32)
    c0, c1  = hand2[0], hand2[1]
    n_comm  = sum(1 for c in board if c >= 0)
    pot     = max(my_bet + opp_bet, 1)
    fpot    = float(pot)

    # [0-16] my_cat
    f[0:17]  = my_cat_onehot(c0, c1, board, n_comm)

    # [17-33] my_range_cats (opp's view via my discards)
    f[17:34] = my_range_cats(my_disc, board, n_comm)

    # [34-50] opp_range_cats (my view via opp discards)
    f[34:51] = opp_range_cats(hand2, opp_disc, board, n_comm)

    # [51-58] board_texture
    f[51:59] = board_texture(board, n_comm)

    # [59-64] line_context
    f[59] = 1. if aggressor_me  else 0.
    f[60] = 1. if aggressor_opp else 0.
    f[61] = 1. if to_call > 0   else 0.
    f[62] = 1. if to_call == 0  else 0.
    f[63] = min(n_bets_me  / 4., 1.)
    f[64] = min(n_bets_opp / 4., 1.)

    # [65-68] pot_ratios
    raise_room = max(MAX_BET - max(my_bet, opp_bet), 0)
    f[65] = to_call    / fpot
    f[66] = my_bet     / fpot
    f[67] = opp_bet    / fpot
    f[68] = raise_room / fpot

    # [69-72] blocker_flags
    f[69:73] = compute_blocker_flags(c0, c1, board, n_comm)

    # [73-75] street one-hot
    if 1 <= street <= 3:
        f[73 + street - 1] = 1.

    # [76] position
    f[76] = 1. if is_bb else 0.

    # [77] pot / MAX_BET
    f[77] = min(pot / float(MAX_BET), 1.0)

    return f

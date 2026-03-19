from collections import Counter
from itertools import combinations
import random
import numpy as np

from game.game import DECK_SIZE, NUM_RANKS, NUM_SUITS
from game.game import rank as _rank, suit as _suit

# All valid 2-card pairs in 27-card deck
_ALL_PAIR = [(i, j) for i in range(DECK_SIZE) for j in range(i + 1, DECK_SIZE)]

# Lazy-loaded evaluator (avoid slow import at module load)
_EVAL = None
_CARD_TABLE = None  # int → treys card int, length DECK_SIZE


def _get_eval():
    global _EVAL, _CARD_TABLE
    if _EVAL is None:
        from gym_env import PokerEnv, WrappedEval
        _EVAL = WrappedEval()
        _CARD_TABLE = [PokerEnv.int_to_card(c) for c in range(DECK_SIZE)]
    return _EVAL, _CARD_TABLE


# ── 17-category hand classification ──────────────────────────────────────────
#
#   0  straight_flush         9  overpair
#   1  full_house            10  top_pair
#   2  flush                 11  middle_pair
#   3  straight              12  bottom_pair
#   4  top_set               13  sf_draw  (4 suited+consecutive)
#   5  middle_set            14  flush_draw  (4+ same suit)
#   6  bottom_set            15  straight_draw  (4 consecutive ranks)
#   7  two_pair              16  high_card
#   8  bottom_two

def hand_category(c0: int, c1: int, board: list) -> int:
    """Classify 2-card hand vs 3-card board into one of 17 categories (0=best)."""
    cards = [c0, c1] + list(board)
    all_r  = [_rank(c) for c in cards]
    all_s  = [_suit(c) for c in cards]
    brd_r  = sorted((_rank(c) for c in board), reverse=True)  # [top, mid, bot]

    rcnt   = Counter(all_r)
    scnt   = Counter(all_s)
    counts = sorted(rcnt.values(), reverse=True)
    max_s  = max(scnt.values())
    uniq_r = sorted(set(all_r))

    has_straight = len(uniq_r) >= 5 and uniq_r[-1] - uniq_r[0] == 4
    has_flush    = max_s >= 5

    if has_flush and has_straight:                           return 0
    if len(counts) >= 2 and counts[0] >= 3 and counts[1] >= 2: return 1
    if has_flush:                                            return 2
    if has_straight:                                         return 3

    if counts[0] == 3:
        trip_r = next(r for r, c in rcnt.items() if c == 3)
        if trip_r == brd_r[0]:  return 4  # top_set
        if trip_r == brd_r[-1]: return 6  # bottom_set
        return 5                           # middle_set

    if len(counts) >= 2 and counts[0] == 2 and counts[1] == 2:
        pair_rs = sorted((r for r, c in rcnt.items() if c == 2), reverse=True)
        return 8 if pair_rs[0] < brd_r[0] else 7

    if counts[0] == 2:
        pair_r = next(r for r, c in rcnt.items() if c == 2)
        if _rank(c0) == _rank(c1) and pair_r > brd_r[0]: return 9   # overpair
        if pair_r == brd_r[0]:                            return 10  # top_pair
        if len(brd_r) >= 2 and pair_r == brd_r[1]:       return 11  # middle_pair
        return 12                                                     # bottom_pair

    # No made hand → check draws
    for s in range(NUM_SUITS):
        s_rs = sorted(_rank(c) for c in cards if _suit(c) == s)
        if len(s_rs) >= 4:
            for i in range(len(s_rs) - 3):
                if s_rs[i + 3] - s_rs[i] == 3:
                    return 13  # sf_draw
    if max_s >= 4:
        return 14  # flush_draw
    for i in range(len(uniq_r) - 3):
        if uniq_r[i + 3] - uniq_r[i] == 3:
            return 15  # straight_draw
    return 16  # high_card


# ── Blocker flags ─────────────────────────────────────────────────────────────

def blocker_flags(c0: int, c1: int, board: list) -> np.ndarray:
    """
    4-dim float array:
      [0] blocks_top_pair   — holds a card matching board's top rank
      [1] blocks_2pair      — holds BOTH top-two board ranks
      [2] blocks_flush      — holds a card of the nut (most common) suit
      [3] blocks_straight   — holds a rank completing a 4-straight on board
    """
    if not board:
        return np.zeros(4, dtype=np.float32)

    brd_r   = sorted((_rank(c) for c in board), reverse=True)
    nut_s   = Counter(_suit(c) for c in board).most_common(1)[0][0]
    my_r    = {_rank(c0), _rank(c1)}

    bt   = float(brd_r[0] in my_r)
    b2p  = float(len(brd_r) >= 2 and brd_r[0] in my_r and brd_r[1] in my_r)
    bfl  = float(_suit(c0) == nut_s or _suit(c1) == nut_s)

    brd_set = set(brd_r)
    bst = 0.
    for r in my_r:
        combined = brd_set | {r}
        for start in range(NUM_RANKS - 3):
            if len(combined & set(range(start, start + 4))) == 4:
                bst = 1.; break
        if bst: break

    return np.array([bt, b2p, bfl, bst], dtype=np.float32)


# ── Board texture ─────────────────────────────────────────────────────────────

def board_texture_features(c0: int, c1: int, board: list) -> np.ndarray:
    """
    6-dim: [board_highest, board_lowest, board_flush_possible, board_connected,
            my_highest, my_lowest]
      board_flush_possible: dominant suit count on board / 3  (wet flush indicator)
      board_connected:      length of longest rank run on board / 2
    Rank values normalized /8.
    """
    if not board:
        return np.zeros(6, dtype=np.float32)
    brd_rs  = sorted(_rank(c) for c in board)
    my_rs   = sorted([_rank(c0), _rank(c1)])

    flush_possible = max(Counter(_suit(c) for c in board).values()) / 3.

    uniq_brd = sorted(set(brd_rs))
    max_run = 1
    cur_run = 1
    for i in range(1, len(uniq_brd)):
        if uniq_brd[i] - uniq_brd[i - 1] == 1:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 1
    connected = max_run / 2.

    return np.array(
        [brd_rs[-1] / 8., brd_rs[0] / 8., flush_possible, connected,
         my_rs[-1] / 8., my_rs[0] / 8.],
        dtype=np.float32
    )


# ── Range → expected hand-category features ──────────────────────────────────

def range_features(range_vec: np.ndarray, board: list) -> np.ndarray:
    """
    17-dim probability-weighted hand-category distribution over a range.

    For each combo k in _ALL_PAIR:
      feat[hand_category(k, board)] += range_vec[k]

    Normalised so values sum to 1 (= expected category distribution).
    """
    feat = np.zeros(17, dtype=np.float32)
    w = np.asarray(range_vec, dtype=np.float32)
    s = w.sum()
    if s < 1e-9:
        feat[16] = 1.  # fallback: all weight on high_card
        return feat
    w = w / s
    for idx, (rc0, rc1) in enumerate(_ALL_PAIR):
        prob = float(w[idx])
        if prob < 1e-9:
            continue
        feat[hand_category(rc0, rc1, board)] += prob
    return feat


# ── Combined feature builder ──────────────────────────────────────────────────

def build_features(c0: int, c1: int, board: list,
                   opp_range: np.ndarray,
                   hero_range: np.ndarray) -> np.ndarray:
    """
    44-dim feature vector (strategic context only, no raw hand-category):
      blocker_flags (4)
      + board_texture (6)
      + range_features(opp_range)  (17)  — expected opp hand-category dist
      + range_features(hero_range) (17)  — expected hero hand-category dist
      = 44

    Hand strength is captured separately via EV computation.
    At inference: score[a] = EV[a] + net(features_a) -> softmax -> strategy.
    """
    return np.concatenate([
        blocker_flags(c0, c1, board),
        board_texture_features(c0, c1, board),
        range_features(opp_range, board),
        range_features(hero_range, board),
    ])


# ── EV computation ────────────────────────────────────────────────────────────

def calculate_ev(board: list, my_pair: tuple, opp_range: np.ndarray,
                 dead: set, threshold: float = 0.001) -> float:
    """
    EV (win rate) for keeping my_pair vs weighted opp range.

    Exact mode (n_samples=None):
      For each opp_pair with P >= threshold:
        Enumerate all (turn, river) pairs from remaining deck.
        ev += P(opp_pair) * (wins / total_boards)

    dead: all cards not in play (hero original hand + board + opp discards)
    """
    ev_obj, card_tbl = _get_eval()
    opp_w = np.asarray(opp_range, dtype=np.float64)

    c0t, c1t = card_tbl[my_pair[0]], card_tbl[my_pair[1]]

    ev = 0.
    total_w = 0.

    for idx, (oc0, oc1) in enumerate(_ALL_PAIR):
        prob = float(opp_w[idx])
        if prob < threshold or oc0 in dead or oc1 in dead:
            continue

        opp_dead = dead | {oc0, oc1}
        remaining = [c for c in range(DECK_SIZE) if c not in opp_dead]
        if len(remaining) < 2:
            continue

        oc0t, oc1t = card_tbl[oc0], card_tbl[oc1]

        boards = list(combinations(remaining, 2))

        wins = 0
        for (turn, river) in boards:
            brd = [card_tbl[c] for c in board] + [card_tbl[turn], card_tbl[river]]
            if ev_obj.evaluate([c0t, c1t], brd) < ev_obj.evaluate([oc0t, oc1t], brd):
                wins += 1

        ev += prob * wins / len(boards)
        total_w += prob

    return ev / total_w
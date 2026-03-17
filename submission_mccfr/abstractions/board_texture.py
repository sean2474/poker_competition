"""
Board texture bucketing for flop, turn, and river.

Each street computes texture INDEPENDENTLY (not recursively).
A small delta (0-2) captures what the new card changed.

Flop:  pairedness(3) x suitedness(3) x connectivity(3) x height(3) = ~81
Turn:  4-card texture(~30) x delta(3) = ~90
River: 5-card texture(~40) x delta(3) = ~120
"""

from abstractions.card_utils import (
    card_rank, card_suit, has_ace, rank_counts, suit_counts,
    sorted_ranks, NUM_RANKS, ACE_RANK_IDX, has_straight_potential,
)


# ═══════════════════════════════════════════════════════════════════
# Shared texture feature helpers
# ═══════════════════════════════════════════════════════════════════

def _pairedness(board: list) -> int:
    """0=unpaired, 1=one pair, 2=two pair/trips+."""
    rc = rank_counts(board)
    mx = max(rc.values())
    num_pairs = sum(1 for v in rc.values() if v >= 2)
    if mx >= 3 or num_pairs >= 2:
        return 2
    if mx == 2:
        return 1
    return 0


def _flush_pressure(board: list) -> int:
    """0=no flush threat, 1=draw possible (3 suited), 2=flush on board (4+ suited)."""
    sc = suit_counts(board)
    mx = max(sc.values())
    if mx >= 4:
        return 2
    if mx >= 3:
        return 1
    return 0


def _straight_pressure(board: list) -> int:
    """0=no straight threat, 1=draw possible, 2=straight on board."""
    rank_set = set(card_rank(c) for c in board)
    if has_straight_potential(rank_set, 5):
        return 2
    if has_straight_potential(rank_set, 4):
        return 1
    return 0


def _height(board: list) -> int:
    """0=low (max<=4), 1=medium (5-7), 2=ace-high."""
    if has_ace(board):
        return 2
    mx = max(card_rank(c) for c in board)
    if mx <= 4:
        return 0
    return 1


def _connectivity(board: list) -> int:
    """0=disconnected, 1=semi-connected, 2=highly connected."""
    ranks = sorted_ranks(board)
    rank_set = set(ranks)

    if has_straight_potential(rank_set, 3):
        gaps = [ranks[i+1] - ranks[i] for i in range(len(ranks)-1)]
        max_gap = max(gaps) if gaps else 0
        span = ranks[-1] - ranks[0]
        if ACE_RANK_IDX in rank_set:
            low_ranks = [r for r in ranks if r != ACE_RANK_IDX]
            if low_ranks:
                span = min(span, max(low_ranks) + 1)
        if span <= 4 and max_gap <= 2:
            return 2
        return 1
    return 0


# ═══════════════════════════════════════════════════════════════════
# Delta: what did the new card change? (used for turn and river)
# ═══════════════════════════════════════════════════════════════════

def _card_delta(prev_board: list, new_card: int) -> int:
    """
    Classify impact of a new card relative to previous board.
    0=blank, 1=scare (pairs/flush/straight completing), 2=ace scare
    """
    prev_ranks = set(card_rank(c) for c in prev_board)
    prev_suits = suit_counts(prev_board)
    nr = card_rank(new_card)
    ns = card_suit(new_card)

    # Ace scare on non-ace board
    if nr == ACE_RANK_IDX and not has_ace(prev_board):
        return 2

    # Pairs the board
    if nr in prev_ranks:
        return 1

    # Flush completing / threatening
    new_suit_count = prev_suits.get(ns, 0) + 1
    if new_suit_count >= 3:
        return 1

    # Straight completing
    all_ranks = prev_ranks | {nr}
    needed = 5 if len(prev_board) >= 4 else 4
    if has_straight_potential(all_ranks, needed):
        return 1

    return 0  # blank


# ═══════════════════════════════════════════════════════════════════
# Flop bucket (3 cards) — same as before
# ═══════════════════════════════════════════════════════════════════

def flop_bucket(board_3: list) -> int:
    """
    pairedness(3) × suitedness(3) × connectivity(3) × height(3) = 81 max.
    """
    p = _pairedness(board_3)
    s = _flush_pressure(board_3)  # 0=rainbow, 1=two-suited, 2=mono (mapped from flush_pressure logic)
    c = _connectivity(board_3)
    h = _height(board_3)
    return p * 27 + s * 9 + c * 3 + h


def flop_bucket_tuple(board_3: list) -> tuple:
    return (_pairedness(board_3), _flush_pressure(board_3),
            _connectivity(board_3), _height(board_3))


# ═══════════════════════════════════════════════════════════════════
# Turn bucket (4 cards) — independent texture + delta
# ═══════════════════════════════════════════════════════════════════

def _turn_texture(board_4: list) -> int:
    """
    Direct 4-card board texture.
    pairedness(3) × flush_pressure(3) × straight_pressure(3) × height(3) = 81
    but many combos impossible, effective ~30.
    """
    p = _pairedness(board_4)
    f = _flush_pressure(board_4)
    s = _straight_pressure(board_4)
    h = _height(board_4)
    return p * 27 + f * 9 + s * 3 + h


def turn_bucket(board_3: list, turn_card: int) -> int:
    """4-card texture × delta(3) = ~90 effective."""
    board_4 = board_3 + [turn_card]
    tex = _turn_texture(board_4)
    delta = _card_delta(board_3, turn_card)
    return tex * 3 + delta


def turn_bucket_tuple(board_3: list, turn_card: int) -> tuple:
    board_4 = board_3 + [turn_card]
    return (_pairedness(board_4), _flush_pressure(board_4),
            _straight_pressure(board_4), _height(board_4),
            _card_delta(board_3, turn_card))


# ═══════════════════════════════════════════════════════════════════
# River bucket (5 cards) — independent texture + delta
# ═══════════════════════════════════════════════════════════════════

def _river_texture(board_5: list) -> int:
    """
    Direct 5-card final board texture.
    pairedness(3) × flush_pressure(3) × straight_pressure(3) × height(3) = 81
    effective ~40.
    """
    p = _pairedness(board_5)
    f = _flush_pressure(board_5)
    s = _straight_pressure(board_5)
    h = _height(board_5)
    return p * 27 + f * 9 + s * 3 + h


def river_bucket(board_3: list, turn_card: int, river_card: int) -> int:
    """5-card texture × delta(3) = ~120 effective."""
    board_5 = board_3 + [turn_card, river_card]
    board_4 = board_3 + [turn_card]
    tex = _river_texture(board_5)
    delta = _card_delta(board_4, river_card)
    return tex * 3 + delta


def river_bucket_tuple(board_3: list, turn_card: int, river_card: int) -> tuple:
    board_5 = board_3 + [turn_card, river_card]
    board_4 = board_3 + [turn_card]
    return (_pairedness(board_5), _flush_pressure(board_5),
            _straight_pressure(board_5), _height(board_5),
            _card_delta(board_4, river_card))


# ═══════════════════════════════════════════════════════════════════
# Convenience: bucket from full board list + street
# ═══════════════════════════════════════════════════════════════════

def board_bucket_for_street(community_cards: list, street: int) -> int:
    if street == 0:
        return 0
    elif street == 1:
        return flop_bucket(community_cards[:3])
    elif street == 2:
        return turn_bucket(community_cards[:3], community_cards[3])
    else:
        return river_bucket(community_cards[:3], community_cards[3], community_cards[4])

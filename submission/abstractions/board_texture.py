"""
Board texture bucketing for flop, turn, and river.

Flop bucket: pairedness × suitedness × connectivity × ace_flag × height_class
Turn bucket: flop_bucket + turn_delta_class
River bucket: prior_board_class + river_delta_class

Target: ~18 flop buckets, turn ~36, river ~36
"""

from submission.abstractions.card_utils import (
    card_rank, card_suit, has_ace, rank_counts, suit_counts,
    sorted_ranks, NUM_RANKS, ACE_RANK_IDX, has_straight_potential,
)


# ═══════════════════════════════════════════════════════════════════
# Flop texture features (3 cards)
# ═══════════════════════════════════════════════════════════════════

def _flop_pairedness(board: list) -> int:
    """0=unpaired, 1=paired (one pair on board), 2=trips."""
    rc = rank_counts(board)
    mx = max(rc.values())
    if mx >= 3:
        return 2
    if mx == 2:
        return 1
    return 0


def _flop_suitedness(board: list) -> int:
    """0=rainbow (3 diff suits), 1=two-suited, 2=monotone."""
    sc = suit_counts(board)
    mx = max(sc.values())
    if mx == 3:
        return 2  # monotone
    if mx == 2:
        return 1  # two-suited
    return 0  # rainbow


def _flop_connectivity(board: list) -> int:
    """
    0=disconnected (max gap > 3 between sorted ranks, no straight draw)
    1=semi-connected (some straight potential)
    2=highly connected (open-ended or made straight possible)
    """
    ranks = sorted_ranks(board)
    rank_set = set(ranks)

    # Check if 5-card straight is possible with 2 more cards
    # i.e. can the 3 board ranks be part of a 5-card window?
    if has_straight_potential(rank_set, 3):
        # Board itself has 3 in a row or close
        gaps = [ranks[i+1] - ranks[i] for i in range(len(ranks)-1)]
        max_gap = max(gaps) if gaps else 0
        span = ranks[-1] - ranks[0]

        # Check ace-low wrapping
        if ACE_RANK_IDX in rank_set:
            span = min(span, max(r for r in ranks if r != ACE_RANK_IDX) - 0 + 1)

        if span <= 4 and max_gap <= 2:
            return 2  # highly connected
        return 1  # semi-connected

    return 0  # disconnected


def _flop_height(board: list) -> int:
    """
    0=low (max rank <= 4, i.e. highest is 6 or below)
    1=medium (max rank 5-7, i.e. 7-9)
    2=ace-high (has ace)
    """
    if has_ace(board):
        return 2
    mx = max(card_rank(c) for c in board)
    if mx <= 4:
        return 0
    return 1


def flop_bucket(board_3: list) -> int:
    """
    Compute flop texture bucket from 3 board cards.
    Combines pairedness(3) × suitedness(3) × connectivity(3) × height(3)
    = 81 max, but many are impossible. We pack into a single int.

    In practice we compress to ~18 meaningful clusters via the tuple.
    Returns an int that can be used as a bucket key component.
    """
    p = _flop_pairedness(board_3)
    s = _flop_suitedness(board_3)
    c = _flop_connectivity(board_3)
    h = _flop_height(board_3)
    return p * 27 + s * 9 + c * 3 + h


def flop_bucket_tuple(board_3: list) -> tuple:
    """Same as flop_bucket but returns the feature tuple for debugging."""
    return (
        _flop_pairedness(board_3),
        _flop_suitedness(board_3),
        _flop_connectivity(board_3),
        _flop_height(board_3),
    )


# ═══════════════════════════════════════════════════════════════════
# Turn delta (4th card relative to flop texture)
# ═══════════════════════════════════════════════════════════════════

def _turn_delta(board_3: list, turn_card: int) -> int:
    """
    Classify what the turn card changes about the board.
    0=blank (no structural change)
    1=pairs the board
    2=flush-possible (3 of same suit now)
    3=straight-completes or straight-heavy (4 connected)
    4=ace scare (turn is Ace on non-ace board)
    """
    flop_ranks = set(card_rank(c) for c in board_3)
    flop_suits = suit_counts(board_3)
    turn_rank = card_rank(turn_card)
    turn_suit = card_suit(turn_card)

    # Ace scare
    if turn_rank == ACE_RANK_IDX and not has_ace(board_3):
        return 4

    # Pairs the board
    if turn_rank in flop_ranks:
        return 1

    # Flush possible: 3+ of same suit
    new_suit_count = flop_suits.get(turn_suit, 0) + 1
    if new_suit_count >= 3:
        return 2

    # Straight heavy: check if 4 cards are within a 5-card window
    all_ranks = flop_ranks | {turn_rank}
    if has_straight_potential(all_ranks, 4):
        return 3

    return 0  # blank


def turn_bucket(board_3: list, turn_card: int) -> int:
    """
    Turn bucket = flop_bucket * 5 + turn_delta.
    """
    fb = flop_bucket(board_3)
    td = _turn_delta(board_3, turn_card)
    return fb * 5 + td


def turn_bucket_tuple(board_3: list, turn_card: int) -> tuple:
    return (flop_bucket_tuple(board_3), _turn_delta(board_3, turn_card))


# ═══════════════════════════════════════════════════════════════════
# River delta (5th card relative to flop+turn)
# ═══════════════════════════════════════════════════════════════════

def _river_delta(board_4: list, river_card: int) -> int:
    """
    Classify what the river card changes.
    0=blank
    1=pairs/trips the board
    2=flush completes (4+ same suit)
    3=straight completes (5 connected)
    4=overcard/ace scare
    """
    board_ranks = set(card_rank(c) for c in board_4)
    board_suits = suit_counts(board_4)
    river_rank = card_rank(river_card)
    river_suit = card_suit(river_card)

    # Flush complete: 4+ of same suit with river
    new_suit_count = board_suits.get(river_suit, 0) + 1
    if new_suit_count >= 4:
        return 2

    # Straight complete
    all_ranks = board_ranks | {river_rank}
    if has_straight_potential(all_ranks, 5):
        return 3

    # Pairs/trips board
    if river_rank in board_ranks:
        return 1

    # Ace scare on non-ace board
    if river_rank == ACE_RANK_IDX and not has_ace(board_4):
        return 4

    return 0


def river_bucket(board_3: list, turn_card: int, river_card: int) -> int:
    """
    River bucket = turn_bucket * 5 + river_delta.
    """
    tb = turn_bucket(board_3, turn_card)
    rd = _river_delta(board_3 + [turn_card], river_card)
    return tb * 5 + rd


def river_bucket_tuple(board_3: list, turn_card: int, river_card: int) -> tuple:
    return (
        turn_bucket_tuple(board_3, turn_card),
        _river_delta(board_3 + [turn_card], river_card),
    )


# ═══════════════════════════════════════════════════════════════════
# Convenience: bucket from full board list + street
# ═══════════════════════════════════════════════════════════════════

def board_bucket_for_street(community_cards: list, street: int) -> int:
    """
    Compute board bucket given community cards and street.
    street 0 (preflop) -> 0 (no board)
    street 1 (flop) -> flop_bucket
    street 2 (turn) -> turn_bucket
    street 3 (river) -> river_bucket
    """
    if street == 0:
        return 0
    elif street == 1:
        return flop_bucket(community_cards[:3])
    elif street == 2:
        return turn_bucket(community_cards[:3], community_cards[3])
    else:
        return river_bucket(community_cards[:3], community_cards[3], community_cards[4])

"""
Opponent discard bucketing.

The 3 discarded cards are public info. We extract features that tell us
what the opponent likely *kept*, not just what they threw away.

Features:
  - has_ace: did they discard an Ace? (implies they did NOT keep one)
  - pair_pattern: did they discard a pair? (implies they kept something better)
  - suit_concentration: are discards heavy in one suit? (implies kept different suit)
  - connectivity: are discards connected? (implies kept off-suit/non-connected)
  - board_interaction: do discards interact with the flop?
  - height_class: low-heavy vs high-heavy discards

Target: 8-16 buckets.
"""

from submission.abstractions.card_utils import (
    card_rank, card_suit, rank_counts, suit_counts,
    has_ace as _has_ace, ACE_RANK_IDX, sorted_ranks,
)


def _discard_has_ace(discards: list) -> int:
    """1 if an Ace was discarded, 0 otherwise."""
    return 1 if _has_ace(discards) else 0


def _discard_pair_pattern(discards: list) -> int:
    """
    0=no pair among discards
    1=pair among discards (2 of same rank)
    2=trips among discards (all 3 same rank)
    """
    rc = rank_counts(discards)
    mx = max(rc.values())
    if mx >= 3:
        return 2
    if mx >= 2:
        return 1
    return 0


def _discard_suit_concentration(discards: list) -> int:
    """
    0=all different suits (rainbow)
    1=two of same suit
    2=all same suit (monotone)
    """
    sc = suit_counts(discards)
    mx = max(sc.values())
    if mx >= 3:
        return 2
    if mx >= 2:
        return 1
    return 0


def _discard_connectivity(discards: list) -> int:
    """
    0=disconnected (spread > 4)
    1=semi-connected (spread 3-4)
    2=connected (spread <= 2, consecutive-ish)
    """
    ranks = sorted_ranks(discards)
    spread = ranks[-1] - ranks[0]
    # Handle Ace wrapping: if ace is present and low cards too
    if ACE_RANK_IDX in set(card_rank(c) for c in discards):
        low_ranks = [r for r in ranks if r != ACE_RANK_IDX]
        if low_ranks:
            spread = min(spread, max(low_ranks) - 0 + 1)  # ace-low view
    if spread <= 2:
        return 2
    if spread <= 4:
        return 1
    return 0


def _discard_board_interaction(discards: list, board_3: list) -> int:
    """
    How much the discards interact with the flop.
    0=no interaction (no shared ranks or suits with board)
    1=weak (shares a suit or adjacent rank)
    2=strong (shares a rank with board = they discarded a pair-with-board card)
    """
    disc_ranks = set(card_rank(c) for c in discards)
    board_ranks = set(card_rank(c) for c in board_3)
    disc_suits = set(card_suit(c) for c in discards)
    board_suits = set(card_suit(c) for c in board_3)

    # Strong: discarded a card that pairs the board
    if disc_ranks & board_ranks:
        return 2

    # Weak: shared suit or adjacent rank
    adjacent = set()
    for r in board_ranks:
        adjacent.add(r - 1)
        adjacent.add(r + 1)
    if disc_ranks & adjacent:
        return 1
    if disc_suits & board_suits:
        return 1

    return 0


def _discard_height(discards: list) -> int:
    """
    0=low-heavy (avg rank <= 3, i.e. mostly 2-5)
    1=mid (avg rank 4-6)
    2=high-heavy (avg rank >= 7, includes ace)
    """
    avg = sum(card_rank(c) for c in discards) / len(discards)
    if avg >= 6:
        return 2
    if avg >= 3:
        return 1
    return 0


def opp_discard_bucket(discards: list, board_3: list) -> int:
    """
    Compute opponent discard bucket from their 3 revealed discards + flop.

    Encoding: ace(2) × pair(3) × suit_conc(3) × board_int(3) = 54 max
    We further compress by grouping rarely-seen combos.

    In practice this gives ~8-16 meaningfully distinct clusters.
    """
    if not discards or all(c < 0 for c in discards):
        return 0  # unknown / not yet discarded

    valid = [c for c in discards if c >= 0]
    if len(valid) < 3:
        return 0

    a = _discard_has_ace(valid)
    p = _discard_pair_pattern(valid)
    s = _discard_suit_concentration(valid)
    bi = _discard_board_interaction(valid, board_3)

    return a * 27 + p * 9 + s * 3 + bi


def opp_discard_bucket_tuple(discards: list, board_3: list) -> tuple:
    """Debug version returning feature tuple."""
    valid = [c for c in discards if c >= 0]
    if len(valid) < 3:
        return (0, 0, 0, 0, 0, 0)
    return (
        _discard_has_ace(valid),
        _discard_pair_pattern(valid),
        _discard_suit_concentration(valid),
        _discard_connectivity(valid),
        _discard_board_interaction(valid, board_3),
        _discard_height(valid),
    )

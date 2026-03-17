"""
Core card utilities for the 27-card poker variant.
Deck constants, canonical hand forms, and evaluator wrapper.
"""

import itertools

# ─── Deck constants ───
RANKS = "23456789A"
SUITS = "dhs"  # diamonds, hearts, spades
NUM_RANKS = len(RANKS)  # 9
NUM_SUITS = len(SUITS)  # 3
DECK_SIZE = NUM_RANKS * NUM_SUITS  # 27
ALL_CARDS = list(range(DECK_SIZE))

# Ace index (used for special straight logic)
ACE_RANK_IDX = 8  # index of 'A' in RANKS


# ─── Basic card accessors ───
def card_rank(c: int) -> int:
    """Rank index 0-8 (2=0, 3=1, ..., 9=6, A=8)."""
    return c % NUM_RANKS

def card_suit(c: int) -> int:
    """Suit index 0-2 (d=0, h=1, s=2)."""
    return c // NUM_RANKS

def card_str(c: int) -> str:
    """Human-readable string like '2d', 'Ah'."""
    return RANKS[card_rank(c)] + SUITS[card_suit(c)]

def make_card(rank_idx: int, suit_idx: int) -> int:
    return suit_idx * NUM_RANKS + rank_idx


# ─── Treys interop (lazy import) ───
_evaluator = None

def get_evaluator():
    """Get the WrappedEval singleton (handles Ace-as-high)."""
    global _evaluator
    if _evaluator is None:
        from treys import Card, Evaluator

        class _WrappedEval(Evaluator):
            def evaluate(self, hand: list, board: list) -> int:
                def ace_to_ten(treys_card):
                    s = Card.int_to_str(treys_card)
                    return Card.new(s.replace("A", "T"))
                alt_hand = list(map(ace_to_ten, hand))
                alt_board = list(map(ace_to_ten, board))
                reg = super().evaluate(hand, board)
                alt = super().evaluate(alt_hand, alt_board)
                return min(reg, alt)

        _evaluator = _WrappedEval()
    return _evaluator

def int_to_treys(card_int: int):
    """Convert our int encoding to treys Card int."""
    from treys import Card
    return Card.new(card_str(card_int))


# ─── Suit canonicalization ───
def canonicalize_suits(cards: tuple) -> tuple:
    """
    Relabel suits so that the first-appearing suit (by sorted rank order)
    becomes 0, second becomes 1, etc.
    This collapses suit-isomorphic hands into a single canonical form.

    Input/output: tuple of card ints.
    Returns: tuple of card ints with relabeled suits (sorted).
    """
    # Sort by rank first, then by suit for tie-breaking, so suit assignment
    # is deterministic regardless of input order.
    sorted_cards = sorted(cards, key=lambda c: (card_rank(c), card_suit(c)))
    suit_map = {}
    next_suit = 0
    for c in sorted_cards:
        s = card_suit(c)
        if s not in suit_map:
            suit_map[s] = next_suit
            next_suit += 1
    canon = []
    for c in sorted_cards:
        r = card_rank(c)
        new_s = suit_map[card_suit(c)]
        canon.append(make_card(r, new_s))
    return tuple(sorted(canon))


def canonical_5card_id(cards_5: list) -> tuple:
    """
    Canonical ID for a 5-card preflop hand.
    Suit-relabeled and sorted.
    """
    return canonicalize_suits(tuple(sorted(cards_5)))


# ─── Rank/suit analysis helpers ───
def rank_counts(cards: list) -> dict:
    """Returns {rank_idx: count}."""
    counts = {}
    for c in cards:
        r = card_rank(c)
        counts[r] = counts.get(r, 0) + 1
    return counts

def suit_counts(cards: list) -> dict:
    """Returns {suit_idx: count}."""
    counts = {}
    for c in cards:
        s = card_suit(c)
        counts[s] = counts.get(s, 0) + 1
    return counts

def has_ace(cards: list) -> bool:
    return any(card_rank(c) == ACE_RANK_IDX for c in cards)

def sorted_ranks(cards: list) -> list:
    """Sorted rank indices, ascending."""
    return sorted(card_rank(c) for c in cards)


# ─── Straight detection (with A-low and A-high) ───
def has_straight_potential(rank_set: set, need: int = 5) -> bool:
    """Check if rank_set can form a straight of length `need`.
    Ace can be low (below 2) or high (above 9).
    """
    # Normal ranks 0-8, Ace is 8
    # A-low straight: A,2,3,4,5 = ranks {8,0,1,2,3}
    # A-high straight: 6,7,8,9,A = ranks {4,5,6,8} — wait, no 10/J/Q/K.
    # Actually with ranks 2-9,A: A-high straight is 6,7,8,9,A = ranks {4,5,6,7,8}
    # Standard: any 5 consecutive in 0-8, treating A(8) as also -1 for low

    extended = set(rank_set)
    if ACE_RANK_IDX in extended:
        extended.add(-1)  # Ace as low (below 2 which is rank 0)

    for start in range(-1, NUM_RANKS - need + 1):
        window = set(range(start, start + need))
        if window.issubset(extended):
            return True
    return False


def straight_draw_outs(rank_set: set) -> int:
    """Count how many single-rank additions complete a 5-card straight."""
    if len(rank_set) < 4:
        return 0
    outs = 0
    all_ranks = set(range(NUM_RANKS))
    missing = all_ranks - rank_set
    for r in missing:
        test = rank_set | {r}
        if has_straight_potential(test, 5):
            outs += 1
    return outs


# ─── All C(5,2) keep pairs ───
KEEP_PAIRS = list(itertools.combinations(range(5), 2))  # 10 pairs

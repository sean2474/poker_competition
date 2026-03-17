"""
5-card hand canonicalization for the 27-card deck.

Suit normalization: remap suits so the most frequent suit = 0,
next most frequent = 1, least = 2. Within same frequency, sort
by original suit index (stable). This ensures two hands that are
suit-permutations of each other share the same canonical key.

Example (3 suits: d=0, h=1, s=2):
  (Ad, Kd, Qd, Jh, Th) → suits d×3, h×1 → map d→0, h→1
  (As, Ks, Qs, Jh, Th) → suits s×3, h×1 → map s→0, h→1
  Both → same canonical form.
"""

NUM_RANKS = 9
NUM_SUITS = 3


def card_rank(c: int) -> int:
    return c % NUM_RANKS


def card_suit(c: int) -> int:
    return c // NUM_RANKS


def make_card(rank: int, suit: int) -> int:
    return rank + suit * NUM_RANKS


def canonicalize(hand5) -> tuple:
    """Return suit-normalized, sorted 5-card tuple."""
    # Count suit frequencies
    counts = [0, 0, 0]
    for c in hand5:
        counts[card_suit(c)] += 1

    # Map: highest freq → 0, next → 1, lowest → 2
    # Tie-break by original suit index (lower suit gets lower canonical)
    sorted_suits = sorted(range(3), key=lambda s: (-counts[s], s))
    suit_map = {s: i for i, s in enumerate(sorted_suits)}

    canonical = tuple(sorted(
        make_card(card_rank(c), suit_map[card_suit(c)])
        for c in hand5
    ))
    return canonical

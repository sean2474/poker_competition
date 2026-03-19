"""

27-card deck: 9 ranks (2-9, A) × 3 suits (d, h, s)
Card encoding: suit * NUM_RANKS + rank   (0-26)

Preflop game:
  SB (player 0) posts 1, BB (player 1) posts 2.
  SB acts first: fold / call / raise.
  Single raise size per level (2.5bb → re-raise → cap).
  Max 3 raises per preflop round.
"""

import random
from typing import List, Tuple

# ── Constants ──────────────────────────────────────────────────────────────────
NUM_RANKS   = 9      # 2,3,4,5,6,7,8,9,A
NUM_SUITS   = 3      # d,h,s
DECK_SIZE   = 27
BIG_BLIND   = 2
SMALL_BLIND = 1
MAX_BET     = 100    # per player

# Preflop raise sizes (total chips committed by raiser)
# Level 0: open to 5 (2.5bb), level 1: 3-bet to 15, level 2: cap 40
_RAISE_SIZES = [5, 15, 40, MAX_BET]  # indexed by raise_count


# ── Card utilities ─────────────────────────────────────────────────────────────

def rank(card: int) -> int:
    return card % NUM_RANKS

def suit(card: int) -> int:
    return card // NUM_RANKS

def make_deck() -> List[int]:
    return list(range(DECK_SIZE))

def deal(n_cards: int, exclude: set = None, rng=None) -> List[int]:
    deck = [c for c in range(DECK_SIZE) if exclude is None or c not in exclude]
    if rng:
        rng.shuffle(deck)
    else:
        random.shuffle(deck)
    return deck[:n_cards]


# ── Canonical hand (suit normalization) ───────────────────────────────────────

def canonicalize(hand5: List[int]) -> Tuple[int, ...]:
    """
    Suit-normalize a 5-card hand so equivalent hands (same ranks, permuted suits)
    map to the same key. Most-common suit → suit 0, ties broken by original suit id.
    """
    cnt = [0] * NUM_SUITS
    for c in hand5:
        cnt[suit(c)] += 1
    # Sort suits: descending frequency, then ascending original id (stable tie-break)
    suit_order = sorted(range(NUM_SUITS), key=lambda s: (-cnt[s], s))
    suit_map = {orig: new for new, orig in enumerate(suit_order)}
    return tuple(sorted(
        suit_map[suit(c)] * NUM_RANKS + rank(c) for c in hand5
    ))

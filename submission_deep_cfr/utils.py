"""
Card helpers, canonicalization, and discard oracle.
All logic mirrors deep_cfr_training/game_state.h exactly.
"""

NUM_RANKS = 9   # 2-9, A
NUM_SUITS = 3   # d, h, s

def card_rank(c: int) -> int: return c % NUM_RANKS
def card_suit(c: int) -> int: return c // NUM_RANKS
def make_card(rank: int, suit: int) -> int: return rank + suit * NUM_RANKS

def card_features(c: int) -> list:
    """4-float encoding: [rank/8, suit0, suit1, suit2]"""
    if c < 0: return [0., 0., 0., 0.]
    s = card_suit(c)
    return [card_rank(c) / 8., float(s == 0), float(s == 1), float(s == 2)]


# ── Suit normalization (for preflop chart key) ─────────────────────────────────

def canonicalize(hand5) -> tuple:
    """Suit-normalize a 5-card hand → canonical sorted tuple."""
    counts = [0, 0, 0]
    for c in hand5: counts[card_suit(c)] += 1
    sorted_suits = sorted(range(3), key=lambda s: (-counts[s], s))
    suit_map = {s: i for i, s in enumerate(sorted_suits)}
    return tuple(sorted(make_card(card_rank(c), suit_map[card_suit(c)]) for c in hand5))


# ── Discard oracle ─────────────────────────────────────────────────────────────

def choose_discard(hand5: list, board3: list) -> tuple:
    """
    Choose which 2 of 5 cards to keep after seeing the flop.
    Returns (keep_i, keep_j) indices into hand5.
    """
    br = [card_rank(c) for c in board3 if c >= 0]
    bs = [card_suit(c) for c in board3 if c >= 0]
    best_score, best = -1, (0, 1)

    for i in range(5):
        for j in range(i + 1, 5):
            r0, r1 = card_rank(hand5[i]), card_rank(hand5[j])
            s0, s1 = card_suit(hand5[i]), card_suit(hand5[j])
            sc = max(r0, r1)
            if r0 == r1:        sc += 20
            if s0 == s1:        sc += 5
            if r0 in br:        sc += 15
            if r1 in br:        sc += 15
            if abs(r0-r1) <= 2: sc += 3
            sct = [0, 0, 0]; sct[s0] += 1; sct[s1] += 1
            for _s in bs: sct[_s] += 1
            if max(sct) >= 4:   sc += 10
            if sc > best_score: best_score = sc; best = (i, j)
    return best

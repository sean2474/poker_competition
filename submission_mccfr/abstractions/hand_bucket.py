"""
Private hand bucketing per street (post-discard).

ALL DETERMINISTIC — no Monte Carlo. Pure card-structure analysis.

Axes per street:
  - made_tier: current hand strength from card structure
  - draw_tier: flush/straight draw potential from suit/rank counting
  - vulnerability: board wetness + hand fragility (deterministic)

Flop: ~24 buckets  |  Turn: ~24 buckets  |  River: ~12 buckets
"""

from abstractions.card_utils import (
    card_rank, card_suit, rank_counts, suit_counts,
    has_ace, has_straight_potential, straight_draw_outs,
    NUM_RANKS, ACE_RANK_IDX,
    get_evaluator, int_to_treys,
)


# ═══════════════════════════════════════════════════════════════════
# Made-hand tier (deterministic card structure)
# ═══════════════════════════════════════════════════════════════════

def _made_tier_from_structure(hand_2: list, board: list) -> int:
    """
    Deterministic made-hand classification from card structure.
    Works with 3, 4, or 5 board cards.

    0=air (high card, no connection)
    1=weak pair (bottom/middle pair, underpair)
    2=top pair / overpair
    3=two pair or better (set, trips, straight, flush, etc.)

    With 5 board cards, uses treys evaluator for exact classification.
    """
    hand_ranks = [card_rank(c) for c in hand_2]
    board_ranks = [card_rank(c) for c in board]

    if not board_ranks:
        return 0

    board_max = max(board_ranks)
    board_rank_set = set(board_ranks)
    pocket_pair = hand_ranks[0] == hand_ranks[1]

    # Count combined rank frequencies
    combined = list(hand_2) + list(board)
    rc = rank_counts(combined)
    max_freq = max(rc.values())
    num_pairs = sum(1 for v in rc.values() if v >= 2)

    # If we have 5 board cards, use exact evaluator
    if len(board) == 5:
        ev = get_evaluator()
        h = [int_to_treys(c) for c in hand_2]
        b = [int_to_treys(c) for c in board]
        rank = ev.evaluate(h, b)
        # treys: 1-10=SF, 167-322=FH, 323-1599=Flush, 1600-1609=Straight,
        # 1610-2467=Trips, 2468-3325=TwoPair, 3326-6185=OnePair, 6186-7462=HC
        if rank <= 1609:
            return 3  # straight or better
        if rank <= 2467:
            return 3  # trips
        if rank <= 3325:
            return 3  # two pair
        if rank <= 4500:
            return 2  # decent pair (top pair range)
        if rank <= 6185:
            return 1  # weak pair
        return 0  # high card

    # For incomplete boards (flop/turn): structural analysis
    # Set or trips (3 of same rank using at least 1 hand card)
    for hr in hand_ranks:
        if rc.get(hr, 0) >= 3:
            return 3

    # Two pair (at least 2 pairs, hand contributes to both or one)
    if num_pairs >= 2:
        hand_contributes = sum(1 for hr in hand_ranks if rc.get(hr, 0) >= 2)
        if hand_contributes >= 1:
            return 3

    # Overpair
    if pocket_pair and hand_ranks[0] > board_max:
        return 2

    # Top pair (hand card matches highest board rank)
    if any(hr == board_max for hr in hand_ranks):
        return 2

    # Middle/bottom pair or underpair
    paired_with_board = any(hr in board_rank_set for hr in hand_ranks)
    if paired_with_board or pocket_pair:
        return 1

    return 0


# ═══════════════════════════════════════════════════════════════════
# Draw tier (deterministic: suit counting + straight outs)
# ═══════════════════════════════════════════════════════════════════

def _draw_tier(hand_2: list, board: list) -> int:
    """
    0=no draw, 1=weak (backdoor or gutshot), 2=OESD, 3=flush draw,
    4=combo draw (flush + straight draw)
    """
    combined = list(hand_2) + list(board)
    sc = suit_counts(combined)
    max_suited = max(sc.values()) if sc else 0

    all_ranks = set(card_rank(c) for c in combined)

    has_flush_draw = max_suited >= 4  # 4 of same suit with cards to come
    has_backdoor_flush = (max_suited == 3 and len(board) == 3)

    s_outs = straight_draw_outs(all_ranks)
    has_oesd = s_outs >= 2
    has_gutshot = s_outs >= 1

    if has_flush_draw and has_oesd:
        return 4
    if has_flush_draw:
        return 3
    if has_oesd:
        return 2
    if has_gutshot or has_backdoor_flush:
        return 1
    return 0


# ═══════════════════════════════════════════════════════════════════
# Vulnerability (deterministic: board texture + hand fragility)
# ═══════════════════════════════════════════════════════════════════

def _vulnerability(hand_2: list, board: list) -> int:
    """
    Deterministic vulnerability based on board texture and hand type.
    0=stable (nutted hands, dry boards)
    1=vulnerable (wet board or marginal hand)

    Key factors:
    - Board is wet (flush draws, straight draws possible)
    - Our hand is a single pair (easily outdrawn)
    - Board has overcards possible
    """
    if len(board) >= 5:
        return 0  # river, no more cards to worry about

    hand_ranks = [card_rank(c) for c in hand_2]
    board_ranks = [card_rank(c) for c in board]
    combined = list(hand_2) + list(board)

    # Board wetness: how many draws are possible?
    sc = suit_counts(board)
    max_board_suit = max(sc.values()) if sc else 0
    board_rank_set = set(board_ranks)

    # Wet if 2+ of same suit on board (flush draw possible for someone)
    wet_flush = max_board_suit >= 2

    # Wet if board has connected ranks
    s_outs = straight_draw_outs(board_rank_set)
    wet_straight = s_outs >= 1

    # Hand fragility: single pair with overcards possible
    made = _made_tier_from_structure(hand_2, board)
    fragile = (made <= 1)  # weak pair or air

    if fragile and (wet_flush or wet_straight):
        return 1
    if made <= 2 and wet_flush and wet_straight:
        return 1
    return 0


# ═══════════════════════════════════════════════════════════════════
# Blocker analysis (deterministic)
# ═══════════════════════════════════════════════════════════════════

def _blocker_tier(hand_2: list, board: list) -> int:
    """
    0=no meaningful blockers
    1=has blockers (blocks flush/straight/set for opponent)
    """
    board_sc = suit_counts(board)
    if not board_sc:
        return 0
    max_board_suit = max(board_sc, key=board_sc.get)
    max_board_suit_count = board_sc[max_board_suit]

    # Block flush: we hold card(s) of the most common board suit
    hand_blocks_flush = False
    if max_board_suit_count >= 3:
        hand_blocks_flush = any(card_suit(c) == max_board_suit for c in hand_2)

    # Block set: we hold a card matching a board rank (blocks their trips)
    board_rank_set = set(card_rank(c) for c in board)
    hand_blocks_set = any(card_rank(c) in board_rank_set for c in hand_2)

    if hand_blocks_flush or hand_blocks_set:
        return 1
    return 0


# ═══════════════════════════════════════════════════════════════════
# Composite bucket functions per street
# ═══════════════════════════════════════════════════════════════════

def flop_hand_bucket(hand_2: list, board_3: list, dead: list = None) -> int:
    """
    Flop hand bucket (post-discard, 3 board cards).
    made(4) × draw_c(3) × vuln(2) = 24 buckets.
    """
    made = _made_tier_from_structure(hand_2, board_3)
    draw = _draw_tier(hand_2, board_3)
    vuln = _vulnerability(hand_2, board_3)

    draw_c = 0
    if draw >= 3:
        draw_c = 2
    elif draw >= 1:
        draw_c = 1

    return made * 6 + draw_c * 2 + vuln


def flop_hand_bucket_tuple(hand_2: list, board_3: list, dead: list = None) -> tuple:
    """Debug: returns (made, draw, vuln)."""
    made = _made_tier_from_structure(hand_2, board_3)
    draw = _draw_tier(hand_2, board_3)
    vuln = _vulnerability(hand_2, board_3)
    return (made, draw, vuln)


def turn_hand_bucket(hand_2: list, board_4: list, dead: list = None) -> int:
    """
    Turn hand bucket (4 board cards).
    made(4) × draw_c(3) × vuln(2) = 24 buckets.
    """
    made = _made_tier_from_structure(hand_2, board_4)
    draw = _draw_tier(hand_2, board_4)
    vuln = _vulnerability(hand_2, board_4)

    draw_c = 0
    if draw >= 3:
        draw_c = 2
    elif draw >= 1:
        draw_c = 1

    return made * 6 + draw_c * 2 + vuln


def _river_made_tier(hand_2: list, board_5: list) -> int:
    """
    Fine-grained river made tier (5 levels). Structure-based, matches C++.
    0=air/bluff candidate, 1=bluff catcher (weak pair),
    2=thin value (overpair/top pair), 3=clear value (two pair/trips),
    4=nutted (straight+)
    """
    ev = get_evaluator()
    h = [int_to_treys(c) for c in hand_2]
    b = [int_to_treys(c) for c in board_5]
    rank = ev.evaluate(h, b)
    # treys: 1-10=SF, 167-322=FH, 323-1599=Flush, 1600-1609=Straight,
    # 1610-2467=Trips, 2468-3325=TwoPair, 3326-6185=OnePair, 6186+=HC
    if rank <= 1609:
        return 4  # straight or better = nutted
    if rank <= 3325:
        return 3  # trips or two pair = clear value
    if rank <= 6185:
        # One pair: top pair / overpair = thin value, else bluff catcher
        hand_ranks = [card_rank(c) for c in hand_2]
        board_max = max(card_rank(c) for c in board_5)
        overpair = hand_ranks[0] == hand_ranks[1] and hand_ranks[0] > board_max
        top_pair = board_max in hand_ranks
        if overpair or top_pair:
            return 2
        return 1
    return 0  # high card = air


def river_hand_bucket(hand_2: list, board_5: list, dead: list = None) -> int:
    """
    River hand bucket (5 board cards, no more draws).
    made(5) × blocker(2) = 10 buckets.
    """
    made = _river_made_tier(hand_2, board_5)
    blocker = _blocker_tier(hand_2, board_5)
    return made * 2 + blocker


# ═══════════════════════════════════════════════════════════════════
# Unified interface
# ═══════════════════════════════════════════════════════════════════

def hand_bucket_for_street(hand_2: list, community: list, street: int,
                           dead: list = None) -> int:
    """
    Compute hand bucket for given street.
    street 0 (preflop): not used (preflop is canonical exact)
    street 1 (flop): flop_hand_bucket
    street 2 (turn): turn_hand_bucket
    street 3 (river): river_hand_bucket
    """
    if street <= 0:
        return 0
    elif street == 1:
        return flop_hand_bucket(hand_2, community[:3], dead)
    elif street == 2:
        return turn_hand_bucket(hand_2, community[:4], dead)
    else:
        return river_hand_bucket(hand_2, community[:5], dead)

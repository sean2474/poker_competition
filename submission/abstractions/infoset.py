"""
Info set key assembly.

Combines all abstraction layers into a single key per street:

Preflop:  (PF, position, line_bucket, canon_5card_id)
Flop:     (F, position, init, line, pressure, board_bkt, opp_disc_bkt, hand_bkt)
Turn:     (T, position, init, line, pressure, board_bkt, opp_disc_bkt, hand_bkt)
River:    (R, position, init, line, pressure, board_bkt, opp_disc_bkt, hand_bkt)

Keys are tuples (hashable, pickle-safe).
"""

from submission.abstractions.card_utils import canonical_5card_id
from submission.abstractions.board_texture import board_bucket_for_street
from submission.abstractions.hand_bucket import hand_bucket_for_street
from submission.abstractions.opp_discard_bucket import opp_discard_bucket
from submission.abstractions.public_state import (
    position_bucket, initiative_bucket_simple, line_bucket, pressure_bucket,
)
from submission.abstractions.action_abs import action_to_short


# ═══════════════════════════════════════════════════════════════════
# Preflop info set key
# ═══════════════════════════════════════════════════════════════════

def preflop_infoset(hand_5: list, is_bb: bool, action_history_str: str) -> tuple:
    """
    Preflop: near-exact hand (canonical 5-card) + action abstraction only.

    Args:
        hand_5: list of 5 card ints (pre-discard hole cards)
        is_bb: True if we are big blind
        action_history_str: string of short action codes for this street
    """
    canon = canonical_5card_id(hand_5)
    pos = position_bucket(is_bb)
    line = line_bucket(action_history_str)
    return ("PF", pos, line, canon)


# ═══════════════════════════════════════════════════════════════════
# Post-discard info set keys (flop/turn/river betting)
# ═══════════════════════════════════════════════════════════════════

def postdiscard_infoset(street: int, hand_2: list, community: list,
                         opp_discards: list, is_bb: bool,
                         hero_last_raiser: bool, villain_last_raiser: bool,
                         action_history_str: str,
                         my_bet: int, opp_bet: int,
                         dead: list = None) -> tuple:
    """
    Post-discard betting info set key.

    Args:
        street: 1 (flop), 2 (turn), 3 (river)
        hand_2: list of 2 card ints (post-discard hole cards)
        community: list of up to 5 community card ints
        opp_discards: list of 3 card ints (opponent's public discards)
        is_bb: True if we are big blind
        hero_last_raiser: True if hero was last aggressor
        villain_last_raiser: True if villain was last aggressor
        action_history_str: short action codes for current street
        my_bet, opp_bet: current bets
        dead: additional dead cards (our discards + opp discards)
    """
    street_tag = {1: "F", 2: "T", 3: "R"}.get(street, "?")

    pos = position_bucket(is_bb)
    init = initiative_bucket_simple(hero_last_raiser, villain_last_raiser)
    line = line_bucket(action_history_str)
    press = pressure_bucket(my_bet, opp_bet)

    board_bkt = board_bucket_for_street(community, street)

    # Opponent discard bucket (uses flop cards)
    board_3 = community[:3] if len(community) >= 3 else community
    opp_disc_bkt = opp_discard_bucket(opp_discards, board_3)

    # Private hand bucket
    hand_bkt = hand_bucket_for_street(hand_2, community, street, dead)

    return (street_tag, pos, init, line, press, board_bkt, opp_disc_bkt, hand_bkt)


# ═══════════════════════════════════════════════════════════════════
# Unified interface: build key from observation dict
# ═══════════════════════════════════════════════════════════════════

def build_infoset_key(observation: dict, hand_cards: list,
                       is_bb: bool, hero_last_raiser: bool,
                       villain_last_raiser: bool,
                       street_action_history: str,
                       my_discards: list = None,
                       opp_discards: list = None) -> tuple:
    """
    Build the appropriate info set key from an observation.

    Args:
        observation: game observation dict
        hand_cards: our current hole cards (5 pre-discard, 2 post-discard)
        is_bb: big blind flag
        hero_last_raiser, villain_last_raiser: initiative flags
        street_action_history: short codes for current street actions
        my_discards: our 3 discarded cards (post-discard)
        opp_discards: opponent's 3 discarded cards (public)

    Returns:
        tuple: info set key
    """
    street = observation["street"]
    community = [c for c in observation["community_cards"] if c != -1]

    if street == 0:
        # Preflop: use canonical 5-card
        return preflop_infoset(hand_cards, is_bb, street_action_history)
    else:
        # Post-discard betting
        my_bet = observation["my_bet"]
        opp_bet = observation["opp_bet"]

        dead = []
        if my_discards:
            dead.extend(c for c in my_discards if c >= 0)
        if opp_discards:
            dead.extend(c for c in opp_discards if c >= 0)

        opp_disc = opp_discards if opp_discards else [-1, -1, -1]

        return postdiscard_infoset(
            street, hand_cards, community, opp_disc,
            is_bb, hero_last_raiser, villain_last_raiser,
            street_action_history, my_bet, opp_bet, dead
        )


# ═══════════════════════════════════════════════════════════════════
# Estimate total info set space sizes
# ═══════════════════════════════════════════════════════════════════

def estimate_infoset_sizes():
    """Print estimated info set space sizes per street."""
    # Preflop: canonical 5-card hands × position(2) × line(6)
    # 27 choose 5 = 80730, but canonical reduces by ~6x (suit iso) = ~13000
    # × 2 × 6 = ~156,000 preflop info sets (exact, no bucketing)
    pf = 13000 * 2 * 6

    # Flop: position(2) × init(3) × line(6) × pressure(5) × board(~81) × opp_disc(~54) × hand(24)
    f = 2 * 3 * 6 * 5 * 81 * 54 * 24

    # Turn: same structure, board bigger, hand ~20
    t = 2 * 3 * 6 * 5 * (81*5) * 54 * 20

    # River: hand ~8
    r = 2 * 3 * 6 * 5 * (81*25) * 54 * 8

    print(f"Preflop info sets (exact canonical): ~{pf:,}")
    print(f"Flop info sets (max theoretical): ~{f:,}")
    print(f"Turn info sets (max theoretical): ~{t:,}")
    print(f"River info sets (max theoretical): ~{r:,}")
    print(f"NOTE: In practice CFR only visits a fraction of these")
    return pf, f, t, r

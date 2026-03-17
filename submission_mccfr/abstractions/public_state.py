"""
Public state bucketing: line, initiative, and pressure.

These capture the betting context that is public information
and strongly influences optimal strategy.

Components:
  - position: SB(0) / BB(1)
  - initiative_bucket: who was the last aggressor
  - line_bucket: betting line class (check-call, bet-raise, etc.)
  - pressure_bucket: relative cost to continue
"""

MAX_BET = 100


# ═══════════════════════════════════════════════════════════════════
# Position
# ═══════════════════════════════════════════════════════════════════

def position_bucket(is_big_blind: bool) -> int:
    """0=SB/IP, 1=BB/OOP."""
    return 1 if is_big_blind else 0


# ═══════════════════════════════════════════════════════════════════
# Initiative
# ═══════════════════════════════════════════════════════════════════

def initiative_bucket(action_history: list) -> int:
    """
    Classify who has initiative based on action history.
    action_history: list of (player_id, action_name) tuples for current street.

    0 = no clear aggressor (checked around, or start of street)
    1 = hero was last aggressor
    2 = villain was last aggressor
    """
    last_aggressor = None
    for player_id, action in action_history:
        if action in ("RAISE", "BET", "RAISE_SMALL", "RAISE_LARGE",
                       "BET_SMALL", "BET_LARGE", "RAISE_HALF", "RAISE_POT",
                       "RAISE_ALLIN", "R"):
            last_aggressor = player_id

    if last_aggressor is None:
        return 0
    return last_aggressor + 1  # 1=player0 aggressor, 2=player1 aggressor


def initiative_bucket_simple(hero_was_last_raiser: bool,
                              villain_was_last_raiser: bool) -> int:
    """
    Simplified version when we track just flags.
    0=no aggressor, 1=hero, 2=villain
    """
    if hero_was_last_raiser:
        return 1
    if villain_was_last_raiser:
        return 2
    return 0


# ═══════════════════════════════════════════════════════════════════
# Line bucket
# ═══════════════════════════════════════════════════════════════════

def line_bucket(action_seq: str) -> int:
    """
    Classify the current street's action sequence into a line bucket.

    action_seq: string of action codes for current street.
      'C'=check/call, 'R'=raise/bet, 'F'=fold (shouldn't reach here)

    0 = unopened / checked to (no bets yet: '', 'C')
    1 = single bet (one raise: 'R', 'CR')
    2 = bet-call ('RC')
    3 = check-raise ('CR' then action, or 'CRC')
    4 = re-raised / multi-raise ('RR', 'CRR', 'RRC', etc.)
    5 = heavy action (3+ raises)
    """
    if not action_seq:
        return 0

    num_raises = action_seq.count('R')

    if num_raises == 0:
        return 0  # checked around
    if num_raises == 1:
        # Was it a donk bet or check-raise?
        idx = action_seq.index('R')
        if idx > 0 and action_seq[idx - 1] == 'C':
            return 3  # check-raise pattern
        if 'C' in action_seq[idx + 1:]:
            return 2  # bet-call
        return 1  # single bet, action pending
    if num_raises == 2:
        return 4  # re-raise
    return 5  # heavy action


# ═══════════════════════════════════════════════════════════════════
# Pressure bucket
# ═══════════════════════════════════════════════════════════════════

def pressure_bucket(my_bet: int, opp_bet: int) -> int:
    """
    How much pressure hero faces relative to the pot.

    0 = free (no bet to face, or we've already matched)
    1 = small pressure (to_call < 25% of pot)
    2 = medium pressure (to_call 25-50% of pot)
    3 = large pressure (to_call 50-75% of pot)
    4 = near-commitment (to_call > 75% of pot, or near max bet)
    """
    to_call = opp_bet - my_bet
    if to_call <= 0:
        return 0

    pot = my_bet + opp_bet
    if pot <= 0:
        return 1

    ratio = to_call / pot
    if ratio < 0.15:
        return 1
    if ratio < 0.30:
        return 2
    if ratio < 0.50:
        return 3
    return 4


# ═══════════════════════════════════════════════════════════════════
# Combined public state tuple
# ═══════════════════════════════════════════════════════════════════

def public_state_key(position: int, init_bucket: int,
                     line_bkt: int, pressure_bkt: int) -> tuple:
    """
    Combine all public state components into a tuple for info set key.
    """
    return (position, init_bucket, line_bkt, pressure_bkt)


def public_state_int(position: int, init_bucket: int,
                     line_bkt: int, pressure_bkt: int) -> int:
    """
    Pack public state into a single int.
    position(2) × initiative(3) × line(6) × pressure(5) = 180
    """
    return position * 90 + init_bucket * 30 + line_bkt * 5 + pressure_bkt

"""
Action abstraction for CFR.

Two contexts:
  - facing_no_bet: CHECK, BET_SMALL, BET_LARGE
  - facing_bet: FOLD, CALL, RAISE_SMALL, RAISE_LARGE
  - high_pressure: FOLD, CALL, JAM

Concrete mapping uses observation's min_raise/max_raise to pick representative values.
"""

# ═══════════════════════════════════════════════════════════════════
# Abstract action definitions
# ═══════════════════════════════════════════════════════════════════

# Context: no bet to face (can check)
NO_BET_ACTIONS = ["CHECK", "BET_SMALL", "BET_LARGE"]

# Context: facing a bet (must fold/call/raise)
FACING_BET_ACTIONS = ["FOLD", "CALL", "RAISE_SMALL", "RAISE_LARGE"]

# Context: high pressure or near all-in (simplified)
HIGH_PRESSURE_ACTIONS = ["FOLD", "CALL", "JAM"]

# ActionType enum values from PokerEnv
_FOLD = 0
_RAISE = 1
_CHECK = 2
_CALL = 3


def get_action_context(valid_actions: list, my_bet: int, opp_bet: int,
                       max_raise: int) -> str:
    """
    Determine which action context we're in.
    Returns 'no_bet', 'facing_bet', or 'high_pressure'.
    """
    to_call = opp_bet - my_bet
    can_raise = valid_actions[_RAISE] if len(valid_actions) > _RAISE else False

    if to_call <= 0:
        return "no_bet"

    if not can_raise or max_raise <= 0:
        return "high_pressure"

    # High pressure if to_call is > 60% of remaining stack capacity
    remaining_capacity = 100 - max(my_bet, opp_bet)
    if remaining_capacity > 0 and to_call / remaining_capacity > 0.6:
        return "high_pressure"

    return "facing_bet"


def get_abstract_actions(context: str) -> list:
    """Return list of abstract action names for a context."""
    if context == "no_bet":
        return NO_BET_ACTIONS
    elif context == "facing_bet":
        return FACING_BET_ACTIONS
    else:
        return HIGH_PRESSURE_ACTIONS


def get_valid_abstract_actions(valid_actions: list, my_bet: int, opp_bet: int,
                                min_raise: int, max_raise: int) -> list:
    """
    Return list of valid abstract actions given game state.
    Filters out actions that aren't actually legal.
    """
    context = get_action_context(valid_actions, my_bet, opp_bet, max_raise)
    abstract = get_abstract_actions(context)

    result = []
    for a in abstract:
        if a == "FOLD" and valid_actions[_FOLD]:
            result.append(a)
        elif a == "CHECK" and valid_actions[_CHECK]:
            result.append(a)
        elif a == "CALL" and valid_actions[_CALL]:
            result.append(a)
        elif a in ("BET_SMALL", "BET_LARGE", "RAISE_SMALL", "RAISE_LARGE", "JAM"):
            if valid_actions[_RAISE]:
                result.append(a)

    # Always have at least FOLD available
    if not result:
        result.append("FOLD")

    return result


def abstract_to_concrete(action: str, min_raise: int, max_raise: int,
                          my_bet: int, opp_bet: int) -> tuple:
    """
    Convert abstract action to concrete (action_type, raise_amount, keep1, keep2).

    Size mapping:
      BET_SMALL / RAISE_SMALL -> lower quartile of [min_raise, max_raise]
      BET_LARGE / RAISE_LARGE -> upper-mid (~65-75%) of [min_raise, max_raise]
      JAM -> max_raise
    """
    if action == "FOLD":
        return (_FOLD, 0, 0, 0)

    if action == "CHECK":
        return (_CHECK, 0, 0, 0)

    if action == "CALL":
        return (_CALL, 0, 0, 0)

    # All raise-type actions
    if max_raise <= 0 or max_raise < min_raise:
        # Can't raise, fall back to call or check
        return (_CALL, 0, 0, 0)

    spread = max_raise - min_raise

    if action == "BET_SMALL" or action == "RAISE_SMALL":
        # Lower quartile: ~25% of spread
        amount = min_raise + int(spread * 0.25)
    elif action == "BET_LARGE" or action == "RAISE_LARGE":
        # Upper-mid: ~70% of spread
        amount = min_raise + int(spread * 0.70)
    elif action == "JAM":
        amount = max_raise
    else:
        amount = min_raise

    amount = max(min_raise, min(amount, max_raise))
    return (_RAISE, amount, 0, 0)


def concrete_to_abstract(action_type: int, raise_amount: int,
                          min_raise: int, max_raise: int,
                          my_bet: int, opp_bet: int) -> str:
    """
    Map a concrete action back to the closest abstract action.
    Used for building history strings from observed opponent actions.
    """
    if action_type == _FOLD:
        return "FOLD"
    if action_type == _CHECK:
        return "CHECK"
    if action_type == _CALL:
        return "CALL"

    # RAISE: classify by size
    if max_raise <= 0 or max_raise <= min_raise:
        return "JAM"

    spread = max_raise - min_raise
    if spread <= 0:
        return "JAM"

    # Where does raise_amount fall in [min_raise, max_raise]?
    position = (raise_amount - min_raise) / spread

    to_call = opp_bet - my_bet
    if to_call > 0:
        # facing_bet context
        if position >= 0.85 or raise_amount >= max_raise * 0.9:
            return "JAM"
        elif position >= 0.45:
            return "RAISE_LARGE"
        else:
            return "RAISE_SMALL"
    else:
        # no_bet context
        if position >= 0.85 or raise_amount >= max_raise * 0.9:
            return "JAM"
        elif position >= 0.45:
            return "BET_LARGE"
        else:
            return "BET_SMALL"


# ═══════════════════════════════════════════════════════════════════
# History string helpers
# ═══════════════════════════════════════════════════════════════════

# Short codes for history string (1-2 chars each for compact keys)
ACTION_SHORT = {
    "FOLD": "F",
    "CHECK": "K",
    "CALL": "C",
    "BET_SMALL": "b",
    "BET_LARGE": "B",
    "RAISE_SMALL": "r",
    "RAISE_LARGE": "R",
    "JAM": "J",
}

def action_to_short(action: str) -> str:
    """Convert abstract action name to short code for history string."""
    return ACTION_SHORT.get(action, "?")

def short_to_action(code: str) -> str:
    """Reverse lookup."""
    rev = {v: k for k, v in ACTION_SHORT.items()}
    return rev.get(code, "FOLD")

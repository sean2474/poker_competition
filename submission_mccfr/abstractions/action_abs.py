"""
Action abstraction for CFR.

Two contexts:
  - no_bet: CHECK, BET_SMALL, BET_LARGE
  - facing_bet: FOLD, CALL, RAISE_SMALL, RAISE_LARGE
"""

NO_BET_ACTIONS = ["CHECK", "BET_SMALL", "BET_LARGE"]
FACING_BET_ACTIONS = ["FOLD", "CALL", "RAISE_SMALL", "RAISE_LARGE"]

_FOLD = 0
_RAISE = 1
_CHECK = 2
_CALL = 3


def get_action_context(valid_actions: list, my_bet: int, opp_bet: int,
                       max_raise: int) -> str:
    """Returns 'no_bet' or 'facing_bet'."""
    to_call = opp_bet - my_bet
    if to_call <= 0:
        return "no_bet"
    return "facing_bet"


def get_abstract_actions(context: str) -> list:
    if context == "no_bet":
        return NO_BET_ACTIONS
    return FACING_BET_ACTIONS


def get_valid_abstract_actions(valid_actions: list, my_bet: int, opp_bet: int,
                                min_raise: int, max_raise: int) -> list:
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
        elif a in ("BET_SMALL", "BET_LARGE", "RAISE_SMALL", "RAISE_LARGE"):
            if valid_actions[_RAISE]:
                result.append(a)

    if not result:
        result.append("FOLD")
    return result


def abstract_to_concrete(action: str, min_raise: int, max_raise: int,
                          my_bet: int, opp_bet: int) -> tuple:
    """
    Size mapping (matches C++ cfr_engine.h training sizing):
      BET_SMALL / RAISE_SMALL -> min_raise + spread * 0.25
      BET_LARGE / RAISE_LARGE -> min_raise + spread * 0.70
    """
    if action == "FOLD":
        return (_FOLD, 0, 0, 0)
    if action == "CHECK":
        return (_CHECK, 0, 0, 0)
    if action == "CALL":
        return (_CALL, 0, 0, 0)

    if max_raise <= 0 or max_raise < min_raise:
        return (_CALL, 0, 0, 0)

    spread = max_raise - min_raise

    if action in ("BET_SMALL", "RAISE_SMALL"):
        amount = min_raise + int(spread * 0.25)
    elif action in ("BET_LARGE", "RAISE_LARGE"):
        amount = min_raise + int(spread * 0.70)
    else:
        amount = min_raise

    amount = max(min_raise, min(amount, max_raise))
    return (_RAISE, amount, 0, 0)


def concrete_to_abstract(action_type: int, raise_amount: int,
                          min_raise: int, max_raise: int,
                          my_bet: int, opp_bet: int) -> str:
    if action_type == _FOLD:
        return "FOLD"
    if action_type == _CHECK:
        return "CHECK"
    if action_type == _CALL:
        return "CALL"

    if max_raise <= 0 or max_raise <= min_raise:
        return "RAISE_LARGE"
    spread = max_raise - min_raise
    if spread <= 0:
        return "RAISE_LARGE"

    position = (raise_amount - min_raise) / spread
    to_call = opp_bet - my_bet

    if to_call > 0:
        return "RAISE_LARGE" if position >= 0.45 else "RAISE_SMALL"
    else:
        return "BET_LARGE" if position >= 0.45 else "BET_SMALL"


ACTION_SHORT = {
    "FOLD": "F",
    "CHECK": "K",
    "CALL": "C",
    "BET_SMALL": "b",
    "BET_LARGE": "B",
    "RAISE_SMALL": "r",
    "RAISE_LARGE": "R",
}

def action_to_short(action: str) -> str:
    return ACTION_SHORT.get(action, "?")

def short_to_action(code: str) -> str:
    rev = {v: k for k, v in ACTION_SHORT.items()}
    return rev.get(code, "FOLD")

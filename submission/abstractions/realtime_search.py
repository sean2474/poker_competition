"""
1-depth lookahead for important decisions.

Given current state + opponent range estimate, evaluates each possible
action by simulating opponent's likely response and computing EV.

Used when:
  - CFR misses (no blueprint for this infoset)
  - Pot is significant (worth spending time on)
"""

import random
from abstractions.card_utils import get_evaluator, int_to_treys, ALL_CARDS, card_rank, ACE_RANK_IDX

_FOLD = 0
_RAISE = 1
_CHECK = 2
_CALL = 3


def _estimate_opp_response(equity_vs_us: float, to_call: int, pot: int):
    """
    Estimate how a reasonable opponent responds to our action.
    Returns (fold_prob, call_prob, raise_prob).
    Based on opponent's equity against our likely range.
    """
    # Opponent's equity is roughly 1 - our_equity
    opp_equity = 1.0 - equity_vs_us

    opp_pot_odds = to_call / (to_call + pot) if to_call > 0 and pot > 0 else 0

    if to_call <= 0:
        # Opponent checks back — no action needed from them
        return (0.0, 1.0, 0.0)

    if opp_equity < opp_pot_odds * 0.6:
        return (0.85, 0.15, 0.0)  # mostly fold
    elif opp_equity < opp_pot_odds:
        return (0.5, 0.45, 0.05)  # mixed fold/call
    elif opp_equity < 0.6:
        return (0.1, 0.75, 0.15)  # mostly call
    elif opp_equity < 0.8:
        return (0.05, 0.5, 0.45)  # call or raise
    else:
        return (0.0, 0.3, 0.7)  # mostly raise


def lookahead_ev(hand: list, community: list, my_bet: int, opp_bet: int,
                  min_raise: int, max_raise: int, valid_actions: list,
                  opp_combos: list, opp_weights: list,
                  num_sims: int = 80) -> dict:
    """
    1-depth lookahead: evaluate EV of each possible action.

    For each action:
      1. We take the action (changes pot/bets)
      2. Estimate opponent response (fold/call/raise)
      3. If showdown, compute equity-weighted outcome

    Returns: {action_name: expected_chips_won}
    """
    ev = get_evaluator()
    my_h = [int_to_treys(c) for c in hand]
    pot = my_bet + opp_bet
    to_call = opp_bet - my_bet
    board_need = 5 - len(community)

    # Pre-sample opponent hands for speed
    if opp_weights and len(opp_weights) == len(opp_combos):
        sampled_opps = [
            opp_combos[random.choices(range(len(opp_combos)), weights=opp_weights, k=1)[0]]
            for _ in range(num_sims)
        ]
    else:
        remaining = [c for c in ALL_CARDS if c not in set(hand) | set(community)]
        sampled_opps = [tuple(random.sample(remaining, 2)) for _ in range(num_sims)]

    results = {}

    # Define candidate actions
    candidates = []
    if valid_actions[_CHECK]:
        candidates.append(("CHECK", _CHECK, 0))
    if valid_actions[_FOLD]:
        candidates.append(("FOLD", _FOLD, 0))
    if valid_actions[_CALL]:
        candidates.append(("CALL", _CALL, 0))
    if valid_actions[_RAISE] and max_raise > 0:
        # Small bet: ~33% pot
        small_amt = max(min_raise, min(int(pot * 0.33), max_raise))
        candidates.append(("BET_SMALL", _RAISE, small_amt))
        # Large bet: ~75% pot
        large_amt = max(min_raise, min(int(pot * 0.75), max_raise))
        if large_amt > small_amt:
            candidates.append(("BET_LARGE", _RAISE, large_amt))

    for name, action_type, raise_amt in candidates:
        total_ev = 0.0

        for opp_pair in sampled_opps:
            opp_pair = list(opp_pair)

            # Complete the board if needed
            used = set(hand) | set(community) | set(opp_pair)
            board_remaining = [c for c in ALL_CARDS if c not in used]
            if board_need > 0 and len(board_remaining) >= board_need:
                extra = random.sample(board_remaining, board_need)
                full_board = community + extra
            else:
                full_board = community

            if len(full_board) < 5:
                continue

            # Evaluate hands
            b = [int_to_treys(c) for c in full_board]
            opp_h = [int_to_treys(c) for c in opp_pair]
            my_rank = ev.evaluate(my_h, b)
            opp_rank = ev.evaluate(opp_h, b)

            # Our equity in this specific matchup
            if my_rank < opp_rank:
                showdown_result = 1.0  # we win
            elif my_rank == opp_rank:
                showdown_result = 0.0  # tie
            else:
                showdown_result = -1.0  # we lose

            # Simulate action outcome
            if action_type == _FOLD:
                total_ev += -my_bet  # lose what we've put in
            elif action_type == _CHECK:
                # Check: pot stays same, goes to showdown or next street
                win_amount = min(my_bet, opp_bet)
                total_ev += showdown_result * win_amount
            elif action_type == _CALL:
                # Call: match opponent's bet, showdown
                new_my_bet = opp_bet
                win_amount = min(new_my_bet, opp_bet)
                total_ev += showdown_result * win_amount
            elif action_type == _RAISE:
                # Raise: we bet raise_amt more
                new_my_bet = opp_bet + raise_amt
                new_pot = new_my_bet + opp_bet
                opp_to_call = new_my_bet - opp_bet

                # Estimate opponent response
                equity_for_opp = 1.0 - (1.0 + showdown_result) / 2.0
                fold_p, call_p, raise_p = _estimate_opp_response(
                    (1.0 + showdown_result) / 2.0, opp_to_call, new_pot
                )

                # Fold: we win current pot
                ev_fold = min(my_bet, opp_bet) + opp_to_call  # wrong, simpler:
                ev_fold = opp_bet  # we win what opp already put in

                # Call: showdown at bigger pot
                ev_call = showdown_result * min(new_my_bet, new_my_bet)

                # Raise: assume we call their raise, showdown
                ev_raise = showdown_result * min(new_my_bet + raise_amt, 100)

                total_ev += fold_p * ev_fold + call_p * ev_call + raise_p * ev_raise

        results[name] = total_ev / max(len(sampled_opps), 1)

    return results


def choose_best_action(hand, community, my_bet, opp_bet, min_raise, max_raise,
                        valid_actions, opp_combos, opp_weights, num_sims=80):
    """
    Run lookahead and return the best action as (action_type, raise_amount).
    """
    evs = lookahead_ev(hand, community, my_bet, opp_bet, min_raise, max_raise,
                        valid_actions, opp_combos, opp_weights, num_sims)

    if not evs:
        return None, None

    best_name = max(evs, key=evs.get)

    # Map back to concrete action
    pot = my_bet + opp_bet
    if best_name == "FOLD":
        return (_FOLD, 0)
    elif best_name == "CHECK":
        return (_CHECK, 0)
    elif best_name == "CALL":
        return (_CALL, 0)
    elif best_name == "BET_SMALL":
        amt = max(min_raise, min(int(pot * 0.33), max_raise))
        return (_RAISE, amt)
    elif best_name == "BET_LARGE":
        amt = max(min_raise, min(int(pot * 0.75), max_raise))
        return (_RAISE, amt)

    return None, None

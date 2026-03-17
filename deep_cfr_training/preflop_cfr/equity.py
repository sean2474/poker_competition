"""
Quick equity estimation for preflop warmup.

equity(p0_hand5, p1_hand5) = P(p0 wins showdown)
  - Both players discard optimally using C++ fast_discard
  - Sample n random 5-card community boards
  - Compare 2-card hands with evaluate_showdown

Used during warmup iterations (iteration <= warmup_iters) as a fast
substitute for the untrained postflop network.
"""

import random

_FULL_DECK = list(range(27))


def preflop_equity(p0_hand5: list, p1_hand5: list, n_boards: int = 20) -> float:
    """
    Monte Carlo equity for two 5-card preflop hands.
    Returns P(p0 wins) in [0, 1].
    """
    from game.features import fast_discard, evaluate_showdown

    dead = set(p0_hand5) | set(p1_hand5)
    remaining = [c for c in _FULL_DECK if c not in dead]

    wins = 0.0
    total = 0

    for _ in range(n_boards):
        if len(remaining) < 5:
            break
        community = random.sample(remaining, 5)
        flop = community[:3]

        ki0, kj0 = fast_discard(p0_hand5, flop)
        ki1, kj1 = fast_discard(p1_hand5, flop)

        p0_kept = [p0_hand5[ki0], p0_hand5[kj0]]
        p1_kept = [p1_hand5[ki1], p1_hand5[kj1]]

        result = evaluate_showdown(p0_kept, p1_kept, community)
        if result > 0:
            wins += 1.0
        elif result == 0:
            wins += 0.5
        total += 1

    return wins / total if total > 0 else 0.5


def warmup_ev(p0_hand5: list, p1_hand5: list, state,
              traversing_player: int, n_boards: int = 15) -> float:
    """
    Terminal EV approximation for warmup phase (before postflop net is trained).
    Uses equity × committed chips.
    """
    eq = preflop_equity(p0_hand5, p1_hand5, n_boards)
    committed = float(state.bets[traversing_player])
    if traversing_player == 0:
        return (2 * eq - 1) * committed
    else:
        return (1 - 2 * eq) * committed

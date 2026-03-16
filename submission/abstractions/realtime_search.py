"""
Real-time subgame solving via mini-CFR.

Key improvement over v1: uses pre-computed MC equity as the value function
for terminal nodes instead of single-sample showdown. This eliminates
per-iteration variance and models future streets properly.

Following Libratus depth-limited solving approach:
  - Pre-compute equity for each (hero_hand, opp_hand) pair via MC rollout
  - Use equity as terminal value (represents expected value of remaining game)
  - Both players' strategies are optimized via CFR+
  - No per-iteration board sampling needed (equity is the expectation)
"""

import random
import numpy as np
from abstractions.card_utils import get_evaluator, int_to_treys, ALL_CARDS

_FOLD = 0
_RAISE = 1
_CHECK = 2
_CALL = 3

BET_SMALL_FRAC = 0.33
BET_LARGE_FRAC = 0.75


def _compute_equities(my_hand, community, opp_hands, num_sims=100):
    """
    Pre-compute MC equity for hero vs each opponent hand.
    Returns array of equities in [0, 1] where 1 = hero always wins.
    This is the "value function" that replaces showdown at terminal nodes.
    """
    ev = get_evaluator()
    my_h = [int_to_treys(c) for c in my_hand]
    board_need = 5 - len(community)
    dead_set = set(my_hand) | set(community)

    equities = np.zeros(len(opp_hands))
    for oi, opp in enumerate(opp_hands):
        opp_h = [int_to_treys(c) for c in opp]
        used = dead_set | set(opp)
        rem = [c for c in ALL_CARDS if c not in used]

        if board_need == 0:
            b = [int_to_treys(c) for c in community[:5]]
            mr = ev.evaluate(my_h, b)
            opr = ev.evaluate(opp_h, b)
            equities[oi] = 1.0 if mr < opr else (0.5 if mr == opr else 0.0)
        elif len(rem) >= board_need:
            wins = ties = 0
            sims = min(num_sims, len(rem) * (len(rem)-1) // 2) if board_need <= 2 else num_sims
            for _ in range(sims):
                extra = random.sample(rem, board_need)
                b = [int_to_treys(c) for c in community + extra]
                mr = ev.evaluate(my_h, b)
                opr = ev.evaluate(opp_h, b)
                if mr < opr: wins += 1
                elif mr == opr: ties += 1
            equities[oi] = (wins + 0.5 * ties) / sims if sims > 0 else 0.5
        else:
            equities[oi] = 0.5
    return equities


def solve_subgame(my_hand: list, community: list,
                   my_bet: int, opp_bet: int,
                   min_raise: int, max_raise: int,
                   valid_actions: list,
                   opp_hands: list, opp_weights: list = None,
                   num_iters: int = 500) -> tuple:
    """
    Depth-limited subgame solving with equity-based value function.

    Terminal payoffs use pre-computed MC equity instead of single showdown.
    equity = 0.7 means hero wins 70% of the time at showdown.
    EV of showdown at stake S = (equity - 0.5) * 2 * S = (2*eq - 1) * S
    """
    pot = my_bet + opp_bet
    can_raise = valid_actions[_RAISE] and max_raise > 0

    bet_s = max(min_raise, min(int(pot * BET_SMALL_FRAC), max_raise)) if can_raise else 0
    bet_l = max(min_raise, min(int(pot * BET_LARGE_FRAC), max_raise)) if can_raise else 0

    facing_bet = (opp_bet > my_bet)
    if facing_bet:
        hero_actions = ["FOLD", "CALL"]
        if can_raise and bet_s > 0:
            hero_actions.append("RAISE_S")
        if can_raise and bet_l > bet_s:
            hero_actions.append("RAISE_L")
    else:
        hero_actions = ["CHECK"]
        if can_raise and bet_s > 0:
            hero_actions.append("BET_S")
        if can_raise and bet_l > bet_s:
            hero_actions.append("BET_L")

    n_hero = len(hero_actions)
    n_opp = len(opp_hands)
    if n_hero <= 1 or n_opp == 0:
        return None, None

    # Normalize opp weights
    if opp_weights and len(opp_weights) == n_opp:
        w_total = sum(opp_weights)
        opp_w = np.array([w / w_total for w in opp_weights]) if w_total > 0 else np.ones(n_opp) / n_opp
    else:
        opp_w = np.ones(n_opp) / n_opp

    # Pre-compute equities (the value function)
    eq = _compute_equities(my_hand, community, opp_hands, num_sims=80)
    # Convert to signed EV multiplier: eq=0.7 → ev_mult=0.4, eq=0.3 → ev_mult=-0.4
    ev_mult = 2.0 * eq - 1.0  # range [-1, 1]

    # Regret tables
    hero_regret = np.zeros(n_hero)
    hero_strat_sum = np.zeros(n_hero)

    opp_regret_after_check = np.zeros((n_opp, 3))  # CHECK/BET_S/BET_L
    opp_regret_after_bets = np.zeros((n_opp, 3))   # FOLD/CALL/RAISE
    opp_regret_after_betl = np.zeros((n_opp, 2))   # FOLD/CALL
    hero_regret_facing_bet = np.zeros(2)            # FOLD/CALL

    # Initialize opp regrets with equity-based priors
    # This avoids the cold-start problem where opp starts with uniform (too much raise)
    for oi in range(n_opp):
        opp_eq = 1.0 - eq[oi]  # opponent's equity
        # After hero bets: opp should fold weak, call medium, raise strong
        opp_regret_after_bets[oi] = np.array([
            max(0.5 - opp_eq, 0) * 10,   # FOLD: high regret if weak
            opp_eq * 5,                    # CALL: proportional to equity
            max(opp_eq - 0.6, 0) * 10     # RAISE: only if strong
        ])
        opp_regret_after_betl[oi] = np.array([
            max(0.5 - opp_eq, 0) * 10,   # FOLD
            opp_eq * 5                     # CALL
        ])
        # After hero checks: opp should check weak, bet strong
        opp_regret_after_check[oi] = np.array([
            max(0.5 - opp_eq, 0) * 5,    # CHECK
            opp_eq * 3,                    # BET_S
            max(opp_eq - 0.7, 0) * 5      # BET_L
        ])

    stake = min(my_bet, opp_bet)

    for _t in range(num_iters):
        # Hero root strategy
        pos = np.maximum(hero_regret, 0)
        tot = pos.sum()
        h_strat = pos / tot if tot > 0 else np.ones(n_hero) / n_hero
        hero_strat_sum += h_strat * max(_t, 1)  # linear weighting

        # Hero facing-bet strategy
        pos_fb = np.maximum(hero_regret_facing_bet, 0)
        tot_fb = pos_fb.sum()
        h_fb = pos_fb / tot_fb if tot_fb > 0 else np.array([0.5, 0.5])

        action_evs = np.zeros(n_hero)

        for oi in range(n_opp):
            w = opp_w[oi]
            sd = ev_mult[oi]  # signed equity multiplier

            if facing_bet:
                evs = []
                evs.append(-my_bet)       # FOLD: lose what we put in
                evs.append(sd * opp_bet)  # CALL: showdown at opp_bet level

                if "RAISE_S" in hero_actions:
                    op = np.maximum(opp_regret_after_bets[oi], 0)
                    ot = op.sum()
                    o_s = op / ot if ot > 0 else np.ones(3) / 3
                    new_bet = opp_bet + bet_s
                    ev_f = opp_bet                              # opp folds → we win their bet
                    ev_c = sd * new_bet                         # opp calls → showdown
                    ev_r = sd * min(new_bet + bet_s, 100)       # opp re-raises → approx
                    evs.append(o_s[0]*ev_f + o_s[1]*ev_c + o_s[2]*ev_r)

                    opp_evs = np.array([-opp_bet, -sd*new_bet, -sd*min(new_bet+bet_s, 100)])
                    opp_avg = np.dot(o_s, opp_evs)
                    opp_regret_after_bets[oi] = np.maximum(
                        opp_regret_after_bets[oi] + w*(opp_evs - opp_avg), 0)

                if "RAISE_L" in hero_actions:
                    new_bet = opp_bet + bet_l
                    op2 = np.maximum(opp_regret_after_betl[oi], 0)
                    ot2 = op2.sum()
                    o_s2 = op2 / ot2 if ot2 > 0 else np.array([0.5, 0.5])
                    ev_f = opp_bet
                    ev_c = sd * new_bet
                    evs.append(o_s2[0]*ev_f + o_s2[1]*ev_c)

                    opp_evs2 = np.array([-opp_bet, -sd*new_bet])
                    opp_avg2 = np.dot(o_s2, opp_evs2)
                    opp_regret_after_betl[oi] = np.maximum(
                        opp_regret_after_betl[oi] + w*(opp_evs2 - opp_avg2), 0)

                for ai in range(len(evs)):
                    action_evs[ai] += w * evs[ai]

            else:
                # Hero first: CHECK / BET_S / BET_L

                # ─ CHECK ─
                op_c = np.maximum(opp_regret_after_check[oi], 0)
                ot_c = op_c.sum()
                o_sc = op_c / ot_c if ot_c > 0 else np.ones(3) / 3

                ev_cc = sd * stake                                                      # both check
                ev_obs = h_fb[0]*(-my_bet) + h_fb[1]*(sd*(stake + bet_s))              # opp bets small
                ev_obl = h_fb[0]*(-my_bet) + h_fb[1]*(sd*(stake + bet_l))              # opp bets large
                ev_check = o_sc[0]*ev_cc + o_sc[1]*ev_obs + o_sc[2]*ev_obl

                opp_evs_c = np.array([
                    -sd*stake,
                    -(h_fb[0]*(-opp_bet) + h_fb[1]*(-sd*(stake+bet_s))),
                    -(h_fb[0]*(-opp_bet) + h_fb[1]*(-sd*(stake+bet_l)))
                ])
                opp_avg_c = np.dot(o_sc, opp_evs_c)
                opp_regret_after_check[oi] = np.maximum(
                    opp_regret_after_check[oi] + w*(opp_evs_c - opp_avg_c), 0)

                if o_sc[1] + o_sc[2] > 0.01:
                    fb_evs = np.array([-my_bet, sd*(stake + bet_s)])
                    fb_avg = np.dot(h_fb, fb_evs)
                    hero_regret_facing_bet = np.maximum(
                        hero_regret_facing_bet + w*(fb_evs - fb_avg), 0)

                action_evs[0] += w * ev_check

                # ─ BET_S ─
                if "BET_S" in hero_actions:
                    idx = hero_actions.index("BET_S")
                    op_bs = np.maximum(opp_regret_after_bets[oi], 0)
                    ot_bs = op_bs.sum()
                    o_sbs = op_bs / ot_bs if ot_bs > 0 else np.ones(3) / 3

                    ev_f = opp_bet                                    # opp folds
                    ev_c = sd * (stake + bet_s)                       # opp calls
                    ev_r = sd * min(stake + bet_s*2, 100)             # opp raises
                    ev_bets = o_sbs[0]*ev_f + o_sbs[1]*ev_c + o_sbs[2]*ev_r

                    opp_evs_bs = np.array([-opp_bet, -sd*(stake+bet_s), -sd*min(stake+bet_s*2, 100)])
                    opp_avg_bs = np.dot(o_sbs, opp_evs_bs)
                    opp_regret_after_bets[oi] = np.maximum(
                        opp_regret_after_bets[oi] + w*(opp_evs_bs - opp_avg_bs), 0)

                    action_evs[idx] += w * ev_bets

                # ─ BET_L ─
                if "BET_L" in hero_actions:
                    idx = hero_actions.index("BET_L")
                    op_bl = np.maximum(opp_regret_after_betl[oi], 0)
                    ot_bl = op_bl.sum()
                    o_sbl = op_bl / ot_bl if ot_bl > 0 else np.array([0.5, 0.5])

                    ev_f = opp_bet                          # opp folds
                    ev_c = sd * (stake + bet_l)              # opp calls
                    ev_betl = o_sbl[0]*ev_f + o_sbl[1]*ev_c

                    opp_evs_bl = np.array([-opp_bet, -sd*(stake+bet_l)])
                    opp_avg_bl = np.dot(o_sbl, opp_evs_bl)
                    opp_regret_after_betl[oi] = np.maximum(
                        opp_regret_after_betl[oi] + w*(opp_evs_bl - opp_avg_bl), 0)

                    action_evs[idx] += w * ev_betl

        # Update hero regret
        avg_ev = np.dot(h_strat, action_evs)
        hero_regret = np.maximum(hero_regret + (action_evs - avg_ev), 0)

    # Average strategy (linear weighted)
    total = hero_strat_sum.sum()
    final_strat = hero_strat_sum / total if total > 0 else np.ones(n_hero) / n_hero

    # Sample action
    chosen_idx = random.choices(range(n_hero), weights=final_strat.tolist(), k=1)[0]
    chosen_name = hero_actions[chosen_idx]

    if chosen_name == "FOLD":
        return (_FOLD, 0)
    elif chosen_name == "CHECK":
        return (_CHECK, 0)
    elif chosen_name == "CALL":
        return (_CALL, 0)
    elif chosen_name in ("BET_S", "RAISE_S"):
        return (_RAISE, bet_s)
    elif chosen_name in ("BET_L", "RAISE_L"):
        return (_RAISE, bet_l)
    return None, None

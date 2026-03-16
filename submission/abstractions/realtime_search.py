"""
Post-discard Full CFR solver.

Unlike MCCFR which samples one hand per iteration, this enumerates ALL
possible opponent hands and updates all strategies simultaneously.
This is the PioSolver/GTO Wizard approach.

Post-discard state space is small enough:
  - 27 cards total, 5 board + 3 hero discard + 3 opp discard = 11 dead
  - Remaining: 16 cards, opp has 2 → C(16,2) = 120 possible opp hands
  - River: exact showdown (no MC needed)
  - Turn: enumerate 1 river card = ~14 runouts
  - Flop: enumerate 2 cards = ~C(14,2) = 91 runouts

With 120 opp hands × 100 iterations → converges in milliseconds.
"""

import random
import numpy as np
from itertools import combinations
from abstractions.card_utils import get_evaluator, int_to_treys, ALL_CARDS

_FOLD = 0
_RAISE = 1
_CHECK = 2
_CALL = 3


def _enumerate_opp_hands(my_hand, community, my_discards, opp_discards):
    """Enumerate all possible opponent 2-card hands."""
    dead = set(my_hand) | set(community)
    for c in my_discards:
        if c >= 0: dead.add(c)
    for c in opp_discards:
        if c >= 0: dead.add(c)
    remaining = [c for c in ALL_CARDS if c not in dead]
    return list(combinations(remaining, 2)), remaining


def _compute_exact_equities(my_hand, community, opp_hands_list, remaining):
    """
    Compute exact equity for each opp hand.
    River: exact showdown. Turn/Flop: enumerate all runouts.
    """
    ev = get_evaluator()
    my_h = [int_to_treys(c) for c in my_hand]
    board_need = 5 - len(community)
    n = len(opp_hands_list)
    equities = np.zeros(n)

    if board_need == 0:
        # River: exact
        b = [int_to_treys(c) for c in community]
        mr = ev.evaluate(my_h, b)
        for oi, opp in enumerate(opp_hands_list):
            opp_h = [int_to_treys(c) for c in opp]
            opr = ev.evaluate(opp_h, b)
            equities[oi] = 1.0 if mr < opr else (0.5 if mr == opr else 0.0)
    else:
        # Turn/Flop: enumerate runouts
        for oi, opp in enumerate(opp_hands_list):
            opp_h = [int_to_treys(c) for c in opp]
            opp_set = set(opp)
            board_cards = [c for c in remaining if c not in opp_set]

            if len(board_cards) < board_need:
                equities[oi] = 0.5
                continue

            # Enumerate all runouts (or sample if too many)
            if board_need == 1:
                runouts = [[c] for c in board_cards]
            elif board_need == 2:
                runouts = list(combinations(board_cards, 2))
                if len(runouts) > 100:
                    runouts = random.sample(runouts, 100)
            else:
                runouts = [random.sample(board_cards, board_need) for _ in range(50)]

            wins = ties = 0
            for extra in runouts:
                b = [int_to_treys(c) for c in list(community) + list(extra)]
                mr = ev.evaluate(my_h, b)
                opr = ev.evaluate(opp_h, b)
                if mr < opr: wins += 1
                elif mr == opr: ties += 1
            total = len(runouts)
            equities[oi] = (wins + 0.5 * ties) / total if total > 0 else 0.5

    return equities


def solve_subgame(my_hand, community, my_bet, opp_bet, min_raise, max_raise,
                   valid_actions, opp_hands, opp_weights=None,
                   num_iters=150, my_discards=None, opp_discards=None):
    """
    Full CFR post-discard solver.

    Enumerates ALL possible opp hands, computes exact equity,
    runs CFR with per-hand opponent strategies.

    Returns (action_type, raise_amount).
    """
    pot = my_bet + opp_bet
    can_raise = valid_actions[_RAISE] and max_raise > 0
    facing_bet = (opp_bet > my_bet)
    stake = min(my_bet, opp_bet)

    # Enumerate all opp hands if not provided
    if not opp_hands:
        my_disc = my_discards or []
        opp_disc = opp_discards or []
        opp_hands, remaining = _enumerate_opp_hands(my_hand, community, my_disc, opp_disc)
    else:
        dead = set(my_hand) | set(community)
        remaining = [c for c in ALL_CARDS if c not in dead]

    n_opp = len(opp_hands)
    if n_opp == 0:
        return None, None

    # Bet sizes
    bet_sizes = []
    if can_raise:
        for frac in [0.33, 0.67, 1.0]:
            amt = max(min_raise, min(int(pot * frac), max_raise))
            if amt not in bet_sizes:
                bet_sizes.append(amt)

    # Hero actions
    if facing_bet:
        hero_actions = ["FOLD", "CALL"]
        hero_amounts = [0, 0]
        for sz in bet_sizes:
            hero_actions.append(f"RAISE_{sz}")
            hero_amounts.append(sz)
    else:
        hero_actions = ["CHECK"]
        hero_amounts = [0]
        for sz in bet_sizes:
            hero_actions.append(f"BET_{sz}")
            hero_amounts.append(sz)

    n_hero = len(hero_actions)
    if n_hero <= 1:
        return None, None

    # Opp weights (discard-aware if available)
    if opp_weights and len(opp_weights) == n_opp:
        w_total = sum(opp_weights)
        opp_w = np.array([w / w_total for w in opp_weights]) if w_total > 0 else np.ones(n_opp) / n_opp
    else:
        opp_w = np.ones(n_opp) / n_opp

    # Compute equities (exact on river, enumerated on turn/flop)
    eq = _compute_exact_equities(my_hand, community, opp_hands, remaining)
    ev_mult = 2.0 * eq - 1.0  # hero's signed EV multiplier per opp hand

    # Regret tables: PER OPP HAND (not bucketed — full enumeration)
    hero_regret = np.zeros(n_hero)
    hero_strat_sum = np.zeros(n_hero)

    # Opp response: per hand, per bet size → [FOLD, CALL]
    n_bets = len(bet_sizes)
    opp_regret_vs_bet = np.zeros((n_opp, max(n_bets, 1), 2))

    # Opp response after hero check: [CHECK_BACK, BET_sz1, BET_sz2, ...]
    n_opp_check_acts = n_bets + 1
    opp_regret_after_check = np.zeros((n_opp, n_opp_check_acts))
    hero_facing_regret = np.zeros(2)  # FOLD/CALL when opp bets

    for _t in range(num_iters):
        # Hero strategy
        pos = np.maximum(hero_regret, 0)
        tot = pos.sum()
        h_strat = pos / tot if tot > 0 else np.ones(n_hero) / n_hero
        hero_strat_sum += h_strat * max(_t, 1)

        # Hero facing bet
        pos_fb = np.maximum(hero_facing_regret, 0)
        tot_fb = pos_fb.sum()
        h_fb = pos_fb / tot_fb if tot_fb > 0 else np.array([0.5, 0.5])

        action_evs = np.zeros(n_hero)

        # Full enumeration over ALL opp hands
        for oi in range(n_opp):
            w = opp_w[oi]
            sd = ev_mult[oi]

            if facing_bet:
                action_evs[0] += w * (-my_bet)      # FOLD
                action_evs[1] += w * (sd * opp_bet)  # CALL

                for si, sz in enumerate(bet_sizes):
                    ai = 2 + si
                    new_bet = opp_bet + sz
                    op = np.maximum(opp_regret_vs_bet[oi, si], 0)
                    ot = op.sum()
                    o_s = op / ot if ot > 0 else np.array([0.5, 0.5])

                    ev_f = opp_bet
                    ev_c = sd * new_bet
                    action_evs[ai] += w * (o_s[0]*ev_f + o_s[1]*ev_c)

                    # Update opp regret
                    opp_evs = np.array([-opp_bet, -sd*new_bet])
                    opp_avg = np.dot(o_s, opp_evs)
                    opp_regret_vs_bet[oi, si] = np.maximum(
                        opp_regret_vs_bet[oi, si] + (opp_evs - opp_avg), 0)
            else:
                # CHECK
                op_c = np.maximum(opp_regret_after_check[oi, :n_opp_check_acts], 0)
                ot_c = op_c.sum()
                o_sc = op_c / ot_c if ot_c > 0 else np.ones(n_opp_check_acts) / n_opp_check_acts

                ev_cc = sd * stake
                ev_check = o_sc[0] * ev_cc
                for osi, osz in enumerate(bet_sizes):
                    ev_opp_bet = h_fb[0]*(-my_bet) + h_fb[1]*(sd*(stake + osz))
                    ev_check += o_sc[1+osi] * ev_opp_bet

                # Opp regret after check
                opp_evs_c = np.zeros(n_opp_check_acts)
                opp_evs_c[0] = -sd * stake
                for osi, osz in enumerate(bet_sizes):
                    opp_evs_c[1+osi] = -(h_fb[0]*(-opp_bet) + h_fb[1]*(-sd*(stake+osz)))
                opp_avg_c = np.dot(o_sc, opp_evs_c)
                opp_regret_after_check[oi, :n_opp_check_acts] = np.maximum(
                    opp_regret_after_check[oi, :n_opp_check_acts] + (opp_evs_c - opp_avg_c), 0)

                # Hero facing-bet regret
                if np.sum(o_sc[1:]) > 0.01 and n_bets > 0:
                    avg_sz = sum(o_sc[1+i]*bet_sizes[i] for i in range(n_bets)) / max(np.sum(o_sc[1:]), 0.01)
                    fb_evs = np.array([-my_bet, sd*(stake + avg_sz)])
                    fb_avg = np.dot(h_fb, fb_evs)
                    hero_facing_regret = np.maximum(hero_facing_regret + w*(fb_evs - fb_avg), 0)

                action_evs[0] += w * ev_check

                # BET sizes
                for si, sz in enumerate(bet_sizes):
                    ai = 1 + si
                    op_b = np.maximum(opp_regret_vs_bet[oi, si], 0)
                    ot_b = op_b.sum()
                    o_sb = op_b / ot_b if ot_b > 0 else np.array([0.5, 0.5])

                    ev_f = opp_bet
                    ev_c = sd * (stake + sz)
                    action_evs[ai] += w * (o_sb[0]*ev_f + o_sb[1]*ev_c)

                    opp_evs_b = np.array([-opp_bet, -sd*(stake+sz)])
                    opp_avg_b = np.dot(o_sb, opp_evs_b)
                    opp_regret_vs_bet[oi, si] = np.maximum(
                        opp_regret_vs_bet[oi, si] + (opp_evs_b - opp_avg_b), 0)

        avg_ev = np.dot(h_strat, action_evs)
        hero_regret = np.maximum(hero_regret + (action_evs - avg_ev), 0)

    # Average strategy
    total = hero_strat_sum.sum()
    final_strat = hero_strat_sum / total if total > 0 else np.ones(n_hero) / n_hero

    chosen_idx = random.choices(range(n_hero), weights=final_strat.tolist(), k=1)[0]
    amt = hero_amounts[chosen_idx]

    if hero_actions[chosen_idx] == "FOLD":
        return (_FOLD, 0)
    elif hero_actions[chosen_idx] == "CHECK":
        return (_CHECK, 0)
    elif hero_actions[chosen_idx] == "CALL":
        return (_CALL, 0)
    else:
        return (_RAISE, amt)

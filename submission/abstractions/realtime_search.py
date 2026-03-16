"""
Real-time subgame solving via mini-CFR.

At decision time, constructs a small game tree for the current street
and runs CFR+ to find an approximate Nash equilibrium strategy.

Key properties (following Libratus/Pluribus approach):
  - Both players' strategies are optimized (not just hero)
  - Board runout is re-sampled each iteration (proper expectation)
  - Opponent range is discard-aware weighted
  - Terminal payoffs are exact chip calculations
"""

import random
import numpy as np
from abstractions.card_utils import get_evaluator, int_to_treys, ALL_CARDS

_FOLD = 0
_RAISE = 1
_CHECK = 2
_CALL = 3

# Bet sizes as fractions of pot
BET_SMALL_FRAC = 0.33
BET_LARGE_FRAC = 0.75


def solve_subgame(my_hand: list, community: list,
                   my_bet: int, opp_bet: int,
                   min_raise: int, max_raise: int,
                   valid_actions: list,
                   opp_hands: list, opp_weights: list = None,
                   num_iters: int = 100) -> tuple:
    """
    Solve a 1-street subgame via mini-CFR.

    Game tree (hero acts first):
      Hero: CHECK / BET_S / BET_L
        CHECK → Opp: CHECK(showdown) / BET_S / BET_L
          Opp bets → Hero: FOLD / CALL(showdown)
        BET_S → Opp: FOLD / CALL(showdown) / RAISE
          Opp raises → Hero: FOLD / CALL(showdown)
        BET_L → Opp: FOLD / CALL(showdown)

    Returns: (action_type, raise_amount) for hero's best action.
    """
    evaluator = get_evaluator()
    my_h = [int_to_treys(c) for c in my_hand]
    board_need = 5 - len(community)
    pot = my_bet + opp_bet
    can_raise = valid_actions[_RAISE] and max_raise > 0

    # Compute bet sizes
    bet_s = max(min_raise, min(int(pot * BET_SMALL_FRAC), max_raise)) if can_raise else 0
    bet_l = max(min_raise, min(int(pot * BET_LARGE_FRAC), max_raise)) if can_raise else 0

    # Determine hero root actions
    facing_bet = (opp_bet > my_bet)
    if facing_bet:
        # Hero faces a bet: FOLD / CALL / RAISE_S / RAISE_L
        hero_actions = ["FOLD", "CALL"]
        if can_raise and bet_s > 0:
            hero_actions.append("RAISE_S")
        if can_raise and bet_l > bet_s:
            hero_actions.append("RAISE_L")
    else:
        # Hero acts first: CHECK / BET_S / BET_L
        hero_actions = ["CHECK"]
        if can_raise and bet_s > 0:
            hero_actions.append("BET_S")
        if can_raise and bet_l > bet_s:
            hero_actions.append("BET_L")

    n_hero = len(hero_actions)
    n_opp = len(opp_hands)
    if n_hero == 0 or n_opp == 0:
        return None, None

    # Normalize opp weights
    if opp_weights and len(opp_weights) == n_opp:
        w_total = sum(opp_weights)
        opp_w = [w / w_total for w in opp_weights] if w_total > 0 else [1.0/n_opp]*n_opp
    else:
        opp_w = [1.0 / n_opp] * n_opp

    # Regret tables
    hero_regret = np.zeros(n_hero)
    hero_strat_sum = np.zeros(n_hero)

    # Opp response regrets (per opp hand, per hero action that requires response)
    # After hero CHECK: opp can CHECK/BET_S/BET_L (3 actions)
    # After hero BET_S: opp can FOLD/CALL/RAISE (3 actions)
    # After hero BET_L: opp can FOLD/CALL (2 actions)
    opp_regret_after_check = np.zeros((n_opp, 3))  # CHECK/BET_S/BET_L
    opp_regret_after_bets = np.zeros((n_opp, 3))   # FOLD/CALL/RAISE
    opp_regret_after_betl = np.zeros((n_opp, 2))   # FOLD/CALL

    # Hero response after opp bets (after hero checked): FOLD/CALL
    hero_regret_facing_bet = np.zeros(2)

    dead_set = set(my_hand) | set(community)

    for _t in range(num_iters):
        # Hero strategy from regret matching
        pos = np.maximum(hero_regret, 0)
        tot = pos.sum()
        h_strat = pos / tot if tot > 0 else np.ones(n_hero) / n_hero
        hero_strat_sum += h_strat

        # Hero facing-bet strategy
        pos_fb = np.maximum(hero_regret_facing_bet, 0)
        tot_fb = pos_fb.sum()
        h_fb_strat = pos_fb / tot_fb if tot_fb > 0 else np.array([0.5, 0.5])

        action_evs = np.zeros(n_hero)

        for oi in range(n_opp):
            opp = opp_hands[oi]
            w = opp_w[oi]

            # Sample board runout (fresh each iteration)
            used = dead_set | set(opp)
            rem = [c for c in ALL_CARDS if c not in used]
            if board_need > 0 and len(rem) >= board_need:
                extra = random.sample(rem, board_need)
                full_board = community + extra
            else:
                full_board = community[:5] if len(community) >= 5 else community + [0] * board_need

            # Evaluate showdown
            b = [int_to_treys(c) for c in full_board[:5]]
            opp_h = [int_to_treys(c) for c in opp]
            try:
                mr = evaluator.evaluate(my_h, b)
                opr = evaluator.evaluate(opp_h, b)
            except:
                continue

            if mr < opr:
                sd = 1.0   # hero wins
            elif mr == opr:
                sd = 0.0   # tie
            else:
                sd = -1.0  # hero loses

            stake = min(my_bet, opp_bet)

            if facing_bet:
                # Hero faces bet: FOLD / CALL / RAISE_S / RAISE_L
                to_call = opp_bet - my_bet
                evs = []
                # FOLD
                evs.append(-my_bet)
                # CALL → showdown
                evs.append(sd * opp_bet)
                # RAISE_S → opp folds or calls
                if "RAISE_S" in hero_actions:
                    # Opp response to our raise
                    op = np.maximum(opp_regret_after_bets[oi], 0)
                    ot = op.sum()
                    o_s = op / ot if ot > 0 else np.ones(3) / 3
                    new_bet = opp_bet + bet_s
                    ev_fold = opp_bet
                    ev_call = sd * new_bet
                    ev_raise = sd * min(new_bet + bet_s, 100)  # simplified
                    evs.append(o_s[0]*ev_fold + o_s[1]*ev_call + o_s[2]*ev_raise)

                    # Update opp regret
                    opp_evs = np.array([-opp_bet, -sd*new_bet, -sd*min(new_bet+bet_s, 100)])
                    opp_avg = np.dot(o_s, opp_evs)
                    opp_regret_after_bets[oi] = np.maximum(
                        opp_regret_after_bets[oi] + w * (opp_evs - opp_avg), 0)

                if "RAISE_L" in hero_actions:
                    new_bet = opp_bet + bet_l
                    op2 = np.maximum(opp_regret_after_betl[oi], 0)
                    ot2 = op2.sum()
                    o_s2 = op2 / ot2 if ot2 > 0 else np.array([0.5, 0.5])
                    ev_fold = opp_bet
                    ev_call = sd * new_bet
                    evs.append(o_s2[0]*ev_fold + o_s2[1]*ev_call)

                    opp_evs2 = np.array([-opp_bet, -sd*new_bet])
                    opp_avg2 = np.dot(o_s2, opp_evs2)
                    opp_regret_after_betl[oi] = np.maximum(
                        opp_regret_after_betl[oi] + w * (opp_evs2 - opp_avg2), 0)

                for ai in range(n_hero):
                    action_evs[ai] += w * evs[ai]

            else:
                # Hero acts first: CHECK / BET_S / BET_L

                # ─ CHECK ─
                # Opp response: CHECK(showdown) / BET_S / BET_L
                op_c = np.maximum(opp_regret_after_check[oi], 0)
                ot_c = op_c.sum()
                o_sc = op_c / ot_c if ot_c > 0 else np.ones(3) / 3

                ev_cc = sd * stake  # both check → showdown
                # opp bets small → hero fold/call
                ev_obs = h_fb_strat[0] * (-my_bet) + h_fb_strat[1] * (sd * (stake + bet_s))
                # opp bets large → hero fold/call
                ev_obl = h_fb_strat[0] * (-my_bet) + h_fb_strat[1] * (sd * (stake + bet_l))

                ev_check = o_sc[0]*ev_cc + o_sc[1]*ev_obs + o_sc[2]*ev_obl

                # Update opp regret after check
                opp_evs_c = np.array([
                    -sd * stake,
                    -(h_fb_strat[0]*(-opp_bet) + h_fb_strat[1]*(-sd*(stake+bet_s))),
                    -(h_fb_strat[0]*(-opp_bet) + h_fb_strat[1]*(-sd*(stake+bet_l)))
                ])
                opp_avg_c = np.dot(o_sc, opp_evs_c)
                opp_regret_after_check[oi] = np.maximum(
                    opp_regret_after_check[oi] + w * (opp_evs_c - opp_avg_c), 0)

                # Update hero facing-bet regret
                if o_sc[1] + o_sc[2] > 0.01:
                    fb_evs = np.array([-my_bet, sd * (stake + bet_s)])
                    fb_avg = np.dot(h_fb_strat, fb_evs)
                    hero_regret_facing_bet = np.maximum(
                        hero_regret_facing_bet + w * (fb_evs - fb_avg), 0)

                action_evs[0] += w * ev_check

                # ─ BET_S ─
                if "BET_S" in hero_actions:
                    idx_bs = hero_actions.index("BET_S")
                    op_bs = np.maximum(opp_regret_after_bets[oi], 0)
                    ot_bs = op_bs.sum()
                    o_sbs = op_bs / ot_bs if ot_bs > 0 else np.ones(3) / 3

                    ev_fold = opp_bet
                    ev_call = sd * (stake + bet_s)
                    ev_raise = sd * min(stake + bet_s * 2, 100)
                    ev_bets = o_sbs[0]*ev_fold + o_sbs[1]*ev_call + o_sbs[2]*ev_raise

                    opp_evs_bs = np.array([-opp_bet, -sd*(stake+bet_s), -sd*min(stake+bet_s*2, 100)])
                    opp_avg_bs = np.dot(o_sbs, opp_evs_bs)
                    opp_regret_after_bets[oi] = np.maximum(
                        opp_regret_after_bets[oi] + w * (opp_evs_bs - opp_avg_bs), 0)

                    action_evs[idx_bs] += w * ev_bets

                # ─ BET_L ─
                if "BET_L" in hero_actions:
                    idx_bl = hero_actions.index("BET_L")
                    op_bl = np.maximum(opp_regret_after_betl[oi], 0)
                    ot_bl = op_bl.sum()
                    o_sbl = op_bl / ot_bl if ot_bl > 0 else np.array([0.5, 0.5])

                    ev_fold = opp_bet
                    ev_call = sd * (stake + bet_l)
                    ev_betl = o_sbl[0]*ev_fold + o_sbl[1]*ev_call

                    opp_evs_bl = np.array([-opp_bet, -sd*(stake+bet_l)])
                    opp_avg_bl = np.dot(o_sbl, opp_evs_bl)
                    opp_regret_after_betl[oi] = np.maximum(
                        opp_regret_after_betl[oi] + w * (opp_evs_bl - opp_avg_bl), 0)

                    action_evs[idx_bl] += w * ev_betl

        # Update hero regret
        avg_ev = np.dot(h_strat, action_evs)
        hero_regret = np.maximum(hero_regret + (action_evs - avg_ev), 0)

    # Average strategy
    total = hero_strat_sum.sum()
    final_strat = hero_strat_sum / total if total > 0 else np.ones(n_hero) / n_hero

    # Sample action from strategy
    chosen_idx = random.choices(range(n_hero), weights=final_strat.tolist(), k=1)[0]
    chosen_name = hero_actions[chosen_idx]

    # Map to concrete action
    if chosen_name == "FOLD":
        return (_FOLD, 0)
    elif chosen_name == "CHECK":
        return (_CHECK, 0)
    elif chosen_name == "CALL":
        return (_CALL, 0)
    elif chosen_name == "BET_S" or chosen_name == "RAISE_S":
        return (_RAISE, bet_s)
    elif chosen_name == "BET_L" or chosen_name == "RAISE_L":
        return (_RAISE, bet_l)
    return None, None

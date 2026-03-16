"""
Real-time subgame solving v4 — expanded action space.

Blueprint uses 2-3 bet sizes. Subgame expands to 5 sizes:
  No bet:    CHECK, BET_25%, BET_50%, BET_75%, BET_ALLIN
  Facing bet: FOLD, CALL, RAISE_50%, RAISE_100%, RAISE_ALLIN

Key: the value of subgame solving comes from finding the PRECISE bet size,
not just "bet or check". With 5 sizes, CFR can find that e.g. 50% pot is
optimal when blueprint only had "small" or "large".

Uses pre-computed MC equity as value function at terminal nodes.
Opponent hands bucketed by equity (4 buckets) for fast convergence.
"""

import random
import numpy as np
from abstractions.card_utils import get_evaluator, int_to_treys, ALL_CARDS

_FOLD = 0
_RAISE = 1
_CHECK = 2
_CALL = 3
OPP_BUCKETS = 4


def _compute_equities(my_hand, community, opp_hands, num_sims=80):
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
            sims = min(num_sims, len(rem)*(len(rem)-1)//2) if board_need <= 2 else num_sims
            for _ in range(sims):
                extra = random.sample(rem, board_need)
                b = [int_to_treys(c) for c in community + extra]
                mr = ev.evaluate(my_h, b)
                opr = ev.evaluate(opp_h, b)
                if mr < opr: wins += 1
                elif mr == opr: ties += 1
            equities[oi] = (wins + 0.5*ties) / sims if sims > 0 else 0.5
        else:
            equities[oi] = 0.5
    return equities


def _make_bet_sizes(pot, min_raise, max_raise):
    """Generate 4 distinct bet sizes from min to max."""
    if max_raise <= 0 or max_raise < min_raise:
        return []
    sizes = set()
    for frac in [0.25, 0.50, 0.75, 1.0]:
        amt = max(min_raise, min(int(pot * frac), max_raise))
        sizes.add(amt)
    # Always include all-in if not already
    sizes.add(max_raise)
    sizes = sorted(sizes)
    # Keep at most 5 distinct sizes
    return sizes[:5]


def solve_subgame(my_hand, community, my_bet, opp_bet, min_raise, max_raise,
                   valid_actions, opp_hands, opp_weights=None,
                   num_iters=500):
    """
    Expanded action space subgame solving.
    Returns (action_type, raise_amount).
    """
    pot = my_bet + opp_bet
    can_raise = valid_actions[_RAISE] and max_raise > 0
    facing_bet = (opp_bet > my_bet)

    # Build expanded action space
    bet_sizes = _make_bet_sizes(pot, min_raise, max_raise) if can_raise else []

    if facing_bet:
        # FOLD, CALL, RAISE_size1, RAISE_size2, ...
        hero_actions = ["FOLD", "CALL"]
        hero_amounts = [0, 0]
        for sz in bet_sizes:
            hero_actions.append(f"RAISE_{sz}")
            hero_amounts.append(sz)
    else:
        # CHECK, BET_size1, BET_size2, ...
        hero_actions = ["CHECK"]
        hero_amounts = [0]
        for sz in bet_sizes:
            hero_actions.append(f"BET_{sz}")
            hero_amounts.append(sz)

    n_hero = len(hero_actions)
    n_opp = len(opp_hands)
    if n_hero <= 1 or n_opp == 0:
        return None, None

    # Normalize weights
    if opp_weights and len(opp_weights) == n_opp:
        w_total = sum(opp_weights)
        opp_w = np.array([w/w_total for w in opp_weights]) if w_total > 0 else np.ones(n_opp)/n_opp
    else:
        opp_w = np.ones(n_opp) / n_opp

    # Pre-compute equities
    eq = _compute_equities(my_hand, community, opp_hands, num_sims=80)
    ev_mult = 2.0 * eq - 1.0

    # Bucket opponent hands
    bucket_bounds = [0.0, 0.3, 0.5, 0.7, 1.01]
    bucket_weight = np.zeros(OPP_BUCKETS)
    bucket_ev = np.zeros(OPP_BUCKETS)
    for oi in range(n_opp):
        opp_eq = 1.0 - eq[oi]
        for bi in range(OPP_BUCKETS):
            if bucket_bounds[bi] <= opp_eq < bucket_bounds[bi+1]:
                bucket_weight[bi] += opp_w[oi]
                bucket_ev[bi] += opp_w[oi] * ev_mult[oi]
                break
    for bi in range(OPP_BUCKETS):
        if bucket_weight[bi] > 0:
            bucket_ev[bi] /= bucket_weight[bi]

    # Opponent response: for each hero bet size, opp can FOLD or CALL
    # (simplified: no opp re-raise to keep tree manageable with many sizes)
    # Per bucket, one regret array for facing each bet size: [FOLD, CALL]
    n_bet_sizes = len(bet_sizes)
    opp_regret = np.zeros((OPP_BUCKETS, max(n_bet_sizes, 1), 2))  # [bucket][size_idx][fold/call]

    # If not facing bet: opp also needs response after hero checks
    # Opp can CHECK or BET with various sizes
    opp_check_regret = np.zeros((OPP_BUCKETS, n_bet_sizes + 1))  # [bucket][check + bet sizes]
    hero_facing_regret = np.zeros(2)  # hero FOLD/CALL when opp bets after check

    hero_regret = np.zeros(n_hero)
    hero_strat_sum = np.zeros(n_hero)
    stake = min(my_bet, opp_bet)

    for _t in range(num_iters):
        # Hero strategy
        pos = np.maximum(hero_regret, 0)
        tot = pos.sum()
        h_strat = pos / tot if tot > 0 else np.ones(n_hero) / n_hero
        hero_strat_sum += h_strat * max(_t, 1)

        # Hero facing bet after check
        pos_fb = np.maximum(hero_facing_regret, 0)
        tot_fb = pos_fb.sum()
        h_fb = pos_fb / tot_fb if tot_fb > 0 else np.array([0.5, 0.5])

        action_evs = np.zeros(n_hero)

        for bi in range(OPP_BUCKETS):
            w = bucket_weight[bi]
            if w < 1e-6:
                continue
            sd = bucket_ev[bi]

            if facing_bet:
                # Hero: FOLD, CALL, RAISE_sz1, RAISE_sz2, ...
                action_evs[0] += w * (-my_bet)       # FOLD
                action_evs[1] += w * (sd * opp_bet)   # CALL → showdown

                for si, sz in enumerate(bet_sizes):
                    ai = 2 + si  # action index
                    new_bet = opp_bet + sz

                    # Opp response to raise
                    op = np.maximum(opp_regret[bi, si], 0)
                    ot = op.sum()
                    o_s = op / ot if ot > 0 else np.array([0.5, 0.5])

                    ev_fold = opp_bet           # opp folds
                    ev_call = sd * new_bet      # showdown
                    hero_ev = o_s[0]*ev_fold + o_s[1]*ev_call
                    action_evs[ai] += w * hero_ev

                    # Update opp regret
                    opp_evs = np.array([-opp_bet, -sd*new_bet])
                    opp_avg = np.dot(o_s, opp_evs)
                    opp_regret[bi, si] = np.maximum(
                        opp_regret[bi, si] + w*(opp_evs - opp_avg), 0)

            else:
                # Hero: CHECK, BET_sz1, BET_sz2, ...

                # ─ CHECK ─
                # Opp can: check back, or bet with any size
                n_opp_acts = n_bet_sizes + 1
                op_c = np.maximum(opp_check_regret[bi, :n_opp_acts], 0)
                ot_c = op_c.sum()
                o_sc = op_c / ot_c if ot_c > 0 else np.ones(n_opp_acts) / n_opp_acts

                ev_check_check = sd * stake  # both check → showdown

                # Opp bets → hero fold/call (use avg bet size for simplicity)
                opp_bet_evs = []
                for osi, osz in enumerate(bet_sizes):
                    ev_opp_bet = h_fb[0]*(-my_bet) + h_fb[1]*(sd*(stake + osz))
                    opp_bet_evs.append(ev_opp_bet)

                ev_check = o_sc[0] * ev_check_check
                for osi in range(n_bet_sizes):
                    ev_check += o_sc[1 + osi] * opp_bet_evs[osi]

                # Opp regret after hero check
                opp_evs_c = np.zeros(n_opp_acts)
                opp_evs_c[0] = -sd * stake  # opp checks back
                for osi, osz in enumerate(bet_sizes):
                    opp_evs_c[1+osi] = -(h_fb[0]*(-opp_bet) + h_fb[1]*(-sd*(stake+osz)))
                opp_avg_c = np.dot(o_sc, opp_evs_c)
                opp_check_regret[bi, :n_opp_acts] = np.maximum(
                    opp_check_regret[bi, :n_opp_acts] + w*(opp_evs_c - opp_avg_c), 0)

                # Hero facing-bet regret (avg over opp bet sizes)
                if sum(o_sc[1:]) > 0.01 and n_bet_sizes > 0:
                    avg_opp_sz = sum(o_sc[1+i]*bet_sizes[i] for i in range(n_bet_sizes)) / max(sum(o_sc[1:]), 0.01)
                    fb_evs = np.array([-my_bet, sd*(stake + avg_opp_sz)])
                    fb_avg = np.dot(h_fb, fb_evs)
                    hero_facing_regret = np.maximum(
                        hero_facing_regret + w*(fb_evs - fb_avg), 0)

                action_evs[0] += w * ev_check

                # ─ BET sizes ─
                for si, sz in enumerate(bet_sizes):
                    ai = 1 + si

                    # Opp response: FOLD / CALL
                    op_b = np.maximum(opp_regret[bi, si], 0)
                    ot_b = op_b.sum()
                    o_sb = op_b / ot_b if ot_b > 0 else np.array([0.5, 0.5])

                    ev_fold = opp_bet
                    ev_call = sd * (stake + sz)
                    hero_ev = o_sb[0]*ev_fold + o_sb[1]*ev_call
                    action_evs[ai] += w * hero_ev

                    # Update opp regret
                    opp_evs_b = np.array([-opp_bet, -sd*(stake+sz)])
                    opp_avg_b = np.dot(o_sb, opp_evs_b)
                    opp_regret[bi, si] = np.maximum(
                        opp_regret[bi, si] + w*(opp_evs_b - opp_avg_b), 0)

        avg_ev = np.dot(h_strat, action_evs)
        hero_regret = np.maximum(hero_regret + (action_evs - avg_ev), 0)

    # Average strategy
    total = hero_strat_sum.sum()
    final_strat = hero_strat_sum / total if total > 0 else np.ones(n_hero) / n_hero

    # Sample action
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

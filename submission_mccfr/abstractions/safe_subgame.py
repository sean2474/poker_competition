"""
Safe Subgame Solving (Libratus-style).

Key difference from unsafe: opponent has a "gadget" option to
terminate the subgame and receive their blueprint counterfactual value.
This guarantees the new strategy is no worse than the blueprint.

Algorithm:
  1. Compute opponent's blueprint CFV for each possible hand
  2. Build subgame with expanded action space + gadget terminate option
  3. Run CFR where opponent can always choose TERMINATE(receive CFV)
  4. Hero's average strategy is safe to use

Reference: "Safe and Nested Subgame Solving" (Brown & Sandholm, NIPS 2017)
"""

import random
import numpy as np
from itertools import combinations
from abstractions.card_utils import get_evaluator, int_to_treys, ALL_CARDS


_FOLD = 0
_RAISE = 1
_CHECK = 2
_CALL = 3
MAX_BET = 100


def compute_blueprint_cfv(hero_hand, community, opp_hands, opp_reach,
                           my_bet, opp_bet, blueprint_strategy_fn):
    """
    Compute opponent's counterfactual value under blueprint strategy.
    
    For each opp hand, CFV = what opp expects to get at this game state
    if both players follow blueprint from here.
    
    Simplified: use MC equity as proxy for blueprint CFV.
    CFV(opp_hand) ≈ opp_equity × stake (from opp's perspective)
    
    This is an approximation — true CFV requires full blueprint traversal.
    But for our 27-card game this is close enough.
    """
    evaluator = get_evaluator()
    my_h = [int_to_treys(c) for c in hero_hand]
    board_need = 5 - len(community)
    dead_set = set(hero_hand) | set(community)
    remaining = [c for c in ALL_CARDS if c not in dead_set]
    stake = min(my_bet, opp_bet)
    
    cfvs = np.zeros(len(opp_hands))
    
    for oi, opp in enumerate(opp_hands):
        opp_h = [int_to_treys(c) for c in opp]
        opp_set = set(opp)
        rem = [c for c in remaining if c not in opp_set]
        
        if board_need == 0:
            b = [int_to_treys(c) for c in community]
            mr = evaluator.evaluate(my_h, b)
            opr = evaluator.evaluate(opp_h, b)
            # From opp's perspective: they win if opr < mr
            if opr < mr:
                cfvs[oi] = stake
            elif opr == mr:
                cfvs[oi] = 0
            else:
                cfvs[oi] = -stake
        elif len(rem) >= board_need:
            wins = ties = sims = 0
            for _ in range(50):
                extra = random.sample(rem, board_need)
                b = [int_to_treys(c) for c in community + extra]
                mr = evaluator.evaluate(my_h, b)
                opr = evaluator.evaluate(opp_h, b)
                if opr < mr: wins += 1
                elif opr == mr: ties += 1
                sims += 1
            opp_eq = (wins + 0.5 * ties) / sims if sims > 0 else 0.5
            cfvs[oi] = (2 * opp_eq - 1) * stake
        else:
            cfvs[oi] = 0
    
    return cfvs


def safe_subgame_solve(hero_hand, community, my_bet, opp_bet,
                        min_raise, max_raise, valid_actions,
                        opp_hands, opp_reach, opp_cfvs,
                        num_iters=200):
    """
    Safe subgame solving with gadget game.
    
    Args:
        hero_hand: our 2 cards
        community: board cards
        my_bet, opp_bet: current bets
        min_raise, max_raise: raise limits
        valid_actions: which actions are valid
        opp_hands: list of possible opponent 2-card hands
        opp_reach: probability weight for each opp hand
        opp_cfvs: blueprint counterfactual value for each opp hand (from opp's perspective)
        num_iters: CFR iterations
    
    Returns: (action_type, raise_amount) — hero's action
    """
    pot = my_bet + opp_bet
    can_raise = valid_actions[_RAISE] and max_raise > 0
    facing_bet = (opp_bet > my_bet)
    stake = min(my_bet, opp_bet)
    n_opp = len(opp_hands)
    
    if n_opp == 0:
        return None, None
    
    # Expanded bet sizes
    bet_sizes = []
    if can_raise:
        for frac in [0.33, 0.50, 0.75, 1.0]:
            amt = max(min_raise, min(int(pot * frac), max_raise))
            if amt not in bet_sizes:
                bet_sizes.append(amt)
    
    # Hero actions
    if facing_bet:
        hero_actions = ["FOLD", "CALL"] + [f"RAISE_{s}" for s in bet_sizes]
        hero_amounts = [0, 0] + list(bet_sizes)
    else:
        hero_actions = ["CHECK"] + [f"BET_{s}" for s in bet_sizes]
        hero_amounts = [0] + list(bet_sizes)
    
    n_hero = len(hero_actions)
    if n_hero <= 1:
        return None, None
    
    # Normalize opp reach
    opp_w = np.array(opp_reach, dtype=np.float64)
    wt = opp_w.sum()
    if wt > 0:
        opp_w /= wt
    else:
        opp_w = np.ones(n_opp) / n_opp
    
    # Pre-compute equity for each opp hand (hero's perspective)
    evaluator = get_evaluator()
    my_h = [int_to_treys(c) for c in hero_hand]
    board_need = 5 - len(community)
    dead_set = set(hero_hand) | set(community)
    remaining = [c for c in ALL_CARDS if c not in dead_set]
    
    hero_eq = np.zeros(n_opp)
    for oi, opp in enumerate(opp_hands):
        opp_h = [int_to_treys(c) for c in opp]
        opp_set = set(opp)
        rem = [c for c in remaining if c not in opp_set]
        if board_need == 0:
            b = [int_to_treys(c) for c in community]
            mr = evaluator.evaluate(my_h, b)
            opr = evaluator.evaluate(opp_h, b)
            hero_eq[oi] = 1.0 if mr < opr else (0.5 if mr == opr else 0.0)
        elif len(rem) >= board_need:
            w = t = 0
            for _ in range(30):
                extra = random.sample(rem, board_need)
                b = [int_to_treys(c) for c in community + extra]
                mr = evaluator.evaluate(my_h, b)
                opr = evaluator.evaluate(opp_h, b)
                if mr < opr: w += 1
                elif mr == opr: w += 0.5
                t += 1
            hero_eq[oi] = w / t if t > 0 else 0.5
        else:
            hero_eq[oi] = 0.5
    
    sd = 2.0 * hero_eq - 1.0  # hero's signed EV multiplier
    
    # Opp actions: FOLD / CALL / RAISE... / TERMINATE(gadget)
    # TERMINATE: opp receives their CFV, hero pays -CFV
    n_opp_actions = 2 + len(bet_sizes) + 1  # +1 for TERMINATE
    # But simplify: FOLD / CALL / TERMINATE
    n_opp_acts = 3  # FOLD, CALL, TERMINATE
    
    # Regret tables
    hero_regret = np.zeros(n_hero)
    hero_strat_sum = np.zeros(n_hero)
    opp_regret = np.zeros((n_opp, n_opp_acts))  # per opp hand
    
    for t in range(num_iters):
        # Hero strategy
        pos = np.maximum(hero_regret, 0)
        tot = pos.sum()
        h_strat = pos / tot if tot > 0 else np.ones(n_hero) / n_hero
        hero_strat_sum += h_strat * max(t, 1)
        
        hero_action_evs = np.zeros(n_hero)
        
        for oi in range(n_opp):
            w = opp_w[oi]
            if w < 1e-8:
                continue
            
            s = sd[oi]  # hero's signed EV multiplier for this opp hand
            cfv = opp_cfvs[oi]  # opp's blueprint CFV
            
            # Opp strategy for this hand
            op = np.maximum(opp_regret[oi], 0)
            ot = op.sum()
            o_strat = op / ot if ot > 0 else np.ones(n_opp_acts) / n_opp_acts
            
            # For each hero action, compute EV considering opp's response
            for ai in range(n_hero):
                if facing_bet:
                    if ai == 0:  # FOLD
                        hero_ev = -my_bet
                    elif ai == 1:  # CALL
                        # Opp doesn't respond to call, goes to showdown
                        hero_ev = s * opp_bet
                    else:  # RAISE
                        sz = hero_amounts[ai]
                        new_bet = opp_bet + sz
                        # Opp responds: FOLD / CALL / TERMINATE
                        ev_fold = opp_bet
                        ev_call = s * new_bet
                        ev_terminate = -cfv  # hero pays opp's CFV (from hero perspective)
                        hero_ev = o_strat[0]*ev_fold + o_strat[1]*ev_call + o_strat[2]*ev_terminate
                else:
                    if ai == 0:  # CHECK
                        # Opp can: check back (showdown) / bet / terminate
                        ev_check = s * stake
                        ev_terminate = -cfv
                        # Simplified: opp checks or terminates
                        hero_ev = o_strat[0]*ev_check + o_strat[1]*(s*(stake+bet_sizes[0]) if bet_sizes else ev_check) + o_strat[2]*ev_terminate
                    else:  # BET
                        sz = hero_amounts[ai]
                        ev_fold = opp_bet
                        ev_call = s * (stake + sz)
                        ev_terminate = -cfv
                        hero_ev = o_strat[0]*ev_fold + o_strat[1]*ev_call + o_strat[2]*ev_terminate
                
                hero_action_evs[ai] += w * hero_ev
            
            # Update opp regret (from opp's perspective: negate hero EV)
            for ai in range(n_hero):
                if facing_bet and ai >= 2:  # hero raised
                    sz = hero_amounts[ai]
                    new_bet = opp_bet + sz
                    opp_evs = np.array([
                        -opp_bet,           # FOLD: lose opp's bet
                        -s * new_bet,        # CALL: showdown (negate hero's EV)
                        cfv                  # TERMINATE: get blueprint CFV
                    ])
                elif not facing_bet and ai >= 1:  # hero bet
                    sz = hero_amounts[ai]
                    opp_evs = np.array([
                        -opp_bet,            # FOLD
                        -s * (stake + sz),   # CALL
                        cfv                  # TERMINATE
                    ])
                else:
                    continue  # no opp response needed for FOLD/CALL/CHECK
                
                opp_avg = np.dot(o_strat, opp_evs)
                # Weight by hero's probability of taking this action
                opp_regret[oi] = np.maximum(
                    opp_regret[oi] + w * h_strat[ai] * (opp_evs - opp_avg), 0)
        
        # Update hero regret
        avg = np.dot(h_strat, hero_action_evs)
        hero_regret = np.maximum(hero_regret + (hero_action_evs - avg), 0)
    
    # Average strategy
    total = hero_strat_sum.sum()
    final = hero_strat_sum / total if total > 0 else np.ones(n_hero) / n_hero
    
    # Sample action
    chosen = random.choices(range(n_hero), weights=final.tolist(), k=1)[0]
    amt = hero_amounts[chosen]
    
    name = hero_actions[chosen]
    if name == "FOLD":
        return (_FOLD, 0)
    elif name == "CHECK":
        return (_CHECK, 0)
    elif name == "CALL":
        return (_CALL, 0)
    else:
        return (_RAISE, amt)

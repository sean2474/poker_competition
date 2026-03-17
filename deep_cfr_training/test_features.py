"""
Feature verification tests.

Checks all 18 extra Python-side features (dims [93-110]) of state_to_features.
Also checks that C++ base (dims [0-92]) is non-zero for normal inputs.

Run:  python test_features.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from game.features import state_to_features
from game.constants import (
    FEATURE_DIM, CPP_FEATURE_DIM,
    A_FOLD, A_CALL, A_CHECK, A_BET_SMALL, A_BET_LARGE,
    A_RAISE_SMALL, A_RAISE_LARGE, A_BET_POT,
)

# Deck helpers: card = rank + suit*9, ranks 0-8, suits 0-2
def c(rank, suit=0): return rank + suit * 9

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_results = []

def check(name, cond, detail=""):
    ok = bool(cond)
    _results.append(ok)
    status = PASS if ok else FAIL
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    return ok


def feat(hero_hand, community, my_bet, opp_bet, street=2, is_bb=False,
         my_disc=None, opp_disc=None, history=None, n_acts=0):
    return state_to_features(
        hero_hand, community, my_bet, opp_bet, street, is_bb,
        my_discards=my_disc or [-1,-1,-1],
        opp_discards=opp_disc or [-1,-1,-1],
        street_bets=[[0,0],[0,0],[0,0],[0,0]],
        history=history or [],
        num_actions_this_street=n_acts,
    )

# Shortcuts: extra feature slice starts at CPP_FEATURE_DIM
def ex(f): return f[CPP_FEATURE_DIM:]   # 18-elem view


print("=" * 55)
print("Feature shape")
print("=" * 55)
f0 = feat([c(5), c(6)], [c(0), c(1), c(2)], 10, 20)
check("total dim == 111",  len(f0) == FEATURE_DIM, f"got {len(f0)}")
check("C++ base nonzero",  np.any(f0[:CPP_FEATURE_DIM] != 0))
check("extra slice len=18", len(ex(f0)) == 18)


print()
print("=" * 55)
print("[0-1] Initiative / last aggressor")
print("=" * 55)
# hero (player 0) bet last
hist_hero_agg = [(0, A_BET_SMALL)]
f = feat([c(5),c(6)], [c(0),c(1),c(2)], 10, 10, is_bb=False, history=hist_hero_agg, n_acts=1)
e = ex(f)
check("hero_last_aggressor=1 when p0 bet (not BB)", e[0] == 1.0, f"got {e[0]}")
check("villain_last_aggressor=0",                  e[1] == 0.0, f"got {e[1]}")

# villain (player 1) raised last; hero is BB (player 1 = hero)
hist_vill_agg = [(0, A_RAISE_SMALL)]
f = feat([c(5),c(6)], [c(0),c(1),c(2)], 10, 10, is_bb=True, history=hist_vill_agg, n_acts=1)
e = ex(f)
check("villain_last_aggressor=1 when p0 raised (BB=p1 is hero)", e[1] == 1.0, f"got {e[1]}")
check("hero_last_aggressor=0",                                    e[0] == 0.0, f"got {e[0]}")

# no aggressor
f = feat([c(5),c(6)], [c(0),c(1),c(2)], 10, 10, history=[(0,A_CHECK),(1,A_CHECK)], n_acts=2)
e = ex(f)
check("no aggressor → both 0",  e[0] == 0.0 and e[1] == 0.0, f"got {e[0]},{e[1]}")


print()
print("=" * 55)
print("[2-3] Action context")
print("=" * 55)
f_bet   = feat([c(5),c(6)], [c(0),c(1),c(2)], 5,  20)  # to_call=15 > 0
f_check = feat([c(5),c(6)], [c(0),c(1),c(2)], 10, 10)  # to_call=0
check("facing_bet=1 when to_call>0",  ex(f_bet)[2]   == 1.0, f"got {ex(f_bet)[2]}")
check("can_check=0 when to_call>0",   ex(f_bet)[3]   == 0.0)
check("facing_bet=0 when to_call=0",  ex(f_check)[2] == 0.0)
check("can_check=1 when to_call=0",   ex(f_check)[3] == 1.0, f"got {ex(f_check)[3]}")


print()
print("=" * 55)
print("[4-7] Line class")
print("=" * 55)
# checked_to: no bet this street, to_call==0
f = feat([c(5),c(6)], [c(0),c(1),c(2)], 10, 10, history=[(0,A_CHECK)], n_acts=1)
e = ex(f)
check("checked_to[4]=1",       e[4]==1.0, f"got {e[4:8]}")
check("others=0",              sum(e[5:8])==0.0)

# facing_lead: first bet, to_call>0, bets_this_street==1
f = feat([c(5),c(6)], [c(0),c(1),c(2)], 5, 20,
         history=[(0,A_CHECK),(1,A_BET_LARGE)], n_acts=2)
e = ex(f)
check("facing_lead[5]=1",      e[5]==1.0, f"got {e[4:8]}")

# facing_raise: to_call>0, bets>=2
f = feat([c(5),c(6)], [c(0),c(1),c(2)], 10, 40,
         history=[(0,A_BET_SMALL),(1,A_RAISE_LARGE)], n_acts=2)
e = ex(f)
check("facing_raise[6]=1",     e[6]==1.0, f"got {e[4:8]}")

# raised_pot: to_call==0, bets>=1 (e.g. hero just raised and opp called)
f = feat([c(5),c(6)], [c(0),c(1),c(2)], 20, 20,
         history=[(0,A_BET_SMALL),(1,A_RAISE_SMALL),(0,A_CALL)], n_acts=3)
e = ex(f)
check("raised_pot[7]=1",       e[7]==1.0, f"got {e[4:8]}")


print()
print("=" * 55)
print("[8-12] Board texture")
print("=" * 55)
# paired board: two cards with same rank
paired_board = [c(3,0), c(3,1), c(7,2)]
f = feat([c(5),c(6)], paired_board, 10, 10)
check("board_paired[8]=1",     ex(f)[8]==1.0, f"got {ex(f)[8]}")

# monotone: all same suit
mono_board = [c(0,1), c(3,1), c(7,1)]
f = feat([c(5),c(6)], mono_board, 10, 10)
check("monotone[9]=1",         ex(f)[9]==1.0, f"got {ex(f)[9]}")

# rainbow: all different suits
rainbow = [c(0,0), c(3,1), c(7,2)]
f = feat([c(5),c(6)], rainbow, 10, 10)
check("rainbow → monotone[9]=0", ex(f)[9]==0.0)

# two-suited: max suit count >= 2
two_suit = [c(0,0), c(3,0), c(7,2)]
f = feat([c(5),c(6)], two_suit, 10, 10)
check("two-suited[10]=1",      ex(f)[10]==1.0, f"got {ex(f)[10]}")

# connected: span<=4
conn_board = [c(2,0), c(3,1), c(5,2)]
f = feat([c(5),c(6)], conn_board, 10, 10)
check("connected[11]=1 (ranks 2,3,5 span=3)", ex(f)[11]==1.0, f"got {ex(f)[11]}")

disco_board = [c(0,0), c(4,1), c(8,2)]
f = feat([c(5),c(6)], disco_board, 10, 10)
check("disconnected[11]=0 (span=8)", ex(f)[11]==0.0, f"got {ex(f)[11]}")

# scare card: turn card completes flush (prev_suits had 2 of same suit)
turn_board = [c(0,1), c(3,1), c(7,0), c(5,1)]  # 3 suited after turn
f = feat([c(2),c(4)], turn_board, 10, 10, street=2)
check("scare[12]=1 (3rd card of suit on turn)", ex(f)[12]==1.0, f"got {ex(f)[12]}")

no_scare = [c(0,0), c(3,1), c(7,2), c(5,0)]  # only 2 of suit 0 on turn
f = feat([c(2),c(4)], no_scare, 10, 10, street=2)
check("scare[12]=0 (no flush completion)", ex(f)[12]==0.0, f"got {ex(f)[12]}")


print()
print("=" * 55)
print("[13-17] Bet ratios")
print("=" * 55)
# my_bet=20, opp_bet=60, pot=80, to_call=40
f = feat([c(5),c(6)], [c(0),c(1),c(2)], 20, 60)
e = ex(f)
pot = 80.0
check("to_call/pot = 40/80 = 0.5",  abs(e[13] - 0.5) < 0.01,  f"got {e[13]:.3f}")
check("opp_bet/pot = 60/80 = 0.75", abs(e[14] - 0.75) < 0.01, f"got {e[14]:.3f}")
check("my_bet/pot  = 20/80 = 0.25", abs(e[15] - 0.25) < 0.01, f"got {e[15]:.3f}")

# max_raise/pot: MAX_BET=100, max(bets)=60, max_r=40, max_r/pot=0.5
check("max_raise/pot = 40/80 = 0.5",  abs(e[16] - 0.5)  < 0.01, f"got {e[16]:.3f}")
check("remaining/100 = 40/100 = 0.4", abs(e[17] - 0.4)  < 0.01, f"got {e[17]:.3f}")

# pot=0 edge case → no div-by-zero
f_zero = feat([c(5),c(6)], [c(0),c(1),c(2)], 0, 0)
check("no crash when pot=0",   np.all(np.isfinite(ex(f_zero))))


print()
print("=" * 55)
total = len(_results)
passed = sum(_results)
symbol = PASS if passed == total else FAIL
print(f"Result: [{symbol}] {passed}/{total} tests passed")
print("=" * 55)
sys.exit(0 if passed == total else 1)

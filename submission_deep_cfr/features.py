"""
119-dim pure-Python feature extraction.
Mirrors deep_cfr_training/cpp/game_state.h state_to_features() exactly.

Layout:
  [0-19]    Hero hand (20)
  [20-39]   Community (20)
  [40-51]   My discards (12)
  [52-63]   Opp discards (12)
  [64-67]   Street one-hot (4)
  [68]      Position/is_bb (1)
  [69-72]   Bet info absolute (4)
  [73-78]   Hand strength (6)
  [79-86]   Betting history: last bet/pot ratio per street/player (8)
  [87-92]   Opp range estimate (6)
  ── extra 26 ──────────────────────────────────────────────────────
  [93-94]   Initiative: hero/villain last aggressor (2)
  [95-96]   Action context: facing_bet, can_check (2)
  [97-100]  Line class: checked_to/facing_lead/facing_raise/raised_pot (4)
  [101-105] Board texture: paired/monotone/two-suited/connected/scare (5)
  [106-110] Bet ratios pot-relative (5)
  [111-118] Bet counts per player per street / 4 (8)
"""

import numpy as np
from utils import card_rank, card_suit, card_features

FEATURE_DIM = 119
MAX_BET     = 100
_AGGRESSIVE = {3, 4, 5, 6, 7}   # BET_SMALL, BET_LARGE, BET_POT, RAISE_SMALL, RAISE_LARGE


def _hand_strength(hand2: list, community: list) -> list:
    if len(hand2) < 2 or hand2[0] < 0 or hand2[1] < 0:
        return [0.5] * 6
    r0, r1 = card_rank(hand2[0]), card_rank(hand2[1])
    s0, s1 = card_suit(hand2[0]), card_suit(hand2[1])
    out = [max(r0, r1) / 8., float(r0 == r1), float(s0 == s1), 0., 0., 0.]
    comm = [c for c in community if c >= 0]
    if comm:
        sc = [0, 0, 0]; sc[s0] += 1; sc[s1] += 1
        for c in comm: sc[card_suit(c)] += 1
        mx = max(sc)
        out[3] = 1. if mx >= 4 else (0.5 if mx >= 3 else 0.)
        gap = abs(r0 - r1)
        out[4] = 1. if gap <= 1 else (0.5 if gap <= 3 else 0.)
        cr = [card_rank(c) for c in comm]; mb = max(cr)
        hit0, hit1 = r0 in cr, r1 in cr
        if hit0 and hit1:   out[5] = 1.
        elif hit0 or hit1:  out[5] = 0.75 if max(r0, r1) == mb else 0.5
    else:
        out[4] = 1. if abs(r0 - r1) <= 1 else (0.5 if abs(r0 - r1) <= 3 else 0.)
    return out


def _opp_range(opp_disc: list, community: list) -> list:
    valid = [c for c in opp_disc if c >= 0]
    if not valid: return [0.5, 0.5, 0.5, 0., 0., 0.5]
    dr = [card_rank(c) for c in valid]; ds = [card_suit(c) for c in valid]
    avg = sum(dr) / len(dr) / 8.
    out = [avg, max(dr) / 8., float(len(dr) != len(set(dr))),
           max(ds.count(s) for s in range(3)) / 3., 0., 1. - avg]
    comm = [c for c in community if c >= 0]
    if comm:
        bs = [0, 0, 0]
        for c in comm: bs[card_suit(c)] += 1
        dom = bs.index(max(bs))
        out[4] = sum(1 for s in ds if s == dom) / 3.
    return out


def build_features(
    hero_hand: list,
    community: list,
    my_bet: int,
    opp_bet: int,
    street: int,
    is_bb: bool,
    my_disc: list,
    opp_disc: list,
    street_last_ratios,     # [[float,float] * 4]
    street_bet_counts,      # [[int,int] * 4]
    history: list,          # [(player_id, cfr_action_id), ...]
    num_acts_this_street: int,
    hero_hand5: list = None,
) -> np.ndarray:
    f = []
    hp = 1 if is_bb else 0

    # [0-19] Hero hand (20): sorted
    if street == 0 and hero_hand5:
        for c in sorted(x for x in hero_hand5 if x >= 0): f.extend(card_features(c))
        for _ in range(5 - sum(1 for x in hero_hand5 if x >= 0)): f.extend([0.]*4)
    else:
        h2 = sorted(x for x in hero_hand[:2] if x >= 0)
        for c in h2: f.extend(card_features(c))
        for _ in range(2 - len(h2)): f.extend([0.]*4)
        f.extend([0.] * 12)

    # [20-39] Community (20): flop sorted, turn/river temporal
    comm = [x for x in community if x >= 0]
    sc = (sorted(comm[:3]) + comm[3:]) if len(comm) >= 3 else comm
    for i in range(5): f.extend(card_features(sc[i]) if i < len(sc) else [0.]*4)

    # [40-51] My discards (12)
    md = [x for x in my_disc if x >= 0][:3]
    for i in range(3): f.extend(card_features(md[i] if i < len(md) else -1))

    # [52-63] Opp discards (12)
    od = [x for x in opp_disc if x >= 0][:3]
    for i in range(3): f.extend(card_features(od[i] if i < len(od) else -1))

    # [64-67] Street one-hot
    f.extend([float(street == s) for s in range(4)])

    # [68] Position
    f.append(1. if is_bb else 0.)

    # [69-72] Bet info absolute
    pot = my_bet + opp_bet
    to_call = max(opp_bet - my_bet, 0)
    f += [my_bet / MAX_BET, opp_bet / MAX_BET, pot / (2 * MAX_BET), to_call / MAX_BET]

    # [73-78] Hand strength (6)
    vis_n    = [0, 3, 4, 5][min(street, 3)]
    vis_comm = comm[:vis_n]
    h2       = hero_hand[:2] if len(hero_hand) >= 2 else hero_hand
    f.extend(_hand_strength(h2, vis_comm))

    # [79-86] Betting history (8): last bet/pot ratio per street per player
    for s in range(4):
        f += [min(float(street_last_ratios[s][hp]),     4.),
              min(float(street_last_ratios[s][1 - hp]), 4.)]

    # [87-92] Opp range (6)
    f.extend(_opp_range(od, vis_comm))

    # ── Extra 26 dims ──────────────────────────────────────────────────────────

    # [93-94] Initiative
    hagg = vagg = 0.
    for p, a in reversed(history):
        if a in _AGGRESSIVE:
            hagg, vagg = (1., 0.) if p == hp else (0., 1.)
            break
    f += [hagg, vagg]

    # [95-96] Action context
    f += [float(to_call > 0), float(to_call == 0)]

    # [97-100] Line class (4, one-hot)
    curr = history[-num_acts_this_street:] if num_acts_this_street > 0 else []
    bets = sum(1 for _, a in curr if a in _AGGRESSIVE)
    lc   = [0., 0., 0., 0.]
    if   to_call == 0 and bets == 0: lc[0] = 1.
    elif to_call > 0  and bets == 1: lc[1] = 1.
    elif to_call > 0  and bets >= 2: lc[2] = 1.
    elif to_call == 0 and bets >= 1: lc[3] = 1.
    f.extend(lc)

    # [101-105] Board texture (5)
    bt = [0., 0., 0., 0., 0.]
    if comm:
        brs = [card_rank(c) for c in comm]
        bss = [card_suit(c) for c in comm]
        bsc = [bss.count(s) for s in range(3)]
        if len(brs) != len(set(brs)):                      bt[0] = 1.  # paired
        if len(comm) >= 3 and max(bsc) == len(comm):       bt[1] = 1.  # monotone
        if max(bsc) >= 2:                                  bt[2] = 1.  # two-suited
        if len(comm) >= 3 and (max(brs) - min(brs)) <= 4: bt[3] = 1.  # connected
        if len(comm) >= 4 and bss[:len(comm)-1].count(bss[-1]) >= 2: bt[4] = 1.
    f.extend(bt)

    # [106-110] Bet ratios (pot-relative)
    sp    = max(float(pot), 1.)
    max_r = max(MAX_BET - max(my_bet, opp_bet), 0)
    f += [min(to_call / sp, 4.), min(opp_bet / sp, 4.), min(my_bet / sp, 4.),
          min(max_r   / sp, 4.), max_r / 100.]

    # [111-118] Bet counts per player per street / 4
    for s in range(4):
        f += [min(street_bet_counts[s][hp]     / 4., 1.),
              min(street_bet_counts[s][1 - hp] / 4., 1.)]

    assert len(f) == FEATURE_DIM, f"feature dim {len(f)} != {FEATURE_DIM}"
    return np.array(f, dtype=np.float32)

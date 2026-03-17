"""
Preflop strategy:
  - Load tabular CFR chart from data/deep_cfr_preflop_chart.pkl
  - Look up (canonical_hand, 's', history_str) → [fold, pass, raise] probs
  - Apply off-size adjustment when opponent opens larger than 2.5bb
"""

import pickle
import os
from utils import card_rank, card_suit, canonicalize

BIG_BLIND = 2

# CFR action IDs used in history string
_ACT_CHAR = {0: 'f', 1: 'c', 2: 'k', 3: 'b', 4: 'B', 5: 'r', 6: 'R', 7: 'p'}

A_FOLD        = 0
A_CALL        = 1
A_CHECK       = 2
A_BET_SMALL   = 3
A_BET_LARGE   = 4
A_RAISE_SMALL = 5
A_RAISE_LARGE = 6
A_BET_POT     = 7

CHART_PATH = os.path.join(os.path.dirname(__file__), 'data', 'deep_cfr_preflop_chart.pkl')


def load_chart() -> dict:
    """Load preflop strategy chart. Returns {} if file not found."""
    if not os.path.exists(CHART_PATH):
        return {}
    with open(CHART_PATH, 'rb') as f:
        return pickle.load(f)


def lookup_chart(chart: dict, hand5: tuple, history: list, facing_chips: int) -> dict:
    """
    Look up strategy for the current preflop infoset.

    1. Canonicalize hand, build history string.
    2. Always query the 's' bucket (trained at 2.5bb).
    3. Apply off-size adjustment if facing_chips > 5 (>2.5bb).

    Returns {cfr_action_id: probability} for actions in valid_actions.
    Returns None if chart is empty (caller should use uniform strategy).
    """
    if not chart:
        return None

    canon    = canonicalize(hand5)
    hist_str = ''.join(_ACT_CHAR.get(a, '?') for _, a in history)

    # Always look up with 's' size bucket (trained at 2.5bb).
    key = (canon, 's', hist_str)
    s_arr = chart.get(key)

    if s_arr is None:
        # Try legacy format without size bucket
        s_arr = chart.get((canon, hist_str))

    if s_arr is None:
        return None   # infoset not in chart → uniform

    total = s_arr.sum()
    if total > 0:
        s_arr = s_arr / total

    # 3-slot chart: [fold_prob, pass_prob, raise_prob]
    return {
        'fold':  float(s_arr[0]),
        'pass':  float(s_arr[1]),   # call or check
        'raise': float(s_arr[2]),
    }


# ── Off-size adjustment ────────────────────────────────────────────────────────

def _score(hand5) -> float:
    ranks = sorted([card_rank(c) for c in hand5], reverse=True)
    suits = [card_suit(c) for c in hand5]
    sc = (ranks[0] + ranks[1]) / 16.0 * 0.40
    rc = {}
    for r in ranks: rc[r] = rc.get(r, 0) + 1
    paired = [r for r, cnt in rc.items() if cnt >= 2]
    if paired: sc += (max(paired) / 8.0) * 0.30
    sc += min((max(suits.count(s) for s in range(3)) - 1) / 2.0, 1.0) * 0.15
    sc += max(0., 1. - (ranks[0] - ranks[1]) / 5.) * 0.15
    return min(sc, 1.0)


def _role(hand5) -> dict:
    ranks = sorted([card_rank(c) for c in hand5], reverse=True)
    suits = [card_suit(c) for c in hand5]
    rc = {}
    for r in ranks: rc[r] = rc.get(r, 0) + 1
    paired  = [r for r, cnt in rc.items() if cnt >= 2]
    ms      = max(suits.count(s) for s in range(3))
    sc      = _score(hand5)
    is_val  = (max(paired) >= 5 if paired else False) or sc >= 0.75
    blocker = any(r >= 6 for r in ranks[:2])
    gap     = ranks[0] - ranks[1]
    bl3     = 0.
    if blocker and (ms >= 3 or gap <= 2): bl3 = min(0.7 + ranks[0] / 8. * 0.3, 1.)
    elif blocker:                          bl3 = 0.4
    elif ms >= 3 and gap <= 2:             bl3 = 0.5
    return {'score': sc, 'is_value': is_val, 'bluff3': bl3}


def defense_cutoff(facing_chips: int) -> float:
    """MDF-based cutoff: hands below this score fold vs facing_chips open."""
    return max(0., min(1. - 3.0 / (facing_chips + 1), 0.95))


def apply_size_adjustment(slot_probs: dict, hand5: tuple, facing_chips: int) -> dict:
    """
    Adjust 2.5bb-trained slot_probs = {fold, pass, raise} for larger opens.

    Policy:
      score < cutoff + not value  → fold everything
      score < cutoff + is_value   → 3-bet or fold (drop call)
      bluff3 < needed + not value → call only (drop bluff raise)
      is_value                    → keep everything
    """
    if facing_chips <= 5:
        return slot_probs    # standard 2.5bb, no adjustment needed

    cutoff        = defense_cutoff(facing_chips)
    bluff3_needed = 0.4 + (facing_chips - 5) * 0.04
    role          = _role(hand5)
    s             = dict(slot_probs)

    if role['score'] < cutoff and not role['is_value']:
        s['fold'] += s.get('pass', 0.) + s.get('raise', 0.)
        s['pass'] = 0.; s['raise'] = 0.
    elif role['score'] < cutoff and role['is_value']:
        s['fold'] += s.get('pass', 0.)
        s['pass'] = 0.                      # 3-bet or fold
    elif role['bluff3'] < bluff3_needed and not role['is_value']:
        s['fold'] += s.get('raise', 0.)
        s['raise'] = 0.                     # call only

    total = sum(s.values())
    if total <= 0: return {'fold': 1., 'pass': 0., 'raise': 0.}
    return {k: v / total for k, v in s.items()}


def resolve_action(slot_probs: dict, valid_cfr_actions: list) -> dict:
    """
    Map {fold/pass/raise} slot probs to {cfr_action_id: prob} for valid actions.

    Slot mapping:
      'fold'  → A_FOLD
      'pass'  → A_CALL or A_CHECK (whichever is valid — never both)
      'raise' → A_RAISE_SMALL or A_BET_SMALL (whichever is valid)
    """
    out = {}
    for a in valid_cfr_actions:
        if   a == A_FOLD:                      out[a] = slot_probs.get('fold', 0.)
        elif a in (A_CALL, A_CHECK):           out[a] = slot_probs.get('pass', 0.)
        elif a in (A_RAISE_SMALL, A_BET_SMALL): out[a] = slot_probs.get('raise', 0.)
    total = sum(out.values())
    if total <= 0:
        n = len(valid_cfr_actions)
        return {a: 1. / n for a in valid_cfr_actions}
    return {k: v / total for k, v in out.items() if v > 0}

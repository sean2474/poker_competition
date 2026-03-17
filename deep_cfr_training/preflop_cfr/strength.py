"""
Preflop hand strength score for off-size open adjustment.

score(canonical_hand5) → float in [0, 1]
  Higher = stronger hand (more likely to continue vs large opens).

Factors (fast, deterministic — no MC needed):
  1. Top-2 rank quality     (40%)  — high cards after discard
  2. Pair quality           (30%)  — best pair rank in hand
  3. Flush potential        (15%)  — 3+ suited cards
  4. Connectivity           (15%)  — gap between top-2 ranks

Used at inference time to apply size-based cutoffs:
  defense_cutoff(facing_chips) = 1 - 3/(facing_chips+1)
  hands with score < cutoff → fold/remove from range
"""

from .canonical import card_rank, card_suit


def preflop_score(canonical_hand5) -> float:
    """Return 0-1 score for a canonical 5-card hand."""
    ranks = sorted([card_rank(c) for c in canonical_hand5], reverse=True)
    suits  = [card_suit(c) for c in canonical_hand5]

    score = 0.0

    # 1. Top-2 rank quality (40%): best pair of cards we can keep
    score += (ranks[0] + ranks[1]) / 16.0 * 0.40   # max = 0.40

    # 2. Pair quality (30%): best pair rank (trips count too)아
    rank_counts: dict = {}
    for r in ranks:
        rank_counts[r] = rank_counts.get(r, 0) + 1
    paired_ranks = [r for r, cnt in rank_counts.items() if cnt >= 2]
    if paired_ranks:
        best_pair_rank = max(paired_ranks)
        score += (best_pair_rank / 8.0) * 0.30      # max = 0.30

    # 3. Flush potential (15%): 3+ suited → can keep flush pair
    suit_counts = [suits.count(s) for s in range(3)]
    max_suited   = max(suit_counts)
    score += min((max_suited - 1) / 2.0, 1.0) * 0.15   # max = 0.15

    # 4. Connectivity (15%): small gap between top-2 ranks
    gap = ranks[0] - ranks[1]
    score += max(0.0, 1.0 - gap / 5.0) * 0.15           # max = 0.15

    return min(score, 1.0)


def defense_cutoff(facing_chips: int, big_blind: int = 2) -> float:
    """
    Minimum score a hand needs to continue vs a given open size.

    MDF = dead_money / (dead_money + to_call) = 3 / (facing + 1)
    cutoff = 1 - MDF  (hands below this score fold)

    Examples (BIG_BLIND=2, dead money = SB+BB = 3):
      2.5bb (5 chips) → MDF=50% → cutoff=0.50
      3bb   (6 chips) → MDF=43% → cutoff=0.57
      5bb  (10 chips) → MDF=27% → cutoff=0.73
      10bb (20 chips) → MDF=14% → cutoff=0.86
    """
    dead_money = int(big_blind * 1.5)          # SB + BB = 3
    mdf = dead_money / (facing_chips + 1)
    return max(0.0, min(1.0 - mdf, 0.95))


def hand_role(canonical_hand5) -> dict:
    """
    Return role classification for a canonical 5-card hand.

    Fields:
      score            float [0,1]  — overall preflop strength
      is_value_continue bool        — strong pair or high-rank: always continue
      bluff3_suitability float [0,1] — suited/connected/blocker hands good for bluff 3-bet
      call_quality     float [0,1]  — medium playability: good to flat-call

    Usage: apply_size_adjustment uses all fields for nuanced cutoffs.
    """
    ranks = sorted([card_rank(c) for c in canonical_hand5], reverse=True)
    suits  = [card_suit(c) for c in canonical_hand5]

    # Count pairs / trips
    rank_counts: dict = {}
    for r in ranks:
        rank_counts[r] = rank_counts.get(r, 0) + 1
    paired_ranks = [r for r, cnt in rank_counts.items() if cnt >= 2]

    # Suit info
    suit_counts = [suits.count(s) for s in range(3)]
    max_suited  = max(suit_counts)

    # Base score (same as preflop_score)
    sc = 0.0
    sc += (ranks[0] + ranks[1]) / 16.0 * 0.40
    if paired_ranks:
        sc += (max(paired_ranks) / 8.0) * 0.30
    sc += min((max_suited - 1) / 2.0, 1.0) * 0.15
    gap = ranks[0] - ranks[1]
    sc += max(0.0, 1.0 - gap / 5.0) * 0.15
    score = min(sc, 1.0)

    # Value continue: high pairs (rank >= 5 = 7,8,9,A) or top pair + suited
    top_pair_rank = max(paired_ranks) if paired_ranks else -1
    is_value = (top_pair_rank >= 5) or (score >= 0.75)

    # Bluff 3-bet suitability: Ax/Kx blocker + suited OR connected
    # blocker: at least one top-3 rank card (6,7,8 = J,Q,K,A in 9-rank deck)
    has_blocker = any(r >= 6 for r in ranks[:2])
    is_suited_good = (max_suited >= 3)  # flush draw potential
    is_connected_good = (gap <= 2)
    bluff3 = 0.0
    if has_blocker and (is_suited_good or is_connected_good):
        bluff3 = 0.7 + (ranks[0] / 8.0) * 0.3
    elif has_blocker:
        bluff3 = 0.4
    elif is_suited_good and is_connected_good:
        bluff3 = 0.5
    bluff3 = min(bluff3, 1.0)

    # Call quality: medium strength, good playability (not top-tier, not trash)
    call_q = 0.0
    if 0.35 < score < 0.70:
        call_q = (score - 0.35) / 0.35 * (1 - abs(score - 0.525) / 0.175)
        call_q = max(0.0, min(call_q, 1.0))

    return {
        'score':              score,
        'is_value_continue':  is_value,
        'bluff3_suitability': bluff3,
        'call_quality':       call_q,
    }


def apply_size_adjustment(strategy: dict, canonical_hand5: tuple,
                          facing_chips: int, big_blind: int = 2,
                          a_fold=0, a_call=1, a_raise=5) -> dict:
    """
    Adjust a 2.5bb baseline strategy for an off-size open using hand role.

    Policy (from tightest to loosest requirement):
      score < cutoff AND not value_continue → fold everything
      score < cutoff AND value_continue     → keep raise, drop call
      cutoff ≤ score AND bluff3 low         → call only (drop bluff raise)
      is_value_continue                     → keep raise regardless
    """
    cutoff = defense_cutoff(facing_chips, big_blind)
    mdf    = 1.0 - cutoff
    # value threshold: top ~1/3 of defending range
    value_thresh = 1.0 - mdf / 3.0

    role = hand_role(canonical_hand5)
    score = role['score']
    is_value = role['is_value_continue']
    bluff3   = role['bluff3_suitability']

    # Bluff 3-bet cutoff scales with open size: larger opens → higher bluff3 needed
    bluff3_needed = 0.4 + (facing_chips - 5) * 0.04   # 5chips(2.5bb)=0.40, 10chips=0.60

    s = dict(strategy)

    if score < cutoff and not is_value:
        # Fold everything — too weak to continue
        s[a_fold] = s.get(a_fold, 0) + s.get(a_call, 0) + s.get(a_raise, 0)
        s[a_call]  = 0.0
        s[a_raise] = 0.0
    elif score < cutoff and is_value:
        # Strong hand but expensive call → 3-bet or fold (no call)
        s[a_fold] = s.get(a_fold, 0) + s.get(a_call, 0)
        s[a_call] = 0.0
    elif bluff3 < bluff3_needed and not is_value:
        # Not good for bluff 3-bet → call only
        s[a_fold]  = s.get(a_fold, 0) + s.get(a_raise, 0)
        s[a_raise] = 0.0
    # else: value range → keep everything

    # renormalize
    total = sum(s.values())
    if total > 0:
        return {k: v / total for k, v in s.items()}
    return {a_fold: 1.0}

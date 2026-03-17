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

    # 2. Pair quality (30%): best pair rank (trips count too)
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


def apply_size_adjustment(strategy: dict, hand_score: float,
                          facing_chips: int, big_blind: int = 2,
                          a_fold=0, a_call=1, a_raise=5) -> dict:
    """
    Adjust a 2.5bb baseline strategy for an off-size open.

    Policy:
      score < cutoff          → fold (remove call AND bluff raise)
      cutoff ≤ score < value  → call only (remove bluff raise)
      score ≥ value_thresh    → keep raise (value 3-bet preserved)

    value_thresh = 1 - MDF/3  (top third of continuing range → value)
    """
    cutoff = defense_cutoff(facing_chips, big_blind)
    mdf    = 1.0 - cutoff
    value_thresh = 1.0 - mdf / 3.0    # top ~1/3 of defenders → value 3-bet

    s = dict(strategy)

    if hand_score < cutoff:
        # fold everything
        s[a_fold] = s.get(a_fold, 0) + s.get(a_call, 0) + s.get(a_raise, 0)
        s[a_call]  = 0.0
        s[a_raise] = 0.0
    elif hand_score < value_thresh:
        # call only — remove bluff raise
        s[a_fold]  = s.get(a_fold, 0) + s.get(a_raise, 0)
        s[a_raise] = 0.0
    # else: keep both call and raise (value range)

    # renormalize
    total = sum(s.values())
    if total > 0:
        return {k: v / total for k, v in s.items()}
    return {a_fold: 1.0}

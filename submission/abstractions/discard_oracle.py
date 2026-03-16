"""
Discard oracle: exact candidate search + approximate evaluation.

NOT part of CFR — runs as a local solver at decision time.

Stage A: Fast heuristic filter over all 10 keep-pair candidates.
Stage B: Detailed MC evaluation of top 2-4 candidates.

Scoring considers:
  - Showdown equity (MC rollout)
  - Draw robustness (flush/straight potential)
  - Nut potential
  - Playability (suited, connected)
  - Blocker value (what we discard denies opponent)
"""

import random
from abstractions.card_utils import (
    card_rank, card_suit, rank_counts, suit_counts,
    has_ace, has_straight_potential, straight_draw_outs,
    KEEP_PAIRS, NUM_RANKS, ACE_RANK_IDX, DECK_SIZE, ALL_CARDS,
    get_evaluator, int_to_treys, canonicalize_suits,
)


# ═══════════════════════════════════════════════════════════════════
# Stage A: Fast heuristic score (no MC, pure card features)
# ═══════════════════════════════════════════════════════════════════

def _fast_score(keep: list, board_3: list) -> float:
    """
    Quick heuristic score for a keep-pair candidate.
    Higher = better. Range roughly [0, 10].
    """
    r0, r1 = card_rank(keep[0]), card_rank(keep[1])
    s0, s1 = card_suit(keep[0]), card_suit(keep[1])

    score = 0.0

    # ─── Made strength with board ───
    board_ranks = [card_rank(c) for c in board_3]
    board_suits = [card_suit(c) for c in board_3]

    # Pair with board (top pair > middle > bottom)
    for kr in [r0, r1]:
        if kr in board_ranks:
            # Higher pair = better
            rank_position = sorted(board_ranks).index(kr) if kr in board_ranks else -1
            if kr == max(board_ranks):
                score += 3.0  # top pair
            elif kr == min(board_ranks):
                score += 1.0  # bottom pair
            else:
                score += 2.0  # middle pair

    # Pocket pair
    if r0 == r1:
        score += 2.0
        # Overpair
        if r0 > max(board_ranks):
            score += 2.0
        # Set possibility (trips with board)
        if r0 in board_ranks:
            score += 4.0  # set!

    # ─── High card value ───
    score += (r0 + r1) / (2 * (NUM_RANKS - 1)) * 1.0  # up to 1.0 for high cards
    if r0 == ACE_RANK_IDX or r1 == ACE_RANK_IDX:
        score += 0.5

    # ─── Suited (flush potential) ───
    if s0 == s1:
        score += 1.5
        # Board suits matching
        matching_board_suit = sum(1 for bs in board_suits if bs == s0)
        if matching_board_suit >= 2:
            score += 3.0  # flush draw with 4 of suit
        elif matching_board_suit >= 1:
            score += 1.0  # backdoor flush

    # ─── Connectivity (straight potential) ───
    gap = abs(r0 - r1)
    combined_ranks = set(board_ranks) | {r0, r1}

    if gap == 1 or (gap == 0):
        score += 1.0  # connected / pair
    elif gap == 2:
        score += 0.5  # one-gapper

    # Straight draw with board
    s_outs = straight_draw_outs(combined_ranks)
    score += s_outs * 0.5

    # ─── Blocker value ───
    # Cards we keep that block opponent from having strong hands
    # If board is paired and we hold a blocker to trips
    rc = rank_counts(board_3)
    for rank, cnt in rc.items():
        if cnt >= 2:
            if r0 == rank or r1 == rank:
                score += 1.5  # we block their set

    return score


# ═══════════════════════════════════════════════════════════════════
# Stage B: Detailed Monte Carlo evaluation
# ═══════════════════════════════════════════════════════════════════

def _mc_equity(keep: list, board_3: list, dead: list,
               num_sims: int = 150) -> float:
    """
    Monte Carlo equity for a keep-pair against random opponent.
    Rolls out turn + river + opponent 2 cards.
    Returns win rate [0, 1].
    """
    ev = get_evaluator()
    used = set(keep) | set(board_3) | set(dead)
    remaining = [c for c in ALL_CARDS if c not in used]

    # Need: 2 more board cards + 2 opponent cards = 4
    if len(remaining) < 4:
        return 0.5

    my_h = [int_to_treys(c) for c in keep]
    wins = 0
    ties = 0
    total = 0

    for _ in range(num_sims):
        sample = random.sample(remaining, 4)
        full_board = board_3 + sample[:2]
        opp = sample[2:4]

        b = [int_to_treys(c) for c in full_board]
        opp_h = [int_to_treys(c) for c in opp]

        my_rank = ev.evaluate(my_h, b)
        opp_rank = ev.evaluate(opp_h, b)
        if my_rank < opp_rank:
            wins += 1
        elif my_rank == opp_rank:
            ties += 1
        total += 1

    if total == 0:
        return 0.5
    return (wins + 0.5 * ties) / total


def _composite_score(keep: list, board_3: list, dead: list,
                     fast: float, mc_equity: float) -> float:
    """
    Combine fast heuristic and MC equity into final score.
    Weights can be tuned.
    """
    # MC equity is the primary signal (range 0-1, scale to ~10)
    # Fast score captures structural features MC might miss
    w_mc = 0.7
    w_fast = 0.3
    return w_mc * (mc_equity * 10) + w_fast * fast


# ═══════════════════════════════════════════════════════════════════
# Main discard oracle
# ═══════════════════════════════════════════════════════════════════

def choose_discard(hand_5: list, board_3: list,
                   opp_discards: list = None,
                   top_k: int = 3,
                   mc_sims: int = 150) -> tuple:
    """
    Choose which 2 cards to keep from 5 hole cards.

    Args:
        hand_5: list of 5 card ints (our hole cards)
        board_3: list of 3 card ints (flop community cards)
        opp_discards: list of 3 card ints (opponent's discards, may be [-1,-1,-1])
        top_k: number of candidates to evaluate in Stage B
        mc_sims: Monte Carlo simulations per candidate in Stage B

    Returns:
        (keep_idx_0, keep_idx_1): indices into hand_5 of the 2 cards to keep
    """
    dead = []
    if opp_discards:
        dead = [c for c in opp_discards if c >= 0]

    # Stage A: fast score all 10 candidates
    candidates = []
    for i, j in KEEP_PAIRS:
        keep = [hand_5[i], hand_5[j]]
        discarded = [hand_5[k] for k in range(5) if k != i and k != j]
        fast = _fast_score(keep, board_3)
        candidates.append((fast, i, j, keep, discarded))

    # Sort by fast score descending, take top_k
    candidates.sort(key=lambda x: -x[0])
    top_candidates = candidates[:top_k]

    # Stage B: MC evaluation of top candidates
    best_score = -1.0
    best_keep = (candidates[0][1], candidates[0][2])

    for fast, i, j, keep, discarded in top_candidates:
        all_dead = dead + discarded
        eq = _mc_equity(keep, board_3, all_dead, num_sims=mc_sims)
        composite = _composite_score(keep, board_3, all_dead, fast, eq)
        if composite > best_score:
            best_score = composite
            best_keep = (i, j)

    return best_keep


def estimate_opp_keep_weights(opp_discards: list, board_3: list,
                               remaining_cards: list) -> dict:
    """
    Estimate likelihood of each possible opponent keep-pair, given their 3 discards.

    Logic: opponent's original 5 cards = keep(2) + discard(3).
    For each possible 2-card combo from remaining_cards, check if that combo
    would be the best keep from the hypothetical 5-card hand.

    Returns: dict mapping (c1, c2) -> weight (higher = more likely kept)
    """
    from itertools import combinations

    opp_disc = [c for c in opp_discards if c >= 0]
    if len(opp_disc) != 3:
        return {}

    weights = {}
    for c1, c2 in combinations(remaining_cards, 2):
        # Hypothetical original 5 cards
        original_5 = [c1, c2] + opp_disc
        keep_pair = [c1, c2]
        keep_score = _fast_score(keep_pair, board_3)

        # Check all 10 possible keep pairs from this 5-card hand
        is_best = True
        for i in range(5):
            for j in range(i + 1, 5):
                alt_keep = [original_5[i], original_5[j]]
                if alt_keep == keep_pair:
                    continue
                alt_score = _fast_score(alt_keep, board_3)
                if alt_score > keep_score + 0.5:  # margin to account for noise
                    is_best = False
                    break
            if not is_best:
                break

        # Weight: 1.0 if this was plausibly the best keep, 0.05 if not
        weights[(c1, c2)] = 1.0 if is_best else 0.05

    return weights


def choose_discard_with_scores(hand_5: list, board_3: list,
                                opp_discards: list = None,
                                mc_sims: int = 100) -> list:
    """
    Debug version: returns all 10 candidates with scores.
    Returns list of (i, j, fast_score, mc_equity, composite) sorted by composite.
    """
    dead = [c for c in (opp_discards or []) if c >= 0]

    results = []
    for i, j in KEEP_PAIRS:
        keep = [hand_5[i], hand_5[j]]
        discarded = [hand_5[k] for k in range(5) if k != i and k != j]
        all_dead = dead + discarded
        fast = _fast_score(keep, board_3)
        eq = _mc_equity(keep, board_3, all_dead, num_sims=mc_sims)
        comp = _composite_score(keep, board_3, all_dead, fast, eq)
        results.append((i, j, fast, eq, comp))

    results.sort(key=lambda x: -x[4])
    return results

#pragma once
#include <algorithm>
#include "constants.h"

/*
 * hand_eval.h — 2-card hand categorization against a board.
 *
 * 17 categories (strongest → weakest):
 *   0  straight_flush
 *   1  full_house
 *   2  flush
 *   3  straight
 *   4  top_set      (trips, rank == board_rank[0])
 *   5  middle_set
 *   6  bottom_set   (trips, rank == board_rank[-1])
 *   7  two_pair
 *   8  bottom_two   (both pairs below top board rank)
 *   9  overpair     (pocket pair above all board ranks)
 *  10  top_pair
 *  11  middle_pair
 *  12  bottom_pair
 *  13  sf_draw      (4 suited+connected)
 *  14  flush_draw   (4+ same suit)
 *  15  straight_draw(4 consecutive ranks)
 *  16  high_card
 */

constexpr int N_CATS = 17;

enum HandCat {
    CAT_SF            = 0,
    CAT_FULL_HOUSE    = 1,
    CAT_FLUSH         = 2,
    CAT_STRAIGHT      = 3,
    CAT_TOP_SET       = 4,
    CAT_MIDDLE_SET    = 5,
    CAT_BOTTOM_SET    = 6,
    CAT_TWO_PAIR      = 7,
    CAT_BOTTOM_TWO    = 8,
    CAT_OVERPAIR      = 9,
    CAT_TOP_PAIR      = 10,
    CAT_MIDDLE_PAIR   = 11,
    CAT_BOTTOM_PAIR   = 12,
    CAT_SF_DRAW       = 13,
    CAT_FLUSH_DRAW    = 14,
    CAT_STRAIGHT_DRAW = 15,
    CAT_HIGH_CARD     = 16,
};

/*
 * compute_blocker_flags — 4-dim float array for (c0, c1) vs current board.
 *
 *   out[0]  blocks_top_pair   : holds a card matching board's top rank
 *   out[1]  blocks_2pair      : holds BOTH top-two board ranks
 *   out[2]  blocks_flush      : holds a card of the nut (most common) suit
 *   out[3]  blocks_straight   : holds a rank that completes a 4-straight on board
 */
inline void compute_blocker_flags(int c0, int c1, const int* community, int n_comm, float* out) {
    out[0]=out[1]=out[2]=out[3]=0.f;
    if (c0 < 0 || n_comm == 0) return;
    int hr[2] = {card_rank(c0), card_rank(c1)};
    int hs[2] = {card_suit(c0), card_suit(c1)};
    int bsc[NUM_SUITS] = {}, b_set[NUM_RANKS] = {};
    int top_r = -1, sec_r = -1;
    for (int i = 0; i < n_comm; i++) {
        if (community[i] < 0) continue;
        int r = card_rank(community[i]), s = card_suit(community[i]);
        bsc[s]++; b_set[r] = 1;
        if (r > top_r) { sec_r = top_r; top_r = r; }
        else if (r > sec_r) sec_r = r;
    }
    int dom = (int)(std::max_element(bsc, bsc + NUM_SUITS) - bsc);
    out[0] = (hr[0]==top_r || hr[1]==top_r) ? 1.f : 0.f;
    out[1] = (top_r>=0 && sec_r>=0 &&
              (hr[0]==top_r || hr[1]==top_r) &&
              (hr[0]==sec_r || hr[1]==sec_r)) ? 1.f : 0.f;
    out[2] = (hs[0]==dom || hs[1]==dom) ? 1.f : 0.f;
    for (int hi = 0; hi < 2 && !out[3]; hi++) {
        int r = hr[hi];
        for (int ws = std::max(0,r-4); ws <= std::min(4,r); ws++) {
            int cnt2 = 0;
            for (int k = ws; k < ws+5; k++) if (k < NUM_RANKS) cnt2 += b_set[k];
            if (cnt2 >= 3 && r >= ws && r < ws+5) { out[3] = 1.f; break; }
        }
    }
}

/*
 * classify_hand — classify (c0, c1) + up to 5 board cards into one HandCat.
 *
 * Only the first min(n_board, 3) board cards are used for category
 * determination (flop-based categorization). Turn/river cards expand
 * the made-hand detection to 5 cards total.
 */
static inline int classify_hand(int c0, int c1, const int* board, int n_board) {
    int use  = (n_board > 5) ? 5 : n_board;
    int n    = 2 + use;

    int cards[7] = {c0, c1, -1, -1, -1, -1, -1};
    for (int i = 0; i < use; i++) cards[2 + i] = board[i];

    int ranks[7], suits[7];
    for (int i = 0; i < n; i++) {
        ranks[i] = card_rank(cards[i]);
        suits[i] = card_suit(cards[i]);
    }

    /* Board ranks, sorted descending */
    int brank[5]; int nb = 0;
    for (int i = 2; i < n; i++) brank[nb++] = ranks[i];
    for (int i = 1; i < nb; i++)
        for (int j = i; j > 0 && brank[j] > brank[j-1]; j--)
            std::swap(brank[j], brank[j-1]);

    /* Rank / suit counts */
    int rcnt[NUM_RANKS] = {}, scnt[NUM_SUITS] = {};
    for (int i = 0; i < n; i++) { rcnt[ranks[i]]++; scnt[suits[i]]++; }
    int max_suit = *std::max_element(scnt, scnt + NUM_SUITS);

    /* ── Made hands (5-card combinations) ──────────────────────────────── */
    if (n >= 5) {
        int sr[7]; std::copy(ranks, ranks + n, sr);
        std::sort(sr, sr + n);
        /* Check any 5-card straight window */
        bool is_str = false;
        for (int lo = 0; lo <= NUM_RANKS - 5; lo++) {
            int hits = 0;
            for (int k = 0; k < n; k++) if (sr[k] >= lo && sr[k] < lo + 5) hits++;
            if (hits >= 5) { is_str = true; break; }
        }
        if (max_suit >= 5 && is_str) return CAT_SF;
        if (max_suit >= 5)           return CAT_FLUSH;
        if (is_str)                  return CAT_STRAIGHT;
    }

    /* Count trips and pairs across all n cards */
    int n_trips = 0, trip_rank = -1, n_pairs = 0, pair_ranks[5], np = 0;
    for (int r = 0; r < NUM_RANKS; r++) {
        if (rcnt[r] >= 3) { n_trips++; trip_rank = r; }
        if (rcnt[r] == 2) { pair_ranks[np++] = r; n_pairs++; }
    }

    if (n_trips >= 1 && n_pairs >= 1) return CAT_FULL_HOUSE;

    if (n_trips >= 1) {
        if (nb == 0)                      return CAT_TOP_SET;
        if (trip_rank == brank[0])        return CAT_TOP_SET;
        if (trip_rank == brank[nb - 1])   return CAT_BOTTOM_SET;
        return CAT_MIDDLE_SET;
    }

    if (n_pairs >= 2) {
        if (nb > 0 && np >= 2 &&
            pair_ranks[np-1] < brank[0] && pair_ranks[np-2] < brank[0])
            return CAT_BOTTOM_TWO;
        return CAT_TWO_PAIR;
    }

    int r0 = card_rank(c0), r1 = card_rank(c1);

    if (n_pairs == 1) {
        int pr = pair_ranks[0];
        if (r0 == r1) {
            bool over = true;
            for (int i = 0; i < nb; i++) if (brank[i] >= r0) { over = false; break; }
            if (over) return CAT_OVERPAIR;
        }
        if (nb == 0)                    return CAT_MIDDLE_PAIR;
        if (pr == brank[0])             return CAT_TOP_PAIR;
        if (pr == brank[nb - 1])        return CAT_BOTTOM_PAIR;
        return CAT_MIDDLE_PAIR;
    }

    /* ── Draw detection ─────────────────────────────────────────────────── */
    if (max_suit >= 4 && n >= 4) {
        for (int s = 0; s < NUM_SUITS; s++) {
            int sr2[7]; int sn = 0;
            for (int i = 0; i < n; i++) if (suits[i] == s) sr2[sn++] = ranks[i];
            if (sn < 4) continue;
            std::sort(sr2, sr2 + sn);
            for (int lo = 0; lo <= NUM_RANKS - 4; lo++) {
                int hits = 0;
                for (int k = 0; k < 4; k++)
                    for (int j = 0; j < sn; j++) if (sr2[j] == lo + k) { hits++; break; }
                if (hits >= 4) return CAT_SF_DRAW;
            }
        }
    }

    if (max_suit >= 4) return CAT_FLUSH_DRAW;

    if (n >= 4) {
        int present[NUM_RANKS] = {};
        for (int i = 0; i < n; i++) present[ranks[i]] = 1;
        for (int lo = 0; lo <= NUM_RANKS - 4; lo++) {
            int hits = 0;
            for (int k = 0; k < 4; k++) if (lo + k < NUM_RANKS && present[lo + k]) hits++;
            if (hits >= 4) return CAT_STRAIGHT_DRAW;
        }
    }

    return CAT_HIGH_CARD;
}

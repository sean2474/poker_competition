/*
 * prob_agent.cpp — C++ MC-equity heuristic agent for the 27-card poker variant.
 *
 * 27-card deck: 9 ranks (2-9,A) × 3 suits (d,h,s)
 * Card encoding: suit * 9 + rank   (card ∈ [0, 26])
 * Action types:  FOLD=0, RAISE=1, CHECK=2, CALL=3, DISCARD=4
 *
 * Compile as shared library:
 *   clang++ -O3 -std=c++17 -shared -fPIC -o prob_agent.so prob_agent.cpp
 *
 * Exported C API (callable via Python ctypes):
 *   mc_equity()      — win-rate estimate via MC simulation
 *   best_discard()   — best (i,j) indices of the 2 cards to keep from 5
 *   betting_action() — returns (action_type, raise_amount)
 */

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

// ── Constants ─────────────────────────────────────────────────────────────────

static constexpr int DECK_SIZE  = 27;
static constexpr int NUM_RANKS  = 9;
static constexpr int NUM_SUITS  = 3;

inline int card_rank(int c) { return c % NUM_RANKS; }
inline int card_suit(int c) { return c / NUM_RANKS; }

// ── 5-card hand evaluator (higher score = better) ─────────────────────────────
//
// Score layout: bits 23-20 = category (0-8), bits 19-0 = tiebreak ranks
//   8=straight-flush  7=quads  6=full-house  5=flush
//   4=straight        3=trips  2=two-pair    1=one-pair  0=high-card

static uint32_t eval5(const int* h) {
    int r[5], s[5];
    int rcnt[NUM_RANKS] = {};
    int scnt[NUM_SUITS] = {};
    for (int i = 0; i < 5; i++) {
        r[i] = card_rank(h[i]);
        s[i] = card_suit(h[i]);
        rcnt[r[i]]++;
        scnt[s[i]]++;
    }

    int sr[5];
    std::copy(r, r + 5, sr);
    std::sort(sr, sr + 5, std::greater<int>());

    bool is_flush = (*std::max_element(scnt, scnt + NUM_SUITS) == 5);

    // Unique sorted ranks (ascending) for straight check
    int ur[5]; int nu = 0;
    {
        int prev = -1;
        for (int i = 4; i >= 0; i--) {
            if (sr[i] != prev) { ur[nu++] = sr[i]; prev = sr[i]; }
        }
    }
    bool is_straight = (nu == 5 && ur[4] - ur[0] == 4);

    // Tiebreak from top 5 sorted ranks
    uint32_t tb = ((uint32_t)sr[0] << 16) | ((uint32_t)sr[1] << 12) |
                  ((uint32_t)sr[2] << 8)  | ((uint32_t)sr[3] << 4)  | sr[4];

    int four = -1, three = -1, pairs[2] = {-1,-1}; int np = 0;
    for (int rk = NUM_RANKS - 1; rk >= 0; rk--) {
        if      (rcnt[rk] == 4) four = rk;
        else if (rcnt[rk] == 3) three = rk;
        else if (rcnt[rk] == 2 && np < 2) pairs[np++] = rk;
    }

    if (is_flush && is_straight) return (8u << 20) | tb;

    if (four >= 0) {
        int kicker = -1;
        for (int i = 0; i < 5; i++) if (rcnt[r[i]] != 4) kicker = r[i];
        return (7u << 20) | ((uint32_t)four << 4) | (uint32_t)kicker;
    }

    if (three >= 0 && np >= 1)
        return (6u << 20) | ((uint32_t)three << 4) | (uint32_t)pairs[0];

    if (is_flush) return (5u << 20) | tb;
    if (is_straight) return (4u << 20) | (uint32_t)sr[0];

    if (three >= 0) {
        int k[2] = {-1,-1}; int nk = 0;
        for (int i = 0; i < 5; i++) if (rcnt[r[i]] != 3 && nk < 2) k[nk++] = r[i];
        std::sort(k, k + 2, std::greater<int>());
        return (3u << 20) | ((uint32_t)three << 8) | ((uint32_t)k[0] << 4) | (uint32_t)k[1];
    }

    if (np == 2) {
        int p0 = std::max(pairs[0], pairs[1]), p1 = std::min(pairs[0], pairs[1]);
        int kicker = -1;
        for (int i = 0; i < 5; i++) if (rcnt[r[i]] == 1) kicker = r[i];
        return (2u << 20) | ((uint32_t)p0 << 8) | ((uint32_t)p1 << 4) | (uint32_t)kicker;
    }

    if (np == 1) {
        int k[3]; int nk = 0;
        for (int i = 0; i < 5; i++) if (rcnt[r[i]] == 1) k[nk++] = r[i];
        std::sort(k, k + 3, std::greater<int>());
        return (1u << 20) | ((uint32_t)pairs[0] << 12) |
               ((uint32_t)k[0] << 8) | ((uint32_t)k[1] << 4) | (uint32_t)k[2];
    }

    return (0u << 20) | tb;  // high card
}

// Best score from all C(7,5)=21 five-card subsets
static uint32_t eval7(const int* h) {
    uint32_t best = 0;
    for (int ex1 = 0; ex1 < 7; ex1++)
    for (int ex2 = ex1 + 1; ex2 < 7; ex2++) {
        int h5[5]; int k = 0;
        for (int i = 0; i < 7; i++)
            if (i != ex1 && i != ex2) h5[k++] = h[i];
        uint32_t s = eval5(h5);
        if (s > best) best = s;
    }
    return best;
}

// Evaluate my_hand(2) + full_board(n_board cards)
// If n_board == 3 → 5 cards total → eval5; if n_board == 5 → 7 cards → eval7
static uint32_t eval_hand(const int* my2, const int* board, int n_board) {
    int all[7];
    all[0] = my2[0]; all[1] = my2[1];
    for (int i = 0; i < n_board; i++) all[2 + i] = board[i];
    if (n_board == 3) return eval5(all);        // 5 cards
    if (n_board >= 5) return eval7(all);        // 7 cards
    // 6 cards: find best 5 from 6
    uint32_t best = 0;
    for (int ex = 0; ex < 6; ex++) {
        int h5[5]; int k = 0;
        for (int i = 0; i < 6; i++) if (i != ex) h5[k++] = all[i];
        uint32_t s = eval5(h5);
        if (s > best) best = s;
    }
    return best;
}

// ── MC equity ─────────────────────────────────────────────────────────────────

extern "C" float mc_equity(
    const int* my_cards,     int n_my,           // 2 hole cards
    const int* community,    int n_community,    // 0-3 board cards
    const int* opp_discards, int n_opp_disc,     // 0-3 known opp discards (-1 = unknown)
    int num_sims,
    unsigned int seed)
{
    std::mt19937 rng(seed);

    bool shown[DECK_SIZE] = {};
    for (int i = 0; i < n_my;       i++) if (my_cards[i] >= 0)     shown[my_cards[i]]     = true;
    for (int i = 0; i < n_community; i++) if (community[i] >= 0)    shown[community[i]]    = true;
    for (int i = 0; i < n_opp_disc;  i++) if (opp_discards[i] >= 0) shown[opp_discards[i]] = true;

    std::vector<int> avail;
    avail.reserve(DECK_SIZE);
    for (int c = 0; c < DECK_SIZE; c++) if (!shown[c]) avail.push_back(c);

    int opp_needed   = 2;
    int board_needed = 5 - n_community;          // fill board to 5
    int sample_size  = opp_needed + board_needed;

    if ((int)avail.size() < sample_size) return 0.5f;

    int wins = 0, ties = 0, valid = 0;

    for (int sim = 0; sim < num_sims; sim++) {
        // Partial Fisher-Yates to sample sample_size elements
        for (int i = 0; i < sample_size; i++) {
            std::uniform_int_distribution<int> d(i, (int)avail.size() - 1);
            std::swap(avail[i], avail[d(rng)]);
        }

        const int* opp_pair = avail.data();
        int full_board[5];
        std::memcpy(full_board, community, n_community * sizeof(int));
        for (int i = 0; i < board_needed; i++)
            full_board[n_community + i] = avail[opp_needed + i];

        uint32_t my_score  = eval_hand(my_cards, full_board, 5);
        uint32_t opp_score = eval_hand(opp_pair,  full_board, 5);

        if      (my_score > opp_score) wins++;
        else if (my_score == opp_score) ties++;
        valid++;

        // Undo swap (restore avail for next iteration)
        for (int i = sample_size - 1; i >= 0; i--) {
            std::uniform_int_distribution<int> d(i, (int)avail.size() - 1);
            // No-op: we just re-sample each iteration (partial shuffle is fine)
        }
        // Actually just re-shuffle each iter: no need to undo
    }

    return valid > 0 ? (float)(wins + 0.5f * ties) / valid : 0.5f;
}

// ── Discard decision ──────────────────────────────────────────────────────────

extern "C" void best_discard(
    const int* hand5,        // 5 hole cards
    const int* community,    int n_community,
    const int* opp_discards, int n_opp_disc,
    int num_sims, unsigned int seed,
    int* out_keep_i,          // output: index 0-4 of first kept card
    int* out_keep_j)          // output: index 0-4 of second kept card
{
    float best_eq = -1.f;
    *out_keep_i = 0; *out_keep_j = 1;

    unsigned int s = seed;
    for (int i = 0; i < 5; i++) {
        for (int j = i + 1; j < 5; j++) {
            int keep[2] = {hand5[i], hand5[j]};
            float eq = mc_equity(keep, 2, community, n_community,
                                 opp_discards, n_opp_disc, num_sims, s++);
            if (eq > best_eq) {
                best_eq = eq;
                *out_keep_i = i;
                *out_keep_j = j;
            }
        }
    }
}

// ── Betting decision ──────────────────────────────────────────────────────────
//
// Returns packed int: high 16 bits = action (0=FOLD,1=RAISE,2=CHECK,3=CALL)
//                     low  16 bits = raise amount (only for RAISE)

extern "C" int betting_action(
    const int* hand2,        // 2 hole cards
    const int* community,    int n_community,
    const int* opp_discards, int n_opp_disc,
    int my_bet, int opp_bet, int min_raise, int max_raise,
    int valid_fold, int valid_raise, int valid_check, int valid_call,
    int num_sims, unsigned int seed)
{
    float equity = mc_equity(hand2, 2, community, n_community,
                             opp_discards, n_opp_disc, num_sims, seed);

    int to_call = opp_bet - my_bet;
    int pot     = my_bet + opp_bet;
    float pot_odds = (to_call > 0 && pot > 0)
                     ? (float)to_call / (to_call + pot)
                     : 0.f;

    if (equity > 0.75f && valid_raise) {
        int amount = (int)(pot * 0.75f);
        amount = std::max(amount, min_raise);
        amount = std::min(amount, max_raise);
        return (1 << 16) | (amount & 0xFFFF);  // RAISE
    }
    if (equity >= pot_odds && equity > 0.35f && valid_call)
        return (3 << 16);  // CALL
    if (valid_check)
        return (2 << 16);  // CHECK
    if (equity >= pot_odds && valid_call)
        return (3 << 16);  // CALL
    return (0 << 16);      // FOLD
}

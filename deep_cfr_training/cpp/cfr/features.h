#pragma once
#include <algorithm>
#include <cstring>
#include <cmath>
#include <unordered_map>
#include "../core/constants.h"
#include "../core/hand_eval.h"   /* classify_hand, N_CATS, HandCat */

// ── C(27,2) = 351 hand pairs ──────────────────────────────────────────────────
constexpr int N_HANDS = 351;

// ── Range validity check ───────────────────────────────────────────────────
// Always-on version: called at init time (once per game, cheap).
inline void _validate_range(const float* range, const char* ctx) {
    float s = 0.f;
    for (int i = 0; i < N_HANDS; i++) {
        if (range[i] < -1e-5f) {
            std::fprintf(stderr,
                "[RANGE ERROR][%s] negative prob at idx=%d: %.8f\n", ctx, i, range[i]);
            std::fflush(stderr);
            std::abort();
        }
        s += range[i];
    }
    if (s < 0.98f || s > 1.02f) {
        std::fprintf(stderr,
            "[RANGE ERROR][%s] sum=%.8f (expected 1.0)\n", ctx, s);
        std::fflush(stderr);
        std::abort();
    }
}
// Hot-loop version: only active when compiled with -DRANGE_VALIDATION.
// Calling _validate_range in the inner DFS loop costs ~80ms/iter at N=5000.
#ifdef RANGE_VALIDATION
#  define _VALIDATE_RANGE_HOT(r, ctx) _validate_range(r, ctx)
#else
#  define _VALIDATE_RANGE_HOT(r, ctx) ((void)0)
#endif

// ── Persistent range functions ────────────────────────────────────────────────
// These replace Python range_tracker by keeping range state INSIDE C++ per game.

// Cached action probs per (board, is_aggressive) for range updates.
// P(bet/raise | hand_i) = sigmoid((strength - 0.4) × 8)
struct _RangeActProbs { float agg[N_HANDS], pass_[N_HANDS]; };

static const _RangeActProbs* _get_range_act_probs(const int* community, int n_comm) {
    static thread_local std::unordered_map<uint64_t, _RangeActProbs> cache;
    uint64_t key = (uint64_t)n_comm;
    for (int i = 0; i < n_comm && i < 5; i++)
        key = key * 31 + (community[i] >= 0 ? (uint64_t)community[i] : 100ULL);
    auto it = cache.find(key);
    if (it != cache.end()) return &it->second;

    _RangeActProbs& ap = cache[key];
    int board5[5] = {-1,-1,-1,-1,-1};
    for (int i = 0; i < n_comm && i < 5; i++) board5[i] = community[i];
    int idx = 0;
    for (int c0 = 0; c0 < 27; c0++)
        for (int c1 = c0+1; c1 < 27; c1++) {
            int cat = classify_hand(c0, c1, board5, n_comm);
            float str = (float)(N_CATS - 1 - cat) / (N_CATS - 1);
            float p = 1.f / (1.f + std::exp(-(str - 0.4f) * 8.f));
            ap.agg[idx]   = std::max(p,       1e-4f);
            ap.pass_[idx] = std::max(1.f - p, 1e-4f);
            idx++;
        }
    return &ap;
}

// Update range when a player is observed taking an action.
// is_aggressive: true = bet/raise, false = check/call (fold zeroes range externally).
inline void range_update_betting(float* range,
                                  const int* community, int n_comm,
                                  bool is_aggressive) {
    const _RangeActProbs* ap = _get_range_act_probs(community, n_comm);
    const float* probs = is_aggressive ? ap->agg : ap->pass_;
    float total = 0.f;
    for (int i = 0; i < N_HANDS; i++) { range[i] *= probs[i]; total += range[i]; }
    if (total > 1e-9f) for (int i = 0; i < N_HANDS; i++) range[i] /= total;
}

// Convert range[N_HANDS] probability array to 17-dim category probs.
inline void range_to_cat_probs(const float* range,
                                const int* community, int n_comm,
                                float* out) {
    for (int k = 0; k < N_CATS; k++) out[k] = 0.f;
    int board5[5] = {-1,-1,-1,-1,-1};
    for (int i = 0; i < n_comm && i < 5; i++) board5[i] = community[i];
    int idx = 0;
    for (int c0 = 0; c0 < 27; c0++)
        for (int c1 = c0+1; c1 < 27; c1++) {
            if (range[idx] > 1e-9f)
                out[classify_hand(c0, c1, board5, n_comm)] += range[idx];
            idx++;
        }
    float s = 0.f; for (int k = 0; k < N_CATS; k++) s += out[k];
    if (s > 1e-9f) for (int k = 0; k < N_CATS; k++) out[k] /= s;
    else           for (int k = 0; k < N_CATS; k++) out[k] = 1.f / N_CATS;
}

// ── RangeFinder forward declarations (linked from rangefinder.cpp) ────────────
extern "C" {
    void c_range_init(const int* dead_cards, int n_dead, float* range);
    void c_range_remove_cards(float* range, const int* cards, int n_cards);
    void c_range_update_discard(float* range, const int* disc3, const int* board3);
    void c_range_update_action(float* range, const float* action_probs);
    void c_range_to_category_probs(const float* range, const int* board,
                                    int n_board, float threshold, float* out);
}

// ── Pair index for (c0, c1) pair (order-independent, 0–350) ──────────────────
inline int pair_idx(int c0, int c1) {
    if (c0 > c1) { int t=c0; c0=c1; c1=t; }
    return c0 * 26 - c0*(c0-1)/2 + (c1 - c0 - 1);
}

// ── my_cat: one-hot 17-dim category for hero's actual hand ───────────────────
static inline void _my_cat_features(int c0, int c1,
                                      const int* community, int n_comm,
                                      float* out) {
    if (c0 < 0 || c1 < 0) {
        for (int i=0; i<N_CATS; i++) out[i] = 1.f/N_CATS;
        return;
    }
    float range[N_HANDS]; std::memset(range, 0, sizeof(range));
    int dead[2] = {c0, c1};
    c_range_init(dead, 2, range);
    if (n_comm > 0) c_range_remove_cards(range, community, n_comm);
    float ap[N_HANDS]; std::memset(ap, 0, sizeof(ap));
    ap[pair_idx(c0, c1)] = 1.f;
    c_range_update_action(range, ap);
    int board5[5] = {-1,-1,-1,-1,-1};
    for (int i=0; i<n_comm && i<5; i++) board5[i] = community[i];
    c_range_to_category_probs(range, board5, n_comm, 0.f, out);
}

// ── my_range_cats: from opp's perspective — what I could have given my discards ────
// dead = community only (opp doesn't know my actual hand);
// update = c_range_update_discard on my own discarded cards (opp saw them).
static inline void _my_range_features(const int* my_disc,
                                       const int* community, int n_comm,
                                       float* out) {
    float range[N_HANDS]; std::memset(range, 0, sizeof(range));
    // dead = community cards only (opponent doesn't know my actual hand)
    if (n_comm > 0) {
        c_range_init(community, n_comm, range);
    } else {
        int no_dead[1] = {-1};
        c_range_init(no_dead, 0, range);
    }
    // Update with my discards (opponent observed them)
    bool has_my_disc = false;
    if (my_disc) for (int i=0; i<3; i++) if (my_disc[i]>=0) { has_my_disc=true; break; }
    if (has_my_disc) {
        int disc3[3] = {-1,-1,-1}, board3[3] = {-1,-1,-1};
        for (int i=0; i<3; i++) disc3[i] = my_disc[i];
        for (int i=0; i<3 && i<n_comm; i++) board3[i] = community[i];
        c_range_update_discard(range, disc3, board3);
    }
    int board5[5] = {-1,-1,-1,-1,-1};
    for (int i=0; i<n_comm && i<5; i++) board5[i] = community[i];
    c_range_to_category_probs(range, board5, n_comm, 0.f, out);
}

// ── opp_range_cats: 17-dim opp range narrowed by their discards ──────────────
static inline void _opp_range_features(int c0, int c1,
                                         const int* opp_disc,
                                         const int* community, int n_comm,
                                         float* out) {
    float range[N_HANDS]; std::memset(range, 0, sizeof(range));
    int dead[2] = {c0, c1};
    int n_dead = (c0>=0 ? 1 : 0) + (c1>=0 ? 1 : 0);
    c_range_init(dead, n_dead, range);
    if (n_comm > 0) c_range_remove_cards(range, community, n_comm);
    bool has_disc = false;
    if (opp_disc) for (int i=0; i<3; i++) if (opp_disc[i]>=0) { has_disc=true; break; }
    if (has_disc) {
        int disc3[3] = {-1,-1,-1}, board3[3] = {-1,-1,-1};
        for (int i=0; i<3; i++) disc3[i] = opp_disc[i];
        for (int i=0; i<3 && i<n_comm; i++) board3[i] = community[i];
        c_range_update_discard(range, disc3, board3);
    }
    int board5[5] = {-1,-1,-1,-1,-1};
    for (int i=0; i<n_comm && i<5; i++) board5[i] = community[i];
    c_range_to_category_probs(range, board5, n_comm, 0.f, out);
}

// ── Full 78-dim feature vector ────────────────────────────────────────────────
//
// [0-16]  my_cat         17  one-hot hand category
// [17-33] my_range_cats  17  uniform (Phase 1/2; Phase 3: Bayesian posterior)
// [34-50] opp_range_cats 17  narrowed by opp discards + betting
// [51-58] board_texture   8  paired/flush_complete/fd_present/connected/high_rank/rainbow/two_suited/coord
// [59-64] line_context    6  aggressor_me/opp, bet_facing, can_check, n_bets_me/4, n_bets_opp/4
// [65-68] pot_ratios      4  to_call/pot, my_bet/pot, opp_bet/pot, raise_room/pot
// [69-72] blocker_flags   4  blocks_top_pair, blocks_2pair, blocks_flush, blocks_straight
// [73-75] street          3  flop/turn/river one-hot
// [76]    position        1  is_bb
// [77]    pot_ratio       1  pot / MAX_BET  (big-pot context for bet sizing)
//
inline void state_to_features(
    const int* hero_hand2,
    const int* community, int n_comm,
    int my_bet, int opp_bet, int street, bool is_bb,
    const int* my_disc, const int* opp_disc,
    float* features,
    const int* street_bet_counts    = nullptr,
    const int* history_players      = nullptr,
    const int* history_actions      = nullptr,
    int        history_len          = 0,
    int        num_acts_this_street = 0,
    const float* opp_range_in       = nullptr,  // persistent range (from PostflopGame)
    const float* my_range_in        = nullptr   // persistent range (from PostflopGame)
) {
    (void)my_disc; (void)num_acts_this_street;
    for (int i=0; i<FEATURE_DIM; i++) features[i] = 0.f;
    int idx     = 0;
    int hp      = is_bb ? 1 : 0;
    int to_call = std::max(opp_bet - my_bet, 0);
    int pot     = my_bet + opp_bet;
    float fpot  = std::max((float)pot, 1.f);

    // [0-16] my_cat
    _my_cat_features(hero_hand2[0], hero_hand2[1], community, n_comm, &features[idx]);
    idx += 17;

    // [17-33] my_range_cats: opp's view of my range
    if (my_range_in != nullptr)
        range_to_cat_probs(my_range_in, community, n_comm, &features[idx]);
    else
        _my_range_features(my_disc, community, n_comm, &features[idx]);
    idx += 17;

    // [34-50] opp_range_cats
    if (opp_range_in != nullptr)
        range_to_cat_probs(opp_range_in, community, n_comm, &features[idx]);
    else
        _opp_range_features(hero_hand2[0], hero_hand2[1], opp_disc, community, n_comm, &features[idx]);
    idx += 17;

    // [51-58] board_texture (8 dims)
    {
        int bsc[3] = {};
        bool paired = false, seen[NUM_RANKS] = {};
        int min_r = NUM_RANKS-1, max_r = 0, nb = 0;
        for (int i=0; i<n_comm; i++) {
            if (community[i]<0) continue;
            int r=card_rank(community[i]), s=card_suit(community[i]);
            bsc[s]++; nb++;
            if (seen[r]) paired=true; seen[r]=true;
            if (r<min_r) min_r=r; if (r>max_r) max_r=r;
        }
        int msc = *std::max_element(bsc, bsc+3);
        int n_suits = (bsc[0]>0) + (bsc[1]>0) + (bsc[2]>0);
        features[idx]   = paired ? 1.f : 0.f;                              // [51] paired
        features[idx+1] = (nb>0 && msc==nb) ? 1.f : 0.f;                  // [52] flush_complete (monotone)
        features[idx+2] = (msc>=2) ? 1.f : 0.f;                           // [53] fd_present (2+ same suit)
        if (nb>=3) features[idx+3] = ((max_r-min_r)<=4) ? 1.f : 0.f;     // [54] connected
        features[idx+4] = (nb>0) ? (float)max_r/8.f : 0.f;               // [55] high_rank/8
        features[idx+5] = (nb>0 && n_suits==3) ? 1.f : 0.f;              // [56] rainbow (3 different suits — no flush possible)
        features[idx+6] = (nb>0 && n_suits==2) ? 1.f : 0.f;              // [57] two_suited (flush draw, not mono)
        features[idx+7] = (paired && (msc>=2||(nb>=3&&(max_r-min_r)<=4))) ? 1.f : 0.f; // [58] coord
        idx += 8;
    }

    // [59-64] line_context (6 dims)
    {
        if (history_players && history_actions && history_len>0) {
            for (int i=history_len-1; i>=0; i--) {
                if (history_actions[i]>=3) {
                    features[idx]   = (history_players[i]==hp) ? 1.f : 0.f;  // [59] aggressor_me
                    features[idx+1] = (history_players[i]!=hp) ? 1.f : 0.f; // [60] aggressor_opp
                    break;
                }
            }
        }
        features[idx+2] = (to_call>0) ? 1.f : 0.f;  // [61] bet_facing
        features[idx+3] = (to_call==0) ? 1.f : 0.f; // [62] can_check (explicit — distinct from 1-bet_facing for net clarity)
        if (street_bet_counts) {
            features[idx+4] = std::min(street_bet_counts[street*2+hp]     / 4.f, 1.f); // [63] n_bets_me/4
            features[idx+5] = std::min(street_bet_counts[street*2+1-hp]   / 4.f, 1.f); // [64] n_bets_opp/4
        }
        idx += 6;
    }

    // [65-68] pot_ratios (4 dims)
    {
        int raise_room = std::max(MAX_BET - std::max(my_bet, opp_bet), 0);
        features[idx]   = (float)to_call     / fpot;  // [65] to_call/pot
        features[idx+1] = (float)my_bet      / fpot;  // [66] my_bet/pot
        features[idx+2] = (float)opp_bet     / fpot;  // [67] opp_bet/pot
        features[idx+3] = (float)raise_room  / fpot;  // [68] raise_room/pot (remaining legal bet space)
        idx += 4;
    }

    // [69-72] blocker_flags (shared with discard via compute_blocker_flags in hand_eval.h)
    compute_blocker_flags(hero_hand2[0], hero_hand2[1], community, n_comm, &features[idx]);
    idx += 4;

    // [73-75] street one-hot (flop=1, turn=2, river=3)
    if (street>=1 && street<=3) features[idx+street-1] = 1.f;
    idx += 3;

    // [76] position
    features[idx] = is_bb ? 1.f : 0.f;
    idx++;

    // [77] pot / MAX_BET  (signals how large the pot is relative to stack cap)
    features[idx] = std::min((float)pot / (float)MAX_BET, 1.f);
    // idx == 78
}


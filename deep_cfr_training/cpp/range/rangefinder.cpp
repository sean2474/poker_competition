/**
 * rangefinder.cpp  —  Opponent 2-card range estimator via Bayesian updates.
 *
 * Range: float[351] probability distribution over opponent's 2-card kept hands.
 * Index: for c0 < c1 in [0,26], idx = packed pair index (0..350).
 *
 * Phase pipeline:
 *   1. INIT     : c_range_init
 *   2. DISCARD  : c_range_remove_cards + c_range_update_discard
 *   3. POSTFLOP : c_range_update_action (Python supplies P(a|hand) per CFR strategy)
 *   4. QUERY    : c_range_get_candidates, c_range_to_category_probs
 *
 * Compiled into libtraversal.{so,dylib} together with traversal.cpp.
 */

#include <cstring>
#include <cmath>
#include <algorithm>
#include <random>
#include "../core/constants.h"
#include "../core/hand_eval.h" /* classify_hand, N_CATS, HandCat */

static constexpr int N_HANDS = 351;   /* C(27,2) */

/* ── Pair index tables (built on first call) ───────────────────────────────── */

static int  _pair_idx[27][27];
static int  _idx_pair[N_HANDS][2];
static bool _tables_ready = false;

static void _build_tables() {
    if (_tables_ready) return;
    memset(_pair_idx, -1, sizeof(_pair_idx));
    int idx = 0;
    for (int a = 0; a < 27; a++)
        for (int b = a + 1; b < 27; b++) {
            _pair_idx[a][b] = idx;
            _idx_pair[idx][0] = a;
            _idx_pair[idx][1] = b;
            idx++;
        }
    _tables_ready = true;
}

static inline int _pidx(int a, int b) {
    return (a < b) ? _pair_idx[a][b] : _pair_idx[b][a];
}

/* ── Normalize ──────────────────────────────────────────────────────────────── */

static void _normalize(float* range) {
    float total = 0.f;
    for (int i = 0; i < N_HANDS; i++) total += range[i];
    if (total > 1e-9f)
        for (int i = 0; i < N_HANDS; i++) range[i] /= total;
}

/* ── sigmoid helper ─────────────────────────────────────────────────────────── */
static inline float _sig(float x) { return 1.f / (1.f + expf(-x)); }

/* ════════════════════════════════════════════════════════════════════════════
 * C API
 * ══════════════════════════════════════════════════════════════════════════ */
extern "C" {

/**
 * c_range_init
 * Initialize uniform range over all 2-card pairs that don't overlap dead_cards.
 *
 * dead_cards[n_dead] : cards known to be unavailable (our hand + board so far)
 * range_out[351]     : output probability array
 */
void c_range_init(const int* dead_cards, int n_dead, float* range_out) {
    _build_tables();
    bool dead[27] = {};
    for (int i = 0; i < n_dead; i++)
        if (dead_cards[i] >= 0 && dead_cards[i] < 27)
            dead[dead_cards[i]] = true;

    for (int i = 0; i < N_HANDS; i++) {
        int c0 = _idx_pair[i][0], c1 = _idx_pair[i][1];
        range_out[i] = (!dead[c0] && !dead[c1]) ? 1.f : 0.f;
    }
    _normalize(range_out);
}

/**
 * c_range_remove_cards
 * Set probability to 0 for any hand containing a newly-revealed card,
 * then renormalize.  Call when flop/turn/river card(s) are dealt.
 *
 * cards[n_cards] : newly dead cards (e.g., board3, opp_disc3)
 */
void c_range_remove_cards(float* range, const int* cards, int n_cards) {
    _build_tables();
    bool rem[27] = {};
    for (int i = 0; i < n_cards; i++)
        if (cards[i] >= 0 && cards[i] < 27) rem[cards[i]] = true;

    for (int i = 0; i < N_HANDS; i++) {
        if (!range[i]) continue;
        int c0 = _idx_pair[i][0], c1 = _idx_pair[i][1];
        if (rem[c0] || rem[c1]) range[i] = 0.f;
    }
    _normalize(range);
}

/**
 * c_range_update_discard
 * Remove hands that overlap with the observed discard cards.
 * Uniform weight among surviving pairs (no fast_score heuristic).
 *
 * Proper discard-weighted update is handled in Python after this call:
 *   Phase 2: run_one_iter samples discard_buffer → DiscardCFR trains
 *   Phase 3: c_range_update_action(range, discard_net_probs) from Python
 *
 * board3 is kept for API compatibility but unused.
 */
void c_range_update_discard(float* range,
                             const int* opp_disc3,
                             const int* /* board3 */) {
    _build_tables();
    bool disc_set[27] = {};
    for (int i = 0; i < 3; i++)
        if (opp_disc3[i] >= 0) disc_set[opp_disc3[i]] = true;
    for (int i = 0; i < N_HANDS; i++) {
        int c0 = _idx_pair[i][0], c1 = _idx_pair[i][1];
        if (disc_set[c0] || disc_set[c1]) range[i] = 0.f;
    }
    _normalize(range);
}

/**
 * c_range_update_action  [Phase 1 / 3 / 4 / 5]
 * Generic Bayesian update: range[i] *= action_probs[i], then renormalize.
 *
 * action_probs[351] : P(observed_action | hand_i) for every candidate pair.
 *   — For preflop: computed from preflop_strategy_sum in Python.
 *   — For postflop: computed by running postflop_net.forward(feats(hand_i))
 *                   in Python for each candidate hand.
 */
void c_range_update_action(float* range, const float* action_probs) {
    _build_tables();
    for (int i = 0; i < N_HANDS; i++)
        range[i] *= std::max(action_probs[i], 1e-6f);
    _normalize(range);
}

/**
 * c_range_get_candidates
 * Return all hands with probability >= min_prob, sorted descending.
 *
 * hands_out[K*2]  : (c0, c1) pairs (flattened)
 * probs_out[K]    : corresponding probabilities
 * Returns K (count of returned hands).
 */
int c_range_get_candidates(const float* range, float min_prob,
                           int* hands_out, float* probs_out) {
    _build_tables();

    /* Collect */
    int count = 0;
    for (int i = 0; i < N_HANDS; i++) {
        if (range[i] >= min_prob) {
            hands_out[count * 2]     = _idx_pair[i][0];
            hands_out[count * 2 + 1] = _idx_pair[i][1];
            probs_out[count]         = range[i];
            count++;
        }
    }

    /* Sort descending by probability (simple insertion for small arrays) */
    for (int i = 1; i < count; i++) {
        float pv = probs_out[i];
        int h0 = hands_out[i*2], h1 = hands_out[i*2+1];
        int j = i - 1;
        while (j >= 0 && probs_out[j] < pv) {
            probs_out[j+1]     = probs_out[j];
            hands_out[(j+1)*2] = hands_out[j*2];
            hands_out[(j+1)*2+1] = hands_out[j*2+1];
            j--;
        }
        probs_out[j+1]     = pv;
        hands_out[(j+1)*2] = h0;
        hands_out[(j+1)*2+1] = h1;
    }
    return count;
}

/**
 * c_range_copy / c_range_entropy  (utility)
 */
void c_range_copy(const float* src, float* dst) {
    memcpy(dst, src, N_HANDS * sizeof(float));
}

float c_range_entropy(const float* range) {
    float h = 0.f;
    for (int i = 0; i < N_HANDS; i++)
        if (range[i] > 1e-9f) h -= range[i] * logf(range[i]);
    return h;
}

/**
 * c_range_to_category_probs
 *
 * Collapse range[351] into N_CATS-dim category probability vector.
 * out[k] = sum of range[i] for hands classified as category k.
 * Uses classify_hand() from hand_eval.h.
 */
void c_range_to_category_probs(const float* range, const int* board,
                                int n_board, float threshold, float* out) {
    _build_tables();
    for (int k = 0; k < N_CATS; k++) out[k] = 0.f;
    for (int i = 0; i < N_HANDS; i++) {
        if (range[i] < threshold) continue;
        int c0 = _idx_pair[i][0], c1 = _idx_pair[i][1];
        out[classify_hand(c0, c1, board, n_board)] += range[i];
    }
}

}  /* extern "C" */

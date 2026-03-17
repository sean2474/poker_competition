/*
 * C++ traversal engine for Deep CFR.
 * Compiled as shared library, called from Python via ctypes.
 * 
 * Build:
 *   g++ -O3 -shared -fPIC -std=c++17 -o libtraversal.so traversal.cpp -lpthread
 *
 * Python usage:
 *   lib = ctypes.CDLL("./libtraversal.so")
 *   lib.run_traversals(...)
 */

#include <cstring>
#include <cstdlib>
#include <cmath>
#include <random>
#include <thread>
#include <mutex>
#include <vector>
#include <algorithm>
#include "game_state.h"

// ═══════════════════════════════════════════════
// Simplified hand evaluator
// ═══════════════════════════════════════════════

static int hand_score(const int* cards7) {
    int rank_count[9] = {};
    int suit_count[3] = {};
    int ranks[7], suits[7];
    
    for (int i = 0; i < 7; i++) {
        ranks[i] = card_rank(cards7[i]);
        suits[i] = card_suit(cards7[i]);
        rank_count[ranks[i]]++;
        suit_count[suits[i]]++;
    }
    
    int score = 0;
    
    // Trips
    int trips_rank = -1;
    for (int r = 8; r >= 0; r--) {
        if (rank_count[r] == 3) { trips_rank = r; break; }
    }
    
    // Pairs
    int pair_ranks[2] = {-1, -1};
    int np = 0;
    for (int r = 8; r >= 0; r--) {
        if (rank_count[r] == 2 && np < 2) pair_ranks[np++] = r;
    }
    
    // Full house
    if (trips_rank >= 0 && np > 0) score = std::max(score, 5000 + trips_rank * 10 + pair_ranks[0]);
    
    // Flush
    int flush_suit = -1;
    for (int s = 0; s < 3; s++) if (suit_count[s] >= 5) flush_suit = s;
    
    // Straight
    bool has_rank[10] = {};
    for (int r = 0; r < 9; r++) if (rank_count[r] > 0) has_rank[r] = true;
    if (has_rank[8]) has_rank[9] = true; // Ace low
    
    int straight_high = -1;
    for (int start = 5; start >= 0; start--) {
        // Check 5 consecutive: start, start+1, ..., start+4
        // But we only have indices 0-9, and straight is 5 consecutive
        if (start + 4 <= 9) {
            bool ok = true;
            for (int i = 0; i < 5; i++) if (!has_rank[start + i]) { ok = false; break; }
            if (ok) { straight_high = start + 4; break; }
        }
    }
    // Also check A-2-3-4-5 (indices 9,0,1,2,3)
    if (straight_high < 0 && has_rank[9] && has_rank[0] && has_rank[1] && has_rank[2] && has_rank[3]) {
        straight_high = 3; // 5-high straight
    }
    
    if (straight_high >= 0) score = std::max(score, 3500 + straight_high);
    if (flush_suit >= 0) score = std::max(score, 3000 + suit_count[flush_suit]);
    
    // Straight flush
    if (flush_suit >= 0 && straight_high >= 0) {
        int suited_ranks[10] = {};
        for (int i = 0; i < 7; i++) {
            if (suits[i] == flush_suit) suited_ranks[ranks[i]] = 1;
        }
        if (suited_ranks[8]) suited_ranks[9] = 1;
        for (int start = 5; start >= 0; start--) {
            if (start + 4 <= 9) {
                bool ok = true;
                for (int i = 0; i < 5; i++) if (!suited_ranks[start + i]) { ok = false; break; }
                if (ok) { score = std::max(score, 6000 + start + 4); break; }
            }
        }
        if (suited_ranks[9] && suited_ranks[0] && suited_ranks[1] && suited_ranks[2] && suited_ranks[3]) {
            score = std::max(score, 6000);
        }
    }
    
    // Two pair / pair / trips (if no better hand)
    if (trips_rank >= 0 && score < 4000) score = std::max(score, 4000 + trips_rank * 10);
    if (np >= 2 && score < 2000) score = std::max(score, 2000 + pair_ranks[0] * 10 + pair_ranks[1]);
    else if (np >= 1 && score < 1000) score = std::max(score, 1000 + pair_ranks[0] * 10);
    
    // High card kicker
    for (int r = 8; r >= 0; r--) {
        if (rank_count[r] > 0) { score += r; break; }
    }
    
    return score;
}

static int evaluate_showdown(const int* p0_hand, const int* p1_hand, const int* community) {
    int cards0[7] = {p0_hand[0], p0_hand[1], community[0], community[1], community[2], community[3], community[4]};
    int cards1[7] = {p1_hand[0], p1_hand[1], community[0], community[1], community[2], community[3], community[4]};
    int s0 = hand_score(cards0);
    int s1 = hand_score(cards1);
    if (s0 > s1) return 1;
    if (s0 < s1) return -1;
    return 0;
}

// ═══════════════════════════════════════════════
// Traversal output buffer (collected by C++, consumed by Python)
// ═══════════════════════════════════════════════

struct TraversalSample {
    float features[FEATURE_DIM];
    float values[NUM_ACTIONS];
    float valid_mask[NUM_ACTIONS];
    int player;       // 0 or 1
    int sample_type;  // 0 = advantage, 1 = strategy
    int iteration;
};

static std::vector<TraversalSample> g_samples;
static std::mutex g_mutex;

// Strategy callback: network inference from Python
// During traversal, we need network output. We pre-compute strategies.
// Simpler approach: pass advantage predictions from Python, do traversal in C++.

// Actually, the cleanest approach: C++ does the FULL traversal with a 
// callback to Python for network inference. But ctypes callbacks are slow.

// Best approach for speed: C++ does traversal with a SIMPLE heuristic strategy
// (uniform or proportional to some fast score), collects (features, action_values),
// and Python uses those to compute advantages and train.

// BUT that defeats the purpose — the strategy used during traversal matters for CFR.

// Compromise: C++ handles GameState + features + showdown.
// Python handles traverse logic + network inference.
// C++ just provides fast primitives.

// ═══════════════════════════════════════════════
// Exported C functions for Python ctypes
// ═══════════════════════════════════════════════

extern "C" {

// Deal a game: shuffle deck, write to output arrays
void deal_game(int* p0_5, int* p1_5, int* community, unsigned int seed) {
    std::mt19937 rng(seed);
    int deck[DECK_SIZE];
    shuffle_deck(deck, rng);
    std::copy(deck, deck + 5, p0_5);
    std::copy(deck + 5, deck + 10, p1_5);
    std::copy(deck + 10, deck + 15, community);
}

// Fast discard
void c_fast_discard(const int* hand5, const int* board3, int* ki, int* kj, unsigned int seed, float temperature) {
    std::mt19937 rng(seed);
    fast_discard(hand5, board3, *ki, *kj, rng, temperature);
}

// Compute features for a game state (outputs full 119-dim vector)
void c_state_features(
    const int* hero_hand2, const int* hero_hand5,
    const int* community, int n_comm,
    int my_bet, int opp_bet, int street, int is_bb,
    const int* my_disc, const int* opp_disc,
    int use_hand5,
    float* features_out,
    const int* street_bets_flat,          // 8 ints [s0p0,s0p1,...] or null
    const float* street_last_ratios_flat, // 8 floats [s0p0,s0p1,...] or null
    const int*   street_bet_counts_flat,  // 8 ints  [s0p0,s0p1,...] or null
    const int*   history_players,         // [history_len] or null
    const int*   history_actions,         // [history_len] or null
    int          history_len,
    int          num_acts_this_street
) {
    int sb[4][2] = {};
    if (street_bets_flat) {
        for (int s = 0; s < 4; s++) {
            sb[s][0] = street_bets_flat[s*2];
            sb[s][1] = street_bets_flat[s*2+1];
        }
    }
    state_to_features(hero_hand2, hero_hand5, community, n_comm,
                       my_bet, opp_bet, street, (bool)is_bb,
                       my_disc, opp_disc, (bool)use_hand5, features_out,
                       street_bets_flat ? sb : nullptr,
                       street_last_ratios_flat,
                       street_bet_counts_flat,
                       history_players, history_actions,
                       history_len, num_acts_this_street);
}

// Evaluate showdown: returns +1 p0 wins, -1 p1 wins, 0 tie
int c_evaluate_showdown(const int* p0_hand, const int* p1_hand, const int* community) {
    return evaluate_showdown(p0_hand, p1_hand, community);
}

// Batch deal + discard: deal N games, discard, return hands
void c_batch_deal_discard(int n, int* p0_hands, int* p1_hands, int* p0_discs, int* p1_discs,
                           int* communities, int* p0_hand5s, int* p1_hand5s,
                           unsigned int base_seed, float temperature) {
    #pragma omp parallel for schedule(dynamic) if(n > 10)
    for (int i = 0; i < n; i++) {
        std::mt19937 rng(base_seed + i);
        int deck[DECK_SIZE];
        shuffle_deck(deck, rng);
        
        int* p0_5 = &p0_hand5s[i * 5];
        int* p1_5 = &p1_hand5s[i * 5];
        int* comm = &communities[i * 5];
        std::copy(deck, deck + 5, p0_5);
        std::copy(deck + 5, deck + 10, p1_5);
        std::copy(deck + 10, deck + 15, comm);
        
        int ki0, kj0, ki1, kj1;
        fast_discard(p0_5, comm, ki0, kj0, rng, temperature);
        p0_hands[i * 2] = p0_5[ki0]; p0_hands[i * 2 + 1] = p0_5[kj0];
        int d = 0;
        for (int j = 0; j < 5; j++) if (j != ki0 && j != kj0) p0_discs[i * 3 + d++] = p0_5[j];
        
        fast_discard(p1_5, comm, ki1, kj1, rng, temperature);
        p1_hands[i * 2] = p1_5[ki1]; p1_hands[i * 2 + 1] = p1_5[kj1];
        d = 0;
        for (int j = 0; j < 5; j++) if (j != ki1 && j != kj1) p1_discs[i * 3 + d++] = p1_5[j];
    }
}

} // extern "C"

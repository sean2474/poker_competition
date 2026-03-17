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
// C++ Postflop Traversal State Machine
// ═══════════════════════════════════════════════

constexpr int MAX_STACK   = 32;
constexpr int MAX_VALID   = 8;
constexpr int MAX_SAMPLES = 60;   // per game: ~10-20 advantage + strategy samples

struct TraversalFrame {
    FullGameState state;
    int   valid[MAX_VALID], n_valid;
    float action_evs[MAX_VALID];    // EV accumulated for each action
    int   action_idx;               // next action index to explore
    float strategy[MAX_VALID];      // current strategy at this node
    float features[FEATURE_DIM];   // cached features (for buffer)
    int   is_traversing;            // 1 = traversing player acts here
};

struct PostflopGame {
    // Per-game deal (set once at init)
    int p0_hand[2], p1_hand[2];
    int p0_hand5[5], p1_hand5[5];
    int community[5];
    int p0_disc[3], p1_disc[3];
    int traversing_player;
    float iteration_weight;         // CFR iteration weight

    // DFS stack
    TraversalFrame stack[MAX_STACK];
    int depth;

    // Inference pending
    int  waiting;
    float pending_feats[FEATURE_DIM];
    int  pending_valid[MAX_VALID], pending_n_valid, pending_player;

    // Output EV
    float ev;
    int   done;

    // Accumulated advantage samples [traversing player]
    float adv_feat[MAX_SAMPLES][FEATURE_DIM];
    float adv_val [MAX_SAMPLES][NUM_ACTIONS];
    float adv_mask[MAX_SAMPLES][NUM_ACTIONS];
    int   adv_street[MAX_SAMPLES], n_adv;

    // Accumulated strategy samples [both players]
    float str_feat [MAX_SAMPLES][FEATURE_DIM];
    float str_val  [MAX_SAMPLES][NUM_ACTIONS];
    float str_mask [MAX_SAMPLES][NUM_ACTIONS];
    int   str_street[MAX_SAMPLES], n_str;
};

// Compute terminal EV for the traversing player
static float terminal_ev(const FullGameState& s, int tp,
                          const int* p0_hand, const int* p1_hand,
                          const int* community) {
    if (s.folded_player >= 0) {
        if (s.folded_player == tp)
            return -(float)s.bets[tp];
        else
            return  (float)s.bets[1 - tp];
    }
    int pot = std::min(s.bets[0], s.bets[1]);
    int sd  = evaluate_showdown(p0_hand, p1_hand, community);
    return (tp == 0) ? (float)(sd * pot) : (float)(-sd * pot);
}

// Extract features from a FullGameState into a flat array
static void extract_features_full(const FullGameState& s, int cp,
                                   const int* p0_hand, const int* p1_hand,
                                   const int* p0_disc, const int* p1_disc,
                                   const int* community,
                                   float* out) {
    const int* my_hand  = (cp == 0) ? p0_hand  : p1_hand;
    const int* my_disc  = (cp == 0) ? p0_disc  : p1_disc;
    const int* opp_disc = (cp == 0) ? p1_disc  : p0_disc;
    int vis_n = s.street == 0 ? 0 : s.street + 2;  // 0/3/4/5
    int vis_n2 = std::min(vis_n, 5);

    int dummy5[5] = {-1,-1,-1,-1,-1};
    const int* h5 = dummy5;  // post-discard: no 5-card hand

    int sb[4][2];
    for (int ss = 0; ss < 4; ss++) {
        sb[ss][0] = s.street_bets[ss][0];
        sb[ss][1] = s.street_bets[ss][1];
    }
    state_to_features(my_hand, h5, community, vis_n2,
                       s.bets[cp], s.bets[1-cp], s.street, cp == 1,
                       my_disc, opp_disc, false, out,
                       sb,
                       &s.street_last_ratios[0][0],
                       &s.street_bet_counts[0][0],
                       s.history_players, s.history_actions, s.history_len,
                       s.num_actions_this_street);
}

// Regret matching: strategy = max(adv, 0) / sum(max(adv,0)) else uniform
static void regret_match(const float* adv, const int* valid, int n,
                          float* strategy) {
    float total = 0.f;
    float best_v = -1e9f; int best_a = valid[0];
    for (int i = 0; i < n; i++) {
        float v = adv[valid[i]];
        if (v > 0) total += v;
        if (v > best_v) { best_v = v; best_a = valid[i]; }
    }
    if (total > 0) {
        for (int i = 0; i < n; i++) {
            strategy[valid[i]] = std::max(adv[valid[i]], 0.f) / total;
        }
    } else {
        for (int i = 0; i < n; i++) strategy[valid[i]] = 1.f / n;
    }
}

// Advance one game's DFS until it needs inference or completes
// strategy: if non-null, the strategy for the current pending node (resumes)
static void advance_game(PostflopGame* g, const float* net_adv, std::mt19937* rng) {
    if (g->done) return;

    // If we were waiting, apply the strategy and continue
    if (g->waiting && net_adv) {
        g->waiting = 0;
        TraversalFrame& f = g->stack[g->depth - 1];
        // compute strategy from net_adv
        regret_match(net_adv, f.valid, f.n_valid, f.strategy);

        // store strategy sample (always, for both players)
        if (g->n_str < MAX_SAMPLES) {
            std::memcpy(g->str_feat[g->n_str], f.features, sizeof(float)*FEATURE_DIM);
            std::memset(g->str_val[g->n_str], 0, sizeof(float)*NUM_ACTIONS);
            std::memset(g->str_mask[g->n_str], 0, sizeof(float)*NUM_ACTIONS);
            for (int i = 0; i < f.n_valid; i++) {
                int a = f.valid[i];
                g->str_val[g->n_str][a]  = f.strategy[a];
                g->str_mask[g->n_str][a] = 1.f;
            }
            g->str_street[g->n_str] = f.state.street;
            g->n_str++;
        }

        if (f.is_traversing) {
            // explore ALL actions (already set up in f.action_idx = 0)
        } else {
            // sample ONE action from strategy
            float r = (rng) ? std::uniform_real_distribution<float>(0,1)(*rng) : 0.5f;
            float cum = 0.f;
            int chosen = f.valid[0];
            for (int i = 0; i < f.n_valid; i++) {
                cum += f.strategy[f.valid[i]];
                if (r <= cum) { chosen = f.valid[i]; break; }
            }
            // mark all other actions as "explored" (only 1 action for opponent)
            f.action_idx = 0;
            // put chosen action first, skip others
            for (int i = 0; i < f.n_valid; i++) {
                if (f.valid[i] == chosen) {
                    std::swap(f.valid[0], f.valid[i]);
                    break;
                }
            }
            f.n_valid = 1;  // only explore chosen action
        }
    }

    // Main DFS loop
    while (g->depth > 0 && !g->waiting) {
        TraversalFrame& f = g->stack[g->depth - 1];

        if (f.action_idx >= f.n_valid) {
            // Frame complete: compute EV and propagate
            float ev;
            if (f.is_traversing) {
                // advantage and EV
                float node_ev = 0.f;
                for (int i = 0; i < f.n_valid; i++)
                    node_ev += f.strategy[f.valid[i]] * f.action_evs[i];
                // store advantage sample
                if (g->n_adv < MAX_SAMPLES) {
                    std::memcpy(g->adv_feat[g->n_adv], f.features, sizeof(float)*FEATURE_DIM);
                    std::memset(g->adv_val[g->n_adv], 0, sizeof(float)*NUM_ACTIONS);
                    std::memset(g->adv_mask[g->n_adv], 0, sizeof(float)*NUM_ACTIONS);
                    for (int i = 0; i < f.n_valid; i++) {
                        int a = f.valid[i];
                        g->adv_val[g->n_adv][a]  = f.action_evs[i] - node_ev;
                        g->adv_mask[g->n_adv][a] = 1.f;
                    }
                    g->adv_street[g->n_adv] = f.state.street;
                    g->n_adv++;
                }
                ev = node_ev;
            } else {
                ev = f.action_evs[0];  // opponent sampled one action
            }
            g->depth--;
            if (g->depth == 0) {
                g->ev = ev;
                g->done = 1;
                return;
            }
            TraversalFrame& parent = g->stack[g->depth - 1];
            parent.action_evs[parent.action_idx - 1] = ev;
            continue;
        }

        // Explore next action
        int action = f.valid[f.action_idx];
        f.action_idx++;
        FullGameState child = f.state.apply(action);

        // Terminal?
        if (child.is_terminal) {
            float ev = terminal_ev(child, g->traversing_player,
                                   g->p0_hand, g->p1_hand, g->community);
            f.action_evs[f.action_idx - 1] = ev;
            continue;
        }

        // Push child frame
        if (g->depth >= MAX_STACK - 1) {
            // Stack overflow: treat as terminal with 0 EV
            f.action_evs[f.action_idx - 1] = 0.f;
            continue;
        }
        TraversalFrame& nf = g->stack[g->depth];
        nf.state = child;
        child.get_valid_actions(nf.valid, nf.n_valid);
        std::memset(nf.action_evs, 0, sizeof(nf.action_evs));
        nf.action_idx = 0;
        std::memset(nf.strategy, 0, sizeof(nf.strategy));
        nf.is_traversing = (child.current_player == g->traversing_player) ? 1 : 0;
        g->depth++;

        // Extract features for this node (needed for both warmup and inference)
        int cp = child.current_player;
        extract_features_full(child, cp,
                               g->p0_hand, g->p1_hand,
                               g->p0_disc, g->p1_disc,
                               g->community,
                               nf.features);

        // Need inference
        std::memcpy(g->pending_feats, nf.features, sizeof(float)*FEATURE_DIM);
        std::memcpy(g->pending_valid, nf.valid, sizeof(int)*nf.n_valid);
        g->pending_n_valid = nf.n_valid;
        g->pending_player  = cp;
        g->waiting         = 1;
        return;  // pause until Python sends strategy
    }
}

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

// ── Batch warmup equity ────────────────────────────────────────────────────
// Replaces Python warmup_ev: MC equity × committed chips, OpenMP-parallel
void c_batch_warmup_ev(
    int n,
    const int* p0_hand5s,          // [n×5]
    const int* p1_hand5s,          // [n×5]
    const int* my_bets,            // [n]
    const int* opp_bets,           // [n]
    const int* traversing_players, // [n]
    float*     evs_out,            // [n]
    unsigned int base_seed,
    int n_boards                   // MC boards per eval (default 15)
) {
    #pragma omp parallel for schedule(dynamic) if(n > 10)
    for (int i = 0; i < n; i++) {
        std::mt19937 rng(base_seed + (unsigned)i * 1013);

        const int* p0h5 = p0_hand5s + i * 5;
        const int* p1h5 = p1_hand5s + i * 5;

        // Build remaining deck
        bool dead[DECK_SIZE] = {};
        for (int j = 0; j < 5; j++) { dead[p0h5[j]] = true; dead[p1h5[j]] = true; }
        int remaining[DECK_SIZE]; int nr = 0;
        for (int c = 0; c < DECK_SIZE; c++) if (!dead[c]) remaining[nr++] = c;

        float wins = 0.f;
        int total = 0;
        for (int b = 0; b < n_boards && nr >= 5; b++) {
            // Sample 5 community cards
            int comm[5];
            int pool[DECK_SIZE]; std::copy(remaining, remaining+nr, pool);
            for (int k = 0; k < 5; k++) {
                std::uniform_int_distribution<int> d(k, nr-1);
                int r2 = d(rng); std::swap(pool[k], pool[r2]);
                comm[k] = pool[k];
            }
            int flop[3] = {comm[0], comm[1], comm[2]};
            int ki0, kj0, ki1, kj1;
            fast_discard(p0h5, flop, ki0, kj0, rng, 0.f);
            fast_discard(p1h5, flop, ki1, kj1, rng, 0.f);
            int p0k[2] = {p0h5[ki0], p0h5[kj0]};
            int p1k[2] = {p1h5[ki1], p1h5[kj1]};
            int sd = evaluate_showdown(p0k, p1k, comm);
            if      (sd > 0) wins += 1.f;
            else if (sd == 0) wins += 0.5f;
            total++;
        }
        float equity = (total > 0) ? wins / total : 0.5f;
        int tp = traversing_players[i];
        float committed = (float)my_bets[i];
        evs_out[i] = (tp == 0) ? (2*equity - 1)*committed : (1 - 2*equity)*committed;
    }
}

// ── Postflop batch traversal C++ state machine ─────────────────────────────
// Allocate N PostflopGame structs (Python holds the pointer as an opaque handle)
PostflopGame* c_postflop_alloc(int n) {
    return new PostflopGame[n]();
}
void c_postflop_free(PostflopGame* games) { delete[] games; }

// Initialize one game at index idx
void c_postflop_init_one(
    PostflopGame* games, int idx,
    const int* initial_state_flat,  // FullGameState serialized as int array
    const int* p0_hand, const int* p1_hand,
    const int* p0_hand5, const int* p1_hand5,
    const int* community,
    const int* p0_disc, const int* p1_disc,
    int traversing_player
) {
    PostflopGame& g = games[idx];
    std::memset(&g, 0, sizeof(PostflopGame));

    // Decode initial state from flat int array (matches Python serialisation)
    // Flat layout: [street, bets0, bets1, cp, is_terminal, folded_player,
    //               min_raise, last_street_bet, num_acts, preflop_override,
    //               street_bets[4][2]=8, history_len, history_players[N], history_actions[N]]
    const int* p = initial_state_flat;
    FullGameState& s0 = g.stack[0].state;
    s0.street                  = *p++;
    s0.bets[0]                 = *p++;  s0.bets[1]         = *p++;
    s0.current_player          = *p++;
    s0.is_terminal             = (bool)*p++;
    s0.folded_player           = *p++;
    s0.min_raise               = *p++;
    s0.last_street_bet         = *p++;
    s0.num_actions_this_street = *p++;
    s0.preflop_open_override   = *p++;
    for (int ss = 0; ss < 4; ss++) {
        s0.street_bets[ss][0] = *p++; s0.street_bets[ss][1] = *p++;
    }
    // street_last_ratios (passed as float, cast from int ptr)
    const float* fp = reinterpret_cast<const float*>(p);
    for (int ss = 0; ss < 4; ss++) {
        s0.street_last_ratios[ss][0] = *fp++; s0.street_last_ratios[ss][1] = *fp++;
    }
    p = reinterpret_cast<const int*>(fp);
    for (int ss = 0; ss < 4; ss++) {
        s0.street_bet_counts[ss][0] = *p++; s0.street_bet_counts[ss][1] = *p++;
    }
    int hlen = *p++;
    s0.history_len = hlen;
    for (int h = 0; h < hlen && h < MAX_HISTORY; h++) s0.history_players[h] = *p++;
    for (int h = 0; h < hlen && h < MAX_HISTORY; h++) s0.history_actions[h] = *p++;

    s0.get_valid_actions(g.stack[0].valid, g.stack[0].n_valid);
    g.stack[0].action_idx    = 0;
    g.stack[0].is_traversing = (s0.current_player == traversing_player) ? 1 : 0;
    g.depth = 1;

    std::copy(p0_hand,  p0_hand  + 2, g.p0_hand);
    std::copy(p1_hand,  p1_hand  + 2, g.p1_hand);
    std::copy(p0_hand5, p0_hand5 + 5, g.p0_hand5);
    std::copy(p1_hand5, p1_hand5 + 5, g.p1_hand5);
    std::copy(community,community + 5, g.community);
    std::copy(p0_disc,  p0_disc  + 3, g.p0_disc);
    std::copy(p1_disc,  p1_disc  + 3, g.p1_disc);

    g.traversing_player = traversing_player;
    g.ev = 0.f; g.done = 0; g.waiting = 0;
    g.n_adv = 0; g.n_str = 0;

    // Prime the first inference request
    extract_features_full(s0, s0.current_player,
                           g.p0_hand, g.p1_hand,
                           g.p0_disc, g.p1_disc, g.community,
                           g.stack[0].features);
    std::memcpy(g.pending_feats, g.stack[0].features, sizeof(float)*FEATURE_DIM);
    std::memcpy(g.pending_valid, g.stack[0].valid, sizeof(int)*g.stack[0].n_valid);
    g.pending_n_valid = g.stack[0].n_valid;
    g.pending_player  = s0.current_player;
    g.waiting = 1;
}

// Collect pending inference requests from all games into output arrays.
// Returns number of pending games.
int c_postflop_collect_pending(
    PostflopGame* games, int n,
    float* feats_out,    // [n_pending × FEATURE_DIM]
    int*   valid_out,    // [n_pending × NUM_ACTIONS]
    int*   n_valid_out,  // [n_pending]
    int*   player_out,   // [n_pending]
    int*   idx_out       // [n_pending] → original game index
) {
    int cnt = 0;
    for (int i = 0; i < n; i++) {
        if (games[i].waiting && !games[i].done) {
            std::memcpy(feats_out + cnt * FEATURE_DIM, games[i].pending_feats,
                        sizeof(float) * FEATURE_DIM);
            std::memset(valid_out + cnt * NUM_ACTIONS, 0, sizeof(int) * NUM_ACTIONS);
            for (int j = 0; j < games[i].pending_n_valid; j++)
                valid_out[cnt * NUM_ACTIONS + j] = games[i].pending_valid[j];
            n_valid_out[cnt] = games[i].pending_n_valid;
            player_out[cnt]  = games[i].pending_player;
            idx_out[cnt]     = i;
            cnt++;
        }
    }
    return cnt;
}

// Resume games with provided strategies (net advantage arrays) and advance.
void c_postflop_resume_batch(
    PostflopGame* games,
    const int*   game_idxs,   // [n_pending]
    const float* net_advs,    // [n_pending × NUM_ACTIONS]
    int n_pending,
    unsigned int base_seed
) {
    #pragma omp parallel for schedule(dynamic) if(n_pending > 4)
    for (int j = 0; j < n_pending; j++) {
        int i = game_idxs[j];
        std::mt19937 rng(base_seed + (unsigned)i * 997);
        const float* adv = net_advs + j * NUM_ACTIONS;
        advance_game(&games[i], adv, &rng);
    }
}

// Collect all buffer samples from all completed games into pre-allocated arrays.
// adv_out_N, str_out_N: max samples to write
void c_postflop_collect_samples(
    PostflopGame* games, int n,
    // Advantage buffer
    float* adv_feats, float* adv_vals, float* adv_masks,
    int*   adv_streets, int*  adv_players, float* adv_iters,
    int*   adv_count_out,
    // Strategy buffer
    float* str_feats, float* str_vals, float* str_masks,
    int*   str_streets, float* str_iters,
    int*   str_count_out,
    float iteration, int traversing_player, int max_samples
) {
    int na = 0, ns = 0;
    for (int i = 0; i < n && na < max_samples; i++) {
        PostflopGame& g = games[i];
        for (int k = 0; k < g.n_adv && na < max_samples; k++, na++) {
            std::memcpy(adv_feats + na*FEATURE_DIM, g.adv_feat[k], sizeof(float)*FEATURE_DIM);
            std::memcpy(adv_vals  + na*NUM_ACTIONS, g.adv_val[k],  sizeof(float)*NUM_ACTIONS);
            std::memcpy(adv_masks + na*NUM_ACTIONS, g.adv_mask[k], sizeof(float)*NUM_ACTIONS);
            adv_streets[na]  = g.adv_street[k];
            adv_players[na]  = traversing_player;
            adv_iters[na]    = iteration;
        }
    }
    for (int i = 0; i < n && ns < max_samples; i++) {
        PostflopGame& g = games[i];
        for (int k = 0; k < g.n_str && ns < max_samples; k++, ns++) {
            std::memcpy(str_feats + ns*FEATURE_DIM, g.str_feat[k], sizeof(float)*FEATURE_DIM);
            std::memcpy(str_vals  + ns*NUM_ACTIONS, g.str_val[k],  sizeof(float)*NUM_ACTIONS);
            std::memcpy(str_masks + ns*NUM_ACTIONS, g.str_mask[k], sizeof(float)*NUM_ACTIONS);
            str_streets[ns] = g.str_street[k];
            str_iters[ns]   = iteration;
        }
    }
    *adv_count_out = na;
    *str_count_out = ns;
}

// Get EVs from completed games
void c_postflop_get_evs(PostflopGame* games, int n, float* evs_out) {
    for (int i = 0; i < n; i++) evs_out[i] = games[i].ev;
}

// How many games are still pending (not done)?
int c_postflop_n_pending(PostflopGame* games, int n) {
    int cnt = 0;
    for (int i = 0; i < n; i++) if (!games[i].done) cnt++;
    return cnt;
}

} // extern "C"

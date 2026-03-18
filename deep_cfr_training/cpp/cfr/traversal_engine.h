#pragma once
#include <cstring>
#include <random>
#include "../core/constants.h"
#include "../core/game_state.h"
#include "features.h"
#include "../heuristic/discard.h"

// ── Traversal state machine ───────────────────────────────────────────────────

constexpr int MAX_STACK   = 32;
constexpr int MAX_VALID   = 8;
constexpr int MAX_SAMPLES = 60;

// One DFS frame
struct TraversalFrame {
    FullGameState state;
    int   valid[MAX_VALID], n_valid;
    float action_evs[MAX_VALID];
    int   action_idx;
    float strategy[MAX_VALID];
    float features[FEATURE_DIM];
    int   is_traversing;
    float my_range_snap[351];   // saved g->my_range before this traversing frame's first action
};

// One game's full traversal state
struct PostflopGame {
    int p0_hand[2], p1_hand[2];
    int p0_hand5[5], p1_hand5[5];
    int community[5];
    int p0_disc[3], p1_disc[3];
    int traversing_player;
    float iteration_weight;
    // Persistent range state: updated at each action, used for features[17-50]
    float opp_range[351];   // opponent range from traversing player's view
    float my_range[351];    // our range from opponent's view

    TraversalFrame stack[MAX_STACK];
    int depth;

    int   waiting;
    float pending_feats[FEATURE_DIM];
    int   pending_valid[MAX_VALID], pending_n_valid, pending_player;

    float ev;
    int   done;

    float adv_feat[MAX_SAMPLES][FEATURE_DIM];
    float adv_val [MAX_SAMPLES][NUM_ACTIONS];
    float adv_mask[MAX_SAMPLES][NUM_ACTIONS];
    int   adv_street[MAX_SAMPLES], n_adv;

    float str_feat [MAX_SAMPLES][FEATURE_DIM];
    float str_val  [MAX_SAMPLES][NUM_ACTIONS];
    float str_mask [MAX_SAMPLES][NUM_ACTIONS];
    int   str_street[MAX_SAMPLES], n_str;
};

// ── Helpers ───────────────────────────────────────────────────────────────────

static inline float terminal_ev(const FullGameState& s, int tp,
                                  const int* p0h, const int* p1h,
                                  const int* comm) {
    if (s.folded_player >= 0)
        return (s.folded_player == tp) ? -(float)s.bets[tp] : (float)s.bets[1-tp];
    int pot = std::min(s.bets[0], s.bets[1]);
    int sd  = evaluate_showdown(p0h, p1h, comm);
    return (tp == 0) ? (float)(sd * pot) : (float)(-sd * pot);
}

static inline void extract_features_full(const FullGameState& s, int cp,
                                           const int* p0h, const int* p1h,
                                           const int* p0d, const int* p1d,
                                           const int* comm, float* out,
                                           const float* opp_range = nullptr,
                                           const float* my_range  = nullptr) {
    const int* my_hand  = (cp == 0) ? p0h : p1h;
    const int* my_disc  = (cp == 0) ? p0d : p1d;
    const int* opp_disc = (cp == 0) ? p1d : p0d;
    int vis_n = std::min(s.street + 2, 5);
    if (s.street == 0) vis_n = 0;
    state_to_features(my_hand, comm, vis_n,
                       s.bets[cp], s.bets[1-cp], s.street, cp==1,
                       my_disc, opp_disc, out,
                       &s.street_bet_counts[0][0],
                       s.history_players, s.history_actions, s.history_len,
                       s.num_actions_this_street,
                       opp_range, my_range);
}

static inline void regret_match(const float* adv, const int* valid, int n, float* strategy) {
    float total = 0.f, best_v = -1e9f; int best_a = valid[0];
    for (int i = 0; i < n; i++) {
        float v = adv[valid[i]];
        if (v > 0) total += v;
        if (v > best_v) { best_v = v; best_a = valid[i]; }
    }
    if (total > 0) { for (int i = 0; i < n; i++) strategy[valid[i]] = std::max(adv[valid[i]], 0.f) / total; }
    else           { for (int i = 0; i < n; i++) strategy[valid[i]] = 1.f / n; }
    (void)best_a;
}

// ── Main advance function ─────────────────────────────────────────────────────

static void advance_game(PostflopGame* g, const float* net_adv, std::mt19937* rng) {
    if (g->done) return;

    if (g->waiting && net_adv) {
        g->waiting = 0;
        TraversalFrame& f = g->stack[g->depth - 1];
        regret_match(net_adv, f.valid, f.n_valid, f.strategy);

        if (g->n_str < MAX_SAMPLES) {
            std::memcpy(g->str_feat[g->n_str], f.features, sizeof(float)*FEATURE_DIM);
            std::memset(g->str_val[g->n_str], 0, sizeof(float)*NUM_ACTIONS);
            std::memset(g->str_mask[g->n_str], 0, sizeof(float)*NUM_ACTIONS);
            for (int i = 0; i < f.n_valid; i++) {
                g->str_val[g->n_str][f.valid[i]]  = f.strategy[f.valid[i]];
                g->str_mask[g->n_str][f.valid[i]] = 1.f;
            }
            g->str_street[g->n_str] = f.state.street;
            g->n_str++;
        }

        if (!f.is_traversing) {
            float r = rng ? std::uniform_real_distribution<float>(0,1)(*rng) : 0.5f;
            float cum = 0.f; int chosen = f.valid[0];
            for (int i = 0; i < f.n_valid; i++) {
                cum += f.strategy[f.valid[i]];
                if (r <= cum) { chosen = f.valid[i]; break; }
            }
            for (int i = 0; i < f.n_valid; i++) if (f.valid[i] == chosen) { std::swap(f.valid[0], f.valid[i]); break; }
            f.n_valid = 1;
            // Opponent acted: update opp_range from traversing player's perspective
            if (chosen >= 0 && f.state.street > 0) {
                int vis_n = std::min(f.state.street + 2, 5);
                range_update_betting(g->opp_range, g->community, vis_n, chosen >= 3);
                _validate_range(g->opp_range, "advance_game:opp_range_after_opp_act");
            }
        }
    }

    while (g->depth > 0 && !g->waiting) {
        TraversalFrame& f = g->stack[g->depth - 1];

        if (f.action_idx >= f.n_valid) {
            float ev;
            if (f.is_traversing) {
                float node_ev = 0.f;
                for (int i = 0; i < f.n_valid; i++) node_ev += f.strategy[f.valid[i]] * f.action_evs[i];
                if (g->n_adv < MAX_SAMPLES) {
                    std::memcpy(g->adv_feat[g->n_adv], f.features, sizeof(float)*FEATURE_DIM);
                    std::memset(g->adv_val[g->n_adv], 0, sizeof(float)*NUM_ACTIONS);
                    std::memset(g->adv_mask[g->n_adv], 0, sizeof(float)*NUM_ACTIONS);
                    for (int i = 0; i < f.n_valid; i++) {
                        g->adv_val[g->n_adv][f.valid[i]]  = f.action_evs[i] - node_ev;
                        g->adv_mask[g->n_adv][f.valid[i]] = 1.f;
                    }
                    g->adv_street[g->n_adv] = f.state.street;
                    g->n_adv++;
                }
                // Restore my_range to pre-frame state after all traversing branches explored
                if (f.state.street > 0)
                    std::memcpy(g->my_range, f.my_range_snap, sizeof(float) * 351);
                ev = node_ev;
            } else {
                ev = f.action_evs[0];
            }
            g->depth--;
            if (g->depth == 0) { g->ev = ev; g->done = 1; return; }
            TraversalFrame& par = g->stack[g->depth - 1];
            par.action_evs[par.action_idx - 1] = ev;
            continue;
        }

        // For traversing nodes: maintain my_range per action branch
        if (f.is_traversing && f.state.street > 0) {
            if (f.action_idx == 0) {
                // First action: snapshot my_range before any update
                _validate_range(g->my_range, "advance_game:my_range_before_snap");
                std::memcpy(f.my_range_snap, g->my_range, sizeof(float) * 351);
            } else {
                // Subsequent actions: restore from snapshot, then update for this action
                std::memcpy(g->my_range, f.my_range_snap, sizeof(float) * 351);
            }
            int vis_n = std::min(f.state.street + 2, 5);
            range_update_betting(g->my_range, g->community, vis_n, f.valid[f.action_idx] >= 3);
            _validate_range(g->my_range, "advance_game:my_range_after_traversing_act");
        }

        int action = f.valid[f.action_idx++];
        FullGameState child = f.state.apply(action);

        if (child.is_terminal) {
            f.action_evs[f.action_idx - 1] = terminal_ev(child, g->traversing_player,
                                                           g->p0_hand, g->p1_hand, g->community);
            continue;
        }

        if (g->depth >= MAX_STACK - 1) { f.action_evs[f.action_idx - 1] = 0.f; continue; }

        TraversalFrame& nf = g->stack[g->depth];
        nf.state = child;
        child.get_valid_actions(nf.valid, nf.n_valid);
        std::memset(nf.action_evs, 0, sizeof(nf.action_evs));
        nf.action_idx = 0;
        std::memset(nf.strategy, 0, sizeof(nf.strategy));
        nf.is_traversing = (child.current_player == g->traversing_player) ? 1 : 0;
        g->depth++;

        extract_features_full(child, child.current_player,
                               g->p0_hand, g->p1_hand,
                               g->p0_disc, g->p1_disc, g->community, nf.features,
                               g->opp_range, g->my_range);
        std::memcpy(g->pending_feats, nf.features, sizeof(float)*FEATURE_DIM);
        std::memcpy(g->pending_valid, nf.valid, sizeof(int)*nf.n_valid);
        g->pending_n_valid = nf.n_valid;
        g->pending_player  = child.current_player;
        g->waiting = 1;
        return;
    }
}

/*
 * C++ traversal engine for Deep CFR — C exports for Python ctypes.
 *
 * Build (Linux/RunPod):
 *   g++ -O3 -shared -fPIC -std=c++17 -fopenmp -o libtraversal.so traversal.cpp range/rangefinder.cpp -lpthread
 * Build (macOS):
 *   clang++ -O3 -shared -fPIC -std=c++17 -o libtraversal.dylib traversal.cpp range/rangefinder.cpp -lpthread
 */

#include <cstring>
#include <random>
#include <vector>
#include "cfr/traversal_engine.h"   // PostflopGame, advance_game (→ cfr/features.h, heuristic/discard.h, core/constants.h)

extern "C" {

// ── Deck & discard ────────────────────────────────────────────────────────────

void deal_game(int* p0_5, int* p1_5, int* community, unsigned int seed) {
    std::mt19937 rng(seed);
    int deck[DECK_SIZE]; shuffle_deck(deck, rng);
    std::copy(deck, deck+5, p0_5); std::copy(deck+5, deck+10, p1_5); std::copy(deck+10, deck+15, community);
}

void c_fast_discard(const int* hand5, const int* board3, int* ki, int* kj,
                     unsigned int seed, float temperature) {
    std::mt19937 rng(seed);
    fast_discard(hand5, board3, *ki, *kj, rng, temperature);
}

int c_evaluate_showdown(const int* p0_hand, const int* p1_hand, const int* community) {
    return evaluate_showdown(p0_hand, p1_hand, community);
}

void c_batch_deal_discard(int n, int* p0h, int* p1h, int* p0d, int* p1d,
                           int* comms, int* p0h5, int* p1h5,
                           unsigned int base_seed, float temperature) {
    #pragma omp parallel for schedule(dynamic) if(n > 10)
    for (int i = 0; i < n; i++) {
        std::mt19937 rng(base_seed + (unsigned)i);
        int deck[DECK_SIZE]; shuffle_deck(deck, rng);
        int* p05=&p0h5[i*5], *p15=&p1h5[i*5], *cm=&comms[i*5];
        std::copy(deck, deck+5, p05); std::copy(deck+5, deck+10, p15); std::copy(deck+10, deck+15, cm);
        int ki0,kj0,ki1,kj1;
        fast_discard(p05, cm, ki0, kj0, rng, temperature);
        p0h[i*2]=p05[ki0]; p0h[i*2+1]=p05[kj0];
        int d=0; for(int j=0;j<5;j++) if(j!=ki0&&j!=kj0) p0d[i*3+d++]=p05[j];
        fast_discard(p15, cm, ki1, kj1, rng, temperature);
        p1h[i*2]=p15[ki1]; p1h[i*2+1]=p15[kj1];
        d=0; for(int j=0;j<5;j++) if(j!=ki1&&j!=kj1) p1d[i*3+d++]=p15[j];
    }
}

// ── Hand category (for Python-side range update) ─────────────────────────────

int c_classify_hand(int c0, int c1, const int* board, int n_board) {
    return classify_hand(c0, c1, board, n_board);
}

void c_blocker_flags(int c0, int c1, const int* board, int n_board, float* out) {
    compute_blocker_flags(c0, c1, board, n_board, out);
}

// ── Per-pending-game raw state (Phase 3 range tracking) ───────────────────────
// Must be called immediately after c_postflop_collect_pending.
// Returns the same count in the same order.
int c_postflop_get_pending_game_info(
    PostflopGame* games, int n_games,
    int* hero_hand_out,  // [cnt * 2]  current player's 2-card hand
    int* community_out,  // [cnt * 5]  full community (fill unused with -1)
    int* my_disc_out,    // [cnt * 3]  current player's discards
    int* opp_disc_out,   // [cnt * 3]  opponent's discards
    int* cp_out,         // [cnt]      current player index (0 or 1)
    int* bet_cp_out,     // [cnt]      current player's chips bet
    int* bet_opp_out,    // [cnt]      opponent's chips bet
    int* game_idx_out    // [cnt]      game index within batch
) {
    int cnt = 0;
    for (int i = 0; i < n_games; i++) {
        PostflopGame& g = games[i];
        if (!g.waiting || g.done) continue;
        int cp = g.pending_player;
        FullGameState& s = g.stack[g.depth > 0 ? g.depth - 1 : 0].state;
        int* hero = (cp == 0) ? g.p0_hand : g.p1_hand;
        int* my_d = (cp == 0) ? g.p0_disc : g.p1_disc;
        int* op_d = (cp == 0) ? g.p1_disc : g.p0_disc;
        for (int j = 0; j < 2; j++) hero_hand_out[cnt*2+j] = hero[j];
        for (int j = 0; j < 5; j++) community_out[cnt*5+j] = g.community[j];
        for (int j = 0; j < 3; j++) my_disc_out[cnt*3+j]   = my_d[j];
        for (int j = 0; j < 3; j++) opp_disc_out[cnt*3+j]  = op_d[j];
        cp_out[cnt]       = cp;
        bet_cp_out[cnt]   = s.bets[cp];
        bet_opp_out[cnt]  = s.bets[1 - cp];
        game_idx_out[cnt] = i;
        cnt++;
    }
    return cnt;
}

// ── Feature extraction ────────────────────────────────────────────────────────

void c_state_features(
    const int* hero_hand2,
    const int* community, int n_comm,
    int my_bet, int opp_bet, int street, int is_bb,
    const int* my_disc, const int* opp_disc,
    float* features_out,
    const int* street_bet_counts_flat,
    const int* history_players,
    const int* history_actions,
    int history_len, int num_acts_this_street
) {
    state_to_features(hero_hand2, community, n_comm,
                       my_bet, opp_bet, street, (bool)is_bb,
                       my_disc, opp_disc, features_out,
                       street_bet_counts_flat,
                       history_players, history_actions, history_len, num_acts_this_street);
}

// ── Batch warmup equity (C++ OpenMP) ─────────────────────────────────────────

void c_batch_warmup_ev(int n,
    const int* p0h5, const int* p1h5,
    const int* my_bets, const int* opp_bets,
    const int* tps, float* evs_out,
    unsigned int base_seed, int n_boards)
{
    #pragma omp parallel for schedule(dynamic) if(n > 10)
    for (int i = 0; i < n; i++) {
        std::mt19937 rng(base_seed + (unsigned)i * 1013);
        const int* p0 = p0h5 + i*5, *p1 = p1h5 + i*5;
        bool dead[DECK_SIZE] = {};
        for(int j=0;j<5;j++){dead[p0[j]]=true;dead[p1[j]]=true;}
        int rem[DECK_SIZE]; int nr=0;
        for(int c=0;c<DECK_SIZE;c++) if(!dead[c]) rem[nr++]=c;
        float wins=0.f; int total=0;
        for(int b=0;b<n_boards&&nr>=5;b++){
            int pool[DECK_SIZE]; std::copy(rem,rem+nr,pool);
            int comm[5];
            for(int k=0;k<5;k++){
                std::uniform_int_distribution<int> d(k,nr-1);
                int r2=d(rng); std::swap(pool[k],pool[r2]);
                comm[k]=pool[k];
            }
            int flop[3]={comm[0],comm[1],comm[2]};
            int ki0,kj0,ki1,kj1;
            fast_discard(p0,flop,ki0,kj0,rng,0.f);
            fast_discard(p1,flop,ki1,kj1,rng,0.f);
            int p0k[2]={p0[ki0],p0[kj0]}, p1k[2]={p1[ki1],p1[kj1]};
            int sd=evaluate_showdown(p0k,p1k,comm);
            if(sd>0) wins+=1.f; else if(sd==0) wins+=0.5f;
            total++;
        }
        float eq=(total>0)?wins/total:0.5f;
        float comm=(float)my_bets[i];
        evs_out[i]=(tps[i]==0)?(2*eq-1)*comm:(1-2*eq)*comm;
    }
}

// ── Discard EV matrix batch ───────────────────────────────────────────────────
// Computes 10×10 ev_matrix[ka][kb] = P(A wins) for N games.
// Replaces N×100 Python evaluate_showdown calls with a single OpenMP C++ batch.

static constexpr int _KP[10][2] = {
    {0,1},{0,2},{0,3},{0,4},{1,2},{1,3},{1,4},{2,3},{2,4},{3,4}
};

void c_compute_discard_ev_matrix_batch(
    int n,
    const int* hand5s_A,   // [n*5]
    const int* hand5s_B,   // [n*5]
    const int* boards3,    // [n*3]
    float* ev_out,          // [n*100] row-major: ev_out[i*100 + ka*10 + kb]
    int n_mc,              // MC samples per pair (0 = exact enum)
    unsigned seed
) {
    #pragma omp parallel for schedule(static) if(n > 2)
    for (int i = 0; i < n; i++) {
        const int* h5A = hand5s_A + i * 5;
        const int* h5B = hand5s_B + i * 5;
        const int* b3  = boards3  + i * 3;
        float*     ev  = ev_out   + i * 100;

        bool in_b3[DECK_SIZE] = {};
        for (int j = 0; j < 3; j++) if (b3[j] >= 0 && b3[j] < DECK_SIZE) in_b3[b3[j]] = true;

        int board5[5] = {b3[0], b3[1], b3[2], -1, -1};
        std::mt19937 rng(seed + (unsigned)i * 1337u);

        for (int ka = 0; ka < 10; ka++) {
            int p0[2] = {h5A[_KP[ka][0]], h5A[_KP[ka][1]]};
            bool in_p0[DECK_SIZE] = {};
            in_p0[p0[0]] = in_p0[p0[1]] = true;

            for (int kb = 0; kb < 10; kb++) {
                int p1[2] = {h5B[_KP[kb][0]], h5B[_KP[kb][1]]};

                int pool[DECK_SIZE]; int n_pool = 0;
                for (int c = 0; c < DECK_SIZE; c++) {
                    if (!in_b3[c] && !in_p0[c] && c != p1[0] && c != p1[1])
                        pool[n_pool++] = c;
                }

                if (n_pool < 2) { ev[ka * 10 + kb] = 0.5f; continue; }

                int n_exact = n_pool * (n_pool - 1) / 2;
                float wins = 0.f;

                if (n_mc == 0 || n_exact <= 91) {
                    for (int t = 0; t < n_pool - 1; t++)
                        for (int r = t + 1; r < n_pool; r++) {
                            board5[3] = pool[t]; board5[4] = pool[r];
                            int sd = evaluate_showdown(p0, p1, board5);
                            wins += (sd > 0) ? 1.f : (sd == 0) ? 0.5f : 0.f;
                        }
                    ev[ka * 10 + kb] = wins / n_exact;
                } else {
                    for (int m = 0; m < n_mc; m++) {
                        int ti = (int)(rng() % (unsigned)n_pool);
                        int ri = (int)(rng() % (unsigned)(n_pool - 1));
                        if (ri >= ti) ri++;
                        board5[3] = pool[ti]; board5[4] = pool[ri];
                        int sd = evaluate_showdown(p0, p1, board5);
                        wins += (sd > 0) ? 1.f : (sd == 0) ? 0.5f : 0.f;
                    }
                    ev[ka * 10 + kb] = wins / n_mc;
                }
            }
        }
    }
}

// ── Discard feature batch builders ───────────────────────────────────────────
// Replace N×10×2 individual c_classify_hand / c_blocker_flags ctypes calls with
// a single C++ call (200K ctypes calls/iter → 1 call).

static constexpr int _KEEP_PAIRS[10][2] = {
    {0,1},{0,2},{0,3},{0,4},{1,2},{1,3},{1,4},{2,3},{2,4},{3,4}
};
static constexpr int _DISCARD_PAIR_DIM = 23;  // cat_oh(17)+hi(1)+lo(1)+blocker(4)

// Compute pair features (23-dim) for all 10 keep-pairs of all N 5-card hands.
// feats_out: [n * 10 * 23] row-major (game-major, then pair-major).
static void _build_pair_feats_for_hand(
    const int* h5, const int* board5, int n_board, float* out)
{
    float bl[4];
    for (int k = 0; k < 10; k++) {
        int c0 = h5[_KEEP_PAIRS[k][0]], c1 = h5[_KEEP_PAIRS[k][1]];
        float* f = out + k * _DISCARD_PAIR_DIM;
        int cat = classify_hand(c0, c1, board5, n_board);
        for (int j = 0; j < 17; j++) f[j] = 0.f; f[cat] = 1.f;
        f[17] = (float)std::max(c0 % NUM_RANKS, c1 % NUM_RANKS) / (NUM_RANKS - 1.f);
        f[18] = (float)std::min(c0 % NUM_RANKS, c1 % NUM_RANKS) / (NUM_RANKS - 1.f);
        compute_blocker_flags(c0, c1, board5, n_board, bl);
        f[19] = bl[0]; f[20] = bl[1]; f[21] = bl[2]; f[22] = bl[3];
    }
}

void c_build_discard_pair_features_batch(
    int n,
    const int* hand5s_A, const int* hand5s_B,  // [n*5] each
    const int* boards3,                          // [n*3] first 3 board cards
    float* feats_A_pair_out,                     // [n*10*23]
    float* feats_B_pair_out                      // [n*10*23]
) {
    #pragma omp parallel for schedule(static) if(n > 4)
    for (int i = 0; i < n; i++) {
        const int* b3 = boards3 + i * 3;
        int board5[5] = {-1,-1,-1,-1,-1}, nb = 0;
        for (int j = 0; j < 3; j++) if (b3[j] >= 0) board5[nb++] = b3[j];
        _build_pair_feats_for_hand(hand5s_A + i*5, board5, nb, feats_A_pair_out + i*10*_DISCARD_PAIR_DIM);
        _build_pair_feats_for_hand(hand5s_B + i*5, board5, nb, feats_B_pair_out + i*10*_DISCARD_PAIR_DIM);
    }
}

// Compute 17-dim opp range category probs for N Player-B states.
// Used to build Player B's context after Player A's discard is known.
void c_opp_cats_narrowed_batch(
    int n,
    const int* hand5s_B,   // [n*5] Player B's 5-card hands (dead cards)
    const int* boards3,    // [n*3] first 3 board cards
    const int* opp_disc3s, // [n*3] Player A's discards (opp from B's view)
    float* cats_out        // [n*17] output
) {
    #pragma omp parallel for schedule(static) if(n > 4)
    for (int i = 0; i < n; i++) {
        const int* h5  = hand5s_B   + i * 5;
        const int* b3  = boards3    + i * 3;
        const int* d3  = opp_disc3s + i * 3;
        float* cats    = cats_out   + i * 17;

        float range[351];
        c_range_init(h5, 5, range);

        int board_cards[3]; int nb = 0;
        for (int j = 0; j < 3; j++) if (b3[j] >= 0) board_cards[nb++] = b3[j];
        if (nb > 0) c_range_remove_cards(range, board_cards, nb);

        bool has_disc = false;
        for (int j = 0; j < 3; j++) if (d3[j] >= 0) { has_disc = true; break; }
        if (has_disc) {
            int board3_arr[3] = {b3[0], b3[1], b3[2]};
            c_range_update_discard(range, d3, board3_arr);
        }

        int board5[5] = {-1,-1,-1,-1,-1};
        for (int j = 0; j < nb; j++) board5[j] = board_cards[j];
        c_range_to_category_probs(range, board5, nb, 0.f, cats);
    }
}

// ── Postflop C++ state machine ────────────────────────────────────────────────

PostflopGame* c_postflop_alloc(int n) { return new PostflopGame[n](); }
void          c_postflop_free(PostflopGame* g) { delete[] g; }

// Batch-compute opp_range[351] and my_range[351] for N games from discard phase.
// Call this AFTER discard decisions are finalized, then pass results to c_postflop_init_one.
void c_compute_postflop_ranges_batch(
    int n, int tp,
    const int* p0_hands,    // [n*2]
    const int* p1_hands,    // [n*2]
    const int* p0_discs,    // [n*3]
    const int* p1_discs,    // [n*3]
    const int* communities, // [n*5]
    float* opp_ranges_out,  // [n*351]
    float* my_ranges_out    // [n*351]
) {
    #pragma omp parallel for schedule(static) if(n > 4)
    for (int i = 0; i < n; i++) {
        const int* tp_hand  = (tp == 0) ? (p0_hands + i*2) : (p1_hands + i*2);
        const int* opp_disc = (tp == 0) ? (p1_discs + i*3) : (p0_discs + i*3);
        const int* tp_disc  = (tp == 0) ? (p0_discs + i*3) : (p1_discs + i*3);
        const int* comm     = communities + i*5;
        float* opp_r = opp_ranges_out + (size_t)i * 351;
        float* my_r  = my_ranges_out  + (size_t)i * 351;

        int n_comm = 0, comm_cards[5];
        for (int j = 0; j < 5; j++) if (comm[j] >= 0) comm_cards[n_comm++] = comm[j];

        // opp_range: dead = tp_hand, remove community, update with opp's observed discards
        int tp_dead[2] = {tp_hand[0], tp_hand[1]};
        c_range_init(tp_dead, 2, opp_r);
        if (n_comm > 0) c_range_remove_cards(opp_r, comm_cards, n_comm);
        bool has_od = false;
        for (int j = 0; j < 3; j++) if (opp_disc[j] >= 0) { has_od = true; break; }
        if (has_od && n_comm >= 3) {
            int b3[3] = {comm[0], comm[1], comm[2]};
            c_range_update_discard(opp_r, opp_disc, b3);
        }

        // my_range: dead = community only, update with tp's observed discards
        if (n_comm > 0) c_range_init(comm_cards, n_comm, my_r);
        else { int nd[1] = {-1}; c_range_init(nd, 0, my_r); }
        bool has_td = false;
        for (int j = 0; j < 3; j++) if (tp_disc[j] >= 0) { has_td = true; break; }
        if (has_td && n_comm >= 3) {
            int b3[3] = {comm[0], comm[1], comm[2]};
            c_range_update_discard(my_r, tp_disc, b3);
        }
    }
}

void c_postflop_init_one(PostflopGame* games, int idx,
    const int* state_flat,
    const int* p0h, const int* p1h,
    const int* p0h5, const int* p1h5,
    const int* community,
    const int* p0d, const int* p1d,
    int tp,
    const float* init_opp_range,  // pre-computed from discard phase (or nullptr)
    const float* init_my_range    // pre-computed from discard phase (or nullptr)
)
{
    PostflopGame& g = games[idx];
    std::memset(&g, 0, sizeof(PostflopGame));
    const int* p = state_flat;
    FullGameState& s0 = g.stack[0].state;
    s0.street=*p++;s0.bets[0]=*p++;s0.bets[1]=*p++;s0.current_player=*p++;
    s0.is_terminal=(bool)*p++;s0.folded_player=*p++;s0.min_raise=*p++;
    s0.last_street_bet=*p++;s0.num_actions_this_street=*p++;s0.preflop_open_override=*p++;
    for(int ss=0;ss<4;ss++){s0.street_bets[ss][0]=*p++;s0.street_bets[ss][1]=*p++;}
    const float* fp=reinterpret_cast<const float*>(p);
    for(int ss=0;ss<4;ss++){s0.street_last_ratios[ss][0]=*fp++;s0.street_last_ratios[ss][1]=*fp++;}
    p=reinterpret_cast<const int*>(fp);
    for(int ss=0;ss<4;ss++){s0.street_bet_counts[ss][0]=*p++;s0.street_bet_counts[ss][1]=*p++;}
    int hlen=*p++; s0.history_len=hlen;
    for(int h=0;h<hlen&&h<MAX_HISTORY;h++) s0.history_players[h]=*p++;
    for(int h=0;h<hlen&&h<MAX_HISTORY;h++) s0.history_actions[h]=*p++;
    s0.get_valid_actions(g.stack[0].valid,g.stack[0].n_valid);
    g.stack[0].action_idx=0;
    g.stack[0].is_traversing=(s0.current_player==tp)?1:0;
    g.depth=1;
    std::copy(p0h,p0h+2,g.p0_hand); std::copy(p1h,p1h+2,g.p1_hand);
    std::copy(p0h5,p0h5+5,g.p0_hand5); std::copy(p1h5,p1h5+5,g.p1_hand5);
    std::copy(community,community+5,g.community);
    std::copy(p0d,p0d+3,g.p0_disc); std::copy(p1d,p1d+3,g.p1_disc);
    g.traversing_player=tp; g.ev=0.f; g.done=0; g.waiting=0; g.n_adv=0; g.n_str=0;

    // ── Initialize persistent ranges (from discard phase if provided) ──────────
    if (init_opp_range != nullptr) {
        std::memcpy(g.opp_range, init_opp_range, sizeof(float) * 351);
    } else {
        const int* tp_hand_p  = (tp == 0) ? p0h : p1h;
        const int* opp_disc_p = (tp == 0) ? p1d : p0d;
        int n_comm = 0, comm_cards[5];
        for (int i = 0; i < 5; i++) if (community[i] >= 0) comm_cards[n_comm++] = community[i];
        int tp_dead[2] = {tp_hand_p[0], tp_hand_p[1]};
        c_range_init(tp_dead, 2, g.opp_range);
        if (n_comm > 0) c_range_remove_cards(g.opp_range, comm_cards, n_comm);
        bool has_od = false; for (int i=0;i<3;i++) if(opp_disc_p[i]>=0){has_od=true;break;}
        if (has_od && n_comm >= 3) {
            int b3[3]={community[0],community[1],community[2]};
            c_range_update_discard(g.opp_range, opp_disc_p, b3);
        }
    }
    _validate_range(g.opp_range, "c_postflop_init_one:opp_range");

    if (init_my_range != nullptr) {
        std::memcpy(g.my_range, init_my_range, sizeof(float) * 351);
    } else {
        const int* tp_disc_p = (tp == 0) ? p0d : p1d;
        int n_comm = 0, comm_cards[5];
        for (int i = 0; i < 5; i++) if (community[i] >= 0) comm_cards[n_comm++] = community[i];
        if (n_comm > 0) c_range_init(comm_cards, n_comm, g.my_range);
        else { int nd[1]={-1}; c_range_init(nd, 0, g.my_range); }
        bool has_td = false; for (int i=0;i<3;i++) if(tp_disc_p[i]>=0){has_td=true;break;}
        if (has_td && n_comm >= 3) {
            int b3[3]={community[0],community[1],community[2]};
            c_range_update_discard(g.my_range, tp_disc_p, b3);
        }
    }
    _validate_range(g.my_range, "c_postflop_init_one:my_range");

    extract_features_full(s0,s0.current_player,g.p0_hand,g.p1_hand,g.p0_disc,g.p1_disc,g.community,g.stack[0].features,g.opp_range,g.my_range);
    std::memcpy(g.pending_feats,g.stack[0].features,sizeof(float)*FEATURE_DIM);
    std::memcpy(g.pending_valid,g.stack[0].valid,sizeof(int)*g.stack[0].n_valid);
    g.pending_n_valid=g.stack[0].n_valid; g.pending_player=s0.current_player; g.waiting=1;
}

int c_postflop_collect_pending(PostflopGame* games, int n,
    float* feats_out, int* valid_out, int* n_valid_out, int* player_out, int* idx_out)
{
    int cnt=0;
    for(int i=0;i<n;i++){
        if(!games[i].waiting||games[i].done) continue;
        std::memcpy(feats_out+cnt*FEATURE_DIM,games[i].pending_feats,sizeof(float)*FEATURE_DIM);
        std::memset(valid_out+cnt*NUM_ACTIONS,0,sizeof(int)*NUM_ACTIONS);
        for(int j=0;j<games[i].pending_n_valid;j++) valid_out[cnt*NUM_ACTIONS+j]=games[i].pending_valid[j];
        n_valid_out[cnt]=games[i].pending_n_valid; player_out[cnt]=games[i].pending_player; idx_out[cnt]=i;
        cnt++;
    }
    return cnt;
}

void c_postflop_resume_batch(PostflopGame* games, const int* game_idxs,
                              const float* net_advs, int n_pending, unsigned int base_seed)
{
    #pragma omp parallel for schedule(dynamic) if(n_pending > 4)
    for(int j=0;j<n_pending;j++){
        int i=game_idxs[j]; std::mt19937 rng(base_seed+(unsigned)i*997);
        advance_game(&games[i], net_advs+j*NUM_ACTIONS, &rng);
    }
}

void c_postflop_collect_samples(PostflopGame* games, int n,
    float* adv_f, float* adv_v, float* adv_m, int* adv_s, int* adv_p, float* adv_i, int* na_out,
    float* str_f, float* str_v, float* str_m, int* str_s, float* str_i, int* ns_out,
    float iteration, int tp, int max_s)
{
    int na=0,ns=0;
    for(int i=0;i<n&&na<max_s;i++){
        auto& g=games[i];
        for(int k=0;k<g.n_adv&&na<max_s;k++,na++){
            std::memcpy(adv_f+na*FEATURE_DIM,g.adv_feat[k],sizeof(float)*FEATURE_DIM);
            std::memcpy(adv_v+na*NUM_ACTIONS,g.adv_val[k], sizeof(float)*NUM_ACTIONS);
            std::memcpy(adv_m+na*NUM_ACTIONS,g.adv_mask[k],sizeof(float)*NUM_ACTIONS);
            adv_s[na]=g.adv_street[k]; adv_p[na]=tp; adv_i[na]=iteration;
        }
    }
    for(int i=0;i<n&&ns<max_s;i++){
        auto& g=games[i];
        for(int k=0;k<g.n_str&&ns<max_s;k++,ns++){
            std::memcpy(str_f+ns*FEATURE_DIM,g.str_feat[k],sizeof(float)*FEATURE_DIM);
            std::memcpy(str_v+ns*NUM_ACTIONS,g.str_val[k], sizeof(float)*NUM_ACTIONS);
            std::memcpy(str_m+ns*NUM_ACTIONS,g.str_mask[k],sizeof(float)*NUM_ACTIONS);
            str_s[ns]=g.str_street[k]; str_i[ns]=iteration;
        }
    }
    *na_out=na; *ns_out=ns;
}

void c_postflop_get_evs(PostflopGame* g, int n, float* evs) { for(int i=0;i<n;i++) evs[i]=g[i].ev; }
int  c_postflop_n_pending(PostflopGame* g, int n) { int c=0; for(int i=0;i<n;i++) if(!g[i].done) c++; return c; }

} // extern "C"

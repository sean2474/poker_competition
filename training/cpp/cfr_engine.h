#pragma once
#include "abstractions.h"
#include <unordered_map>
#include <string>
#include <cstring>
#include <fstream>
#include <numeric>
#include <mutex>
#include <atomic>
#include <thread>

constexpr int MAX_BET = 100;
constexpr int SMALL_BLIND = 1;
constexpr int BIG_BLIND = 2;
constexpr int MAX_ACTIONS = 4;
constexpr int MAX_HISTORY = 32;

// ═══════════════════════════════════════════════
// Action context + valid actions
// ═══════════════════════════════════════════════

enum ActionCtx : uint8_t { CTX_NO_BET = 0, CTX_FACING_BET = 1 };
enum AbsAction : uint8_t {
    A_FOLD=0, A_CALL=1, A_CHECK=2, A_BET_SMALL=3, A_BET_LARGE=4,
    A_RAISE_SMALL=5, A_RAISE_LARGE=6, A_JAM=7
};

// Short codes for history string matching Python
inline char action_short(AbsAction a) {
    switch(a) {
        case A_FOLD: return 'F'; case A_CALL: return 'C'; case A_CHECK: return 'K';
        case A_BET_SMALL: return 'b'; case A_BET_LARGE: return 'B';
        case A_RAISE_SMALL: return 'r'; case A_RAISE_LARGE: return 'R';
        case A_JAM: return 'J';
    }
    return '?';
}

// ═══════════════════════════════════════════════
// Game State for CFR traversal
// ═══════════════════════════════════════════════

struct GameState {
    int street = 0;
    int bets[2] = {SMALL_BLIND, BIG_BLIND};
    int current_player = 0; // SB first preflop
    int min_raise = BIG_BLIND;
    int last_street_bet = 0;
    int last_aggressor = -1;
    char street_history[MAX_HISTORY] = {};
    int hist_len = 0;
    bool is_terminal = false;
    int winner = -2; // -2=none, -1=showdown, 0/1=folded
    int folded_player = -1;

    ActionCtx get_action_ctx() const {
        int to_call = bets[1 - current_player] - bets[current_player];
        if (to_call <= 0) return CTX_NO_BET;
        return CTX_FACING_BET;
    }

    // Returns count of valid actions, fills `out`
    int get_valid_actions(AbsAction out[MAX_ACTIONS]) const {
        int cp = current_player, opp = 1 - cp;
        int to_call = bets[opp] - bets[cp];
        int max_raise = MAX_BET - std::max(bets[0], bets[1]);
        bool can_raise = (max_raise > 0 && min_raise <= max_raise);
        int n = 0;

        if (to_call <= 0) {
            out[n++] = A_CHECK;
            if (can_raise) { out[n++] = A_BET_SMALL; out[n++] = A_BET_LARGE; }
        } else {
            out[n++] = A_FOLD;
            out[n++] = A_CALL;
            if (can_raise) { out[n++] = A_RAISE_SMALL; out[n++] = A_RAISE_LARGE; }
        }
        return n;
    }

    GameState apply(AbsAction action) const {
        GameState s = *this;
        int cp = s.current_player, opp = 1 - cp;
        int max_raise = MAX_BET - std::max(s.bets[0], s.bets[1]);

        s.street_history[s.hist_len++] = action_short(action);

        if (action == A_FOLD) {
            s.is_terminal = true; s.winner = opp; s.folded_player = cp;
            return s;
        }
        if (action == A_CHECK) {
            bool end_street = false;
            if (s.street == 0 && cp == 1) end_street = true;
            else if (s.street >= 1 && cp == 0) end_street = true;
            if (end_street) s._advance_street();
            else s.current_player = opp;
            return s;
        }
        if (action == A_CALL) {
            s.bets[cp] = s.bets[opp];
            if (!(s.street == 0 && cp == 0 && s.bets[cp] == BIG_BLIND))
                s._advance_street();
            else s.current_player = opp;
            return s;
        }

        // Raise/bet types
        int raise_amount;
        int spread = max_raise - s.min_raise;
        if (action == A_BET_LARGE || action == A_RAISE_LARGE) raise_amount = s.min_raise + (int)(spread * 0.70);
        else raise_amount = s.min_raise + (int)(spread * 0.25);

        raise_amount = std::max(s.min_raise, std::min(raise_amount, max_raise));
        s.bets[cp] = s.bets[opp] + raise_amount;
        int raise_so_far = s.bets[opp] - s.last_street_bet;
        int min_raise_nl = raise_so_far + raise_amount;
        s.min_raise = std::min(min_raise_nl, MAX_BET - std::max(s.bets[0], s.bets[1]));
        s.last_aggressor = cp;
        s.current_player = opp;
        return s;
    }

    void _advance_street() {
        street++;
        if (street > 3) { is_terminal = true; winner = -1; }
        else {
            min_raise = BIG_BLIND;
            last_street_bet = bets[0];
            current_player = 1; // BB first post-flop
            hist_len = 0;
            std::memset(street_history, 0, MAX_HISTORY);
            last_aggressor = -1;
        }
    }
};

// ═══════════════════════════════════════════════
// Discard Oracle (fast heuristic + optional MC)
// ═══════════════════════════════════════════════

inline double fast_discard_score(const int keep[2], const int board3[3]) {
    int r0 = card_rank(keep[0]), r1 = card_rank(keep[1]);
    int s0 = card_suit(keep[0]), s1 = card_suit(keep[1]);
    int br[3], bs[3];
    for (int i = 0; i < 3; i++) { br[i] = card_rank(board3[i]); bs[i] = card_suit(board3[i]); }
    int bmax = std::max({br[0], br[1], br[2]});

    double score = 0;

    // Pair with board
    for (int kr : {r0, r1}) {
        for (int i = 0; i < 3; i++) {
            if (kr == br[i]) {
                if (kr == bmax) score += 3.0;
                else if (kr == std::min({br[0],br[1],br[2]})) score += 1.0;
                else score += 2.0;
                break;
            }
        }
    }

    // Pocket pair
    if (r0 == r1) {
        score += 2.0;
        if (r0 > bmax) score += 2.0;
        for (int i = 0; i < 3; i++) if (r0 == br[i]) score += 4.0;
    }

    // High cards
    score += (r0 + r1) / (2.0 * (NUM_RANKS - 1));
    if (r0 == ACE_RANK || r1 == ACE_RANK) score += 0.5;

    // Suited
    if (s0 == s1) {
        score += 1.5;
        int match = 0;
        for (int i = 0; i < 3; i++) if (bs[i] == s0) match++;
        if (match >= 2) score += 3.0;
        else if (match >= 1) score += 1.0;
    }

    // Connectivity
    int gap = std::abs(r0 - r1);
    if (gap <= 1) score += 1.0;
    else if (gap == 2) score += 0.5;

    // Straight outs with board
    bool rs[NUM_RANKS] = {};
    for (int i = 0; i < 3; i++) rs[br[i]] = true;
    rs[r0] = true; rs[r1] = true;
    score += straight_draw_outs(rs) * 0.5;

    // Blocker
    int freq[NUM_RANKS] = {};
    for (int i = 0; i < 3; i++) freq[br[i]]++;
    for (int r : {r0, r1}) if (freq[r] >= 2) score += 1.5;

    return score;
}

inline double mc_equity(const int keep[2], const int board3[3], const int* dead, int ndead,
                        int num_sims = 60) {
    bool used[DECK_SIZE] = {};
    used[keep[0]] = used[keep[1]] = true;
    for (int i = 0; i < 3; i++) used[board3[i]] = true;
    for (int i = 0; i < ndead; i++) used[dead[i]] = true;

    int remaining[DECK_SIZE]; int nr = 0;
    for (int i = 0; i < DECK_SIZE; i++) if (!used[i]) remaining[nr++] = i;

    if (nr < 4) return 0.5;

    auto& rng = get_rng();
    int wins = 0, ties = 0, total = 0;
    for (int s = 0; s < num_sims; s++) {
        // Fisher-Yates partial shuffle for 4 cards
        for (int i = 0; i < 4; i++) {
            int j = i + rng() % (nr - i);
            std::swap(remaining[i], remaining[j]);
        }
        int full_board[5] = {board3[0], board3[1], board3[2], remaining[0], remaining[1]};
        int opp[2] = {remaining[2], remaining[3]};
        int r = eval::compare_hands(keep, opp, full_board);
        if (r == -1) wins++;
        else if (r == 0) ties++;
        total++;
    }
    return (total > 0) ? (wins + 0.5 * ties) / total : 0.5;
}

// Returns (keep_i, keep_j) indices into hand_5
inline std::pair<int,int> choose_discard(const int hand5[5], const int board3[3],
                                          const int* opp_disc, int nopp_disc,
                                          int top_k = 3, int mc_sims = 60) {
    // Stage A: fast score all 10
    struct Candidate { double score; int i, j; int keep[2]; int disc[3]; };
    Candidate cands[10];
    int nc = 0;
    for (int i = 0; i < 5; i++) for (int j = i+1; j < 5; j++) {
        cands[nc].i = i; cands[nc].j = j;
        cands[nc].keep[0] = hand5[i]; cands[nc].keep[1] = hand5[j];
        int di = 0;
        for (int k = 0; k < 5; k++) if (k != i && k != j) cands[nc].disc[di++] = hand5[k];
        cands[nc].score = fast_discard_score(cands[nc].keep, board3);
        nc++;
    }

    // Sort by fast score descending
    std::sort(cands, cands + nc, [](const Candidate& a, const Candidate& b) { return a.score > b.score; });

    // Stage B: MC on top_k
    int best_i = cands[0].i, best_j = cands[0].j;
    double best_score = -1;
    int tk = std::min(top_k, nc);
    for (int c = 0; c < tk; c++) {
        // Dead = opp discards + our discards
        int dead[6]; int nd = 0;
        for (int d = 0; d < nopp_disc; d++) dead[nd++] = opp_disc[d];
        for (int d = 0; d < 3; d++) dead[nd++] = cands[c].disc[d];
        double eq = mc_equity(cands[c].keep, board3, dead, nd, mc_sims);
        double composite = 0.7 * eq * 10.0 + 0.3 * cands[c].score;
        if (composite > best_score) {
            best_score = composite;
            best_i = cands[c].i;
            best_j = cands[c].j;
        }
    }
    return {best_i, best_j};
}

// ═══════════════════════════════════════════════
// Infoset Key (compact 64-bit hash)
// ═══════════════════════════════════════════════

// FNV-1a hash
inline uint64_t fnv_hash(const uint8_t* data, size_t len) {
    uint64_t h = 14695981039346656037ULL;
    for (size_t i = 0; i < len; i++) {
        h ^= data[i];
        h *= 1099511628211ULL;
    }
    return h;
}

struct InfoKeyData {
    uint8_t buf[64];
    int len = 0;
    void add_u8(uint8_t v) { buf[len++] = v; }
    void add_u16(uint16_t v) { std::memcpy(buf + len, &v, 2); len += 2; }
    void add_u32(uint32_t v) { std::memcpy(buf + len, &v, 4); len += 4; }
    void add_str(const char* s, int n) { std::memcpy(buf + len, s, n); len += n; }
    uint64_t hash() const { return fnv_hash(buf, len); }
};

inline uint64_t make_preflop_key(const int hand5[5], bool is_bb, const char* hist, int hist_len) {
    auto canon = canonicalize_5(hand5, 5);
    int pos = is_bb ? 1 : 0;
    int line = line_bucket_fn(hist);

    InfoKeyData k;
    k.add_u8(0); // PF marker
    k.add_u8(pos);
    k.add_u8(line);
    for (int i = 0; i < 5; i++) k.add_u8(canon.cards[i]);
    return k.hash();
}

inline uint64_t make_postdiscard_key(int street, const int hand2[2], const int* community,
                                      const int opp_disc[3], bool is_bb,
                                      bool hero_agg, bool villain_agg,
                                      const char* hist, int hist_len,
                                      int my_bet, int opp_bet, const int* dead, int ndead,
                                      ActionCtx actx, int nactions) {
    int pos = is_bb ? 1 : 0;
    int init = hero_agg ? 1 : (villain_agg ? 2 : 0);
    int line = line_bucket_fn(hist);
    int press = pressure_bucket(my_bet, opp_bet);
    int board_bkt = board_bucket_for_street(community, street);
    int board3[3] = {community[0], community[1], community[2]};
    int opp_disc_bkt = opp_discard_bucket_fn(opp_disc, board3);
    int hand_bkt = hand_bucket_for_street(hand2, community, street);

    InfoKeyData k;
    k.add_u8(street);
    k.add_u8(pos);
    k.add_u8(init);
    k.add_u8(line);
    k.add_u8(press);
    k.add_u16(board_bkt);
    k.add_u8(opp_disc_bkt);
    k.add_u8(hand_bkt);
    k.add_u8(actx);
    k.add_u8(nactions);
    return k.hash();
}

// ═══════════════════════════════════════════════
// CFR Node
// ═══════════════════════════════════════════════

struct CFRNode {
    double regret_sum[MAX_ACTIONS] = {};
    double strategy_sum[MAX_ACTIONS] = {};
    uint8_t num_actions = 0;
    uint8_t action_type = 0; // index into ACTION_LISTS

    void init(int n, uint8_t atype) { num_actions = n; action_type = atype; }

    void get_strategy(double out[MAX_ACTIONS], double reach_weight, int t) {
        double pos[MAX_ACTIONS], total = 0;
        for (int i = 0; i < num_actions; i++) {
            pos[i] = std::max(regret_sum[i], 0.0);
            total += pos[i];
        }
        if (total > 0) {
            for (int i = 0; i < num_actions; i++) out[i] = pos[i] / total;
        } else {
            for (int i = 0; i < num_actions; i++) out[i] = 1.0 / num_actions;
        }
        // Linear CFR: weight strategy_sum by iteration number
        double weight = reach_weight * std::max(t, 1);
        for (int i = 0; i < num_actions; i++) strategy_sum[i] += weight * out[i];
    }

    void get_average_strategy(double out[MAX_ACTIONS]) const {
        double total = 0;
        for (int i = 0; i < num_actions; i++) total += strategy_sum[i];
        if (total > 0) {
            for (int i = 0; i < num_actions; i++) out[i] = strategy_sum[i] / total;
        } else {
            for (int i = 0; i < num_actions; i++) out[i] = 1.0 / num_actions;
        }
    }
};

// Action list type encoding (matches Python convert_to_python.py)
inline uint8_t get_action_list_type(const AbsAction* acts, int n) {
    if (n == 2 && acts[0] == A_FOLD && acts[1] == A_CALL) return 0;
    if (n == 4 && acts[0] == A_FOLD && acts[1] == A_CALL) return 1;  // FOLD,CALL,RAISE_S,RAISE_L
    if (n == 1 && acts[0] == A_CHECK) return 2;
    if (n == 3 && acts[0] == A_CHECK) return 3;  // CHECK,BET_S,BET_L
    return 255;
}

// ═══════════════════════════════════════════════
// CFR Trainer
// ═══════════════════════════════════════════════

struct CFRTrainer {
    std::unordered_map<uint64_t, CFRNode> nodes;
    std::mutex node_mutex;  // protects map insert only
    std::atomic<int> iterations{0};

    CFRNode& get_node(uint64_t key, int nactions, uint8_t atype) {
        // Fast path: node already exists (no lock needed for read in practice,
        // but we use lock for insert safety)
        {
            auto it = nodes.find(key);
            if (it != nodes.end()) return it->second;
        }
        std::lock_guard<std::mutex> lock(node_mutex);
        // Double-check after acquiring lock
        auto it = nodes.find(key);
        if (it != nodes.end()) return it->second;
        CFRNode& node = nodes[key];
        node.init(nactions, atype);
        return node;
    }

    uint64_t make_key(const GameState& state, int cp,
                      const int* p_hand, const int* p_hand5,
                      const int* community, const int* opp_disc,
                      const int* p_disc, ActionCtx actx, int nactions) {
        bool is_bb = (cp == 1);
        if (state.street == 0) {
            return make_preflop_key(p_hand5, is_bb, state.street_history, state.hist_len);
        }
        bool hero_agg = (state.last_aggressor == cp);
        bool villain_agg = (state.last_aggressor == (1 - cp));
        int dead[6]; int nd = 0;
        for (int i = 0; i < 3; i++) dead[nd++] = p_disc[i];
        for (int i = 0; i < 3; i++) dead[nd++] = opp_disc[i];
        return make_postdiscard_key(
            state.street, p_hand, community, opp_disc,
            is_bb, hero_agg, villain_agg,
            state.street_history, state.hist_len,
            state.bets[cp], state.bets[1-cp], dead, nd, actx, nactions
        );
    }

    // Returns (util_p0, util_p1)
    std::pair<double,double> cfr(const GameState& state,
                                  const int p0_hand[2], const int p1_hand[2],
                                  const int p0_hand5[5], const int p1_hand5[5],
                                  const int community[5],
                                  const int p0_disc[3], const int p1_disc[3],
                                  double reach_0, double reach_1) {
        if (state.is_terminal) {
            int pot = std::min(state.bets[0], state.bets[1]);
            if (state.folded_player >= 0) {
                return (state.folded_player == 0) ?
                    std::make_pair(-(double)pot, (double)pot) :
                    std::make_pair((double)pot, -(double)pot);
            }
            int r = eval::compare_hands(p0_hand, p1_hand, community);
            if (r == -1) return {(double)pot, -(double)pot};
            if (r == 1) return {-(double)pot, (double)pot};
            return {0.0, 0.0};
        }

        int cp = state.current_player;
        AbsAction actions[MAX_ACTIONS];
        int n = state.get_valid_actions(actions);
        ActionCtx actx = state.get_action_ctx();
        uint8_t atype = get_action_list_type(actions, n);

        const int* my_hand = (cp == 0) ? p0_hand : p1_hand;
        const int* my_hand5 = (cp == 0) ? p0_hand5 : p1_hand5;
        const int* opp_disc = (cp == 0) ? p1_disc : p0_disc;
        const int* my_disc = (cp == 0) ? p0_disc : p1_disc;

        uint64_t key = make_key(state, cp, my_hand, my_hand5, community, opp_disc, my_disc, actx, n);
        CFRNode& node = get_node(key, n, atype);

        double reach = (cp == 0) ? reach_0 : reach_1;
        double strategy[MAX_ACTIONS];
        node.get_strategy(strategy, reach, iterations);

        double action_utils[MAX_ACTIONS] = {};
        double node_util[2] = {};

        for (int i = 0; i < n; i++) {
            GameState ns = state.apply(actions[i]);
            auto [u0, u1] = (cp == 0) ?
                cfr(ns, p0_hand, p1_hand, p0_hand5, p1_hand5, community, p0_disc, p1_disc,
                    reach_0 * strategy[i], reach_1) :
                cfr(ns, p0_hand, p1_hand, p0_hand5, p1_hand5, community, p0_disc, p1_disc,
                    reach_0, reach_1 * strategy[i]);
            action_utils[i] = (cp == 0) ? u0 : u1;
            node_util[0] += strategy[i] * u0;
            node_util[1] += strategy[i] * u1;
        }

        // CFR+ regret update
        double my_util = node_util[cp];
        double opp_reach = (cp == 0) ? reach_1 : reach_0;
        for (int i = 0; i < n; i++) {
            node.regret_sum[i] = std::max(
                node.regret_sum[i] + opp_reach * (action_utils[i] - my_util), 0.0);
        }

        return {node_util[0], node_util[1]};
    }

    void train_one() {
        int deck[DECK_SIZE];
        shuffle_deck(deck);
        int p0_5[5], p1_5[5], community[5];
        std::copy(deck, deck+5, p0_5);
        std::copy(deck+5, deck+10, p1_5);
        std::copy(deck+10, deck+15, community);

        int board3[3] = {community[0], community[1], community[2]};

        auto [ki0, kj0] = choose_discard(p0_5, board3, nullptr, 0, 3, 60);
        int p0_hand[2] = {p0_5[ki0], p0_5[kj0]};
        int p0_disc[3]; { int d=0; for(int i=0;i<5;i++) if(i!=ki0&&i!=kj0) p0_disc[d++]=p0_5[i]; }

        auto [ki1, kj1] = choose_discard(p1_5, board3, p0_disc, 3, 3, 60);
        int p1_hand[2] = {p1_5[ki1], p1_5[kj1]};
        int p1_disc[3]; { int d=0; for(int i=0;i<5;i++) if(i!=ki1&&i!=kj1) p1_disc[d++]=p1_5[i]; }

        GameState state;
        cfr(state, p0_hand, p1_hand, p0_5, p1_5, community, p0_disc, p1_disc, 1.0, 1.0);
        iterations.fetch_add(1, std::memory_order_relaxed);
    }

    void train_parallel(int num_iterations, int num_threads) {
        // Pre-warm: run single-threaded first to populate most nodes
        // This reduces lock contention during parallel phase
        int warmup = std::min(1000, num_iterations / 10);
        for (int i = 0; i < warmup; i++) train_one();
        int remaining = num_iterations - warmup;

        std::vector<std::thread> threads;
        int per_thread = remaining / num_threads;
        int extra = remaining % num_threads;

        for (int t = 0; t < num_threads; t++) {
            int count = per_thread + (t < extra ? 1 : 0);
            threads.emplace_back([this, count]() {
                for (int i = 0; i < count; i++) {
                    train_one();
                }
            });
        }
        for (auto& t : threads) t.join();
    }

    // Save binary format for Python conversion
    void save_binary(const char* path) {
        std::ofstream f(path, std::ios::binary);
        // Header: iterations(4), num_nodes(4)
        uint32_t ni = iterations, nn = nodes.size();
        f.write((char*)&ni, 4);
        f.write((char*)&nn, 4);
        // Each node: key(8), atype(1), nactions(1), avg_strategy[4](32), confidence(8)
        for (auto& [key, node] : nodes) {
            f.write((char*)&key, 8);
            f.write((char*)&node.action_type, 1);
            f.write((char*)&node.num_actions, 1);
            double avg[MAX_ACTIONS];
            node.get_average_strategy(avg);
            f.write((char*)avg, MAX_ACTIONS * 8);
            // Confidence = sum of strategy_sum (proxy for visit count)
            double conf = 0;
            for (int i = 0; i < node.num_actions; i++) conf += node.strategy_sum[i];
            f.write((char*)&conf, 8);
        }
        f.close();
    }

    // Save checkpoint (full regret + strategy_sum)
    void save_checkpoint(const char* path) {
        std::ofstream f(path, std::ios::binary);
        uint32_t ni = iterations, nn = nodes.size();
        f.write((char*)&ni, 4);
        f.write((char*)&nn, 4);
        for (auto& [key, node] : nodes) {
            f.write((char*)&key, 8);
            f.write((char*)&node.action_type, 1);
            f.write((char*)&node.num_actions, 1);
            f.write((char*)node.regret_sum, MAX_ACTIONS * 8);
            f.write((char*)node.strategy_sum, MAX_ACTIONS * 8);
        }
        f.close();
    }

    void load_checkpoint(const char* path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return;
        uint32_t ni, nn;
        f.read((char*)&ni, 4); f.read((char*)&nn, 4);
        iterations = ni;
        nodes.clear();
        nodes.reserve(nn);
        for (uint32_t i = 0; i < nn; i++) {
            uint64_t key; uint8_t atype, nact;
            f.read((char*)&key, 8);
            f.read((char*)&atype, 1);
            f.read((char*)&nact, 1);
            CFRNode& node = nodes[key];
            node.init(nact, atype);
            f.read((char*)node.regret_sum, MAX_ACTIONS * 8);
            f.read((char*)node.strategy_sum, MAX_ACTIONS * 8);
        }
        f.close();
    }
};

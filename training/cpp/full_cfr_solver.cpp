/*
 * Post-discard Full CFR solver as a Python C extension.
 * Compiled to .so, called from Python at runtime.
 *
 * Enumerates all possible opponent hands, traverses full multi-street
 * game tree (flop→turn→river), updates all strategies simultaneously.
 *
 * Designed for 27-card deck post-discard:
 *   - ~105 possible opp hands
 *   - Turn: ~14 runout cards  
 *   - River: ~13 runout cards
 *   - Game tree: ~20 nodes per street × 3 streets = ~60 nodes
 *
 * 100 iterations ≈ few hundred ms in C++
 */

// This will be compiled as a standalone executable that reads input from stdin
// and writes the chosen action to stdout. Python calls it via subprocess.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <random>
#include <chrono>

// ─── Card basics (same as card_utils.h) ───
constexpr int NUM_RANKS = 9;
constexpr int NUM_SUITS = 3;
constexpr int DECK_SIZE = 27;
constexpr int MAX_BET = 100;

inline int card_rank(int c) { return c % NUM_RANKS; }
inline int card_suit(int c) { return c / NUM_RANKS; }

// ─── Hand evaluation (simplified 7-card evaluator) ───
namespace eval {
enum HandCat { STRAIGHT_FLUSH=0, FULL_HOUSE=1, FLUSH=2, STRAIGHT=3,
               THREE_OF_KIND=4, TWO_PAIR=5, ONE_PAIR=6, HIGH_CARD=7 };

struct HandRank {
    int cat, pri, sec, kick[5];
    bool operator<(const HandRank& o) const {
        if (cat != o.cat) return cat < o.cat;
        if (pri != o.pri) return pri > o.pri;
        if (sec != o.sec) return sec > o.sec;
        for (int i = 0; i < 5; i++) if (kick[i] != o.kick[i]) return kick[i] > o.kick[i];
        return false;
    }
};

inline int check_straight(const int r[5]) {
    bool is = true;
    for (int i = 1; i < 5; i++) if (r[i] != r[i-1]+1) { is = false; break; }
    if (is) return r[4];
    if (r[0]==0 && r[1]==1 && r[2]==2 && r[3]==3 && r[4]==8) return 3;
    return -1;
}

inline HandRank evaluate_5(const int c[5]) {
    int ranks[5], suits[5];
    for (int i = 0; i < 5; i++) { ranks[i] = card_rank(c[i]); suits[i] = card_suit(c[i]); }
    int sr[5]; std::copy(ranks, ranks+5, sr); std::sort(sr, sr+5);
    bool fl = (suits[0]==suits[1] && suits[1]==suits[2] && suits[2]==suits[3] && suits[3]==suits[4]);
    int sh = check_straight(sr);
    int freq[NUM_RANKS] = {};
    for (int i = 0; i < 5; i++) freq[ranks[i]]++;
    int trips=-1, pairs[2]={-1,-1}, pc=0;
    for (int r = NUM_RANKS-1; r >= 0; r--) {
        if (freq[r]==3) trips=r;
        else if (freq[r]==2 && pc<2) pairs[pc++]=r;
    }
    HandRank res; res.pri=res.sec=-1; std::fill(res.kick, res.kick+5, -1);
    if (fl && sh>=0) { res.cat=STRAIGHT_FLUSH; res.pri=sh; }
    else if (trips>=0 && pc>=1) { res.cat=FULL_HOUSE; res.pri=trips; res.sec=pairs[0]; }
    else if (fl) { res.cat=FLUSH; for (int i=0;i<5;i++) res.kick[i]=sr[4-i]; }
    else if (sh>=0) { res.cat=STRAIGHT; res.pri=sh; }
    else if (trips>=0) { res.cat=THREE_OF_KIND; res.pri=trips; int k=0; for(int r=NUM_RANKS-1;r>=0;r--) if(freq[r]==1) res.kick[k++]=r; }
    else if (pc>=2) { res.cat=TWO_PAIR; res.pri=std::max(pairs[0],pairs[1]); res.sec=std::min(pairs[0],pairs[1]); for(int r=NUM_RANKS-1;r>=0;r--) if(freq[r]==1){res.kick[0]=r;break;} }
    else if (pc==1) { res.cat=ONE_PAIR; res.pri=pairs[0]; int k=0; for(int r=NUM_RANKS-1;r>=0;r--) if(freq[r]==1) res.kick[k++]=r; }
    else { res.cat=HIGH_CARD; for(int i=0;i<5;i++) res.kick[i]=sr[4-i]; }
    return res;
}

inline HandRank evaluate_7(const int h[2], const int b[5]) {
    int all[7] = {h[0],h[1],b[0],b[1],b[2],b[3],b[4]};
    HandRank best; best.cat = HIGH_CARD+1;
    for (int i=0;i<7;i++) for (int j=i+1;j<7;j++) {
        int five[5]; int k=0;
        for (int x=0;x<7;x++) if(x!=i&&x!=j) five[k++]=all[x];
        HandRank r = evaluate_5(five);
        if (r < best) best = r;
    }
    return best;
}

// +1 hero wins, -1 hero loses, 0 tie
inline int compare(const int h0[2], const int h1[2], const int b[5]) {
    HandRank r0 = evaluate_7(h0, b), r1 = evaluate_7(h1, b);
    if (r0 < r1) return 1;
    if (r1 < r0) return -1;
    return 0;
}
} // namespace eval

// ─── CFR Node ───
struct Node {
    float regret[4] = {};
    float strat_sum[4] = {};
    int n_actions = 0;

    void get_strategy(float out[4], int t) {
        float pos[4], total = 0;
        for (int i = 0; i < n_actions; i++) { pos[i] = std::max(regret[i], 0.0f); total += pos[i]; }
        if (total > 0) for (int i = 0; i < n_actions; i++) out[i] = pos[i] / total;
        else for (int i = 0; i < n_actions; i++) out[i] = 1.0f / n_actions;
        for (int i = 0; i < n_actions; i++) strat_sum[i] += out[i] * std::max(t, 1);
    }
};

// ─── Full CFR Solver ───
struct FullCFR {
    int hero[2], community[5], n_community;
    int remaining[DECK_SIZE]; int n_remaining;
    
    struct OppHand { int cards[2]; };
    std::vector<OppHand> opp_hands;
    
    // Flat array: 2 * 4 * 101 * 101 * 256 = ~21M entries
    // But most bet values are sparse. Use compact encoding:
    // Quantize bets to 16 levels: 0,1,2,3,4,5,6,8,10,15,20,30,50,75,100 → 4 bits
    // Key: hero(1) × street(2) × my_bet_q(4) × opp_bet_q(4) × oi(8) = 19 bits = 512K entries
    static constexpr int TABLE_SIZE = 1 << 19; // 512K × 20 bytes = 10MB
    Node* nodes;
    
    // Pre-computed showdown result per opp hand for current board
    int8_t sd_cache[256];
    
    std::mt19937 rng;
    int node_count = 0;
    
    FullCFR() : rng(std::random_device{}()) {
        nodes = new Node[TABLE_SIZE]();  // zero-initialized on heap
    }
    ~FullCFR() { delete[] nodes; }
    
    static int quantize_bet(int bet) {
        if (bet <= 0) return 0;
        if (bet <= 2) return bet;
        if (bet <= 5) return 3;
        if (bet <= 10) return 4;
        if (bet <= 15) return 5;
        if (bet <= 20) return 6;
        if (bet <= 30) return 7;
        if (bet <= 40) return 8;
        if (bet <= 50) return 9;
        if (bet <= 60) return 10;
        if (bet <= 70) return 11;
        if (bet <= 80) return 12;
        if (bet <= 90) return 13;
        if (bet <= 95) return 14;
        return 15;
    }
    
    inline int make_key(bool is_hero, int street, int my_bet, int opp_bet, int oi) {
        int k = (is_hero ? 1 : 0);
        k = (k << 2) | (street & 3);
        k = (k << 4) | quantize_bet(my_bet);
        k = (k << 4) | quantize_bet(opp_bet);
        k = (k << 8) | (oi & 255);
        return k & (TABLE_SIZE - 1);
    }
    
    inline Node& get_node(int key, int n_actions) {
        Node& n = nodes[key];
        if (n.n_actions == 0) { n.n_actions = n_actions; node_count++; }
        return n;
    }
    
    int bet_sizes[2]; int n_bets;
    
    void compute_bet_sizes(int pot, int max_raise, int min_raise) {
        n_bets = 0;
        if (max_raise <= 0 || max_raise < min_raise) return;
        int s1 = std::max(min_raise, std::min((int)(pot * 0.5), max_raise));
        bet_sizes[n_bets++] = s1;
        int s2 = std::max(min_raise, std::min((int)(pot * 1.0), max_raise));
        if (s2 != s1) bet_sizes[n_bets++] = s2;
    }
    
    // Recursive CFR traversal
    float cfr(int oi, const int board5[5], int my_bet, int opp_bet,
              int min_raise, int street, bool is_hero_turn, int t) {
        int pot = my_bet + opp_bet;
        int max_raise = MAX_BET - std::max(my_bet, opp_bet);
        int to_call = is_hero_turn ? (opp_bet - my_bet) : (my_bet - opp_bet);
        
        // Terminal: past river — use pre-computed showdown
        if (street > 3) {
            return sd_cache[oi] * std::min(my_bet, opp_bet);
        }
        
        compute_bet_sizes(pot, max_raise, min_raise);
        bool can_raise = (max_raise > 0 && min_raise <= max_raise);
        
        int n_actions;
        if (to_call > 0) n_actions = 2 + (can_raise ? n_bets : 0); // FOLD, CALL, RAISE...
        else n_actions = 1 + (can_raise ? n_bets : 0); // CHECK, BET...
        
        uint64_t key = make_key(is_hero_turn, street, my_bet, opp_bet, oi);
        Node& node = get_node(key, n_actions);
        float strat[4];
        node.get_strategy(strat, t);
        
        float action_ev[4] = {};
        
        if (is_hero_turn) {
            if (to_call > 0) {
                // FOLD
                action_ev[0] = -my_bet;
                // CALL → next street or showdown
                int new_my = opp_bet;
                if (street == 3) {
                    action_ev[1] = sd_cache[oi] * new_my;
                } else {
                    action_ev[1] = cfr(oi, board5, new_my, opp_bet, 2, street+1, false, t);
                }
                // RAISE
                for (int si = 0; si < n_bets && can_raise; si++) {
                    int new_my2 = opp_bet + bet_sizes[si];
                    action_ev[2+si] = cfr(oi, board5, new_my2, opp_bet,
                                           std::min(bet_sizes[si], max_raise), street, false, t);
                }
            } else {
                // CHECK → opp turn
                action_ev[0] = cfr(oi, board5, my_bet, opp_bet, min_raise, street, false, t);
                // BET
                for (int si = 0; si < n_bets && can_raise; si++) {
                    int new_my = my_bet + bet_sizes[si];
                    action_ev[1+si] = cfr(oi, board5, new_my, opp_bet,
                                           std::min(bet_sizes[si], max_raise), street, false, t);
                }
            }
            
            float ev = 0;
            for (int i = 0; i < n_actions; i++) ev += strat[i] * action_ev[i];
            for (int i = 0; i < n_actions; i++)
                node.regret[i] = std::max(node.regret[i] + (action_ev[i] - ev), 0.0f);
            return ev;
            
        } else {
            // Opponent turn
            if (to_call > 0) {
                // OPP FOLD
                action_ev[0] = opp_bet; // hero wins
                // OPP CALL
                int new_opp = my_bet;
                if (street == 3) {
                    action_ev[1] = sd_cache[oi] * my_bet;
                } else {
                    action_ev[1] = cfr(oi, board5, my_bet, new_opp, 2, street+1, true, t);
                }
                // OPP RAISE
                for (int si = 0; si < n_bets && can_raise; si++) {
                    int new_opp2 = my_bet + bet_sizes[si];
                    action_ev[2+si] = cfr(oi, board5, my_bet, new_opp2,
                                           std::min(bet_sizes[si], max_raise), street, true, t);
                }
            } else {
                // OPP CHECK → next street
                if (street == 3) {
                    action_ev[0] = sd_cache[oi] * std::min(my_bet, opp_bet);
                } else {
                    action_ev[0] = cfr(oi, board5, my_bet, opp_bet, 2, street+1, true, t);
                }
                // OPP BET
                for (int si = 0; si < n_bets && can_raise; si++) {
                    int new_opp = opp_bet + bet_sizes[si];
                    action_ev[1+si] = cfr(oi, board5, my_bet, new_opp,
                                           std::min(bet_sizes[si], max_raise), street, true, t);
                }
            }
            
            // Opp minimizes hero EV
            float opp_ev[4];
            for (int i = 0; i < n_actions; i++) opp_ev[i] = -action_ev[i];
            float avg = 0;
            for (int i = 0; i < n_actions; i++) avg += strat[i] * opp_ev[i];
            for (int i = 0; i < n_actions; i++)
                node.regret[i] = std::max(node.regret[i] + (opp_ev[i] - avg), 0.0f);
            
            float hero_ev = 0;
            for (int i = 0; i < n_actions; i++) hero_ev += strat[i] * action_ev[i];
            return hero_ev;
        }
    }
    
    // Main solve: read from command line args, output action
    void solve(int my_bet, int opp_bet, int min_raise, int street, int num_iters) {
        // Enumerate board runouts
        for (int t = 0; t < num_iters; t++) {
            // Sample remaining board cards
            int board5[5];
            std::copy(community, community + n_community, board5);
            int board_need = 5 - n_community;
            
            if (board_need > 0) {
                // Shuffle remaining and pick
                std::vector<int> deck(remaining, remaining + n_remaining);
                for (int i = 0; i < board_need; i++) {
                    int j = i + rng() % (deck.size() - i);
                    std::swap(deck[i], deck[j]);
                    board5[n_community + i] = deck[i];
                }
            }
            
            // Pre-compute showdown results for this board
            for (int oi = 0; oi < (int)opp_hands.size(); oi++) {
                bool overlap = false;
                for (int i = n_community; i < 5; i++) {
                    if (board5[i] == opp_hands[oi].cards[0] || board5[i] == opp_hands[oi].cards[1]) {
                        overlap = true; break;
                    }
                }
                if (overlap) { sd_cache[oi] = 0; continue; }
                sd_cache[oi] = eval::compare(hero, opp_hands[oi].cards, board5);
            }
            
            for (int oi = 0; oi < (int)opp_hands.size(); oi++) {
                if (sd_cache[oi] == 0 && opp_hands[oi].cards[0] != opp_hands[oi].cards[1]) {
                    // Check if it was overlap (sd_cache=0 could also be a tie)
                    bool overlap = false;
                    for (int i = n_community; i < 5; i++) {
                        if (board5[i] == opp_hands[oi].cards[0] || board5[i] == opp_hands[oi].cards[1]) {
                            overlap = true; break;
                        }
                    }
                    if (overlap) continue;
                }
                cfr(oi, board5, my_bet, opp_bet, min_raise, street, true, t);
            }
        }
    }
    
    void get_root_strategy(int my_bet, int opp_bet, int street,
                            float avg_strat[4], int& n_actions_out) {
        int pot = my_bet + opp_bet;
        int max_raise = MAX_BET - std::max(my_bet, opp_bet);
        int to_call = opp_bet - my_bet;
        compute_bet_sizes(pot, max_raise, 2);
        bool can_raise = (max_raise > 0);
        
        if (to_call > 0) n_actions_out = 2 + (can_raise ? n_bets : 0);
        else n_actions_out = 1 + (can_raise ? n_bets : 0);
        
        std::fill(avg_strat, avg_strat + 4, 0.0f);
        int count = 0;
        for (int oi = 0; oi < (int)opp_hands.size(); oi++) {
            int key = make_key(true, street, my_bet, opp_bet, oi);
            Node& node = nodes[key];
            if (node.n_actions == 0) continue;
            float total = 0;
            for (int i = 0; i < n_actions_out; i++) total += node.strat_sum[i];
            if (total > 0) {
                for (int i = 0; i < n_actions_out; i++) avg_strat[i] += node.strat_sum[i] / total;
            } else {
                for (int i = 0; i < n_actions_out; i++) avg_strat[i] += 1.0f / n_actions_out;
            }
            count++;
        }
        if (count > 0) {
            for (int i = 0; i < n_actions_out; i++) avg_strat[i] /= count;
        }
    }
};

// ─── Main: reads game state from args, outputs action ───
int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: full_cfr_solve hero0 hero1 comm0..commN my_bet opp_bet min_raise street dead0..deadN num_iters\n");
        return 1;
    }
    
    // Parse args: hero0 hero1 [community...] my_bet opp_bet min_raise street [dead...] num_iters
    // Format: hero0 hero1 n_comm comm... my_bet opp_bet min_raise street n_dead dead... num_iters
    int idx = 1;
    FullCFR solver;
    
    solver.hero[0] = atoi(argv[idx++]);
    solver.hero[1] = atoi(argv[idx++]);
    
    int n_comm = atoi(argv[idx++]);
    solver.n_community = n_comm;
    for (int i = 0; i < n_comm; i++) solver.community[i] = atoi(argv[idx++]);
    
    int my_bet = atoi(argv[idx++]);
    int opp_bet = atoi(argv[idx++]);
    int min_raise = atoi(argv[idx++]);
    int street = atoi(argv[idx++]);
    
    int n_dead = atoi(argv[idx++]);
    bool dead[DECK_SIZE] = {};
    dead[solver.hero[0]] = dead[solver.hero[1]] = true;
    for (int i = 0; i < n_comm; i++) dead[solver.community[i]] = true;
    for (int i = 0; i < n_dead; i++) { int c = atoi(argv[idx++]); dead[c] = true; }
    
    int num_iters = atoi(argv[idx++]);
    
    // Build remaining cards and opp hands
    solver.n_remaining = 0;
    for (int c = 0; c < DECK_SIZE; c++) {
        if (!dead[c]) solver.remaining[solver.n_remaining++] = c;
    }
    
    // Enumerate opp hands
    for (int i = 0; i < solver.n_remaining; i++) {
        for (int j = i+1; j < solver.n_remaining; j++) {
            FullCFR::OppHand oh;
            oh.cards[0] = solver.remaining[i];
            oh.cards[1] = solver.remaining[j];
            solver.opp_hands.push_back(oh);
        }
    }
    
    auto t0 = std::chrono::steady_clock::now();
    
    solver.solve(my_bet, opp_bet, min_raise, street, num_iters);
    
    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    
    // Get root strategy
    float avg[4]; int n_actions;
    solver.get_root_strategy(my_bet, opp_bet, street, avg, n_actions);
    
    // Determine bet sizes for output
    int pot = my_bet + opp_bet;
    int max_raise = MAX_BET - std::max(my_bet, opp_bet);
    solver.compute_bet_sizes(pot, max_raise, min_raise);
    
    int to_call = opp_bet - my_bet;
    
    // Sample action from strategy
    float r = (float)rand() / RAND_MAX;
    float cumul = 0;
    int chosen = 0;
    for (int i = 0; i < n_actions; i++) {
        cumul += avg[i];
        if (r < cumul) { chosen = i; break; }
        chosen = i;
    }
    
    // Output: action_type amount elapsed_ms strategy
    if (to_call > 0) {
        // FOLD=0, CALL=1, RAISE_sz=2+
        if (chosen == 0) printf("0 0");  // FOLD
        else if (chosen == 1) printf("3 0");  // CALL
        else printf("1 %d", solver.bet_sizes[chosen-2]);  // RAISE
    } else {
        // CHECK=0, BET_sz=1+
        if (chosen == 0) printf("2 0");  // CHECK
        else printf("1 %d", solver.bet_sizes[chosen-1]);  // BET
    }
    
    printf(" %.0f", elapsed * 1000);
    
    // Print strategy for debugging
    fprintf(stderr, "opp_hands=%d nodes=%d time=%.0fms strat=[", 
            (int)solver.opp_hands.size(), solver.node_count, elapsed*1000);
    for (int i = 0; i < n_actions; i++) fprintf(stderr, "%.2f%s", avg[i], i<n_actions-1?",":"");
    fprintf(stderr, "]\n");
    
    return 0;
}

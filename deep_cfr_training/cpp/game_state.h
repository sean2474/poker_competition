#pragma once
#include <vector>
#include <array>
#include <algorithm>
#include <cstring>
#include <random>
#include <cmath>

// ═══════════════════════════════════════════════
// Constants matching Python game_env.py
// ═══════════════════════════════════════════════

constexpr int MAX_BET = 100;
constexpr int SMALL_BLIND = 1;
constexpr int BIG_BLIND = 2;
constexpr int NUM_RANKS = 9;   // 2-9,A
constexpr int NUM_SUITS = 3;   // d,h,s
constexpr int DECK_SIZE = 27;
constexpr int NUM_ACTIONS = 8;
constexpr int FEATURE_DIM = 119;   // 93 base (C++) + 26 extra
constexpr int BASE_FEATURE_DIM = 93;

// Actions
constexpr int A_FOLD = 0;
constexpr int A_CALL = 1;
constexpr int A_CHECK = 2;
constexpr int A_BET_SMALL = 3;
constexpr int A_BET_LARGE = 4;
constexpr int A_RAISE_SMALL = 5;
constexpr int A_RAISE_LARGE = 6;
constexpr int A_BET_POT = 7;

inline int card_rank(int c) { return c % NUM_RANKS; }
inline int card_suit(int c) { return c / NUM_RANKS; }

// ═══════════════════════════════════════════════
// GameState
// ═══════════════════════════════════════════════

struct GameState {
    int street = 0;
    int bets[2] = {SMALL_BLIND, BIG_BLIND};
    int current_player = 0;  // SB first preflop
    bool is_terminal = false;
    int folded_player = -1;
    int min_raise = BIG_BLIND;
    int num_actions_this_street = 0;
    int street_bets[4][2] = {};  // [street][player] max raise amount (0-100)

    void get_valid_actions(int* actions, int& n) const {
        n = 0;
        int cp = current_player;
        int opp = 1 - cp;
        int to_call = bets[opp] - bets[cp];
        int max_raise = MAX_BET - std::max(bets[0], bets[1]);
        // Preflop: allow all-in even if < min_raise (incomplete raise is ok)
        bool can_raise = max_raise > 0 && (street > 0 ? min_raise <= max_raise : true);

        if (to_call > 0) {
            actions[n++] = A_FOLD;
            actions[n++] = A_CALL;
            if (can_raise) {
                if (street == 0) {
                    actions[n++] = A_RAISE_SMALL;  // single preflop raise size
                } else {
                    actions[n++] = A_RAISE_SMALL;
                    actions[n++] = A_RAISE_LARGE;
                }
            }
        } else {
            actions[n++] = A_CHECK;
            if (can_raise) {
                if (street == 0) {
                    actions[n++] = A_BET_SMALL;  // single preflop open size
                } else {
                    int pot = bets[0] + bets[1];
                    // Postflop: pot-relative sizing 33% / 75% / 100%
                    int small_amt = std::max(min_raise, std::min(pot * 33 / 100, max_raise));
                    int large_amt = std::max(min_raise, std::min(pot * 75 / 100, max_raise));
                    int pot_amt   = std::max(min_raise, std::min(pot,             max_raise));
                    int thresh = std::max(2, pot / 20);
                    actions[n++] = A_BET_SMALL;
                    if (std::abs(pot_amt - large_amt) > thresh)
                        actions[n++] = A_BET_POT;
                    if (std::abs(large_amt - small_amt) > thresh)
                        actions[n++] = A_BET_LARGE;
                }
            }
        }
    }

    GameState apply(int action) const {
        GameState s = *this;
        int cp = s.current_player;
        int opp = 1 - cp;
        int max_raise = MAX_BET - std::max(s.bets[0], s.bets[1]);
        int st = s.street;

        if (action == A_FOLD) {
            s.is_terminal = true;
            s.folded_player = cp;
            return s;
        }

        if (action == A_CHECK) {
            s.num_actions_this_street++;
            if (s.num_actions_this_street >= 2 && s.bets[0] == s.bets[1]) {
                s.advance_street();
            } else {
                s.current_player = opp;
            }
            return s;
        }

        if (action == A_CALL) {
            s.bets[cp] = s.bets[opp];
            s.num_actions_this_street++;
            if (!(s.street == 0 && cp == 0 && s.bets[cp] == BIG_BLIND)) {
                s.advance_street();
            } else {
                s.current_player = opp;
            }
            return s;
        }

        // Raise/bet
        s.num_actions_this_street++;
        int pot = s.bets[0] + s.bets[1];
        int mn = s.min_raise;
        int raise_amt;
        if (st == 0) {
            // Preflop tiered sizing (mn = min_raise = previous raise amount)
            if (mn <= BIG_BLIND)                        // open
                raise_amt = std::max(mn, std::min(3 * BIG_BLIND, max_raise));      // 6
            else if (mn <= 3 * BIG_BLIND)               // 3-bet
                raise_amt = std::max(mn, std::min(3 * mn, max_raise));             // 18
            else                                        // 4-bet+: raise to MAX_BET (50BB per-hand cap)
                raise_amt = max_raise;
        } else {
            // Postflop: pot-relative sizing
            if (action == A_BET_SMALL || action == A_RAISE_SMALL)
                raise_amt = std::max(mn, std::min(pot * 33 / 100, max_raise));
            else if (action == A_BET_POT)
                raise_amt = std::max(mn, std::min(pot, max_raise));
            else
                raise_amt = std::max(mn, std::min(pot * 75 / 100, max_raise));
        }
        // Preflop: allow all-in even if < standard size (incomplete raise)
        if (st == 0)
            raise_amt = std::min(raise_amt, max_raise);
        else
            raise_amt = std::max(s.min_raise, std::min(raise_amt, max_raise));
        if (st < 4) s.street_bets[st][cp] = std::max(s.street_bets[st][cp], raise_amt);
        s.bets[cp] = s.bets[opp] + raise_amt;
        s.min_raise = std::max(raise_amt, s.min_raise);
        s.current_player = opp;
        return s;
    }

    void advance_street() {
        if (street >= 3) {
            is_terminal = true;
        } else {
            street++;
            // Post-flop: BB (player 1) acts first
            current_player = (street >= 1) ? 1 : 0;
            min_raise = BIG_BLIND;
            num_actions_this_street = 0;
        }
    }
};

// ═══════════════════════════════════════════════
// FullGameState — all fields needed for traversal
// ═══════════════════════════════════════════════

constexpr int MAX_HISTORY = 30;

struct FullGameState {
    int  street                  = 0;
    int  bets[2]                 = {SMALL_BLIND, BIG_BLIND};
    int  current_player          = 0;
    bool is_terminal             = false;
    int  folded_player           = -1;
    int  min_raise               = BIG_BLIND;
    int  last_street_bet         = 0;
    int  num_actions_this_street = 0;
    int  preflop_open_override   = -1;   // -1 = use default 2.5bb

    int   street_bets[4][2]        = {};
    float street_last_ratios[4][2] = {};
    int   street_bet_counts[4][2]  = {};

    int history_players[MAX_HISTORY] = {};
    int history_actions[MAX_HISTORY] = {};
    int history_len                  = 0;

    void get_valid_actions(int* actions, int& n) const {
        n = 0;
        int cp = current_player, opp = 1 - cp;
        int to_call  = bets[opp] - bets[cp];
        int max_r    = MAX_BET - std::max(bets[0], bets[1]);
        bool can_r   = max_r > 0 && (street == 0 || min_raise <= max_r);
        if (to_call > 0) {
            actions[n++] = A_FOLD; actions[n++] = A_CALL;
            if (can_r) {
                actions[n++] = A_RAISE_SMALL;
                if (street > 0) actions[n++] = A_RAISE_LARGE;
            }
        } else {
            actions[n++] = A_CHECK;
            if (can_r) {
                if (street == 0) {
                    actions[n++] = A_BET_SMALL;
                } else {
                    int pot = bets[0]+bets[1], thresh = std::max(2, pot/20);
                    int sa  = std::max(min_raise, std::min(pot*33/100, max_r));
                    int la  = std::max(min_raise, std::min(pot*75/100, max_r));
                    int pa  = std::max(min_raise, std::min(pot,        max_r));
                    actions[n++] = A_BET_SMALL;
                    if (std::abs(pa-la) > thresh) actions[n++] = A_BET_POT;
                    if (std::abs(la-sa) > thresh) actions[n++] = A_BET_LARGE;
                }
            }
        }
    }

    FullGameState apply(int action) const {
        FullGameState s = *this;
        int cp = s.current_player, opp = 1 - cp;
        int max_r = MAX_BET - std::max(s.bets[0], s.bets[1]);
        // record in history
        if (s.history_len < MAX_HISTORY) {
            s.history_players[s.history_len] = cp;
            s.history_actions[s.history_len] = action;
            s.history_len++;
        }
        s.num_actions_this_street++;
        if (action == A_FOLD) { s.is_terminal = true; s.folded_player = cp; return s; }
        if (action == A_CHECK) {
            if (s.num_actions_this_street >= 2 && s.bets[0]==s.bets[1]) s.next_street();
            else s.current_player = opp;
            return s;
        }
        if (action == A_CALL) {
            s.bets[cp] = s.bets[opp];
            if (!(s.street==0 && cp==0 && s.bets[cp]==BIG_BLIND)) s.next_street();
            else s.current_player = opp;
            return s;
        }
        // raise/bet
        int pot = s.bets[0]+s.bets[1], mn = s.min_raise, raise_amt;
        int open_amt = (s.preflop_open_override > 0)
                       ? s.preflop_open_override
                       : (int)std::round(1.5f * BIG_BLIND);   // 2.5bb default
        if (s.street == 0) {
            if      (mn <= BIG_BLIND)  raise_amt = std::max(mn, std::min(open_amt, max_r));
            else if (mn <= open_amt)   raise_amt = std::max(mn, std::min(3*mn,     max_r));
            else                       raise_amt = max_r;
            raise_amt = std::min(raise_amt, max_r);
        } else {
            if (action==A_BET_SMALL||action==A_RAISE_SMALL) raise_amt=std::max(mn,std::min(pot*33/100,max_r));
            else if (action==A_BET_POT)                      raise_amt=std::max(mn,std::min(pot,       max_r));
            else                                             raise_amt=std::max(mn,std::min(pot*75/100,max_r));
            raise_amt = std::max(mn, std::min(raise_amt, max_r));
        }
        float pot_before = (float)(s.bets[0]+s.bets[1]);
        float ratio = (pot_before > 0) ? raise_amt / pot_before : 0.f;
        s.street_bets[s.street][cp]        = std::max(s.street_bets[s.street][cp], raise_amt);
        s.street_last_ratios[s.street][cp] = ratio;
        s.street_bet_counts[s.street][cp]++;
        s.bets[cp]   = s.bets[opp] + raise_amt;
        s.min_raise  = std::max(raise_amt, s.min_raise);
        s.current_player = opp;
        return s;
    }

private:
    void next_street() {
        if (street >= 3) { is_terminal = true; }
        else {
            street++;
            current_player = (street >= 1) ? 1 : 0;
            last_street_bet = std::max(bets[0], bets[1]);
            min_raise = BIG_BLIND;
            num_actions_this_street = 0;
        }
    }
};


// ═══════════════════════════════════════════════
// Card utilities
// ═══════════════════════════════════════════════

inline void shuffle_deck(int* deck, std::mt19937& rng) {
    for (int i = 0; i < DECK_SIZE; i++) deck[i] = i;
    for (int i = DECK_SIZE - 1; i > 0; i--) {
        std::uniform_int_distribution<int> dist(0, i);
        std::swap(deck[i], deck[dist(rng)]);
    }
}

// ═══════════════════════════════════════════════
// Feature extraction (matches Python exactly)
// ═══════════════════════════════════════════════

inline void card_features(int card, float* out) {
    if (card < 0) {
        out[0] = out[1] = out[2] = out[3] = 0.0f;
        return;
    }
    out[0] = (float)card_rank(card) / (NUM_RANKS - 1);
    int s = card_suit(card);
    out[1] = (s == 0) ? 1.0f : 0.0f;
    out[2] = (s == 1) ? 1.0f : 0.0f;
    out[3] = (s == 2) ? 1.0f : 0.0f;
}

inline void hand_strength_features(const int* hand2, const int* community, int n_comm, float* out) {
    // Fast deterministic features (no MC)
    if (hand2[0] < 0 || hand2[1] < 0) {
        for (int i = 0; i < 6; i++) out[i] = 0.0f;
        return;
    }
    int r0 = card_rank(hand2[0]), r1 = card_rank(hand2[1]);
    int s0 = card_suit(hand2[0]), s1 = card_suit(hand2[1]);

    // 1. High card
    out[0] = (float)std::max(r0, r1) / (NUM_RANKS - 1);
    // 2. Pocket pair
    out[1] = (r0 == r1) ? 1.0f : 0.0f;
    // 3. Suited
    out[2] = (s0 == s1) ? 1.0f : 0.0f;

    // 4. Flush draw
    out[3] = 0.0f;
    if (n_comm > 0) {
        int suit_counts[3] = {};
        suit_counts[s0]++; suit_counts[s1]++;
        for (int i = 0; i < n_comm; i++)
            if (community[i] >= 0) suit_counts[card_suit(community[i])]++;
        int mx = *std::max_element(suit_counts, suit_counts + 3);
        out[3] = (mx >= 4) ? 1.0f : (mx >= 3) ? 0.5f : 0.0f;
    }

    // 5. Connectedness
    int gap = std::abs(r0 - r1);
    out[4] = (gap <= 1) ? 1.0f : (gap <= 3) ? 0.5f : 0.0f;

    // 6. Board hit
    out[5] = 0.0f;
    if (n_comm > 0) {
        bool hit0 = false, hit1 = false;
        int max_board = -1;
        for (int i = 0; i < n_comm; i++) {
            if (community[i] >= 0) {
                int br = card_rank(community[i]);
                if (br == r0) hit0 = true;
                if (br == r1) hit1 = true;
                max_board = std::max(max_board, br);
            }
        }
        if (hit0 && hit1) out[5] = 1.0f;
        else if (hit0 || hit1) {
            out[5] = 0.5f;
            if (std::max(r0, r1) == max_board) out[5] = 0.75f;
        }
    }
}

inline void opp_range_features(const int* opp_disc, const int* community, int n_comm, float* out) {
    bool has_disc = false;
    for (int i = 0; i < 3; i++) if (opp_disc[i] >= 0) has_disc = true;

    if (!has_disc) {
        out[0] = out[1] = out[2] = 0.5f;
        out[3] = out[4] = out[5] = 0.0f;
        return;
    }

    int disc_ranks[3], disc_suits[3];
    int nd = 0;
    for (int i = 0; i < 3; i++) {
        if (opp_disc[i] >= 0) {
            disc_ranks[nd] = card_rank(opp_disc[i]);
            disc_suits[nd] = card_suit(opp_disc[i]);
            nd++;
        }
    }

    // 1. Avg discarded rank
    float sum_r = 0; for (int i = 0; i < nd; i++) sum_r += disc_ranks[i];
    float avg = sum_r / nd / (NUM_RANKS - 1);
    out[0] = avg;

    // 2. Max discarded rank
    int mx = *std::max_element(disc_ranks, disc_ranks + nd);
    out[1] = (float)mx / (NUM_RANKS - 1);

    // 3. Discarded pair
    bool has_pair = false;
    for (int i = 0; i < nd && !has_pair; i++)
        for (int j = i+1; j < nd; j++)
            if (disc_ranks[i] == disc_ranks[j]) { has_pair = true; break; }
    out[2] = has_pair ? 1.0f : 0.0f;

    // 4. Suit concentration
    int sc[3] = {};
    for (int i = 0; i < nd; i++) sc[disc_suits[i]]++;
    out[3] = (float)*std::max_element(sc, sc+3) / 3.0f;

    // 5. Board suit match
    out[4] = 0.0f;
    if (n_comm > 0) {
        int board_suits[3] = {};
        for (int i = 0; i < n_comm; i++)
            if (community[i] >= 0) board_suits[card_suit(community[i])]++;
        int dom = (int)(std::max_element(board_suits, board_suits+3) - board_suits);
        int match = 0;
        for (int i = 0; i < nd; i++) if (disc_suits[i] == dom) match++;
        out[4] = (float)match / 3.0f;
    }

    // 6. Kept rank estimate
    out[5] = 1.0f - avg;
}

// Build full 119-dim feature vector
// Permutation invariant: hero hand and flop cards are sorted before encoding
inline void state_to_features(
    const int* hero_hand2, const int* hero_hand5,
    const int* community, int n_comm,
    int my_bet, int opp_bet, int street, bool is_bb,
    const int* my_disc, const int* opp_disc,
    bool use_hand5,
    float* features,
    const int street_bets[4][2] = nullptr,        // [street][player] max chip raise (fallback)
    const float* street_last_ratios = nullptr,     // [4][2] flat: LAST bet/pot per street/player
    const int*   street_bet_counts  = nullptr,     // [4][2] flat: raise count per street/player
    const int*   history_players    = nullptr,     // [history_len]
    const int*   history_actions    = nullptr,     // [history_len]
    int          history_len        = 0,
    int          num_acts_this_street = 0
) {
    // Zero-init entire 119-dim buffer
    for (int i = 0; i < FEATURE_DIM; i++) features[i] = 0.0f;
    int idx = 0;

    // Hero hand (20 floats) — sorted for permutation invariance
    if (use_hand5 && hero_hand5) {
        // Preflop: sort all 5 cards
        int h5[5]; std::copy(hero_hand5, hero_hand5+5, h5);
        std::sort(h5, h5+5);
        for (int i = 0; i < 5; i++) card_features(h5[i], &features[idx + i*4]);
    } else {
        // Post-discard: sort 2-card hand
        int h2[2] = {hero_hand2[0], hero_hand2[1]};
        if (h2[0] > h2[1]) std::swap(h2[0], h2[1]);
        card_features(h2[0], &features[idx]);
        card_features(h2[1], &features[idx+4]);
        for (int i = 8; i < 20; i++) features[idx+i] = 0.0f;
    }
    idx += 20;

    // Community (20 floats) — flop sorted for permutation invariance
    // Layout: [flop_0, flop_1, flop_2, turn, river]
    // Sort flop (3 cards) but keep turn/river in temporal order
    int sorted_comm[5] = {-1, -1, -1, -1, -1};
    if (n_comm >= 3) {
        int flop[3] = {community[0], community[1], community[2]};
        std::sort(flop, flop + 3);
        sorted_comm[0] = flop[0]; sorted_comm[1] = flop[1]; sorted_comm[2] = flop[2];
        if (n_comm >= 4) sorted_comm[3] = community[3];
        if (n_comm >= 5) sorted_comm[4] = community[4];
    } else {
        for (int i = 0; i < n_comm; i++) sorted_comm[i] = community[i];
    }
    for (int i = 0; i < 5; i++) {
        if (i < n_comm && sorted_comm[i] >= 0)
            card_features(sorted_comm[i], &features[idx + i*4]);
        else
            for (int j = 0; j < 4; j++) features[idx + i*4 + j] = 0.0f;
    }
    idx += 20;

    // My discards (12 floats)
    for (int i = 0; i < 3; i++) {
        if (my_disc && my_disc[i] >= 0)
            card_features(my_disc[i], &features[idx + i*4]);
        else
            for (int j = 0; j < 4; j++) features[idx + i*4 + j] = 0.0f;
    }
    idx += 12;

    // Opp discards (12 floats)
    for (int i = 0; i < 3; i++) {
        if (opp_disc && opp_disc[i] >= 0)
            card_features(opp_disc[i], &features[idx + i*4]);
        else
            for (int j = 0; j < 4; j++) features[idx + i*4 + j] = 0.0f;
    }
    idx += 12;

    // Street one-hot (4)
    for (int s = 0; s < 4; s++) features[idx++] = (street == s) ? 1.0f : 0.0f;

    // Position (1)
    features[idx++] = is_bb ? 1.0f : 0.0f;

    // Bet info (4)
    int pot = my_bet + opp_bet;
    features[idx++] = (float)my_bet / MAX_BET;
    features[idx++] = (float)opp_bet / MAX_BET;
    features[idx++] = (float)pot / (2 * MAX_BET);
    features[idx++] = (float)std::max(opp_bet - my_bet, 0) / MAX_BET;

    // Hand strength (6)
    int vis_comm[5];
    int vis_n = std::min(n_comm, (street == 0) ? 0 : (street == 1) ? 3 : (street == 2) ? 4 : 5);
    for (int i = 0; i < vis_n; i++) vis_comm[i] = community[i];

    if (street > 0 && hero_hand2[0] >= 0) {
        hand_strength_features(hero_hand2, vis_comm, vis_n, &features[idx]);
    } else if (use_hand5 && hero_hand5) {
        // Preflop basic features
        int ranks[5];
        for (int i = 0; i < 5; i++) ranks[i] = card_rank(hero_hand5[i]);
        std::sort(ranks, ranks+5, std::greater<int>());
        features[idx] = (float)ranks[0] / (NUM_RANKS - 1);
        bool hp = false;
        for (int i = 0; i < 5 && !hp; i++)
            for (int j = i+1; j < 5; j++)
                if (ranks[i] == ranks[j]) { hp = true; break; }
        features[idx+1] = hp ? 1.0f : 0.0f;
        int suits[5];
        for (int i = 0; i < 5; i++) suits[i] = card_suit(hero_hand5[i]);
        int max_sc = 0;
        for (int s = 0; s < 3; s++) {
            int cnt = 0;
            for (int i = 0; i < 5; i++) if (suits[i] == s) cnt++;
            max_sc = std::max(max_sc, cnt);
        }
        features[idx+2] = (float)max_sc / 5.0f;
        features[idx+3] = features[idx+4] = features[idx+5] = 0.0f;
    } else {
        for (int i = 0; i < 6; i++) features[idx+i] = 0.5f;
    }
    idx += 6;

    // Betting history (8): LAST bet/pot ratio per player per street
    // Overridden by street_last_ratios if provided, else falls back to /MAX_BET
    for (int s = 0; s < 4; s++) {
        int hp = is_bb ? 1 : 0;
        float my_r = 0.0f, op_r = 0.0f;
        if (street_last_ratios) {
            my_r = std::min(street_last_ratios[s*2 + hp],     4.0f);
            op_r = std::min(street_last_ratios[s*2 + 1 - hp], 4.0f);
        } else if (street_bets) {
            my_r = std::min((float)street_bets[s][hp]     / (float)MAX_BET, 1.0f);
            op_r = std::min((float)street_bets[s][1 - hp] / (float)MAX_BET, 1.0f);
        }
        features[idx++] = my_r;
        features[idx++] = op_r;
    }

    // Opp range (6)
    int opp_d[3] = {-1, -1, -1};
    if (opp_disc) { opp_d[0] = opp_disc[0]; opp_d[1] = opp_disc[1]; opp_d[2] = opp_disc[2]; }
    opp_range_features(opp_d, vis_comm, vis_n, &features[idx]);
    idx += 6;
    // idx == 93

    // ── Extra 26 dims ─────────────────────────────────────────────────────
    int hp = is_bb ? 1 : 0;
    int to_call = std::max(opp_bet - my_bet, 0);

    // [93-94] Initiative: hero/villain last aggressor
    if (history_players && history_actions && history_len > 0) {
        for (int i = history_len - 1; i >= 0; i--) {
            int act = history_actions[i];
            if (act >= 3) {  // aggressive action
                features[idx]     = (history_players[i] == hp) ? 1.0f : 0.0f;
                features[idx + 1] = (history_players[i] != hp) ? 1.0f : 0.0f;
                break;
            }
        }
    }
    idx += 2;

    // [95-96] Action context
    features[idx++] = (to_call > 0) ? 1.0f : 0.0f;
    features[idx++] = (to_call == 0) ? 1.0f : 0.0f;

    // [97-100] Line class
    {
        int bets_this = 0;
        int start = history_len - num_acts_this_street;
        if (start < 0) start = 0;
        if (history_actions) {
            for (int i = start; i < history_len; i++)
                if (history_actions[i] >= 3) bets_this++;
        }
        if      (to_call == 0 && bets_this == 0) features[idx]   = 1.0f;  // checked_to
        else if (to_call > 0  && bets_this == 1) features[idx+1] = 1.0f;  // facing_lead
        else if (to_call > 0  && bets_this >= 2) features[idx+2] = 1.0f;  // facing_raise
        else if (to_call == 0 && bets_this >= 1) features[idx+3] = 1.0f;  // raised_pot
        idx += 4;
    }

    // [101-105] Board texture
    {
        int board_r[5], board_s[5];
        int sc[3] = {};
        bool paired = false;
        bool seen_r[9] = {};
        int min_r2 = 8, max_r2 = 0;
        for (int i = 0; i < n_comm; i++) {
            int c = community[i];
            if (c < 0) continue;
            int r = card_rank(c), s = card_suit(c);
            board_r[i] = r; board_s[i] = s;
            sc[s]++;
            if (seen_r[r]) paired = true;
            seen_r[r] = true;
            min_r2 = std::min(min_r2, r); max_r2 = std::max(max_r2, r);
        }
        int max_sc = *std::max_element(sc, sc+3);
        features[idx]   = paired ? 1.0f : 0.0f;
        features[idx+1] = (n_comm >= 3 && max_sc == n_comm) ? 1.0f : 0.0f;  // monotone
        features[idx+2] = (max_sc >= 2) ? 1.0f : 0.0f;                      // two-suited
        if (n_comm >= 3) features[idx+3] = ((max_r2 - min_r2) <= 4) ? 1.0f : 0.0f; // connected
        if (n_comm >= 4) {  // scare card
            int prev_sc[3] = {};
            for (int i = 0; i < n_comm-1; i++) prev_sc[board_s[i]]++;
            if (prev_sc[board_s[n_comm-1]] >= 2) features[idx+4] = 1.0f;
        }
        idx += 5;
    }

    // [106-110] Bet ratios (pot-relative)
    {
        int pot = my_bet + opp_bet;
        float sp = std::max((float)pot, 1.0f);
        int max_r2 = std::max(MAX_BET - std::max(my_bet, opp_bet), 0);
        features[idx++] = std::min((float)to_call  / sp, 4.0f);
        features[idx++] = std::min((float)opp_bet   / sp, 4.0f);
        features[idx++] = std::min((float)my_bet    / sp, 4.0f);
        features[idx++] = std::min((float)max_r2    / sp, 4.0f);
        features[idx++] = max_r2 / 100.0f;
    }

    // [111-118] Bet counts per player per street
    if (street_bet_counts) {
        for (int s = 0; s < 4; s++) {
            features[idx++] = std::min(street_bet_counts[s*2 + hp]     / 4.0f, 1.0f);
            features[idx++] = std::min(street_bet_counts[s*2 + 1 - hp] / 4.0f, 1.0f);
        }
    } else {
        for (int i = 0; i < 8; i++) features[idx++] = 0.0f;
    }
    // idx == 119
}

// ═══════════════════════════════════════════════
// Fast heuristic discard (matching Python)
// ═══════════════════════════════════════════════

// Keep pairs: C(5,2) = 10 combinations
constexpr int KEEP_PAIRS[10][2] = {
    {0,1},{0,2},{0,3},{0,4},{1,2},{1,3},{1,4},{2,3},{2,4},{3,4}
};

inline float fast_score(const int* keep, const int* board3) {
    int r0 = card_rank(keep[0]), r1 = card_rank(keep[1]);
    int s0 = card_suit(keep[0]), s1 = card_suit(keep[1]);
    float score = 0;

    // Pair bonus
    if (r0 == r1) score += 10.0f;

    // High card
    score += std::max(r0, r1) * 0.5f;

    // Suited bonus
    if (s0 == s1) score += 3.0f;

    // Board connection
    for (int i = 0; i < 3; i++) {
        if (board3[i] < 0) continue;
        int br = card_rank(board3[i]);
        if (br == r0 || br == r1) score += 5.0f;  // board pair
        if (std::abs(br - r0) <= 1 || std::abs(br - r1) <= 1) score += 1.0f;
        // Flush with board
        int bs = card_suit(board3[i]);
        if (bs == s0 && s0 == s1) score += 2.0f;
    }

    return score;
}

inline void fast_discard(const int* hand5, const int* board3,
                          int& ki, int& kj, std::mt19937& rng,
                          float temperature = 0.05f) {
    float scores[10];
    float max_score = -1e9f;
    for (int p = 0; p < 10; p++) {
        int keep[2] = {hand5[KEEP_PAIRS[p][0]], hand5[KEEP_PAIRS[p][1]]};
        scores[p] = fast_score(keep, board3);
        max_score = std::max(max_score, scores[p]);
    }

    if (temperature <= 0) {
        int best = 0;
        for (int i = 1; i < 10; i++) if (scores[i] > scores[best]) best = i;
        ki = KEEP_PAIRS[best][0]; kj = KEEP_PAIRS[best][1];
        return;
    }

    // Softmax with temperature
    float probs[10], sum = 0;
    for (int i = 0; i < 10; i++) {
        probs[i] = std::exp((scores[i] - max_score) / temperature);
        sum += probs[i];
    }
    for (int i = 0; i < 10; i++) probs[i] /= sum;

    std::discrete_distribution<int> dist(probs, probs + 10);
    int chosen = dist(rng);
    ki = KEEP_PAIRS[chosen][0]; kj = KEEP_PAIRS[chosen][1];
}

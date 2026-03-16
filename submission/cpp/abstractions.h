#pragma once
#include "card_utils.h"

// ═══════════════════════════════════════════════
// Board texture bucketing
// ═══════════════════════════════════════════════

inline int board_pairedness(const int* board, int n) {
    int freq[NUM_RANKS] = {};
    for (int i = 0; i < n; i++) freq[card_rank(board[i])]++;
    int mx = 0, np = 0;
    for (int r = 0; r < NUM_RANKS; r++) { if (freq[r] > mx) mx = freq[r]; if (freq[r] >= 2) np++; }
    if (mx >= 3 || np >= 2) return 2;
    return (mx == 2) ? 1 : 0;
}

inline int board_flush_pressure(const int* board, int n) {
    int sc[NUM_SUITS] = {};
    for (int i = 0; i < n; i++) sc[card_suit(board[i])]++;
    int mx = 0; for (int s = 0; s < NUM_SUITS; s++) if (sc[s] > mx) mx = sc[s];
    if (mx >= 4) return 2; if (mx >= 3) return 1; return 0;
}

inline bool has_straight_potential(const bool rs[NUM_RANKS], int need) {
    for (int start = -1; start <= NUM_RANKS - need; start++) {
        int cnt = 0;
        for (int j = 0; j < need; j++) {
            int r = start + j;
            if (r == -1) { if (rs[ACE_RANK]) cnt++; }
            else if (r >= 0 && r < NUM_RANKS && rs[r]) cnt++;
        }
        if (cnt >= need) return true;
    }
    return false;
}

inline int board_straight_pressure(const int* board, int n) {
    bool rs[NUM_RANKS] = {};
    for (int i = 0; i < n; i++) rs[card_rank(board[i])] = true;
    if (has_straight_potential(rs, 5)) return 2;
    if (has_straight_potential(rs, 4)) return 1;
    return 0;
}

inline int board_height(const int* board, int n) {
    bool ace = false; int mx = 0;
    for (int i = 0; i < n; i++) { int r = card_rank(board[i]); if (r == ACE_RANK) ace = true; if (r > mx) mx = r; }
    if (ace) return 2; return (mx <= 4) ? 0 : 1;
}

inline int board_connectivity(const int* board, int n) {
    bool rs[NUM_RANKS] = {};
    int ranks[5];
    for (int i = 0; i < n; i++) { ranks[i] = card_rank(board[i]); rs[ranks[i]] = true; }
    std::sort(ranks, ranks + n);
    if (has_straight_potential(rs, 3)) {
        int span = ranks[n-1] - ranks[0], max_gap = 0;
        for (int i = 1; i < n; i++) { int g = ranks[i]-ranks[i-1]; if (g > max_gap) max_gap = g; }
        if (rs[ACE_RANK]) { int alt = 0; for (int i = 0; i < n; i++) if (ranks[i] != ACE_RANK && ranks[i] > alt) alt = ranks[i]; span = std::min(span, alt+1); }
        if (span <= 4 && max_gap <= 2) return 2;
        return 1;
    }
    return 0;
}

inline int card_delta(const int* prev, int pn, int nc) {
    int nr = card_rank(nc), ns = card_suit(nc);
    bool rp[NUM_RANKS] = {}; int sc[NUM_SUITS] = {}; bool pa = false;
    for (int i = 0; i < pn; i++) { rp[card_rank(prev[i])] = true; sc[card_suit(prev[i])]++; if (card_rank(prev[i]) == ACE_RANK) pa = true; }
    if (nr == ACE_RANK && !pa) return 2;
    if (rp[nr]) return 1;
    if (sc[ns] + 1 >= 3) return 1;
    rp[nr] = true;
    if (has_straight_potential(rp, (pn >= 4) ? 5 : 4)) return 1;
    return 0;
}

inline int flop_bucket(const int b[3]) { return board_pairedness(b,3)*27 + board_flush_pressure(b,3)*9 + board_connectivity(b,3)*3 + board_height(b,3); }

inline int turn_bucket(const int b3[3], int tc) {
    int b4[4] = {b3[0],b3[1],b3[2],tc};
    int tex = board_pairedness(b4,4)*27 + board_flush_pressure(b4,4)*9 + board_straight_pressure(b4,4)*3 + board_height(b4,4);
    return tex*3 + card_delta(b3,3,tc);
}

inline int river_bucket(const int b3[3], int tc, int rc) {
    int b5[5] = {b3[0],b3[1],b3[2],tc,rc};
    int b4[4] = {b3[0],b3[1],b3[2],tc};
    int tex = board_pairedness(b5,5)*27 + board_flush_pressure(b5,5)*9 + board_straight_pressure(b5,5)*3 + board_height(b5,5);
    return tex*3 + card_delta(b4,4,rc);
}

inline int board_bucket_for_street(const int* c, int st) {
    if (st == 0) return 0;
    if (st == 1) return flop_bucket(c);
    if (st == 2) return turn_bucket(c, c[3]);
    return river_bucket(c, c[3], c[4]);
}

// ═══════════════════════════════════════════════
// Hand bucket (deterministic, no MC)
// ═══════════════════════════════════════════════

inline int straight_draw_outs(const bool rs[NUM_RANKS]) {
    int outs = 0;
    for (int r = 0; r < NUM_RANKS; r++) {
        if (rs[r]) continue;
        bool t[NUM_RANKS]; std::copy(rs, rs+NUM_RANKS, t); t[r] = true;
        if (has_straight_potential(t, 5)) outs++;
    }
    return outs;
}

inline int made_tier_structure(const int h[2], const int* b, int bn) {
    int hr[2] = {card_rank(h[0]), card_rank(h[1])};
    int bmax = -1; bool brs[NUM_RANKS] = {};
    for (int i = 0; i < bn; i++) { int r = card_rank(b[i]); brs[r] = true; if (r > bmax) bmax = r; }
    int freq[NUM_RANKS] = {};
    freq[hr[0]]++; freq[hr[1]]++;
    for (int i = 0; i < bn; i++) freq[card_rank(b[i])]++;
    int np = 0; for (int r = 0; r < NUM_RANKS; r++) if (freq[r] >= 2) np++;

    if (bn == 5) {
        auto rank = eval::evaluate_7(h, b);
        if (rank.category <= eval::STRAIGHT) return 3;
        if (rank.category <= eval::TWO_PAIR) return 3;
        if (rank.category == eval::ONE_PAIR) {
            bool overpair = (hr[0] == hr[1] && hr[0] > bmax);
            bool top_pair = (hr[0] == bmax || hr[1] == bmax);
            return (overpair || top_pair) ? 2 : 1;
        }
        return 0;
    }
    for (int i = 0; i < 2; i++) if (freq[hr[i]] >= 3) return 3;
    if (np >= 2) { for (int i = 0; i < 2; i++) if (freq[hr[i]] >= 2) return 3; }
    if (hr[0] == hr[1] && hr[0] > bmax) return 2;
    if (hr[0] == bmax || hr[1] == bmax) return 2;
    if (brs[hr[0]] || brs[hr[1]] || hr[0] == hr[1]) return 1;
    return 0;
}

inline int draw_tier(const int h[2], const int* b, int bn) {
    int sc[NUM_SUITS] = {}; bool rs[NUM_RANKS] = {};
    for (int i = 0; i < 2; i++) { sc[card_suit(h[i])]++; rs[card_rank(h[i])] = true; }
    for (int i = 0; i < bn; i++) { sc[card_suit(b[i])]++; rs[card_rank(b[i])] = true; }
    int ms = 0; for (int s = 0; s < NUM_SUITS; s++) if (sc[s] > ms) ms = sc[s];
    bool fd = (ms >= 4), bd = (ms == 3 && bn == 3);
    int so = straight_draw_outs(rs);
    if (fd && so >= 2) return 4;
    if (fd) return 3;
    if (so >= 2) return 2;
    if (so >= 1 || bd) return 1;
    return 0;
}

inline int vulnerability(const int h[2], const int* b, int bn) {
    if (bn >= 5) return 0;
    int sc[NUM_SUITS] = {}; bool brs[NUM_RANKS] = {};
    for (int i = 0; i < bn; i++) { sc[card_suit(b[i])]++; brs[card_rank(b[i])] = true; }
    int ms = 0; for (int s = 0; s < NUM_SUITS; s++) if (sc[s] > ms) ms = sc[s];
    bool wf = (ms >= 2), ws = (straight_draw_outs(brs) >= 1);
    int made = made_tier_structure(h, b, bn);
    if (made <= 1 && (wf || ws)) return 1;
    if (made <= 2 && wf && ws) return 1;
    return 0;
}

inline int blocker_tier(const int h[2], const int* b, int bn) {
    int sc[NUM_SUITS] = {};
    for (int i = 0; i < bn; i++) sc[card_suit(b[i])]++;
    int msi = 0; for (int s = 1; s < NUM_SUITS; s++) if (sc[s] > sc[msi]) msi = s;
    bool bf = false;
    if (sc[msi] >= 3) for (int i = 0; i < 2; i++) if (card_suit(h[i]) == msi) bf = true;
    bool brs[NUM_RANKS] = {};
    for (int i = 0; i < bn; i++) brs[card_rank(b[i])] = true;
    bool bs = false;
    for (int i = 0; i < 2; i++) if (brs[card_rank(h[i])]) bs = true;
    return (bf || bs) ? 1 : 0;
}

inline int river_made_tier(const int h[2], const int b5[5]) {
    auto r = eval::evaluate_7(h, b5);
    if (r.category <= eval::STRAIGHT) return 4;
    if (r.category <= eval::TWO_PAIR) return 3;
    if (r.category == eval::ONE_PAIR) {
        // Top pair or overpair = thin value, otherwise bluff catcher
        int bmax = 0;
        for (int i = 0; i < 5; i++) { int br = card_rank(b5[i]); if (br > bmax) bmax = br; }
        bool overpair = (card_rank(h[0]) == card_rank(h[1]) && card_rank(h[0]) > bmax);
        bool top_pair = (card_rank(h[0]) == bmax || card_rank(h[1]) == bmax);
        if (overpair || top_pair) return 2;
        return 1;
    }
    return 0;
}

inline int flop_hand_bucket(const int h[2], const int b3[3]) {
    int m = made_tier_structure(h,b3,3), d = draw_tier(h,b3,3), v = vulnerability(h,b3,3);
    return m*6 + ((d>=3)?2:(d>=1)?1:0)*2 + v;
}
inline int turn_hand_bucket(const int h[2], const int b4[4]) {
    int m = made_tier_structure(h,b4,4), d = draw_tier(h,b4,4), v = vulnerability(h,b4,4);
    return m*6 + ((d>=3)?2:(d>=1)?1:0)*2 + v;
}
inline int river_hand_bucket(const int h[2], const int b5[5]) {
    return river_made_tier(h,b5)*2 + blocker_tier(h,b5,5);
}
inline int hand_bucket_for_street(const int h[2], const int* c, int st) {
    if (st <= 0) return 0;
    if (st == 1) return flop_hand_bucket(h, c);
    if (st == 2) return turn_hand_bucket(h, c);
    return river_hand_bucket(h, c);
}

// ═══════════════════════════════════════════════
// Opp discard bucket
// ═══════════════════════════════════════════════

inline int opp_discard_bucket_fn(const int disc[3], const int b3[3]) {
    int a = 0; for (int i = 0; i < 3; i++) if (card_rank(disc[i]) == ACE_RANK) a = 1;
    int freq[NUM_RANKS] = {}; for (int i = 0; i < 3; i++) freq[card_rank(disc[i])]++;
    int mx = 0; for (int r = 0; r < NUM_RANKS; r++) if (freq[r] > mx) mx = freq[r];
    int pp = (mx >= 3) ? 2 : (mx >= 2) ? 1 : 0;
    int sc[NUM_SUITS] = {}; for (int i = 0; i < 3; i++) sc[card_suit(disc[i])]++;
    int msc = 0; for (int s = 0; s < NUM_SUITS; s++) if (sc[s] > msc) msc = sc[s];
    int scc = (msc >= 3) ? 2 : (msc >= 2) ? 1 : 0;
    bool dr[NUM_RANKS]={}, br[NUM_RANKS]={};
    for (int i = 0; i < 3; i++) { dr[card_rank(disc[i])]=true; br[card_rank(b3[i])]=true; }
    int bi = 0;
    for (int r = 0; r < NUM_RANKS; r++) if (dr[r] && br[r]) { bi = 2; break; }
    if (!bi) { for (int r = 0; r < NUM_RANKS; r++) { if (!dr[r]) continue; if ((r>0&&br[r-1])||(r<NUM_RANKS-1&&br[r+1])) { bi=1; break; } } }
    if (!bi) { bool ds[NUM_SUITS]={}, bsu[NUM_SUITS]={}; for(int i=0;i<3;i++){ds[card_suit(disc[i])]=true;bsu[card_suit(b3[i])]=true;} for(int s=0;s<NUM_SUITS;s++) if(ds[s]&&bsu[s]){bi=1;break;} }
    return a*27 + pp*9 + scc*3 + bi;
}

// ═══════════════════════════════════════════════
// Public state
// ═══════════════════════════════════════════════

inline int pressure_bucket(int my_bet, int opp_bet) {
    int tc = opp_bet - my_bet;
    if (tc <= 0) return 0;
    int pot = my_bet + opp_bet;
    if (pot <= 0) return 1;
    double ratio = (double)tc / pot;
    if (ratio < 0.15) return 1;
    if (ratio < 0.30) return 2;
    if (ratio < 0.50) return 3;
    return 4;
}

inline int line_bucket_fn(const char* hist) {
    int nr = 0, first_r = -1;
    for (int i = 0; hist[i]; i++) {
        char c = hist[i];
        if (c == 'R' || c == 'r' || c == 'B' || c == 'b' || c == 'J') {
            if (first_r < 0) first_r = i;
            nr++;
        }
    }
    if (nr == 0) return 0;
    if (nr == 1) {
        if (first_r > 0 && (hist[first_r-1] == 'C' || hist[first_r-1] == 'K')) return 3;
        for (int i = first_r+1; hist[i]; i++) if (hist[i] == 'C') return 2;
        return 1;
    }
    if (nr == 2) return 4;
    return 5;
}

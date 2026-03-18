#pragma once
#include <algorithm>
#include "../core/constants.h"

// ── Hand evaluator (7-card, simplified for 27-card deck) ─────────────────────

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
    int trips_rank = -1;
    for (int r = 8; r >= 0; r--)
        if (rank_count[r] == 3) { trips_rank = r; break; }
    int pair_ranks[2] = {-1,-1}; int np = 0;
    for (int r = 8; r >= 0; r--)
        if (rank_count[r] == 2 && np < 2) pair_ranks[np++] = r;
    if (trips_rank >= 0 && np > 0)
        score = std::max(score, 5000 + trips_rank * 10 + pair_ranks[0]);
    int flush_suit = -1;
    for (int s = 0; s < 3; s++) if (suit_count[s] >= 5) flush_suit = s;
    bool has_rank[10] = {};
    for (int r = 0; r < 9; r++) if (rank_count[r] > 0) has_rank[r] = true;
    if (has_rank[8]) has_rank[9] = true;
    int straight_high = -1;
    for (int st = 5; st >= 0; st--) {
        if (st + 4 <= 9) {
            bool ok = true;
            for (int i = 0; i < 5; i++) if (!has_rank[st + i]) { ok = false; break; }
            if (ok) { straight_high = st + 4; break; }
        }
    }
    if (straight_high < 0 && has_rank[9] && has_rank[0] && has_rank[1] && has_rank[2] && has_rank[3])
        straight_high = 3;
    if (straight_high >= 0) score = std::max(score, 3500 + straight_high);
    if (flush_suit >= 0) score = std::max(score, 3000 + suit_count[flush_suit]);
    if (flush_suit >= 0 && straight_high >= 0) {
        int sr[10] = {};
        for (int i = 0; i < 7; i++) if (suits[i] == flush_suit) sr[ranks[i]] = 1;
        if (sr[8]) sr[9] = 1;
        for (int st = 5; st >= 0; st--) {
            if (st + 4 <= 9) {
                bool ok = true;
                for (int i = 0; i < 5; i++) if (!sr[st+i]) { ok = false; break; }
                if (ok) { score = std::max(score, 6000 + st + 4); break; }
            }
        }
        if (sr[9]&&sr[0]&&sr[1]&&sr[2]&&sr[3]) score = std::max(score, 6000);
    }
    if (trips_rank >= 0 && score < 4000) score = std::max(score, 4000 + trips_rank * 10);
    if (np >= 2 && score < 2000) score = std::max(score, 2000 + pair_ranks[0]*10 + pair_ranks[1]);
    else if (np >= 1 && score < 1000) score = std::max(score, 1000 + pair_ranks[0]*10);
    for (int r = 8; r >= 0; r--) if (rank_count[r] > 0) { score += r; break; }
    return score;
}

inline int evaluate_showdown(const int* p0, const int* p1, const int* comm) {
    int c0[7] = {p0[0],p0[1],comm[0],comm[1],comm[2],comm[3],comm[4]};
    int c1[7] = {p1[0],p1[1],comm[0],comm[1],comm[2],comm[3],comm[4]};
    int s0 = hand_score(c0), s1 = hand_score(c1);
    return (s0 > s1) ? 1 : (s0 < s1) ? -1 : 0;
}

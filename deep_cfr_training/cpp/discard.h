#pragma once
#include <algorithm>
#include <cmath>
#include <random>
#include "constants.h"

// ── Deck shuffle ──────────────────────────────────────────────────────────────

inline void shuffle_deck(int* deck, std::mt19937& rng) {
    for (int i = 0; i < DECK_SIZE; i++) deck[i] = i;
    for (int i = DECK_SIZE - 1; i > 0; i--) {
        std::uniform_int_distribution<int> dist(0, i);
        std::swap(deck[i], deck[dist(rng)]);
    }
}

// ── Hand evaluator (7-card, simplified for 27-card deck) ──────────────────────

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

// ── Fast heuristic discard ────────────────────────────────────────────────────

constexpr int KEEP_PAIRS[10][2] = {
    {0,1},{0,2},{0,3},{0,4},{1,2},{1,3},{1,4},{2,3},{2,4},{3,4}
};

inline float fast_score(const int* keep, const int* board3) {
    int r0 = card_rank(keep[0]), r1 = card_rank(keep[1]);
    int s0 = card_suit(keep[0]), s1 = card_suit(keep[1]);
    float sc = 0;
    if (r0 == r1) sc += 10.f;
    sc += std::max(r0, r1) * 0.5f;
    if (s0 == s1) sc += 3.f;
    for (int i = 0; i < 3; i++) {
        if (board3[i] < 0) continue;
        int br = card_rank(board3[i]), bs = card_suit(board3[i]);
        if (br == r0 || br == r1) sc += 5.f;
        if (std::abs(br-r0) <= 1 || std::abs(br-r1) <= 1) sc += 1.f;
        if (bs == s0 && s0 == s1) sc += 2.f;
    }
    return sc;
}

inline void fast_discard(const int* hand5, const int* board3,
                          int& ki, int& kj, std::mt19937& rng,
                          float temperature = 0.05f) {
    float scores[10]; float max_sc = -1e9f;
    for (int p = 0; p < 10; p++) {
        int keep[2] = {hand5[KEEP_PAIRS[p][0]], hand5[KEEP_PAIRS[p][1]]};
        scores[p] = fast_score(keep, board3);
        max_sc = std::max(max_sc, scores[p]);
    }
    if (temperature <= 0) {
        int best = 0;
        for (int i = 1; i < 10; i++) if (scores[i] > scores[best]) best = i;
        ki = KEEP_PAIRS[best][0]; kj = KEEP_PAIRS[best][1]; return;
    }
    float probs[10], sum = 0;
    for (int i = 0; i < 10; i++) {
        probs[i] = std::exp((scores[i]-max_sc)/temperature); sum += probs[i];
    }
    for (int i = 0; i < 10; i++) probs[i] /= sum;
    std::discrete_distribution<int> dist(probs, probs+10);
    int chosen = dist(rng);
    ki = KEEP_PAIRS[chosen][0]; kj = KEEP_PAIRS[chosen][1];
}

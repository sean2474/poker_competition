#pragma once
#include <algorithm>
#include <cmath>
#include <random>
#include "../core/constants.h"
#include "showdown.h"

// ── Deck shuffle ──────────────────────────────────────────────────────────────

inline void shuffle_deck(int* deck, std::mt19937& rng) {
    for (int i = 0; i < DECK_SIZE; i++) deck[i] = i;
    for (int i = DECK_SIZE - 1; i > 0; i--) {
        std::uniform_int_distribution<int> dist(0, i);
        std::swap(deck[i], deck[dist(rng)]);
    }
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

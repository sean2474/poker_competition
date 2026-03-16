#pragma once
#include <cstdint>
#include <array>
#include <string>
#include <vector>
#include <algorithm>
#include <cassert>
#include <random>

// ─── Deck constants ───
constexpr int NUM_RANKS = 9;   // 2-9, A
constexpr int NUM_SUITS = 3;   // d, h, s
constexpr int DECK_SIZE = 27;
constexpr int ACE_RANK = 8;    // rank index of Ace

inline int card_rank(int c) { return c % NUM_RANKS; }
inline int card_suit(int c) { return c / NUM_RANKS; }
inline int make_card(int rank, int suit) { return suit * NUM_RANKS + rank; }

inline const char* RANK_CHARS = "23456789A";
inline const char* SUIT_CHARS = "dhs";

inline std::string card_str(int c) {
    std::string s(1, RANK_CHARS[card_rank(c)]);
    s += SUIT_CHARS[card_suit(c)];
    return s;
}

// ─── Suit canonicalization ───
// Sort by rank then suit, relabel suits in order of first appearance
struct CanonResult {
    std::array<int, 5> cards;
    int count;
};

inline CanonResult canonicalize_5(const int* cards, int n) {
    // Sort by (rank, suit)
    std::array<int, 5> sorted_cards;
    for (int i = 0; i < n; i++) sorted_cards[i] = cards[i];
    std::sort(sorted_cards.begin(), sorted_cards.begin() + n,
        [](int a, int b) {
            int ra = card_rank(a), rb = card_rank(b);
            if (ra != rb) return ra < rb;
            return card_suit(a) < card_suit(b);
        });

    int suit_map[3] = {-1, -1, -1};
    int next_suit = 0;
    std::array<int, 5> canon;
    for (int i = 0; i < n; i++) {
        int s = card_suit(sorted_cards[i]);
        if (suit_map[s] == -1) suit_map[s] = next_suit++;
        canon[i] = make_card(card_rank(sorted_cards[i]), suit_map[s]);
    }
    std::sort(canon.begin(), canon.begin() + n);
    return {canon, n};
}

// ─── Hand evaluation ───
// For the 27-card variant, we implement a simple but correct 7-card evaluator.
// We enumerate all C(7,5) = 21 five-card combos and take the best.
//
// 5-card hand rankings (lower = better):
//   Straight Flush, Full House, Flush, Straight, Trips, Two Pair, One Pair, High Card
//   (No four-of-a-kind possible with 3 suits)

namespace eval {

// Hand category (lower = better)
enum HandCat : int {
    STRAIGHT_FLUSH = 0,
    FULL_HOUSE = 1,
    FLUSH = 2,
    STRAIGHT = 3,
    THREE_OF_KIND = 4,
    TWO_PAIR = 5,
    ONE_PAIR = 6,
    HIGH_CARD = 7,
};

struct HandRank {
    int category;  // HandCat
    int primary;   // primary rank (e.g. trips rank, pair rank)
    int secondary; // secondary rank (e.g. pair in full house)
    int kickers[5]; // kicker ranks in descending order
    
    bool operator<(const HandRank& o) const {
        if (category != o.category) return category < o.category;
        if (primary != o.primary) return primary > o.primary;  // higher rank = better
        if (secondary != o.secondary) return secondary > o.secondary;
        for (int i = 0; i < 5; i++) {
            if (kickers[i] != o.kickers[i]) return kickers[i] > o.kickers[i];
        }
        return false; // equal
    }
    bool operator==(const HandRank& o) const {
        if (category != o.category || primary != o.primary || secondary != o.secondary) return false;
        for (int i = 0; i < 5; i++) if (kickers[i] != o.kickers[i]) return false;
        return true;
    }
};

// Check for straight in a sorted (ascending) rank array of length 5
// Ace can be low (below 2) or high (above 9)
// Returns the high card rank of the straight, or -1 if not a straight
inline int check_straight(const int ranks[5]) {
    // Normal straight check
    bool is_straight = true;
    for (int i = 1; i < 5; i++) {
        if (ranks[i] != ranks[i-1] + 1) { is_straight = false; break; }
    }
    if (is_straight) return ranks[4]; // high card

    // Ace-low: A,2,3,4,5 = ranks sorted would be {0,1,2,3,8}
    if (ranks[0] == 0 && ranks[1] == 1 && ranks[2] == 2 && ranks[3] == 3 && ranks[4] == ACE_RANK) {
        return 3; // 5 is the high card (rank index 3)
    }

    // Ace-high: 6,7,8,9,A = ranks {4,5,6,7,8}
    // This is already handled by normal check since A=8 and 9=7, so 4,5,6,7,8 works
    
    return -1;
}

inline HandRank evaluate_5(const int cards[5]) {
    int ranks[5], suits[5];
    for (int i = 0; i < 5; i++) {
        ranks[i] = card_rank(cards[i]);
        suits[i] = card_suit(cards[i]);
    }
    
    // Sort ranks ascending
    int sr[5];
    std::copy(ranks, ranks + 5, sr);
    std::sort(sr, sr + 5);
    
    // Check flush
    bool is_flush = (suits[0] == suits[1] && suits[1] == suits[2] && 
                     suits[2] == suits[3] && suits[3] == suits[4]);
    
    // Check straight
    int straight_high = check_straight(sr);
    bool is_straight = (straight_high >= 0);
    
    // Count rank frequencies
    int freq[NUM_RANKS] = {};
    for (int i = 0; i < 5; i++) freq[ranks[i]]++;
    
    // Find pairs, trips, etc.
    int trips_rank = -1, pairs[2] = {-1, -1}, pair_count = 0;
    // Iterate from high to low for ranking purposes
    for (int r = NUM_RANKS - 1; r >= 0; r--) {
        if (freq[r] == 3) trips_rank = r;
        else if (freq[r] == 2 && pair_count < 2) pairs[pair_count++] = r;
    }
    
    HandRank result;
    result.primary = result.secondary = -1;
    std::fill(result.kickers, result.kickers + 5, -1);
    
    if (is_flush && is_straight) {
        result.category = STRAIGHT_FLUSH;
        result.primary = straight_high;
    } else if (trips_rank >= 0 && pair_count >= 1) {
        result.category = FULL_HOUSE;
        result.primary = trips_rank;
        result.secondary = pairs[0];
    } else if (is_flush) {
        result.category = FLUSH;
        // Kickers are all 5 ranks sorted descending
        for (int i = 0; i < 5; i++) result.kickers[i] = sr[4 - i];
    } else if (is_straight) {
        result.category = STRAIGHT;
        result.primary = straight_high;
    } else if (trips_rank >= 0) {
        result.category = THREE_OF_KIND;
        result.primary = trips_rank;
        int ki = 0;
        for (int r = NUM_RANKS - 1; r >= 0; r--)
            if (freq[r] == 1) result.kickers[ki++] = r;
    } else if (pair_count >= 2) {
        result.category = TWO_PAIR;
        result.primary = std::max(pairs[0], pairs[1]);
        result.secondary = std::min(pairs[0], pairs[1]);
        for (int r = NUM_RANKS - 1; r >= 0; r--)
            if (freq[r] == 1) { result.kickers[0] = r; break; }
    } else if (pair_count == 1) {
        result.category = ONE_PAIR;
        result.primary = pairs[0];
        int ki = 0;
        for (int r = NUM_RANKS - 1; r >= 0; r--)
            if (freq[r] == 1) result.kickers[ki++] = r;
    } else {
        result.category = HIGH_CARD;
        for (int i = 0; i < 5; i++) result.kickers[i] = sr[4 - i];
    }
    
    return result;
}

// Evaluate best 5-card hand from 2 hole + 5 board = 7 cards
// Returns HandRank (lower category = better)
inline HandRank evaluate_7(const int hand[2], const int board[5]) {
    int all[7];
    all[0] = hand[0]; all[1] = hand[1];
    for (int i = 0; i < 5; i++) all[i + 2] = board[i];
    
    // Try all C(7,5) = 21 combos
    HandRank best;
    best.category = HIGH_CARD + 1; // worse than anything
    
    for (int i = 0; i < 7; i++) {
        for (int j = i + 1; j < 7; j++) {
            // Skip cards i and j, take remaining 5
            int five[5];
            int k = 0;
            for (int x = 0; x < 7; x++) {
                if (x != i && x != j) five[k++] = all[x];
            }
            HandRank r = evaluate_5(five);
            if (r < best) best = r;
        }
    }
    
    return best;
}

// Compare two hands: -1 = hand0 wins, 0 = tie, 1 = hand1 wins
inline int compare_hands(const int h0[2], const int h1[2], const int board[5]) {
    HandRank r0 = evaluate_7(h0, board);
    HandRank r1 = evaluate_7(h1, board);
    if (r0 < r1) return -1;
    if (r1 < r0) return 1;
    return 0;
}

} // namespace eval

// ─── Random number generator ───
inline std::mt19937& get_rng() {
    static thread_local std::mt19937 rng(std::random_device{}());
    return rng;
}

inline void shuffle_deck(int deck[DECK_SIZE]) {
    for (int i = 0; i < DECK_SIZE; i++) deck[i] = i;
    std::shuffle(deck, deck + DECK_SIZE, get_rng());
}

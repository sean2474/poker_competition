#pragma once

// ── Game constants (match Python game/constants.py) ──────────────────────────
constexpr int MAX_BET          = 100;
constexpr int SMALL_BLIND      = 1;
constexpr int BIG_BLIND        = 2;
constexpr int NUM_RANKS        = 9;    // 2-9, A
constexpr int NUM_SUITS        = 3;    // d, h, s
constexpr int DECK_SIZE        = 27;
constexpr int NUM_ACTIONS      = 8;
constexpr int FEATURE_DIM      = 119;  // 93 base + 26 extra
constexpr int BASE_FEATURE_DIM = 93;
constexpr int MAX_HISTORY      = 30;

// Action IDs
constexpr int A_FOLD       = 0;
constexpr int A_CALL       = 1;
constexpr int A_CHECK      = 2;
constexpr int A_BET_SMALL  = 3;
constexpr int A_BET_LARGE  = 4;
constexpr int A_RAISE_SMALL = 5;
constexpr int A_RAISE_LARGE = 6;
constexpr int A_BET_POT    = 7;

// Card helpers
inline int card_rank(int c) { return c % NUM_RANKS; }
inline int card_suit(int c) { return c / NUM_RANKS; }

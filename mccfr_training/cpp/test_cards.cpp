#include "card_utils.h"
#include <iostream>
#include <cassert>
#include <cstring>

using namespace std;

void test_basics() {
    cout << "=== Basic Card Tests ===" << endl;
    
    // card_rank / card_suit / card_str
    assert(card_rank(0) == 0 && card_suit(0) == 0);  // 2d
    assert(card_str(0) == "2d");
    assert(card_str(26) == "As");
    assert(card_str(9) == "2h");
    assert(card_str(3) == "5d");
    assert(make_card(0, 0) == 0);
    assert(make_card(8, 2) == 26);  // As
    cout << "  card accessors OK" << endl;
    
    // Print full deck
    cout << "  Full deck:" << endl;
    for (int i = 0; i < DECK_SIZE; i++) {
        cout << "  " << i << "=" << card_str(i);
        if ((i + 1) % 9 == 0) cout << endl;
        else cout << " ";
    }
}

void test_canonicalize() {
    cout << "=== Canonical Tests ===" << endl;
    
    // 9d 9h 2s 3d 4h  vs  9h 9s 2d 3h 4s -> should be same
    int h1[] = {make_card(7,0), make_card(7,1), make_card(0,2), make_card(1,0), make_card(2,1)};
    int h2[] = {make_card(7,1), make_card(7,2), make_card(0,0), make_card(1,1), make_card(2,2)};
    
    auto c1 = canonicalize_5(h1, 5);
    auto c2 = canonicalize_5(h2, 5);
    
    cout << "  h1 canon:";
    for (int i = 0; i < 5; i++) cout << " " << c1.cards[i];
    cout << endl;
    cout << "  h2 canon:";
    for (int i = 0; i < 5; i++) cout << " " << c2.cards[i];
    cout << endl;
    
    bool match = true;
    for (int i = 0; i < 5; i++) if (c1.cards[i] != c2.cards[i]) match = false;
    assert(match);
    cout << "  suit iso OK" << endl;
}

void test_evaluator() {
    cout << "=== Evaluator Tests ===" << endl;
    
    using namespace eval;
    
    // Test straight flush: 2d 3d 4d 5d 6d
    {
        int cards[] = {make_card(0,0), make_card(1,0), make_card(2,0), make_card(3,0), make_card(4,0)};
        auto r = evaluate_5(cards);
        cout << "  2d-6d straight flush: cat=" << r.category << endl;
        assert(r.category == STRAIGHT_FLUSH);
    }
    
    // Test flush: 2d 4d 6d 8d Ad
    {
        int cards[] = {make_card(0,0), make_card(2,0), make_card(4,0), make_card(6,0), make_card(8,0)};
        auto r = evaluate_5(cards);
        cout << "  2d 4d 6d 8d Ad flush: cat=" << r.category << endl;
        assert(r.category == FLUSH);
    }
    
    // Test straight: 5d 6h 7s 8d 9h
    {
        int cards[] = {make_card(3,0), make_card(4,1), make_card(5,2), make_card(6,0), make_card(7,1)};
        auto r = evaluate_5(cards);
        cout << "  5-9 straight: cat=" << r.category << endl;
        assert(r.category == STRAIGHT);
    }
    
    // Test A-low straight: A 2 3 4 5
    {
        int cards[] = {make_card(8,0), make_card(0,1), make_card(1,2), make_card(2,0), make_card(3,1)};
        auto r = evaluate_5(cards);
        cout << "  A-low straight: cat=" << r.category << " high=" << r.primary << endl;
        assert(r.category == STRAIGHT);
        assert(r.primary == 3); // 5 is high
    }
    
    // Test A-high straight: 6 7 8 9 A
    {
        int cards[] = {make_card(4,0), make_card(5,1), make_card(6,2), make_card(7,0), make_card(8,1)};
        auto r = evaluate_5(cards);
        cout << "  A-high straight: cat=" << r.category << " high=" << r.primary << endl;
        assert(r.category == STRAIGHT);
        assert(r.primary == ACE_RANK);
    }
    
    // Test full house: 5d 5h 5s 3d 3h
    {
        int cards[] = {make_card(3,0), make_card(3,1), make_card(3,2), make_card(1,0), make_card(1,1)};
        auto r = evaluate_5(cards);
        cout << "  full house 5s over 3s: cat=" << r.category << endl;
        assert(r.category == FULL_HOUSE);
        assert(r.primary == 3); // trips 5
        assert(r.secondary == 1); // pair 3
    }
    
    // Test trips: 8d 8h 8s 2d 3h
    {
        int cards[] = {make_card(6,0), make_card(6,1), make_card(6,2), make_card(0,0), make_card(1,1)};
        auto r = evaluate_5(cards);
        cout << "  trips 8s: cat=" << r.category << endl;
        assert(r.category == THREE_OF_KIND);
    }
    
    // Test two pair: 9d 9h 3d 3h 8s
    {
        int cards[] = {make_card(7,0), make_card(7,1), make_card(1,0), make_card(1,1), make_card(6,2)};
        auto r = evaluate_5(cards);
        cout << "  two pair 9s and 3s: cat=" << r.category << endl;
        assert(r.category == TWO_PAIR);
    }
    
    // Test one pair: Ad Ah 7d 4s 2h
    {
        int cards[] = {make_card(8,0), make_card(8,1), make_card(5,0), make_card(2,2), make_card(0,1)};
        auto r = evaluate_5(cards);
        cout << "  pair of aces: cat=" << r.category << endl;
        assert(r.category == ONE_PAIR);
        assert(r.primary == ACE_RANK);
    }
    
    // Test high card: Ad 8s 6h 4d 2s
    {
        int cards[] = {make_card(8,0), make_card(6,2), make_card(4,1), make_card(2,0), make_card(0,2)};
        auto r = evaluate_5(cards);
        cout << "  high card ace: cat=" << r.category << endl;
        assert(r.category == HIGH_CARD);
    }
    
    cout << "  all 5-card evals OK" << endl;
}

void test_compare() {
    cout << "=== Compare Hands (7-card) ===" << endl;
    
    using namespace eval;
    
    // Overpair 9-9 vs underpair 2-2 on board 4d 6h 8s Ad 3h (no straight possible for 2-2)
    {
        int h0[] = {make_card(7,0), make_card(7,1)}; // 9d 9h
        int h1[] = {make_card(0,0), make_card(0,1)}; // 2d 2h
        int board[] = {make_card(2,0), make_card(4,1), make_card(6,2), make_card(8,0), make_card(1,1)};
        // board: 4d 6h 8s Ad 3h -- disconnected, no straight for 2-2
        int r = compare_hands(h0, h1, board);
        cout << "  9-9 vs 2-2 on 4d6h8sAd3h: " << r << " (expect -1=h0 wins)" << endl;
        assert(r == -1);
    }
    
    // Flush vs pair
    {
        int h0[] = {make_card(0,0), make_card(2,0)}; // 2d 4d (flush draw)
        int h1[] = {make_card(7,1), make_card(5,2)}; // 9h 7s
        int board[] = {make_card(1,0), make_card(4,0), make_card(6,0), make_card(3,1), make_card(8,2)};
        // board: 3d 6d 8d 5h As -> h0 has flush (2d 3d 4d 6d 8d)
        int r = compare_hands(h0, h1, board);
        cout << "  flush vs non-flush: " << r << " (expect -1)" << endl;
        assert(r == -1);
    }
    
    cout << "  compare OK" << endl;
}

void test_shuffle() {
    cout << "=== Shuffle Test ===" << endl;
    int deck[DECK_SIZE];
    shuffle_deck(deck);
    
    // Check all cards present
    bool seen[DECK_SIZE] = {};
    for (int i = 0; i < DECK_SIZE; i++) {
        assert(deck[i] >= 0 && deck[i] < DECK_SIZE);
        seen[deck[i]] = true;
    }
    for (int i = 0; i < DECK_SIZE; i++) assert(seen[i]);
    cout << "  shuffle OK" << endl;
}

int main() {
    test_basics();
    test_canonicalize();
    test_evaluator();
    test_compare();
    test_shuffle();
    
    cout << endl << "ALL CARD_UTILS TESTS PASSED" << endl;
    return 0;
}

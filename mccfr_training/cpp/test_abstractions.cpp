#include "abstractions.h"
#include <iostream>
#include <cassert>
using namespace std;

void test_board_bucket() {
    cout << "=== Board Bucket Tests ===" << endl;
    
    // Rainbow unpaired disconnected low: 2d 4h 6s
    int b1[] = {make_card(0,0), make_card(2,1), make_card(4,2)};
    int fb1 = flop_bucket(b1);
    cout << "  2d 4h 6s flop: " << fb1 << endl;
    
    // Monotone: 2d 5d 8d
    int b2[] = {make_card(0,0), make_card(3,0), make_card(6,0)};
    int fb2 = flop_bucket(b2);
    cout << "  2d 5d 8d flop: " << fb2 << " (expect flush_pressure=1 for 3-card)" << endl;
    
    // Paired: 3d 3h 7s
    int b3[] = {make_card(1,0), make_card(1,1), make_card(5,2)};
    int fb3 = flop_bucket(b3);
    cout << "  3d 3h 7s flop: " << fb3 << endl;
    assert(board_pairedness(b3, 3) == 1);
    
    // Turn + river bucket
    int community[] = {make_card(0,0), make_card(3,1), make_card(6,2), make_card(1,0), make_card(4,1)};
    for (int st = 0; st < 4; st++) {
        int bb = board_bucket_for_street(community, st);
        cout << "  street " << st << ": bucket=" << bb << endl;
    }
    cout << "  board bucket OK" << endl;
}

void test_hand_bucket() {
    cout << "=== Hand Bucket Tests ===" << endl;
    
    // Overpair 9-9 on low board 2d 3h 5s
    int hand_op[] = {make_card(7,0), make_card(7,1)};
    int board_low[] = {make_card(0,0), make_card(1,1), make_card(3,2)};
    int fb = flop_hand_bucket(hand_op, board_low);
    int made = made_tier_structure(hand_op, board_low, 3);
    cout << "  overpair 9-9 on 2-3-5: made=" << made << " bucket=" << fb << endl;
    assert(made == 2); // overpair = top pair tier
    
    // Air: 2d 3d on 7h 8s As
    int hand_air[] = {make_card(0,0), make_card(1,0)};
    int board_high[] = {make_card(5,1), make_card(6,2), make_card(8,2)};
    int made_air = made_tier_structure(hand_air, board_high, 3);
    cout << "  air 2d3d on 7h8sAs: made=" << made_air << endl;
    assert(made_air == 0);
    
    // Set: 5d 5h on 5s 2d 3h
    int hand_set[] = {make_card(3,0), make_card(3,1)};
    int board_set[] = {make_card(3,2), make_card(0,0), make_card(1,1)};
    int made_set = made_tier_structure(hand_set, board_set, 3);
    cout << "  set 5-5-5: made=" << made_set << endl;
    assert(made_set == 3);
    
    // River bucket
    int community[] = {make_card(0,0), make_card(1,1), make_card(3,2), make_card(2,0), make_card(4,1)};
    int rb = river_hand_bucket(hand_op, community);
    cout << "  9d9h river bucket=" << rb << endl;
    
    // Turn
    int tb = turn_hand_bucket(hand_op, community); // uses first 4
    cout << "  9d9h turn bucket=" << tb << endl;
    
    cout << "  hand bucket OK" << endl;
}

void test_opp_discard() {
    cout << "=== Opp Discard Bucket Tests ===" << endl;
    
    int board[] = {make_card(3,0), make_card(5,1), make_card(7,2)};
    
    // Discarded Ace + low cards
    int disc1[] = {make_card(8,0), make_card(0,1), make_card(1,2)};
    int b1 = opp_discard_bucket_fn(disc1, board);
    cout << "  Ad,2h,3s: bucket=" << b1 << endl;
    
    // Discarded pair
    int disc2[] = {make_card(1,0), make_card(1,1), make_card(4,2)};
    int b2 = opp_discard_bucket_fn(disc2, board);
    cout << "  3d,3h,6s: bucket=" << b2 << endl;
    
    cout << "  opp discard OK" << endl;
}

void test_public_state() {
    cout << "=== Public State Tests ===" << endl;
    
    assert(pressure_bucket(5, 5) == 0);
    assert(pressure_bucket(2, 3) == 2);  // ratio=0.2
    assert(pressure_bucket(1, 50) == 4); // ratio=0.96
    cout << "  pressure OK" << endl;
    
    assert(line_bucket_fn("") == 0);
    assert(line_bucket_fn("K") == 0);
    assert(line_bucket_fn("b") == 1);
    assert(line_bucket_fn("bC") == 2);
    assert(line_bucket_fn("Kb") == 3);
    assert(line_bucket_fn("bR") == 4);
    assert(line_bucket_fn("bRR") == 5);
    cout << "  line bucket OK" << endl;
}

void test_draw_tier() {
    cout << "=== Draw Tier Tests ===" << endl;
    
    // 4 diamonds: 2d 5d on 3d 4d 9s -> flush draw
    int hand_fd[] = {make_card(0,0), make_card(3,0)};
    int board_fd[] = {make_card(1,0), make_card(2,0), make_card(7,2)};
    int dt = draw_tier(hand_fd, board_fd, 3);
    cout << "  4 diamonds: draw_tier=" << dt << endl;
    assert(dt >= 3); // flush draw or combo
    
    // No draw: 2d 3h on 7s 9d As (disconnected, rainbow)
    int hand_nd[] = {make_card(0,0), make_card(1,1)};
    int board_nd[] = {make_card(5,2), make_card(7,0), make_card(8,2)};
    int dt2 = draw_tier(hand_nd, board_nd, 3);
    cout << "  no draw: draw_tier=" << dt2 << endl;
    
    cout << "  draw tier OK" << endl;
}

int main() {
    test_board_bucket();
    test_hand_bucket();
    test_opp_discard();
    test_public_state();
    test_draw_tier();
    
    cout << endl << "ALL ABSTRACTION TESTS PASSED" << endl;
    return 0;
}

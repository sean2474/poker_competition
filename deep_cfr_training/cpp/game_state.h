#pragma once
#include <algorithm>
#include <cstring>
#include <cmath>
#include "constants.h"

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



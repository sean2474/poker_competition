"""Python GameState — mirrors C++ game_state.h logic exactly."""

from .constants import (
    MAX_BET, SMALL_BLIND, BIG_BLIND,
    A_FOLD, A_CALL, A_CHECK,
    A_BET_SMALL, A_BET_LARGE, A_RAISE_SMALL, A_RAISE_LARGE, A_BET_POT,
)


class GameState:
    def __init__(self):
        self.street = 0
        self.bets = [SMALL_BLIND, BIG_BLIND]
        self.current_player = 0       # SB acts first preflop
        self.is_terminal = False
        self.folded_player = -1
        self.min_raise = BIG_BLIND
        self.history = []
        self.last_street_bet = 0
        self.num_actions_this_street = 0
        self.street_bets        = [[0, 0], [0, 0], [0, 0], [0, 0]]    # [street][player] max raise amt (chips, for C++)
        self.street_last_ratios = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]  # LAST bet/pot this street
        self.street_bet_counts  = [[0, 0], [0, 0], [0, 0], [0, 0]]     # num bets/raises per player per street
        self.preflop_open_override = None  # override open raise_amt (chips above BB) for training diversity

    def copy(self):
        s = GameState()
        s.street = self.street
        s.bets = list(self.bets)
        s.current_player = self.current_player
        s.is_terminal = self.is_terminal
        s.folded_player = self.folded_player
        s.min_raise = self.min_raise
        s.history = list(self.history)
        s.last_street_bet = self.last_street_bet
        s.num_actions_this_street = self.num_actions_this_street
        s.street_bets        = [list(b) for b in self.street_bets]
        s.street_last_ratios = [list(b) for b in self.street_last_ratios]
        s.street_bet_counts  = [list(b) for b in self.street_bet_counts]
        s.preflop_open_override = self.preflop_open_override
        return s

    def get_valid_actions(self):
        cp, opp = self.current_player, 1 - self.current_player
        to_call = self.bets[opp] - self.bets[cp]
        max_raise = MAX_BET - max(self.bets)
        # Preflop: always allow all-in even if < min_raise
        can_raise = max_raise > 0 and (self.street == 0 or self.min_raise <= max_raise)

        actions = []
        if to_call > 0:
            actions.append(A_FOLD)
            actions.append(A_CALL)
            if can_raise:
                actions.append(A_RAISE_SMALL)           # preflop: single size; postflop: 33%
                if self.street > 0:
                    actions.append(A_RAISE_LARGE)        # postflop only: 75%
        else:
            actions.append(A_CHECK)
            if can_raise:
                if self.street == 0:
                    actions.append(A_BET_SMALL)          # preflop: single open size
                else:
                    pot = self.bets[0] + self.bets[1]
                    mn = self.min_raise
                    small_amt = max(mn, min(int(pot * 0.33), max_raise))
                    large_amt = max(mn, min(int(pot * 0.75), max_raise))
                    pot_amt   = max(mn, min(pot,             max_raise))
                    thresh    = max(2, pot // 20)
                    actions.append(A_BET_SMALL)
                    if abs(pot_amt - large_amt) > thresh:
                        actions.append(A_BET_POT)
                    if abs(large_amt - small_amt) > thresh:
                        actions.append(A_BET_LARGE)
        return actions

    def apply(self, action):
        s = self.copy()
        cp, opp = s.current_player, 1 - s.current_player
        max_raise = MAX_BET - max(s.bets)

        s.history.append((cp, action))

        if action == A_FOLD:
            s.is_terminal = True
            s.folded_player = cp
            return s

        if action == A_CHECK:
            s.num_actions_this_street += 1
            if s.num_actions_this_street >= 2 and s.bets[0] == s.bets[1]:
                s._advance_street()
            else:
                s.current_player = opp
            return s

        if action == A_CALL:
            s.bets[cp] = s.bets[opp]
            s.num_actions_this_street += 1
            # SB limp preflop: BB gets option
            if not (s.street == 0 and cp == 0 and s.bets[cp] == BIG_BLIND):
                s._advance_street()
            else:
                s.current_player = opp
            return s

        # ── Raise / Bet ──────────────────────────────────────────────────────
        s.num_actions_this_street += 1
        pot = s.bets[0] + s.bets[1]
        mn  = s.min_raise

        if s.street == 0:
            # Preflop tiered sizing
            _OPEN_AMT = s.preflop_open_override if s.preflop_open_override else int(round(1.5 * BIG_BLIND))
            if mn <= BIG_BLIND:                  # open
                raise_amt = max(mn, min(_OPEN_AMT, max_raise))
            elif mn <= _OPEN_AMT:                # 3-bet
                raise_amt = max(mn, min(3 * mn, max_raise))
            else:                                # 4-bet+: all-in (MAX_BET cap)
                raise_amt = max_raise
            raise_amt = min(raise_amt, max_raise)   # allow incomplete raise (all-in)
        else:
            # Postflop pot-relative
            if action in (A_BET_SMALL, A_RAISE_SMALL):
                raise_amt = max(mn, min(int(pot * 0.33), max_raise))
            elif action == A_BET_POT:
                raise_amt = max(mn, min(pot, max_raise))
            else:
                raise_amt = max(mn, min(int(pot * 0.75), max_raise))
            raise_amt = max(s.min_raise, min(raise_amt, max_raise))

        pot_before = pot  # pot before the bet/raise
        ratio = raise_amt / max(float(pot_before), 1.0)
        s.street_bets[s.street][cp]       = max(s.street_bets[s.street][cp], raise_amt)
        s.street_last_ratios[s.street][cp] = ratio          # last (not max) — most recent bet size
        s.street_bet_counts[s.street][cp] += 1               # count re-raises
        s.bets[cp]   = s.bets[opp] + raise_amt
        s.min_raise  = max(raise_amt, s.min_raise)
        s.current_player = opp
        return s

    def _advance_street(self):
        if self.street >= 3:
            self.is_terminal = True
        else:
            self.street += 1
            self.current_player = 1 if self.street >= 1 else 0
            self.last_street_bet = max(self.bets)
            self.min_raise = BIG_BLIND
            self.num_actions_this_street = 0

"""
Opponent range tracker — Bayesian update of P(opponent's 2-card held hand).

Updates:
  1. record_preflop_action() → applied lazily after discards are known
  2. apply_preflop_updates()  → uses preflop chart + known 5-card hand
  3. update_discard()        → removes impossible hands (contains discarded cards)
  4. update_postflop_action() → batch StrategyNet inference for P(action | pair)
  5. get_cats()               → 17-dim category distribution for features [34-50]
"""

import numpy as np
import torch

from features import (
    _ALL_PAIRS, _range_to_cats, state_to_features, FEATURE_DIM,
)
from action import FOLD, RAISE, CHECK, CALL, NUM_ACTIONS


class OppRangeTracker:
    """Tracks a 351-dim probability distribution over opponent's kept 2-card hand."""

    def __init__(self):
        self._probs: np.ndarray = None   # shape (351,), float32
        self._pf_pending: list  = []     # (slot, size_bucket, hist_str) to apply post-discard

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def reset(self, my_cards5: list):
        """Call at start of each hand with hero's 5 preflop cards."""
        dead = set(c for c in my_cards5 if c >= 0)
        self._probs = np.array(
            [0. if (a in dead or b in dead) else 1. for a, b in _ALL_PAIRS],
            dtype=np.float32,
        )
        s = self._probs.sum()
        if s > 0:
            self._probs /= s
        self._pf_pending = []

    # ── Preflop ───────────────────────────────────────────────────────────────

    def record_preflop_action(self, action_type: int,
                               size_bucket: str, hist_str: str):
        """Store observed opp preflop action; applied in apply_preflop_updates()."""
        if action_type == FOLD:    slot = 0
        elif action_type == RAISE: slot = 2
        else:                      slot = 1
        self._pf_pending.append((slot, size_bucket, hist_str))

    def apply_preflop_updates(self, opp_disc: list, chart: dict,
                               canonicalize):
        """
        Apply stored preflop updates once opponent discards are known.
        Uses the known 5-card hand {kept_pair ∪ opp_disc} to look up exact
        preflop chart probability → precise Bayesian update.
        """
        if not self._pf_pending or self._probs is None:
            return
        disc = [c for c in opp_disc if c >= 0]
        if len(disc) != 3:
            return
        disc_set = set(disc)

        for slot, bkt, hist in self._pf_pending:
            weights = np.empty(len(_ALL_PAIRS), dtype=np.float32)
            for i, (c0, c1) in enumerate(_ALL_PAIRS):
                if self._probs[i] < 1e-9 or c0 in disc_set or c1 in disc_set:
                    weights[i] = 0.
                    continue
                hand5   = [c0, c1] + disc
                key     = (canonicalize(hand5), bkt, hist)
                strat   = chart.get(key)
                if strat is not None and strat.sum() > 0:
                    weights[i] = max(float(strat[slot]) / float(strat.sum()), 1e-6)
                else:
                    weights[i] = 1.0 / 3.0

            new = self._probs * weights
            s   = new.sum()
            if s > 1e-9:
                self._probs = new / s

        self._pf_pending = []

    def get_probs_for_discard(self, chart, canonicalize,
                               dead_cards: list, n_completions: int = 40) -> np.ndarray:
        """
        Return a range distribution adjusted for preflop actions, for use at
        discard time (before opp discards are known).

        Does NOT modify _probs.  Instead samples 5-card completions for each
        valid 2-card pair to compute marginal P(pf_action | pair), returning
        a temporary weighted distribution for EV computation.

        dead_cards: our 5 cards + 3 board cards (8 total).
        n_completions: MC samples per pair (higher = more accurate, ~40 is fast).
        """
        if self._probs is None:
            return None
        if not self._pf_pending:
            return self._probs.copy()

        dead_set  = set(int(c) for c in dead_cards if c >= 0)
        available = [c for c in range(27) if c not in dead_set]
        probs     = self._probs.copy()

        for slot, bkt, hist in self._pf_pending:
            weights = np.ones(len(_ALL_PAIRS), dtype=np.float32)
            for i, (c0, c1) in enumerate(_ALL_PAIRS):
                if probs[i] < 1e-9:
                    continue
                if c0 in dead_set or c1 in dead_set:
                    weights[i] = 0.
                    continue
                pool = [c for c in available if c != c0 and c != c1]
                if len(pool) < 3:
                    weights[i] = 1. / 3
                    continue
                total = 0.0
                for _ in range(n_completions):
                    extras = np.random.choice(pool, 3, replace=False).tolist()
                    hand5  = [c0, c1] + extras
                    key    = (canonicalize(hand5), bkt, hist)
                    strat  = chart.get(key)
                    if strat is not None and strat.sum() > 0:
                        total += float(strat[slot]) / float(strat.sum())
                    else:
                        total += 1. / 3
                weights[i] = max(total / n_completions, 1e-6)

            new = probs * weights
            s   = new.sum()
            if s > 1e-9:
                probs = new / s

        return probs

    # ── Discard ───────────────────────────────────────────────────────────────

    def update_discard(self, opp_disc: list):
        """Remove all pairs containing opponent's discarded cards."""
        if self._probs is None:
            return
        disc_set = set(c for c in opp_disc if c >= 0)
        if not disc_set:
            return
        new = self._probs.copy()
        for i, (a, b) in enumerate(_ALL_PAIRS):
            if a in disc_set or b in disc_set:
                new[i] = 0.
        s = new.sum()
        if s > 1e-9:
            self._probs = new / s

    # ── Postflop ──────────────────────────────────────────────────────────────

    def update_postflop_action(self, action_type, strategy_net,
                               my_bet, opp_bet, board, n_board, street,
                               opp_is_bb, my_disc, opp_disc,
                               aggressor_me, aggressor_opp,
                               n_bets_me, n_bets_opp):
        """Placeholder — postflop range update not yet implemented."""
        pass

    # ── Output ────────────────────────────────────────────────────────────────

    def get_cats(self, board: list, n_board: int) -> np.ndarray:
        """Returns 17-dim hand-category distribution for use as opp_range feature."""
        if self._probs is None:
            return np.ones(17, dtype=np.float32) / 17.0
        return _range_to_cats(self._probs, board, n_board)

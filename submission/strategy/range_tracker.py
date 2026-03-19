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

    def update_postflop_action(
        self,
        action_type: int,
        strategy_net,
        my_bet: int, opp_bet: int,
        board: list, n_board: int, street: int,
        opp_is_bb: bool,
        my_disc: list, opp_disc: list,
        aggressor_me: bool, aggressor_opp: bool,
        n_bets_me: int, n_bets_opp: int,
    ):
        """
        Bayesian update: P(pair) *= P(observed_action | opp_hand = pair).
        Uses batch NN inference with vectorized numpy feature builder instead
        of 91 separate state_to_features() calls.
        """
        if self._probs is None or strategy_net is None:
            return
        valid_idx = [i for i, p in enumerate(self._probs) if p > 1e-9]
        if not valid_idx:
            return

        batch = self._build_batch_feats(
            valid_idx, board, n_board, opp_is_bb,
            my_bet, opp_bet, street,
            my_disc, opp_disc,
            aggressor_me, aggressor_opp, n_bets_me, n_bets_opp,
        )

        with torch.no_grad():
            logits = strategy_net(
                torch.from_numpy(batch).float()
            ).numpy()                               # (n, 8)

        # Softmax → action probability
        logits -= logits.max(axis=1, keepdims=True)
        probs_all = np.exp(logits)
        probs_all /= probs_all.sum(axis=1, keepdims=True)

        # Training-action slots for each game action
        if   action_type == FOLD:   t_slots = [0]
        elif action_type == CALL:   t_slots = [1]
        elif action_type == CHECK:  t_slots = [2]
        elif action_type == RAISE:  t_slots = [3, 4, 5, 6, 7]
        else:
            return

        action_p = probs_all[:, t_slots].sum(axis=1)  # (n,)

        # Bayesian update
        weights = np.ones(len(_ALL_PAIRS), dtype=np.float32)
        for k, i in enumerate(valid_idx):
            weights[i] = max(float(action_p[k]), 1e-6)

        new = self._probs * weights
        s   = new.sum()
        if s > 1e-9:
            self._probs = new / s

    # ── Output ────────────────────────────────────────────────────────────────

    def get_cats(self, board: list, n_board: int) -> np.ndarray:
        """Returns 17-dim hand-category distribution for use as opp_range feature."""
        if self._probs is None:
            return np.ones(17, dtype=np.float32) / 17.0
        return _range_to_cats(self._probs, board, n_board)

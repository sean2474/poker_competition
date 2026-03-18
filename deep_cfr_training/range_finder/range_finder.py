"""RangeFinder — Bayesian range estimator for one player's 2-card hand."""

import ctypes
import numpy as np

from ._lib import lib
from .constants import N_HANDS, N_CATS, pidx, hand2_feat20


class RangeFinder:
    """
    Bayesian range estimator for ONE player's 2-card kept hand distribution.

    Maintains range[351]: P(player holds each candidate pair).

    Phase pipeline:
      init(dead)                      → uniform over non-dead pairs
      update_preflop_action(action)   → heuristic hand-strength update
      remove_cards(board3)            → zero out revealed cards
      update_discard(opp_disc3, board3) → fast_score softmax oracle
      update_from_net(net, action, ...) → NNUE-style batch net inference
      update_action(action_probs)     → generic Bayesian multiply
    """

    def __init__(self):
        self._range = (ctypes.c_float * N_HANDS)()
        self._rng   = np.random.default_rng()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _ptr(self):
        return ctypes.cast(self._range, ctypes.POINTER(ctypes.c_float))

    @staticmethod
    def _int_arr(lst, size=None):
        lst = [int(x) for x in (lst or [])]
        if size is not None:
            lst = (lst + [-1] * size)[:size]
        return (ctypes.c_int * len(lst))(*lst)

    @staticmethod
    def _float_arr(arr):
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # ── Phase API ─────────────────────────────────────────────────────────────

    def init(self, dead_cards):
        """Uniform range over pairs not containing dead_cards."""
        arr = self._int_arr(dead_cards)
        lib.c_range_init(arr, len(dead_cards), self._ptr())

    def remove_cards(self, cards):
        """Zero-out hands containing revealed cards and renormalize."""
        arr = self._int_arr(cards)
        lib.c_range_remove_cards(self._ptr(), arr, len(cards))

    def update_discard(self, opp_disc3, board3):
        """Remove hands containing any discarded card; uniform over survivors.
        (board3 is passed for API compatibility but unused after fast_score removal.)"""
        lib.c_range_update_discard(
            self._ptr(),
            self._int_arr(opp_disc3, size=3),
            self._int_arr(board3,    size=3),
        )

    def update_action(self, action_probs):
        """[Any phase] Generic Bayesian multiply: range *= action_probs, renorm."""
        ap = np.ascontiguousarray(action_probs, dtype=np.float32)
        assert len(ap) == N_HANDS
        lib.c_range_update_action(self._ptr(), self._float_arr(ap))

    def update_from_net(self, net, observed_action: int,
                        community, my_bet: int, opp_bet: int,
                        street: int, opp_is_bb: bool,
                        opp_disc3, our_disc3,
                        street_bets=None, history=None,
                        street_last_ratios=None, street_bet_counts=None,
                        num_acts_this_street: int = 0,
                        device=None):
        """
        [Phase 3/4/5] NNUE-style batch update.

        Computes base features once (hero_hand=dummy), tiles for all active
        candidates, patches dims 0-19 per candidate, runs one batch forward pass.
        """
        import torch
        from game.features import state_to_features

        cands = self.get_candidates(min_prob=0.0)
        if not cands:
            return

        base = state_to_features(
            hero_hand=[-1, -1], community=community,
            my_bet=opp_bet, opp_bet=my_bet,
            street=street, is_bb=opp_is_bb,
            my_discards=opp_disc3, opp_discards=our_disc3,
            street_bets=street_bets, history=history,
            street_last_ratios=street_last_ratios,
            street_bet_counts=street_bet_counts,
            num_actions_this_street=num_acts_this_street,
        )
        base[0:20] = 0.

        N = len(cands)
        feats        = np.tile(base, (N, 1))
        hand_indices = np.empty(N, dtype=np.int32)

        for k, (c0, c1, _) in enumerate(cands):
            feats[k, 0:20]  = hand2_feat20(c0, c1)
            hand_indices[k] = pidx(c0, c1)

        t = torch.from_numpy(feats).float()
        if device is not None:
            t = t.to(device)

        net.eval()
        with torch.no_grad():
            probs = torch.softmax(net(t), dim=1).cpu().numpy()

        action_probs = np.zeros(N_HANDS, dtype=np.float32)
        action_probs[hand_indices] = probs[:, observed_action]
        lib.c_range_update_action(self._ptr(), self._float_arr(action_probs))

    # ── Query ─────────────────────────────────────────────────────────────────

    def get_candidates(self, min_prob: float = 1e-3):
        """List of (c0, c1, prob) for hands above min_prob, sorted descending."""
        hands_out = (ctypes.c_int   * (N_HANDS * 2))()
        probs_out = (ctypes.c_float * N_HANDS)()
        n = lib.c_range_get_candidates(
            self._ptr(), ctypes.c_float(min_prob), hands_out, probs_out
        )
        return [(int(hands_out[i*2]), int(hands_out[i*2+1]), float(probs_out[i]))
                for i in range(n)]

    def get_range_array(self) -> np.ndarray:
        """Raw copy of range[351]."""
        return np.frombuffer(self._range, dtype=np.float32).copy()

    def category_probs(self, board, threshold: float = 1e-3) -> np.ndarray:
        """
        17-dim hand-category probability vector.
        out[k] = sum of range[i] for hands classified as category k.
        Hands below threshold are skipped.
        """
        brd = self._int_arr(board, size=5)
        n_bd = sum(1 for c in board if c >= 0)
        out  = (ctypes.c_float * N_CATS)()
        lib.c_range_to_category_probs(
            self._ptr(), brd, ctypes.c_int(n_bd),
            ctypes.c_float(threshold),
            ctypes.cast(out, ctypes.POINTER(ctypes.c_float))
        )
        return np.frombuffer(out, dtype=np.float32).copy()

    def entropy(self) -> float:
        return float(lib.c_range_entropy(self._ptr()))

    def copy(self) -> 'RangeFinder':
        rf = RangeFinder()
        lib.c_range_copy(self._ptr(), rf._ptr())
        return rf

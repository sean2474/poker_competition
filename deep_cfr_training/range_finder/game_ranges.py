"""GameRanges — dual range tracker (opp + self) for one game from hero's POV."""

import ctypes
import numpy as np

from ._lib import lib
from .constants import N_HANDS, pidx
from .range_finder import RangeFinder


class GameRanges:
    """
    Maintains both range distributions from hero's perspective:

      opp_rf  : what does opp hold?          (updated on opp's actions)
      my_rf   : what does opp think I hold?  (updated on our own actions)

    Phase 3 feature:
      category_features(board) → float32[34]  (17 opp + 17 mine)
      Append to 119-dim base → 153-dim network input.
    """

    def __init__(self, our_hand5, opp_dead_cards=None):
        self.opp_rf = RangeFinder()
        self.my_rf  = RangeFinder()

        dead = list(our_hand5) + (list(opp_dead_cards) if opp_dead_cards else [])
        self.opp_rf.init(dead_cards=dead)
        self.my_rf.init(dead_cards=[])

    # ── Board / card events ───────────────────────────────────────────────────

    def reveal_board(self, board_cards):
        """New board cards dealt — remove from both ranges."""
        self.opp_rf.remove_cards(board_cards)
        self.my_rf.remove_cards(board_cards)

    # ── Discard events ────────────────────────────────────────────────────────

    def observe_opp_discard(self, opp_disc3, board3):
        """Opp discarded 3 specific cards — update opp_rf via discard oracle."""
        self.opp_rf.update_discard(opp_disc3, board3)

    def observe_our_discard(self, our_disc3, board3, our_hand5):
        """We discarded — collapse my_rf to one-hot on our actual kept pair."""
        our_keep = [c for c in our_hand5 if c not in our_disc3]
        if len(our_keep) == 2:
            arr = np.zeros(N_HANDS, dtype=np.float32)
            arr[pidx(our_keep[0], our_keep[1])] = 1.0
            lib.c_range_update_action(
                self.my_rf._ptr(),
                self.my_rf._float_arr(arr)
            )

    # ── Action observations ───────────────────────────────────────────────────

    def observe_opp_action(self, net, action: int, community,
                           my_bet, opp_bet, street, opp_is_bb,
                           opp_disc3, our_disc3, **kwargs):
        """Opp takes action → update opp_rf via NNUE net inference."""
        self.opp_rf.update_from_net(
            net, action, community, my_bet, opp_bet,
            street, opp_is_bb, opp_disc3, our_disc3, **kwargs
        )

    def observe_our_action(self, net, action: int, community,
                           my_bet, opp_bet, street, our_is_bb,
                           our_disc3, opp_disc3, **kwargs):
        """We take action → update my_rf (symmetric, swap bet/disc roles)."""
        self.my_rf.update_from_net(
            net, action, community, opp_bet, my_bet,
            street, our_is_bb, our_disc3, opp_disc3, **kwargs
        )

    # ── Query ─────────────────────────────────────────────────────────────────

    def category_features(self, board, threshold: float = 1e-3) -> np.ndarray:
        """34-dim [opp_cat17 | my_cat17] for Phase 3 network input."""
        return np.concatenate([
            self.opp_rf.category_probs(board, threshold),
            self.my_rf.category_probs(board, threshold),
        ])

    def opp_range_array(self) -> np.ndarray:
        return self.opp_rf.get_range_array()

    def my_range_array(self) -> np.ndarray:
        return self.my_rf.get_range_array()

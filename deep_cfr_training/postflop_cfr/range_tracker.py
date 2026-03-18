"""
Phase 3 range tracking — per-game Bayesian range updates from betting actions.

Phase 1: no range tracking (warmup, heuristic play)
Phase 2: discard-based range only (already in C++ state_to_features)
Phase 3: discard range + betting-action Bayesian update

For each game in the PostflopBatch, maintains:
  opp_rf : opponent range from MY perspective (dead = my_hand + community)
  my_rf  : my range from OPP perspective    (dead = community only)

Betting update uses hand-category heuristic:
  P(bet/raise | cat) = sigmoid((strength - 0.4) * 8)
  P(check/call | cat) = 1 - P(bet/raise | cat)
  where strength = (N_CATS - 1 - cat_idx) / (N_CATS - 1)

NOTE: Phase 3 will upgrade to net-based action probs once convergence is verified.
"""

import ctypes
import numpy as np

from range_finder import RangeFinder, N_CATS, N_HANDS

# ── All 351 hand pairs (precomputed) ─────────────────────────────────────────

_ALL_PAIRS = np.array(
    [(a, b) for a in range(27) for b in range(a + 1, 27)],
    dtype=np.int32,
)  # shape (351, 2)

# ── Board→category lookup cache ──────────────────────────────────────────────

_board_cat_cache: dict = {}   # board_key → np.ndarray(351, int32)
_c_classify_fn = None         # set lazily from game.features


def _get_c_classify():
    global _c_classify_fn
    if _c_classify_fn is None:
        from game.features import _c_lib
        _c_classify_fn = _c_lib.c_classify_hand
    return _c_classify_fn


def get_category_table(community: np.ndarray) -> np.ndarray:
    """
    Returns int32[351] of classify_hand(c0, c1, board) for the current board.
    Cached by board key to avoid recomputation for same board.
    """
    n_comm   = int(np.sum(community >= 0))
    board_key = tuple(int(c) for c in community[:n_comm])
    if board_key in _board_cat_cache:
        return _board_cat_cache[board_key]

    fn    = _get_c_classify()
    board = (ctypes.c_int * 5)(*[int(community[i]) if i < n_comm else -1
                                  for i in range(5)])
    cats  = np.array([fn(_ALL_PAIRS[i, 0], _ALL_PAIRS[i, 1], board, n_comm)
                      for i in range(N_HANDS)], dtype=np.int32)
    _board_cat_cache[board_key] = cats
    return cats


# ── Action probability heuristic ─────────────────────────────────────────────

def action_probs_heuristic(cats: np.ndarray, is_aggressive: bool) -> np.ndarray:
    """
    P(action | hand_category) for all 351 hands using sigmoid hand strength.
    Strong categories (low index) → more likely to bet/raise.
    """
    strength = (N_CATS - 1 - cats.astype(np.float32)) / (N_CATS - 1)
    p_agg    = 1.0 / (1.0 + np.exp(-(strength - 0.4) * 8))
    probs    = p_agg if is_aggressive else (1.0 - p_agg)
    # Clamp to avoid zeros (causes issues with log-prob updates)
    return np.clip(probs, 1e-4, 1.0).astype(np.float32)


# ── Per-game range state ──────────────────────────────────────────────────────

class GameRangeState:
    """
    Tracks my_rf and opp_rf for one game from street 1 onward.

    Initialized once (from discards) when first seen in collect_pending.
    Updated at each subsequent collect_pending call (betting action inferred
    from state transition: was there a bet to call at the NEW node?).
    """

    __slots__ = ('opp_rf', 'my_rf',
                 'prev_my_bet', 'prev_opp_bet', 'prev_cp',
                 'prev_community', 'initialized')

    def __init__(self):
        self.opp_rf        = None
        self.my_rf         = None
        self.prev_my_bet   = -1
        self.prev_opp_bet  = -1
        self.prev_cp       = -1
        self.prev_community = None
        self.initialized   = False

    def init(self, hand2: np.ndarray,
             community: np.ndarray,
             my_disc:   np.ndarray,
             opp_disc:  np.ndarray):
        """
        Initialize from discards — same semantics as C++ _opp_range_features
        and _my_range_features, but maintained in Python for incremental updates.

        opp_rf: dead = my_hand + community, update = opp_disc
        my_rf:  dead = community only,      update = my_disc
        """
        n_comm  = int(np.sum(community >= 0))
        board   = [int(c) for c in community[:n_comm]]
        board3  = (board + [-1, -1, -1])[:3]
        board5  = (board + [-1]*5)[:5]

        # opp_rf: opponent range from my perspective
        dead_opp = [int(hand2[0]), int(hand2[1])]
        self.opp_rf = RangeFinder()
        self.opp_rf.init(dead_cards=dead_opp)
        if n_comm > 0:
            self.opp_rf.remove_cards(board)
        if any(int(c) >= 0 for c in opp_disc):
            self.opp_rf.update_discard([int(c) for c in opp_disc], board3)

        # my_rf: my range from opponent's perspective
        self.my_rf = RangeFinder()
        self.my_rf.init(dead_cards=board if n_comm > 0 else [])
        if any(int(c) >= 0 for c in my_disc):
            self.my_rf.update_discard([int(c) for c in my_disc], board3)

        self.initialized = True

    def update_from_betting(self, to_call: int, prev_cp: int,
                             community: np.ndarray):
        """
        Called when a new node arrives for this game.

        to_call > 0 at the NEW node means the PREVIOUS player was aggressive
        (bet/raise). to_call == 0 means they were passive (check/call).

        prev_cp: which player (0 or 1) acted to reach the current node.
        """
        if not self.initialized:
            return
        is_aggressive = (to_call > 0)
        cats          = get_category_table(community)
        probs         = action_probs_heuristic(cats, is_aggressive)

        if prev_cp == 1:
            # Opponent (opp_rf, from MY perspective) acted
            self.opp_rf.update_action(probs)
        else:
            # I acted (my_rf, from OPP perspective)
            self.my_rf.update_action(probs)

    def category_features(self, community: np.ndarray) -> tuple:
        """
        Returns (my_cats[17], opp_cats[17]) for feature override.
        """
        n_comm  = int(np.sum(community >= 0))
        board5  = list(community[:n_comm]) + [-1] * (5 - n_comm)
        uniform = np.ones(N_CATS, dtype=np.float32) / N_CATS
        my_cats  = self.my_rf.category_probs(board5, 0.0)  if self.my_rf  else uniform
        opp_cats = self.opp_rf.category_probs(board5, 0.0) if self.opp_rf else uniform
        return my_cats, opp_cats


# ── Batch update: override dims 17-50 in feature matrix ─────────────────────

def apply_range_features(feats: np.ndarray,
                          game_infos: tuple,
                          game_states: dict):
    """
    For each pending game, update GameRangeState and override dims [17-50].

    game_infos: return value of batch.get_pending_game_info(cnt)
    game_states: dict[game_idx → GameRangeState]

    Returns feats (modified in-place).
    """
    hand2_arr, comm_arr, my_disc_arr, opp_disc_arr, \
    cp_arr, bet_cp_arr, bet_op_arr, gidx_arr = game_infos

    cnt = len(gidx_arr)
    for i in range(cnt):
        gidx      = int(gidx_arr[i])
        hand2     = hand2_arr[i]
        community = comm_arr[i]
        my_disc   = my_disc_arr[i]
        opp_disc  = opp_disc_arr[i]
        cp        = int(cp_arr[i])
        bet_cp    = int(bet_cp_arr[i])
        bet_op    = int(bet_op_arr[i])
        to_call   = max(bet_op - bet_cp, 0)

        if gidx not in game_states:
            gs = GameRangeState()
            gs.init(hand2, community, my_disc, opp_disc)
            game_states[gidx] = gs
        else:
            gs = game_states[gidx]
            # Update from previous action
            if gs.prev_cp >= 0:
                gs.update_from_betting(to_call, gs.prev_cp, community)

        # Record current state for next call
        gs.prev_cp      = cp
        gs.prev_my_bet  = bet_cp
        gs.prev_opp_bet = bet_op

        # Override dims 17-50 with Python-computed range categories
        if gs.initialized:
            my_cats, opp_cats = gs.category_features(community)
            feats[i, 17:34] = my_cats   # my_range_cats (opp's view of me)
            feats[i, 34:51] = opp_cats  # opp_range_cats (my view of opp)

    return feats

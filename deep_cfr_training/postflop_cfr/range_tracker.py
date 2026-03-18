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

from range_finder import N_CATS, N_HANDS

# ── All 351 hand pairs (precomputed) ─────────────────────────────────────────

_ALL_PAIRS = np.array(
    [(a, b) for a in range(27) for b in range(a + 1, 27)],
    dtype=np.int32,
)  # shape (351, 2)

# ── Per-card dead-mask: _CARD_MASK[c] = bool[351], True if pair contains c ───
# Used for vectorized range initialization and discard update.
_CARD_MASK = np.zeros((27, N_HANDS), dtype=bool)
for _i, (_a, _b) in enumerate(_ALL_PAIRS.tolist()):
    _CARD_MASK[_a, _i] = True
    _CARD_MASK[_b, _i] = True


def _dead_mask(cards) -> np.ndarray:
    """Returns bool[351]: True for pairs containing any card in `cards`."""
    m = np.zeros(N_HANDS, dtype=bool)
    for c in cards:
        if 0 <= int(c) < 27:
            m |= _CARD_MASK[int(c)]
    return m


def _init_range(dead_cards) -> np.ndarray:
    """float32[351]: uniform over non-dead pairs."""
    r = np.where(_dead_mask(dead_cards), 0., 1.).astype(np.float32)
    s = r.sum()
    return r / s if s > 0 else r


def _apply_discard(r: np.ndarray, disc3) -> np.ndarray:
    """Remove impossible hands (discard-contained), renormalize."""
    m = _dead_mask([c for c in disc3 if int(c) >= 0])
    r[m] = 0.
    s = r.sum()
    if s > 0: r /= s
    return r


def _update_action(r: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Bayesian multiply: r *= probs, renormalize."""
    r *= probs
    s = r.sum()
    if s > 0: r /= s
    return r

# ── Board→category lookup cache ──────────────────────────────────────────────

_board_cat_cache:   dict = {}   # board_key → int32[351]
_board_onehot_cache: dict = {} # board_key → float32[351, 17]  (for matmul)
_act_probs_cache:   dict = {}  # (board_key, is_aggressive) → float32[351]
_c_classify_fn = None         # set lazily from game.features


def _get_c_classify():
    global _c_classify_fn
    if _c_classify_fn is None:
        from game.features import _c_lib
        _c_classify_fn = _c_lib.c_classify_hand
    return _c_classify_fn


def get_onehot_table(community: np.ndarray) -> np.ndarray:
    """
    Returns float32[351, 17] one-hot table for matmul-based category features.
    Cached per board. Enables: range (k,351) @ onehot (351,17) = cats (k,17).
    """
    n_comm    = int(np.sum(community >= 0))
    board_key = tuple(int(c) for c in community[:n_comm])
    if board_key in _board_onehot_cache:
        return _board_onehot_cache[board_key]
    cats  = get_category_table(community)
    oh    = np.zeros((N_HANDS, N_CATS), dtype=np.float32)
    oh[np.arange(N_HANDS), cats] = 1.
    _board_onehot_cache[board_key] = oh
    return oh


def get_action_probs(community: np.ndarray, is_aggressive: bool) -> np.ndarray:
    """
    Returns float32[351] action probs for (board, aggressive) — cached.
    Avoids recomputing sigmoid for same board+action across all pending games.
    """
    n_comm    = int(np.sum(community >= 0))
    board_key = tuple(int(c) for c in community[:n_comm])
    key       = (board_key, bool(is_aggressive))
    if key in _act_probs_cache:
        return _act_probs_cache[key]
    cats  = get_category_table(community)
    probs = action_probs_heuristic(cats, is_aggressive)
    _act_probs_cache[key] = probs
    return probs


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
    Tracks opp_range and my_range for one game as plain numpy float32[351].

    Pure numpy (no RangeFinder objects) — eliminates C++ ctypes overhead for
    range maintenance.  Only get_category_table uses C++ (cached per board).
    """

    __slots__ = ('opp_range', 'my_range', 'prev_cp', 'initialized')

    def __init__(self):
        self.opp_range  = None
        self.my_range   = None
        self.prev_cp    = -1
        self.initialized = False

    def init(self, hand2: np.ndarray,
             community: np.ndarray,
             my_disc:   np.ndarray,
             opp_disc:  np.ndarray):
        """Initialize ranges from discards using vectorized numpy ops."""
        # opp_range: dead = my_hand + community, update = opp_disc
        dead_opp = list(hand2) + [int(c) for c in community if int(c) >= 0]
        self.opp_range = _init_range(dead_opp)
        if any(int(c) >= 0 for c in opp_disc):
            self.opp_range = _apply_discard(self.opp_range, opp_disc)

        # my_range: dead = community only, update = my_disc
        dead_my = [int(c) for c in community if int(c) >= 0]
        self.my_range = _init_range(dead_my)
        if any(int(c) >= 0 for c in my_disc):
            self.my_range = _apply_discard(self.my_range, my_disc)

        self.initialized = True

    def update_from_betting(self, to_call: int, prev_cp: int,
                             community: np.ndarray):
        """Bayesian update using cached action_probs (board+aggressive memoized)."""
        if not self.initialized:
            return
        probs = get_action_probs(community, to_call > 0)   # cached per board+agg
        if prev_cp == 1:
            self.opp_range = _update_action(self.opp_range, probs)
        else:
            self.my_range  = _update_action(self.my_range,  probs)

    def category_features(self, community: np.ndarray) -> tuple:
        """Returns (my_cats[17], opp_cats[17]) via np.bincount (no C++ calls)."""
        uniform    = np.ones(N_CATS, dtype=np.float32) / N_CATS
        if not self.initialized:
            return uniform, uniform
        cats_table = get_category_table(community)   # cached int32[351]

        def _cats(r):
            out = np.bincount(cats_table,
                              weights=r.astype(np.float64),
                              minlength=N_CATS).astype(np.float32)
            s = out.sum()
            return out / s if s > 0 else uniform

        return _cats(self.my_range), _cats(self.opp_range)


# ── Batch update: override dims 17-50 in feature matrix ─────────────────────

_UNIFORM17 = np.ones(N_CATS, dtype=np.float32) / N_CATS


def apply_range_features(feats: np.ndarray,
                          game_infos: tuple,
                          game_states: dict):
    """
    Update GameRangeState per game and override dims [17-50] in feats.

    Optimized:
    1. action_probs cached by (board, aggressive) → no repeated sigmoid
    2. category_features batched per board via matmul (range @ onehot)
       eliminates per-game Python function call overhead
    """
    hand2_arr, comm_arr, my_disc_arr, opp_disc_arr, \
    cp_arr, bet_cp_arr, bet_op_arr, gidx_arr = game_infos
    cnt = len(gidx_arr)

    # ── Phase 1: init / collect updates ──────────────────────────────────────
    active  = []                             # (i, gs, comm) for initialized
    pending = []                             # (gs, is_agg, prev_cp) to update

    for i in range(cnt):
        gidx    = int(gidx_arr[i])
        hand2   = hand2_arr[i]
        comm    = comm_arr[i]
        to_call = max(int(bet_op_arr[i]) - int(bet_cp_arr[i]), 0)

        if gidx not in game_states:
            gs = GameRangeState()
            gs.init(hand2, comm, my_disc_arr[i], opp_disc_arr[i])
            game_states[gidx] = gs
        else:
            gs = game_states[gidx]
            if gs.prev_cp >= 0:
                pending.append((gs, to_call > 0, gs.prev_cp, comm))

        gs.prev_cp = int(cp_arr[i])
        if gs.initialized:
            active.append((i, gs, comm))

    # ── Batch apply pending range updates grouped by (board, is_agg) ────────
    if pending:
        from collections import defaultdict
        upd_groups: dict = defaultdict(list)  # (board_key, is_agg, which) → [gs]
        for gs, is_agg, prev_cp, comm in pending:
            n_c = int(np.sum(comm >= 0))
            bk  = tuple(int(c) for c in comm[:n_c])
            which = 'opp' if prev_cp == 1 else 'my'
            upd_groups[(bk, is_agg, which)].append((gs, comm))

        for (bk, is_agg, which), group in upd_groups.items():
            probs = get_action_probs(group[0][1], is_agg)  # cached
            k = len(group)
            if k == 1:
                gs = group[0][0]
                r = gs.opp_range if which == 'opp' else gs.my_range
                r *= probs; s = r.sum()
                if s > 0: r /= s
            else:
                # Batch multiply: (k, 351) *= probs[None, :]
                ranges = np.stack(
                    [gs.opp_range if which == 'opp' else gs.my_range
                     for gs, _ in group])
                ranges *= probs[np.newaxis, :]
                norms = ranges.sum(axis=1, keepdims=True)
                np.divide(ranges, np.maximum(norms, 1e-9), out=ranges)
                for j, (gs, _) in enumerate(group):
                    if which == 'opp':
                        gs.opp_range = ranges[j]
                    else:
                        gs.my_range  = ranges[j]

    if not active:
        return feats

    # ── Phase 2: batch category_features grouped by board ────────────────────
    from collections import defaultdict
    board_groups: dict = defaultdict(list)   # board_key → [(feat_idx, gs)]
    board_comm:  dict = {}                   # board_key → community array

    for i, gs, comm in active:
        n_c = int(np.sum(comm >= 0))
        key = tuple(int(c) for c in comm[:n_c])
        board_groups[key].append((i, gs))
        if key not in board_comm:
            board_comm[key] = comm

    for board_key, group in board_groups.items():
        oh = get_onehot_table(board_comm[board_key])   # (351, 17) cached
        k  = len(group)

        # Stack all ranges: (k, 351)
        my_stack  = np.empty((k, N_HANDS), dtype=np.float32)
        opp_stack = np.empty((k, N_HANDS), dtype=np.float32)
        for j, (_, gs) in enumerate(group):
            my_stack[j]  = gs.my_range
            opp_stack[j] = gs.opp_range

        # Batch matmul: (k, 351) @ (351, 17) → (k, 17)
        my_cats  = my_stack  @ oh
        opp_cats = opp_stack @ oh

        # Normalize row-wise
        my_s  = my_cats.sum(axis=1,  keepdims=True)
        opp_s = opp_cats.sum(axis=1, keepdims=True)
        np.divide(my_cats,  np.maximum(my_s,  1e-9), out=my_cats)
        np.divide(opp_cats, np.maximum(opp_s, 1e-9), out=opp_cats)

        for j, (feat_idx, _) in enumerate(group):
            feats[feat_idx, 17:34] = my_cats[j]
            feats[feat_idx, 34:51] = opp_cats[j]

    return feats

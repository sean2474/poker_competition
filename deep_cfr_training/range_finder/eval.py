"""Rangefinder accuracy evaluation — training-only, not needed at inference."""

import numpy as np

from .range_finder import RangeFinder
from .constants import N_HANDS, pidx


def eval_rangefinder_mse(trainer, n_games: int = 100) -> dict:
    """
    MSE and true-hand probability for rangefinder accuracy measurement.

    Called by postflop_cfr/runner.py at checkpoint intervals.
    Returns:
      mse_uniform / mse_post_discard : MSE vs one-hot true hand
      p_true_uniform / p_true_post_discard : mean P(true hand)
    """
    from game.features import batch_deal_discard

    r = batch_deal_discard(n_games)
    _, _, p0d, p1d, comms, p0h5, p1h5 = r

    mse_uniform = [];      mse_post_discard = []
    p_true_uniform = [];   p_true_discard   = []

    for i in range(n_games):
        board3   = list(comms[i][:3])
        h5_our   = list(p0h5[i])
        h5_opp   = list(p1h5[i])
        opp_disc = list(p1d[i])
        opp_keep = [c for c in h5_opp if c not in opp_disc]
        if len(opp_keep) != 2:
            continue

        true_idx = pidx(opp_keep[0], opp_keep[1])
        true_hot = np.zeros(N_HANDS, dtype=np.float32)
        true_hot[true_idx] = 1.0

        rf = RangeFinder()
        rf.init(dead_cards=h5_our)
        arr = rf.get_range_array()
        mse_uniform.append(float(np.mean((arr - true_hot) ** 2)))
        p_true_uniform.append(float(arr[true_idx]))

        rf.remove_cards(board3)
        rf.update_discard(opp_disc, board3)
        arr2 = rf.get_range_array()
        mse_post_discard.append(float(np.mean((arr2 - true_hot) ** 2)))
        p_true_discard.append(float(arr2[true_idx]))

    def _m(lst): return float(np.mean(lst)) if lst else 0.0
    return {
        'mse_uniform':         _m(mse_uniform),
        'mse_post_discard':    _m(mse_post_discard),
        'p_true_uniform':      _m(p_true_uniform),
        'p_true_post_discard': _m(p_true_discard),
        'n_games':             len(mse_uniform),
    }

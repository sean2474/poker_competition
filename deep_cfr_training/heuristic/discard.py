"""
DiscardHeuristic — fast_discard C++ heuristic as IDiscardModel.

Uses fast_score softmax sampling. opp_cats / is_bb are accepted but ignored.
"""

import ctypes
import os as _os
import random
import numpy as np

from interfaces import IDiscardModel
from discard_cfr.features import KEEP_PAIRS, N_KEEP_PAIRS

_cpp_dir  = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'cpp')
_trav_lib = None
for _ext in ['libtraversal.so', 'libtraversal.dylib']:
    _p = _os.path.join(_cpp_dir, _ext)
    if _os.path.exists(_p):
        _trav_lib = ctypes.CDLL(_p)
        break

if _trav_lib is not None:
    _trav_lib.c_fast_discard.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_uint,
        ctypes.c_float,
    ]


class DiscardHeuristic(IDiscardModel):
    """C++ fast_score softmax discard heuristic."""

    def __init__(self, temperature: float = 0.05):
        self.temperature = temperature

    def get_strategy(self, hand5, board3,
                     opp_cats: np.ndarray = None,
                     is_bb: bool = False) -> np.ndarray:
        if _trav_lib is None:
            return np.ones(N_KEEP_PAIRS, dtype=np.float32) / N_KEEP_PAIRS

        h5   = (ctypes.c_int * 5)(*[int(c) for c in hand5[:5]])
        b3   = (ctypes.c_int * 3)(*[int(c) for c in board3[:3]])
        ki   = ctypes.c_int()
        kj   = ctypes.c_int()
        seed = random.randint(0, 2**31)
        _trav_lib.c_fast_discard(h5, b3, ctypes.byref(ki), ctypes.byref(kj),
                                 ctypes.c_uint(seed),
                                 ctypes.c_float(self.temperature))
        chosen_ki, chosen_kj = ki.value, kj.value
        for idx, (ai, aj) in enumerate(KEEP_PAIRS):
            if hand5[ai] == chosen_ki and hand5[aj] == chosen_kj:
                strat = np.zeros(N_KEEP_PAIRS, dtype=np.float32)
                strat[idx] = 1.0
                return strat
        return np.ones(N_KEEP_PAIRS, dtype=np.float32) / N_KEEP_PAIRS

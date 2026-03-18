"""ctypes loader for librangefinder — private, do not import directly."""

import os
import ctypes

_cpp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cpp')
lib = None
for _ext in ['libtraversal.so', 'libtraversal.dylib',
             'librangefinder.so', 'librangefinder.dylib']:
    _p = os.path.join(_cpp_dir, _ext)
    if os.path.exists(_p):
        try:
            lib = ctypes.CDLL(_p)
            lib.c_range_init  # probe — raises AttributeError if not in this lib
            break
        except AttributeError:
            lib = None

if lib is None:
    raise RuntimeError(
        f"librangefinder / libtraversal not found in {_cpp_dir}.\n"
        "Run: clang++ -O3 -shared -fPIC -std=c++17 -o "
        "libtraversal.dylib traversal.cpp rangefinder.cpp"
    )

# ── Function signatures ───────────────────────────────────────────────────────

lib.c_range_init.argtypes = [
    ctypes.POINTER(ctypes.c_int), ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
]
lib.c_range_remove_cards.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_int), ctypes.c_int,
]
lib.c_range_update_discard.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
]
lib.c_range_update_action.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
]
lib.c_range_get_candidates.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.c_float,
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float),
]
lib.c_range_get_candidates.restype = ctypes.c_int
lib.c_range_entropy.argtypes = [ctypes.POINTER(ctypes.c_float)]
lib.c_range_entropy.restype  = ctypes.c_float
lib.c_range_copy.argtypes    = [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
]
lib.c_range_to_category_probs.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.c_float,
    ctypes.POINTER(ctypes.c_float),
]

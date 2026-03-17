"""Backward-compat shim — all symbols re-exported from game/ package."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from game import *  # noqa: F401,F403
from game.constants import *  # noqa: F401,F403

MAX_BET     = 100
SMALL_BLIND = 1
BIG_BLIND   = 2

# Abstract action IDs (shared preflop + postflop, routing done at inference)
A_FOLD       = 0
A_CALL       = 1
A_CHECK      = 2
A_BET_SMALL  = 3   # postflop: 33% pot  |  preflop: unused (check only)
A_BET_LARGE  = 4   # postflop: 75% pot
A_RAISE_SMALL = 5  # postflop: 33% pot  |  preflop: single open/re-raise size
A_RAISE_LARGE = 6  # postflop: 75% pot  |  preflop: unused
A_BET_POT    = 7   # postflop: 100% pot |  preflop: unused

NUM_ACTIONS = 8
FEATURE_DIM = 93   # see features.py for layout

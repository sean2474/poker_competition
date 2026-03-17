MAX_BET     = 100
SMALL_BLIND = 1
BIG_BLIND   = 2

# Abstract action IDs (shared preflop + postflop, routing done at inference)
A_FOLD       = 0
A_CALL       = 1
A_CHECK      = 2
A_BET_SMALL  = 3   # postflop: 33% pot
A_BET_LARGE  = 4   # postflop: 75% pot
A_BET_POT    = 7   # postflop: 100% pot
A_RAISE_SMALL = 5  # postflop: 33% pot
A_RAISE_LARGE = 6  # postflop: 75% pot

NUM_ACTIONS     = 8
FEATURE_DIM = 119  # C++ c_state_features outputs all 119 dims directly

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
FEATURE_DIM = 77   # range-based: my_cat(17)+my_range(17)+opp_range(17)+board(8)+line(6)+pot(4)+blocker(4)+street(3)+pos(1)

"""
pytest engineTest.py -v

Test suite for the BS AEC Environment (BSEnv.py).

Coverage targets:
    - Initialization invariants
    - playCards transitions
    - passChallenge transitions
    - challenge transitions (honest and bluff)
    - Win condition detection (both paths)
    - Cross-cutting card count invariant
    - Rewards are added
"""

# Action space:
# 0 declare 1 card, 0 of which are honest
# 1 declare 1 card, 1 of which are honest
# 2 declare 2 cards, 0 of which are honest
# 3 declare 2 cards, 1 of which are honest
# 4 d 2 c, 2 o w a h
# 5 3, 0
# 6 3, 1
# 7 3, 2
# 8 3, 3
# 9 4, 0
# 10 4, 1
# 11 4, 2
# 12 4, 3
# 13 4, 4
# 14 Challenge
# 15 Pass challenge

import core.BSEnv as BSEnv
from pettingzoo.test import api_test

api_test(BSEnv.BSEnv(), num_cycles=1000)

env = BSEnv.BSEnv(renderMode="human", max_iter=1)
env.reset(seed=42) # maintain seed 42!

# TODO: implement baseline test for environment function

# Engine legality
# Turn progression
# Observation privacy
# action masking
# reward event


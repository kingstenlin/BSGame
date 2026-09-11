from core import Action, GameState
import core.utils
import random
from RL.utils import *
# given a masked action space and an observation space, return a decision

# Observation vector layout:
# [0:13]: counts of each rank, normalized by 4
# [13, 14]: cyclic encoding of current rank
# [15]: pile size, normalized by deck size
# [16]: claim quantity, normalized by 4 (0 if no claim)
# [17]: current phase (0 for DECLARE, 1 for CHALLENGE)
# Next NUM_PLAYERS: hand sizes, normalized by deck size

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

# -------------------------
# play card behavior
def play_Honest(obs, act):
    #plays all possible truthful cards. if not possible, minimally lies
    d = {1 : 1, 2 : 4, 3 : 8, 4 : 13} # the action corresponding to playing honest {key} cards
    handArr = getHandArr(obs)
    rankInd = getRankToPlay(obs)[0]
    if rankInd != 0:
        return d[handArr[rankInd]]
    else:
        return 0

def play_Menace(obs, act):
    # always four card lie
    return 12

def play_Wild(obs, act):
    # sneak in a four card lie every once in a while. otherwise honest
    x = random.random()
    if x < 0.2:
        return 12
    else:
        return play_Honest(obs, act)
# --------------------------
# challenge behavior

def challenge_NonConfrontational(obs, act):
    return 15 # never challenge

def challenge_Menace(obs, act):
    return 14 # always challenge

def challenge_BigNumberHater(obs, act):
    if getPrevClaim(obs)[0]: # if more than two cards, challenge
        return 14
    else:
        return 15

def challenge_LogicalConservative(obs, act):
    # if agent's own hand denies, call it out
    rankInd = getRankToPlay(obs)[0]
    ct = getPrevClaim(obs)[0]
    if obs[rankInd] > (4 - ct):
        return 14
    else:
        return 15

def challenge_Random(obs, act):
    return random.sample([14, 15], 1)
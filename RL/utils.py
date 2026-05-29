import core.BSEnv as BSEnv
import core.GameState as GameState
import numpy as np

indToString = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]


def getRankToPlay(obs) -> (int, str):
    """
    :param obs:
    :return: rankInd: int, rank as string
    """
    a = obs[13]
    b = obs[14]
    if a > 0:
        rankInd = int(np.round(13 * np.arccos(2 * (b - 0.5)) / 2 / np.pi))
    else:
        rankInd = 13 - int(np.round(13 * np.arccos(2 * (b - 0.5)) / 2 / np.pi))

    return rankInd, indToString[rankInd]

def getPhase(act) -> GameState.Phase:
    return GameState.Phase.DECLARE if act[14] == 0 else GameState.Phase.CHALLENGE


def getPrevClaim(obs) -> (int, str):
    """

    :param obs: Observation vector
    :return: count: int, rank: str
    """
    return int(np.round(obs[16] * 4)), getRankToPlay(obs)[1]


def getHandArr(obs):
    """returns an array (following rankInd) of counts"""
    arr = [0] * 13
    for i in range(13):
        arr[i] += int(round(obs[i] * 4))

    return arr




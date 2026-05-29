import core.BSEnv as BSEnv
import numpy as np
from RL.utils import *
indToString = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]

env = BSEnv.BSEnv()
env.reset(seed=42)

verboseActionsDict = {i : f"{honest} true cards and {quantity - honest} false cards"
for i, (quantity, honest) in enumerate(BSEnv.declareActions)}
verboseActionsDict[14] = "Challenge"
verboseActionsDict[15] = "Pass"

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        actionSpace = info["action_mask"]

        print("-" * 40)
        print(f"Player {agent}'s turn to {"DECLARE" if actionSpace[14] == 0 else "CHALLENGE"}")

        handString = ""
        for i in range(13):
            if observation[i] != 0:
                handString += (indToString[i] * int(observation[i] * 4))
        print(f"Hand of length {len(handString)}: {", ".join(handString)}")

        print(f"Current pile size: {int(np.round(observation[15] * 52))}")

        # decode the rank to play from the cyclic encoding
        a = observation[13]
        b = observation[14]
        if a > 0:
            rankInd = int(np.round(13 * np.arccos(2 * (b - 0.5)) / 2 / np.pi))
        else:
            rankInd = 13 - int(np.round(13 * np.arccos(2 * (b - 0.5)) / 2 / np.pi))

        if actionSpace[14] != 0:
            # declare specifics
            print(f"Previous claim: {int(np.round(observation[16] * 4))} copies of {indToString[rankInd]}")
        else:
           # play specifics
            print(f"Rank to play: {indToString[rankInd]}")
        print("Possible actions: ")
        print("*" * 10)

        for i in range(len(actionSpace)):
            if actionSpace[i] != 0:
                print(f"Action {i}: {verboseActionsDict[i]}")
        print("*" * 10)
        action = int(input("Input action (int): "))

        while actionSpace[action] == 0:
            action = int(input("Invalid. Try again (int): "))

    env.step(action)

env.close()
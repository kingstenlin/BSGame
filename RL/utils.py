from __future__ import annotations

import os
import torch

from core.BSEnv import OBS_DIM, NUM_ACTIONS

from RL.agents import PolicyAgent
from RL.trainWithPool import BSPolicy
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

"""
checkpoint_utils.py — Load trained PolicyAgent instances from disk.

Deliberately kept separate from agents.py: BSPolicy is defined in
trainWithPool.py, and trainWithPool.py imports PolicyAgent from
agents.py. If agents.py imported BSPolicy back from trainWithPool.py,
that would be a circular import. This module sits "above" both and
imports from each freely.

Usage
─────
    from RL.checkpoint_utils import load_policy_agent

    bot_agent = load_policy_agent("checkpoints/policy_0050000.pt")
    session.add_bot(player_id, bot_agent)
"""


def load_policy_agent(
    checkpoint_path: str,
    device: str = "cpu",
    hidden_dim: int = 128,
    is_trainable: bool = False,
) -> PolicyAgent:
    """
    Reconstruct a BSPolicy and load its weights from a training
    checkpoint, wrapping the result in a PolicyAgent ready to `.act()`.

    checkpoint_path : path to a .pt file. Supports both the full
                       trainer checkpoint dict written by
                       BSTrainer.save() (which has a "policy" key
                       alongside optimizer state, pool snapshots,
                       etc.) and a bare state_dict saved on its own.
    device          : "cpu" or "cuda". Bots served over a websocket
                       are almost always fine on "cpu" — no need to
                       fight the training process for GPU memory.
    hidden_dim      : must match what the checkpoint was trained
                       with. trainWithPool.py's default is 128; if
                       you trained with a different --hidden-dim
                       (or add a CLI flag for it later), pass that
                       value here or loading will fail with a
                       state_dict shape mismatch.
    is_trainable    : keep False for serving. PolicyAgent only computes
                       log_prob/value (for a training buffer) when
                       True — irrelevant and wasteful for a bot that's
                       just playing games.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint at {checkpoint_path!r}")

    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict) and "policy" in ckpt:
        state_dict = ckpt["policy"]
    else:
        # Bare state_dict, saved without the surrounding trainer dict.
        state_dict = ckpt

    policy = BSPolicy(OBS_DIM, NUM_ACTIONS, hidden_dim)
    policy.load_state_dict(state_dict)
    policy.to(device)
    policy.eval()  # no-op for dropout/batchnorm here, but cheap and correct habit

    return PolicyAgent(policy, device, is_trainable=is_trainable)




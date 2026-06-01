"""
bots.py – Pluggable bot agents for the BS Card Game server
------------------------------------------------------------
All bots must subclass BaseBot and implement:

    def act(self, obs: np.ndarray, action_mask: np.ndarray) -> int

`obs` and `action_mask` are exactly what BSEnv provides (the raw numpy arrays
from env.last()).  Return an integer action index.

To add your own bot:
  1. Subclass BaseBot
  2. Implement act()
  3. Register it in server.py's BOT_REGISTRY dict

Bundled bots
------------
  RandomBot  – picks uniformly at random from valid actions
  HonestBot  – always plays honestly (prefers true cards) and never challenges
"""

import random
from abc import ABC, abstractmethod

import numpy as np

import core.BSEnv as BSEnv

IND_TO_STRING = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]


# ── base class ────────────────────────────────────────────────────────────────

class BaseBot(ABC):
    """All bots must subclass this."""

    @abstractmethod
    def act(self, obs: np.ndarray, action_mask: np.ndarray) -> int:
        """
        Parameters
        ----------
        obs         : observation vector from BSEnv (same as env.last()[0])
        action_mask : binary mask from info["action_mask"] (same as env.last()[4])

        Returns
        -------
        int : a valid action index
        """


# ── helpers ───────────────────────────────────────────────────────────────────

def _valid(action_mask) -> list[int]:
    return [i for i, v in enumerate(action_mask) if v != 0]


def _decode_rank(obs) -> int:
    a, b = obs[13], obs[14]
    angle = np.arccos(np.clip(2 * (b - 0.5), -1, 1))
    idx = int(np.round(13 * angle / (2 * np.pi)))
    if a <= 0:
        idx = 13 - idx
    return max(0, min(12, idx))


def _hand_counts(obs) -> list[int]:
    return [int(obs[i] * 4) for i in range(13)]


def _is_challenge_phase(action_mask) -> bool:
    return bool(action_mask[14] != 0)


# ── RandomBot ─────────────────────────────────────────────────────────────────

class RandomBot(BaseBot):
    """Chooses uniformly at random from all valid actions."""

    def act(self, obs, action_mask) -> int:
        return random.choice(_valid(action_mask))


# ── HonestBot ─────────────────────────────────────────────────────────────────

class HonestBot(BaseBot):
    """
    Declare phase: plays as many TRUE copies of the required rank as possible.
    If it has none, plays the action with the fewest bluff cards.
    Challenge phase: never challenges (passes when allowed).
    """

    def act(self, obs, action_mask) -> int:
        valid = _valid(action_mask)

        if _is_challenge_phase(action_mask):
            # prefer Pass (15) over Challenge (14)
            if 15 in valid:
                return 15
            return 14   # forced challenge

        # ── declare phase ──
        rank_idx = _decode_rank(obs)
        hand = _hand_counts(obs)
        true_in_hand = hand[rank_idx]

        # find declare actions with exactly `true_in_hand` honest cards
        # (or as many as possible), fewest bluffs
        best_action = None
        best_score = float("inf")   # lower honest_deficit + bluffs is better

        for action_id in valid:
            if action_id >= len(BSEnv.declareActions):
                continue   # skip Challenge/Pass
            quantity, honest = BSEnv.declareActions[action_id]
            bluff = quantity - honest
            deficit = max(0, honest - true_in_hand)
            score = deficit * 10 + bluff   # minimise lying
            if score < best_score:
                best_score = score
                best_action = action_id

        if best_action is not None:
            return best_action
        # fallback
        return valid[0]


# ── AggressiveBot ─────────────────────────────────────────────────────────────

class AggressiveBot(BaseBot):
    """
    Declare phase : always plays the maximum quantity (lots of bluff).
    Challenge phase: challenges 70% of the time.
    """

    def act(self, obs, action_mask) -> int:
        valid = _valid(action_mask)

        if _is_challenge_phase(action_mask):
            if 14 in valid and random.random() < 0.70:
                return 14   # Challenge
            if 15 in valid:
                return 15   # Pass
            return valid[0]

        # pick declare action with highest quantity (most cards claimed)
        declare_valid = [a for a in valid if a < len(BSEnv.declareActions)]
        if declare_valid:
            return max(declare_valid, key=lambda a: BSEnv.declareActions[a][0])
        return valid[0]


# ── ConservativeBot ───────────────────────────────────────────────────────────

class ConservativeBot(BaseBot):
    """
    Declare phase : plays the minimum quantity possible.
    Challenge phase: never challenges.
    """

    def act(self, obs, action_mask) -> int:
        valid = _valid(action_mask)

        if _is_challenge_phase(action_mask):
            if 15 in valid:
                return 15
            return valid[0]

        declare_valid = [a for a in valid if a < len(BSEnv.declareActions)]
        if declare_valid:
            return min(declare_valid, key=lambda a: BSEnv.declareActions[a][0])
        return valid[0]
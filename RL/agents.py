"""
agents.py — Agent abstractions for BSEnv opponent pool.

All agents expose a single method:
    act(obs, mask) -> (action, log_prob, value)

log_prob and value are None for non-policy agents. The trainer
uses this to decide whether to buffer a transition for training.

Naive agents
────────────
Three rule-based strategies that cover different behavioural extremes.
Having variety in the naive pool forces the learning agent to handle
multiple exploit patterns rather than specialising against one.

    RandomAgent       — uniform random over legal actions
    ConservativeAgent — never challenges, plays as honestly as possible
    AggressiveAgent   — always bluffs maximally, always challenges
    ThresholdAgent    — reasonable baseline: honest when possible,
                        challenges when pile looks suspicious

Observation index reference (from BSEnv):
    [0:13]  own hand composition, counts normalised by 4
    [13]    sin component of current rank (cyclic)
    [14]    cos component of current rank (cyclic)
    [15]    pile size / 52
    [16]    last claim quantity / 4
    [17]    phase  (0 = DECLARE, 1 = CHALLENGE)
    [18:21] hand sizes / 52
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
import torch

# Action-space constants duplicated here to avoid circular import with BSEnv
_DECLARE_ACTIONS = [
    (qty, honest)
    for qty in range(1, 5)
    for honest in range(0, qty + 1)
]  # indices 0-13
_ACTION_CHALLENGE = 14
_ACTION_PASS      = 15

# Observation indices
_IDX_PILE  = 15
_IDX_PHASE = 17

ActResult = Tuple[int, Optional[float], Optional[float]]  # (action, log_prob, value)


# ────────────────────────────────────────────────────────────────────────────
# Abstract base
# ────────────────────────────────────────────────────────────────────────────

class Agent(ABC):
    """
    Minimal interface shared by all agents (policy and rule-based).

    act() returns (action, log_prob, value).
    log_prob and value are None for rule-based agents — the trainer uses
    their presence to decide whether to buffer the transition.
    """

    @abstractmethod
    def act(self, obs: np.ndarray, mask: np.ndarray) -> ActResult:
        ...

    @staticmethod
    def _valid_actions(mask: np.ndarray) -> np.ndarray:
        return np.where(mask)[0]

    @staticmethod
    def _is_challenge_phase(obs: np.ndarray) -> bool:
        return obs[_IDX_PHASE] > 0.5


# ────────────────────────────────────────────────────────────────────────────
# Policy agent (wraps a BSPolicy — used for current model and past checkpoints)
# ────────────────────────────────────────────────────────────────────────────

class PolicyAgent(Agent):
    """
    Wraps a BSPolicy. Returns log_prob and value so the trainer can
    buffer transitions for the PPO update.

    Set is_trainable=True for the current model, False for past
    checkpoints. The flag is informational — the trainer checks it
    to decide whether to add transitions to the training buffer.
    """

    def __init__(self, policy, device: str, is_trainable: bool = False):
        self.policy      = policy
        self.device      = device
        self.is_trainable = is_trainable

    @torch.no_grad()
    def act(self, obs: np.ndarray, mask: np.ndarray) -> ActResult:
        obs_t  = torch.tensor(obs,  dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.tensor(mask, dtype=torch.bool,    device=self.device)

        logits, value = self.policy(obs_t)
        logits = logits.squeeze(0).masked_fill(~mask_t, float("-inf"))
        dist   = torch.distributions.Categorical(logits=logits)
        action = dist.sample()

        if self.is_trainable:
            return action.item(), dist.log_prob(action).item(), value.squeeze().item()
        else:
            # Past checkpoints: act but don't expose training signal
            return action.item(), None, None


# ────────────────────────────────────────────────────────────────────────────
# Naive agents
# ────────────────────────────────────────────────────────────────────────────

class RandomAgent(Agent):
    """
    Samples uniformly from valid actions. Floor baseline — any learned
    policy that cannot consistently beat this has a fundamental problem.
    """

    def __init__(self, rng: Optional[np.random.Generator] = None):
        self._rng = rng or np.random.default_rng()

    def act(self, obs: np.ndarray, mask: np.ndarray) -> ActResult:
        valid = self._valid_actions(mask)
        return int(self._rng.choice(valid)), None, None


class ConservativeAgent(Agent):
    """
    Never challenges. Plays as honestly as possible.

    DECLARE: prefer qty=1 honest if any matching card held (action 1),
             else qty=1 full bluff (action 0). Ignores partial bluffs.
    CHALLENGE: always passes (action 15).

    Teaches the learning agent to exploit passive opponents — a player
    facing this agent can bluff freely.
    """

    def act(self, obs: np.ndarray, mask: np.ndarray) -> ActResult:
        if self._is_challenge_phase(obs):
            return _ACTION_PASS, None, None

        # action 1 = (qty=1, honest=1); action 0 = (qty=1, honest=0)
        if mask[1]:   # can play 1 honest card
            return 1, None, None
        elif mask[0]: # forced to bluff
            return 0, None, None
        else:
            # Fallback: first valid declare action
            valid = self._valid_actions(mask)
            return int(valid[0]), None, None


class AggressiveAgent(Agent):
    """
    Always bluffs maximally and always challenges.

    DECLARE: pick highest available qty with honest=0 (pure bluff).
    CHALLENGE: always challenges (action 14).

    Teaches the learning agent to handle challenge pressure and to
    value honest cards for navigating forced-challenge sequences.
    """

    def act(self, obs: np.ndarray, mask: np.ndarray) -> ActResult:
        if self._is_challenge_phase(obs):
            return _ACTION_CHALLENGE, None, None

        # Scan from the end of declare actions: highest qty, honest=0
        # declareActions order: [(1,0),(1,1),(2,0),(2,1),(2,2),(3,0),...]
        # Indices of (qty, 0) actions: 0, 2, 5, 9
        full_bluff_indices = [i for i, (qty, honest) in enumerate(_DECLARE_ACTIONS)
                              if honest == 0]
        for idx in reversed(full_bluff_indices):
            if mask[idx]:
                return idx, None, None

        # Fallback: first valid action
        valid = self._valid_actions(mask)
        return int(valid[0]), None, None


class ThresholdAgent(Agent):
    """
    The most "reasonable" naive agent — closest to intuitive human play.

    DECLARE: play qty=1 honestly if possible; otherwise qty=1 bluff.
    CHALLENGE: challenge if normalised pile size > threshold (default 0.15,
               roughly 8 cards), else pass.

    Provides a mid-difficulty opponent. Beating this consistently requires
    the learning agent to learn context-sensitive bluffing rather than
    just exploiting pure passivity or pure aggression.
    """

    def __init__(self, challenge_threshold: float = 0.15):
        self._threshold = challenge_threshold

    def act(self, obs: np.ndarray, mask: np.ndarray) -> ActResult:
        if self._is_challenge_phase(obs):
            pile_norm = obs[_IDX_PILE]
            if pile_norm > self._threshold:
                return _ACTION_CHALLENGE, None, None
            else:
                return _ACTION_PASS, None, None

        # DECLARE: prefer (qty=1, honest=1), fall back to (qty=1, honest=0)
        if mask[1]:
            return 1, None, None
        elif mask[0]:
            return 0, None, None
        else:
            valid = self._valid_actions(mask)
            return int(valid[0]), None, None


# ────────────────────────────────────────────────────────────────────────────
# Convenience constructor
# ────────────────────────────────────────────────────────────────────────────

def make_naive_agents(rng: Optional[np.random.Generator] = None) -> list[Agent]:
    """Return one instance of each naive agent type."""
    r = rng or np.random.default_rng()
    return [
        RandomAgent(rng=r),
        ConservativeAgent(),
        AggressiveAgent(),
        ThresholdAgent(challenge_threshold=0.15),
    ]
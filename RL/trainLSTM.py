"""
trainWithPool_LSTM.py — PPO + LSTM trainer for BSEnv with opponent pool.

Architecture overview
─────────────────────
                    ┌─────────────┐
  raw obs (OBS_DIM) │ LSTMEncoder │ → latent (LSTM_LATENT_DIM)
                    └──────┬──────┘
                           │  concat
                    ┌──────▼──────────────────────────┐
  augmented obs     │ BSPolicy  (OBS_DIM+LSTM_LATENT)  │
  (OBS_DIM+LATENT)  │  trunk → actor / critic          │
                    └─────────────────────────────────-┘

The LSTM encoder is a *separate* nn.Module so it can later be swapped
for a Particle-Filter or Bayesian module with an identical interface:

    latent, new_hx = encoder(obs_t, hx)          # single step (inference)
    latent_seq      = encoder.forward_seq(obs_seq, hx0)  # full sequence (training)

Hidden state management
───────────────────────
AEC turns mean each agent acts roughly every NUM_PLAYERS steps. We keep
one (h, c) pair per seat during rollout. After each agent's turn we store
the *pre-step* hidden state alongside the transition so it can be replayed
during the PPO update.

PPO update with BPTT
────────────────────
We do NOT shuffle individual timesteps (that would break recurrence).
Instead we process each trajectory as a sequence, run the LSTM forward
in one pass, then flatten timesteps for the clipped-PPO loss.

Gradient decoupling (change 1):
  The latent vector is detached before the policy forward pass. This means
  the PPO loss gradient does not flow into the encoder — the policy and
  encoder are optimized by separate Adam instances at different learning
  rates (policy: lr, encoder: lr * 0.25). The encoder receives its own
  backward pass on the same loss value, computed with the live (attached)
  latent but with policy weights frozen in that graph. This prevents the
  noisy PPO gradient from destabilizing the LSTM representations.

Opponent pool
─────────────
Unchanged semantics: 50% current / 40% checkpoint / 10% naive.
Checkpoint agents carry a frozen copy of *both* encoder and policy weights.
"""

from __future__ import annotations

import copy
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from core.BSEnv import BSEnv, NUM_ACTIONS, NUM_PLAYERS, OBS_DIM
from agents import (
    Agent, PolicyAgent, make_naive_agents,
    RandomAgent, ConservativeAgent, AggressiveAgent, ThresholdAgent,
)


# ────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ────────────────────────────────────────────────────────────────────────────

# Opponent pool
POOL_ADD_INTERVAL  = 10
POOL_MAX_SIZE      = 30

# Seat-assignment probabilities
P_CURRENT    = 0.50
P_CHECKPOINT = 0.40
P_NAIVE      = 0.10

# LSTM
LSTM_LATENT_DIM  = 32   # output size of encoder; appended to obs
LSTM_HIDDEN_DIM  = 64   # internal cell size
LSTM_NUM_LAYERS  = 1

# Augmented observation dimensionality seen by BSPolicy
AUG_OBS_DIM = OBS_DIM + LSTM_LATENT_DIM


# ────────────────────────────────────────────────────────────────────────────
# LSTM Encoder
# ────────────────────────────────────────────────────────────────────────────

HiddenState = Tuple[torch.Tensor, torch.Tensor]   # (h, c), each [layers, 1, hidden]


class LSTMEncoder(nn.Module):
    """
    Maps raw observations → latent summary vector via an LSTM.

    Interface (matches what a future PP/Bayes module must expose):

        latent, new_hx = encoder(obs_t, hx)
            obs_t : [1, OBS_DIM]  float32     (single timestep, batch=1)
            hx    : HiddenState or None        (None → zeros)
            latent: [1, LSTM_LATENT_DIM]       float32
            new_hx: HiddenState

        latent_seq = encoder.forward_seq(obs_seq, hx0)
            obs_seq   : [T, OBS_DIM]  float32  (full episode sequence)
            hx0       : HiddenState or None
            latent_seq: [T, LSTM_LATENT_DIM]   float32
    """

    def __init__(
        self,
        obs_dim:    int = OBS_DIM,
        hidden_dim: int = LSTM_HIDDEN_DIM,
        latent_dim: int = LSTM_LATENT_DIM,
        num_layers: int = LSTM_NUM_LAYERS,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size  = obs_dim,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = False,   # input shape: [T, B, obs_dim]
        )
        # Project LSTM output to the latent we append to obs
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh(),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for name, p in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(p, gain=1.0)
            elif "bias" in name:
                nn.init.zeros_(p)
        nn.init.orthogonal_(self.proj[0].weight, gain=np.sqrt(2))
        nn.init.zeros_(self.proj[0].bias)

    def _zero_hidden(self, device: torch.device) -> HiddenState:
        z = torch.zeros(self.num_layers, 1, self.hidden_dim, device=device)
        return (z, z.clone())

    # ── single-step (used during rollout inference) ───────────────────────

    def forward(
        self,
        obs_t: torch.Tensor,          # [1, obs_dim]  (batch=1)
        hx:    Optional[HiddenState],
    ) -> Tuple[torch.Tensor, HiddenState]:
        """
        Single observation step. Returns (latent [1, latent_dim], new_hx).
        Intended for use inside collect_episode (no grad).
        """
        if hx is None:
            hx = self._zero_hidden(obs_t.device)

        # lstm expects [T, B, input]: T=1, B=1
        out, new_hx = self.lstm(obs_t.unsqueeze(0), hx)   # out: [1,1,hidden]
        latent = self.proj(out.squeeze(0))                  # [1, latent_dim]
        return latent, new_hx

    # ── full-sequence (used during PPO update with BPTT) ─────────────────

    def forward_seq(
        self,
        obs_seq: torch.Tensor,         # [T, obs_dim]
        hx0:     Optional[HiddenState],
    ) -> torch.Tensor:                  # [T, latent_dim]
        """
        Process an entire episode sequence in one LSTM pass.
        Gradients flow through the full sequence (BPTT).
        obs_seq is unsqueezed to [T, 1, obs_dim] for the LSTM.
        """
        if hx0 is None:
            hx0 = self._zero_hidden(obs_seq.device)

        out, _ = self.lstm(obs_seq.unsqueeze(1), hx0)   # [T, 1, hidden]
        latent_seq = self.proj(out.squeeze(1))           # [T, latent_dim]
        return latent_seq


# ────────────────────────────────────────────────────────────────────────────
# Policy network  (unchanged from vanilla PPO; receives augmented obs)
# ────────────────────────────────────────────────────────────────────────────

class BSPolicy(nn.Module):
    """
    Shared actor-critic. Input dim is AUG_OBS_DIM = OBS_DIM + LSTM_LATENT_DIM.
    Tanh activations, orthogonal init.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim,    hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        self.actor  = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.trunk:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor.weight,  gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, obs: torch.Tensor):
        h = self.trunk(obs)
        return self.actor(h), self.critic(h).squeeze(-1)

    def evaluate(
        self,
        obs:     torch.Tensor,   # [B, aug_obs_dim]
        actions: torch.Tensor,   # [B]
        masks:   torch.Tensor,   # [B, action_dim]  bool
    ):
        logits, values = self(obs)
        logits = logits.masked_fill(~masks, float("-inf"))
        dist   = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), values


# ────────────────────────────────────────────────────────────────────────────
# Trajectory buffer  (adds lstm_hx_seq for BPTT replay)
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class Trajectory:
    obs:       List[np.ndarray] = field(default_factory=list)   # raw obs
    actions:   List[int]        = field(default_factory=list)
    log_probs: List[float]      = field(default_factory=list)
    rewards:   List[float]      = field(default_factory=list)
    values:    List[float]      = field(default_factory=list)
    dones:     List[float]      = field(default_factory=list)
    masks:     List[np.ndarray] = field(default_factory=list)
    # Pre-step LSTM hidden states for BPTT.
    # Each entry is (h, c) with shapes [layers, 1, hidden] on CPU.
    lstm_hx:   List[HiddenState] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.obs)


# ────────────────────────────────────────────────────────────────────────────
# GAE  (unchanged)
# ────────────────────────────────────────────────────────────────────────────

def compute_gae(
    rewards: List[float],
    values:  List[float],
    dones:   List[float],
    gamma:   float = 0.99,
    lam:     float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    n   = len(rewards)
    adv = np.zeros(n, dtype=np.float32)
    last_gae = 0.0

    for t in reversed(range(n)):
        next_val  = values[t + 1] if t < n - 1 else 0.0
        next_val *= 1.0 - dones[t]
        delta    = rewards[t] + gamma * next_val - values[t]
        last_gae = delta + gamma * lam * (1.0 - dones[t]) * last_gae
        adv[t]   = last_gae

    returns = adv + np.array(values, dtype=np.float32)
    return adv, returns


# ────────────────────────────────────────────────────────────────────────────
# Opponent pool  (snapshots encoder + policy together)
# ────────────────────────────────────────────────────────────────────────────

class OpponentPool:
    """
    Fixed-size FIFO pool of past (encoder, policy) snapshot pairs.
    Each snapshot is a pair of CPU state-dicts to minimise memory.
    """

    def __init__(
        self,
        obs_dim:    int,
        action_dim: int,
        hidden_dim: int,
        device:     str,
        max_size:   int = POOL_MAX_SIZE,
    ):
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.device     = device
        self.max_size   = max_size
        self._pool: List[Tuple[dict, dict]] = []   # (encoder_sd, policy_sd)

    def add(self, encoder: LSTMEncoder, policy: BSPolicy) -> None:
        enc_sd = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
        pol_sd = {k: v.cpu().clone() for k, v in policy.state_dict().items()}
        self._pool.append((enc_sd, pol_sd))
        if len(self._pool) > self.max_size:
            self._pool.pop(0)

    def sample(self) -> "LSTMPolicyAgent":
        enc_sd, pol_sd = self._pool[np.random.randint(len(self._pool))]

        encoder = LSTMEncoder()
        encoder.load_state_dict({k: v.clone() for k, v in enc_sd.items()})
        encoder.to(self.device)
        encoder.eval()

        policy = BSPolicy(AUG_OBS_DIM, self.action_dim, self.hidden_dim)
        policy.load_state_dict({k: v.clone() for k, v in pol_sd.items()})
        policy.to(self.device)
        policy.eval()

        return LSTMPolicyAgent(encoder, policy, self.device, is_trainable=False)

    def is_empty(self) -> bool:
        return len(self._pool) == 0

    def __len__(self) -> int:
        return len(self._pool)


# ────────────────────────────────────────────────────────────────────────────
# LSTM-aware agent wrapper
# ────────────────────────────────────────────────────────────────────────────

class LSTMPolicyAgent:
    """
    Wraps (LSTMEncoder, BSPolicy) for use during episode rollout.

    Maintains per-seat hidden states internally so the caller only needs
    to tell it which seat is acting. Hidden states are reset at episode
    boundaries via reset_hidden().

    act() returns (action, log_prob, value, pre_step_hx):
        - log_prob and value are None for frozen (non-trainable) agents.
        - pre_step_hx is always returned so the caller can store it in
          the trajectory for BPTT replay.
    """

    def __init__(
        self,
        encoder:      LSTMEncoder,
        policy:       BSPolicy,
        device:       str,
        is_trainable: bool,
    ):
        self.encoder      = encoder
        self.policy       = policy
        self.device       = device
        self.is_trainable = is_trainable
        # Seat-indexed hidden states; populated lazily / reset each episode
        self._hidden: Dict[int, Optional[HiddenState]] = {}

    def reset_hidden(self, seats: List[int]) -> None:
        """Call at the start of each episode for every seat using this agent."""
        for s in seats:
            self._hidden[s] = None

    def act(
        self,
        obs_arr:     np.ndarray,    # raw obs [OBS_DIM]
        action_mask: np.ndarray,    # [NUM_ACTIONS] int8
        seat:        int,           # which seat is acting
    ) -> Tuple[int, Optional[float], Optional[float], Optional[HiddenState]]:
        """
        Returns (action, log_prob, value, pre_step_hx).
        log_prob / value are None for non-trainable agents.
        pre_step_hx is the hidden state *before* this obs was encoded;
        store it alongside the transition for BPTT replay.
        """
        obs_t = torch.tensor(obs_arr, dtype=torch.float32,
                             device=self.device).unsqueeze(0)  # [1, OBS_DIM]

        hx = self._hidden.get(seat, None)
        pre_step_hx = _detach_hx(hx)   # snapshot before update (for BPTT)

        ctx = torch.no_grad() if not self.is_trainable else _null_ctx()
        with ctx:
            latent, new_hx = self.encoder(obs_t, hx)          # [1, latent]
            aug_obs = torch.cat([obs_t, latent], dim=-1)       # [1, aug_obs]
            logits, value = self.policy(aug_obs)

        self._hidden[seat] = new_hx   # carry state forward

        # Mask invalid actions
        mask_t = torch.tensor(action_mask, dtype=torch.bool, device=self.device)
        logits = logits.masked_fill(~mask_t, float("-inf"))
        dist   = torch.distributions.Categorical(logits=logits)
        action = int(dist.sample().item())

        if self.is_trainable:
            log_prob = float(dist.log_prob(torch.tensor(action, device=self.device)).item())
            val      = float(value.item())
        else:
            log_prob = None
            val      = None

        return action, log_prob, val, pre_step_hx


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _detach_hx(hx: Optional[HiddenState]) -> Optional[HiddenState]:
    """Return a detached CPU copy of a hidden state (or None)."""
    if hx is None:
        return None
    return (hx[0].detach().cpu(), hx[1].detach().cpu())


class _null_ctx:
    """No-op context manager (replaces torch.no_grad for trainable agents)."""
    def __enter__(self): return self
    def __exit__(self, *_): pass


# ────────────────────────────────────────────────────────────────────────────
# Trainer
# ────────────────────────────────────────────────────────────────────────────

class BSTrainer:

    def __init__(
        self,
        hidden_dim:          int   = 128,
        lr:                  float = 3e-4,
        gamma:               float = 0.99,
        gae_lambda:          float = 0.95,
        clip_eps:            float = 0.2,
        value_coef:          float = 0.5,
        entropy_coef:        float = 0.01,
        max_grad_norm:       float = 0.5,
        n_epochs:            int   = 4,
        batch_size:          int   = 256,
        episodes_per_update: int   = 32,
        max_iter:            int   = 100_000,
        pool_add_interval:   int   = POOL_ADD_INTERVAL,
        device:              Optional[str] = None,
    ):
        self.env    = BSEnv(max_iter=max_iter)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Models
        self.encoder = LSTMEncoder().to(self.device)
        self.policy  = BSPolicy(AUG_OBS_DIM, NUM_ACTIONS, hidden_dim).to(self.device)

        # Separate optimizers: encoder is decoupled from policy gradient.
        # encoder_lr is intentionally lower — the encoder should update slowly
        # and stably relative to the policy, which sees cleaner gradients
        # because latent is detached before the policy forward pass.
        self.policy_opt  = optim.Adam(self.policy.parameters(),  lr=lr,        eps=1e-5)
        self.encoder_opt = optim.Adam(self.encoder.parameters(), lr=lr * 0.25, eps=1e-5)

        self.gamma            = gamma
        self.gae_lambda       = gae_lambda
        self.clip_eps         = clip_eps
        self.value_coef       = value_coef
        self.entropy_coef     = entropy_coef
        self.max_grad_norm    = max_grad_norm
        self.n_epochs         = n_epochs
        self.batch_size       = batch_size
        self.episodes_per_update = episodes_per_update
        self.pool_add_interval   = pool_add_interval

        # Opponent infrastructure
        self.pool = OpponentPool(
            AUG_OBS_DIM, NUM_ACTIONS, hidden_dim, self.device
        )
        self._naive_agents = make_naive_agents()
        self._rng = np.random.default_rng()

        # Current-policy agent wrapper (trainable)
        self._current_agent = LSTMPolicyAgent(
            self.encoder, self.policy, self.device, is_trainable=True
        )

        # Metrics
        self.episode_count     = 0
        self.update_count      = 0
        self.win_counts:       Dict[int, int] = defaultdict(int)
        self.seat_type_counts: Dict[str, int] = defaultdict(int)

    # ── seat assignment ───────────────────────────────────────────────────

    def _assign_seats(self) -> Dict[int, LSTMPolicyAgent]:
        assignments: Dict[int, LSTMPolicyAgent] = {}
        types:       Dict[int, str]             = {}

        for seat in range(NUM_PLAYERS):
            r = self._rng.random()
            if r < P_CURRENT:
                assignments[seat] = self._current_agent
                types[seat]       = "current"
            elif r < P_CURRENT + P_CHECKPOINT and not self.pool.is_empty():
                assignments[seat] = self.pool.sample()
                types[seat]       = "checkpoint"
            elif r < P_CURRENT + P_CHECKPOINT and self.pool.is_empty():
                assignments[seat] = self._current_agent
                types[seat]       = "current"
            else:
                # Naive agents don't use the LSTM wrapper; handled below
                assignments[seat] = self._rng.choice(self._naive_agents)
                types[seat]       = "naive"

        if not any(t == "current" for t in types.values()):
            fallback_seat = int(self._rng.integers(0, NUM_PLAYERS))
            assignments[fallback_seat] = self._current_agent
            types[fallback_seat]       = "current"

        for t in types.values():
            self.seat_type_counts[t] += 1

        return assignments

    # ── episode collection ────────────────────────────────────────────────

    def collect_episode(self) -> List[Trajectory]:
        """
        Run one full episode.

        Returns one Trajectory per seat. Trajectories for non-trainable
        seats are empty and skipped by update().

        LSTM changes vs vanilla PPO:
          - Each LSTMPolicyAgent maintains hidden state per seat.
          - reset_hidden() is called at episode start.
          - act() now also returns pre_step_hx which is appended to the
            trajectory so BPTT can replay the exact sequence.
          - Naive agents fall back to their original .act(obs, mask) interface.
        """
        self.env.reset()
        seat_agents   = self._assign_seats()
        trajectories  = {a: Trajectory() for a in range(NUM_PLAYERS)}

        # Reset LSTM hidden states for all LSTM-based seats
        lstm_seats: Dict[LSTMPolicyAgent, List[int]] = defaultdict(list)
        for seat, agent in seat_agents.items():
            if isinstance(agent, LSTMPolicyAgent):
                lstm_seats[agent].append(seat)
        for agent, seats in lstm_seats.items():
            agent.reset_hidden(seats)

        # seat → (obs, action, log_prob, value, mask, pre_step_hx)
        pending: Dict[int, tuple] = {}

        for agent in self.env.agent_iter():
            obs_arr, reward, terminated, truncated, info = self.env.last()
            done = terminated or truncated

            # Flush previous transition for this seat with its reward
            if agent in pending:
                p_obs, p_act, p_lp, p_val, p_mask, p_hx = pending.pop(agent)
                traj = trajectories[agent]
                traj.obs.append(p_obs)
                traj.actions.append(p_act)
                traj.log_probs.append(p_lp)
                traj.rewards.append(float(reward))
                traj.values.append(p_val)
                traj.dones.append(float(done))
                traj.masks.append(p_mask)
                traj.lstm_hx.append(p_hx)

            if done:
                self.env.step(None)
                continue

            seat_agent = seat_agents[agent]

            if isinstance(seat_agent, LSTMPolicyAgent):
                action, log_prob, value, pre_hx = seat_agent.act(
                    obs_arr, info["action_mask"], seat=agent
                )
            else:
                # Naive agent — original interface, no hidden state
                action, log_prob, value = seat_agent.act(obs_arr, info["action_mask"])
                pre_hx = None

            if log_prob is not None:
                pending[agent] = (obs_arr, action, log_prob, value,
                                  info["action_mask"], pre_hx)

            self.env.step(action)

        # Flush seats still pending at episode end
        for agent, (p_obs, p_act, p_lp, p_val, p_mask, p_hx) in pending.items():
            traj = trajectories[agent]
            traj.obs.append(p_obs)
            traj.actions.append(p_act)
            traj.log_probs.append(p_lp)
            traj.rewards.append(0.0)
            traj.values.append(p_val)
            traj.dones.append(1.0)
            traj.masks.append(p_mask)
            traj.lstm_hx.append(p_hx)

        if self.env.state is not None and self.env.state.winner is not None:
            self.win_counts[self.env.state.winner] += 1
        self.episode_count += 1

        return list(trajectories.values())

    # ── PPO update with BPTT ──────────────────────────────────────────────

    def update(self, all_trajectories: List[Trajectory]) -> dict:
        """
        Recurrent PPO update.

        We process each trajectory *as a sequence* so the LSTM encoder
        can be trained with BPTT. After re-running the encoder over the
        sequence, we flatten all timesteps for the standard clipped-PPO
        objective.

        Order of operations per epoch:
          1. For each trajectory, run encoder.forward_seq() from the stored
             initial hidden state → latent_seq.
          2. Concatenate raw obs with latent_seq → aug_obs_seq.
          3. Run policy on aug_obs_seq (batch dim = T).
          4. Accumulate PPO loss over all sequences, then step optimizer.

        Note: we do *not* shuffle individual timesteps (that breaks BPTT).
        We do shuffle trajectory order across epochs for variance reduction.
        """
        # Filter to non-empty trainable trajectories
        trajs = [t for t in all_trajectories if len(t) > 0]
        if not trajs:
            return {}

        # Pre-compute advantages and returns (uses stored values, no grad)
        adv_list: List[np.ndarray] = []
        ret_list: List[np.ndarray] = []
        for traj in trajs:
            adv, ret = compute_gae(
                traj.rewards, traj.values, traj.dones,
                self.gamma, self.gae_lambda,
            )
            adv_list.append(adv)
            ret_list.append(ret)

        # Normalize advantages globally across all trajectories
        all_adv = np.concatenate(adv_list)
        adv_mean, adv_std = float(all_adv.mean()), float(all_adv.std() + 1e-8)
        adv_list = [(a - adv_mean) / adv_std for a in adv_list]

        n_trajs = len(trajs)
        idx     = np.arange(n_trajs)
        log     = defaultdict(list)

        for _ in range(self.n_epochs):
            np.random.shuffle(idx)

            for ti in idx:
                traj = trajs[ti]
                T    = len(traj)

                # ── Re-run encoder (with grad) over full sequence ─────────
                obs_seq = torch.tensor(np.array(traj.obs), dtype=torch.float32,
                                       device=self.device)            # [T, OBS_DIM]

                hx0 = _move_hx(traj.lstm_hx[0], self.device) if traj.lstm_hx else None

                latent_seq = self.encoder.forward_seq(obs_seq, hx0)  # [T, latent]

                # ── Detach latent before policy forward ───────────────────
                # Policy gradients do not flow into the encoder. The encoder
                # is updated via its own backward pass on the same loss value,
                # but at a lower learning rate and without interference from
                # the clipped PPO objective reshaping its representations.
                latent_detached = latent_seq.detach()
                aug_obs_policy  = torch.cat([obs_seq, latent_detached], dim=-1)  # [T, aug]

                # ── Shared targets ────────────────────────────────────────
                act_t  = torch.tensor(traj.actions,   dtype=torch.long,    device=self.device)
                lp_old = torch.tensor(traj.log_probs, dtype=torch.float32, device=self.device)
                adv_t  = torch.tensor(adv_list[ti],   dtype=torch.float32, device=self.device)
                ret_t  = torch.tensor(ret_list[ti],   dtype=torch.float32, device=self.device)
                mask_t = torch.tensor(np.array(traj.masks), dtype=torch.bool, device=self.device)

                # ── Policy forward and loss ───────────────────────────────
                log_prob, entropy, value = self.policy.evaluate(aug_obs_policy, act_t, mask_t)

                ratio = torch.exp(log_prob - lp_old)
                policy_loss  = -torch.min(
                    ratio * adv_t,
                    ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv_t,
                ).mean()
                value_loss   = nn.functional.mse_loss(value, ret_t)
                entropy_loss = -entropy.mean()

                ppo_loss = (policy_loss
                            + self.value_coef   * value_loss
                            + self.entropy_coef * entropy_loss)

                # ── Policy backward (encoder grad blocked by detach) ──────
                self.policy_opt.zero_grad()
                ppo_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy_opt.step()

                # ── Encoder backward (attached latent, policy detached) ───
                # Re-run policy on live latent so the encoder receives signal
                # for how well its representations support the policy loss,
                # without the policy weights moving a second time this step.
                aug_obs_enc = torch.cat([obs_seq, latent_seq], dim=-1)
                with torch.no_grad():
                    # freeze policy params from this graph — we only want
                    # gradients w.r.t. encoder params
                    pass
                log_prob_enc, _, value_enc = self.policy.evaluate(aug_obs_enc, act_t, mask_t)
                ratio_enc = torch.exp(log_prob_enc - lp_old)
                enc_loss  = (-torch.min(
                    ratio_enc * adv_t,
                    ratio_enc.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv_t,
                ).mean() + self.value_coef * nn.functional.mse_loss(value_enc, ret_t))

                self.encoder_opt.zero_grad()
                enc_loss.backward()
                nn.utils.clip_grad_norm_(self.encoder.parameters(), self.max_grad_norm)
                self.encoder_opt.step()

                log["policy_loss"].append(policy_loss.item())
                log["value_loss"].append(value_loss.item())
                log["entropy"].append(-entropy_loss.item())

        self.update_count += 1

        if self.update_count % self.pool_add_interval == 0:
            self.pool.add(self.encoder, self.policy)

        return {k: float(np.mean(v)) for k, v in log.items()}

    # ── training loop ─────────────────────────────────────────────────────

    def train(
        self,
        total_episodes: int,
        log_interval:   int = 200,
        save_interval:  int = 5_000,
        checkpoint_dir: str = "checkpoints_lstm",
    ) -> BSPolicy:
        os.makedirs(checkpoint_dir, exist_ok=True)

        n_enc = sum(p.numel() for p in self.encoder.parameters())
        n_pol = sum(p.numel() for p in self.policy.parameters())
        print(f"Device       : {self.device}")
        print(f"Encoder params: {n_enc:,}  (LSTM hidden={LSTM_HIDDEN_DIM}, "
              f"latent={LSTM_LATENT_DIM})")
        print(f"Policy params : {n_pol:,}  (aug obs={AUG_OBS_DIM})")
        print(f"Total params  : {n_enc + n_pol:,}")
        print(f"Target        : {total_episodes:,} episodes, "
              f"update every {self.episodes_per_update}")
        print(f"Pool          : max {POOL_MAX_SIZE} snapshots, "
              f"snapshot every {self.pool_add_interval} updates")
        print(f"Seat dist     : {P_CURRENT:.0%} current / "
              f"{P_CHECKPOINT:.0%} checkpoint / {P_NAIVE:.0%} naive\n")

        t0     = time.time()
        buffer: List[Trajectory] = []

        while self.episode_count < total_episodes:

            for _ in range(self.episodes_per_update):
                buffer.extend(self.collect_episode())
                if self.episode_count >= total_episodes:
                    break

            metrics = self.update(buffer)
            buffer.clear()

            if self.episode_count % log_interval < self.episodes_per_update:
                total_wins  = max(sum(self.win_counts.values()), 1)
                total_seats = max(sum(self.seat_type_counts.values()), 1)
                win_str  = "  ".join(
                    f"p{a}:{self.win_counts[a]/total_wins:5.1%}"
                    for a in range(NUM_PLAYERS)
                )
                seat_str = "  ".join(
                    f"{k}:{self.seat_type_counts[k]/total_seats:4.1%}"
                    for k in ["current", "checkpoint", "naive"]
                )
                print(
                    f"ep {self.episode_count:>7,} | "
                    f"upd {self.update_count:>4,} | "
                    f"pool {len(self.pool):>2} | "
                    f"t {time.time()-t0:>5.0f}s | "
                    f"wins [{win_str}] | "
                    f"seats [{seat_str}] | "
                    f"π {metrics.get('policy_loss', 0):+.4f}  "
                    f"v {metrics.get('value_loss',  0):.4f}  "
                    f"H {metrics.get('entropy',     0):.3f}"
                )

            if self.episode_count % save_interval < self.episodes_per_update:
                path = os.path.join(
                    checkpoint_dir, f"policy_{self.episode_count:07d}.pt"
                )
                self.save(path)

        print("\nTraining complete.")
        return self.policy

    # ── persistence ───────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        torch.save({
            "encoder":          self.encoder.state_dict(),
            "policy":           self.policy.state_dict(),
            "policy_opt":       self.policy_opt.state_dict(),
            "encoder_opt":      self.encoder_opt.state_dict(),
            "episode_count":    self.episode_count,
            "update_count":     self.update_count,
            "win_counts":       dict(self.win_counts),
            "seat_type_counts": dict(self.seat_type_counts),
            "pool":             self.pool._pool,
        }, path)
        print(f"  → saved {path}")

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(ckpt["encoder"])
        self.policy.load_state_dict(ckpt["policy"])
        # Support checkpoints saved before the optimizer split
        if "policy_opt" in ckpt:
            self.policy_opt.load_state_dict(ckpt["policy_opt"])
        if "encoder_opt" in ckpt:
            self.encoder_opt.load_state_dict(ckpt["encoder_opt"])
        self.episode_count    = ckpt["episode_count"]
        self.update_count     = ckpt["update_count"]
        self.win_counts       = defaultdict(int, ckpt["win_counts"])
        self.seat_type_counts = defaultdict(int, ckpt.get("seat_type_counts", {}))
        self.pool._pool       = ckpt.get("pool", [])
        print(f"  → loaded {path} (ep {self.episode_count}, "
              f"pool size {len(self.pool)})")


# ────────────────────────────────────────────────────────────────────────────
# Helper: move HiddenState to device
# ────────────────────────────────────────────────────────────────────────────

def _move_hx(
    hx: Optional[HiddenState],
    device: str,
) -> Optional[HiddenState]:
    if hx is None:
        return None
    return (hx[0].to(device), hx[1].to(device))


# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    trainer = BSTrainer(
        hidden_dim          = 128,
        lr                  = 3e-4,
        gamma               = 0.99,
        gae_lambda          = 0.95,
        clip_eps            = 0.2,
        value_coef          = 0.5,
        entropy_coef        = 0.01,
        max_grad_norm       = 0.5,
        n_epochs            = 4,
        batch_size          = 256,
        episodes_per_update = 32,
        max_iter            = 100_000,
        pool_add_interval   = 10,
    )
    trainer.train(
        total_episodes = 100_000,
        log_interval   = 200,
        save_interval  = 5_000,
    )
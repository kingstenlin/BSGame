"""
trainWithPool.py — Self-play PPO trainer for BSEnv with opponent pool.

Opponent pool
─────────────
Each episode assigns an agent to each of the 3 seats. The distribution
across all seat-assignments targets ~50/40/10:

    50%  current policy      (trainable — transitions buffered)
    40%  past checkpoint     (frozen copy of an earlier policy snapshot)
    10%  naive agent         (rule-based: Random, Conservative, Aggressive,
                              or Threshold — sampled uniformly)

Implementation:
  - Each seat is sampled independently from the distribution above.
  - If no seat received the current policy (probability 0.5³ = 12.5%),
    one seat is re-assigned to current policy to guarantee training data
    every episode.
  - The checkpoint pool is initially empty; fallback to current policy
    until the first snapshot is taken (every POOL_ADD_INTERVAL updates).

Only transitions from current-policy seats are added to the training
buffer. Past-checkpoint and naive-agent seats act normally (keeping the
game valid) but their transitions are discarded.

AEC reward offset
─────────────────
env.last() returns cumulative reward since the agent LAST acted.
Transitions are buffered and flushed with their reward on the agent's
following turn (or when processed as dead). See collect_episode for detail.
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
from RL.agents import (
    Agent, PolicyAgent, make_naive_agents,
    RandomAgent, ConservativeAgent, AggressiveAgent, ThresholdAgent,
)


# ────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ────────────────────────────────────────────────────────────────────────────

# Opponent pool
POOL_ADD_INTERVAL  = 10    # add snapshot to pool every N updates
POOL_MAX_SIZE      = 30    # evict oldest when pool exceeds this

# Seat-assignment probabilities (must sum to 1.0)
P_CURRENT    = 0.50
P_CHECKPOINT = 0.40
P_NAIVE      = 0.10


# ────────────────────────────────────────────────────────────────────────────
# Policy network
# ────────────────────────────────────────────────────────────────────────────

class BSPolicy(nn.Module):
    """
    Shared actor-critic network. Tanh activations, orthogonal init.
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
        obs:     torch.Tensor,   # [B, obs_dim]
        actions: torch.Tensor,   # [B]
        masks:   torch.Tensor,   # [B, action_dim]  bool
    ):
        logits, values = self(obs)
        logits = logits.masked_fill(~masks, float("-inf"))
        dist   = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), values


# ────────────────────────────────────────────────────────────────────────────
# Trajectory buffer
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class Trajectory:
    obs:       List[np.ndarray] = field(default_factory=list)
    actions:   List[int]        = field(default_factory=list)
    log_probs: List[float]      = field(default_factory=list)
    rewards:   List[float]      = field(default_factory=list)
    values:    List[float]      = field(default_factory=list)
    dones:     List[float]      = field(default_factory=list)
    masks:     List[np.ndarray] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.obs)


# ────────────────────────────────────────────────────────────────────────────
# GAE
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
# Opponent pool
# ────────────────────────────────────────────────────────────────────────────

class OpponentPool:
    """
    Manages a fixed-size FIFO pool of past policy snapshots.

    Checkpoints are stored as CPU state-dicts (not full BSPolicy objects)
    to minimise memory. A new BSPolicy is instantiated on each sample()
    call — the cost is a dict copy and a load_state_dict, which is
    negligible compared to episode rollout time.
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
        self._pool: List[dict] = []   # CPU state-dicts, oldest-first

    def add(self, policy: BSPolicy) -> None:
        """Snapshot current policy onto CPU and append to pool."""
        state = {k: v.cpu().clone() for k, v in policy.state_dict().items()}
        self._pool.append(state)
        if len(self._pool) > self.max_size:
            self._pool.pop(0)   # evict oldest

    def sample(self) -> PolicyAgent:
        """
        Sample a uniformly random past snapshot.
        Returns a frozen PolicyAgent (is_trainable=False).
        Raises if pool is empty — check is_empty() first.
        """
        state = self._pool[np.random.randint(len(self._pool))]
        policy = BSPolicy(self.obs_dim, self.action_dim, self.hidden_dim)
        policy.load_state_dict({k: v.clone() for k, v in state.items()})
        policy.to(self.device)
        policy.eval()
        return PolicyAgent(policy, self.device, is_trainable=False)

    def is_empty(self) -> bool:
        return len(self._pool) == 0

    def __len__(self) -> int:
        return len(self._pool)


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
        self.policy = BSPolicy(OBS_DIM, NUM_ACTIONS, hidden_dim).to(self.device)
        self.opt    = optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)

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
            OBS_DIM, NUM_ACTIONS, hidden_dim, self.device
        )
        self._naive_agents = make_naive_agents()
        self._rng = np.random.default_rng()

        # The current-policy agent wrapper (is_trainable=True)
        self._current_agent = PolicyAgent(self.policy, self.device, is_trainable=True)

        # Metrics
        self.episode_count = 0
        self.update_count  = 0
        self.win_counts:      Dict[int, int] = defaultdict(int)
        self.seat_type_counts: Dict[str, int] = defaultdict(int)

    # ── seat assignment ───────────────────────────────────────────────────

    def _assign_seats(self) -> Dict[int, Agent]:
        """
        Sample an agent for each of the 3 seats.

        Distribution per seat (independent):
            P_CURRENT    = 0.50  → current policy  (trainable)
            P_CHECKPOINT = 0.40  → random past snapshot (frozen)
            P_NAIVE      = 0.10  → random naive agent

        If the pool is empty, checkpoint slots fall back to current policy.
        If no seat received the current policy after sampling, one seat is
        re-assigned to current policy to guarantee training data.
        """
        assignments: Dict[int, Agent] = {}
        types:       Dict[int, str]   = {}

        for seat in range(NUM_PLAYERS):
            r = self._rng.random()
            if r < P_CURRENT:
                assignments[seat] = self._current_agent
                types[seat]       = "current"
            elif r < P_CURRENT + P_CHECKPOINT and not self.pool.is_empty():
                assignments[seat] = self.pool.sample()
                types[seat]       = "checkpoint"
            elif r < P_CURRENT + P_CHECKPOINT and self.pool.is_empty():
                # Pool not yet populated — fall back to current policy
                assignments[seat] = self._current_agent
                types[seat]       = "current"
            else:
                assignments[seat] = self._rng.choice(self._naive_agents)
                types[seat]       = "naive"

        # Guarantee at least one trainable seat per episode
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
        Run one full episode with the seat assignment drawn from the pool.

        Returns one Trajectory per seat. Trajectories for non-trainable
        seats will be empty (len 0) and are skipped by update().

        Pending-dict pattern (AEC reward offset):
            When seat S acts we buffer (obs, action, log_prob, value, mask).
            The reward for that action arrives via env.last() on S's NEXT
            turn. We flush the pending entry with that reward at that point.
        """
        self.env.reset()
        seat_agents   = self._assign_seats()
        trajectories  = {a: Trajectory() for a in range(NUM_PLAYERS)}

        # seat → (obs, action, log_prob, value, mask)
        pending: Dict[int, tuple] = {}

        for agent in self.env.agent_iter():
            obs_arr, reward, terminated, truncated, info = self.env.last()
            done = terminated or truncated

            # Flush the previous transition for this seat now that its
            # reward has arrived.
            if agent in pending:
                p_obs, p_act, p_lp, p_val, p_mask = pending.pop(agent)
                traj = trajectories[agent]
                traj.obs.append(p_obs)
                traj.actions.append(p_act)
                traj.log_probs.append(p_lp)
                traj.rewards.append(float(reward))
                traj.values.append(p_val)
                traj.dones.append(float(done))
                traj.masks.append(p_mask)

            if done:
                self.env.step(None)
                continue

            # Select action for this seat
            seat_agent = seat_agents[agent]
            action, log_prob, value = seat_agent.act(obs_arr, info["action_mask"])

            # Only buffer if this seat is the trainable current policy
            if log_prob is not None:
                pending[agent] = (obs_arr, action, log_prob, value, info["action_mask"])

            self.env.step(action)

        # Flush any seats still pending (acted on the very last step)
        for agent, (p_obs, p_act, p_lp, p_val, p_mask) in pending.items():
            traj = trajectories[agent]
            traj.obs.append(p_obs)
            traj.actions.append(p_act)
            traj.log_probs.append(p_lp)
            traj.rewards.append(0.0)
            traj.values.append(p_val)
            traj.dones.append(1.0)
            traj.masks.append(p_mask)

        if self.env.state is not None and self.env.state.winner is not None:
            self.win_counts[self.env.state.winner] += 1
        self.episode_count += 1

        return list(trajectories.values())

    # ── PPO update ────────────────────────────────────────────────────────

    def update(self, all_trajectories: List[Trajectory]) -> dict:
        """
        PPO update on all trainable transitions collected since last update.
        Non-trainable trajectories (empty) are silently skipped.
        """
        obs_l, act_l, lp_l, adv_l, ret_l, mask_l = [], [], [], [], [], []

        for traj in all_trajectories:
            if len(traj) == 0:
                continue
            adv, ret = compute_gae(
                traj.rewards, traj.values, traj.dones,
                self.gamma, self.gae_lambda,
            )
            obs_l.extend(traj.obs)
            act_l.extend(traj.actions)
            lp_l.extend(traj.log_probs)
            adv_l.extend(adv.tolist())
            ret_l.extend(ret.tolist())
            mask_l.extend(traj.masks)

        if not obs_l:
            return {}

        obs_t  = torch.tensor(np.array(obs_l),  dtype=torch.float32, device=self.device)
        act_t  = torch.tensor(act_l,             dtype=torch.long,    device=self.device)
        lp_t   = torch.tensor(lp_l,              dtype=torch.float32, device=self.device)
        adv_t  = torch.tensor(adv_l,             dtype=torch.float32, device=self.device)
        ret_t  = torch.tensor(ret_l,             dtype=torch.float32, device=self.device)
        mask_t = torch.tensor(np.array(mask_l),  dtype=torch.bool,    device=self.device)

        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        n   = len(obs_l)
        idx = np.arange(n)
        log = defaultdict(list)

        for _ in range(self.n_epochs):
            np.random.shuffle(idx)
            for start in range(0, n, self.batch_size):
                b = idx[start : start + self.batch_size]

                log_prob, entropy, value = self.policy.evaluate(
                    obs_t[b], act_t[b], mask_t[b]
                )

                ratio = torch.exp(log_prob - lp_t[b])
                adv_b = adv_t[b]

                policy_loss = -torch.min(
                    ratio * adv_b,
                    ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv_b,
                ).mean()

                value_loss   = nn.functional.mse_loss(value, ret_t[b])
                entropy_loss = -entropy.mean()

                loss = (policy_loss
                        + self.value_coef   * value_loss
                        + self.entropy_coef * entropy_loss)

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.opt.step()

                log["policy_loss"].append(policy_loss.item())
                log["value_loss"].append(value_loss.item())
                log["entropy"].append(-entropy_loss.item())

        self.update_count += 1

        # Add snapshot to pool on schedule
        if self.update_count % self.pool_add_interval == 0:
            self.pool.add(self.policy)

        return {k: float(np.mean(v)) for k, v in log.items()}

    # ── training loop ─────────────────────────────────────────────────────

    def train(
        self,
        total_episodes: int,
        log_interval:   int = 200,
        save_interval:  int = 5_000,
        checkpoint_dir: str = "checkpoints",
        resume_from:    Optional[str] = None,
    ) -> BSPolicy:
        os.makedirs(checkpoint_dir, exist_ok=True)

        if resume_from is not None:
            self.load(resume_from)
            print(f"  Resuming from ep {self.episode_count:,} → target {total_episodes:,}\n")
            if self.episode_count >= total_episodes:
                print("  Nothing to do — checkpoint already at or past target.")
                return self.policy

        n_params = sum(p.numel() for p in self.policy.parameters())
        print(f"Device     : {self.device}")
        print(f"Parameters : {n_params:,}")
        print(f"Progress   : ep {self.episode_count:,} → {total_episodes:,}")
        print(f"Update every: {self.episodes_per_update} episodes")
        print(f"Pool       : max {POOL_MAX_SIZE} snapshots, "
              f"snapshot every {self.pool_add_interval} updates")
        print(f"Seat dist  : {P_CURRENT:.0%} current / "
              f"{P_CHECKPOINT:.0%} checkpoint / {P_NAIVE:.0%} naive\n")

        t0 = time.time()
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
            "policy":           self.policy.state_dict(),
            "optimizer":        self.opt.state_dict(),
            "episode_count":    self.episode_count,
            "update_count":     self.update_count,
            "win_counts":       dict(self.win_counts),
            "seat_type_counts": dict(self.seat_type_counts),
            "pool":             self.pool._pool,   # list of CPU state-dicts
        }, path)
        print(f"  → saved {path}")

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])
        self.opt.load_state_dict(ckpt["optimizer"])
        self.episode_count    = ckpt["episode_count"]
        self.update_count     = ckpt["update_count"]
        self.win_counts       = defaultdict(int, ckpt["win_counts"])
        self.seat_type_counts = defaultdict(int, ckpt.get("seat_type_counts", {}))
        self.pool._pool       = ckpt.get("pool", [])
        print(f"  → loaded {path} (ep {self.episode_count}, "
              f"pool size {len(self.pool)})")


# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse as _ap
    _p = _ap.ArgumentParser(description="Train vanilla PPO agent for BSEnv")
    _p.add_argument("--total-episodes",      type=int,   default=100_000)
    _p.add_argument("--resume",              type=str,   default=None,
                    help="Path to checkpoint to resume from")
    _p.add_argument("--checkpoint-dir",      type=str,   default="checkpoints")
    _p.add_argument("--save-interval",       type=int,   default=5_000)
    _p.add_argument("--log-interval",        type=int,   default=200)
    _p.add_argument("--lr",                  type=float, default=3e-4)
    _p.add_argument("--episodes-per-update", type=int,   default=32)
    _a = _p.parse_args()

    trainer = BSTrainer(
        hidden_dim          = 128,
        lr                  = _a.lr,
        gamma               = 0.99,
        gae_lambda          = 0.95,
        clip_eps            = 0.2,
        value_coef          = 0.5,
        entropy_coef        = 0.01,
        max_grad_norm       = 0.5,
        n_epochs            = 4,
        batch_size          = 256,
        episodes_per_update = _a.episodes_per_update,
        max_iter            = 100_000,
        pool_add_interval   = 10,
    )
    trainer.train(
        total_episodes = _a.total_episodes,
        log_interval   = _a.log_interval,
        save_interval  = _a.save_interval,
        checkpoint_dir = _a.checkpoint_dir,
        resume_from    = _a.resume,
    )
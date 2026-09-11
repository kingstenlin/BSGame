"""
train.py — Self-play PPO trainer for BSEnv (PettingZoo AEC).

Architecture
────────────
All three agents share one policy (actor-critic). Each episode, every
agent's transitions are collected and combined into a single PPO update.
This is standard self-play: the policy improves from the perspective of
all seats simultaneously.

AEC reward offset
─────────────────
In PettingZoo AEC, env.last() returns the reward accumulated since that
agent LAST acted — not the reward from the step that just ran. Concretely:

    agent 0 acts  →  reward_0 stored in cumulative_rewards[0]
    agent 1 acts  →  reward_1 stored in cumulative_rewards[1]
    agent 2 acts  →  reward_2 stored in cumulative_rewards[2]
    agent 0 acts  →  env.last() NOW returns reward_0 to agent 0

Transitions are therefore buffered per-agent and flushed with their reward
on the agent's following turn (or when they're processed as dead).

Prerequisites
─────────────
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# BSEnv exposes these module-level constants
from core.BSEnv import BSEnv, NUM_ACTIONS, NUM_PLAYERS, OBS_DIM


# ────────────────────────────────────────────────────────────────────────────
# Policy network
# ────────────────────────────────────────────────────────────────────────────

class BSPolicy(nn.Module):
    """
    Shared actor-critic network.

    trunk  → shared feature extractor (Tanh activations, standard for PPO)
    actor  → logit per action  (masked before sampling / log_prob)
    critic → scalar state value V(s)

    Orthogonal initialisation with PPO-standard gains.
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

    @torch.no_grad()
    def act(self, obs: torch.Tensor, mask: torch.Tensor):
        """
        Sample one action under mask. obs/mask are 1-D (single step).
        Returns (action_int, log_prob_float, value_float).
        """
        logits, value = self(obs.unsqueeze(0))
        logits = logits.squeeze(0).masked_fill(~mask, float("-inf"))
        dist   = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item(), value.item()

    def evaluate(
        self,
        obs:     torch.Tensor,   # [B, obs_dim]
        actions: torch.Tensor,   # [B]
        masks:   torch.Tensor,   # [B, action_dim]  bool
    ):
        """
        Re-compute log_prob, entropy, value for a batch.
        Called during the PPO update — gradients flow here.
        """
        logits, values = self(obs)
        logits = logits.masked_fill(~masks, float("-inf"))
        dist   = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), values


# ────────────────────────────────────────────────────────────────────────────
# Per-agent trajectory buffer
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class Trajectory:
    """Stores one agent's transitions from one episode."""
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
):
    """
    Generalised Advantage Estimation over a single agent's trajectory.

    done[t] = 1 means the episode ended at step t; next_value is zeroed
    and the GAE does not propagate backward through terminal transitions.

    Returns (advantages, returns) as float32 numpy arrays.
    """
    n   = len(rewards)
    adv = np.zeros(n, dtype=np.float32)
    last_gae = 0.0

    for t in reversed(range(n)):
        # Bootstrap value: zero at episode end, else V(s_{t+1})
        next_val  = values[t + 1] if t < n - 1 else 0.0
        next_val *= 1.0 - dones[t]          # zero if this step is terminal

        delta    = rewards[t] + gamma * next_val - values[t]
        last_gae = delta + gamma * lam * (1.0 - dones[t]) * last_gae
        adv[t]   = last_gae

    returns = adv + np.array(values, dtype=np.float32)
    return adv, returns


# ────────────────────────────────────────────────────────────────────────────
# Trainer
# ────────────────────────────────────────────────────────────────────────────

class BSTrainer:
    """
    Self-play PPO trainer.

    collect_episode()  →  one full game; returns per-agent Trajectory list
    update()           →  PPO update on a batch of trajectories
    train()            →  outer loop: collect → update → log → checkpoint
    """

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
        device:              Optional[str] = None,
    ):
        self.env    = BSEnv(max_iter=max_iter)
        # self.device = device or ("mps" if torch.mps.is_available() else "cpu")
        self.device = device or "cpu"
        self.policy = BSPolicy(OBS_DIM, NUM_ACTIONS, hidden_dim).to(self.device)
        self.opt    = optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)

        # Hyperparameters
        self.gamma            = gamma
        self.gae_lambda       = gae_lambda
        self.clip_eps         = clip_eps
        self.value_coef       = value_coef
        self.entropy_coef     = entropy_coef
        self.max_grad_norm    = max_grad_norm
        self.n_epochs         = n_epochs
        self.batch_size       = batch_size
        self.episodes_per_update = episodes_per_update

        # Metrics
        self.episode_count = 0
        self.update_count  = 0
        self.win_counts: Dict[int, int] = defaultdict(int)


    # ── episode collection ───────────────────────────────────────────────────

    def collect_episode(self) -> List[Trajectory]:
        """
        Run one complete self-play episode under the current policy.
        Returns one Trajectory per agent (3 total).

        Pending-dict pattern
        ────────────────────
        When agent A acts we buffer (obs, action, log_prob, value, mask).
        The reward for that action arrives via env.last() on A's NEXT turn.
        We flush the buffered transition with that reward at that point.
        Dead-agent iterations deliver terminal rewards the same way.
        """
        self.env.reset()
        trajectories: Dict[int, Trajectory] = {a: Trajectory() for a in range(NUM_PLAYERS)}

        # agent → (obs_arr, action, log_prob, value, mask_arr)
        pending: Dict[int, tuple] = {}

        for agent in self.env.agent_iter():
            obs_arr, reward, terminated, truncated, info = self.env.last()
            done = terminated or truncated

            # Flush the previous transition for this agent using the reward
            # that just arrived from env.last().
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

            # Select action under current policy
            obs_t  = torch.tensor(obs_arr,            dtype=torch.float32, device=self.device)
            mask_t = torch.tensor(info["action_mask"], dtype=torch.bool,   device=self.device)
            action, log_prob, value = self.policy.act(obs_t, mask_t)

            pending[agent] = (obs_arr, action, log_prob, value, info["action_mask"])
            self.env.step(action)

        # Agents still pending after the loop ended their last turn exactly as
        # the episode terminated. They may not have been iterated as dead agents.
        # Give them done=True and reward=0 (conservative; terminal rewards should
        # already have been delivered via cumulative_rewards when the loop ran).
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

    # ── PPO update ───────────────────────────────────────────────────────────

    def update(self, all_trajectories: List[Trajectory]) -> dict:
        """
        PPO update on a combined batch of trajectories from all agents
        and all episodes since the last update.

        Steps:
          1. Compute GAE per trajectory (preserves episode boundaries)
          2. Flatten all transitions into tensors
          3. Normalise advantages across the full batch
          4. n_epochs passes of minibatch PPO
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

        obs_t  = torch.tensor(np.array(obs_l),   dtype=torch.float32, device=self.device)
        act_t  = torch.tensor(act_l,              dtype=torch.long,    device=self.device)
        lp_t   = torch.tensor(lp_l,               dtype=torch.float32, device=self.device)
        adv_t  = torch.tensor(adv_l,              dtype=torch.float32, device=self.device)
        ret_t  = torch.tensor(ret_l,              dtype=torch.float32, device=self.device)
        mask_t = torch.tensor(np.array(mask_l),   dtype=torch.bool,    device=self.device)

        # Normalise advantages across the entire batch
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

                ratio   = torch.exp(log_prob - lp_t[b])
                adv_b   = adv_t[b]

                # Clipped policy objective
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
        return {k: float(np.mean(v)) for k, v in log.items()}

    # ── training loop ────────────────────────────────────────────────────────

    def train(
        self,
        total_episodes: int,
        log_interval:   int = 200,
        save_interval:  int = 5_000,
        checkpoint_dir: str = "checkpoints",
    ) -> BSPolicy:
        os.makedirs(checkpoint_dir, exist_ok=True)

        n_params = sum(p.numel() for p in self.policy.parameters())
        print(f"Device : {self.device}")
        print(f"Params : {n_params:,}")
        print(f"Target : {total_episodes:,} episodes, "
              f"update every {self.episodes_per_update}\n")

        t0 = time.time()
        buffer: List[Trajectory] = []

        while self.episode_count < total_episodes:

            # ── collect ──────────────────────────────────────────────────
            for _ in range(self.episodes_per_update):
                buffer.extend(self.collect_episode())
                if self.episode_count >= total_episodes:
                    break

            # ── update ───────────────────────────────────────────────────
            metrics = self.update(buffer)
            buffer.clear()

            # ── log ──────────────────────────────────────────────────────
            if self.episode_count % log_interval < self.episodes_per_update:
                total_wins = max(sum(self.win_counts.values()), 1)
                win_str = "  ".join(
                    f"p{a}:{self.win_counts[a]/total_wins:5.1%}"
                    for a in range(NUM_PLAYERS)
                )
                print(
                    f"ep {self.episode_count:>7,} | "
                    f"upd {self.update_count:>4,} | "
                    f"t {time.time()-t0:>6.0f}s | "
                    f"wins [{win_str}] | "
                    f"π {metrics.get('policy_loss', 0):+.4f}  "
                    f"v {metrics.get('value_loss',  0):.4f}  "
                    f"H {metrics.get('entropy',     0):.3f}"
                )

            # ── checkpoint ───────────────────────────────────────────────
            if self.episode_count % save_interval < self.episodes_per_update:
                path = os.path.join(
                    checkpoint_dir, f"policy_{self.episode_count:07d}.pt"
                )
                self.save(path)

        print("\nTraining complete.")
        return self.policy

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        torch.save({
            "policy":        self.policy.state_dict(),
            "optimizer":     self.opt.state_dict(),
            "episode_count": self.episode_count,
            "update_count":  self.update_count,
            "win_counts":    dict(self.win_counts),
        }, path)
        print(f"  → saved {path}")

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])
        self.opt.load_state_dict(ckpt["optimizer"])
        self.episode_count = ckpt["episode_count"]
        self.update_count  = ckpt["update_count"]
        self.win_counts    = defaultdict(int, ckpt["win_counts"])
        print(f"  → loaded {path}")


# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────
import cProfile


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
    )

    trainer.train(
        total_episodes = 100_000,
        log_interval   = 200,
        save_interval  = 5_000,
    )
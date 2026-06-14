"""
evaluate.py — Agent evaluation suite for BSEnv.

Run a configurable tournament between any mix of:
  - PPO policy checkpoint        (--ppo path/to/checkpoint.pt)
  - PPO + LSTM policy checkpoint (--lstm path/to/checkpoint.pt)
  - Naive agents                 (--naive random|conservative|aggressive|threshold|all)
  - Human player                 (--human)

Usage examples
──────────────
# 1v1v1: PPO vs LSTM vs ThresholdAgent, 500 games
python evaluate.py \
    --ppo  checkpoints/policy_0100000.pt \
    --lstm checkpoints_lstm/policy_0100000.pt \
    --naive threshold \
    --games 500

# All naive agents round-robin, 200 games
python evaluate.py --naive all --games 200

# Human vs two naive agents, single game with verbose output
python evaluate.py --human --naive aggressive --naive conservative --games 1 --verbose

# LSTM vs three different naive agents (fill remaining seats)
python evaluate.py --lstm checkpoints_lstm/policy_0100000.pt --naive all --games 1000

Seats are filled left-to-right from the agents specified on the command line.
If more than 3 agents are specified, all pairwise 3-seat subsets are tested.
If fewer than 3 agents are specified, the remaining seats are filled with
RandomAgent instances.

Output
──────
Per-run summary table printed to stdout.
Optional CSV export with --csv path/to/results.csv.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# ── project imports ──────────────────────────────────────────────────────────
from core.BSEnv import BSEnv, NUM_ACTIONS, NUM_PLAYERS, OBS_DIM
from agents import (
    Agent, PolicyAgent, make_naive_agents,
    RandomAgent, ConservativeAgent, AggressiveAgent, ThresholdAgent,
)


# ────────────────────────────────────────────────────────────────────────────
# Lazy import: LSTM types only needed if an LSTM checkpoint is requested
# ────────────────────────────────────────────────────────────────────────────

def _import_lstm():
    """Import LSTM classes from the LSTM trainer module."""
    try:
        from trainLSTM import (
            LSTMEncoder, BSPolicy as LSTMBSPolicy,
            LSTMPolicyAgent, AUG_OBS_DIM,
        )
        return LSTMEncoder, LSTMBSPolicy, LSTMPolicyAgent, AUG_OBS_DIM
    except ImportError as e:
        print(f"[error] Could not import LSTM module: {e}")
        print("        Make sure trainLSTM.py is on the Python path.")
        sys.exit(1)


def _import_ppo():
    """Import vanilla PPO classes from the standard trainer module."""
    try:
        from trainWithPool import BSPolicy
        return BSPolicy
    except ImportError as e:
        print(f"[error] Could not import PPO module: {e}")
        print("        Make sure trainWithPool.py is on the Python path.")
        sys.exit(1)


# ────────────────────────────────────────────────────────────────────────────
# Checkpoint loaders
# ────────────────────────────────────────────────────────────────────────────

def load_ppo_agent(
    path:       str,
    device:     str,
    hidden_dim: int = 128,
    label:      Optional[str] = None,
) -> "EvalAgent":
    """Load a vanilla PPO checkpoint and return an EvalAgent wrapper."""
    BSPolicy = _import_ppo()
    ckpt = torch.load(path, map_location=device)

    policy = BSPolicy(OBS_DIM, NUM_ACTIONS, hidden_dim)
    policy.load_state_dict(ckpt["policy"])
    policy.to(device)
    policy.eval()

    ep = ckpt.get("episode_count", "?")
    name = label or f"PPO(ep={ep})"
    return EvalPolicyAgent(PolicyAgent(policy, device, is_trainable=False), name=name)


def load_lstm_agent(
    path:       str,
    device:     str,
    hidden_dim: int = 128,
    label:      Optional[str] = None,
) -> "EvalAgent":
    """Load a PPO+LSTM checkpoint and return an EvalAgent wrapper."""
    LSTMEncoder, LSTMBSPolicy, LSTMPolicyAgent, AUG_OBS_DIM = _import_lstm()
    ckpt = torch.load(path, map_location=device)

    encoder = LSTMEncoder()
    encoder.load_state_dict(ckpt["encoder"])
    encoder.to(device)
    encoder.eval()

    policy = LSTMBSPolicy(AUG_OBS_DIM, NUM_ACTIONS, hidden_dim)
    policy.load_state_dict(ckpt["policy"])
    policy.to(device)
    policy.eval()

    ep = ckpt.get("episode_count", "?")
    name = label or f"LSTM(ep={ep})"
    inner = LSTMPolicyAgent(encoder, policy, device, is_trainable=False)
    return EvalLSTMAgent(inner, name=name)


# ────────────────────────────────────────────────────────────────────────────
# EvalAgent wrappers — unified interface for the evaluation loop
# ────────────────────────────────────────────────────────────────────────────
#
# The evaluation loop calls:
#   agent.act(obs, mask, seat)   → int (action)
#   agent.reset()                → None (called at episode start)
#
# Human agents print the game state and read from stdin.
# ────────────────────────────────────────────────────────────────────────────

class EvalAgent:
    """Abstract base for evaluation-time agent wrappers."""
    name: str

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray, mask: np.ndarray, seat: int) -> int:
        raise NotImplementedError


class EvalPolicyAgent(EvalAgent):
    """Wraps a vanilla PolicyAgent."""
    def __init__(self, inner: PolicyAgent, name: str):
        self._inner = inner
        self.name   = name

    def act(self, obs: np.ndarray, mask: np.ndarray, seat: int) -> int:
        action, _, _ = self._inner.act(obs, mask)
        return action


class EvalLSTMAgent(EvalAgent):
    """Wraps an LSTMPolicyAgent; manages per-seat hidden state."""
    def __init__(self, inner, name: str):
        self._inner = inner
        self.name   = name
        self._seat: Optional[int] = None

    def reset(self) -> None:
        # Hidden state is reset via reset_hidden; we need to know the seat first,
        # so actual reset happens on first act() call of a new episode.
        self._needs_reset = True

    def act(self, obs: np.ndarray, mask: np.ndarray, seat: int) -> int:
        if getattr(self, "_needs_reset", True):
            self._inner.reset_hidden([seat])
            self._needs_reset = False
            self._seat = seat
        action, _, _, _ = self._inner.act(obs, mask, seat=seat)
        return action


class EvalNaiveAgent(EvalAgent):
    """Wraps any naive Agent."""
    def __init__(self, inner: Agent, name: str):
        self._inner = inner
        self.name   = name

    def act(self, obs: np.ndarray, mask: np.ndarray, seat: int) -> int:
        action, _, _ = self._inner.act(obs, mask)
        return action


class HumanAgent(EvalAgent):
    """Interactive human player — reads from stdin."""
    name = "Human"

    # Declare action descriptions for display
    _DECLARE_LABELS = [
        f"declare {qty} card(s), {honest} honest"
        for qty in range(1, 5)
        for honest in range(0, qty + 1)
    ]

    def act(self, obs: np.ndarray, mask: np.ndarray, seat: int) -> int:
        phase    = "CHALLENGE" if obs[17] > 0.5 else "DECLARE"
        pile_sz  = int(round(obs[15] * 52))
        claim_q  = int(round(obs[16] * 4))
        hand_sz  = int(round(obs[18] * 52))

        print(f"\n{'─'*50}")
        print(f"  [YOU — seat {seat}]  phase: {phase}")
        print(f"  Pile: {pile_sz} cards   Last claim: {claim_q or '—'}")
        print(f"  Your hand size: {hand_sz}")
        print()

        valid = np.where(mask)[0]
        action_map = {}
        print("  Valid actions:")
        for i, a in enumerate(valid):
            if a == 14:
                label = "Challenge"
            elif a == 15:
                label = "Pass challenge"
            else:
                label = self._DECLARE_LABELS[a]
            print(f"    [{i}] {label}  (action {a})")
            action_map[i] = int(a)

        while True:
            try:
                choice = int(input(f"\n  Your choice [0-{len(valid)-1}]: ").strip())
                if choice in action_map:
                    print(f"  → {action_map[choice]}: "
                          f"{'Challenge' if action_map[choice]==14 else 'Pass' if action_map[choice]==15 else self._DECLARE_LABELS[action_map[choice]]}")
                    return action_map[choice]
            except (ValueError, KeyboardInterrupt):
                pass
            print("  Invalid — try again.")


# ────────────────────────────────────────────────────────────────────────────
# Factory helpers
# ────────────────────────────────────────────────────────────────────────────

_NAIVE_MAP = {
    "random":       lambda: EvalNaiveAgent(RandomAgent(),       "Random"),
    "conservative": lambda: EvalNaiveAgent(ConservativeAgent(), "Conservative"),
    "aggressive":   lambda: EvalNaiveAgent(AggressiveAgent(),   "Aggressive"),
    "threshold":    lambda: EvalNaiveAgent(ThresholdAgent(),    "Threshold"),
}


def make_naive_eval_agent(kind: str) -> EvalAgent:
    if kind not in _NAIVE_MAP:
        raise ValueError(f"Unknown naive agent '{kind}'. "
                         f"Choose from: {list(_NAIVE_MAP)}")
    return _NAIVE_MAP[kind]()


# ────────────────────────────────────────────────────────────────────────────
# Match result
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class MatchResult:
    """Stats for a single game."""
    seat_names: List[str]           # name of agent in each seat
    winner_seat: Optional[int]      # None = truncated
    n_moves:     int
    truncated:   bool


@dataclass
class TournamentStats:
    """Aggregate stats across multiple games for one matchup."""
    seat_names:    List[str]
    shuffled:      bool             = False
    wins:          Dict[int, int]   = field(default_factory=lambda: defaultdict(int))  # by canonical agent index
    seat_wins:     Dict[int, int]   = field(default_factory=lambda: defaultdict(int))  # by physical seat
    truncations:   int              = 0
    total_games:   int              = 0
    total_moves:   int              = 0
    move_samples:  List[int]        = field(default_factory=list)

    def record(self, result: MatchResult, physical_seat: Optional[int] = None) -> None:
        self.total_games += 1
        self.total_moves += result.n_moves
        self.move_samples.append(result.n_moves)
        if result.truncated:
            self.truncations += 1
        elif result.winner_seat is not None:
            self.wins[result.winner_seat] += 1          # canonical agent index
            ps = physical_seat if physical_seat is not None else result.winner_seat
            self.seat_wins[ps] += 1                     # physical seat

    def seat_win_rate(self, seat: int) -> float:
        """Win rate by canonical agent index (seat-normalized in shuffled mode)."""
        completed = self.total_games - self.truncations
        if completed == 0:
            return 0.0
        return self.wins[seat] / completed

    def physical_seat_win_rate(self, seat: int) -> float:
        """Win rate by physical seat position (only meaningful in fixed mode)."""
        completed = self.total_games - self.truncations
        if completed == 0:
            return 0.0
        return self.seat_wins[seat] / completed

    @property
    def mean_moves(self) -> float:
        return float(np.mean(self.move_samples)) if self.move_samples else 0.0

    @property
    def median_moves(self) -> float:
        return float(np.median(self.move_samples)) if self.move_samples else 0.0


# ────────────────────────────────────────────────────────────────────────────
# Core evaluation loop
# ────────────────────────────────────────────────────────────────────────────

def run_game(
    env:     BSEnv,
    agents:  List[EvalAgent],   # len == NUM_PLAYERS, indexed by seat
    verbose: bool = False,
    seed:    Optional[int] = None,
) -> MatchResult:
    """
    Run one full game. Returns a MatchResult.

    agents[i] acts when env.agent_selection == i.
    """
    for agent in agents:
        agent.reset()

    env.reset(seed=seed)

    n_moves   = 0
    truncated = False

    for seat in env.agent_iter():
        obs, reward, terminated, trunc, info = env.last()
        done = terminated or trunc

        if done:
            env.step(None)
            if trunc:
                truncated = True
            continue

        mask   = info["action_mask"]
        action = agents[seat].act(obs, mask, seat=seat)

        if verbose:
            _print_step(seat, agents[seat].name, action, env)

        env.step(action)
        n_moves += 1

    winner_seat = None
    if env.state is not None and env.state.winner is not None:
        winner_seat = env.state.winner

    return MatchResult(
        seat_names  = [a.name for a in agents],
        winner_seat = winner_seat,
        n_moves     = n_moves,
        truncated   = truncated,
    )


def run_tournament(
    agents:     List[EvalAgent],
    n_games:    int,
    max_iter:   int  = 5_000,
    verbose:    bool = False,
    seed_start: int  = 0,
) -> TournamentStats:
    """
    Run n_games with a fixed seat assignment.
    Wins are tracked by seat index.
    """
    env   = BSEnv(max_iter=max_iter)
    stats = TournamentStats(seat_names=[a.name for a in agents], shuffled=False)
    use_progress = n_games >= 20 and not verbose

    t0 = time.time()
    for i in range(n_games):
        result = run_game(env, agents, verbose=verbose, seed=seed_start + i)
        stats.record(result)

        if use_progress and (i + 1) % max(1, n_games // 20) == 0:
            pct  = (i + 1) / n_games * 100
            bar  = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            rate = (i + 1) / (time.time() - t0 + 1e-9)
            print(f"\r  [{bar}] {pct:5.1f}%  {i+1}/{n_games}  ({rate:.1f} games/s)",
                  end="", flush=True)

    if use_progress:
        print()

    return stats


def run_shuffled_tournament(
    agents:     List[EvalAgent],
    n_games:    int,
    max_iter:   int  = 5_000,
    verbose:    bool = False,
    seed_start: int  = 0,
) -> TournamentStats:
    """
    Run n_games with agents rotated evenly through all seat permutations.

    All 3! = 6 permutations are used. Games are distributed as evenly as
    possible across permutations (remainder games go to the first perms).
    Wins are tracked both by seat index and by agent identity so both
    positional and agent-level win rates can be reported.
    """
    perms      = list(itertools.permutations(range(len(agents))))
    n_perms    = len(perms)
    base, rem  = divmod(n_games, n_perms)
    env        = BSEnv(max_iter=max_iter)
    stats      = TournamentStats(seat_names=[a.name for a in agents], shuffled=True)
    use_progress = n_games >= 20 and not verbose

    game_i = 0
    t0     = time.time()

    for pi, perm in enumerate(perms):
        perm_games   = base + (1 if pi < rem else 0)
        seated       = [agents[perm[s]] for s in range(NUM_PLAYERS)]

        for _ in range(perm_games):
            result = run_game(env, seated, verbose=verbose, seed=seed_start + game_i)
            # Remap winner seat back to original agent index before recording
            remapped = MatchResult(
                seat_names  = [a.name for a in agents],   # canonical order
                winner_seat = perm[result.winner_seat] if result.winner_seat is not None else None,
                n_moves     = result.n_moves,
                truncated   = result.truncated,
            )
            stats.record(remapped, physical_seat=result.winner_seat)
            game_i += 1

            if use_progress and game_i % max(1, n_games // 20) == 0:
                pct  = game_i / n_games * 100
                bar  = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                rate = game_i / (time.time() - t0 + 1e-9)
                print(f"\r  [{bar}] {pct:5.1f}%  {game_i}/{n_games}  ({rate:.1f} games/s)",
                      end="", flush=True)

    if use_progress:
        print()

    return stats


# ────────────────────────────────────────────────────────────────────────────
# Multi-matchup: round-robin over all 3-seat subsets
# ────────────────────────────────────────────────────────────────────────────

def run_round_robin(
    agents:     List[EvalAgent],
    n_games:    int,
    max_iter:   int  = 5_000,
    verbose:    bool = False,
    shuffle:    bool = True,
) -> List[TournamentStats]:
    """
    If more than 3 agents are provided, run every combination of 3.
    With shuffle=True (default), each matchup uses run_shuffled_tournament
    so win rates are seat-normalized. With shuffle=False the fixed seat
    order is preserved (useful for debugging positional effects).
    """
    while len(agents) < NUM_PLAYERS:
        agents.append(EvalNaiveAgent(RandomAgent(), "Random(filler)"))

    runner = run_shuffled_tournament if shuffle else run_tournament

    if len(agents) == NUM_PLAYERS:
        return [runner(agents, n_games, max_iter, verbose)]

    all_stats = []
    combos = list(itertools.combinations(range(len(agents)), NUM_PLAYERS))
    print(f"\n  {len(combos)} matchup(s) × {n_games} games each "
          f"({'seat-shuffled' if shuffle else 'fixed seats'})\n")

    for combo in combos:
        trio = [agents[i] for i in combo]
        print(f"  Matchup: {' vs '.join(a.name for a in trio)}")
        stats = runner(trio, n_games, max_iter, verbose)
        all_stats.append(stats)

    return all_stats


# ────────────────────────────────────────────────────────────────────────────
# Display
# ────────────────────────────────────────────────────────────────────────────

def _print_step(seat: int, name: str, action: int, env: BSEnv) -> None:
    from core.BSEnv import declareActions
    phase = env.state.current_phase.name if env.state else "?"
    if action == 14:
        label = "CHALLENGE"
    elif action == 15:
        label = "PASS"
    elif action < 14:
        qty, honest = declareActions[action]
        label = f"declare {qty} ({honest} honest)"
    else:
        label = f"action {action}"
    print(f"  seat {seat} [{name:>14s}]  {phase:<10s}  {label}")


def print_stats(stats: TournamentStats) -> None:
    completed  = stats.total_games - stats.truncations
    mode_label = "seat-shuffled" if stats.shuffled else "fixed seats"

    print(f"\n  ┌{'─'*56}┐")
    print(f"  │  Agents: {' vs '.join(stats.seat_names):<47}│")
    print(f"  │  Mode: {mode_label:<49}│")
    print(f"  ├{'─'*56}┤")
    print(f"  │  Games:      {stats.total_games:<6}  "
          f"Completed: {completed:<6}  "
          f"Truncated: {stats.truncations:<5}  │")
    print(f"  │  Moves/game: mean {stats.mean_moves:>5.1f}   "
          f"median {stats.median_moves:>5.1f}{' '*15}│")
    print(f"  ├{'─'*56}┤")

    if stats.shuffled:
        # Primary table: per-agent win rates (seat-normalized)
        print(f"  │  {'Agent':<22}  {'Wins':>6}  {'Win rate':>9}  {'':>5}  │")
        print(f"  ├{'─'*56}┤")
        for i, name in enumerate(stats.seat_names):
            w    = stats.wins[i]
            rate = stats.seat_win_rate(i)
            print(f"  │  {name:<22}  {w:>6}  {rate:>8.1%}  {'':>5}  │")

        # Secondary breakdown: raw positional win rates
        print(f"  ├{'─'*56}┤")
        print(f"  │  Positional breakdown (averaged across agents):{' '*7}│")
        for s in range(NUM_PLAYERS):
            rate = stats.physical_seat_win_rate(s)
            bar  = "█" * int(rate * 20) + "░" * (20 - int(rate * 20))
            print(f"  │    p{s}  {bar}  {rate:>5.1%}{' '*13}│")
    else:
        # Fixed-seat mode: show per-seat as before
        print(f"  │  {'Agent':<22}  {'Wins':>6}  {'Win rate':>9}  {'Seat':>4}  │")
        print(f"  ├{'─'*56}┤")
        for i, name in enumerate(stats.seat_names):
            w    = stats.wins[i]
            rate = stats.seat_win_rate(i)
            print(f"  │  {name:<22}  {w:>6}  {rate:>8.1%}  {'p'+str(i):>4}  │")

    print(f"  └{'─'*56}┘\n")


def print_all_stats(all_stats: List[TournamentStats]) -> None:
    for stats in all_stats:
        print_stats(stats)

    if len(all_stats) > 1:
        # Aggregate by agent name across all matchups
        total_wins:   Dict[str, int] = defaultdict(int)
        total_appear: Dict[str, int] = defaultdict(int)
        for stats in all_stats:
            completed = stats.total_games - stats.truncations
            for seat, name in enumerate(stats.seat_names):
                total_wins[name]   += stats.wins[seat]
                total_appear[name] += completed

        print(f"  ┌{'─'*40}┐")
        print(f"  │  Overall win rate (all matchups)       │")
        print(f"  ├{'─'*40}┤")
        for name in sorted(total_appear, key=lambda n: -total_wins.get(n, 0)):
            appear = total_appear[name]
            wins   = total_wins.get(name, 0)
            rate   = wins / appear if appear > 0 else 0.0
            print(f"  │  {name:<20}  {rate:>6.1%}  ({wins}/{appear})  │")
        print(f"  └{'─'*40}┘\n")


# ────────────────────────────────────────────────────────────────────────────
# CSV export
# ────────────────────────────────────────────────────────────────────────────

def export_csv(all_stats: List[TournamentStats], path: str) -> None:
    rows = []
    for stats in all_stats:
        matchup   = " vs ".join(stats.seat_names)
        completed = stats.total_games - stats.truncations
        for seat, name in enumerate(stats.seat_names):
            wins = stats.wins[seat]
            rate = wins / completed if completed > 0 else 0.0
            rows.append({
                "matchup":    matchup,
                "seat":       seat,
                "agent":      name,
                "games":      stats.total_games,
                "completed":  completed,
                "truncated":  stats.truncations,
                "wins":       wins,
                "win_rate":   f"{rate:.4f}",
                "mean_moves": f"{stats.mean_moves:.1f}",
            })

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  → CSV saved to {path}")


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="BSEnv agent evaluation suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--ppo",    metavar="PATH", action="append", default=[],
                   help="Path to a vanilla PPO checkpoint (repeatable)")
    p.add_argument("--lstm",   metavar="PATH", action="append", default=[],
                   help="Path to a PPO+LSTM checkpoint (repeatable)")
    p.add_argument("--naive",  metavar="KIND", action="append", default=[],
                   help="Add a naive agent: random|conservative|aggressive|threshold|all "
                        "(repeatable; 'all' adds one of each)")
    p.add_argument("--human",  action="store_true",
                   help="Add a human player (reads from stdin)")
    p.add_argument("--games",  type=int, default=100,
                   help="Number of games per matchup (default: 100)")
    p.add_argument("--max-iter", type=int, default=5_000,
                   help="Max moves per game before truncation (default: 5000)")
    p.add_argument("--hidden-dim", type=int, default=128,
                   help="Policy hidden dim — must match checkpoint (default: 128)")
    p.add_argument("--device", default=None,
                   help="torch device (default: cuda if available, else cpu)")
    p.add_argument("--verbose", action="store_true",
                   help="Print every action (recommended only for --games 1)")
    p.add_argument("--no-shuffle", action="store_true",
                   help="Disable seat rotation; report raw positional win rates")
    p.add_argument("--csv",    metavar="PATH", default=None,
                   help="Export results to CSV")
    p.add_argument("--seed",   type=int, default=0,
                   help="Starting RNG seed (default: 0)")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    agents: List[EvalAgent] = []

    # ── Load PPO checkpoints ────────────────────────────────────────────────
    for path in args.ppo:
        print(f"  Loading PPO checkpoint: {path}")
        agents.append(load_ppo_agent(path, device, args.hidden_dim))

    # ── Load LSTM checkpoints ───────────────────────────────────────────────
    for path in args.lstm:
        print(f"  Loading LSTM checkpoint: {path}")
        agents.append(load_lstm_agent(path, device, args.hidden_dim))

    # ── Naive agents ────────────────────────────────────────────────────────
    naive_kinds = {
        "all": ["random", "conservative", "aggressive", "threshold"],
    }
    for kind in args.naive:
        for k in naive_kinds.get(kind, [kind]):
            agents.append(make_naive_eval_agent(k))
            print(f"  Added naive agent: {k}")

    # ── Human ───────────────────────────────────────────────────────────────
    if args.human:
        agents.append(HumanAgent())
        print("  Added human player")

    if not agents:
        print("\n  No agents specified — running default: all 4 naive agents.\n")
        for k in ["random", "conservative", "aggressive", "threshold"]:
            agents.append(make_naive_eval_agent(k))

    shuffle = not args.no_shuffle
    print(f"\n  Agents ({len(agents)}): {', '.join(a.name for a in agents)}")
    print(f"  Games per matchup: {args.games}  |  Device: {device}  |  "
          f"Seats: {'shuffled' if shuffle else 'fixed'}\n")

    # ── Run ─────────────────────────────────────────────────────────────────
    all_stats = run_round_robin(
        agents   = agents,
        n_games  = args.games,
        max_iter = args.max_iter,
        verbose  = args.verbose,
        shuffle  = shuffle,
    )

    print_all_stats(all_stats)

    if args.csv:
        export_csv(all_stats, args.csv)


# ────────────────────────────────────────────────────────────────────────────
# Programmatic API (import and call directly without CLI)
# ────────────────────────────────────────────────────────────────────────────

def quick_eval(
    agents:   List[EvalAgent],
    n_games:  int = 200,
    max_iter: int = 5_000,
    verbose:  bool = False,
) -> List[TournamentStats]:
    """
    Convenience function for use in notebooks or other scripts.

    Example:
        from evaluate import quick_eval, load_ppo_agent, load_lstm_agent, make_naive_eval_agent

        ppo  = load_ppo_agent("checkpoints/policy_0100000.pt", device="cpu")
        lstm = load_lstm_agent("checkpoints_lstm/policy_0100000.pt", device="cpu")
        rand = make_naive_eval_agent("random")

        stats = quick_eval([ppo, lstm, rand], n_games=500)
        print_all_stats(stats)
    """
    all_stats = run_round_robin(agents, n_games, max_iter, verbose)
    return all_stats


if __name__ == "__main__":
    main()
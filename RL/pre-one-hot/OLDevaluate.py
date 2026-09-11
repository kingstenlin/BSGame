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
        print("        Make sure trainWithPool_LSTM.py is on the Python path.")
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

# ════════════════════════════════════════════════════════════════════════════
# LSTM DIAGNOSTICS
# ════════════════════════════════════════════════════════════════════════════

# ────────────────────────────────────────────────────────────────────────────
# 1. Hidden state activation analysis
# ────────────────────────────────────────────────────────────────────────────

def analyze_hidden_state(
    lstm_eval_agent,
    n_games:    int = 50,
    max_iter:   int = 5_000,
    seed_start: int = 0,
    opponent:   Optional[EvalAgent] = None,
) -> dict:
    """
    Collect hidden state trajectories across n_games and report:
      - mean / std of h_t across timesteps (per-game and pooled)
      - mean / std of latent vectors at the policy input
      - dead-unit rate: fraction of hidden units with std < 0.01 across time

    If the LSTM is using memory, h_t should vary meaningfully across
    timesteps within a game. Near-zero std = encoder collapsed to constant.
    """
    encoder = lstm_eval_agent._inner.encoder
    device  = lstm_eval_agent._inner.device
    opp     = opponent or EvalNaiveAgent(RandomAgent(), "Random")
    env     = BSEnv(max_iter=max_iter)

    all_h_stds:      List[float]      = []
    all_latent_stds: List[float]      = []
    all_h_vecs:      List[np.ndarray] = []

    for game_i in range(n_games):
        agents = [lstm_eval_agent, opp, opp]
        lstm_eval_agent.reset()
        env.reset(seed=seed_start + game_i)
        h_seq:      List[np.ndarray] = []
        latent_seq: List[np.ndarray] = []

        for seat in env.agent_iter():
            obs, _, terminated, truncated, info = env.last()
            if terminated or truncated:
                env.step(None)
                continue

            if seat == 0:
                obs_t = torch.tensor(obs, dtype=torch.float32,
                                     device=device).unsqueeze(0)
                hx = lstm_eval_agent._inner._hidden.get(0, None)
                with torch.no_grad():
                    latent, new_hx = encoder(obs_t, hx)
                h_seq.append(new_hx[0].squeeze().cpu().numpy())
                latent_seq.append(latent.squeeze().cpu().numpy())

            action = agents[seat].act(obs, info["action_mask"], seat=seat)
            env.step(action)

        if len(h_seq) > 1:
            h_arr = np.stack(h_seq)
            l_arr = np.stack(latent_seq)
            all_h_stds.append(float(h_arr.std(axis=0).mean()))
            all_latent_stds.append(float(l_arr.std(axis=0).mean()))
            all_h_vecs.extend(h_seq)

    h_pool    = np.stack(all_h_vecs)
    unit_stds = h_pool.std(axis=0)
    dead_rate = float((unit_stds < 0.01).mean())

    results = {
        "h_temporal_std_mean": float(np.mean(all_h_stds)),
        "h_temporal_std_std":  float(np.std(all_h_stds)),
        "latent_std_mean":     float(np.mean(all_latent_stds)),
        "latent_std_std":      float(np.std(all_latent_stds)),
        "dead_unit_rate":      dead_rate,
        "hidden_unit_stds":    unit_stds,
        "n_games":             n_games,
    }
    _print_hidden_analysis(results)
    return results


def _print_hidden_analysis(r: dict) -> None:
    print(f"\n  ┌{'─'*52}┐")
    print(f"  │  LSTM Hidden State Analysis ({r['n_games']} games){' '*10}│")
    print(f"  ├{'─'*52}┤")
    h_note = "← near-zero = collapsed" if r["h_temporal_std_mean"] < 0.02 else "← varying (good)"
    l_note = "← saturated/ignored"     if r["latent_std_mean"]     < 0.02 else "← informative (good)"
    print(f"  │  h_t temporal std (mean ± std across games):{' '*6}│")
    print(f"  │    {r['h_temporal_std_mean']:.4f} ± {r['h_temporal_std_std']:.4f}  {h_note:<24}│")
    print(f"  │  Latent std at policy input (mean ± std):{' '*9}│")
    print(f"  │    {r['latent_std_mean']:.4f} ± {r['latent_std_std']:.4f}  {l_note:<24}│")
    print(f"  │  Dead hidden units (std < 0.01): {r['dead_unit_rate']:>5.1%}{' '*13}│")
    stds   = r["hidden_unit_stds"]
    bins   = [0, 0.01, 0.05, 0.1, 0.2, float("inf")]
    labels = ["<0.01", "0.01-0.05", "0.05-0.1", "0.1-0.2", ">0.2"]
    print(f"  ├{'─'*52}┤")
    print(f"  │  Hidden unit std distribution:{' '*21}│")
    for i, label in enumerate(labels):
        count = int(((stds >= bins[i]) & (stds < bins[i+1])).sum())
        bar   = "█" * min(count, 28)
        print(f"  │    {label:<10} {bar:<28} {count:>3}  │")
    print(f"  └{'─'*52}┘\n")


# ────────────────────────────────────────────────────────────────────────────
# 2. Zeroed hidden state ablation
# ────────────────────────────────────────────────────────────────────────────

class _ZeroedHiddenLSTMAgent(EvalLSTMAgent):
    """EvalLSTMAgent with hidden state forced to zero before every act()."""
    def act(self, obs: np.ndarray, mask: np.ndarray, seat: int) -> int:
        self._inner._hidden[seat] = None
        self._needs_reset = False
        action, _, _, _ = self._inner.act(obs, mask, seat=seat)
        return action


def ablation_zeroed_hidden(
    lstm_eval_agent,
    opponents: List[EvalAgent],
    n_games:   int = 200,
    max_iter:  int = 5_000,
) -> None:
    """
    Compare LSTM (normal) vs LSTM (zeroed hx every step).
    If win rates are indistinguishable, the policy is not using its memory.
    """
    zeroed = _ZeroedHiddenLSTMAgent(
        lstm_eval_agent._inner,
        name=f"{lstm_eval_agent.name}[zeroed-hx]",
    )

    print(f"\n  Ablation: {lstm_eval_agent.name} vs zeroed-hx version")
    opps = list(opponents[:NUM_PLAYERS - 1])
    while len(opps) < NUM_PLAYERS - 1:
        opps.append(EvalNaiveAgent(RandomAgent(), "Random(filler)"))

    normal_stats = run_shuffled_tournament([lstm_eval_agent] + opps, n_games, max_iter)
    zeroed_stats = run_shuffled_tournament([zeroed]          + opps, n_games, max_iter)

    nwr   = normal_stats.seat_win_rate(0)
    zwr   = zeroed_stats.seat_win_rate(0)
    delta = nwr - zwr

    if abs(delta) < 0.03:
        verdict = "⚠  No meaningful difference — LSTM may not use memory"
    elif delta > 0:
        verdict = "✓  Normal LSTM outperforms — hidden state contributes"
    else:
        verdict = "✗  Zeroed LSTM wins — hidden state is hurting performance"

    print(f"\n  ┌{'─'*52}┐")
    print(f"  │  Zeroed Hidden State Ablation{' '*22}│")
    print(f"  ├{'─'*52}┤")
    print(f"  │  {lstm_eval_agent.name:<30}  win rate: {nwr:>5.1%}  │")
    print(f"  │  {zeroed.name:<30}  win rate: {zwr:>5.1%}  │")
    print(f"  ├{'─'*52}┤")
    print(f"  │  Δ = {delta:+.1%}   {verdict:<38}│")
    print(f"  └{'─'*52}┘\n")


# ════════════════════════════════════════════════════════════════════════════
# SCENARIO TESTING
# ════════════════════════════════════════════════════════════════════════════

RANK_NAMES       = ["A","2","3","4","5","6","7","8","9","10","J","Q","K"]
_RANK_NAME_TO_IDX = {n: i for i, n in enumerate(RANK_NAMES)}

_DECLARE_LABELS = [
    f"declare {qty} ({honest} honest)"
    for qty in range(1, 5)
    for honest in range(0, qty + 1)
]
_ACTION_LABELS = _DECLARE_LABELS + ["Challenge", "Pass"]


@dataclass
class ScenarioSpec:
    """
    Specifies a single-step game state for probing agent decisions.

    hand:         Rank strings from acting agent's perspective, e.g. ["A","A","K","3"]
    current_rank: Rank being declared, e.g. "A"
    phase:        "DECLARE" or "CHALLENGE"
    pile_size:    Cards in the central pile (0-52)
    claim_qty:    Quantity in the last declaration (0 = no prior claim)
    hand_sizes:   [curr, next, prev] player hand sizes; defaults to [len(hand), 17, 17]

    Example — impossible four-of-a-kind bluff:
        ScenarioSpec(
            hand=["A","2","3","K"], current_rank="A",
            phase="CHALLENGE", pile_size=8, claim_qty=4,
        )
    The agent holds 1 Ace, so a claim of 4 is impossible. Does it challenge?
    """
    hand:         List[str]
    current_rank: str                  = "A"
    phase:        str                  = "CHALLENGE"
    pile_size:    int                  = 0
    claim_qty:    int                  = 0
    hand_sizes:   Optional[List[int]]  = None

    def __post_init__(self) -> None:
        for r in self.hand:
            if r not in _RANK_NAME_TO_IDX:
                raise ValueError(f"Unknown rank '{r}'. Use: {RANK_NAMES}")
        if self.current_rank not in _RANK_NAME_TO_IDX:
            raise ValueError(f"Unknown current_rank '{self.current_rank}'")
        if self.phase not in ("DECLARE", "CHALLENGE"):
            raise ValueError("phase must be 'DECLARE' or 'CHALLENGE'")

    def to_obs(self) -> np.ndarray:
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        for r in self.hand:
            obs[_RANK_NAME_TO_IDX[r]] += 0.25
        ri = _RANK_NAME_TO_IDX[self.current_rank]
        obs[13] = np.sin(2 * np.pi * ri / 13) / 2 + 0.5
        obs[14] = np.cos(2 * np.pi * ri / 13) / 2 + 0.5
        obs[15] = self.pile_size / 52.0
        obs[16] = self.claim_qty / 4.0
        obs[17] = 0.0 if self.phase == "DECLARE" else 1.0
        sizes   = self.hand_sizes or [len(self.hand), 17, 17]
        for i, s in enumerate(sizes):
            obs[18 + i] = s / 52.0
        return obs

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(NUM_ACTIONS, dtype=np.int8)
        if self.phase == "CHALLENGE":
            mask[14] = 1
            mask[15] = 1
        else:
            matches = sum(1 for r in self.hand if r == self.current_rank)
            for i, (qty, honest) in enumerate(
                [(q, h) for q in range(1, 5) for h in range(0, q + 1)]
            ):
                if qty <= len(self.hand) and honest <= matches:
                    mask[i] = 1
        return mask


def _get_greedy_action(agent: EvalAgent, obs: np.ndarray, mask: np.ndarray) -> int:
    """Return argmax action for a policy agent without sampling."""
    mask_t = torch.tensor(mask, dtype=torch.bool)
    if isinstance(agent, EvalLSTMAgent):
        enc    = agent._inner.encoder
        pol    = agent._inner.policy
        dev    = agent._inner.device
        obs_t  = torch.tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0)
        with torch.no_grad():
            latent, _ = enc(obs_t, None)
            logits, _ = pol(torch.cat([obs_t, latent], dim=-1))
    elif isinstance(agent, EvalPolicyAgent):
        dev   = agent._inner.device
        obs_t = torch.tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0)
        with torch.no_grad():
            logits, _ = agent._inner.policy(obs_t)
    else:
        return -1
    logits = logits.squeeze(0).masked_fill(~mask_t.to(logits.device), float("-inf"))
    return int(logits.argmax().item())


def run_scenario(
    agent:     EvalAgent,
    scenario:  ScenarioSpec,
    n_samples: int = 1000,
) -> None:
    """
    Query an agent's action distribution over a hand scenario.

    Samples n_samples actions stochastically and reports the empirical
    distribution, plus the greedy (argmax) action for policy agents.
    The LSTM hidden state is reset between samples (scenario is stateless).
    """
    obs  = scenario.to_obs()
    mask = scenario.action_mask()

    print(f"\n  ┌{'─'*60}┐")
    print(f"  │  Scenario: {agent.name:<48}│")
    print(f"  ├{'─'*60}┤")
    print(f"  │  Hand:          {', '.join(scenario.hand):<43}│")
    print(f"  │  Current rank:  {scenario.current_rank:<43}│")
    print(f"  │  Phase:         {scenario.phase:<43}│")
    print(f"  │  Pile size:     {scenario.pile_size:<43}│")
    print(f"  │  Claim qty:     {(scenario.claim_qty if scenario.claim_qty else '—')!s:<43}│")
    valid_labels = [_ACTION_LABELS[i] for i in range(NUM_ACTIONS) if mask[i]]
    print(f"  │  Valid actions: {', '.join(valid_labels):<43}│")
    print(f"  ├{'─'*60}┤")

    counts: Dict[int, int] = defaultdict(int)
    for _ in range(n_samples):
        if isinstance(agent, EvalLSTMAgent):
            agent._inner.reset_hidden([0])
            agent._needs_reset = False
        counts[agent.act(obs, mask, seat=0)] += 1

    greedy = _get_greedy_action(agent, obs, mask)

    print(f"  │  Distribution ({n_samples} samples):{' '*30}│")
    print(f"  ├{'─'*60}┤")
    for a in [i for i in range(NUM_ACTIONS) if mask[i]]:
        freq  = counts[a] / n_samples
        bar   = "█" * int(freq * 32) + "░" * (32 - int(freq * 32))
        gmark = " ◄ greedy" if a == greedy else ""
        print(f"  │  {_ACTION_LABELS[a]:<18} {bar} {freq:>5.1%}{gmark:<9}│")

    valid_actions = [i for i in range(NUM_ACTIONS) if mask[i]]
    probs   = np.array([counts[a] / n_samples for a in valid_actions])
    probs   = probs[probs > 0]
    entropy = float(-np.sum(probs * np.log(probs + 1e-9)))
    maxH    = float(np.log(len(valid_actions)))
    spread  = "concentrated" if entropy < maxH * 0.4 else "spread"
    print(f"  ├{'─'*60}┤")
    print(f"  │  Entropy: {entropy:.3f} / {maxH:.3f} max  ({spread}){' '*22}│")
    print(f"  └{'─'*60}┘\n")


# ── Pre-built scenarios ───────────────────────────────────────────────────

SCENARIOS: Dict[str, ScenarioSpec] = {

    # Agent holds 1 Ace; opponent claims 4 Aces → only 3 remain in deck
    "impossible_four_of_a_kind": ScenarioSpec(
        hand=["A","2","3","K"], current_rank="A",
        phase="CHALLENGE", pile_size=8, claim_qty=4,
    ),

    # Agent holds all 4 of the current rank → any claim is complementarily provable
    "agent_holds_all_four": ScenarioSpec(
        hand=["A","A","A","A","2","3"], current_rank="A",
        phase="CHALLENGE", pile_size=4, claim_qty=4,
    ),

    # Declare phase, agent has many matching cards
    "declare_with_honest_cards": ScenarioSpec(
        hand=["A","A","A","5","7"], current_rank="A",
        phase="DECLARE", pile_size=0, claim_qty=0,
    ),

    # Declare phase, agent has no matching cards (pure bluff required)
    "forced_bluff": ScenarioSpec(
        hand=["2","3","K","Q"], current_rank="A",
        phase="DECLARE", pile_size=20, claim_qty=0,
    ),

    # Challenge phase, small pile, low claim — hard to justify challenge
    "small_pile_low_claim": ScenarioSpec(
        hand=["A","5","7","9"], current_rank="A",
        phase="CHALLENGE", pile_size=3, claim_qty=1,
    ),

    # Challenge phase, large pile, maximum claim — very suspicious
    "large_pile_high_claim": ScenarioSpec(
        hand=["A","5","7","9"], current_rank="A",
        phase="CHALLENGE", pile_size=30, claim_qty=4,
    ),
}


# ════════════════════════════════════════════════════════════════════════════
# Extended __main__ (replaces the original stub)
# ════════════════════════════════════════════════════════════════════════════

def _build_full_parser() -> argparse.ArgumentParser:
    p = build_parser()

    diag = p.add_argument_group("LSTM diagnostics")
    diag.add_argument("--analyze-hidden", action="store_true",
                      help="Hidden state activation analysis on LSTM agents")
    diag.add_argument("--ablate-zeroed", action="store_true",
                      help="Zeroed-hx ablation on LSTM agents")
    diag.add_argument("--hidden-games", type=int, default=50,
                      help="Games for hidden state analysis (default: 50)")

    scen = p.add_argument_group("Scenario testing")
    scen.add_argument("--scenario", metavar="NAME", action="append", default=[],
                      help=f"Pre-built scenario name (repeatable). "
                           f"See --list-scenarios for options.")
    scen.add_argument("--scenario-samples", type=int, default=1000,
                      help="Samples per scenario query (default: 1000)")
    scen.add_argument("--list-scenarios", action="store_true",
                      help="Print available scenarios and exit")
    return p

if __name__ == "__main__":
    _parser = _build_full_parser()
    _args   = _parser.parse_args()

    if _args.list_scenarios:
        print("\n  Pre-built scenarios:")
        for _name, _spec in SCENARIOS.items():
            print(f"    {_name:<35} hand={_spec.hand}  rank={_spec.current_rank}"
                  f"  phase={_spec.phase}  claim={_spec.claim_qty}")
        print()
        sys.exit(0)

    _device = _args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    _agents: List[EvalAgent] = []
    _lstm_agents: List[EvalLSTMAgent] = []

    for _path in _args.ppo:
        print(f"  Loading PPO checkpoint: {_path}")
        _agents.append(load_ppo_agent(_path, _device, _args.hidden_dim))

    for _path in _args.lstm:
        print(f"  Loading LSTM checkpoint: {_path}")
        _a = load_lstm_agent(_path, _device, _args.hidden_dim)
        _agents.append(_a)
        _lstm_agents.append(_a)

    _naive_kinds = {"all": ["random", "conservative", "aggressive", "threshold"]}
    for _kind in _args.naive:
        for _k in _naive_kinds.get(_kind, [_kind]):
            _agents.append(make_naive_eval_agent(_k))
            print(f"  Added naive agent: {_k}")

    if _args.human:
        _agents.append(HumanAgent())
        print("  Added human player")

    if not _agents:
        print("\n  No agents specified — running default: all 4 naive agents.\n")
        for _k in ["random", "conservative", "aggressive", "threshold"]:
            _agents.append(make_naive_eval_agent(_k))

    _shuffle = not _args.no_shuffle
    print(f"\n  Agents ({len(_agents)}): {', '.join(a.name for a in _agents)}")
    print(f"  Games per matchup: {_args.games}  |  Device: {_device}  |  "
          f"Seats: {'shuffled' if _shuffle else 'fixed'}\n")

    # ── Tournament ───────────────────────────────────────────────────────
    _all_stats = run_round_robin(
        agents=_agents, n_games=_args.games,
        max_iter=_args.max_iter, verbose=_args.verbose, shuffle=_shuffle,
    )
    print_all_stats(_all_stats)
    if _args.csv:
        export_csv(_all_stats, _args.csv)

    # ── LSTM diagnostics ─────────────────────────────────────────────────
    _naive_opp = EvalNaiveAgent(RandomAgent(), "Random")

    if _args.analyze_hidden:
        for _la in _lstm_agents:
            print(f"\n  Analyzing hidden state: {_la.name}")
            analyze_hidden_state(_la, n_games=_args.hidden_games,
                                 max_iter=_args.max_iter, opponent=_naive_opp)

    if _args.ablate_zeroed:
        for _la in _lstm_agents:
            ablation_zeroed_hidden(_la, opponents=[_naive_opp],
                                   n_games=_args.games, max_iter=_args.max_iter)

    # ── Scenario testing ─────────────────────────────────────────────────
    if _args.scenario:
        _policy_agents = [a for a in _agents
                          if isinstance(a, (EvalPolicyAgent, EvalLSTMAgent))]
        for _sname in _args.scenario:
            if _sname not in SCENARIOS:
                print(f"  [warn] Unknown scenario '{_sname}'. "
                      f"Use --list-scenarios.")
                continue
            print(f"\n  ══ Scenario: {_sname} ══")
            for _agent in _policy_agents:
                run_scenario(_agent, SCENARIOS[_sname],
                             n_samples=_args.scenario_samples)
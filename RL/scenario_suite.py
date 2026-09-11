"""
scenario_suite.py — Structured behavioral test suite for BSEnv agents.

Each sweep fixes all variables except one and measures how the agent's
action distribution changes as that variable moves across its range.
This isolates the agent's sensitivity to each input dimension and reveals
whether it is responding strategically or ignoring / misusing information.

Usage
─────
# Test a PPO checkpoint
python scenario_suite.py --ppo checkpoints/policy_0100000.pt

# Test both PPO and LSTM side by side
python scenario_suite.py \
    --ppo  checkpoints/policy_0100000.pt \
    --lstm checkpoints_lstm/policy_0100000.pt

# Run only specific sweep groups
python scenario_suite.py --ppo ... --groups claim_qty hand_size

# Export to CSV
python scenario_suite.py --ppo ... --csv results/scenarios.csv

# Adjust samples per scenario (default 2000)
python scenario_suite.py --ppo ... --samples 5000

Sweep groups
────────────
  claim_qty         : how challenge rate changes with claim size (fixed hand)
  claim_impossibility: challenge rate when claim + agent holdings exceed 4
  hand_size         : challenge rate as agent hand grows (fixed rank counts)
  current_rank_count: challenge rate vs how many of active rank agent holds
  pile_size         : challenge rate vs pile size (memoryless invariance test)
  opponent_hand     : challenge rate vs opponent hand size
  declare_honesty   : declare quantity/honesty choices vs hand composition
  declare_hand_size : declare choices as hand grows
  cross_rank_leakage: challenge rate vs irrelevant rank counts (leakage test)
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── project imports ──────────────────────────────────────────────────────────
from RL.evaluate import (
    EvalAgent, EvalPolicyAgent, EvalLSTMAgent,
    ScenarioSpec, run_scenario,
    load_ppo_agent, load_lstm_agent,
    RANK_NAMES, NUM_ACTIONS,
    _ACTION_LABELS, _get_greedy_action,
)

import torch


# ────────────────────────────────────────────────────────────────────────────
# Core sweep infrastructure
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class SweepResult:
    """Result of one scenario query within a sweep."""
    label:      str                  # human-readable value of the swept variable
    challenge:  float                # P(challenge)
    pass_:      float                # P(pass)
    dominant:   str                  # highest-probability action label
    entropy:    float                # entropy over valid actions
    raw_counts: Dict[int, int]       # action → count


@dataclass
class Sweep:
    """A named sweep varying one variable."""
    name:        str
    description: str
    phase:       str                 # "CHALLENGE" or "DECLARE"
    scenarios:   List[Tuple[str, ScenarioSpec]]   # (label, spec)
    expected:    str                 # what strategic reasoning predicts
    flag:        Optional[str] = None  # anomaly flag if behavior is surprising


def _sample(
    agent:    EvalAgent,
    spec:     ScenarioSpec,
    n:        int,
) -> Dict[int, int]:
    """Sample n actions from agent on spec, return count dict."""
    obs  = spec.to_obs()
    mask = spec.action_mask()
    counts: Dict[int, int] = {i: 0 for i in range(NUM_ACTIONS) if mask[i]}

    for _ in range(n):
        if isinstance(agent, EvalLSTMAgent):
            agent._inner.reset_hidden([0])
            agent._needs_reset = False
        counts[agent.act(obs, mask, seat=0)] += 1

    return counts


def _sweep_result(label: str, counts: Dict[int, int], n: int) -> SweepResult:
    total     = sum(counts.values())
    challenge = counts.get(14, 0) / total
    pass_     = counts.get(15, 0) / total
    dominant  = _ACTION_LABELS[max(counts, key=counts.get)]
    probs     = np.array([v / total for v in counts.values()])
    probs     = probs[probs > 0]
    entropy   = float(-np.sum(probs * np.log(probs + 1e-9)))
    return SweepResult(label, challenge, pass_, dominant, entropy, counts)


def run_sweep(
    agent:   EvalAgent,
    sweep:   Sweep,
    n:       int = 2000,
) -> List[SweepResult]:
    results = []
    for label, spec in sweep.scenarios:
        counts = _sample(agent, spec, n)
        results.append(_sweep_result(label, counts, n))
    return results


# ────────────────────────────────────────────────────────────────────────────
# Display
# ────────────────────────────────────────────────────────────────────────────

def print_sweep(
    sweep:   Sweep,
    results: List[SweepResult],
    agent_name: str,
) -> None:
    width = 72
    print(f"\n  ╔{'═'*width}╗")
    print(f"  ║  {agent_name:<{width-2}}║")
    print(f"  ║  Sweep: {sweep.name:<{width-9}}║")
    print(f"  ║  {sweep.description:<{width-2}}║")
    print(f"  ╠{'═'*width}╣")
    print(f"  ║  Expected: {sweep.expected:<{width-11}}║")
    print(f"  ╠{'═'*width}╣")

    if sweep.phase == "CHALLENGE":
        # Challenge phase: show challenge vs pass rate as primary signal
        print(f"  ║  {'Variable':<20}  {'Challenge':>9}  {'Pass':>6}  "
              f"{'Bar (challenge rate)':^28}  {'H':>4}  ║")
        print(f"  ╠{'═'*width}╣")
        for r in results:
            bar_len = int(r.challenge * 28)
            bar     = "█" * bar_len + "░" * (28 - bar_len)
            flag    = " ◄" if _is_anomalous_challenge(r, sweep) else "  "
            print(f"  ║  {r.label:<20}  {r.challenge:>8.1%}  {r.pass_:>6.1%}  "
                  f"{bar}  {r.entropy:>4.2f}{flag}║")
    else:
        # Declare phase: show dominant action and entropy
        print(f"  ║  {'Variable':<20}  {'Dominant action':<28}  "
              f"{'Prob':>6}  {'H':>4}  ║")
        print(f"  ╠{'═'*width}╣")
        for r in results:
            dom_action = max(r.raw_counts, key=r.raw_counts.get)
            dom_prob   = r.raw_counts[dom_action] / sum(r.raw_counts.values())
            flag       = " ◄" if r.entropy < 0.15 else "  "
            print(f"  ║  {r.label:<20}  {r.dominant:<28}  "
                  f"{dom_prob:>6.1%}  {r.entropy:>4.2f}{flag}║")

    print(f"  ╚{'═'*width}╝")


def _is_anomalous_challenge(r: SweepResult, sweep: Sweep) -> bool:
    """Flag results that seem strategically inconsistent."""
    if "impossible" in sweep.name.lower() and r.challenge < 0.60:
        return True
    if "leakage" in sweep.name.lower() and r.challenge < 0.35:
        return True
    return False


def print_agent_header(name: str) -> None:
    print(f"\n\n  {'═'*72}")
    print(f"  Agent: {name}")
    print(f"  {'═'*72}")


# ────────────────────────────────────────────────────────────────────────────
# Sweep definitions
# ────────────────────────────────────────────────────────────────────────────

def _build_sweeps() -> Dict[str, Sweep]:
    sweeps = {}

    # ── 1. Claim quantity sweep ───────────────────────────────────────────
    # Fixed: agent holds 1 Ace (current rank), pile=10, hand=8
    # Varying: claim quantity 1→4
    # Expected: challenge rate increases monotonically with claim quantity
    sweeps["claim_qty"] = Sweep(
        name        = "claim_qty",
        description = "Claim quantity 1→4, agent holds 1 of active rank, pile=10",
        phase       = "CHALLENGE",
        expected    = "Challenge rate increases monotonically with claim quantity",
        scenarios   = [
            (f"claim={q}", ScenarioSpec(
                hand=["A","5","7","9","2","K","Q"], current_rank="A",
                phase="CHALLENGE", pile_size=10, claim_qty=q,
                hand_sizes=[7, 12, 12],
            ))
            for q in [1, 2, 3, 4]
        ],
    )

    # ── 2. Claim impossibility: claim + holdings > 4 ─────────────────────
    # The key strategic test: when agent holds N of active rank,
    # any claim > (4 - N) is mathematically impossible
    # Expected: strong challenge whenever claim makes total > 4
    sweeps["claim_impossibility"] = Sweep(
        name        = "claim_impossibility",
        description = "Agent holdings + claim quantity exceeds 4 (impossible claims)",
        phase       = "CHALLENGE",
        expected    = "Near-certain challenge whenever holdings + claim > 4",
        scenarios   = [
            # 1 in hand + claim 4 = impossible (only 3 remain)
            ("hold=1 claim=4 [impossible]", ScenarioSpec(
                hand=["A","5","7","9"], current_rank="A",
                phase="CHALLENGE", pile_size=8, claim_qty=4,
                hand_sizes=[4, 15, 15],
            )),
            # 2 in hand + claim 3 = impossible
            ("hold=2 claim=3 [impossible]", ScenarioSpec(
                hand=["A","A","5","7"], current_rank="A",
                phase="CHALLENGE", pile_size=8, claim_qty=3,
                hand_sizes=[4, 15, 15],
            )),
            # 2 in hand + claim 2 = possible (exactly uses remaining)
            ("hold=2 claim=2 [possible]", ScenarioSpec(
                hand=["A","A","5","7"], current_rank="A",
                phase="CHALLENGE", pile_size=8, claim_qty=2,
                hand_sizes=[4, 15, 15],
            )),
            # 3 in hand + claim 2 = impossible
            ("hold=3 claim=2 [impossible]", ScenarioSpec(
                hand=["A","A","A","5"], current_rank="A",
                phase="CHALLENGE", pile_size=8, claim_qty=2,
                hand_sizes=[4, 15, 15],
            )),
            # 3 in hand + claim 1 = possible
            ("hold=3 claim=1 [possible]", ScenarioSpec(
                hand=["A","A","A","5"], current_rank="A",
                phase="CHALLENGE", pile_size=8, claim_qty=1,
                hand_sizes=[4, 15, 15],
            )),
            # 4 in hand + claim 1 = impossible (none remain)
            ("hold=4 claim=1 [impossible]", ScenarioSpec(
                hand=["A","A","A","A","5","7"], current_rank="A",
                phase="CHALLENGE", pile_size=8, claim_qty=1,
                hand_sizes=[6, 15, 15],
            )),
            # 4 in hand + claim 4 = impossible
            ("hold=4 claim=4 [impossible]", ScenarioSpec(
                hand=["A","A","A","A","5","7"], current_rank="A",
                phase="CHALLENGE", pile_size=8, claim_qty=4,
                hand_sizes=[6, 15, 15],
            )),
        ],
    )

    # ── 3. Agent hand size sweep ──────────────────────────────────────────
    # Fixed: holds 1 Ace, claim=3, pile=10, rank counts constant
    # Varying: total hand size via adding irrelevant cards
    # Expected: strategically invariant (irrelevant rank counts shouldn't matter)
    # Tests: cross-rank leakage hypothesis
    sweeps["hand_size"] = Sweep(
        name        = "hand_size",
        description = "Agent hand size 4→17, current-rank count fixed at 1, claim=3",
        phase       = "CHALLENGE",
        expected    = "Challenge rate invariant to hand size (irrelevant ranks shouldn't matter)",
        scenarios   = [
            (f"hand={sz}", ScenarioSpec(
                hand=["A"] + ["5","7","9","2","K","Q","J","8","6","3","4","10","2","3","K","Q"][:sz-1],
                current_rank="A", phase="CHALLENGE",
                pile_size=10, claim_qty=3,
                hand_sizes=[sz, 17-sz//2, 17-sz//2],
            ))
            for sz in [4, 6, 8, 10, 12, 14, 17]
        ],
    )

    # ── 4. Current rank count sweep ───────────────────────────────────────
    # Fixed: total hand=8, claim=3, pile=10
    # Varying: how many of active rank agent holds (0→4)
    # Expected: challenge rate increases sharply as holdings approach claim
    sweeps["current_rank_count"] = Sweep(
        name        = "current_rank_count",
        description = "Agent holdings of active rank 0→4, claim=3, total hand fixed at 8",
        phase       = "CHALLENGE",
        expected    = "Challenge rate increases as holdings approach and exceed claim",
        scenarios   = [
            (f"hold={n} of rank", ScenarioSpec(
                hand=["A"]*n + ["5","7","9","2","K","Q","J"][:8-n],
                current_rank="A", phase="CHALLENGE",
                pile_size=10, claim_qty=3,
                hand_sizes=[8, 12, 12],
            ))
            for n in [0, 1, 2, 3, 4]
        ],
    )

    # ── 5. Pile size sweep ────────────────────────────────────────────────
    # Fixed: hold 1 Ace, claim=2, hand=6
    # Varying: pile size 0→40
    # Expected: memoryless PPO should show weak/no sensitivity
    # LSTM should show sensitivity (large pile = more history = more info)
    sweeps["pile_size"] = Sweep(
        name        = "pile_size",
        description = "Pile size 0→40, agent holds 1 of active rank, claim=2",
        phase       = "CHALLENGE",
        expected    = "PPO: weak/flat response. LSTM: should respond to pile growth",
        scenarios   = [
            (f"pile={p}", ScenarioSpec(
                hand=["A","5","7","9","2","K"], current_rank="A",
                phase="CHALLENGE", pile_size=p, claim_qty=2,
                hand_sizes=[6, 12, 12],
            ))
            for p in [0, 4, 8, 12, 16, 20, 30, 40]
        ],
    )

    # ── 6. Opponent hand size sweep ───────────────────────────────────────
    # Fixed: hold 1 Ace, claim=2, pile=10, own hand=6
    # Varying: next opponent hand size 2→20
    # Expected: larger opponent hand = more desperate = more likely bluffing
    # So challenge rate should increase as opponent hand grows
    sweeps["opponent_hand"] = Sweep(
        name        = "opponent_hand",
        description = "Next opponent hand size 2→20, own hand fixed, claim=2",
        phase       = "CHALLENGE",
        expected    = "Challenge rate increases with opponent hand size (desperation bluffing)",
        scenarios   = [
            (f"opp_hand={opp}", ScenarioSpec(
                hand=["A","5","7","9","2","K"], current_rank="A",
                phase="CHALLENGE", pile_size=10, claim_qty=2,
                hand_sizes=[6, opp, 12],
            ))
            for opp in [2, 4, 6, 8, 10, 12, 15, 20]
        ],
    )

    # ── 7. Cross-rank leakage test ────────────────────────────────────────
    # Fixed: hold exactly 1 Ace, claim=3, pile=10, same total hand size
    # Varying: which irrelevant ranks fill the rest of the hand
    # Expected: challenge rate invariant (irrelevant ranks shouldn't matter)
    sweeps["cross_rank_leakage"] = Sweep(
        name        = "cross_rank_leakage",
        description = "Irrelevant rank composition varies, active-rank count fixed at 1, claim=3",
        phase       = "CHALLENGE",
        expected    = "Challenge rate invariant to which irrelevant ranks are held",
        scenarios   = [
            ("filler=low  [2,3,4,5,6,7]", ScenarioSpec(
                hand=["A","2","3","4","5","6","7"], current_rank="A",
                phase="CHALLENGE", pile_size=10, claim_qty=3,
                hand_sizes=[7, 12, 12],
            )),
            ("filler=mid  [5,6,7,8,9,10]", ScenarioSpec(
                hand=["A","5","6","7","8","9","10"], current_rank="A",
                phase="CHALLENGE", pile_size=10, claim_qty=3,
                hand_sizes=[7, 12, 12],
            )),
            ("filler=high [9,10,J,Q,K,K]", ScenarioSpec(
                hand=["A","9","10","J","Q","K","K"], current_rank="A",
                phase="CHALLENGE", pile_size=10, claim_qty=3,
                hand_sizes=[7, 12, 12],
            )),
            ("filler=near [2,2,K,K,Q,Q]", ScenarioSpec(
                hand=["A","2","2","K","K","Q","Q"], current_rank="A",
                phase="CHALLENGE", pile_size=10, claim_qty=3,
                hand_sizes=[7, 12, 12],
            )),
            ("filler=clustered [2,2,2,3,3,3]", ScenarioSpec(
                hand=["A","2","2","2","3","3","3"], current_rank="A",
                phase="CHALLENGE", pile_size=10, claim_qty=3,
                hand_sizes=[7, 12, 12],
            )),
            ("filler=spread [2,4,6,8,10,Q]", ScenarioSpec(
                hand=["A","2","4","6","8","10","Q"], current_rank="A",
                phase="CHALLENGE", pile_size=10, claim_qty=3,
                hand_sizes=[7, 12, 12],
            )),
        ],
    )

    # ── 8. Declare: honest cards available ───────────────────────────────
    # Fixed: pile=0, hand=8, rank=Ace
    # Varying: how many Aces agent holds (0→4)
    # Expected: when holding more of active rank, agent should declare
    # more honestly and in larger quantities
    sweeps["declare_honesty"] = Sweep(
        name        = "declare_honesty",
        description = "Aces held 0→4 in declare phase, total hand=8",
        phase       = "DECLARE",
        expected    = "More Aces → larger honest declarations, less pure bluffing",
        scenarios   = [
            (f"hold={n} Aces", ScenarioSpec(
                hand=["A"]*n + ["5","7","9","2","K","Q","J"][:8-n],
                current_rank="A", phase="DECLARE",
                pile_size=0, claim_qty=0,
                hand_sizes=[8, 12, 12],
            ))
            for n in [0, 1, 2, 3, 4]
        ],
    )

    # ── 9. Declare: hand size effect ─────────────────────────────────────
    # Fixed: hold 2 Aces, rank=Ace, pile=5
    # Varying: total hand size 4→17
    # Expected: larger hand → lower urgency, potentially smaller declarations
    # Tests: whether hand size influences declare quantity choice
    sweeps["declare_hand_size"] = Sweep(
        name        = "declare_hand_size",
        description = "Total hand size 4→17 in declare phase, always hold 2 Aces",
        phase       = "DECLARE",
        expected    = "Larger hand may reduce urgency; smaller hand = more aggressive shedding",
        scenarios   = [
            (f"hand={sz}", ScenarioSpec(
                hand=["A","A"] + ["5","7","9","2","K","Q","J","8","6","3","4","10","K","Q","J"][:sz-2],
                current_rank="A", phase="DECLARE",
                pile_size=5, claim_qty=0,
                hand_sizes=[sz, 12, 12],
            ))
            for sz in [4, 6, 8, 10, 12, 14, 17]
        ],
    )

    # ── 10. Declare: pile size effect ────────────────────────────────────
    # Fixed: hold 1 Ace, hand=6, rank=Ace
    # Varying: pile size 0→30
    # Expected: larger pile = higher risk of challenge = more honest play
    # or alternatively more conservative quantity declarations
    sweeps["declare_pile_size"] = Sweep(
        name        = "declare_pile_size",
        description = "Pile size 0→30 in declare phase, hold 1 Ace, hand=6",
        phase       = "DECLARE",
        expected    = "Larger pile may induce more honest/conservative declarations",
        scenarios   = [
            (f"pile={p}", ScenarioSpec(
                hand=["A","5","7","9","2","K"], current_rank="A",
                phase="DECLARE", pile_size=p, claim_qty=0,
                hand_sizes=[6, 12, 12],
            ))
            for p in [0, 5, 10, 15, 20, 25, 30]
        ],
    )

    # ── 11. Rank cycle position sweep ────────────────────────────────────
    # Fixed: agent always holds the card 3 ranks ahead of current rank
    # (near-future card), claim=2, pile=8, hand=6
    # Varying: which rank is active (cycles through all 13)
    # Expected: consistent behavior since strategic situation is identical
    # Tests: whether one-hot + cyclic encoding treats ranks symmetrically
    sweeps["rank_symmetry"] = Sweep(
        name        = "rank_symmetry",
        description = "Active rank cycles through all 13, agent always holds rank+3 card, claim=2",
        phase       = "CHALLENGE",
        expected    = "Challenge rate symmetric across all active ranks (same relative situation)",
        scenarios   = [
            (f"rank={RANK_NAMES[r]}", ScenarioSpec(
                hand=[RANK_NAMES[(r+3) % 13], "5", "7", "9", "2", "K"],
                current_rank=RANK_NAMES[r],
                phase="CHALLENGE", pile_size=8, claim_qty=2,
                hand_sizes=[6, 12, 12],
            ))
            for r in range(13)
        ],
    )

    # ── 12. Endgame: near-win opponent ───────────────────────────────────
    # Fixed: claim=2, pile=10, own hand=8
    # Varying: opponent hand size from large (safe) to 1 (desperate)
    # Expected: when opponent has 1-2 cards, challenge rate should spike
    # since they're desperate to shed last cards and likely bluffing
    sweeps["endgame_opponent"] = Sweep(
        name        = "endgame_opponent",
        description = "Opponent near winning (small hand), claim=2, own hand=8",
        phase       = "CHALLENGE",
        expected    = "Challenge rate spikes when opponent hand is tiny (desperation)",
        scenarios   = [
            (f"opp_hand={opp}", ScenarioSpec(
                hand=["A","5","7","9","2","K","Q","J"], current_rank="A",
                phase="CHALLENGE", pile_size=10, claim_qty=2,
                hand_sizes=[8, opp, 12],
            ))
            for opp in [20, 15, 10, 7, 5, 3, 2, 1]
        ],
    )

    # ── 13. Own endgame: agent near winning ──────────────────────────────
    # Fixed: claim=2, pile=10, opponent hand=12
    # Varying: own hand from large to tiny
    # Expected: when agent has few cards, higher risk tolerance,
    # more aggressive challenging to end game faster
    sweeps["own_endgame"] = Sweep(
        name        = "own_endgame",
        description = "Agent near winning (own hand shrinks), claim=2, opp hand=12",
        phase       = "CHALLENGE",
        expected    = "Challenge behavior may shift as agent approaches win",
        scenarios   = [
            (f"own_hand={sz}", ScenarioSpec(
                hand=["A"] + ["5","7","9","2","K","Q","J","8","6","3","4","10"][:sz-1],
                current_rank="A", phase="CHALLENGE",
                pile_size=10, claim_qty=2,
                hand_sizes=[sz, 12, 12],
            ))
            for sz in [17, 14, 10, 7, 5, 3, 2, 1]
            if sz >= 1
        ],
    )

    # ── 14. Three-way hand size: all players ─────────────────────────────
    # Tests whether the agent integrates all three hand sizes jointly
    # by varying the prev player's hand while holding others fixed
    sweeps["prev_opponent_hand"] = Sweep(
        name        = "prev_opponent_hand",
        description = "Previous opponent hand size varies, claim=2, own=6, next=12",
        phase       = "CHALLENGE",
        expected    = "Weak or no sensitivity (prev player didn't make the claim)",
        scenarios   = [
            (f"prev_hand={prev}", ScenarioSpec(
                hand=["A","5","7","9","2","K"], current_rank="A",
                phase="CHALLENGE", pile_size=10, claim_qty=2,
                hand_sizes=[6, 12, prev],
            ))
            for prev in [2, 5, 8, 12, 15, 20]
        ],
    )

    return sweeps


# ────────────────────────────────────────────────────────────────────────────
# CSV export
# ────────────────────────────────────────────────────────────────────────────

def export_sweep_csv(
    all_results: Dict[str, Dict[str, List[SweepResult]]],
    sweeps:      Dict[str, Sweep],
    path:        str,
) -> None:
    """
    all_results[agent_name][sweep_name] = List[SweepResult]
    """
    rows = []
    for agent_name, agent_results in all_results.items():
        for sweep_name, results in agent_results.items():
            sweep = sweeps[sweep_name]
            for i, (r, (label, _)) in enumerate(
                zip(results, sweep.scenarios)
            ):
                rows.append({
                    "agent":       agent_name,
                    "sweep":       sweep_name,
                    "phase":       sweep.phase,
                    "label":       r.label,
                    "challenge":   f"{r.challenge:.4f}",
                    "pass":        f"{r.pass_:.4f}",
                    "dominant":    r.dominant,
                    "entropy":     f"{r.entropy:.4f}",
                })

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  → CSV saved to {path}")


# ────────────────────────────────────────────────────────────────────────────
# Summary: sensitivity table across all sweeps
# ────────────────────────────────────────────────────────────────────────────

def print_sensitivity_summary(
    all_results: Dict[str, Dict[str, List[SweepResult]]],
    sweeps:      Dict[str, Sweep],
) -> None:
    """
    For each agent and each sweep, report the range of challenge rates
    (or dominant action entropy for declare sweeps) as a single-line
    sensitivity measure. High range = sensitive to that variable.
    """
    print(f"\n\n  {'═'*72}")
    print(f"  SENSITIVITY SUMMARY")
    print(f"  {'═'*72}")
    print(f"  {'Sweep':<30}  {'Phase':<9}", end="")
    for agent_name in all_results:
        print(f"  {agent_name[:16]:<16}", end="")
    print()
    print(f"  {'─'*72}")

    for sweep_name, sweep in sweeps.items():
        print(f"  {sweep_name:<30}  {sweep.phase:<9}", end="")
        for agent_name, agent_results in all_results.items():
            if sweep_name not in agent_results:
                print(f"  {'—':>16}", end="")
                continue
            results = agent_results[sweep_name]
            if sweep.phase == "CHALLENGE":
                rates = [r.challenge for r in results]
                rng   = max(rates) - min(rates)
                print(f"  Δ={rng:>5.1%} [{min(rates):.0%}→{max(rates):.0%}]", end="")
            else:
                entropies = [r.entropy for r in results]
                rng       = max(entropies) - min(entropies)
                print(f"  ΔH={rng:>5.3f} [{min(entropies):.2f}→{max(entropies):.2f}]", end="")
        print()

    print(f"  {'═'*72}\n")


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="BSEnv structured behavioral scenario suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--ppo",    metavar="PATH", action="append", default=[],
                   help="PPO checkpoint path (repeatable)")
    p.add_argument("--lstm",   metavar="PATH", action="append", default=[],
                   help="LSTM checkpoint path (repeatable)")
    p.add_argument("--hidden-dim", type=int, default=128,
                   help="Policy hidden dim (default: 128)")
    p.add_argument("--device", default=None,
                   help="torch device (default: auto)")
    p.add_argument("--samples", type=int, default=2000,
                   help="Samples per scenario (default: 2000)")
    p.add_argument("--groups", metavar="NAME", nargs="+", default=None,
                   help="Run only these sweep groups (default: all)")
    p.add_argument("--csv", metavar="PATH", default=None,
                   help="Export results to CSV")
    p.add_argument("--list-groups", action="store_true",
                   help="List available sweep groups and exit")
    return p


def main() -> None:
    args   = build_parser().parse_args()
    sweeps = _build_sweeps()

    if args.list_groups:
        print("\n  Available sweep groups:")
        for name, sweep in sweeps.items():
            print(f"    {name:<30}  {sweep.description}")
        print()
        sys.exit(0)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    agents: List[EvalAgent] = []

    for path in args.ppo:
        print(f"  Loading PPO:  {path}")
        agents.append(load_ppo_agent(path, device, args.hidden_dim))

    for path in args.lstm:
        print(f"  Loading LSTM: {path}")
        agents.append(load_lstm_agent(path, device, args.hidden_dim))

    if not agents:
        print("[error] No agents specified. Use --ppo or --lstm.")
        sys.exit(1)

    active_sweeps = {
        k: v for k, v in sweeps.items()
        if args.groups is None or k in args.groups
    }

    print(f"\n  Agents:  {', '.join(a.name for a in agents)}")
    print(f"  Sweeps:  {', '.join(active_sweeps)}")
    print(f"  Samples: {args.samples} per scenario\n")

    all_results: Dict[str, Dict[str, List[SweepResult]]] = {
        a.name: {} for a in agents
    }

    for sweep_name, sweep in active_sweeps.items():
        for agent in agents:
            results = run_sweep(agent, sweep, n=args.samples)
            all_results[agent.name][sweep_name] = results
            print_sweep(sweep, results, agent.name)

    print_sensitivity_summary(all_results, active_sweeps)

    if args.csv:
        export_sweep_csv(all_results, active_sweeps, args.csv)


if __name__ == "__main__":
    main()
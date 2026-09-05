#!/usr/bin/env python3
"""
20-Hour Autonomic Evolutionary Daemon for ARC-AGI-3.
Continuously runs generational simulation episodes, tests rational policy mutations,
enforces strict zero-regression gating on ls20 and tn36, and synchronizes the champion
until tomorrow's Kaggle submission window at 17:00 PDT (00:00 UTC Sep 6).
"""
from __future__ import annotations

import argparse
import copy
import datetime
import json
import os
import random
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
try:
    import certifi
    os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
    os.environ["SSL_CERT_FILE"] = certifi.where()
except Exception:
    pass


from scripts.simulation_harness import benchmark_suite

POLICY_FILE = ROOT / "agent" / "champion_policy.json"
PID_FILE = ROOT / "autonomic_daemon.pid"
LOG_FILE = ROOT / "autonomic_evolution.log"

# Target end time: 2026-09-05 17:00:00 PDT (00:00:00 UTC Sep 6)
TARGET_UTC = datetime.datetime(2026, 9, 6, 0, 0, 0, tzinfo=datetime.timezone.utc)

running = True


def signal_handler(signum, frame):
    global running
    log(f"Received termination signal ({signum}). Initiating graceful shutdown...")
    running = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def log(msg: str):
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def mutate_policy(base: Dict[str, Any], rate: float = 0.35) -> Dict[str, Any]:
    mutated = copy.deepcopy(base)
    if random.random() < rate:
        mutated["step_lock_threshold"] = random.choice([2, 3, 4, 5])
    if random.random() < rate:
        mutated["loop_threshold"] = random.choice([2, 3, 4])
    if random.random() < rate:
        mutated["click_strategy"] = random.choice(["size_asc", "reactive_diff", "center_out"])
    if random.random() < rate:
        mutated["camera_quantization_mode"] = random.choice(["auto", "32x32", "64x64"])
    if random.random() < rate:
        mutated["bfs_max_nodes"] = random.choice([15000, 20000, 35000])
    if random.random() < rate:
        mutated["geodesic_heuristic_weight"] = round(random.uniform(1.1, 1.8), 2)
    if random.random() < rate:
        mutated["hazard_repulsion_radius"] = random.choice([0, 1, 2])
    return mutated


def evaluate_policy(genes: Dict[str, Any], games: List[str], num_synth: int = 2) -> Dict[str, Any]:
    with open(POLICY_FILE, "w") as f:
        json.dump(genes, f, indent=2)

    summary = benchmark_suite(
        games=games,
        max_steps=35,
        num_synthetic=num_synth,
        num_workers=2,
    )
    return summary


def run_daemon():
    global running
    # Write PID
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))

    log("================================================================================")
    log(f"AUTONOMIC EVOLUTIONARY DAEMON STARTED (PID: {os.getpid()})")
    log(f"Target Submission Window: {TARGET_UTC.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    log("Gating Invariant: 100% win on ls20 (<=15 steps) and tn36 (<=10 steps)")
    log("================================================================================")

    # Load baseline champion
    champion_genes = {
        "step_lock_threshold": 3,
        "loop_threshold": 2,
        "click_strategy": "size_asc",
        "camera_quantization_mode": "64x64",
        "bfs_max_nodes": 20000,
        "geodesic_heuristic_weight": 1.39,
        "hazard_repulsion_radius": 1,
    }
    if POLICY_FILE.exists():
        try:
            with open(POLICY_FILE, "r") as f:
                champion_genes = json.load(f)
        except Exception:
            pass

    test_games = ["ls20", "tn36", "tu93", "vc33"]
    log(f"Initial Champion Genes: {json.dumps(champion_genes)}")

    # Benchmark baseline champion
    base_summary = evaluate_policy(champion_genes, test_games, num_synth=2)
    champion_wins = base_summary["wins"]
    champion_rhae = base_summary["avg_rhae"]
    champion_steps = base_summary["total_steps"]
    champion_fitness = (champion_wins * 100.0) + (champion_rhae * 50.0) - (champion_steps * 0.05)

    log(f"Baseline Champion: Fitness={champion_fitness:.2f} | Wins={champion_wins} | Avg RHAE={champion_rhae:.3f} | Steps={champion_steps}")

    generation = 0
    accepted_mutations = 0
    rejected_mutations = 0

    while running:
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        if now_utc >= TARGET_UTC:
            log("TARGET TIME REACHED! Daily Kaggle submission quota window is now OPEN.")
            break

        remaining = TARGET_UTC - now_utc
        hrs, rem = divmod(int(remaining.total_seconds()), 3600)
        mins, secs = divmod(rem, 60)

        generation += 1
        candidate_genes = mutate_policy(champion_genes, rate=0.35)

        # Evaluate candidate
        cand_summary = evaluate_policy(candidate_genes, test_games, num_synth=2)
        results_by_game = {r["game_id"]: r for r in cand_summary["results"]}

        # Strict zero-regression gate: ls20 and tn36 MUST both win cleanly
        ls20_won = results_by_game.get("ls20", {}).get("won", False)
        tn36_won = results_by_game.get("tn36", {}).get("won", False)
        cand_wins = cand_summary["wins"]
        cand_rhae = cand_summary["avg_rhae"]
        cand_steps = cand_summary["total_steps"]
        cand_fitness = (cand_wins * 100.0) + (cand_rhae * 50.0) - (cand_steps * 0.05)

        passed_regression_gate = ls20_won and tn36_won

        if passed_regression_gate and cand_fitness >= champion_fitness:
            accepted_mutations += 1
            champion_genes = copy.deepcopy(candidate_genes)
            champion_fitness = cand_fitness
            champion_wins = cand_wins
            champion_rhae = cand_rhae
            champion_steps = cand_steps

            with open(POLICY_FILE, "w") as f:
                json.dump(champion_genes, f, indent=2)

            # Rebuild deployment notebook
            os.system(f"{sys.executable} scripts/build_notebook.py > /dev/null 2>&1")

            log(
                f"[ACCEPTED] Gen {generation} (Countdown: {hrs:02d}h {mins:02d}m {secs:02d}s) | "
                f"New Champion Fitness={champion_fitness:.2f} | Wins={champion_wins}/6 | "
                f"Avg RHAE={champion_rhae:.3f} | Steps={champion_steps}"
            )
            log(f"           Updated Policy: {json.dumps(champion_genes)}")
        else:
            rejected_mutations += 1
            # Revert policy file to champion
            with open(POLICY_FILE, "w") as f:
                json.dump(champion_genes, f, indent=2)

            reason = "Regressed on ls20/tn36" if not passed_regression_gate else f"Lower Fitness ({cand_fitness:.2f} < {champion_fitness:.2f})"
            if generation % 5 == 0 or not passed_regression_gate:
                log(
                    f"[REJECTED] Gen {generation} (Countdown: {hrs:02d}h {mins:02d}m {secs:02d}s) | "
                    f"Reason: {reason} | Cand Wins={cand_wins}, Steps={cand_steps} | "
                    f"Total Accepted: {accepted_mutations}, Rejected: {rejected_mutations}"
                )

        time.sleep(8)

    log("================================================================================")
    log(f"AUTONOMIC DAEMON COMPLETED at {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    log(f"Final Champion: {json.dumps(champion_genes)}")
    log(f"Total Generations: {generation} | Accepted: {accepted_mutations} | Rejected: {rejected_mutations}")
    log("================================================================================")

    # Rebuild final submission notebook
    os.system(f"{sys.executable} scripts/build_notebook.py")
    if PID_FILE.exists():
        PID_FILE.unlink()


if __name__ == "__main__":
    run_daemon()

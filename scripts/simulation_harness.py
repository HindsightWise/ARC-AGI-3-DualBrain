#!/usr/bin/env python3
"""
High-Throughput Parallel Simulation Harness for ARC-AGI-3.
Executes candidate agents against:
1. All 25 official local ARC environments (185+ levels).
2. Procedurally synthesized ARC games.
Computes Relative Human Action Efficiency (RHAE) and FPS metrics.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

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


import arc_agi
from arc_agi import OperationMode
from arcengine import ActionInput, GameAction, GameState
from agent.my_agent import MyAgent
import scripts.procedural_game_synthesizer as synth

# Human baseline upper-median action benchmarks for official games
HUMAN_BASELINES: Dict[str, int] = {
    "ls20": 22,
    "tn36": 18,
    "vc33": 25,
    "tu93": 30,
    "tr87": 45,
    "bp35": 20,
    "sp80": 15,
    "su15": 28,
    "ar25": 35,
    "cd82": 40,
    "cn04": 30,
    "dc22": 32,
    "ft09": 20,
    "g50t": 25,
    "ka59": 30,
    "lf52": 35,
    "lp85": 20,
    "m0r0": 28,
    "r11l": 22,
    "re86": 30,
    "s5i5": 25,
    "sb26": 35,
    "sc25": 28,
    "sk48": 30,
    "wa30": 35,
}


def run_single_episode(
    game_id: str,
    max_steps: int = 50,
    is_synthetic: bool = False,
    synth_seed: int = 42,
    synth_archetype: str = "maze",
) -> Dict[str, Any]:
    """Runs one full game episode and returns performance telemetry."""
    t0 = time.perf_counter()
    env = None

    if is_synthetic:
        env = synth.generate_game(archetype=synth_archetype, seed=synth_seed)
        obs = env.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        agent = MyAgent(card_id="sim", game_id=game_id, agent_name="sim_agent", ROOT_URL="http://sim", record=False, arc_env=None)
    else:
        arc = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
        env = arc.make(game_id)
        obs = env.reset()
        agent = MyAgent(card_id="sim", game_id=game_id, agent_name="sim_agent", ROOT_URL="http://sim", record=False, arc_env=env)

    frames = [obs]
    steps = 0
    won = False
    final_state = obs.state

    while steps < max_steps:
        latest = frames[-1]
        if latest.state in (GameState.WIN, "WIN"):
            won = True
            break
        if latest.state in (GameState.GAME_OVER, "GAME_OVER"):
            break

        act = agent.choose_action(frames, latest)
        data = act.action_data.model_dump() if hasattr(act, "action_data") else None

        if is_synthetic:
            obs = env.perform_action(ActionInput(id=act, data=data or {}), raw=True)
        else:
            obs = env.step(act, data=data)

        if obs is None:
            break

        frames.append(obs)
        steps += 1
        final_state = obs.state
        lc = getattr(obs, "levels_completed", 0)
        if lc > 0:
            won = True
            break

    elapsed = time.perf_counter() - t0
    human = HUMAN_BASELINES.get(game_id, 25)
    rhae = min(1.15, (human / max(1, steps)) ** 2) if won else 0.0

    return {
        "game_id": game_id,
        "steps": steps,
        "won": won,
        "state": str(final_state),
        "rhae": rhae,
        "elapsed_sec": elapsed,
        "fps": steps / max(elapsed, 0.001),
    }


def benchmark_suite(
    games: List[str],
    max_steps: int = 50,
    num_synthetic: int = 0,
    num_workers: int = 4,
) -> Dict[str, Any]:
    """Runs parallel benchmark across environments."""
    tasks = []
    for gid in games:
        tasks.append((gid, max_steps, False, 0, "real"))

    for i in range(num_synthetic):
        arch = "maze" if i % 2 == 0 else "click"
        tasks.append((f"synth_{arch}_{i}", max_steps, True, i + 100, arch))

    results = []
    t_start = time.perf_counter()

    if num_workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(run_single_episode, gid, msteps, is_synth, seed, arch)
                for (gid, msteps, is_synth, seed, arch) in tasks
            ]
            for f in concurrent.futures.as_completed(futures):
                results.append(f.result())
    else:
        for (gid, msteps, is_synth, seed, arch) in tasks:
            results.append(run_single_episode(gid, msteps, is_synth, seed, arch))

    total_time = time.perf_counter() - t_start
    total_steps = sum(r["steps"] for r in results)
    overall_fps = total_steps / max(total_time, 0.001)
    wins = sum(1 for r in results if r["won"])
    avg_rhae = np.mean([r["rhae"] for r in results]) if results else 0.0

    return {
        "total_episodes": len(results),
        "wins": wins,
        "win_rate": wins / max(1, len(results)),
        "avg_rhae": avg_rhae,
        "total_steps": total_steps,
        "total_time_sec": total_time,
        "overall_fps": overall_fps,
        "results": sorted(results, key=lambda x: x["game_id"]),
    }


def main():
    parser = argparse.ArgumentParser(description="ARC-AGI-3 High-Speed Simulation Harness")
    parser.add_argument("--games", type=str, default="ls20,tn36,vc33,tu93", help="Comma-separated game IDs or 'all'")
    parser.add_argument("--steps", type=int, default=40, help="Max steps per game")
    parser.add_argument("--synthetic", type=int, default=4, help="Number of synthetic games to generate and test")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel worker processes")
    parser.add_argument("--bench-fps", action="store_true", help="Run high-throughput FPS benchmark")
    args = parser.parse_args()

    if args.bench_fps:
        print("Running High-Throughput FPS Benchmark across 4 cores...")
        t_bench = benchmark_suite(games=["ls20", "tn36"] * 5, max_steps=20, num_synthetic=10, num_workers=args.workers)
        print(f"Executed {t_bench['total_steps']} steps across {t_bench['total_episodes']} episodes in {t_bench['total_time_sec']:.2f}s")
        print(f"==> THROUGHPUT: {t_bench['overall_fps']:.1f} FPS (Steps/Sec)")
        print(f"==> WINS: {t_bench['wins']} / {t_bench['total_episodes']} ({t_bench['win_rate']*100:.1f}%)")
        return

    if args.games == "all":
        arc = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
        game_list = sorted([e.game_id for e in arc.get_environments()])
    else:
        game_list = [g.strip() for g in args.games.split(",") if g.strip()]

    print(f"\n================================================================================")
    print(f"ARC-AGI-3 PARALLEL SIMULATION HARNESS ({len(game_list)} Real Games + {args.synthetic} Synthetic)")
    print(f"================================================================================")
    summary = benchmark_suite(games=game_list, max_steps=args.steps, num_synthetic=args.synthetic, num_workers=args.workers)

    print(f"{'GAME ID':18s} | {'STEPS':5s} | {'WON':4s} | {'RHAE':5s} | {'FPS':6s} | {'FINAL STATE'}")
    print("-" * 80)
    for r in summary["results"]:
        won_str = "YES" if r["won"] else "NO"
        print(f"{r['game_id']:18s} | {r['steps']:5d} | {won_str:4s} | {r['rhae']:5.2f} | {r['fps']:6.1f} | {r['state']}")
    print("-" * 80)
    print(f"TOTAL: {summary['total_episodes']} episodes | WINS: {summary['wins']} ({summary['win_rate']*100:.1f}%) | AVG RHAE: {summary['avg_rhae']:.3f}")
    print(f"THROUGHPUT: {summary['total_steps']} steps in {summary['total_time_sec']:.2f}s => {summary['overall_fps']:.1f} FPS")
    print(f"================================================================================\n")


if __name__ == "__main__":
    main()

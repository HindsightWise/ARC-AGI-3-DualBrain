#!/usr/bin/env python3
"""
Autonomic Evolutionary Policy Optimizer (The Red Queen Engine) for ARC-AGI-3.
Mutates and evolves cognitive heuristics, camera quantization thresholds,
avatar stability locks, and click prioritization across thousands of simulated episodes.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

from scripts.simulation_harness import benchmark_suite

POLICY_FILE = ROOT / "agent" / "champion_policy.json"

DEFAULT_GENOME: Dict[str, Any] = {
    "step_lock_threshold": 3,          # Steps required to permanently lock avatar step size
    "loop_threshold": 3,               # Hash repeats before triggering orthogonal escape
    "click_strategy": "size_asc",       # 'size_asc', 'reactive_diff', 'center_out'
    "camera_quantization_mode": "auto",# 'auto', '32x32', '64x64'
    "bfs_max_nodes": 20000,            # Max search nodes in configuration state space
    "geodesic_heuristic_weight": 1.2,  # Manhattan multiplier in A* path planning
    "hazard_repulsion_radius": 1,      # Guard margin around known lethal tiles
}


class Genome:
    def __init__(self, genes: Optional[Dict[str, Any]] = None):
        self.genes = copy.deepcopy(genes or DEFAULT_GENOME)
        self.fitness: float = 0.0
        self.wins: int = 0
        self.avg_rhae: float = 0.0
        self.total_steps: int = 0

    def mutate(self, mutation_rate: float = 0.3) -> Genome:
        mutated = copy.deepcopy(self.genes)
        if random.random() < mutation_rate:
            mutated["step_lock_threshold"] = random.choice([2, 3, 4, 5])
        if random.random() < mutation_rate:
            mutated["loop_threshold"] = random.choice([2, 3, 4])
        if random.random() < mutation_rate:
            mutated["click_strategy"] = random.choice(["size_asc", "reactive_diff", "center_out"])
        if random.random() < mutation_rate:
            mutated["camera_quantization_mode"] = random.choice(["auto", "32x32", "64x64"])
        if random.random() < mutation_rate:
            mutated["bfs_max_nodes"] = random.choice([10000, 20000, 40000])
        if random.random() < mutation_rate:
            mutated["geodesic_heuristic_weight"] = round(random.uniform(1.0, 1.8), 2)
        if random.random() < mutation_rate:
            mutated["hazard_repulsion_radius"] = random.choice([0, 1, 2])
        return Genome(mutated)


def evaluate_genome(genome: Genome, test_games: List[str], num_synthetic: int = 2) -> float:
    """Evaluates genome fitness using the parallel simulation harness."""
    # Write temporary policy for agent to read
    with open(POLICY_FILE, "w") as f:
        json.dump(genome.genes, f, indent=2)

    summary = benchmark_suite(
        games=test_games,
        max_steps=35,
        num_synthetic=num_synthetic,
        num_workers=2,
    )

    wins = summary["wins"]
    avg_rhae = summary["avg_rhae"]
    steps = summary["total_steps"]

    # Fitness: Heavily rewards wins and high RHAE, penalizes total actions
    fitness = (wins * 100.0) + (avg_rhae * 50.0) - (steps * 0.05)
    genome.fitness = fitness
    genome.wins = wins
    genome.avg_rhae = avg_rhae
    genome.total_steps = steps
    return fitness


def run_evolution(
    generations: int = 5,
    population_size: int = 4,
    test_games: List[str] = ["ls20", "tn36"],
    num_synthetic: int = 2,
) -> Genome:
    """Runs the Red Queen evolutionary loop over population."""
    print(f"\n================================================================================")
    print(f"RED QUEEN EVOLUTIONARY OPTIMIZER (Generations: {generations}, Pop: {population_size})")
    print(f"Test Suite: {test_games} + {num_synthetic} Synthetic")
    print(f"================================================================================")

    # Load existing champion if available
    champion = None
    if POLICY_FILE.exists():
        try:
            with open(POLICY_FILE, "r") as f:
                champion = Genome(json.load(f))
                print("Loaded existing champion policy.")
        except Exception:
            pass

    if champion is None:
        champion = Genome(DEFAULT_GENOME)

    # Initial population
    population = [champion]
    for _ in range(population_size - 1):
        population.append(champion.mutate(mutation_rate=0.5))

    for gen in range(1, generations + 1):
        print(f"\n--- Generation {gen} / {generations} ---")
        for i, ind in enumerate(population):
            fit = evaluate_genome(ind, test_games, num_synthetic)
            print(f"  Candidate {i+1}: Wins={ind.wins}, RHAE={ind.avg_rhae:.3f}, Steps={ind.total_steps}, Fitness={fit:.2f}")

        # Sort by fitness descending
        population.sort(key=lambda x: x.fitness, reverse=True)
        best = population[0]
        print(f"==> Gen {gen} Leader: Fitness={best.fitness:.2f}, Wins={best.wins}, Avg RHAE={best.avg_rhae:.3f}")

        if best.fitness > champion.fitness or gen == 1:
            champion = copy.deepcopy(best)
            with open(POLICY_FILE, "w") as f:
                json.dump(champion.genes, f, indent=2)
            print(f"*** New Champion Policy Saved to {POLICY_FILE} ***")

        # Breed next generation (elitism + mutations)
        next_pop = [champion, champion.mutate(0.2)]
        while len(next_pop) < population_size:
            parent = random.choice(population[:2])
            next_pop.append(parent.mutate(0.4))
        population = next_pop

    print(f"\n================================================================================")
    print(f"EVOLUTION COMPLETE! Top Performing Genome:")
    print(json.dumps(champion.genes, indent=2))
    print(f"Fitness: {champion.fitness:.2f} | Wins: {champion.wins} | Avg RHAE: {champion.avg_rhae:.3f}")
    print(f"================================================================================\n")
    return champion


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARC-AGI-3 Evolutionary Policy Optimizer")
    parser.add_argument("--generations", type=int, default=3, help="Number of evolutionary generations")
    parser.add_argument("--pop", type=int, default=3, help="Population size per generation")
    parser.add_argument("--games", type=str, default="ls20,tn36", help="Test games")
    parser.add_argument("--synthetic", type=int, default=2, help="Synthetic games per evaluation")
    args = parser.parse_args()

    games = [g.strip() for g in args.games.split(",") if g.strip()]
    run_evolution(generations=args.generations, population_size=args.pop, test_games=games, num_synthetic=args.synthetic)

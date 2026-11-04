# SPDX-License-Identifier: MIT-0
"""k_eff ablation on the 8 click-only public games (runs/2026-10-21_keff-v1-*.jsonl).

Same agent, same games and seeds; only the ACTION6 candidate set changes:
segments (connected components, 5 priority groups) < grid8 (64 block centres) < pixels (4096).
Prediction (docs/novelty.md item 3): with N_g, travel and h unchanged, actions scale like k and
per-level score like k^-2. Paired comparisons use levels solved under both modes, which selects
easy levels; unsolved counts are reported next to them.

    .venv/bin/python arc3/scripts/stage3_keff.py
"""
import math
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common import schema  # noqa: E402

MODES = ("segments", "grid8", "pixels")


def main():
    rows = {m: schema.read(ROOT / "runs" / f"2026-10-21_keff-v1-{m}.jsonl") for m in MODES}
    print("mode       levels solved  games with a solved level (any seed)  mean game score  mean k (candidates/state)  mean T per level")
    for m in MODES:
        rs = rows[m]
        games = {(r["game"], r["seed"]): r for r in rs}
        solved = [r for r in rs if r["solved"]]
        print(f"{m:<10} {len(solved):13d}  {len({r['game'] for r in solved}):8d}"
              f"  {st.mean(r['game_score'] for r in games.values()):15.3f}"
              f"  {st.mean(r['k_eff'] for r in rs if r['k_eff']):25.1f}"
              f"  {st.mean(r['tested_pairs'] for r in rs if r['tested_pairs'] is not None):16.1f}")
        per_seed = {}
        for (g, s), r in games.items():
            per_seed.setdefault(s, []).append(r["game_score"])
        print(f"           seeds: {len(per_seed)}, runs: {len(games)}, mean game score by seed: "
              + ", ".join(f"{s}:{st.mean(v):.3f}" for s, v in sorted(per_seed.items())))
    print("\npaired levels (same game, level, seed) solved under both modes")
    key = lambda r: (r["game"], r["level"], r["seed"])  # noqa: E731
    base = {key(r): r for r in rows["segments"]}
    for m in MODES[1:]:
        other = {key(r): r for r in rows[m]}
        both = [k for k in base if k in other and base[k]["solved"] and other[k]["solved"]]
        only_base = sum(1 for k in base if base[k]["solved"] and not (k in other and other[k]["solved"]))
        only_other = sum(1 for k in other if other[k]["solved"] and not (k in base and base[k]["solved"]))
        print(f"  segments vs {m}: both {len(both)}, only segments {only_base}, only {m} {only_other}")
        for k in both:
            a, b = base[k], other[k]
            print(f"    {k[0]} L{k[1]} s{k[2]}: actions {a['actions_a']} -> {b['actions_a']} "
                  f"(x{b['actions_a'] / a['actions_a']:.1f}); k {a['k_eff']:.0f} -> {b['k_eff']:.0f} "
                  f"(x{b['k_eff'] / a['k_eff']:.1f}); T {a['tested_pairs']} -> {b['tested_pairs']}; "
                  f"score {a['level_score']:.3f} -> {b['level_score']:.3f}")
        if both:
            ra = [math.log(other[k]["actions_a"] / base[k]["actions_a"]) for k in both]
            rk = [math.log(other[k]["k_eff"] / base[k]["k_eff"]) for k in both]
            print(f"    geo-mean action ratio x{math.exp(st.mean(ra)):.2f} vs geo-mean k ratio x{math.exp(st.mean(rk)):.2f}")


if __name__ == "__main__":
    main()

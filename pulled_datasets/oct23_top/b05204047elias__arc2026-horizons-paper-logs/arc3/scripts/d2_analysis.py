# SPDX-License-Identifier: MIT-0
"""D2 main experiment: confirmatory tests on held-out games and descriptive tables (docs/d2-plan.md).

Reads runs/2026-10-22_d2-{seg-groups,seg-flat,px-flat,px-groups}.jsonl (one row per reached level).

    .venv/bin/python arc3/scripts/d2_analysis.py
"""
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common import schema  # noqa: E402

CONDS = ("seg-groups", "seg-flat", "px-flat", "px-groups")
DEV = ("lp85", "r11l", "vc33")
HOLDOUT = ("ft09", "lf52", "s5i5", "su15", "tn36")
SEEDS = tuple(range(10))
BOOT = 10_000


def load():
    runs = {}
    for c in CONDS:
        for r in schema.read(ROOT / "runs" / f"2026-10-22_d2-{c}.jsonl"):
            runs.setdefault((c, r["game"], r["seed"]), []).append(r)
    expect = {(c, g, s) for c in CONDS for g in DEV + HOLDOUT for s in SEEDS}
    if set(runs) != expect:
        raise SystemExit(f"sample mismatch: missing {sorted(expect - set(runs))[:10]}, extra {sorted(set(runs) - expect)[:10]}")
    for key, rows in runs.items():
        rows.sort(key=lambda r: r["level"])
        levels = [r["level"] for r in rows]
        if levels != list(range(1, len(levels) + 1)):
            raise SystemExit(f"{key}: levels {levels} are not 1..n")
        if any(r["max_actions"] != 3000 for r in rows) or rows[0]["stop_reason"] not in ("max_actions", "win"):
            raise SystemExit(f"{key}: not a complete 3,000-action run ({rows[0]['stop_reason']})")
    return runs


def solved(rows):
    return sum(r["solved"] for r in rows)


def contrast(runs, games, weights):
    """d[g][s] = sum_c w_c * solved(c, g, s)."""
    return {g: np.array([sum(w * solved(runs[(c, g, s)]) for c, w in weights.items()) for s in SEEDS], float)
            for g in games}


def cluster_boot(d, rng):
    games = list(d)
    est = float(np.mean([d[g].mean() for g in games]))
    reps = np.empty(BOOT)
    for b in range(BOOT):
        gs = rng.choice(len(games), len(games))
        reps[b] = np.mean([d[games[i]][rng.integers(0, len(SEEDS), len(SEEDS))].mean() for i in gs])
    lo, hi = np.quantile(reps, [0.025, 0.975])
    return est, float(lo), float(hi)


CONTRASTS = {
    "H1 ordering on segments (seg-groups - seg-flat)": {"seg-groups": 1, "seg-flat": -1},
    "H2 ordering on pixels (px-groups - px-flat)": {"px-groups": 1, "px-flat": -1},
    "H3 interaction ([seg-groups - seg-flat] - [px-groups - px-flat])":
        {"seg-groups": 1, "seg-flat": -1, "px-groups": -1, "px-flat": 1},
    "E1 compression at fixed ordering (seg-groups - px-groups)": {"seg-groups": 1, "px-groups": -1},
}
DIRECTIONAL = ("H1", "H2")


def main():
    runs = load()
    rng = np.random.default_rng(0)
    print("== confirmatory, held-out games", HOLDOUT, f"(seeds {SEEDS[0]}-{SEEDS[-1]}; levels solved per game run)")
    for name, w in CONTRASTS.items():
        d = contrast(runs, HOLDOUT, w)
        est, lo, hi = cluster_boot(d, rng)
        per_game = {g: round(float(d[g].mean()), 2) for g in HOLDOUT}
        line = f"{name}: mean {est:+.3f} [{lo:+.3f}, {hi:+.3f}]; per game {per_game}"
        if name[:2] in DIRECTIONAL:
            verdict = "SUPPORTED" if lo > 0 else "CONTRADICTED" if hi < 0 else "INCONCLUSIVE"
            same = sum(1 for g in HOLDOUT if d[g].mean() > 0)
            line += f"; games with positive mean {same}/{len(HOLDOUT)} -> {verdict}"
        print(line)

    print("\n== descriptive, all games (columns: levels solved per seed; game score 0-100; click hit rate summed over"
          " tests; first-effect rank = mean over level records of per-state means, found states only;"
          " same-object repeat share summed; conflicts/same-pixel; exception kinds)")
    for g in DEV + HOLDOUT:
        for c in CONDS:
            rs = [runs[(c, g, s)] for s in SEEDS]
            d = [r["diag"] for rows in rs for r in rows]
            tests = sum(x["click_tests"] for x in d)
            ranks = [x["first_effect_rank_mean"] for x in d if x["first_effect_rank_mean"] is not None]
            kinds = {}
            for x in d:
                for k, v in x["error_kinds"].items():
                    kinds[k] = kinds.get(k, 0) + v
            print(f"  {'dev ' if g in DEV else 'hold'} {g} {c:<10} solved {st.mean(solved(x) for x in rs):4.1f}"
                  f"  score {st.mean(x[0]['game_score'] for x in rs):6.2f}"
                  f"  hit {sum(x['click_effective'] for x in d) / max(1, tests):.3f}"
                  f"  rank {st.mean(ranks) if ranks else float('nan'):6.2f}"
                  f"  repeat {sum(x['seg_repeat_tests'] for x in d) / max(1, tests):.3f}"
                  f"  confl {sum(x['conflicts'] for x in d)}/{sum(x['conflicts_same_point'] for x in d)}"
                  f"  errors {kinds}")

    print("\n== common-level analysis as pre-registered: levels REACHED by both conditions (same game, seed)")
    print("   (click hit rate on those levels; actions only where both solved, since an unsolved level's actions"
          " are just the remaining budget)")
    for a, b in (("seg-groups", "seg-flat"), ("px-groups", "px-flat"), ("seg-groups", "px-groups")):
        pairs = higher = 0
        for g in DEV + HOLDOUT:
            for s in SEEDS:
                la = {r["level"]: r for r in runs[(a, g, s)]}
                lb = {r["level"]: r for r in runs[(b, g, s)]}
                for lv in set(la) & set(lb):
                    da, db = la[lv]["diag"], lb[lv]["diag"]
                    if da["click_tests"] and db["click_tests"]:
                        pairs += 1
                        higher += int(da["click_effective"] / da["click_tests"] > db["click_effective"] / db["click_tests"])
        print(f"  {a} vs {b}: {pairs} reached-by-both level records with clicks; {a} has the higher hit rate in {higher}")
    print("\n== deviation (documented in docs/results-d2.md): levels SOLVED by both, action ratio")
    for a, b in (("seg-groups", "seg-flat"), ("px-groups", "px-flat"), ("seg-groups", "px-groups")):
        ratios, both_solved = [], 0
        for g in DEV + HOLDOUT:
            for s in SEEDS:
                la = {r["level"]: r for r in runs[(a, g, s)]}
                lb = {r["level"]: r for r in runs[(b, g, s)]}
                for lv in set(la) & set(lb):
                    if la[lv]["solved"] and lb[lv]["solved"]:
                        both_solved += 1
                        ratios.append(np.log(lb[lv]["actions_a"] / la[lv]["actions_a"]))
        gm = float(np.exp(np.mean(ratios))) if ratios else float("nan")
        print(f"  {a} vs {b}: {both_solved} levels solved by both; geo-mean actions({b})/actions({a}) = {gm:.2f}")


if __name__ == "__main__":
    main()

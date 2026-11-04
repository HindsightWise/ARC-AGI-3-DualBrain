# SPDX-License-Identifier: MIT-0
"""Print every number quoted in docs/results-stage1.md from the committed JSONL logs.

    .venv/bin/python arc3/scripts/stage1_tables.py
"""
import math
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common import fit, schema  # noqa: E402

RUNS = ROOT / "runs"
SWEEPS = {f: RUNS / f"2026-10-21_synth-v1-{f}.jsonl" for f in ("random", "ring")}
RESET = RUNS / "2026-10-21_synth-reset-v2.jsonl"
REPS = 1000


def slope_line(label, x, y, clusters):
    s, _ = fit.ols(x, y)
    lo, hi = fit.bootstrap_slope(x, y, reps=REPS)
    clo, chi = fit.bootstrap_slope(x, y, reps=REPS, clusters=clusters)
    print(f"  {label:<38} n={len(x):4d}  slope={s:+.3f}  row CI [{lo:+.3f}, {hi:+.3f}]  "
          f"graph-cluster CI [{clo:+.3f}, {chi:+.3f}]")


def table_slopes():
    print("Table 1. Synthetic sweep fits (solved levels; bootstrap reps=%d)" % REPS)
    for fam, path in SWEEPS.items():
        rows = schema.read(path)
        for p in sorted({r["p_noop"] for r in rows}):
            sel = [r for r in rows if r["p_noop"] == p and r["solved"]]
            tries = [r["graph_tries"] for r in rows if r["p_noop"] == p]
            print(f"{fam} p_noop={p}: {len(sel)}/{sum(r['p_noop'] == p for r in rows)} solved; graphs drawn "
                  f"per accepted level: mean {np.mean(tries):.2f}, max {max(tries)}, "
                  f"accepted on first draw {sum(t == 1 for t in tries)}/{len(tries)}")
            # rows from the same (n, k, p_noop, seed) can share a graph across D: cluster on it
            cl = np.array([f"{r['n']}-{r['k']}-{r['p_noop']}-{r['seed']}" for r in sel])
            ngk = np.array([r["states_discovered_Ng"] * r["k_eff"] for r in sel], float)
            t = np.array([r["tested_pairs"] for r in sel], float)
            a = np.array([r["actions_a"] for r in sel], float)
            h = np.array([r["h_baseline"] for r in sel], float)
            slope_line("log a ~ log(Ng*k)        (pred +1)", np.log(ngk), np.log(a), cl)
            slope_line("log a ~ log(tested pairs) (pred +1)", np.log(t), np.log(a), cl)
            slope_line("log (h/a)^2 ~ log(Ng*k/h)  (pred -2)", np.log(ngk / h), 2 * np.log(h / a), cl)
            slope_line("log (h/a)^2 ~ log(T/h)     (pred -2)", np.log(t / h), 2 * np.log(h / a), cl)
            slope_line("log (a/T) ~ log(T/h)        (overhead)", np.log(t / h), np.log(a / t), cl)
            c = a / ngk
            print(f"  c = a/(Ng*k): median {np.median(c):.2f}, IQR [{np.quantile(c, .25):.2f}, "
                  f"{np.quantile(c, .75):.2f}]; a/T: median {np.median(a / t):.2f}")


def table_reset():
    print("\nTable 2. RESET ablation (same graphs; k=4 moves; +1 RESET column when present)")
    rows = schema.read(RESET)
    by = {(r["family"], r["reset_mode"], r["n"], r["goal_depth_D"], r["seed"]): r for r in rows}
    for fam in ("random", "ring"):
        keys = [k[2:] for k in by if k[0] == fam and k[1] == "none"]
        for mode in ("none", "action", "known"):
            rs = [by[(fam, mode) + k] for k in keys]
            sol = [r for r in rs if r["solved"]]
            fails = {o: sum(r["outcome"] == o for r in rs) for o in ("budget", "stuck", "exhausted")}
            print(f"  {fam:<6} {mode:<6} solved {len(sol):3d}/{len(rs)}  median actions "
                  f"{st.median(r['actions_a'] for r in sol):7.1f}  "
                  f"mean uncapped (h/a)^2 {st.mean(r['rhae_uncapped'] for r in rs):.4f}  unsolved {fails}")
        for mode in ("known", "action"):
            both = [k for k in keys if by[(fam, "none") + k]["solved"] and by[(fam, mode) + k]["solved"]]
            lr = [math.log(by[(fam, mode) + k]["actions_a"] / by[(fam, "none") + k]["actions_a"]) for k in both]
            print(f"  {fam:<6} {mode:<6} vs none on {len(both)} levels solved by both: geo-mean actions ratio "
                  f"{math.exp(st.mean(lr)):.2f} (cheaper in {sum(x < 0 for x in lr)}, equal in {sum(x == 0 for x in lr)})")


if __name__ == "__main__":
    table_slopes()
    table_reset()

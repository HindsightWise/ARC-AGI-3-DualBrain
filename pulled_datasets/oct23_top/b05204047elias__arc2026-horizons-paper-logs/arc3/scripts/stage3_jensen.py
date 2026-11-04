# SPDX-License-Identifier: MIT-0
"""Jensen experiment tables from runs/2026-10-21_synth-jensen-v1-*.jsonl.

For every graph (family, n, D, graph seed) and exploration policy, 30 explorer seeds give a
distribution of action counts a. Per graph and policy:
  E[a], E[score] with score = min(1.15, (h/a)^2) (unsolved = 0), (h/E[a])^2, sd of log a.
The operational claim (docs/novelty.md, item 5) is that ranking policies by mean actions and
ranking them by expected score can disagree, because the score is convex in a.

    .venv/bin/python arc3/scripts/stage3_jensen.py
"""
import itertools
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common import schema  # noqa: E402

FILES = [ROOT / "runs" / f"2026-10-21_synth-jensen-v1-{f}.jsonl" for f in ("random", "ring")]
POLICIES = ("nf", "nf-static", "random-walk")


def score(r):
    return min(1.15, (r["h_baseline"] / r["actions_a"]) ** 2) if r["solved"] else 0.0


def main():
    for path in FILES:
        rows = schema.read(path)
        by = defaultdict(list)
        for r in rows:
            by[(r["n"], r["goal_depth_D"], r["seed"], r["policy"])].append(r)
        graphs = sorted({k[:3] for k in by})
        print(f"\n== {path.name}: {len(rows)} runs, {len(graphs)} graphs, "
              f"{len(rows) // max(1, len(by))} explorer seeds per (graph, policy); "
              f"unsolved {sum(not r['solved'] for r in rows)}")
        stats = {}
        for key, rs in by.items():
            a = np.array([r["actions_a"] for r in rs if r["solved"]], float)
            h = rs[0]["h_baseline"]
            stats[key] = {
                "Ea": a.mean() if len(a) else math.inf,
                "Escore": st.mean(score(r) for r in rs),
                "score_at_Ea": min(1.15, (h / a.mean()) ** 2) if len(a) else 0.0,
                "sdlog": float(np.log(a).std()) if len(a) > 1 else 0.0,
                "capped": sum(1 for r in rs if r["solved"] and (h / r["actions_a"]) ** 2 > 1.15),
            }
        print("  policy        mean E[a]   mean E[score]  mean (h/E[a])^2  mean sd(log a)  runs capped")
        for pol in POLICIES:
            ss = [stats[g + (pol,)] for g in graphs]
            print(f"  {pol:<12} {st.mean(s['Ea'] for s in ss):10.1f}  {st.mean(s['Escore'] for s in ss):13.4f}  "
                  f"{st.mean(s['score_at_Ea'] for s in ss):15.4f}  {st.mean(s['sdlog'] for s in ss):14.2f}  "
                  f"{sum(s['capped'] for s in ss):11d}")
        print("  pairwise, per graph: the policy with the smaller E[a] has the smaller E[score] (rank reversal);")
        print("  'significant' = both differences have 95% bootstrap CIs excluding 0 (1000 reps, runs resampled")
        print("  independently per policy; unsolved runs count as a = budget for E[a])")
        rng = np.random.default_rng(0)
        for p, q in itertools.combinations(POLICIES, 2):
            rev = sig = tie = 0
            for g in graphs:
                sp, sq = stats[g + (p,)], stats[g + (q,)]
                if sp["Ea"] == sq["Ea"] or sp["Escore"] == sq["Escore"]:
                    tie += 1
                    continue
                if (sp["Ea"] < sq["Ea"]) == (sp["Escore"] > sq["Escore"]):
                    continue
                rev += 1
                ap, aq = (np.array([r["actions_a"] for r in by[g + (x,)]], float) for x in (p, q))
                cp, cq = (np.array([score(r) for r in by[g + (x,)]]) for x in (p, q))
                ia = rng.integers(0, len(ap), (1000, len(ap)))
                ib = rng.integers(0, len(aq), (1000, len(aq)))
                da = ap[ia].mean(1) - aq[ib].mean(1)
                ds = cp[ia].mean(1) - cq[ib].mean(1)
                ok = lambda d: np.quantile(d, 0.025) > 0 or np.quantile(d, 0.975) < 0  # noqa: E731
                sig += bool(ok(da) and ok(ds))
            print(f"    {p} vs {q}: reversals {rev}/{len(graphs)} graphs, significant {sig} (ties {tie})")
        gap = [stats[g + (pol,)]["Escore"] - stats[g + (pol,)]["score_at_Ea"] for g in graphs for pol in POLICIES]
        print(f"  Jensen gap E[score] - score(E[a]) over graph x policy: min {min(gap):+.4f}, "
              f"median {st.median(gap):+.4f}, max {max(gap):+.4f} (negative only through the 1.15 cap "
              f"or unsolved runs)")


if __name__ == "__main__":
    main()

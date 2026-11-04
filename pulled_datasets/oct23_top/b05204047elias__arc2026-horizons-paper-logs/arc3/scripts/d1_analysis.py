# SPDX-License-Identifier: MIT-0
"""D1 tables and the pre-registered go/no-go verdict (docs/d1-plan.md).

Reads runs/2026-10-21_d1-{seg-groups,seg-flat,px-flat,px-groups}.jsonl (one row per reached level).

    .venv/bin/python arc3/scripts/d1_analysis.py
"""
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common import schema  # noqa: E402

CONDS = ("seg-groups", "seg-flat", "px-flat", "px-groups")
MAIN_RUNS = 8 * 4 * 10   # main experiment: 8 click games x 4 conditions x 10 seeds (game runs)
WORKERS = 3
MAX_HOURS = 24.0
GAMES = ("lp85", "r11l", "vc33")
SEEDS = (0, 1, 2, 3, 4)
NEED = 4                 # seeds out of 5 for a mechanism to count as reproduced


def load():
    runs = {}  # (cond, game, seed) -> list of level rows sorted by level
    for c in CONDS:
        for r in schema.read(ROOT / "runs" / f"2026-10-21_d1-{c}.jsonl"):
            runs.setdefault((c, r["game"], r["seed"]), []).append(r)
    for rows in runs.values():
        rows.sort(key=lambda r: r["level"])
    # the pre-registered sample, nothing more or less (Codex review 2026-10-21)
    expect = {(c, g, s) for c in CONDS for g in GAMES for s in SEEDS}
    if set(runs) != expect:
        raise SystemExit(f"sample mismatch: missing {sorted(expect - set(runs))}, extra {sorted(set(runs) - expect)}")
    for key, rows in runs.items():
        levels = [r["level"] for r in rows]
        if levels != list(range(1, len(levels) + 1)):
            raise SystemExit(f"{key}: levels {levels} are not 1..n (duplicate or appended rerun?)")
        if any(r["max_actions"] != 3000 for r in rows) or rows[0]["stop_reason"] not in ("max_actions", "win"):
            raise SystemExit(f"{key}: not a complete 3,000-action run ({rows[0]['stop_reason']})")
    return runs


def solved(rows):
    return sum(r["solved"] for r in rows)


def ordering_sign(a, b):
    """+1 if run a beats run b (more levels; tie -> fewer actions on levels both solved), -1 if worse, 0 tie."""
    if solved(a) != solved(b):
        return 1 if solved(a) > solved(b) else -1
    common = {r["level"] for r in a if r["solved"]} & {r["level"] for r in b if r["solved"]}
    if not common:
        return 0
    ca = sum(r["actions_a"] for r in a if r["level"] in common)
    cb = sum(r["actions_a"] for r in b if r["level"] in common)
    return (ca < cb) - (ca > cb)


def main():
    runs = load()
    games = sorted({g for _, g, _ in runs})
    seeds = sorted({s for _, _, s in runs})
    print(f"{len(runs)} game runs; games {games}; seeds {seeds}\n")

    print("cond        game  lv solved/seed  score   hit-rate  1st-eff rank  seg-repeat  cov-seg  cov-obj"
          "  confl/same-pt  errors  frontier-empty(unsolved)  sec")
    for c in CONDS:
        for g in games:
            rs = [runs[(c, g, s)] for s in seeds if (c, g, s) in runs]
            lv = [r for rows in rs for r in rows]
            d = [r["diag"] for r in lv]
            tests = sum(x["click_tests"] for x in d)

            def mean(key):
                vals = [x[key] for x in d if x[key] is not None]
                return f"{st.mean(vals):.3f}" if vals else "  -  "
            print(f"{c:<11} {g}  {st.mean(solved(x) for x in rs):14.1f}  {st.mean(x[0]['game_score'] for x in rs):6.2f}"
                  f"  {sum(x['click_effective'] for x in d) / max(1, tests):8.3f}  {mean('first_effect_rank_mean'):>12}"
                  f"  {sum(x['seg_repeat_tests'] for x in d) / max(1, tests):10.3f}  {mean('coverage_seg_mean'):>7}"
                  f"  {mean('coverage_obj_mean'):>7}  {sum(x['conflicts'] for x in d):5d}/{sum(x['conflicts_same_point'] for x in d):<3d}  {sum(x['errors'] for x in d):6d}"
                  f"  {sum(1 for r in lv if not r['solved'] and r['diag']['frontier_empty']):24d}"
                  f"  {st.mean(x[0]['game_seconds'] for x in rs):5.0f}")
    print()

    verdict = {}
    for g in games:
        m1 = [ordering_sign(runs[("seg-groups", g, s)], runs[("seg-flat", g, s)]) for s in seeds]
        m1_ok = max(m1.count(1), m1.count(-1)) >= NEED
        m2 = []
        for s in seeds:
            base = runs[("seg-groups", g, s)]
            m2.append(any(solved(runs[(c, g, s)]) > solved(base) for c in ("px-flat", "px-groups")))
        m2_ok = sum(m2) >= NEED
        m3 = [any(r["diag"]["conflicts_same_point"] > 0 or r["diag"]["error_kinds"].get("outcome_flip", 0) > 0
                  for r in runs[("seg-groups", g, s)]) for s in seeds]
        m3_ok = sum(m3) >= NEED
        verdict[g] = (m1_ok, m2_ok, m3_ok)
        print(f"{g}: M1 ordering signs (groups vs flat, +1 = groups better) {m1} -> {'REPRODUCED' if m1_ok else 'no'}")
        print(f"      M2 pixels solve more levels than seg-groups, per seed {m2} -> {'REPRODUCED' if m2_ok else 'no'}")
        print(f"      M3 same-point conflict or outcome_flip per seed {m3} -> {'REPRODUCED' if m3_ok else 'no'}")

    games_ok = [g for g, v in verdict.items() if any(v)]
    secs = [rows[0]["game_seconds"] for rows in runs.values()]
    hours = st.mean(secs) * MAIN_RUNS / WORKERS / 3600
    print(f"\ngames with a reproduced mechanism: {games_ok} ({len(games_ok)}/{len(games)}; need >= 2)")
    print(f"throughput: mean {st.mean(secs):.0f} s per game run -> main experiment {MAIN_RUNS} runs / {WORKERS} workers"
          f" = {hours:.1f} h (need <= {MAX_HOURS:.0f} h)")
    go = len(games_ok) >= 2 and hours <= MAX_HOURS
    print(f"\nVERDICT: {'GO' if go else 'NO-GO'}")


if __name__ == "__main__":
    main()

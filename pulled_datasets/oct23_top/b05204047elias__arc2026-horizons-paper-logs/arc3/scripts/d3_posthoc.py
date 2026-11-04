# SPDX-License-Identifier: MIT-0
"""Exploratory D3 co-solved actions; reads existing logs, never runs games.

Run from the repository root: .venv/bin/python arc3/scripts/d3_posthoc.py
Bootstrap whole games only, keeping each game's co-solved pairs together.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GAMES = ("vc33", "lp85", "r11l", "ft09", "lf52", "s5i5", "su15", "tn36")


def load(path):
    records = {}
    totals = defaultdict(int)
    for line in path.read_text().splitlines():
        row = json.loads(line)
        key = (row["game"], row["seed"], row["level"])
        if key in records:
            raise ValueError(f"Duplicate level record: {key}")
        if row["backend"] != "local_comp" or row["actions_a"] <= 0:
            raise ValueError(f"Invalid backend/actions: {key}")
        records[key] = row
        totals[key[:2]] += row["actions_a"]
    expected = {(g, s) for g in GAMES for s in range(10)}
    if set(totals) != expected or any(n != 3000 for n in totals.values()):
        raise ValueError("Expected 80 complete runs of 3,000 actions")
    return records


def counts(pairs):
    a, b = np.asarray(pairs).T
    return [int(np.sum(b < a)), int(np.sum(b > a)), int(np.sum(b == a))]


def main():
    baseline = load(ROOT / "runs/2026-10-22_d2-seg-groups.jsonl")
    b4 = load(ROOT / "runs/2026-10-22_d3-seg-groups-b4.jsonl")
    levels = defaultdict(list)
    games = defaultdict(list)
    for key in sorted(baseline.keys() & b4.keys()):
        a, b = baseline[key], b4[key]
        if a["solved"] and b["solved"]:
            pair = (a["actions_a"], b["actions_a"])
            levels[key[0], key[2]].append(pair)
            games[key[0]].append(pair)

    print("## 事後分析（非預先登記）\n")
    print("|遊戲|關卡|n|基線 actions 中位數|b4 actions 中位數|b4 較少|較多|相同|")
    print("|---|---|---|---|---|---|---|---|")
    for g in GAMES:
        for game, level in sorted(levels):
            if game == g:
                pairs = levels[game, level]
                a, b = np.median(pairs, axis=0)
                fewer, more, ties = counts(pairs)
                print(f"|{g}|{level}|{len(pairs)}|{a:g}|{b:g}|{fewer}|{more}|{ties}|")
    pooled = [p for pairs in games.values() for p in pairs]
    fewer, more, ties = counts(pooled)
    print(f"\n合計 {len(pooled)} 配對：b4 較少 {fewer}／較多 {more}／相同 {ties}。")

    # Equal game weights; threshold is on all co-solved seed×level pairs.
    eligible = sorted(g for g in games if len(games[g]) >= 5)
    medians = []
    for g in eligible:
        a, b = np.asarray(games[g], dtype=float).T
        medians.append(np.median(np.log(b / a)))
    medians = np.asarray(medians)
    if not len(medians):
        raise ValueError("No games with at least five co-solved pairs")
    rng = np.random.default_rng(0)
    reps = rng.choice(medians, size=(10_000, len(medians))).mean(axis=1)
    estimate = medians.mean()
    lo, hi = np.quantile(reps, [0.025, 0.975])
    print("\n≥5 配對的遊戲（每遊戲 median log(a_b4/a_base)）：" +
          "、".join(f"{g} n={len(games[g])}: {m:+.6f}" for g, m in zip(eligible, medians)) + "。")
    print(f"等權平均 = {estimate:+.6f}；95% 遊戲群集 bootstrap CI [{lo:+.6f}, {hi:+.6f}]。")
    print(f"取 exp = {np.exp(estimate):.4f} [{np.exp(lo):.4f}, {np.exp(hi):.4f}]。")
    print("僅重抽遊戲、保留其全部配對；10,000 次，default_rng(0)，百分位區間。")
    print("此為事後、探索性分析；條件化於雙方成功，不能估計整體解題成本或因果機制。")


if __name__ == "__main__":
    main()

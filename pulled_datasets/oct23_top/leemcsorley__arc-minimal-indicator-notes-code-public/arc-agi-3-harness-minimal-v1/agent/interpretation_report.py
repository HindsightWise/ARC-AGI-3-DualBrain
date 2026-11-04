"""Read replay JSON without executing HTML/code: python -m src.agent.interpretation_report PATH.

Compare off/observe/feedback runs with the same games/model settings. Reports recorded
CPU-job wall time, coverage, failures, checks and game outcomes; no inferred causality.
"""
from collections import Counter
import json
from pathlib import Path
import re
import sys


def summarize(data):
    statuses, checks, changes = Counter(), Counter(), Counter()
    seen, unmatched, steps = set(), 0, 0
    latest = data.get("interpretation") or {}
    setup = latest.get("setup", {})
    for event in data.get("events", []):
        ip = (event.get("v2") or {}).get("interpretation") or {}
        if ip.get("metrics"):
            latest = {"mode": ip.get("mode"), **ip["metrics"]}
        if ip.get("setup"):
            setup = ip["setup"]
        for step in (event.get("v2") or {}).get("steps", []):
            item = step.get("interpretation") or {}
            snap = item.get("after") or {}
            if not snap:
                continue
            key = tuple(snap.get(k) for k in ("game", "level", "attempt", "step", "revision"))
            if key in seen:
                continue
            seen.add(key)
            steps += 1
            statuses[snap.get("status", "missing")] += 1
            report = item.get("report") or {}
            unmatched += report.get("unmatched_cells", 0)
            changes.update(e["kind"] for e in report.get("changes", []))
            checks.update(q["status"] for q in item.get("expectations", []))
    return {"game": data.get("game_id"), "score": data.get("final_score"), "interpretation": latest,
            "setup": setup,
            "observed_steps": steps, "statuses": dict(statuses), "changes": dict(changes),
            "expectations": dict(checks), "unmatched_changed_cells": unmatched,
            "levels_completed": max((int(e.get("levels_completed") or 0) for e in data.get("events", [])), default=0)}


def summarize_directory(path):
    root = Path(path)
    files = [root] if root.is_file() else sorted(root.rglob("*.html"))
    runs, errors = [], []
    for f in files:
        try:
            match = re.search(r'<script id="arc-replay-data"[^>]*>(.*?)</script>', f.read_text(encoding="utf-8"), re.S)
            if not match:
                raise ValueError("missing replay JSON")
            runs.append({"file": str(f), **summarize(json.loads(match[1]))})
        except (OSError, ValueError, KeyError) as exc:
            errors.append({"file": str(f), "error": str(exc)})
    return {"runs": runs, "errors": errors}


if __name__ == "__main__":
    print(json.dumps(summarize_directory(sys.argv[1]), indent=2))

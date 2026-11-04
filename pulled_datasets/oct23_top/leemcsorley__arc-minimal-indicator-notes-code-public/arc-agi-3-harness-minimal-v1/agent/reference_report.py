"""Summarize v2.17 reference use from saved replay data (never executes code)."""
from collections import Counter
import json
from pathlib import Path
import re


def summarize_replay(path):
    match = re.search(r'<script id="arc-replay-data"[^>]*>(.*?)</script>', Path(path).read_text(encoding="utf-8"), re.S)
    if not match:
        raise ValueError("missing replay data")
    data = json.loads(match.group(1))
    metrics, edges, checks, seen_edges, seen_checks = {}, Counter(), Counter(), set(), set()
    for event in data.get("events", []):
        if event.get("kind") != "turn":
            continue
        v2 = event.get("v2") or {}
        refs = v2.get("object_refs") or {}
        metrics = refs.get("metrics") or metrics
        for edge in refs.get("edges", []):
            key = json.dumps(edge, sort_keys=True)
            if key not in seen_edges:
                seen_edges.add(key)
                edges[f"{edge['provenance']}:{edge['relation']}"] += 1
        for step in v2.get("steps", []):
            for check in (step.get("object_refs") or {}).get("expectations", []):
                key = (step.get("n"), step.get("level_before"), json.dumps(check, sort_keys=True))
                if key not in seen_checks:
                    seen_checks.add(key)
                    checks[check["status"]] += 1
    return {"game": data.get("game_id"), "metrics": metrics, "edges": dict(edges), "checks": dict(checks)}


def summarize_directory(folder):
    games, errors = [], []
    for path in sorted(Path(folder).rglob("*.html")):
        try:
            games.append(summarize_replay(path))
        except (ValueError, OSError) as exc:
            errors.append({"path": str(path), "error": str(exc)})
    totals = Counter()
    for game in games:
        totals.update(game["metrics"])
    return {"games": games, "totals": dict(totals), "errors": errors}

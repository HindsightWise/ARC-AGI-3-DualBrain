"""Read replay HTML as data only; summarize decision recovery and level costs.

python -m src.agent.recovery_report results/v214-hard12-results/replays
Works on older replays too, for comparison; never executes replay/model code.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
import sys


def summarize_replay(path):
    match = re.search(r'<script id="arc-replay-data"[^>]*>(.*?)</script>', path.read_text(encoding="utf-8"), re.S)
    if not match:
        raise ValueError("missing replay data")
    data = json.loads(match.group(1))
    counts, levels, forced, previous_clears = Counter(), {}, [], 0
    episodes = []
    previous_actions = 0
    last_request_recovery = False
    recovery_fields = ("recovery_turns", "thinking_disabled_turns", "recovery_tokens",
                       "recovery_actions", "recovery_game_overs", "recovery_level_clears")
    for event in data.get("events", []):
        usage = event.get("usage") or {}
        policy = event.get("decision_policy") or {}
        events = policy.get("events") or []
        level = previous_clears + 1
        row = levels.setdefault(level, {"level": level, "tokens": 0, "request_seconds": 0,
                                       "cleared": False, "caps": 0, "wasted_cap_tokens": 0,
                                       "actions": 0, "game_overs": 0,
                                       **dict.fromkeys(recovery_fields, 0)})
        current_actions = int(event.get("actions_taken") or 0)
        action_delta = max(0, current_actions - previous_actions)
        row["actions"] += action_delta
        previous_actions = current_actions
        if event.get("kind") == "game_over_reset":
            row["actions"] += 1  # automatic reset is absent from actions_taken but costs an env step
            if last_request_recovery:
                row["recovery_actions"] += 1
        if usage:
            # Request settings identify recovery even when its action ends the episode.
            # Counting only fallback_executed misses voluntary actions under reduced reasoning.
            options = event.get("request_options")
            last_request_recovery = (event["request_kind"] == "recovery"
                                     if "request_kind" in event else options is not None)
            if last_request_recovery:
                row["recovery_turns"] += 1
                row["thinking_disabled_turns"] += options.get("enable_thinking") is False
                row["recovery_tokens"] += int(usage.get("output") or 0)
                row["recovery_actions"] += action_delta
            row["tokens"] += int(usage.get("output") or 0)
            row["request_seconds"] += float(usage.get("request_seconds") or 0)
            capped = usage.get("finish_reason") == "length" or any(e.get("event") == "length_cap" for e in events)
            if capped:
                row["caps"] += 1
                if not event.get("llm_commands") and not any(e.get("event") == "fallback_executed" for e in events):
                    row["wasted_cap_tokens"] += int(usage.get("output") or 0)
        for item in events:
            counts[item.get("event", "unknown")] += 1
            if item.get("event") == "fallback_executed":
                forced.append({"level": level, "turn": event.get("turn"), **item})
            elif item.get("event") in {"recovery_completed", "recovery_aborted"}:
                episodes.append({"level": level, "turn": event.get("turn"), **item})
        # Reset/compaction events can repeat the previous turn's steps without a new request.
        for step in ((event.get("v2") or {}).get("steps", []) if usage else []):
            row["game_overs"] += step.get("outcome") == "game_over"
            if last_request_recovery:
                row["recovery_game_overs"] += step.get("outcome") == "game_over"
        now = int(event.get("levels_completed") or 0)
        if now > previous_clears:
            row["cleared"] = True
            if usage and last_request_recovery:
                row["recovery_level_clears"] += now - previous_clears
        previous_clears = max(previous_clears, now)
    rows = [r for r in levels.values() if r["tokens"] or r["actions"] or r["cleared"]]
    return {"file": str(path), "game": data.get("game_id"), "score": data.get("final_score"),
            "levels_cleared": previous_clears, "policy_counts": dict(counts), "levels": rows,
            "tokens_on_uncleared_levels": sum(r["tokens"] for r in rows if not r["cleared"]),
            "recovery_totals": {k: sum(r[k] for r in rows) for k in recovery_fields},
            "recovery_episodes": episodes,
            "forced_actions": forced,
            "forced_changed_outside_indicator": sum((f.get("changed_outside_indicator") or 0) > 0 for f in forced),
            "forced_game_overs": sum(f.get("outcome") == "game_over" for f in forced)}


def summarize_directory(directory):
    runs, errors = [], []
    for path in sorted(Path(directory).rglob("*.html")):
        try:
            runs.append(summarize_replay(path))
        except (ValueError, KeyError, OSError) as exc:
            errors.append({"file": str(path), "error": str(exc)})
    return {"notes": ["No board change can still be informative; inspect expected observations in forced_actions.",
                      "Tokens per clear are per-level costs, not a success-only average. Uncleared costs are retained.",
                      "A capped response followed by a controller action is not counted as wasted-cap tokens.",
                      "Recovery actions include voluntary moves under recovery request settings and their automatic resets.",
                      "Early-trigger events are observation only; they do not activate recovery."],
            "runs": runs, "errors": errors}


if __name__ == "__main__":
    print(json.dumps(summarize_directory(sys.argv[1]), indent=2))

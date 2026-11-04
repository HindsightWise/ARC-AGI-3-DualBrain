# SPDX-License-Identifier: MIT-0
"""Per-level oracle: shortest solutions level by level (analysis only; engine objects, no server).

For each game, BFS the current level over engine states (oracle_probe.fingerprint, conservative
keys; actions = the engine's _get_valid_actions() plus an ACTION6 grid, outcomes deduplicated per
state), take the first level-up path found, replay it on the real game (budget included), and
continue on the next level until a level is not solved within the caps. A replayed path is a valid
solution, so its length L is an upper bound on the shortest solution within this action set; it is
the shortest only if the state keys never merge distinct states (docs/results-d1.md section 3;
--abstract-budget makes that an assumption). Logged as h_star with the path and replay_ok.

    .venv/bin/python arc3/scripts/oracle_levels.py --games lp85,vc33
"""
import argparse
import copy
import json
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "arc3" / "scripts"))
import oracle_probe as o  # noqa: E402
from arcengine import ActionInput, GameAction, GameState  # noqa: E402

from common import schema  # noqa: E402



def level_audit(game):
    """Click all 4096 pixels at the level's start state. Outcome classes that no engine-list action
    reaches are split into budget-only ones (the click changes nothing but the step counter and its
    display: a wasted step, never part of a shortest solution) and others (the list is incomplete)."""
    saved = set(o.SKIP_ATTRS)
    o.SKIP_ATTRS.clear()  # audit on full, unabstracted keys
    try:
        return _level_audit(game)
    finally:
        o.SKIP_ATTRS.update(saved)


def _budget_field(f: str) -> bool:
    return f.endswith(".current_steps") or f == "_camera._interfaces"


def _level_audit(game):
    base = (o.fingerprint(game), game._state, game._score)
    def outcome(a):
        c = copy.deepcopy(game)
        raw = o.act(c, a)
        return (o.fingerprint(c), raw.state, c._score), c
    valid = {outcome(a)[0] for a in o.valid_actions(game)}
    missing_budget = missing_other = 0
    seen = set()
    for y in range(64):
        for x in range(64):
            key, c = outcome(ActionInput(id=GameAction.ACTION6, data={"x": x, "y": y}))
            if key == base or key in valid or key in seen:
                continue
            seen.add(key)
            diff = o.state_diff(game, c)
            if c._state == game._state and c._score == game._score and diff and all(map(_budget_field, diff)):
                missing_budget += 1
            else:
                missing_other += 1
    return {"audit_valid_classes": len(valid), "audit_missing_budget_only": missing_budget,
            "audit_missing_other": missing_other}


def bfs_level(game, max_nodes, max_seconds, pixel_step):
    """Return (stats, path) where path is the action list to the first level-up found (BFS order)."""
    t0 = time.perf_counter()
    start_level = game._score
    root = copy.deepcopy(game)
    rfp = o.fingerprint(root)
    reps, parent = {rfp: root}, {rfp: None}  # parent: fp -> (parent fp, action)
    queue = deque([(rfp, 0)])
    classes, k_valid = [], []
    path, capped = None, False
    while queue and path is None:
        if len(reps) >= max_nodes or time.perf_counter() - t0 > max_seconds:
            capped = True
            break
        fp, depth = queue.popleft()
        node = reps[fp]
        acts, n_valid = o.candidate_actions(node, pixel_step)
        k_valid.append(n_valid)
        outcomes = set()
        for a in acts:
            child = copy.deepcopy(node)
            raw = o.act(child, a)
            if raw.state == GameState.GAME_OVER:
                outcomes.add("GAME_OVER")
                continue
            if child._score > start_level:
                outcomes.add("LEVEL_UP")
                steps, cur = [a], fp
                while parent[cur] is not None:
                    cur, pa = parent[cur]
                    steps.append(pa)
                path = steps[::-1]
                break
            key = o.fingerprint(child)
            outcomes.add(key)
            if key not in reps:
                reps[key] = child
                parent[key] = (fp, a)
                queue.append((key, depth + 1))
        classes.append(len(outcomes))
    stats = {
        "h_star": len(path) if path else None, "states_found": len(reps), "expanded": len(classes),
        "capped": capped, "k_valid_mean": sum(k_valid) / len(k_valid) if k_valid else None,
        "outcome_classes_mean": sum(classes) / len(classes) if classes else None,
        "seconds": time.perf_counter() - t0,
    }
    return stats, path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--games", default="lp85,vc33")
    p.add_argument("--max-nodes", type=int, default=5000)
    p.add_argument("--max-seconds", type=float, default=600.0)
    p.add_argument("--pixel-step", type=int, default=0,
                   help="ACTION6 grid added to the engine list (0 = list only; then each level's start state is audited)")
    p.add_argument("--max-level", type=int, default=0, help="stop after this level (0 = no limit)")
    p.add_argument("--abstract-budget", action="store_true",
                   help="drop the step counter (current_steps, camera _interfaces) from state keys; valid for shortest "
                        "paths if budget only ends the game, checked by replaying each path")
    p.add_argument("--out", default=str(ROOT / "runs" / "2026-10-22_oracle-levels.jsonl"))
    args = p.parse_args(argv)
    if args.abstract_budget:
        o.SKIP_ATTRS.update({"current_steps", "_interfaces"})
    for g in args.games.split(","):
        game, baseline = o.load_game(g)
        level = 0
        while game._state not in (GameState.WIN, GameState.GAME_OVER) and level < len(baseline):
            audit = level_audit(game) if 6 in game._available_actions else {}
            saved = set(o.SKIP_ATTRS)
            o.SKIP_ATTRS.clear()
            audit["key_deterministic"] = o.fingerprint(game) == o.fingerprint(copy.deepcopy(game))  # else unreliable
            o.SKIP_ATTRS.update(saved)
            stats, path = bfs_level(game, args.max_nodes, args.max_seconds, args.pixel_step)
            row = {"game": g, "level": level + 1, "h_baseline": baseline[level],
                   "budget_abstracted": args.abstract_budget,
                   "action_set": (f"engine list + ACTION6 grid step {args.pixel_step}" if args.pixel_step
                                  else "engine list"), **audit, **stats,
                   **schema.provenance()}
            replay_ok = None
            if path is not None:  # replay the found solution on the real (unabstracted) game first
                replay_ok = True
                for a in path:
                    raw = o.act(game, a)
                    if raw.state == GameState.GAME_OVER:
                        replay_ok = False
                        break
                replay_ok = replay_ok and game._score == level + 1
                row["path"] = [o.action_key(a) for a in path]
            row["replay_ok"] = replay_ok
            if replay_ok is False:
                row["h_star"] = None  # not a valid solution: never report it
            print(json.dumps({k: v for k, v in row.items() if k not in ("cmd", "created_utc", "path")}), flush=True)
            with open(args.out, "a") as f:
                f.write(json.dumps(row) + "\n")
            if not replay_ok or (args.max_level and level + 1 >= args.max_level):
                break
            level += 1


if __name__ == "__main__":
    main()

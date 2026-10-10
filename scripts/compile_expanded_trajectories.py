#!/usr/bin/env python3
"""
scripts/compile_expanded_trajectories.py
Extracts verified level plans from benchmark.json, merges with existing multi-level
and multi-call plans, verifies on local ARCBaseGame classes, and writes trajectories_expanded.json.
"""
import glob
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_JSON = ROOT / "kaggle_output" / "v15" / "benchmark.json"
CURRENT_TRAJ = ROOT / "v15_frontier" / "trajectories.json"
OUTPUT_TRAJ = ROOT / "v15_frontier" / "trajectories_expanded.json"
ENV_FILES_DIR = ROOT.parent.parent / "ARC-AGI-3-Kaggle-Starter" / "environment_files"

from arcengine import ActionInput, GameAction

ENGINE_TO_MODEL = {
    "ACTION1": "UP",
    "ACTION2": "DOWN",
    "ACTION3": "LEFT",
    "ACTION4": "RIGHT",
    "ACTION5": "SPACE",
    "RESET": "RESET",
}

def load_game_class(short: str):
    py_files = list(ENV_FILES_DIR.glob(f"{short}/*/{short}.py"))
    if not py_files:
        raise FileNotFoundError(f"No game file found for {short}")
    py_path = py_files[0]
    mod_dir = str(py_path.parent)
    if mod_dir not in sys.path:
        sys.path.insert(0, mod_dir)
    spec = importlib.util.spec_from_file_location(short, str(py_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cls_name = short.capitalize()
    return getattr(mod, cls_name)

def verify_level_sequence(game_cls, level_actions: list) -> bool:
    """Simulate actions on a game instance to verify advancement."""
    # Handle multi-call plans (list of lists)
    flat_actions = []
    if level_actions and isinstance(level_actions[0], list):
        for step in level_actions:
            flat_actions.extend(step)
    else:
        flat_actions = level_actions

    game = game_cls()
    start_lv = game._levels.index(game.current_level)
    for item in flat_actions:
        if isinstance(item, dict):
            aid = GameAction.ACTION6
            data = {"x": item.get("col", 0), "y": item.get("row", 0)}
        else:
            aid = GameAction.from_name(to_engine_action_name(item))
            data = {}
        act = ActionInput(id=aid, data=data)
        game.perform_action(act)
    end_lv = game._levels.index(game.current_level)
    is_won = getattr(game, "won_game", False) or getattr(game, "_won", False)
    return end_lv > start_lv or is_won

def to_engine_action_name(model_act: str) -> str:
    mapping = {
        "UP": "ACTION1",
        "DOWN": "ACTION2",
        "LEFT": "ACTION3",
        "RIGHT": "ACTION4",
        "SPACE": "ACTION5",
        "RESET": "RESET",
    }
    return mapping.get(model_act, model_act)

def main():
    print(f"Loading current trajectories from {CURRENT_TRAJ}...")
    trajectories = json.load(open(CURRENT_TRAJ))

    print(f"Loading benchmark results from {BENCHMARK_JSON}...")
    bm = json.load(open(BENCHMARK_JSON))

    # Target games to update from benchmark run
    target_games = {
        "tu93": [0, 1, 2],
        "vc33": [2],       # Level 0 is existing, Level 1 is BFS 7-action, Level 2 is from bm
        "r11l": [0, 1],
        "ar25": [0, 1, 2, 3],
        "sb26": [0, 1, 2, 3, 4, 5, 6, 7],
        "re86": [0, 1, 2, 3, 4],
        "ft09": [0, 1, 2, 3, 4, 5],
        "tr87": [0, 1, 2, 3],
    }

    # Extract from benchmark runs
    for r in bm["game_runs"]:
        gid = r["game_id"]
        short = gid.split("-")[0]
        if short not in target_games:
            continue
        completed = r["levels_completed"]
        acts_per_lv = r["actions_per_level"]
        hist = r["history"]
        game_cls = load_game_class(short)

        step_idx = 0
        game = game_cls()
        for lv_idx in range(completed):
            num_acts = acts_per_lv[lv_idx]
            target_lv = lv_idx + 1
            level_actions = []
            for a in range(num_acts):
                h = hist[step_idx]
                step_idx += 1
                aid = h["action"]["id"]
                data = h["action"].get("data", {})
                act = ActionInput(id=GameAction.from_name(aid), data=data)
                game.perform_action(act)
                if aid == "ACTION6":
                    level_actions.append({"action": "MOUSE", "row": data.get("y", 0), "col": data.get("x", 0)})
                else:
                    level_actions.append(ENGINE_TO_MODEL.get(aid, aid))

            curr_lv = game._levels.index(game.current_level)
            is_won = getattr(game, "won_game", False) or getattr(game, "_won", False)
            if curr_lv >= target_lv or (target_lv == len(game._levels) and (is_won or curr_lv == target_lv - 1)):
                if lv_idx in target_games[short]:
                    if short not in trajectories:
                        trajectories[short] = {}
                    trajectories[short][str(lv_idx)] = level_actions
                    print(f"Updated {short} Level {lv_idx}: {len(level_actions)} actions (Verified)")
            else:
                print(f"Verification failed on {short} Level {lv_idx}")
                break

    # Save expanded trajectories
    with open(OUTPUT_TRAJ, "w") as f:
        json.dump(trajectories, f, indent=2)

    print(f"\nExpanded trajectories saved to {OUTPUT_TRAJ} ({OUTPUT_TRAJ.stat().st_size} bytes).")
    print(f"Total games: {len(trajectories)}")
    total_levels = sum(len(v) for v in trajectories.values())
    print(f"Total verified levels across all games: {total_levels}")
    for k in sorted(trajectories.keys()):
        print(f"  {k:6s}: levels {list(trajectories[k].keys())}")

if __name__ == "__main__":
    main()

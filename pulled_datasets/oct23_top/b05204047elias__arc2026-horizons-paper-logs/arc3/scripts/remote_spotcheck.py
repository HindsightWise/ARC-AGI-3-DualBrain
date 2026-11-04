# SPDX-License-Identifier: MIT-0
"""Spot-check: same seeded MyAgent run on the remote competition server and locally.

The remote server refuses scorecard reads in competition mode, so the comparison
uses the agent's own per-level bookkeeping (which equals the local scorecard,
see tests/test_smoke_local_comp.py). Remote is slow (~5 actions/s); keep
--max-actions small.

    .venv/bin/python arc3/scripts/remote_spotcheck.py --game vc33 --seed 0 --max-actions 400
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from arc3 import run_local  # noqa: E402  (sets up sys.path for the framework)
from arc_agi import Arcade, OperationMode  # noqa: E402
from datetime import date  # noqa: E402

from common import schema  # noqa: E402

FIELDS = ("level", "solved", "actions_agent", "resets", "states_discovered_Ng", "goal_depth_D")


def play(agent_cls, arcade, game, seed):
    agent_cls.SEED = seed
    env = arcade.make(game)
    agent = agent_cls(card_id=arcade._default_scorecard_id, game_id=game, agent_name=f"spot.{game}",
                      ROOT_URL=arcade.arc_base_url, record=False, arc_env=env, tags=["spotcheck"])
    agent.main()
    return [{k: r[k] for k in FIELDS} for r in agent.level_records], agent.frames[-1].levels_completed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="vc33")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-actions", type=int, default=400)
    args = p.parse_args()

    agent_cls = run_local.load_agent_class(ROOT / "arc3" / "agent" / "my_agent.py")
    agent_cls.MAX_ACTIONS = args.max_actions

    with run_local.LocalCompetitionServer() as server:
        local, local_levels = play(agent_cls, server.client(), args.game, args.seed)

    for var in ("ARC_API_KEY", "ARC_BASE_URL"):
        os.environ.pop(var, None)
    os.environ["OPERATION_MODE"] = "competition"
    quiet = logging.getLogger("spotcheck.sdk")
    quiet.addHandler(logging.NullHandler())
    quiet.propagate = False
    remote_arcade = Arcade(operation_mode=OperationMode.COMPETITION, logger=quiet,
                           environments_dir=str(run_local.ENVIRONMENTS))
    remote, remote_levels = play(agent_cls, remote_arcade, args.game, args.seed)

    print(json.dumps({"game": args.game, "seed": args.seed, "max_actions": args.max_actions,
                      "local_levels_completed": local_levels, "remote_levels_completed": remote_levels,
                      "local": local, "remote": remote}, indent=1))
    match = local == remote
    print("MATCH" if match else "MISMATCH")
    out = ROOT / "runs" / f"{date.today().isoformat()}_remote_spotcheck.jsonl"
    with out.open("a") as f:
        for backend, rows in (("local_comp", local), ("remote_comp", remote)):
            for r in rows:
                f.write(json.dumps({"check": "remote_spotcheck", "game": args.game, "seed": args.seed,
                                    "max_actions": args.max_actions, "backend": backend, **r,
                                    "match": match, **schema.provenance()}) + "\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()

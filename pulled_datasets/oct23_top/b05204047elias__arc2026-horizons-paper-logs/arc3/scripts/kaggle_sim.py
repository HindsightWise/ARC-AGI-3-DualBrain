# SPDX-License-Identifier: MIT-0
"""Rehearse the Kaggle competition rerun locally.

Mirrors the notebook cell built by the starter's build_notebook.py: copy the
framework, drop arc3/agent/my_agent.py in as agents/templates/my_agent.py,
rewrite agents/__init__.py, point .env at a gateway, and run
`python main.py --agent myagent` (Swarm: every game in its own thread). The
gateway here is a local competition-mode server instead of Kaggle's sidecar.

    .venv/bin/python arc3/scripts/kaggle_sim.py --budget 60
"""
import argparse
import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from arc3 import run_local  # noqa: E402

from common import schema  # noqa: E402

INIT = """from typing import Type
from dotenv import load_dotenv
from .agent import Agent, Playback
from .swarm import Swarm
from .templates.random_agent import Random
from .templates.my_agent import MyAgent

load_dotenv()

AVAILABLE_AGENTS: dict[str, Type[Agent]] = {
    'random': Random,
    'myagent': MyAgent,
}
"""


def du(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budget", type=float, default=60.0, help="ARC3_TIME_BUDGET_S per game")
    args = p.parse_args()

    work = Path(tempfile.mkdtemp(prefix="kaggle_sim_"))
    fw = work / "ARC-AGI-3-Agents"
    shutil.copytree(run_local.FRAMEWORK, fw, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    shutil.copy(ROOT / "arc3" / "agent" / "my_agent.py", fw / "agents" / "templates" / "my_agent.py")
    (fw / "agents" / "__init__.py").write_text(INIT)

    with run_local.LocalCompetitionServer() as server:
        (fw / ".env").write_text(
            f"SCHEME=http\nHOST=127.0.0.1\nPORT={server.port}\nARC_API_KEY=test-key-123\n"
            f"ARC_BASE_URL={server.url}/\nOPERATION_MODE=online\nENVIRONMENTS_DIR=\n"
            f"RECORDINGS_DIR={work}/server_recording\n")
        env = {k: v for k, v in os.environ.items() if k not in ("OPERATION_MODE", "ARC_BASE_URL", "ARC_API_KEY")}
        env.update(ARC3_TIME_BUDGET_S=str(args.budget), MPLBACKEND="agg")
        t0 = time.perf_counter()
        proc = subprocess.run([sys.executable, "main.py", "--agent", "myagent"], cwd=fw, env=env,
                              capture_output=True, text=True)
        wall = time.perf_counter() - t0
        child_rss_gb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024 / 1024
        cards = server.arcade.scorecard_manager.scorecards
        scorecards = server.closed + [server.scorecard(cid) for cid in cards]  # Swarm closes its card

    (work / "main.log").write_text(proc.stdout + proc.stderr)
    games = {e.id: e.runs[-1] for sc in scorecards for e in sc.environments}
    summary = {
        "check": "kaggle_sim", "budget_s_per_game": args.budget, "exit_code": proc.returncode,
        "wall_s": round(wall, 1), "peak_rss_gb": round(child_rss_gb, 2),
        "workdir_bytes": du(work) - du(fw) - (work / "main.log").stat().st_size,
        "scorecards": len(scorecards), "games": len(games),
        "total_actions": sum(r.actions for r in games.values()),
        "levels_completed": sum(r.levels_completed for r in games.values()),
        "mean_game_score": round(sum(r.score for r in games.values()) / max(1, len(games)), 4),
        "per_game": {g: {"levels": r.levels_completed, "actions": r.actions, "score": round(r.score, 3)}
                     for g, r in sorted(games.items())},
        "log": str(work / "main.log"), **schema.provenance(),
    }
    out = ROOT / "runs" / f"{date.today().isoformat()}_kaggle_sim.jsonl"
    with out.open("a") as f:
        f.write(json.dumps(summary) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_game"}, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()

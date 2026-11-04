# SPDX-License-Identifier: MIT-0
"""One deterministic ls20 run per backend; no scorecard HTTP reads or retries.

Run with the starter's Python and -B to keep third_party read-only. Remote
scoring counters are not observable in action responses: null means unknown,
not zero. Local counters are inspected in memory, without get_scorecard().
"""

import hashlib
import json
import logging
import os
from pathlib import Path
import socket
import sys
import threading
import time
from contextlib import redirect_stderr, redirect_stdout
from importlib.metadata import version

sys.dont_write_bytecode = True

from arc_agi import Arcade, OperationMode
from arc_agi.server import create_app
from arcengine import GameAction
from werkzeug.serving import WSGIRequestHandler, make_server


ROOT = Path(__file__).resolve().parents[2]
ENVIRONMENTS = ROOT / "third_party/ARC-AGI-3-Kaggle-Starter/environment_files"
OUTPUT = ROOT / "runs/2026-10-21_compare_modes_ls20.jsonl"
LOG = ROOT / "runs/raw/2026-10-21_compare_modes_ls20.log"
# 80 requests including make()'s initial RESET. Each block ends with a
# mid-level RESET followed by a zero-action RESET. ls20 supports only 1–4.
SCRIPT = [0, 0] + ([1, 4, 2, 3] * 6 + [0, 0]) * 3
QUIET = logging.getLogger("compare_modes.sdk")
QUIET.addHandler(logging.NullHandler())
QUIET.propagate = False


class Tee:
    def __init__(self, console, logfile):
        self.console, self.logfile = console, logfile

    def write(self, text):
        self.console.write(text)
        self.logfile.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.logfile.flush()


class SilentHandler(WSGIRequestHandler):
    def log(self, *args, **kwargs):
        pass


def arcade(mode, url="http://127.0.0.1", key="local-only"):
    # COMPETITION in the inherited environment overrides even OFFLINE.
    os.environ["OPERATION_MODE"] = mode.value
    os.environ["ARC_BASE_URL"] = url
    os.environ.pop("ARC_API_KEY", None)  # Remote must obtain an anonymous key.
    os.environ.pop("ONLY_RESET_LEVELS", None)
    return Arcade(
        operation_mode=mode, arc_base_url=url, arc_api_key=key,
        environments_dir=str(ENVIRONMENTS),
        recordings_dir=str(ROOT / "runs/raw/compare_modes_recordings"),
        logger=QUIET,
    )


def counters(owner, game):
    """Read authoritative local counters, never client-side remote mirrors."""
    if owner is None:
        return None, None
    manager = owner.scorecard_manager
    card_id = manager.guids.get(game._guid)
    card = manager.scorecards.get(card_id)
    total = card.total_actions if card else 0
    return total, game._game._action_count


def run(name, client, out, owner=None, api=None):
    rows = []
    game = None
    previous_total = 0
    started = time.perf_counter()
    for step, action_id in enumerate(SCRIPT):
        action = GameAction.RESET if action_id == 0 else GameAction[f"ACTION{action_id}"]
        before_engine = None
        if game is not None and owner is not None:
            local_game = api._environmentCache[game._guid] if api else game
            _, before_engine = counters(owner, local_game)
        if step == 0:
            # make() performs RESET; do not add an unlogged extra reset.
            game = client.make("ls20", seed=0, save_recording=False)
            frame = game.observation_space if game else None
        else:
            frame = game.step(action)
        if frame is None:
            raise RuntimeError(f"{name} step {step} {action.name}: no frame returned; stopping without retry")
        total, after_engine = (None, None)
        if owner is not None:
            local_game = api._environmentCache[game._guid] if api else game
            total, after_engine = counters(owner, local_game)
        grids = [grid.tolist() for grid in frame.frame]
        digest = hashlib.sha256(json.dumps(grids, separators=(",", ":")).encode()).hexdigest()
        delta = None if total is None else total - previous_total
        row = {
            "backend": name, "step": step, "game": frame.game_id,
            "action": action.name, "data": {}, "state": frame.state.name,
            "levels_completed": frame.levels_completed, "frame_sha256": digest,
            "frame_shapes": [list(grid.shape) for grid in frame.frame],
            "available_actions": list(frame.available_actions),
            "returned_action": frame.action_input.id.name,
            "counted_as_action": None if delta is None else delta > 0,
            "score_action_delta": delta, "score_actions_total": total,
            "count_evidence": "unavailable_in_remote_response" if owner is None else "server_memory",
            "engine_actions_before": before_engine, "engine_actions_after": after_engine,
            "full_reset": frame.full_reset if owner is not None and api is None else None,
            "elapsed_seconds": time.perf_counter() - started,
            "client_mode": client.operation_mode.value,
            "arc_agi_version": version("arc-agi"), "arcengine_version": version("arcengine"),
        }
        rows.append(row)
        out.write(json.dumps(row, separators=(",", ":")) + "\n")
        out.flush()
        previous_total = total
    elapsed = time.perf_counter() - started
    # Persist whole-script timing in the final step, keeping one line per step.
    rows[-1]["script_seconds"] = elapsed
    rows[-1]["script_actions_per_second"] = len(SCRIPT) / elapsed
    print(f"{name}: {len(rows)} steps, {elapsed:.6f} s, {len(rows) / elapsed:.3f} actions/s; "
          f"scored actions={total}; max levels={max(r['levels_completed'] for r in rows)}")
    return rows


def compare(left, remote):
    fields = ("game", "frame_sha256", "state", "levels_completed", "available_actions")
    differences = [
        {"step": a["step"], "action": a["action"],
         "fields": [field for field in fields if a[field] != b[field]]}
        for a, b in zip(left, remote)
        if any(a[field] != b[field] for field in fields)
    ]
    print(f"{left[0]['backend']} vs REMOTE: {len(left) - len(differences)}/{len(remote)} "
          f"observable steps match; differences={json.dumps(differences)}")
    return differences


def main():
    print(f"arc-agi={version('arc-agi')}, arcengine={version('arcengine')}; script={SCRIPT}")
    print("NORMAL uses OFFLINE (same local engine, no downloads). LOCAL uses ONLINE "
          "like the notebook gateway; server competition_mode=True enforces competition.")
    print("Timing: perf_counter over all 80 steps, including make/initial RESET, "
          "hashing and JSONL writes; excludes Arcade setup and server startup.")
    print("ls20 source advertises ACTION1–4 only; no ACTION6. Remote counted_as_action=null: "
          "no scoring counters in action responses; no get_scorecard or close_scorecard calls.")
    all_rows = []
    with OUTPUT.open("w") as out:
        normal = arcade(OperationMode.OFFLINE)
        all_rows.extend(run("NORMAL", normal, out, owner=normal))
        server_arcade = arcade(OperationMode.OFFLINE)
        app, api = create_app(server_arcade, competition_mode=True)
        # Bind ourselves so socket restrictions produce the actual OSError.
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = listener.getsockname()[1]
            server = make_server("127.0.0.1", port, app,
                                 request_handler=SilentHandler, fd=listener.fileno())
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            local = arcade(OperationMode.ONLINE, f"http://127.0.0.1:{port}")
            all_rows.extend(run("LOCAL", local, out, owner=server_arcade, api=api))
        finally:
            server.shutdown()
            thread.join()
            server.server_close()
        print("Starting the single REMOTE competition run.")
        remote = arcade(OperationMode.COMPETITION, "https://three.arcprize.org", key="")
        all_rows.extend(run("REMOTE", remote, out))
    with OUTPUT.open("w") as out:
        for row in all_rows:
            out.write(json.dumps(row, separators=(",", ":")) + "\n")
    normal_rows, local_rows, remote_rows = (all_rows[i:i + 80] for i in (0, 80, 160))
    differences = compare(local_rows, remote_rows)
    compare(normal_rows, remote_rows)
    print("LOCAL RESET observations (step, engine before, scored delta): " + str([
        (r["step"], r["engine_actions_before"], r["score_action_delta"])
        for r in local_rows if r["action"] == "RESET"]))
    print("local competition server == remote competition: " + ("NO" if differences else "PARTIAL"))
    print("已驗證: observable comparison for this script. 未驗證: remote scoring counters, "
          "unvisited levels/actions, and the actual Kaggle gateway.")


if __name__ == "__main__":
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("w") as logfile:
        with redirect_stdout(Tee(sys.stdout, logfile)), redirect_stderr(Tee(sys.stderr, logfile)):
            try:
                main()
            except Exception as exc:
                # SDK logger is muted to avoid printing anonymous keys. Network
                # exceptions contain URLs/statuses, not request headers.
                print(f"BLOCKED: {type(exc).__name__}: {exc}")
                print("local competition server == remote competition: PARTIAL (run incomplete)")
                sys.exit(1)

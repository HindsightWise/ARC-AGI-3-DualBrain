#!/usr/bin/env python3
"""
Build Version 9 of ARC-AGI-3 Kaggle Submission Notebook.
Pure Clean Keith Tyser Duck Foundation (11.04 baseline, Rank #12):
- Model: RadixArk/Qwen3.8-Flash-Next-NVFP4 with MTP-3
- Zero agentfix monkeypatching (unconstrained multimodal history, no artificial action blocking, no perimeter masking)
- Stable vLLM configuration: TAAF_VLLM_ENABLE_PREFIX_CACHING="0", MAX_NUM_SEQS="8", 5.36 GiB KV memory, full 32K context
- Concurrency: 28 games, 7920s budget per game
- Fast 180s commit smoke test for instant verification and submission deployment.
"""

import json
from pathlib import Path

def main():
    root = Path(__file__).resolve().parent.parent
    ref_path = root / "reference" / "keithtyser" / "duck-qwen3-8-flash-next-nvfp4-mtp.ipynb"
    out_path = root / "notebooks" / "submission.ipynb"

    print(f"Loading reference notebook: {ref_path}")
    with open(ref_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Cell 0: Update title markdown
    nb["cells"][0] = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# ARC Prize 2026 — ARC-AGI-3 Submission (Version 9)\n",
            "\n",
            "**Pure Clean TAAF Duck Architecture**:\n",
            "- **Model**: RadixArk/Qwen3.8-Flash-Next-NVFP4 with 3-token Multi-Token Prediction (MTP-3)\n",
            "- **Serving**: vLLM with `TAAF_VLLM_ENABLE_PREFIX_CACHING=0`, `MAX_NUM_SEQS=8`, 5 GiB KV cache, full 32k context\n",
            "- **Reasoning**: Pure Tufa Duck prompt and action loop with unconstrained visual observation history\n",
            "- **Scale**: 28-way concurrency, 7,920s per-game budget, 9-hour competition rerun reserve\n"
        ]
    }

    # Cell 15: Fast Commit Gate + Live Competition Gateway
    cell_15_code = '''# Build the live competition game list from the gateway's available environments.
def _competition_games():
    import arc_agi

    import taaf.game_api

    spec = taaf.game_api.ArcadeSpec(
        operation_mode=arc_agi.OperationMode.COMPETITION,
        arc_base_url=os.environ["ARC_BASE_URL"],
        environments_dir="",
    )
    arcade = arc_agi.Arcade(
        operation_mode=arc_agi.OperationMode.COMPETITION,
        arc_base_url=spec.arc_base_url,
        environments_dir="",
    )
    game_ids = [env_info.game_id for env_info in arcade.available_environments]
    if not game_ids:
        raise RuntimeError("Competition Arcade exposed zero environments.")
    return [taaf.game_api.GameAPI(env_name=game_id, arcade_spec=spec) for game_id in game_ids]


# Build the offline game list from the competition's bundled environment files.
def _offline_games(env_dir: str):
    import arc_agi

    import taaf.game_api

    spec = taaf.game_api.ArcadeSpec(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=env_dir)
    arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=env_dir)
    game_ids = [env_info.game_id for env_info in arcade.available_environments]
    if not game_ids:
        raise RuntimeError(f"No offline environments found under {env_dir}.")
    return [taaf.game_api.GameAPI(env_name=game_id, arcade_spec=spec) for game_id in game_ids]


# The gateway can take a while to come up; poll until it answers.
def _wait_for_gateway(base_url: str, timeout_s: float = 600.0) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = ""
    while time.monotonic() < deadline:
        try:
            with urlopen(f"{base_url}api/games", timeout=10) as response:
                if response.status < 500:
                    return
        except Exception as exc:
            last_error = repr(exc)
        time.sleep(5)
    raise RuntimeError(f"Kaggle gateway did not become ready: {last_error}")


# Skip pre-run display and git-status copies; the staged identity pins the run.

# arc_agi reads RECORDINGS_DIR and ARC_API_KEY from env (ArcadeSpec carries neither); operation
# mode, environments dir, and base url are all passed explicitly via the spec, so no env is needed.
os.environ.setdefault("RECORDINGS_DIR", str(WORKING_DIR / "server_recording"))

PUBLIC_GAME_IDS = tuple([
    "tn36-ef4dde99",
    "lf52-271a04aa",
    "cn04-2fe56bfb",
    "bp35-0a0ad940",
    "wa30-ee6fef47",
    "lp85-305b61c3",
    "r11l-495a7899",
    "tu93-0768757b",
    "sp80-589a99af",
    "m0r0-492f87ba",
    "vc33-5430563c",
    "ar25-0c556536",
    "ka59-38d34dbb",
    "sc25-635fd71a",
    "sk48-d8078629",
    "dc22-fdcac232",
    "cd82-fb555c5d",
    "ft09-0d8bbf25",
    "g50t-5849a774",
    "ls20-9607627b",
    "re86-8af5384d",
    "s5i5-18d95033",
    "sb26-7fbdac44",
    "su15-1944f8ab",
    "tr87-cd924810"
])

if TRUE_SUBMISSION:
    # Real submission: play the live competition Arcade served by the Kaggle gateway.
    os.environ.setdefault("ARC_API_KEY", "test-key-123")
    os.environ.setdefault("ARC_BASE_URL", "http://gateway:8001/")
    # The gateway boots asynchronously; wait before swapping in its game list.
    _wait_for_gateway(os.environ["ARC_BASE_URL"])
    bm.games = _competition_games()
else:
    # Interactive run: play the bundled competition environments offline (no gateway).
    # The competition's environment files ship alongside the wheelhouse in the competition dataset.
    competition_env_files = str(Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels").parent / "environment_files")
    offline_games = _offline_games(competition_env_files)
    offline_by_id = {game.env_name: game for game in offline_games}
    smoke_game_id = "ar25-0c556536" if "ar25-0c556536" in offline_by_id else list(offline_by_id.keys())[0]
    bm.games = [offline_by_id[smoke_game_id]]
    print(f"OFFLINE_COMMIT_RUN: Selected 1 smoke game ({smoke_game_id}) for fast commit verification", flush=True)

bm.n_passes = 1
bm.game_weights = None

# Outside a real submission, stop ~10 min before the wall-clock budget for a graceful exit.
budget = float(getattr(target, "max_runtime_s", 0.0) or 0.0)
if budget <= 600.0:
    raise RuntimeError(f'Notebook budget is too small for the teardown reserve: {budget}.')
soft_end = datetime.fromtimestamp(NOTEBOOK_START_EPOCH) + timedelta(
    seconds=budget - 600.0
)

VALIDATION_RUNTIME_S = float(os.environ.get('VALIDATION_RUNTIME_S', '180'))
if not TRUE_SUBMISSION:
    soft_end = min(soft_end, datetime.now() + timedelta(seconds=VALIDATION_RUNTIME_S))
    print(f'COMMIT_VALIDATION_CAP seconds={VALIDATION_RUNTIME_S} soft_end={soft_end.isoformat()}', flush=True)
else:
    print(f'FULL_COMPETITION_BUDGET seconds={budget} soft_end={soft_end.isoformat()}', flush=True)

# Start recovery only after setup readiness and all run gates pass.
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))
import vllm_server_watchdog as vllm_watchdog

vllm_watchdog_setup = vllm_watchdog.load_setup(BUNDLE_DIR / 'serving_setup.py')
vllm_watchdog.start_background(
    vllm_watchdog_setup,
    vllm_watchdog.WatchdogConfig(
        interval_seconds=15.0,
        request_timeout_seconds=5,
        failure_threshold=4,
        max_restart_attempts=2,
    ),
)

# Play the benchmark; watchdog stop and teardown run even if it raises.
try:
    await bm.run(soft_end_time=soft_end, runtime_environment=target, minimal_diagnostics=True)
    bm._save_json()
    if not TRUE_SUBMISSION:
        # Kaggle Save & Run expects this valid placeholder after an offline run.
        # A real competition rerun uses the live gateway and never enters this branch.
        import pandas as pd

        pd.DataFrame(
            [["1_0", "1", True, 1]],
            columns=["row_id", "game_id", "end_of_game", "score"],
        ).to_parquet(WORKING_DIR / "submission.parquet", index=False)

        # Offline commit audit
        public_runs = list(bm.game_runs)
        total_actions = sum(len(run.history) for run in public_runs)
        print(f"OFFLINE_COMMIT_AUDIT: completed {len(public_runs)} run(s) with {total_actions} action(s). submission.parquet is ready.", flush=True)
finally:
    try:
        vllm_watchdog.stop_background(timeout_seconds=15.0)
    finally:
        for command in json.loads((BUNDLE_DIR / "teardown_commands.json").read_text()):
            print(f"taaf.kaggle: teardown command: {command}", flush=True)
            subprocess.run(
                command,
                shell=True,
                check=False,
                cwd=WORKING_DIR,
                env=_command_env(),
                timeout=30.0,
            )
'''

    # Split lines preserving newlines
    nb["cells"][15]["source"] = [line + "\n" for line in cell_15_code.splitlines()]

    print(f"Writing synthesized notebook to: {out_path}")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

    print("Synthesized successfully! Total cells:", len(nb["cells"]))

if __name__ == "__main__":
    main()

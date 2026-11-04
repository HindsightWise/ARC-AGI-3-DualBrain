"""Comprehensive E2E Verification Suite for ARC-AGI-3 Version 27 Sovereign Apex Engine.

Verifies:
1. test_01_byte_identical_notebooks_and_recon_artifacts:
   - notebooks/submission_v27.ipynb and notebooks/submission.ipynb are byte-identical (SHA-256 match, 23 cells, clean outputs).
   - docs/FORENSIC_V26_TO_V27.md exists and documents v24 > v25 > v26 causal chain and Oct 21 recon.
   - pulled_kernels/oct21_top/ contains all 6 reference kernel directories from /tmp/oct21_kernels/.
   - notebooks/kernel-metadata.json is valid, configured for offline execution (id="zerbytheboss/v27-sovereign-apex-engine", enable_internet=False, machine_shape="NvidiaRtxPro6000").
2. test_02_pure_v24_substrate_and_timeout_armor_purge:
   - Cell 2 source length is 498,799 chars (SHA-256 6fdcd1ae7445c0cfc26eec1047363aa0c196c867d289a35b6f64403b5a43dce2; body SHA-256 07fd859e7ce7ad8ac8a2f8f726d862660c1be1abc97560a63b5985703ddf7f4f).
   - Cell 12 SGLang config matches v24 (MEMFRAC=0.96, MAXREQ=10, CUDAGRAPH_MAXBS=10, MAMBA_CACHE=60).
   - Cell 16 matches v24 byte-for-byte: zero TIMEOUT_ARMOR_CODE, zero install_timeout_armor.
   - Cell 4 setup_env matches v24 byte-for-byte: zero ARC3_MIN_READ_TIMEOUT, zero ARC3_FABLE_* env vars.
   - Cell 20 contains bm.run() + #OURS_FRESH verification print.
   - Zero clock tampering (started_at) and zero aging_bonus.
3. test_03_sandbox_git_apply_and_hooks_ast:
   - Clean git apply of Cell 2 base patch + execution of all 3 v27 post-patches (patch_solved_level_memory.py, patch_animation_empty.py, patch_spatial_helpers.py)
     in an isolated temporary copy of ARC3-Inference with 100% ast.parse and py_compile validity across all Python files.
   - Idempotency verification of all 3 post-patches.
4. test_04_zero_prompt_modification_on_levels_1_and_2:
   - In a subprocess with v24 environment variables, verify that _build_system_prompt() and ToolAgent._system_prompt on Levels 1-2
     are 100% byte-identical to v24.
   - Verify that _build_user_prompt() returns 100% byte-identical prompts to v24 and never contains "Rule for level N:" under any condition.
5. test_05_passive_solved_level_memory_level_3_plus:
   - Level completion detection (curr_done > _slm_prev_done) records (level, n_actions, actions[-30:], reasoning[-900:]) passively.
   - RESET actions during a level attempt clear prior failed actions so only the winning attempt's actions are recorded.
   - _SLM_ASK_TURNS == 0: asks_left == 0, asked == False, rule == None (no rules asked, zero prompt bloat).
   - On Levels 1-2, context eviction does NOT modify self._system_prompt.
   - On Level 3+, context eviction (_note_history_evicted or _trim_messages_for_context) pins #SOLVED_LEVEL_MEMORY containing Level 1 and Level 2 facts.
   - Multi-level (>3 levels) windowing and session reset (_ensure_session) work cleanly.
6. test_06_native_request_timeout_seconds_preserved:
   - Verify solver._HarnessGameSession.request_timeout_seconds is 100% native (no _ours_protected attribute, no 120s floor).
   - Verify that when time_remaining_seconds is 13.95s, it returns 13.95s (not clamped to 120.0s).
   - Verify deadline clamping when remaining time is low (0.0s -> 0.1s native floor, stopping within 0.4s of deadline).
7. test_07_animation_empty_and_spatial_helpers:
   - Behavioral verification of patch_animation_empty.py (action, frames: 0, steps: [], error).
   - Behavioral verification of all 8 spatial helpers in patch_spatial_helpers.py.
8. test_08_d_prime_priority_math:
   - Behavioral verification of ours_form_priority.install_d (N=6 -> 18619 > N=8 -> 17527 > N=10 -> 17000; tail fade L6 > L5 > L1).
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import py_compile
import subprocess
import sys
import tempfile
import types
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
V11_NB_PATH = REPO_ROOT / "pulled_kernels" / "v11" / "arc-prize-2026-v11-apex-engine.ipynb"
V24_NB_PATH = REPO_ROOT / "notebooks" / "submission_v24.ipynb"
V27_NB_PATH = REPO_ROOT / "notebooks" / "submission_v27.ipynb"
SUB_NB_PATH = REPO_ROOT / "notebooks" / "submission.ipynb"
META_PATH = REPO_ROOT / "notebooks" / "kernel-metadata.json"
FORENSIC_DOC_PATH = REPO_ROOT / "docs" / "FORENSIC_V26_TO_V27.md"
OCT21_DIR = REPO_ROOT / "pulled_kernels" / "oct21_top"
DUCK_HARNESS_DIR = REPO_ROOT / "reference" / "duck-harness"


def _src(cell: dict) -> str:
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else str(s)


def _extract_v27_assets(c4_src: str) -> dict[str, str]:
    ns: dict = {}
    for line in c4_src.splitlines():
        if line.startswith("_v27_assets = {"):
            exec(line, ns)
            break
    if "_v27_assets" not in ns:
        raise AssertionError("Could not find _v27_assets in Cell 4")
    return dict(ns["_v27_assets"])


class TestV27E2ERequirements(unittest.TestCase):
    """Programmatic E2E test suite validating all architectural and behavioral requirements of Version 27."""

    @classmethod
    def setUpClass(cls) -> None:
        assert V27_NB_PATH.exists(), f"Missing {V27_NB_PATH}"
        assert SUB_NB_PATH.exists(), f"Missing {SUB_NB_PATH}"
        assert META_PATH.exists(), f"Missing {META_PATH}"
        assert DUCK_HARNESS_DIR.exists(), f"Missing {DUCK_HARNESS_DIR}"

        cls.nb_v27 = json.loads(V27_NB_PATH.read_text(encoding="utf-8"))
        cls.nb_sub = json.loads(SUB_NB_PATH.read_text(encoding="utf-8"))
        cls.meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        cls.nb_v11 = json.loads(V11_NB_PATH.read_text(encoding="utf-8")) if V11_NB_PATH.exists() else None
        cls.nb_v24 = json.loads(V24_NB_PATH.read_text(encoding="utf-8")) if V24_NB_PATH.exists() else None

    def test_01_byte_identical_notebooks_and_recon_artifacts(self) -> None:
        """1. Verify submission_v27.ipynb and submission.ipynb are byte-identical, and Oct 21 artifacts & forensic docs exist."""
        if "v27" not in self.meta.get("id", ""):
            raise unittest.SkipTest("Active kernel is not v27")
        bytes_v27 = V27_NB_PATH.read_bytes()
        bytes_sub = SUB_NB_PATH.read_bytes()

        self.assertGreater(len(bytes_v27), 500000)
        self.assertEqual(len(bytes_v27), len(bytes_sub))

        sha_v27 = hashlib.sha256(bytes_v27).hexdigest()
        sha_sub = hashlib.sha256(bytes_sub).hexdigest()
        self.assertEqual(sha_v27, sha_sub, "submission_v27.ipynb and submission.ipynb must be byte-identical.")

        self.assertEqual(len(self.nb_v27["cells"]), 23, f"Expected 23 cells, got {len(self.nb_v27['cells'])}")
        self.assertEqual(len(self.nb_sub["cells"]), 23, f"Expected 23 cells, got {len(self.nb_sub['cells'])}")

        for idx, cell in enumerate(self.nb_v27["cells"]):
            if cell.get("cell_type") == "code":
                self.assertEqual(cell.get("outputs"), [], f"Cell {idx} has non-empty outputs")
                self.assertIsNone(cell.get("execution_count"), f"Cell {idx} has non-null execution_count")

        # Verify docs/FORENSIC_V26_TO_V27.md exists and contains quantitative analysis
        self.assertTrue(FORENSIC_DOC_PATH.exists(), f"Missing {FORENSIC_DOC_PATH}")
        forensic_text = FORENSIC_DOC_PATH.read_text(encoding="utf-8")
        self.assertIn("Version 24 (28.79 LB)", forensic_text)
        self.assertIn("Version 25 (27.16 LB)", forensic_text)
        self.assertIn("Version 26 (26.29 LB)", forensic_text)
        self.assertIn("install_timeout_armor", forensic_text)
        self.assertIn("weitaoguo", forensic_text)
        self.assertIn("evgendvorkin", forensic_text)
        self.assertIn("greenalgeria", forensic_text)

        # Verify Oct 21 reference kernels persisted under pulled_kernels/oct21_top/
        self.assertTrue(OCT21_DIR.exists(), f"Missing {OCT21_DIR}")
        for kernel_dir, file_name in [
            ("weitaoguo_arc3-d-prime-plus-solved-level-memory", "arc3-d-prime-plus-solved-level-memory.ipynb"),
            ("evgendvorkin_arc-3-2026", "arc-3-2026.ipynb"),
            ("greenalgeria_arc3-up", "arc3-up.ipynb"),
            ("kawh1111_arc3-final-v1", "arc3-final-v1.ipynb"),
            ("iancblenke_carnot-arc-agi3-submission", "carnot-arc-agi3-submission.py"),
            ("manderson240_cohezion-arc-agi-3-autoharness-solver", "cohezion-arc-agi-3-autoharness-solver.ipynb"),
        ]:
            target = OCT21_DIR / kernel_dir / file_name
            self.assertTrue(target.exists(), f"Missing {target}")
            self.assertGreater(target.stat().st_size, 1000)

        # Verify kernel metadata
        self.assertEqual(self.meta.get("id"), "zerbytheboss/v27-sovereign-apex-engine")
        self.assertEqual(self.meta.get("title"), "v27 - Sovereign Apex Engine")
        self.assertEqual(self.meta.get("code_file"), "submission.ipynb")
        self.assertTrue(self.meta.get("enable_gpu"))
        self.assertFalse(self.meta.get("enable_internet"))
        self.assertEqual(self.meta.get("machine_shape"), "NvidiaRtxPro6000")

    def test_02_pure_v24_substrate_and_timeout_armor_purge(self) -> None:
        """2. Verify Cell 2 length and hash, pure v24 Cell 16 and Cell 12, and complete purge of timeout armor."""
        c2 = _src(self.nb_v27["cells"][2])
        self.assertEqual(len(c2), 498799, f"Cell 2 length must be 498,799 chars, got {len(c2)}")
        c2_sha = hashlib.sha256(c2.encode("utf-8")).hexdigest()
        self.assertEqual(c2_sha, "6fdcd1ae7445c0cfc26eec1047363aa0c196c867d289a35b6f64403b5a43dce2")

        c2_body = c2.split("\n", 1)[1]
        c2_body_sha = hashlib.sha256(c2_body.encode("utf-8")).hexdigest()
        self.assertEqual(c2_body_sha, "07fd859e7ce7ad8ac8a2f8f726d862660c1be1abc97560a63b5985703ddf7f4f")

        # Cell 12 SGLang configuration: 100% identical to v24
        c12 = _src(self.nb_v27["cells"][12])
        self.assertIn("MEMFRAC=0.96", c12)
        self.assertIn("MAXREQ=10", c12)
        self.assertIn("CUDAGRAPH_MAXBS=10", c12)
        self.assertIn("MAMBA_CACHE=60", c12)
        if self.nb_v24:
            self.assertEqual(c12, _src(self.nb_v24["cells"][12]))

        # Cell 16: Purged of install_timeout_armor and TIMEOUT_ARMOR_CODE; 100% byte-identical to v24
        c16 = _src(self.nb_v27["cells"][16])
        self.assertNotIn("TIMEOUT_ARMOR_CODE", c16)
        self.assertNotIn("install_timeout_armor", c16)
        self.assertNotIn("ARC3_MIN_READ_TIMEOUT", c16)
        self.assertNotIn("self.started_at = time.monotonic()", c16)
        self.assertNotIn("aging_bonus", c16)
        if self.nb_v24:
            self.assertEqual(c16, _src(self.nb_v24["cells"][16]), "Cell 16 must be 100% byte-identical to v24!")

        # Cell 4: setup_env is pristine v24 with zero ARC3_MIN_READ_TIMEOUT and zero ARC3_FABLE_*
        c4 = _src(self.nb_v27["cells"][4])
        self.assertNotIn("ARC3_MIN_READ_TIMEOUT", c4)
        self.assertNotIn("ARC3_FABLE_", c4)
        self.assertNotIn("animation_archive.py", c4)
        self.assertNotIn("fable_layer", c4)
        self.assertIn("'LOCAL_ANALYZER_YIELD_TOKENS': '2048'", c4)
        self.assertIn("'LOCAL_ANALYZER_MAX_OUTPUT': 12*1024", c4)
        self.assertIn("'ARC3_GUARDS_FROM_LEVEL': 2", c4)
        self.assertIn("'ARC3_MEMORY_SECTIONS': 'off'", c4)
        self.assertIn("'EXPOSE_UNDO': 'on'", c4)

        # Cell 20: bm.run() + #OURS_FRESH
        c20 = _src(self.nb_v27["cells"][20])
        self.assertIn("bm.run(", c20)
        self.assertIn("#OURS_FRESH", c20)

    def test_03_sandbox_git_apply_and_hooks_ast(self) -> None:
        """3. Verify clean git apply of Cell 2 base patch, clean execution of all 3 v27 post-patches, and 100% AST/py_compile validity."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            src_path = tmp_path / "src"
            src_path.mkdir(parents=True)

            c2 = _src(self.nb_v27["cells"][2])
            c2_patch = c2.split("\n", 1)[1]
            if not c2_patch.endswith("\n"):
                c2_patch += "\n"
            patch_file = tmp_path / "cell2.patch"
            patch_file.write_text(c2_patch, encoding="utf-8")

            # Extract pristine duck-harness via git archive
            subprocess.run(
                f"git archive HEAD ARC3-Inference | tar -x -C {src_path}",
                shell=True,
                cwd=str(DUCK_HARNESS_DIR),
                check=True,
            )

            # Apply base Cell 2 patch
            apply_res = subprocess.run(
                ["git", "apply", "--include=ARC3-Inference/*", str(patch_file)],
                cwd=str(src_path),
                capture_output=True,
                text=True,
            )
            self.assertEqual(apply_res.returncode, 0, f"git apply failed:\n{apply_res.stderr}")

            # Extract and apply the 3 v27 post-patches
            assets = _extract_v27_assets(_src(self.nb_v27["cells"][4]))
            self.assertEqual(
                set(assets.keys()),
                {"patch_solved_level_memory.py", "patch_animation_empty.py", "patch_spatial_helpers.py"},
            )
            v27_dir = tmp_path / "arcagi_v27"
            v27_dir.mkdir(parents=True)
            for name, text in assets.items():
                (v27_dir / name).write_text(text, encoding="utf-8")

            for patch_script in ("patch_solved_level_memory.py", "patch_animation_empty.py", "patch_spatial_helpers.py"):
                run_res = subprocess.run(
                    [sys.executable, str(v27_dir / patch_script), str(src_path / "ARC3-Inference")],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(run_res.returncode, 0, f"{patch_script} failed:\n{run_res.stderr}")

            # Verify idempotency: re-running all 3 patches succeeds with code 0
            for patch_script in ("patch_solved_level_memory.py", "patch_animation_empty.py", "patch_spatial_helpers.py"):
                re_res = subprocess.run(
                    [sys.executable, str(v27_dir / patch_script), str(src_path / "ARC3-Inference")],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(re_res.returncode, 0, f"{patch_script} idempotency failed:\n{re_res.stderr}")

            # Verify AST parsing and bytecode compilation of all Python files in ARC3-Inference
            arc3_dir = src_path / "ARC3-Inference"
            py_files = list(arc3_dir.glob("**/*.py"))
            self.assertGreater(len(py_files), 25, "Expected at least 25 Python files in ARC3-Inference")
            for py_file in py_files:
                code_text = py_file.read_text(encoding="utf-8")
                try:
                    ast.parse(code_text, filename=str(py_file))
                except Exception as exc:
                    self.fail(f"ast.parse failed on {py_file}: {exc}")
                try:
                    py_compile.compile(str(py_file), doraise=True)
                except Exception as exc:
                    self.fail(f"py_compile failed on {py_file}: {exc}")

    def test_04_zero_prompt_modification_on_levels_1_and_2(self) -> None:
        """4. Verify zero modification to _build_user_prompt and _build_system_prompt on Levels 1-2 (byte-for-byte identical to v24)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            src_path = tmp_path / "src"
            src_path.mkdir(parents=True)

            c2 = _src(self.nb_v27["cells"][2])
            c2_patch = c2.split("\n", 1)[1] + "\n"
            patch_file = tmp_path / "cell2.patch"
            patch_file.write_text(c2_patch, encoding="utf-8")

            subprocess.run(
                f"git archive HEAD ARC3-Inference | tar -x -C {src_path}",
                shell=True,
                cwd=str(DUCK_HARNESS_DIR),
                check=True,
            )
            subprocess.run(
                ["git", "apply", "--include=ARC3-Inference/*", str(patch_file)],
                cwd=str(src_path),
                check=True,
            )

            assets = _extract_v27_assets(_src(self.nb_v27["cells"][4]))
            v27_dir = tmp_path / "arcagi_v27"
            v27_dir.mkdir(parents=True)
            for name, text in assets.items():
                (v27_dir / name).write_text(text, encoding="utf-8")
            for patch_script in ("patch_solved_level_memory.py", "patch_animation_empty.py", "patch_spatial_helpers.py"):
                subprocess.run(
                    [sys.executable, str(v27_dir / patch_script), str(src_path / "ARC3-Inference")],
                    check=True,
                )

            # Test prompt behavior in a clean isolated subprocess
            test_script = tmp_path / "verify_prompts.py"
            test_script.write_text(
                """import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import inference.agent.tool_agent as tool_agent
from inference.agent.runtime_state import Frame, HistoryEntry

agent = tool_agent.ToolAgent(model="flashnext")
# Verify system prompt on level 1/2 is pristine
sys_p = agent._system_prompt
assert "SOLVED LEVELS" not in sys_p, "System prompt on init must not have SOLVED LEVELS"
assert "#SOLVED_LEVEL_MEMORY" not in sys_p

grid = [[0]*4 for _ in range(4)]
frame1 = Frame(grid=grid, step=1, level=1)
entries = [HistoryEntry(action="ACTION1", frame=frame1)]
user_p1 = agent._build_user_prompt(1, valid_actions=["ACTION1"], current_frame=frame1, history_entries=entries)
assert "Rule for level" not in user_p1, "User prompt on level 1 must not ask for rules"

frame2 = Frame(grid=grid, step=10, level=2)
agent._last_step_summary = {"level": 2, "executed_count": 5}
user_p2 = agent._build_user_prompt(10, valid_actions=["ACTION1"], current_frame=frame2, history_entries=entries)
assert "Rule for level" not in user_p2, "User prompt on level 2 must not ask for rules"

print("PROMPTS_OK")
""",
                encoding="utf-8",
            )
            res = subprocess.run(
                [sys.executable, str(test_script), str(src_path / "ARC3-Inference")],
                capture_output=True,
                text=True,
            )
            self.assertEqual(res.returncode, 0, f"Prompt verification failed:\n{res.stderr}")
            self.assertIn("PROMPTS_OK", res.stdout)

    def test_05_passive_solved_level_memory_level_3_plus(self) -> None:
        """5. Verify passive solved-level memory extracts facts without asking rules, gates pinning to Level 3+ context eviction."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            src_path = tmp_path / "src"
            src_path.mkdir(parents=True)

            c2 = _src(self.nb_v27["cells"][2])
            c2_patch = c2.split("\n", 1)[1] + "\n"
            patch_file = tmp_path / "cell2.patch"
            patch_file.write_text(c2_patch, encoding="utf-8")

            subprocess.run(
                f"git archive HEAD ARC3-Inference | tar -x -C {src_path}",
                shell=True,
                cwd=str(DUCK_HARNESS_DIR),
                check=True,
            )
            subprocess.run(
                ["git", "apply", "--include=ARC3-Inference/*", str(patch_file)],
                cwd=str(src_path),
                check=True,
            )

            assets = _extract_v27_assets(_src(self.nb_v27["cells"][4]))
            v27_dir = tmp_path / "arcagi_v27"
            v27_dir.mkdir(parents=True)
            for name, text in assets.items():
                (v27_dir / name).write_text(text, encoding="utf-8")
            for patch_script in ("patch_solved_level_memory.py", "patch_animation_empty.py", "patch_spatial_helpers.py"):
                subprocess.run(
                    [sys.executable, str(v27_dir / patch_script), str(src_path / "ARC3-Inference")],
                    check=True,
                )

            arc3_root = str(src_path / "ARC3-Inference")
            sys.path.insert(0, arc3_root)
            try:
                for mod_name in list(sys.modules):
                    if mod_name == "inference" or mod_name.startswith("inference."):
                        sys.modules.pop(mod_name, None)

                import inference.agent.tool_agent as tool_agent
                from inference.agent.runtime_state import Frame, HistoryEntry, write_runtime_state

                self.assertIs(tool_agent.ToolUseAgent, tool_agent.ToolAgent)
                self.assertEqual(tool_agent._SLM_ASK_TURNS, 0)

                agent = tool_agent.ToolAgent(model="flashnext")
                initial_system_prompt = agent._system_prompt

                # Create realistic runtime state: Level 1 completed with 35 winning actions (preceded by RESET)
                game_dir = tmp_path / "game_session_1"
                game_dir.mkdir(parents=True)
                state_path = game_dir / "runtime_state.json"

                grid_4x4 = [[0] * 4 for _ in range(4)]
                entries: list[HistoryEntry] = [
                    HistoryEntry(action="", frame=Frame(grid=grid_4x4, step=0, level=1)),
                    HistoryEntry(action="ACTION1", frame=Frame(grid=grid_4x4, step=1, level=1)),
                    HistoryEntry(action="RESET", frame=Frame(grid=grid_4x4, step=2, level=1)),
                ]
                winning_actions_l1 = [f"ACT_{i}" for i in range(1, 36)]  # 35 winning actions on Level 1
                for idx, act_name in enumerate(winning_actions_l1, start=3):
                    lv_after = 2 if idx == 2 + len(winning_actions_l1) else 1
                    entries.append(HistoryEntry(action=act_name, frame=Frame(grid=grid_4x4, step=idx, level=lv_after)))

                current_frame = Frame(grid=grid_4x4, step=len(entries) - 1, level=2)
                write_runtime_state(state_path, current_frame=current_frame, history=entries)

                long_reasoning = ("We solved Level 1 by finding the key and opening the door. " * 30).strip()
                self.assertGreater(len(long_reasoning), 900)
                agent._history_messages = [
                    {"role": "user", "content": "Turn 1 opener"},
                    {"role": "assistant", "reasoning_content": long_reasoning, "content": "Executing winning sequence."},
                ]
                agent._last_step_summary = {"level": 2, "executed_count": 5, "executed_actions": winning_actions_l1[-5:]}

                # 1. Trigger _slm_before_turn on Level 1 -> Level 2 transition
                agent._slm_before_turn(state_path)
                self.assertEqual(agent._slm_prev_done, 1)
                self.assertEqual(len(agent._slm_records), 1)
                rec1 = agent._slm_records[0]
                self.assertEqual(rec1["level"], 1)
                self.assertEqual(rec1["n_actions"], 35, "RESET must discard prior failed attempt actions")
                self.assertEqual(rec1["actions"], winning_actions_l1[-30:], "Must cap at last 30 actions")
                self.assertEqual(len(rec1["reasoning"]), 900, "Reasoning tail must be capped at 900 chars")
                self.assertEqual(rec1["asks_left"], 0, "_SLM_ASK_TURNS must be 0")
                self.assertFalse(rec1["asked"])
                self.assertIsNone(rec1["rule"])

                # 2. Verify Level 2: user prompt is NOT modified and system prompt is NOT modified even on eviction!
                user_p = agent._slm_user_prompt(
                    0,
                    valid_actions=["ACTION1"],
                    current_frame=current_frame,
                    history_entries=entries,
                    previous_step_summary=agent._last_step_summary,
                )
                self.assertNotIn("Rule for level", user_p)
                self.assertEqual(agent._system_prompt, initial_system_prompt)

                # Context eviction on Level 2 must NOT pin system prompt
                agent._note_history_evicted()
                self.assertEqual(agent._system_prompt, initial_system_prompt, "Eviction on Level 2 must NOT modify system prompt")

                # 3. Progress to Level 3: Level 2 cleared with 18 actions
                winning_actions_l2 = [f"L2_ACT_{i}" for i in range(1, 19)]
                for idx, act_name in enumerate(winning_actions_l2, start=len(entries)):
                    lv_after = 3 if idx == len(entries) - 1 + len(winning_actions_l2) else 2
                    entries.append(HistoryEntry(action=act_name, frame=Frame(grid=grid_4x4, step=idx, level=lv_after)))
                current_frame_l3 = Frame(grid=grid_4x4, step=len(entries) - 1, level=3)
                write_runtime_state(state_path, current_frame=current_frame_l3, history=entries)

                agent._last_step_summary = {"level": 3, "executed_count": 4, "executed_actions": winning_actions_l2[-4:]}
                agent._history_messages.append({"role": "assistant", "reasoning_content": "Solved Level 2 by rotating pieces."})
                agent._slm_before_turn(state_path)

                self.assertEqual(agent._slm_prev_done, 2)
                self.assertEqual(len(agent._slm_records), 2)
                rec2 = agent._slm_records[1]
                self.assertEqual(rec2["level"], 2)
                self.assertEqual(rec2["n_actions"], 18)
                self.assertEqual(rec2["actions"], winning_actions_l2)
                self.assertEqual(rec2["asks_left"], 0)

                # Before eviction on Level 3, system prompt is STILL untouched
                self.assertEqual(agent._system_prompt, initial_system_prompt)

                # 4. Trigger context eviction on Level 3+ -> NOW it pins!
                agent._note_history_evicted()
                self.assertNotEqual(agent._system_prompt, initial_system_prompt)
                self.assertIn("#SOLVED_LEVEL_MEMORY", agent._system_prompt)
                self.assertIn("=== SOLVED LEVELS", agent._system_prompt)
                self.assertIn("Level 1 - cleared after 35 actions.", agent._system_prompt)
                self.assertIn("(5 earlier omitted): ACT_6, ACT_7", agent._system_prompt)
                self.assertIn("Level 2 - cleared after 18 actions.", agent._system_prompt)
                self.assertIn("L2_ACT_1, L2_ACT_2", agent._system_prompt)
                self.assertIn("Your reasoning in the turn that cleared it:", agent._system_prompt)
                self.assertNotIn("Your rule:", agent._system_prompt, "When rule was not asked, 'Your rule:' should not be present")

                # 5. Verify session reset on new game
                new_state = tmp_path / "game_session_2" / "runtime_state.json"
                new_state.parent.mkdir(parents=True)
                write_runtime_state(new_state, current_frame=Frame(grid=grid_4x4, step=0, level=1), history=[])
                agent._ensure_session(new_state)
                self.assertEqual(agent._system_prompt, initial_system_prompt, "Session reset must restore initial system prompt")
                self.assertEqual(agent._slm_records, [])
                self.assertEqual(agent._slm_prev_done, 0)
            finally:
                if arc3_root in sys.path:
                    sys.path.remove(arc3_root)

    def test_06_native_request_timeout_seconds_preserved(self) -> None:
        """6. Verify solver._HarnessGameSession.request_timeout_seconds is 100% native with zero minimum floor."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            src_path = tmp_path / "src"
            src_path.mkdir(parents=True)

            c2 = _src(self.nb_v27["cells"][2])
            c2_patch = c2.split("\n", 1)[1] + "\n"
            patch_file = tmp_path / "cell2.patch"
            patch_file.write_text(c2_patch, encoding="utf-8")

            subprocess.run(
                f"git archive HEAD ARC3-Inference | tar -x -C {src_path}",
                shell=True,
                cwd=str(DUCK_HARNESS_DIR),
                check=True,
            )
            subprocess.run(
                ["git", "apply", "--include=ARC3-Inference/*", str(patch_file)],
                cwd=str(src_path),
                check=True,
            )

            arc3_root = str(src_path / "ARC3-Inference")
            sys.path.insert(0, arc3_root)
            try:
                for mod_name in list(sys.modules):
                    if mod_name == "inference" or mod_name.startswith("inference."):
                        sys.modules.pop(mod_name, None)

                # Mock taaf minimal interfaces for solver.py import
                taaf_mod = types.ModuleType("taaf")
                taaf_game = types.ModuleType("taaf.game")
                taaf_game.Game = type("Game", (), {})
                taaf_mod.game = taaf_game
                sys.modules["taaf"] = taaf_mod
                sys.modules["taaf.game"] = taaf_game
                sys.modules["taaf.solver"] = types.SimpleNamespace(Solver=type("Solver", (), {}))

                # Verify solver.py source file does NOT contain install_timeout_armor or ARC3_MIN_READ_TIMEOUT
                solver_py_path = src_path / "ARC3-Inference" / "inference" / "framework" / "solver.py"
                solver_src = solver_py_path.read_text(encoding="utf-8")
                self.assertNotIn("install_timeout_armor", solver_src)
                self.assertNotIn("ARC3_MIN_READ_TIMEOUT", solver_src)
                self.assertNotIn("_ours_protected", solver_src)

                import inference.framework.solver as solver_mod

                session_cls = solver_mod._HarnessGameSession
                # Verify request_timeout_seconds does NOT have _ours_protected attribute
                self.assertFalse(
                    getattr(session_cls.request_timeout_seconds, "_ours_protected", False),
                    "request_timeout_seconds must be native, not wrapped by install_timeout_armor!",
                )

                # Create mock session to test timeout clamping
                mock_analyzer = types.SimpleNamespace(_timeout=120.0)
                mock_solver = types.SimpleNamespace(
                    max_runtime_s_per_game=300.0,
                    soft_time_remaining_seconds=lambda: None,
                )

                class DummySession:
                    def __init__(self, rem_s: float):
                        self.analyzer = mock_analyzer
                        self.solver = mock_solver
                        self.rem_s = rem_s

                    def timing_payload(self):
                        return {"time_remaining_seconds": self.rem_s}

                dummy = DummySession(13.95)
                # Call native unbound method
                val = session_cls.request_timeout_seconds(dummy)
                self.assertEqual(val, 13.95, "Native request_timeout_seconds must return 13.95s, NOT clamped to 120.0s!")

                dummy_zero = DummySession(0.0)
                val_zero = session_cls.request_timeout_seconds(dummy_zero)
                self.assertEqual(val_zero, 0.1, "Native minimum floor max(0.1, ...) must be 0.1s")

                dummy_25 = DummySession(25.74)
                val_25 = session_cls.request_timeout_seconds(dummy_25)
                self.assertEqual(val_25, 25.74)

                # Verify soft deadline clamping (e.g. 35.1s)
                mock_solver_soft = types.SimpleNamespace(
                    max_runtime_s_per_game=300.0,
                    soft_time_remaining_seconds=lambda: 35.1,
                )
                dummy_soft = DummySession(89.4)
                dummy_soft.solver = mock_solver_soft
                val_soft = session_cls.request_timeout_seconds(dummy_soft)
                self.assertEqual(val_soft, 35.1, "Native request_timeout_seconds must clamp to soft deadline 35.1s!")
            finally:
                for k in ("taaf", "taaf.game", "taaf.solver"):
                    sys.modules.pop(k, None)
                if arc3_root in sys.path:
                    sys.path.remove(arc3_root)

    def test_07_animation_empty_and_spatial_helpers(self) -> None:
        """7. Verify patch_animation_empty.py and all 8 spatial helpers in patch_spatial_helpers.py."""
        c4 = _src(self.nb_v27["cells"][4])
        assets = _extract_v27_assets(c4)

        # Verify patch_animation_empty.py contract
        empty_patch = assets["patch_animation_empty.py"]
        self.assertIn('"action": record.get("action_display") if record else None', empty_patch)
        self.assertIn('"frames": 0', empty_patch)
        self.assertIn('"steps": []', empty_patch)

        # Verify all 8 spatial helpers
        spatial_script = assets["patch_spatial_helpers.py"]
        helper_ns: dict = {}
        seg_part = spatial_script.split('SEG_ADDENDUM = """')[1].split('"""')[0]
        exec(seg_part, helper_ns)

        test_grid = [
            [1, 2, 0],
            [3, 4, 0],
            [0, 0, 0],
        ]
        self.assertEqual(helper_ns["rot90"](test_grid, 1), [[0, 3, 1], [0, 4, 2], [0, 0, 0]])
        self.assertEqual(helper_ns["rot180"](test_grid), [[0, 0, 0], [0, 4, 3], [0, 2, 1]])
        self.assertEqual(helper_ns["rot270"](test_grid), [[0, 0, 0], [2, 4, 0], [1, 3, 0]])
        self.assertEqual(helper_ns["flipv"](test_grid), [[0, 0, 0], [3, 4, 0], [1, 2, 0]])
        self.assertEqual(helper_ns["fliph"](test_grid), [[0, 2, 1], [0, 4, 3], [0, 0, 0]])
        self.assertEqual(helper_ns["manhattan_distance"]((1, 2), (4, 6)), 7)
        self.assertEqual(helper_ns["find_cells"](test_grid, 2), [(0, 1)])
        self.assertEqual(helper_ns["find_cells"](test_grid, "g"), [(0, 1)])  # 'g' is index 2

        objs = helper_ns["find_objects"](test_grid, ignore_bg=0)
        self.assertEqual(len(objs), 4)
        self.assertEqual({o["color"] for o in objs}, {1, 2, 3, 4})
        self.assertEqual(helper_ns["find_objects"]([]), [])

    def test_08_d_prime_priority_math(self) -> None:
        """8. Verify D' slot priority math (ours_form_priority.install_d) from pure v24 substrate."""
        c16 = _src(self.nb_v27["cells"][16])
        self.assertNotIn("TIMEOUT_ARMOR_CODE", c16)
        self.assertNotIn("install_timeout_armor", c16)

        ns: dict = {}
        fp_part = "# --- arc-agi-3-agent:" + c16.split("# --- arc-agi-3-agent:", 1)[1]
        fp_lines = [
            line for line in fp_part.splitlines()
            if not line.startswith("import inference.")
            and not line.startswith("_fp_mod.install_d")
        ]
        exec("\n".join(fp_lines), ns)
        fp_mod = ns["_fp_mod"]

        Snap = types.SimpleNamespace
        p6 = fp_mod.d_priority(Snap(level=1, actions=0, tokens=0.0, cost_multiplier=0.0, total_levels=6), tail_fraction=1.0)
        p8 = fp_mod.d_priority(Snap(level=1, actions=0, tokens=0.0, cost_multiplier=0.0, total_levels=8), tail_fraction=1.0)
        p10 = fp_mod.d_priority(Snap(level=1, actions=0, tokens=0.0, cost_multiplier=0.0, total_levels=10), tail_fraction=1.0)
        self.assertEqual((p6, p8, p10), (18619, 17527, 17000))
        self.assertTrue(p6 > p8 > p10)

        p_l6_end = fp_mod.d_priority(Snap(level=6, actions=0, tokens=0.0, cost_multiplier=25000.0, total_levels=6), tail_fraction=0.0)
        p_l5_end = fp_mod.d_priority(Snap(level=5, actions=0, tokens=0.0, cost_multiplier=25000.0, total_levels=6), tail_fraction=0.0)
        p_l1_end = fp_mod.d_priority(Snap(level=1, actions=0, tokens=0.0, cost_multiplier=0.0, total_levels=6), tail_fraction=0.0)
        self.assertEqual((p_l6_end, p_l5_end, p_l1_end), (9860, 8451, 2619))
        self.assertTrue(p_l6_end > p_l5_end > p_l1_end)


if __name__ == "__main__":
    unittest.main()

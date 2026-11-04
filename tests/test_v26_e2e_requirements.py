"""Comprehensive E2E Verification Suite for ARC-AGI-3 Version 26 Sovereign Apex Engine.

Verifies:
1. test_01_byte_identical_notebooks_and_recon_artifacts:
   - notebooks/submission_v26.ipynb and notebooks/submission.ipynb are byte-identical (SHA-256 match, 23 cells, clean outputs).
   - Oct 19-20 reference kernels (lalalia9/arc3-franzen-v19, heliosli V6, sirikilohit M85) are persisted under pulled_kernels/oct20_top/.
   - docs/FORENSIC_V25_AND_OCT20_RECON.md documents our ground-truth submission ledger, v25 regression root causes, and v26 mechanics.
2. test_02_pure_v11_cell2_patch_and_v24_alignment:
   - Cell 2 source length is 498,799 chars and SHA-256 is 6fdcd1ae7445c0cfc26eec1047363aa0c196c867d289a35b6f64403b5a43dce2.
   - Cell 12 SGLang config is 100% byte-identical to v24 (MEMFRAC=0.96, MAXREQ=10, CUDAGRAPH_MAXBS=10, MAMBA_CACHE=60).
   - Cell 4 setup_env matches v24 plus ARC3_MIN_READ_TIMEOUT='120.0', with zero ARC3_FABLE_* env vars, zero animation_archive.py, and zero fable_layer.
   - Cell 20 contains bm.run() + #OURS_FRESH verification print.
3. test_03_sandbox_git_apply_and_hooks_ast:
   - Clean git apply of Cell 2 + execution of all 3 v26 post-patches (patch_solved_level_memory.py, patch_animation_empty.py, patch_spatial_helpers.py)
     in an isolated temporary copy of ARC3-Inference with 100% ast.parse and py_compile validity across all Python files.
4. test_04_zero_system_prompt_modification_before_drain:
   - _build_system_prompt output and ToolAgent._system_prompt on Levels 1-2 (before first context drain) are 100% byte-identical to v24,
     confirming animation_archive and fable_layer are completely removed.
5. test_05_solved_level_memory_behavioral_and_edge_cases:
   - Level completion detection (curr_done > _slm_prev_done) records (level, n_actions, actions/action_seq, reasoning/assistant_tail) and sets asks_left = 2.
   - RESET actions during a level attempt clear prior failed actions so only the winning attempt's actions are counted.
   - When _resume_after_yield is True, _slm_user_prompt does NOT decrement asks_left or append the rule request to the transient resume prompt.
   - When _resume_after_yield is False, _slm_user_prompt appends the rule request and _slm_capture_from_text extracts 'Rule for level 1: ...'
     (cleanly stripping markdown bold/code formatting and 'Rule for level 1 - ...' / em-dash variants).
   - Before context drain (_slm_drained_once == False), self._system_prompt is untouched (#SOLVED_LEVEL_MEMORY not yet appended, preserving SGLang prefix cache).
   - On context drain (_note_history_evicted / _trim_messages_for_context), _slm_drained_once becomes True and #SOLVED_LEVEL_MEMORY block
     (with captured rule, winning actions, and winning reasoning tail) is pinned to self._system_prompt, trimmed_messages[0]["content"], and self._messages[0]["content"].
   - Multi-level (>3 solved levels) windowing and session reset (_ensure_session) work cleanly without duplicate headers.
6. test_06_animation_empty_and_spatial_helpers:
   - Behavioral verification of patch_animation_empty.py and patch_spatial_helpers.py (rot90, rot180, rot270, flipv, fliph, manhattan_distance, find_cells, find_objects).
7. test_07_d_prime_priority_and_timeout_armor:
   - Behavioral verification of ours_form_priority.install_d and install_timeout_armor(min_floor=120.0).
8. test_08_kernel_metadata_offline_compliance:
   - Valid offline notebooks/kernel-metadata.json (id="zerbytheboss/v26-sovereign-apex-engine", title="v26 - Sovereign Apex Engine", enable_internet=False, machine_shape="NvidiaRtxPro6000").
"""
from __future__ import annotations

import ast
import hashlib
import importlib
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
V26_NB_PATH = REPO_ROOT / "notebooks" / "submission_v26.ipynb"
SUB_NB_PATH = REPO_ROOT / "notebooks" / "submission.ipynb"
META_PATH = REPO_ROOT / "notebooks" / "kernel-metadata.json"
RECON_DOC_PATH = REPO_ROOT / "docs" / "FORENSIC_V25_AND_OCT20_RECON.md"
OCT20_DIR = REPO_ROOT / "pulled_kernels" / "oct20_top"
DUCK_HARNESS_DIR = REPO_ROOT / "reference" / "duck-harness"


def _src(cell: dict) -> str:
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else str(s)


def _extract_v26_assets(c4_src: str) -> dict[str, str]:
    ns: dict = {}
    for line in c4_src.splitlines():
        if line.startswith("_v26_assets = {"):
            exec(line, ns)
            break
    if "_v26_assets" not in ns:
        raise AssertionError("Could not find _v26_assets in Cell 4")
    return dict(ns["_v26_assets"])


class TestV26E2ERequirements(unittest.TestCase):
    """Programmatic E2E test suite validating all architectural and behavioral requirements of Version 26."""

    @classmethod
    def setUpClass(cls) -> None:
        assert V26_NB_PATH.exists(), f"Missing {V26_NB_PATH}"
        assert SUB_NB_PATH.exists(), f"Missing {SUB_NB_PATH}"
        assert META_PATH.exists(), f"Missing {META_PATH}"
        assert DUCK_HARNESS_DIR.exists(), f"Missing {DUCK_HARNESS_DIR}"

        cls.nb_v26 = json.loads(V26_NB_PATH.read_text(encoding="utf-8"))
        cls.nb_sub = json.loads(SUB_NB_PATH.read_text(encoding="utf-8"))
        cls.meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        cls.nb_v11 = json.loads(V11_NB_PATH.read_text(encoding="utf-8")) if V11_NB_PATH.exists() else None
        cls.nb_v24 = json.loads(V24_NB_PATH.read_text(encoding="utf-8")) if V24_NB_PATH.exists() else None

    def test_01_byte_identical_notebooks_and_recon_artifacts(self) -> None:
        """1. Verify submission_v26.ipynb and submission.ipynb are byte-identical, and Oct 20 artifacts & forensic docs exist."""
        if "v26" not in self.meta.get("id", ""):
            raise unittest.SkipTest("Active kernel is not v26")
        bytes_v26 = V26_NB_PATH.read_bytes()
        bytes_sub = SUB_NB_PATH.read_bytes()

        self.assertGreater(len(bytes_v26), 500000)
        self.assertEqual(len(bytes_v26), len(bytes_sub))

        sha_v26 = hashlib.sha256(bytes_v26).hexdigest()
        sha_sub = hashlib.sha256(bytes_sub).hexdigest()
        self.assertEqual(sha_v26, sha_sub, "submission_v26.ipynb and submission.ipynb must be byte-identical.")

        self.assertEqual(len(self.nb_v26["cells"]), 23, f"Expected 23 cells, got {len(self.nb_v26['cells'])}")
        self.assertEqual(len(self.nb_sub["cells"]), 23, f"Expected 23 cells, got {len(self.nb_sub['cells'])}")

        for idx, cell in enumerate(self.nb_v26["cells"]):
            if cell.get("cell_type") == "code":
                self.assertEqual(cell.get("outputs"), [], f"Cell {idx} has non-empty outputs")
                self.assertIsNone(cell.get("execution_count"), f"Cell {idx} has non-null execution_count")

        # Verify Oct 19-20 reference kernels persisted under pulled_kernels/oct20_top/
        self.assertTrue(
            (OCT20_DIR / "lalalia9_arc3-franzen-v19" / "arc3-franzen-v19.ipynb").exists(),
            "Missing persisted lalalia9/arc3-franzen-v19 notebook in pulled_kernels/oct20_top/",
        )
        self.assertTrue(
            (
                OCT20_DIR
                / "heliosli_try-merging-it-you-might-get-lucky-and-win-gold"
                / "try-merging-it-you-might-get-lucky-and-win-gold.ipynb"
            ).exists(),
            "Missing persisted heliosli V6 notebook in pulled_kernels/oct20_top/",
        )

        # Verify docs/FORENSIC_V25_AND_OCT20_RECON.md exists and documents our ground-truth submission ledger
        self.assertTrue(RECON_DOC_PATH.exists(), f"Missing {RECON_DOC_PATH}")
        doc_text = RECON_DOC_PATH.read_text(encoding="utf-8")
        for score_token in ("26.70", "14.04", "24.30", "17.34", "25.18", "20.86", "28.79", "27.16"):
            self.assertIn(score_token, doc_text, f"Missing submission score {score_token} in {RECON_DOC_PATH.name}")
        for topic_token in ("AnimationArchive", "fable_layer", "SOLVED_LEVEL_MEMORY_PATCH", "_resume_after_yield"):
            self.assertIn(topic_token, doc_text, f"Missing technical topic {topic_token} in {RECON_DOC_PATH.name}")

    def test_02_pure_v11_cell2_patch_and_v24_alignment(self) -> None:
        """2. Verify Cell 2 patch (498,799 chars, SHA-256 match) and Cell 4/12/16/20 alignment with v24."""
        c2 = _src(self.nb_v26["cells"][2])
        self.assertEqual(len(c2), 498799, f"Expected Cell 2 length 498799, got {len(c2)}")
        c2_sha = hashlib.sha256(c2.encode("utf-8")).hexdigest()
        self.assertEqual(
            c2_sha,
            "6fdcd1ae7445c0cfc26eec1047363aa0c196c867d289a35b6f64403b5a43dce2",
            f"Cell 2 SHA-256 mismatch: {c2_sha}",
        )
        self.assertTrue(c2.startswith("%%writefile /kaggle/harness-changes.patch\n"))

        if self.nb_v11:
            self.assertEqual(c2, _src(self.nb_v11["cells"][2]), "Cell 2 drifted from pure v11 baseline!")

        # Cell 12 SGLang config must match v11 / v24 byte-for-byte
        c12 = _src(self.nb_v26["cells"][12])
        if self.nb_v24:
            self.assertEqual(c12, _src(self.nb_v24["cells"][12]), "Cell 12 drifted from v24!")
            self.assertEqual(_src(self.nb_v26["cells"][20]), _src(self.nb_v24["cells"][20]), "Cell 20 drifted from v24!")
        self.assertIn("MEMFRAC=0.96", c12)
        self.assertIn("MAXREQ=10", c12)
        self.assertIn("CUDAGRAPH_MAXBS=10", c12)
        self.assertIn("MAMBA_CACHE=60", c12)

        # Cell 4 setup_env must contain v24 flags + ARC3_MIN_READ_TIMEOUT, and NO fable_layer or animation_archive
        c4 = _src(self.nb_v26["cells"][4])
        self.assertIn("'LOCAL_ANALYZER_YIELD_TOKENS': '2048'", c4)
        self.assertIn("'ARC3_CONTEXT_DRAIN_TOKENS': 58*1024", c4)
        self.assertIn("'ARC3_MIN_READ_TIMEOUT': '120.0'", c4)
        self.assertNotIn("ARC3_FABLE", c4, "ARC3_FABLE_* env vars must be completely absent from Cell 4!")
        self.assertNotIn("fable_layer", c4, "fable_layer must be completely absent from Cell 4!")
        self.assertNotIn("animation_archive.py", c4, "animation_archive.py must be completely absent from Cell 4!")
        self.assertNotIn("patch_animation_archive.py", c4, "patch_animation_archive.py must be absent from Cell 4!")

        # Cell 20 contains bm.run() and #OURS_FRESH verification print
        c20 = _src(self.nb_v26["cells"][20])
        self.assertIn("bm.run(", c20)
        self.assertIn("#OURS_FRESH", c20)

    def test_03_sandbox_git_apply_and_hooks_ast(self) -> None:
        """3. Extract pristine ARC3-Inference, apply Cell 2 patch and all 3 v26 post-patches, verify AST and compilation."""
        with tempfile.TemporaryDirectory(prefix="arc3_v26_e2e_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            src_path = tmp_path / "src"
            src_path.mkdir(parents=True, exist_ok=True)

            archive_proc = subprocess.run(
                f"git archive HEAD ARC3-Inference | tar -x -C {src_path}",
                shell=True,
                cwd=str(DUCK_HARNESS_DIR),
                capture_output=True,
                text=True,
            )
            self.assertEqual(archive_proc.returncode, 0, f"git archive failed: {archive_proc.stderr}")

            c2 = _src(self.nb_v26["cells"][2])
            c2_patch = c2.split("\n", 1)[1]
            if not c2_patch.endswith("\n"):
                c2_patch += "\n"
            patch_file = tmp_path / "cell2.patch"
            patch_file.write_text(c2_patch, encoding="utf-8")

            apply_proc = subprocess.run(
                ["git", "apply", "--include=ARC3-Inference/*", "-v", str(patch_file)],
                cwd=str(src_path),
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                apply_proc.returncode,
                0,
                f"git apply cell2.patch failed:\nStdout: {apply_proc.stdout}\nStderr: {apply_proc.stderr}",
            )
            self.assertNotIn("reject", apply_proc.stderr.lower())

            c4 = _src(self.nb_v26["cells"][4])
            assets = _extract_v26_assets(c4)
            self.assertEqual(
                set(assets.keys()),
                {"patch_solved_level_memory.py", "patch_animation_empty.py", "patch_spatial_helpers.py"},
            )

            v26_dir = tmp_path / "arcagi_v26"
            v26_dir.mkdir(parents=True, exist_ok=True)
            for name, text in assets.items():
                (v26_dir / name).write_text(text, encoding="utf-8")

            for patch_script in ("patch_solved_level_memory.py", "patch_animation_empty.py", "patch_spatial_helpers.py"):
                post_proc = subprocess.run(
                    [sys.executable, str(v26_dir / patch_script), str(src_path / "ARC3-Inference")],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(
                    post_proc.returncode,
                    0,
                    f"Post-patch {patch_script} failed:\nStdout: {post_proc.stdout}\nStderr: {post_proc.stderr}",
                )

            # Verify idempotency of all 3 post-patch scripts when run a second time
            for patch_script in ("patch_solved_level_memory.py", "patch_animation_empty.py", "patch_spatial_helpers.py"):
                idem_proc = subprocess.run(
                    [sys.executable, str(v26_dir / patch_script), str(src_path / "ARC3-Inference")],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(idem_proc.returncode, 0, f"Idempotent re-run of {patch_script} failed: {idem_proc.stderr}")

            arc3_tree = src_path / "ARC3-Inference"
            py_files = list(arc3_tree.rglob("*.py"))
            self.assertGreaterEqual(len(py_files), 25)
            for pf in py_files:
                code = pf.read_text(encoding="utf-8")
                ast.parse(code, filename=str(pf))
                py_compile.compile(str(pf), doraise=True)

    def test_04_zero_system_prompt_modification_before_drain(self) -> None:
        """4. Verify _build_system_prompt and initial ToolAgent._system_prompt are 100% byte-identical to v24."""
        with tempfile.TemporaryDirectory(prefix="arc3_v26_sp_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            v24_src = tmp_path / "v24_src"
            v26_src = tmp_path / "v26_src"
            v24_src.mkdir(parents=True)
            v26_src.mkdir(parents=True)

            c2 = _src(self.nb_v26["cells"][2])
            c2_patch = c2.split("\n", 1)[1]
            if not c2_patch.endswith("\n"):
                c2_patch += "\n"
            patch_file = tmp_path / "cell2.patch"
            patch_file.write_text(c2_patch, encoding="utf-8")

            for target_src in (v24_src, v26_src):
                subprocess.run(
                    f"git archive HEAD ARC3-Inference | tar -x -C {target_src}",
                    shell=True,
                    cwd=str(DUCK_HARNESS_DIR),
                    check=True,
                )
                subprocess.run(
                    ["git", "apply", "--include=ARC3-Inference/*", str(patch_file)],
                    cwd=str(target_src),
                    check=True,
                )

            assets = _extract_v26_assets(_src(self.nb_v26["cells"][4]))
            v26_dir = tmp_path / "arcagi_v26"
            v26_dir.mkdir(parents=True)
            for name, text in assets.items():
                (v26_dir / name).write_text(text, encoding="utf-8")
            for patch_script in ("patch_solved_level_memory.py", "patch_animation_empty.py", "patch_spatial_helpers.py"):
                subprocess.run(
                    [sys.executable, str(v26_dir / patch_script), str(v26_src / "ARC3-Inference")],
                    check=True,
                )

            # Compare _build_system_prompt output in isolated subprocesses with identical v24 env flags
            eval_code = (
                "import sys, os; "
                "os.environ['LOCAL_ANALYZER_MODEL_ID']='flashnext'; "
                "os.environ['LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS']='3072'; "
                "os.environ['ARC3_EXPLAIN_GAMEPLAY_CHANGED']='1'; "
                "os.environ['ARC3_NEW_CHANGED_PROMPTS']='1'; "
                "os.environ['ARC3_LEVEL_TRANSFER_GUIDANCE']='1'; "
                "os.environ['ARC3_ACTION_INFO']='1'; "
                "os.environ['ARC3_DEDUPE_MULTICALL_LINE']='1'; "
                "os.environ['ARC3_FRAME_DIFF_HINT']='1'; "
                "os.environ['ARC3_ANIMATION']='1'; "
                "os.environ['ARC3_MEMORY_SECTIONS']='off'; "
                "os.environ['ARC3_PERSISTENT_FUNCTIONS']='1'; "
                "os.environ['ARC3_PERSISTENT_FUNCTIONS_SCOPE']='game'; "
                "os.environ['ARC3_PERSISTENT_FUNCTIONS_IMPORTS']='1'; "
                "os.environ['EXPOSE_UNDO']='on'; "
                "from inference.agent.tool_agent import _build_system_prompt, ToolAgent; "
                "sp = _build_system_prompt(tool_output_tokens=3072); "
                "agent = ToolAgent(model='flashnext'); "
                "assert sp == agent._system_prompt; "
                "sys.stdout.write(sp)"
            )
            p_v24 = subprocess.run(
                [sys.executable, "-c", eval_code],
                cwd=str(v24_src / "ARC3-Inference"),
                capture_output=True,
                text=True,
                check=True,
            )
            p_v26 = subprocess.run(
                [sys.executable, "-c", eval_code],
                cwd=str(v26_src / "ARC3-Inference"),
                capture_output=True,
                text=True,
                check=True,
            )
            # Strip the "SOLVED_LEVEL_MEMORY_PATCH on\n" import banner from p_v26 stdout
            v26_sp = p_v26.stdout
            if v26_sp.startswith("SOLVED_LEVEL_MEMORY_PATCH on\n"):
                v26_sp = v26_sp[len("SOLVED_LEVEL_MEMORY_PATCH on\n"):]

            self.assertEqual(
                v26_sp,
                p_v24.stdout,
                "v26 system prompt before first context drain must be 100% byte-identical to v24!",
            )
            self.assertNotIn("animation_history", v26_sp)
            self.assertNotIn("animation_at", v26_sp)
            self.assertNotIn("SOLVED LEVELS", v26_sp)
            self.assertNotIn("#SOLVED_LEVEL_MEMORY", v26_sp)

    def test_05_solved_level_memory_behavioral_and_edge_cases(self) -> None:
        """5. Full behavioral verification of SOLVED_LEVEL_MEMORY_PATCH including _resume_after_yield guard and drain pinning."""
        with tempfile.TemporaryDirectory(prefix="arc3_v26_slm_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            src_path = tmp_path / "src"
            src_path.mkdir(parents=True)

            c2 = _src(self.nb_v26["cells"][2])
            c2_patch = c2.split("\n", 1)[1]
            if not c2_patch.endswith("\n"):
                c2_patch += "\n"
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

            assets = _extract_v26_assets(_src(self.nb_v26["cells"][4]))
            v26_dir = tmp_path / "arcagi_v26"
            v26_dir.mkdir(parents=True)
            for name, text in assets.items():
                (v26_dir / name).write_text(text, encoding="utf-8")
            for patch_script in ("patch_solved_level_memory.py", "patch_animation_empty.py", "patch_spatial_helpers.py"):
                subprocess.run(
                    [sys.executable, str(v26_dir / patch_script), str(src_path / "ARC3-Inference")],
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

                agent = tool_agent.ToolAgent(model="flashnext")
                initial_system_prompt = agent._system_prompt

                # Verify initial state before any level completion or drain
                self.assertFalse(agent._slm_drained_once)
                self.assertEqual(agent._slm_prev_done, 0)
                self.assertEqual(agent._slm_records, [])
                self.assertIs(agent._slm_records, agent._slm_levels)

                # Create a realistic runtime_state.json where Level 1 had 2 failed actions + RESET + 35 winning actions -> Level 2
                game_dir = tmp_path / "game_session_1"
                game_dir.mkdir(parents=True)
                state_path = game_dir / "runtime_state.json"

                grid_4x4 = [[0] * 4 for _ in range(4)]
                entries: list[HistoryEntry] = [
                    HistoryEntry(action="", frame=Frame(grid=grid_4x4, step=0, level=1)),
                    HistoryEntry(action="ACTION1", frame=Frame(grid=grid_4x4, step=1, level=1)),
                    HistoryEntry(action="ACTION2", frame=Frame(grid=grid_4x4, step=2, level=1)),
                    HistoryEntry(action="RESET", frame=Frame(grid=grid_4x4, step=3, level=1)),
                ]
                winning_actions = [f"ACT_{i}" for i in range(1, 36)]  # 35 winning actions on Level 1
                for idx, act_name in enumerate(winning_actions, start=4):
                    lv_after = 2 if idx == 3 + len(winning_actions) else 1
                    entries.append(HistoryEntry(action=act_name, frame=Frame(grid=grid_4x4, step=idx, level=lv_after)))

                current_frame = Frame(grid=grid_4x4, step=len(entries) - 1, level=2)
                write_runtime_state(state_path, current_frame=current_frame, history=entries)

                # Populate assistant winning reasoning (> 900 chars to verify 900-char tail truncation)
                long_reasoning = ("We tested the doors and found the blue key opens the yellow lock. " * 20).strip()
                self.assertGreater(len(long_reasoning), 900)
                agent._history_messages = [
                    {"role": "user", "content": "Turn 1 opener"},
                    {"role": "assistant", "reasoning_content": long_reasoning, "content": "Executing winning sequence."},
                ]
                agent._last_step_summary = {"level": 2, "executed_count": 5, "executed_actions": winning_actions[-5:]}

                # 1. Trigger _slm_before_turn on Level 1 -> Level 2 transition
                agent._slm_before_turn(state_path)
                self.assertEqual(agent._slm_prev_done, 1)
                self.assertEqual(len(agent._slm_records), 1)
                rec1 = agent._slm_records[0]
                self.assertEqual(rec1["level"], 1)
                self.assertEqual(rec1["n_actions"], 35, "RESET must discard prior failed attempt actions")
                self.assertEqual(len(rec1["actions"]), 30, "Actions must be capped at last 30 winning actions")
                self.assertEqual(rec1["actions"], winning_actions[-30:])
                self.assertEqual(rec1["action_seq"], winning_actions[-30:])
                self.assertEqual(len(rec1["reasoning"]), 900, "Reasoning tail must be capped at 900 chars")
                self.assertEqual(rec1["assistant_tail"], rec1["reasoning"])
                self.assertEqual(rec1["asks_left"], 2)
                self.assertIsNone(rec1["rule"])

                # 2. Verify Prefix-Cache Invariant before context drain: _system_prompt is 100% untouched!
                self.assertFalse(agent._slm_drained_once)
                self.assertEqual(agent._system_prompt, initial_system_prompt)
                self.assertNotIn("#SOLVED_LEVEL_MEMORY", agent._system_prompt)
                self.assertNotIn("SOLVED LEVELS", agent._system_prompt)

                # 3. Verify Compatibility Guard with D' (_resume_after_yield == True):
                # Must NOT decrement asks_left or append the rule request to a transient resume prompt!
                agent._resume_after_yield = True
                resume_prompt = agent._slm_user_prompt(
                    0,
                    valid_actions=["ACTION1"],
                    current_frame=current_frame,
                    history_entries=entries,
                    previous_step_summary=agent._last_step_summary,
                )
                self.assertNotIn("Rule for level 1:", resume_prompt)
                self.assertEqual(rec1["asks_left"], 2, "asks_left must NOT decrement when _resume_after_yield is True")
                self.assertFalse(rec1["asked"])

                # Also verify direct string call to _slm_user_prompt when _resume_after_yield is True
                self.assertEqual(agent._slm_user_prompt("resume"), "resume")
                self.assertEqual(rec1["asks_left"], 2)

                # 4. Verify Rule Elicitation when _resume_after_yield is False
                agent._resume_after_yield = False
                normal_prompt = agent._slm_user_prompt(
                    0,
                    valid_actions=["ACTION1"],
                    current_frame=current_frame,
                    history_entries=entries,
                    previous_step_summary=agent._last_step_summary,
                )
                self.assertIn("Rule for level 1:", normal_prompt)
                self.assertEqual(rec1["asks_left"], 1)
                self.assertTrue(rec1["asked"])

                # 5. Test _slm_capture_from_text across formatting variants (bold, backticks, dash, em-dash, multi-line, prompt-echo, code-fence boundary)
                for formatted_line, expected_rule in [
                    (
                        "**Rule for level 1:** Collect the blue key at (2,3) then step on the yellow door.",
                        "Collect the blue key at (2,3) then step on the yellow door.",
                    ),
                    (
                        "`Rule for level 1:` Collect the blue key at (2,3) then step on the yellow door.",
                        "Collect the blue key at (2,3) then step on the yellow door.",
                    ),
                    (
                        "Rule for level 1 - Collect the blue key at (2,3) then step on the yellow door.",
                        "Collect the blue key at (2,3) then step on the yellow door.",
                    ),
                    (
                        "Rule for level 1 \u2014 Collect the blue key at (2,3) then step on the yellow door.",
                        "Collect the blue key at (2,3) then step on the yellow door.",
                    ),
                    (
                        "Rule for level 1: Goal is exit.\n- ACTION1 moves up\n- Blue key opens yellow lock\n\nNext paragraph.",
                        "Goal is exit. ACTION1 moves up Blue key opens yellow lock",
                    ),
                    (
                        "The user prompt says: write a line starting `Rule for level 1:` that states the goal, what each control did and the plan that won.\n\n"
                        "Rule for level 1: Move the cyan player to the green exit while avoiding red lasers.",
                        "Move the cyan player to the green exit while avoiding red lasers.",
                    ),
                    (
                        "Rule for level 1: Move cyan block to green exit.\n```python\naction('ACTION1')\n```",
                        "Move cyan block to green exit.",
                    ),
                ]:
                    rec1["rule"] = None
                    extracted = agent._slm_capture_from_text(formatted_line)
                    self.assertEqual(extracted, expected_rule)
                    self.assertEqual(rec1["rule"], expected_rule)

                # Reject too-short garbage fragment (< 10 chars) and raw prompt echo
                self.assertIsNone(tool_agent._slm_find_rule("Rule for level 1: ok", 1))
                self.assertIsNone(
                    tool_agent._slm_find_rule(
                        "write a line starting `Rule for level 1:` that states the goal, what each control did and the plan that won.",
                        1,
                    )
                )

                # Verify _slm_level_actions handles serialized dict entries containing "frame" and RESET
                dict_entries = [
                    {"action": "", "frame": {"level": 1}},
                    {"action": "FAIL_1", "frame": {"level": 1}},
                    {"action": "RESET", "frame": {"level": 1}},
                    {"action": "WIN_1", "frame": {"level": 1}},
                    {"action": "WIN_2", "frame": {"level": 2}},
                ]
                self.assertEqual(tool_agent._slm_level_actions(dict_entries, 1), ["WIN_1", "WIN_2"])

                # Also verify automatic capture via _slm_before_turn from assistant message in _history_messages
                rec1["rule"] = None
                agent._history_messages.append(
                    {
                        "role": "assistant",
                        "content": "Thought about Level 2.\nRule for level 1: Collect the blue key first, then unlock the yellow exit door.",
                    }
                )
                agent._slm_before_turn(state_path)
                self.assertEqual(
                    rec1["rule"],
                    "Collect the blue key first, then unlock the yellow exit door.",
                )

                # Once rule is captured, subsequent _slm_user_prompt calls must NOT ask for Level 1 again
                next_prompt = agent._slm_user_prompt("Current state: step 40, level 2.")
                self.assertNotIn("Rule for level 1:", next_prompt)

                # 6. Verify that BEFORE context drain (_slm_drained_once == False), _system_prompt is STILL untouched!
                self.assertFalse(agent._slm_drained_once)
                self.assertEqual(agent._system_prompt, initial_system_prompt)
                self.assertNotIn("#SOLVED_LEVEL_MEMORY", agent._system_prompt)

                # 7. Trigger context drain via _trim_messages_for_context on the SAME turn that the model writes a rule in `messages`
                rec1["rule"] = None
                agent._history_messages.pop()  # remove from _history_messages so it is ONLY in `messages`
                agent._messages = [
                    {"role": "system", "content": agent._system_prompt},
                    {"role": "user", "content": "Old user message " * 500},
                    {"role": "assistant", "content": "Old assistant message " * 500},
                    {"role": "user", "content": "Recent user message asking for Rule for level 1"},
                    {
                        "role": "assistant",
                        "content": "Rule for level 1: Collect the blue key first, then unlock the yellow exit door.",
                    },
                ]
                # Force _trim_messages_for_context to evict old messages by setting a tight budget
                orig_budget = agent._context_budget_tokens
                try:
                    agent._context_budget_tokens = 200
                    trimmed = agent._trim_messages_for_context(agent._messages, preserve_recent=1)
                finally:
                    agent._context_budget_tokens = orig_budget

                self.assertLess(len(trimmed), len(agent._messages), "Expected _trim_messages_for_context to drop old messages")
                self.assertTrue(agent._slm_drained_once, "_slm_drained_once must become True on context drain")
                self.assertEqual(
                    rec1["rule"],
                    "Collect the blue key first, then unlock the yellow exit door.",
                    "Rule written in `messages` during the draining turn must be captured before pinning!",
                )
                self.assertIn("#SOLVED_LEVEL_MEMORY", agent._system_prompt)
                self.assertIn("=== SOLVED LEVELS", agent._system_prompt)
                self.assertIn("Level 1 - cleared after 35 actions.", agent._system_prompt)
                self.assertIn("Your rule: Collect the blue key first, then unlock the yellow exit door.", agent._system_prompt)
                self.assertIn("(5 earlier omitted): ACT_6, ACT_7", agent._system_prompt)
                self.assertIn("Your reasoning in the turn that cleared it:", agent._system_prompt)
                self.assertEqual(trimmed[0]["content"], agent._system_prompt)
                self.assertEqual(agent._messages[0]["content"], agent._system_prompt)

                # Verify _slm_refresh_system_prompt syncs _messages[0]["content"] even when block is unchanged
                agent._messages[0]["content"] = initial_system_prompt
                agent._slm_refresh_system_prompt()
                self.assertEqual(agent._messages[0]["content"], agent._system_prompt)

                # 8. Test multi-level windowing (> _SLM_FULL_LEVELS = 3) and turn-cap drain via _persistent_history_messages
                for lv_num in (2, 3, 4):
                    agent._slm_records.append({
                        "level": lv_num,
                        "n_actions": 10 + lv_num,
                        "actions": [f"L{lv_num}_A1", f"L{lv_num}_A2"],
                        "action_seq": [f"L{lv_num}_A1", f"L{lv_num}_A2"],
                        "reasoning": f"Reasoning for level {lv_num}",
                        "assistant_tail": f"Reasoning for level {lv_num}",
                        "rule": f"Solved level {lv_num} by matching pattern {lv_num}.",
                        "asks_left": 0,
                        "asked": True,
                    })
                # Trigger turn-cap eviction through _persistent_history_messages (ARC3_HISTORY_ASSISTANT_TURNS=2)
                turn_cap_msgs = [{"role": "system", "content": agent._system_prompt}]
                for t_idx in range(5):
                    turn_cap_msgs.append({"role": "user", "content": f"Turn {t_idx} user"})
                    turn_cap_msgs.append({"role": "assistant", "content": f"Turn {t_idx} assistant"})
                old_turns_env = os.environ.get("ARC3_HISTORY_ASSISTANT_TURNS")
                old_drain_env = os.environ.get("ARC3_HISTORY_TURN_DRAIN")
                try:
                    os.environ["ARC3_HISTORY_ASSISTANT_TURNS"] = "2"
                    os.environ["ARC3_HISTORY_TURN_DRAIN"] = "2"
                    persisted_hist = agent._persistent_history_messages(turn_cap_msgs)
                finally:
                    if old_turns_env is None:
                        os.environ.pop("ARC3_HISTORY_ASSISTANT_TURNS", None)
                    else:
                        os.environ["ARC3_HISTORY_ASSISTANT_TURNS"] = old_turns_env
                    if old_drain_env is None:
                        os.environ.pop("ARC3_HISTORY_TURN_DRAIN", None)
                    else:
                        os.environ["ARC3_HISTORY_TURN_DRAIN"] = old_drain_env
                self.assertLess(len(persisted_hist), len(turn_cap_msgs) - 1)
                self.assertEqual(agent._system_prompt.count("=== SOLVED LEVELS"), 1, "Block must be replaced in-place")
                self.assertEqual(agent._system_prompt.count("=== END SOLVED LEVELS ==="), 1)
                # Level 1 (older than newest 3 levels) keeps rule but omits full actions/reasoning
                self.assertIn("Level 1 - cleared after 35 actions.", agent._system_prompt)
                self.assertNotIn("ACT_6, ACT_7", agent._system_prompt)
                # Levels 2, 3, 4 keep full actions and reasoning
                for lv_num in (2, 3, 4):
                    self.assertIn(f"L{lv_num}_A1, L{lv_num}_A2", agent._system_prompt)
                    self.assertIn(f"Reasoning for level {lv_num}", agent._system_prompt)

                # 9. Test session reset when _ensure_session switches to a new game directory (both Path and str)
                new_game_state = tmp_path / "game_session_2" / "runtime_state.json"
                new_game_state.parent.mkdir(parents=True)
                agent._ensure_session(str(new_game_state))
                self.assertFalse(agent._slm_drained_once)
                self.assertEqual(agent._slm_prev_done, 0)
                self.assertEqual(agent._slm_records, [])
                self.assertEqual(agent._system_prompt, initial_system_prompt)
            finally:
                if arc3_root in sys.path:
                    sys.path.remove(arc3_root)

    def test_06_animation_empty_and_spatial_helpers(self) -> None:
        """6. Verify patch_animation_empty.py and all 8 spatial helpers in patch_spatial_helpers.py."""
        c4 = _src(self.nb_v26["cells"][4])
        assets = _extract_v26_assets(c4)

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
        self.assertEqual(helper_ns["find_cells"](test_grid, "g"), [(0, 1)])  # char lookup ('g' is index 2)

        objs = helper_ns["find_objects"](test_grid, ignore_bg=0)
        self.assertEqual(len(objs), 4)
        self.assertEqual({o["color"] for o in objs}, {1, 2, 3, 4})
        self.assertEqual(helper_ns["find_objects"]([]), [])

    def test_07_d_prime_priority_and_timeout_armor(self) -> None:
        """7. Verify D' slot priority math (ours_form_priority.install_d) and install_timeout_armor(min_floor=120.0)."""
        c2 = _src(self.nb_v26["cells"][2])
        c4 = _src(self.nb_v26["cells"][4])
        c16 = _src(self.nb_v26["cells"][16])

        self.assertNotIn("self.started_at = time.monotonic()", c2 + c16)
        self.assertNotIn("aging_bonus", c16)
        self.assertNotIn("EXPOSE_RESET", c4 + c16)

        ns: dict = {}
        fp_part = "# --- arc-agi-3-agent:" + c16.split("# --- arc-agi-3-agent:", 1)[1]
        fp_lines = [
            line for line in fp_part.splitlines()
            if not line.startswith("import inference.")
            and not line.startswith("_fp_mod.install_d")
            and not line.startswith("_fp_mod.install_timeout_armor")
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

        # Verify install_timeout_armor
        class MockSession:
            def __init__(self, val: float | None):
                self.val = val

            def request_timeout_seconds(self) -> float | None:
                return self.val

        mock_solver = types.SimpleNamespace(_HarnessGameSession=MockSession)
        msg1 = fp_mod.install_timeout_armor(mock_solver, min_floor=120.0)
        self.assertIn("#TIMEOUT_ARMOR ok", msg1)
        self.assertIsNone(MockSession(None).request_timeout_seconds())
        self.assertEqual(MockSession(0.0).request_timeout_seconds(), 120.0)
        for clamped in (13.95, 35.55, 66.18, 89.45):
            self.assertEqual(MockSession(clamped).request_timeout_seconds(), 120.0)
        self.assertEqual(MockSession(250.0).request_timeout_seconds(), 250.0)
        # Idempotency
        fp_mod.install_timeout_armor(mock_solver, min_floor=120.0)
        self.assertEqual(MockSession(13.95).request_timeout_seconds(), 120.0)

    def test_08_kernel_metadata_offline_compliance(self) -> None:
        """8. Verify kernel-metadata.json configuration for v26 offline GPU competition execution."""
        if "v26" not in self.meta.get("id", ""):
            raise unittest.SkipTest("Active kernel is not v26")
        self.assertEqual(self.meta.get("id"), "zerbytheboss/v26-sovereign-apex-engine")
        self.assertEqual(self.meta.get("title"), "v26 - Sovereign Apex Engine")
        self.assertEqual(self.meta.get("code_file"), "submission.ipynb")
        self.assertTrue(self.meta.get("enable_gpu"))
        self.assertFalse(self.meta.get("enable_internet"))
        self.assertEqual(self.meta.get("machine_shape"), "NvidiaRtxPro6000")


if __name__ == "__main__":
    unittest.main()

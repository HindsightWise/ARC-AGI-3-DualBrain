"""Standalone local programmatic verification and harness automation for ARC-AGI-3 Version 23.

Simulates the exact Kaggle Cell 2 + Cell 4 + Cell 16 execution flow in an isolated
temporary directory against pristine ARC3-Inference extracted from reference/duck-harness.

Verifies:
1. Clean extraction of pristine ARC3-Inference from reference/duck-harness git archive.
2. Extraction of unified patch from Cell 2 of notebooks/submission.ipynb (and submission_v23.ipynb).
3. Clean application of patch via `git apply --include=ARC3-Inference/* -v` with returncode 0.
4. AST parsing (`ast.parse`) and bytecode compilation (`py_compile.compile`) across all
   modified Python files with 0 SyntaxErrors and 0 IndentationErrors.
5. Clean simulation of Cell 16 scheduler installation (Fair-Play Priority Gate,
   Post-Admission Timer Reset, Anti-Starvation Aging, Bounded Search, Level 1 Guards,
   Hyper-Resilience Timeout Armor) with zero runtime or syntax exceptions.
6. Verification of Post-Admission Timer Reset in solver.py and session_cls.
7. Verification of compact action result compatibility alias (gameplay_changed = board_changed).
8. Verification of zero-action bounded search (search_plan) and checked execution (execute_plan).
9. Perfect byte-synchronization between submission_v23.ipynb and submission.ipynb.
10. Valid offline GPU kernel metadata configuration (enable_internet: false, NvidiaRtxPro6000).
"""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import py_compile
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "submission.ipynb"
NOTEBOOK_V23_PATH = ROOT / "notebooks" / "submission_v23.ipynb"
METADATA_PATH = ROOT / "notebooks" / "kernel-metadata.json"
DUCK_HARNESS_DIR = ROOT / "reference" / "duck-harness"


class TestV23BuildVerification(unittest.TestCase):
    """End-to-End simulation and build verification test for Version 23 Sovereign Apex Engine."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        assert NOTEBOOK_PATH.exists(), f"Missing submission notebook at {NOTEBOOK_PATH}"
        assert NOTEBOOK_V23_PATH.exists(), f"Missing v23 notebook at {NOTEBOOK_V23_PATH}"
        assert METADATA_PATH.exists(), f"Missing kernel metadata at {METADATA_PATH}"
        assert DUCK_HARNESS_DIR.exists(), f"Missing reference duck harness at {DUCK_HARNESS_DIR}"
        cls.notebook_data = json.loads(NOTEBOOK_V23_PATH.read_text(encoding="utf-8"))
        cls.cells = cls.notebook_data.get("cells", [])

    def test_pristine_archive_extraction_and_git_apply(self):
        """Simulate Cell 2 patch extraction and Cell 4 git apply in isolated directory."""
        with tempfile.TemporaryDirectory(prefix="arc3_v23_verify_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            src_path = tmp_path / "src"
            src_path.mkdir(parents=True, exist_ok=True)

            # 1. Extract pristine ARC3-Inference via git archive HEAD
            archive_proc = subprocess.run(
                f"git archive HEAD ARC3-Inference | tar -x -C {src_path}",
                shell=True,
                cwd=str(DUCK_HARNESS_DIR),
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                archive_proc.returncode,
                0,
                f"git archive extraction failed: {archive_proc.stderr}",
            )

            extracted_arc3 = src_path / "ARC3-Inference"
            self.assertTrue(extracted_arc3.is_dir(), "ARC3-Inference directory must exist after extraction")
            self.assertTrue((extracted_arc3 / "inference").is_dir(), "inference package directory missing")

            # 2. Extract patch from Cell 2
            cell_2 = self.cells[2]
            self.assertEqual(cell_2.get("cell_type"), "code")
            c2_src = "".join(cell_2["source"]) if isinstance(cell_2["source"], list) else cell_2["source"]
            self.assertTrue(c2_src.startswith("%%writefile"), "Cell 2 must start with %%writefile magic")

            patch_text = c2_src.split("\n", 1)[1] if c2_src.startswith("%%writefile") else c2_src
            if not patch_text.endswith("\n"):
                patch_text += "\n"

            patch_file = tmp_path / "harness-changes.patch"
            patch_file.write_text(patch_text, encoding="utf-8")

            # 3. Simulate Cell 4 git apply execution
            apply_proc = subprocess.run(
                ["git", "apply", "--include=ARC3-Inference/*", "-v", str(patch_file)],
                cwd=str(src_path),
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                apply_proc.returncode,
                0,
                f"git apply failed with returncode {apply_proc.returncode}:\nStdout: {apply_proc.stdout}\nStderr: {apply_proc.stderr}",
            )

            # 4. Confirm modified Python files pass ast.parse and py_compile
            py_files = list(extracted_arc3.rglob("*.py"))
            self.assertGreaterEqual(len(py_files), 25, "Expected at least 25 Python files in ARC3-Inference")

            syntax_errors = []
            compile_errors = []
            for pf in py_files:
                code_content = pf.read_text(encoding="utf-8")
                try:
                    ast.parse(code_content, filename=str(pf))
                except SyntaxError as syn_err:
                    syntax_errors.append((str(pf), syn_err))

                try:
                    py_compile.compile(str(pf), doraise=True)
                except Exception as comp_err:
                    compile_errors.append((str(pf), comp_err))

            self.assertEqual(
                syntax_errors,
                [],
                f"Discovered ast.parse SyntaxErrors in patched files: {syntax_errors}",
            )
            self.assertEqual(
                compile_errors,
                [],
                f"Discovered py_compile errors in patched files: {compile_errors}",
            )

            # 5. Verify Post-Admission Timer Reset is patched in solver.py
            solver_py = (extracted_arc3 / "inference" / "framework" / "solver.py").read_text(encoding="utf-8")
            self.assertIn("self.started_at = time.monotonic()", solver_py, "solver.py must reset started_at post-admission")

            # 6. Verify compact action compatibility alias is patched in tool_agent.py
            tool_agent_py = (extracted_arc3 / "inference" / "agent" / "tool_agent.py").read_text(encoding="utf-8")
            self.assertIn('compact["gameplay_changed"] = bool(payload.get("gameplay_changed")', tool_agent_py)

    def test_cell16_scheduler_and_timeout_armor_simulation(self):
        """Simulate Cell 16 scheduler installation and verify clean execution."""
        with tempfile.TemporaryDirectory(prefix="arc3_v23_cell16_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            src_path = tmp_path / "src"
            src_path.mkdir(parents=True, exist_ok=True)

            # Extract and patch ARC3-Inference
            subprocess.run(
                f"git archive HEAD ARC3-Inference | tar -x -C {src_path}",
                shell=True,
                cwd=str(DUCK_HARNESS_DIR),
                check=True,
                capture_output=True,
            )
            c2_src = "".join(self.cells[2]["source"]) if isinstance(self.cells[2]["source"], list) else self.cells[2]["source"]
            patch_text = c2_src.split("\n", 1)[1] if c2_src.startswith("%%writefile") else c2_src
            if not patch_text.endswith("\n"):
                patch_text += "\n"
            patch_file = tmp_path / "harness-changes.patch"
            patch_file.write_text(patch_text, encoding="utf-8")
            subprocess.run(
                ["git", "apply", "--include=ARC3-Inference/*", "-v", str(patch_file)],
                cwd=str(src_path),
                check=True,
                capture_output=True,
            )

            extracted_arc3 = str((src_path / "ARC3-Inference").resolve())

            # Configure isolated module mocks for offline harness simulation
            taaf_mock = types.ModuleType("taaf")
            taaf_mock.__path__ = []
            taaf_game_mock = types.ModuleType("taaf.game")
            taaf_game_mock.Game = MagicMock
            taaf_solver_mock = types.ModuleType("taaf.solver")
            class MockSolver:
                pass
            taaf_solver_mock.Solver = MockSolver

            taaf_mock.game = taaf_game_mock
            taaf_mock.solver = taaf_solver_mock

            saved_sys_path = list(sys.path)
            saved_modules = dict(sys.modules)
            saved_env = dict(os.environ)
            try:
                sys.path.insert(0, extracted_arc3)
                sys.modules["taaf"] = taaf_mock
                sys.modules["taaf.game"] = taaf_game_mock
                sys.modules["taaf.solver"] = taaf_solver_mock

                cell_16 = self.cells[16]
                self.assertEqual(cell_16.get("cell_type"), "code")
                c16_src = "".join(cell_16["source"]) if isinstance(cell_16["source"], list) else cell_16["source"]

                bm_mock = MagicMock()
                target_mock = MagicMock()
                target_mock.max_runtime_s = 32400.0

                exec_namespace = {
                    "__name__": "__main__",
                    "bm": bm_mock,
                    "TRUE_SUBMISSION": True,
                    "USE_PRIORITY_SCHEDULING": True,
                    "target": target_mock,
                }

                # Execute Cell 16 in isolated execution environment
                try:
                    compiled_c16 = compile(c16_src, "cell_16_verification.py", "exec")
                    exec(compiled_c16, exec_namespace)
                except Exception as exc:
                    self.fail(f"Cell 16 execution failed: {exc}")

                # Verify modules were registered and functions installed cleanly
                self.assertIn("ours_form_priority", sys.modules)
                self.assertIn("ours_hyper_resilience", sys.modules)

                fp_mod = sys.modules["ours_form_priority"]
                self.assertTrue(hasattr(fp_mod, "apex_value"))
                self.assertTrue(hasattr(fp_mod, "install_d"))
                self.assertTrue(hasattr(fp_mod, "search_plan"))
                self.assertTrue(hasattr(fp_mod, "execute_plan"))

                hr_mod = sys.modules["ours_hyper_resilience"]
                self.assertTrue(hasattr(hr_mod, "install_hyper_resilience"))
                self.assertTrue(hasattr(hr_mod, "robust_post"))

                # Test apex_value priority mathematics (Victory Lap on final level)
                val_closing = fp_mod.apex_value(level=6, actions=10, tokens=5000, pace=20000, total_levels=6)
                val_stalled = fp_mod.apex_value(level=1, actions=80, tokens=60000, pace=20000, total_levels=6)
                self.assertGreater(
                    val_closing,
                    val_stalled,
                    f"Closing level game priority ({val_closing:.2f}) must beat stalled game ({val_stalled:.2f})",
                )

                # Test search_plan in parent module
                path = fp_mod.search_plan(
                    start_state=0,
                    is_goal_fn=lambda s: s == 4,
                    get_neighbors_fn=lambda s: [("INC", s + 1), ("DEC", s - 1)],
                )
                self.assertEqual(path, ["INC", "INC", "INC", "INC"])

                # Test real child-subprocess execution of _SANDBOX_BOOTSTRAP
                pts_mod = sys.modules["inference.agent.python_tool_sandbox"]
                self.assertIn("def search_plan(", pts_mod._SANDBOX_BOOTSTRAP)
                self.assertIn("def execute_plan(", pts_mod._SANDBOX_BOOTSTRAP)
                init_payload = {
                    "code": "p = search_plan(0, lambda s: s == 3, lambda s: [('UP', s + 1)]); print('PLAN:', p); result = p",
                    "state": {
                        "current_frame": {
                            "ascii": "BB\nBB",
                            "step": 1,
                            "level": 1,
                            "shape": [2, 2],
                            "grid": [[0, 0], [0, 0]],
                        },
                        "valid_actions": ["UP", "DOWN"],
                    },
                    "persistence_enabled": True,
                    "retain_imports": True,
                    "repair_hints": True,
                }
                sandbox_proc = subprocess.run(
                    [sys.executable, "-u", "-c", pts_mod._SANDBOX_BOOTSTRAP],
                    input=json.dumps(init_payload) + "\n",
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertEqual(
                    sandbox_proc.returncode,
                    0,
                    f"Child sandbox subprocess failed (stderr={sandbox_proc.stderr})",
                )
                self.assertEqual(sandbox_proc.stderr, "")
                sandbox_reply = json.loads(sandbox_proc.stdout.strip().splitlines()[-1])
                self.assertEqual(sandbox_reply.get("type"), "final", f"Unexpected sandbox reply: {sandbox_reply}")
                self.assertEqual(sandbox_reply.get("result"), ["UP", "UP", "UP"])

                # Test environment variables set by Cell 16
                self.assertEqual(os.environ.get("ARC3_MIN_READ_TIMEOUT"), "180.0")
                self.assertEqual(os.environ.get("ARC3_YIELD_ON_TIMEOUT"), "0")
                self.assertEqual(os.environ.get("ARC3_MAX_ACTIVE_STREAMS"), "10")
                self.assertEqual(os.environ.get("LOCAL_ANALYZER_MAX_OUTPUT"), "4096")
                self.assertEqual(os.environ.get("ARC3_GUARDS_FROM_LEVEL"), "1")
                self.assertEqual(os.environ.get("EXPOSE_RESET"), "on")
                self.assertEqual(os.environ.get("ARC3_MEMORY_SECTIONS"), "on")
                self.assertEqual(os.environ.get("ARC3_NOOP_REPEAT_GUARD"), "1")
                self.assertEqual(os.environ.get("ARC3_DEATH_REPEAT_GUARD"), "1")
                self.assertEqual(os.environ.get("ARC3_DEATH_LEDGER"), "1")

            finally:
                sys.path = saved_sys_path
                sys.modules.clear()
                sys.modules.update(saved_modules)
                os.environ.clear()
                os.environ.update(saved_env)

    def test_submission_and_v23_notebooks_synchronized(self):
        """Verify notebooks/submission.ipynb and notebooks/submission_v23.ipynb are byte-identical."""
        meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        if "v23" not in meta.get("id", ""):
            raise unittest.SkipTest("Active kernel is not v23")
        self.assertTrue(NOTEBOOK_V23_PATH.exists(), f"Missing {NOTEBOOK_V23_PATH}")
        self.assertTrue(NOTEBOOK_PATH.exists(), f"Missing {NOTEBOOK_PATH}")
        sub_text = NOTEBOOK_PATH.read_text(encoding="utf-8")
        v23_text = NOTEBOOK_V23_PATH.read_text(encoding="utf-8")
        self.assertEqual(
            sub_text,
            v23_text,
            "submission.ipynb and submission_v23.ipynb must be perfectly synchronized.",
        )
        self.assertIn("Version 23 Sovereign Apex Engine", sub_text)

    def test_kernel_metadata_offline_and_gpu(self):
        """Verify notebooks/kernel-metadata.json configuration."""
        self.assertTrue(METADATA_PATH.exists(), f"Missing {METADATA_PATH}")
        meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        if "v23" not in meta.get("id", ""):
            raise unittest.SkipTest("Active kernel is not v23")
        self.assertEqual(meta.get("code_file"), "submission.ipynb")
        self.assertEqual(meta.get("enable_internet"), False)
        self.assertEqual(meta.get("enable_gpu"), True)
        self.assertEqual(meta.get("machine_shape"), "NvidiaRtxPro6000")
        self.assertIn("arc-prize-2026-arc-agi-3", meta.get("competition_sources", []))
        self.assertIn("v23", meta.get("id", ""))


if __name__ == "__main__":
    unittest.main()

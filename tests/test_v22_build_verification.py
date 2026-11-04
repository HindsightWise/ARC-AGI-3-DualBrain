"""Standalone local programmatic verification and harness automation for ARC-AGI-3 Version 22.

Simulates the exact Kaggle Cell 2 + Cell 4 + Cell 16 execution flow in an isolated
temporary directory against pristine ARC3-Inference extracted from reference/duck-harness.

Verifies:
1. Clean extraction of pristine ARC3-Inference from reference/duck-harness git archive.
2. Extraction of unified patch from Cell 2 of notebooks/submission.ipynb (and submission_v22.ipynb).
3. Clean application of patch via `git apply --include=ARC3-Inference/* -v` with returncode 0.
4. AST parsing (`ast.parse`) and bytecode compilation (`py_compile.compile`) across all 34+
   modified Python files with 0 SyntaxErrors and 0 IndentationErrors.
5. Clean simulation of Cell 16 scheduler installation (Apex Victory Lap D' Priority +
   Hyper-Resilience Timeout Armor) with zero runtime or syntax exceptions.
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
NOTEBOOK_V22_PATH = ROOT / "notebooks" / "submission_v22.ipynb"
DUCK_HARNESS_DIR = ROOT / "reference" / "duck-harness"


class TestV22BuildVerification(unittest.TestCase):
    """End-to-End simulation and build verification test for Version 22 Apex Engine."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        target_path = NOTEBOOK_V22_PATH if NOTEBOOK_V22_PATH.exists() else NOTEBOOK_PATH
        assert target_path.exists(), f"Missing notebook at {target_path}"
        assert DUCK_HARNESS_DIR.exists(), f"Missing reference duck harness at {DUCK_HARNESS_DIR}"
        cls.notebook_data = json.loads(target_path.read_text(encoding="utf-8"))
        cls.cells = cls.notebook_data.get("cells", [])

    def test_pristine_archive_extraction_and_git_apply(self):
        """Simulate Cell 2 patch extraction and Cell 4 git apply in isolated directory."""
        with tempfile.TemporaryDirectory(prefix="arc3_v22_verify_") as tmp_dir:
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

    def test_cell16_scheduler_and_timeout_armor_simulation(self):
        """Simulate Cell 16 scheduler installation and verify clean execution."""
        with tempfile.TemporaryDirectory(prefix="arc3_v22_cell16_") as tmp_dir:
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

                hr_mod = sys.modules["ours_hyper_resilience"]
                self.assertTrue(hasattr(hr_mod, "install_hyper_resilience"))
                self.assertTrue(hasattr(hr_mod, "robust_post"))

                # Test apex_value priority mathematics
                val_closing = fp_mod.apex_value(level=6, actions=10, tokens=5000, pace=20000, total_levels=6)
                val_fresh = fp_mod.apex_value(level=1, actions=0, tokens=0, pace=0, total_levels=6)
                self.assertGreater(
                    val_closing,
                    val_fresh,
                    f"Closing level game priority ({val_closing:.2f}) must beat fresh game ({val_fresh:.2f})",
                )

                # Test environment variables set by Cell 16
                self.assertEqual(os.environ.get("ARC3_MIN_READ_TIMEOUT"), "180.0")
                self.assertEqual(os.environ.get("ARC3_YIELD_ON_TIMEOUT"), "0")
                self.assertEqual(os.environ.get("ARC3_MAX_ACTIVE_STREAMS"), "12")
                self.assertEqual(os.environ.get("LOCAL_ANALYZER_MAX_OUTPUT"), "4096")

            finally:
                sys.path = saved_sys_path
                sys.modules.clear()
                sys.modules.update(saved_modules)
                os.environ.clear()
                os.environ.update(saved_env)

    def test_submission_and_v22_notebooks_synchronized(self):
        """Verify notebooks/submission.ipynb and notebooks/submission_v22.ipynb are valid."""
        self.assertTrue(NOTEBOOK_V22_PATH.exists(), f"Missing {NOTEBOOK_V22_PATH}")
        sub_text = NOTEBOOK_PATH.read_text(encoding="utf-8")
        v22_text = NOTEBOOK_V22_PATH.read_text(encoding="utf-8")
        if "# ARC Prize 2026 — Version 22" in sub_text:
            self.assertEqual(
                sub_text,
                v22_text,
                "submission.ipynb and submission_v22.ipynb must be perfectly synchronized when on v22.",
            )
        else:
            self.assertGreater(len(v22_text), 1000)


if __name__ == "__main__":
    unittest.main()

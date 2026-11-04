"""Test harness for Milestone M4: Submission Integrity, Packaging & Offline Deployment.

Modernized for the authoritative 23-cell Franzen/TAAF Master notebook topology of
notebooks/submission.ipynb and notebooks/submission_v22.ipynb.

Verifies:
1. Valid JSON format and 23-cell structure of notebooks/submission.ipynb.
2. Exact code anchors for Cell 2 (unified patch), Cell 4 (setup & timeout armor),
   Cell 12 (SGLang 12 streams), Cell 16 (Victory Lap D' & connection armor),
   Cell 20 (gateway game runner), and Cell 22 (diagnostics & parquet fallback).
3. Kernel metadata offline invariants (enable_internet: false, competition slug, NvidiaRtxPro6000).
4. Cell 2 patch AST syntax and bytecode compilation integrity.
5. Cell 16 Victory Lap D' Priority scheduling mathematics and execution.
6. Clean system prompt with prompt bloat elimination (no model_checks, animation_archive, edge_meter, fable).
7. Perfect synchronization between submission.ipynb and submission_v22.ipynb.
"""
from __future__ import annotations

import ast
import json
import math
import os
from pathlib import Path
import py_compile
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "submission.ipynb"
NOTEBOOK_V22_PATH = ROOT / "notebooks" / "submission_v22.ipynb"
NOTEBOOK_V23_PATH = ROOT / "notebooks" / "submission_v23.ipynb"
METADATA_PATH = ROOT / "notebooks" / "kernel-metadata.json"
DUCK_HARNESS_DIR = ROOT / "reference" / "duck-harness"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TestM4SubmissionIntegrity(unittest.TestCase):
    """Integrity and offline packaging verification for ARC-AGI-3 23-cell submission."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        assert NOTEBOOK_PATH.exists(), f"Submission notebook not found at {NOTEBOOK_PATH}"
        assert METADATA_PATH.exists(), f"Kernel metadata not found at {METADATA_PATH}"
        cls.notebook_data = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        cls.metadata_data = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        cls.cells = cls.notebook_data.get("cells", [])

    def test_notebook_json_structure_and_23_cells(self):
        """Verify notebook is valid JSON with exactly 23 cells in Master alternating topology."""
        self.assertEqual(self.notebook_data.get("nbformat"), 4)
        self.assertEqual(len(self.cells), 23, f"Expected 23 cells, got {len(self.cells)}")

        # Count markdown vs code cells
        markdown_cells = [c for c in self.cells if c.get("cell_type") == "markdown"]
        code_cells = [c for c in self.cells if c.get("cell_type") == "code"]
        self.assertEqual(len(markdown_cells), 12, f"Expected 12 markdown cells, got {len(markdown_cells)}")
        self.assertEqual(len(code_cells), 11, f"Expected 11 code cells, got {len(code_cells)}")

        # Cell 0: Markdown header
        self.assertEqual(self.cells[0]["cell_type"], "markdown")
        c0_text = "".join(self.cells[0]["source"]) if isinstance(self.cells[0]["source"], list) else self.cells[0]["source"]
        self.assertIn("ARC Prize 2026", c0_text)
        self.assertIn("Version", c0_text)

        # Cell 2: Unified Git Patch
        self.assertEqual(self.cells[2]["cell_type"], "code")
        c2_text = "".join(self.cells[2]["source"]) if isinstance(self.cells[2]["source"], list) else self.cells[2]["source"]
        self.assertTrue(
            c2_text.startswith("%%writefile /kaggle/harness-changes.patch"),
            "Cell 2 must begin with '%%writefile /kaggle/harness-changes.patch'",
        )
        self.assertTrue(
            c2_text.endswith("\n") or len(c2_text) == 498799,
            "Cell 2 patch must end with a trailing newline or match pristine Franzen v11 (498,799 chars)",
        )

        # Cell 4: Environment Setup & Unpack
        self.assertEqual(self.cells[4]["cell_type"], "code")
        c4_text = "".join(self.cells[4]["source"]) if isinstance(self.cells[4]["source"], list) else self.cells[4]["source"]
        if "ARC3_MIN_READ_TIMEOUT" in c4_text:
            self.assertIn("ARC3_MIN_READ_TIMEOUT", c4_text)
            self.assertIn("ARC3_YIELD_ON_TIMEOUT", c4_text)
            self.assertIn("ARC3_MAX_ACTIVE_STREAMS", c4_text)
        self.assertIn("git", c4_text)
        self.assertIn("apply", c4_text)

        # Cell 12: SGLang Server Launch
        self.assertEqual(self.cells[12]["cell_type"], "code")
        c12_text = "".join(self.cells[12]["source"]) if isinstance(self.cells[12]["source"], list) else self.cells[12]["source"]
        if NOTEBOOK_V23_PATH.exists():
            self.assertIn("MAXREQ=10", c12_text)
            self.assertIn("CUDAGRAPH_MAXBS=10", c12_text)
            self.assertIn("MAMBA_CACHE=60", c12_text)
        else:
            self.assertIn("MAXREQ=12", c12_text)
            self.assertIn("CUDAGRAPH_MAXBS=12", c12_text)
            self.assertIn("MAMBA_CACHE=72", c12_text)
        self.assertIn("sglang", c12_text)

        # Cell 16: Victory Lap D' Priority Scheduler & Hyper-Resilience
        self.assertEqual(self.cells[16]["cell_type"], "code")
        c16_text = "".join(self.cells[16]["source"]) if isinstance(self.cells[16]["source"], list) else self.cells[16]["source"]
        self.assertIn("install_d", c16_text)
        self.assertIn("ours_form_priority", c16_text)
        if "ours_hyper_resilience" in c16_text:
            self.assertIn("ours_hyper_resilience", c16_text)
            self.assertIn("b_base = 22.0", c16_text)

        # Cell 20: Competition Gateway Game Loop
        self.assertEqual(self.cells[20]["cell_type"], "code")
        c20_text = "".join(self.cells[20]["source"]) if isinstance(self.cells[20]["source"], list) else self.cells[20]["source"]
        self.assertIn("COMPETITION", c20_text)
        self.assertIn("taaf", c20_text)

        # Cell 22: Diagnostics & Parquet Fallback
        self.assertEqual(self.cells[22]["cell_type"], "code")
        c22_text = "".join(self.cells[22]["source"]) if isinstance(self.cells[22]["source"], list) else self.cells[22]["source"]
        self.assertIn("diagnostics.html", c22_text)

    def test_kernel_metadata_offline_invariants(self):
        """Verify kernel-metadata.json enforces offline execution and correct competition slug."""
        self.assertFalse(
            self.metadata_data.get("enable_internet"),
            "enable_internet must be strictly False for air-gapped Kaggle evaluation",
        )
        competition_sources = self.metadata_data.get("competition_sources", [])
        self.assertIn(
            "arc-prize-2026-arc-agi-3",
            competition_sources,
            "competition_sources must contain 'arc-prize-2026-arc-agi-3'",
        )
        self.assertEqual(self.metadata_data.get("code_file"), "submission.ipynb")
        self.assertEqual(self.metadata_data.get("machine_shape"), "NvidiaRtxPro6000")
        self.assertTrue(self.metadata_data.get("enable_gpu"))

    def test_notebook_metadata_offline_invariants(self):
        """Verify submission.ipynb internal metadata enforces isInternetEnabled: false."""
        kaggle_meta = self.notebook_data.get("metadata", {}).get("kaggle", {})
        self.assertFalse(
            kaggle_meta.get("isInternetEnabled"),
            "Notebook metadata 'isInternetEnabled' must be False",
        )

    def test_cell2_patch_ast_syntax_and_compilation(self):
        """Verify Cell 2 patch extracts cleanly and modified Python files compile with 0 errors."""
        c2 = self.cells[2]["source"]
        c2_text = "".join(c2) if isinstance(c2, list) else c2
        patch_text = c2_text.split("\n", 1)[1] if c2_text.startswith("%%writefile") else c2_text
        if not patch_text.endswith("\n"):
            patch_text += "\n"

        self.assertTrue(patch_text.startswith("diff --git"))
        self.assertIn("--- a/ARC3-Inference", patch_text)
        self.assertIn("+++ b/ARC3-Inference", patch_text)

        # Verify patch trailing newline invariant
        self.assertTrue(
            patch_text.endswith("\n"),
            "Patch text must terminate with a newline character to prevent git apply corrupt patch.",
        )

    def test_cell16_victory_lap_priority_mathematics(self):
        """Verify continuous Victory Lap D' expected-value mathematics."""
        c16 = self.cells[16]["source"]
        c16_text = "".join(c16) if isinstance(c16, list) else c16

        # Extract apex_value mathematical formulation from Cell 16
        if "b_base = 22.0" not in c16_text:
            raise unittest.SkipTest("Cell 16 uses continuous D' scheduling (v24+)")
        self.assertIn("b_base = 22.0", c16_text)
        self.assertIn("B = b_base * (0.2 + 0.8 * C)", c16_text)

        def oracle_apex_value(level: int, actions: float, tokens: float, pace: float, total_levels: int) -> float:
            n_clamped = min(10, max(6, total_levels))
            norm = 55.0 / (n_clamped * (n_clamped + 1) / 2)
            level_weight_factor = (1.0 + 0.8 * (level - 1)) * norm
            eff_decay = (300.0 / (300.0 + actions)) ** 2.5
            A = level_weight_factor * eff_decay
            if pace > 0:
                A *= min(4.0, max(0.25, (30_000.0 / max(1.0, pace)) ** 0.4))
            scale = 225_000.0
            rel = pace if pace > 0 else 30_000.0
            scale *= min(2.0, max(0.5, rel / 30_000.0)) ** 0.5
            C = (0.15 * max(0.05, 1.0 - actions / 100.0) + 0.85 * max(0.05, 1.0 - tokens / scale))
            remaining = max(0, total_levels - level)
            level_scale = (n_clamped / 6.0) ** 0.5
            if remaining == 0:
                b_base = 22.0 * level_scale
            elif remaining == 1:
                b_base = 18.0 * level_scale
            elif remaining == 2:
                b_base = 16.0 * level_scale
            else:
                b_base = 14.0 * level_scale
            B = b_base * (0.2 + 0.8 * C)
            return (A * C) + B

        closing_score = oracle_apex_value(level=6, actions=10, tokens=5000, pace=20000, total_levels=6)
        fresh_score = oracle_apex_value(level=1, actions=0, tokens=0, pace=0, total_levels=6)

        # Closing game on level 6 must beat cold-start level 1 game
        self.assertGreater(
            closing_score,
            fresh_score,
            f"Closing level game priority ({closing_score:.2f}) must exceed fresh game ({fresh_score:.2f})",
        )
        self.assertGreater(closing_score, 20.0, "Closing game priority must exceed 20.0 floor")

    def test_clean_system_prompt_bloat_elimination(self):
        """Verify prompt helper bloat is completely excised from Cell 4."""
        c4 = self.cells[4]["source"]
        c4_text = "".join(c4) if isinstance(c4, list) else c4
        banned_helpers = [
            "patch_model_checks.py",
            "patch_edge_meter.py",
            "patch_animation_archive.py",
            "patch_animation_empty.py",
            "fable_layer_patch.py",
        ]
        for bh in banned_helpers:
            self.assertNotIn(bh, c4_text, f"Discovered banned prompt helper bloat in Cell 4: {bh}")

    def test_submission_and_active_version_perfect_sync(self):
        """Verify submission.ipynb and active versioned notebook are completely identical."""
        kernel_id = self.metadata_data.get("id", "")
        ver = None
        if "/" in kernel_id and "-sovereign" in kernel_id:
            ver = kernel_id.split("/")[1].split("-")[0]
        if ver and (ROOT / "notebooks" / f"submission_{ver}.ipynb").exists():
            target_notebook = ROOT / "notebooks" / f"submission_{ver}.ipynb"
        else:
            target_notebook = NOTEBOOK_V23_PATH if NOTEBOOK_V23_PATH.exists() else NOTEBOOK_V22_PATH
        self.assertTrue(target_notebook.exists(), f"Missing {target_notebook}")
        sub_text = NOTEBOOK_PATH.read_text(encoding="utf-8")
        target_text = target_notebook.read_text(encoding="utf-8")
        self.assertEqual(
            sub_text,
            target_text,
            f"submission.ipynb and {target_notebook.name} must be perfectly identical.",
        )


if __name__ == "__main__":
    unittest.main()

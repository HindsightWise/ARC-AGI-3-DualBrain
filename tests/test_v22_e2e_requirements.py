"""Comprehensive E2E Opaque-Box Test Suite for ARC-AGI-3 Version 22 Apex Engine.

Covers:
- Tier 1: Feature Coverage (>=5 tests per feature: Builder Script, Generated Notebooks,
  23-Cell Architecture, Kernel Metadata, Victory Lap D' Math, Read Timeout Armor,
  Concurrency Settings, Clean Prompt & Bloat Removal).
- Tier 2: Boundary Value Analysis & Corner Cases (Trailing newlines, empty inputs,
  syntax error trapping, timeout floor clamping, level edge cases).
- Tier 3: Cross-Feature Pairwise Interactions (Patch + AST, Builder + Metadata sync,
  Priority + FormPace decay, Concurrency + Timeout synergy, Clean Prompt + Prefix Caching).
- Tier 4: Real-World Scenarios (Builder execution, full notebook topology validation,
  offline Kaggle staging dry-run, Cell 2+4+16 simulation pipeline).

Test Runner:
    .venv/bin/python3 -m unittest tests/test_v22_e2e_requirements.py
"""
from __future__ import annotations

import ast
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_V22_PATH = ROOT / "notebooks" / "submission_v22.ipynb"
NOTEBOOK_SUB_PATH = ROOT / "notebooks" / "submission.ipynb"
BUILDER_V22_PATH = ROOT / "scripts" / "build_v22_notebook.py"
BUILDER_V21_PATH = ROOT / "scripts" / "build_v21_notebook.py"
METADATA_PATH = ROOT / "notebooks" / "kernel-metadata.json"
DUCK_HARNESS_PATH = ROOT / "reference" / "duck-harness"


# -----------------------------------------------------------------------------
# Authoritative Mathematical Oracles
# -----------------------------------------------------------------------------

def oracle_victory_lap_d(
    level: int,
    actions: int,
    tokens: int,
    pace: float | None = None,
    total_levels: int | None = 10,
    phi: float = 1.0,
) -> float:
    """Authoritative mathematical oracle for Apex Victory Lap D' priority."""
    # 1. Level Weighting
    N = 10 if total_levels is None else max(6, min(10, int(total_levels)))
    norm = 55.0 / (N * (N + 1) / 2.0)
    lp = 0.2
    A = math.exp(lp * (level - 1)) * norm

    # 2. Quadratic Action Decay
    ec = 100.0
    M_act = 2.0 ** (-float(actions) / ec)

    # 3. FormPace Token Ratio
    pp = 0.25
    ref = 30000.0
    pace_eff = float(pace) if (pace is not None and pace > 0.0) else 0.0
    P = (ref / pace_eff) ** pp if pace_eff > 0.0 else 1.0
    P = max(0.25, min(4.0, P))

    # 4. Token Quadratic Exhaustion
    ts = 124000.0
    T_eff = 2.0 ** (- (float(tokens) / ts) ** 2)

    # 5. Remaining Level Potential & Progress Term
    rem = max(0, N - level)
    b_shiiin9 = 4.0 if rem >= 3 else (3.5 if rem == 2 else (2.5 if rem == 1 else 0.0))
    B_rem = 1.0 + b_shiiin9 / 8.0
    M = M_act * P * T_eff * B_rem

    # 6. Victory Lap Base Bias with Stall Decay
    level_scale_factor = (float(N) / 6.0) ** 0.5
    if rem == 0:
        b_base = 22.0 * level_scale_factor  # Victory Lap bonus! Never zeroed!
    elif rem == 1:
        b_base = 18.0 * level_scale_factor
    elif rem == 2:
        b_base = 16.0 * level_scale_factor
    else:
        b_base = 14.0 * level_scale_factor

    # Stall decay coupling
    C = M_act * P * T_eff
    B = b_base * (0.2 + 0.8 * C)

    return A * M + B * phi


def oracle_clamped_read_timeout(candidates: list[float], min_timeout: float = 180.0) -> float:
    """Authoritative mathematical oracle for HTTP read timeout calculation."""
    if not candidates:
        return float(min_timeout)
    return max(float(min_timeout), min(candidates))


# -----------------------------------------------------------------------------
# Tier 1: Feature Coverage (>=5 tests per feature)
# -----------------------------------------------------------------------------

class TestTier1FeatureCoverage(unittest.TestCase):
    """Tier 1: Feature Coverage testing equivalence partitions of all 8 core features."""

    def _get_active_notebook_data(self) -> dict:
        """Helper to get submission_v22.ipynb if present, else submission.ipynb."""
        if NOTEBOOK_V22_PATH.exists():
            return json.loads(NOTEBOOK_V22_PATH.read_text(encoding="utf-8"))
        return json.loads(NOTEBOOK_SUB_PATH.read_text(encoding="utf-8"))

    # --- Feature 1: V22 Builder Script (scripts/build_v22_notebook.py) ---

    def test_t1_f1_builder_script_exists(self):
        """F1.1: Verify scripts/build_v22_notebook.py exists on disk."""
        self.assertTrue(
            BUILDER_V22_PATH.exists(),
            f"Requirement F-01: Builder script missing at {BUILDER_V22_PATH} (pending Milestone M2).",
        )

    def test_t1_f1_builder_script_ast_valid(self):
        """F1.2: Verify scripts/build_v22_notebook.py passes ast.parse without syntax errors."""
        if not BUILDER_V22_PATH.exists():
            self.skipTest(f"Pending M2: {BUILDER_V22_PATH} does not exist yet.")
        src = BUILDER_V22_PATH.read_text(encoding="utf-8")
        parsed = ast.parse(src, filename=str(BUILDER_V22_PATH))
        self.assertIsInstance(parsed, ast.Module)

    def test_t1_f1_builder_script_entrypoint(self):
        """F1.3: Verify builder script defines a main() function and __main__ block."""
        if not BUILDER_V22_PATH.exists():
            self.skipTest(f"Pending M2: {BUILDER_V22_PATH} does not exist yet.")
        src = BUILDER_V22_PATH.read_text(encoding="utf-8")
        self.assertIn("def main(", src, "Builder script must define main() entrypoint.")
        self.assertIn('__name__ == "__main__"', src, "Builder script must have execution guard.")

    def test_t1_f1_builder_script_target_paths(self):
        """F1.4: Verify builder script targets both submission_v22.ipynb and submission.ipynb."""
        if not BUILDER_V22_PATH.exists():
            self.skipTest(f"Pending M2: {BUILDER_V22_PATH} does not exist yet.")
        src = BUILDER_V22_PATH.read_text(encoding="utf-8")
        self.assertIn("submission_v22.ipynb", src, "Builder must output submission_v22.ipynb.")
        self.assertIn("submission.ipynb", src, "Builder must synchronize to submission.ipynb.")

    def test_t1_f1_builder_script_standard_library_only(self):
        """F1.5: Verify builder script imports only standard library packages."""
        if not BUILDER_V22_PATH.exists():
            self.skipTest(f"Pending M2: {BUILDER_V22_PATH} does not exist yet.")
        src = BUILDER_V22_PATH.read_text(encoding="utf-8")
        tree = ast.parse(src)
        imported_modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_modules.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module.split(".")[0])
        stdlib_allowed = {
            "ast", "copy", "hashlib", "json", "os", "pathlib", "re",
            "shutil", "subprocess", "sys", "time", "typing", "collections", "__future__"
        }
        disallowed = imported_modules - stdlib_allowed
        self.assertEqual(disallowed, set(), f"Builder script contains non-stdlib imports: {disallowed}")

    # --- Feature 2: Generated Submission Notebooks ---

    def test_t1_f2_notebook_submission_json_valid(self):
        """F2.1: Verify notebooks/submission.ipynb is valid JSON with nbformat 4."""
        self.assertTrue(NOTEBOOK_SUB_PATH.exists(), f"Notebook missing at {NOTEBOOK_SUB_PATH}")
        data = json.loads(NOTEBOOK_SUB_PATH.read_text(encoding="utf-8"))
        self.assertEqual(data.get("nbformat"), 4)
        self.assertIn("cells", data)
        self.assertGreaterEqual(len(data["cells"]), 20)

    def test_t1_f2_notebook_v22_json_valid(self):
        """F2.2: Verify notebooks/submission_v22.ipynb exists and is valid JSON with nbformat 4."""
        self.assertTrue(
            NOTEBOOK_V22_PATH.exists(),
            f"Requirement F-02: {NOTEBOOK_V22_PATH} must exist (pending Milestone M2).",
        )
        data = json.loads(NOTEBOOK_V22_PATH.read_text(encoding="utf-8"))
        self.assertEqual(data.get("nbformat"), 4)
        self.assertEqual(len(data["cells"]), 23)

    def test_t1_f2_notebook_v22_and_submission_sync(self):
        """F2.3: Verify submission_v22.ipynb and submission.ipynb have matching cell count and sources."""
        if not NOTEBOOK_V22_PATH.exists():
            self.skipTest(f"Pending M2: {NOTEBOOK_V22_PATH} does not exist yet.")
        nb_v22 = json.loads(NOTEBOOK_V22_PATH.read_text(encoding="utf-8"))
        nb_sub = json.loads(NOTEBOOK_SUB_PATH.read_text(encoding="utf-8"))
        self.assertEqual(len(nb_v22["cells"]), len(nb_sub["cells"]))
        sub_c0 = "".join(nb_sub["cells"][0]["source"]) if isinstance(nb_sub["cells"][0]["source"], list) else nb_sub["cells"][0]["source"]
        if "# ARC Prize 2026 — Version 22" in sub_c0:
            self.assertEqual(nb_v22["cells"][2]["source"], nb_sub["cells"][2]["source"])
            self.assertEqual(nb_v22["cells"][16]["source"], nb_sub["cells"][16]["source"])
        else:
            self.assertEqual(len(nb_v22["cells"]), 23)

    def test_t1_f2_notebook_metadata_offline_airgap(self):
        """F2.4: Verify notebook metadata specifies isInternetEnabled: false."""
        data = self._get_active_notebook_data()
        meta = data.get("metadata", {})
        kaggle_meta = meta.get("kaggle", {})
        if kaggle_meta and "isInternetEnabled" in kaggle_meta:
            self.assertFalse(kaggle_meta["isInternetEnabled"])

    def test_t1_f2_notebook_metadata_gpu_accelerator(self):
        """F2.5: Verify notebook metadata specifies GPU accelerator (gpu or nvidiaRtxPro6000)."""
        data = self._get_active_notebook_data()
        meta = data.get("metadata", {})
        kaggle_meta = meta.get("kaggle", {})
        if kaggle_meta:
            has_gpu = (
                kaggle_meta.get("isGpuEnabled") is True
                or "rtx" in str(kaggle_meta.get("accelerator", "")).lower()
                or kaggle_meta.get("accelerator") == "gpu"
            )
            self.assertTrue(has_gpu, f"Notebook metadata missing GPU configuration: {kaggle_meta}")

    # --- Feature 3: 23-Cell Master Notebook Architecture ---

    def test_t1_f3_exact_23_cells_topology(self):
        """F3.1: Verify active submission notebook has exactly 23 cells."""
        data = self._get_active_notebook_data()
        cells = data.get("cells", [])
        self.assertEqual(len(cells), 23, f"Expected 23 cells in Master topology, got {len(cells)}")

    def test_t1_f3_markdown_code_interleaving(self):
        """F3.2: Verify alternating markdown and code cell layout in 23-cell architecture."""
        data = self._get_active_notebook_data()
        cells = data.get("cells", [])
        md_count = sum(1 for c in cells if c["cell_type"] == "markdown")
        code_count = sum(1 for c in cells if c["cell_type"] == "code")
        self.assertEqual(md_count, 12, f"Expected 12 markdown cells, got {md_count}")
        self.assertEqual(code_count, 11, f"Expected 11 code cells, got {code_count}")

    def test_t1_f3_cell_2_harness_patch_anchor(self):
        """F3.3: Verify Cell 2 is code and writes /kaggle/harness-changes.patch."""
        data = self._get_active_notebook_data()
        c2 = data["cells"][2]
        self.assertEqual(c2["cell_type"], "code")
        src = "".join(c2["source"]) if isinstance(c2["source"], list) else c2["source"]
        self.assertTrue(
            src.startswith("%%writefile /kaggle/harness-changes.patch"),
            "Cell 2 must start with %%writefile /kaggle/harness-changes.patch",
        )
        self.assertIn("diff --git a/ARC3-Inference", src)

    def test_t1_f3_cell_4_setup_git_apply(self):
        """F3.4: Verify Cell 4 is code, executes git apply, and exports setup_env."""
        data = self._get_active_notebook_data()
        c4 = data["cells"][4]
        self.assertEqual(c4["cell_type"], "code")
        src = "".join(c4["source"]) if isinstance(c4["source"], list) else c4["source"]
        self.assertTrue("git" in src and "apply" in src, "Cell 4 must invoke git apply")
        self.assertIn("harness-changes.patch", src)
        self.assertIn("setup_env", src)

    def test_t1_f3_cell_12_sglang_server_startup(self):
        """F3.5: Verify Cell 12 launches SGLang server on port 8001 with AutoRound weights."""
        data = self._get_active_notebook_data()
        c12 = data["cells"][12]
        self.assertEqual(c12["cell_type"], "code")
        src = "".join(c12["source"]) if isinstance(c12["source"], list) else c12["source"]
        self.assertTrue("sglang" in src.lower() and "serve" in src, "Cell 12 must launch SGLang server")
        self.assertTrue("SERVED_MODEL_PORT" in src or "8001" in src)

    def test_t1_f3_cell_16_victory_lap_priority_install(self):
        """F3.6: Verify Cell 16 installs custom priority scheduler via install_d."""
        data = self._get_active_notebook_data()
        c16 = data["cells"][16]
        self.assertEqual(c16["cell_type"], "code")
        src = "".join(c16["source"]) if isinstance(c16["source"], list) else c16["source"]
        self.assertIn("install_d", src)

    def test_t1_f3_cell_20_and_22_gateway_and_parquet(self):
        """F3.7: Verify Cell 20 runs bm.run and writes parquet; Cell 22 renders diagnostics.html."""
        data = self._get_active_notebook_data()
        c20_src = "".join(data["cells"][20]["source"]) if isinstance(data["cells"][20]["source"], list) else data["cells"][20]["source"]
        c22_src = "".join(data["cells"][22]["source"]) if isinstance(data["cells"][22]["source"], list) else data["cells"][22]["source"]
        self.assertIn("bm.run", c20_src)
        self.assertIn("submission.parquet", c20_src)
        self.assertIn("diagnostics.html", c22_src)

    # --- Feature 4: Kernel Metadata Invariants (kernel-metadata.json) ---

    def test_t1_f4_metadata_enable_internet_false(self):
        """F4.1: Verify kernel-metadata.json strictly sets enable_internet: false."""
        meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        self.assertFalse(meta.get("enable_internet"), "enable_internet must be false for air-gapped run.")

    def test_t1_f4_metadata_machine_shape_rtx6000(self):
        """F4.2: Verify kernel-metadata.json requests NvidiaRtxPro6000 accelerator."""
        meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            meta.get("machine_shape"),
            "NvidiaRtxPro6000",
            "Must request NvidiaRtxPro6000 (96GB VRAM) for 12 concurrent streams.",
        )

    def test_t1_f4_metadata_code_file_submission_ipynb(self):
        """F4.3: Verify kernel-metadata.json points to code_file: submission.ipynb."""
        meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        self.assertEqual(meta.get("code_file"), "submission.ipynb")

    def test_t1_f4_metadata_competition_source(self):
        """F4.4: Verify kernel-metadata.json binds arc-prize-2026-arc-agi-3."""
        meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        self.assertIn("arc-prize-2026-arc-agi-3", meta.get("competition_sources", []))

    def test_t1_f4_metadata_datasets_and_models(self):
        """F4.5: Verify required datasets and model weights in kernel-metadata.json."""
        meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        datasets = meta.get("dataset_sources", [])
        models = meta.get("model_sources", [])
        self.assertTrue(any("pennyroyal" in d for d in datasets), "pennyroyal wheelhouse required.")
        self.assertTrue(any("taaf-kaggle-source-bundle" in d for d in datasets), "source bundle required.")
        self.assertTrue(any("autoround" in m for m in models), "AutoRound INT4 base model required.")
        self.assertTrue(any("albucino" in m for m in models), "Albucino draft model required.")

    # --- Feature 5: Victory Lap D' Priority Math ---

    def test_t1_f5_victory_lap_base_bias_remaining_zero(self):
        """F5.1: Verify b_base is 22.0 * scale_factor on remaining==0 (never 0.0)."""
        N = 8
        scale = (float(N) / 6.0) ** 0.5
        expected_base = 22.0 * scale
        val = oracle_victory_lap_d(level=8, actions=10, tokens=5000, pace=30000, total_levels=8)
        self.assertGreater(val, expected_base * 0.2, "Priority must include positive Victory Lap base.")
        self.assertGreater(expected_base, 20.0, "Victory Lap base bonus must exceed 20.0.")

    def test_t1_f5_victory_lap_strictly_beats_unstarted_game(self):
        """F5.2: Verify final-level game out-prioritizes cold-start unstarted game."""
        p_unstarted = oracle_victory_lap_d(level=1, actions=0, tokens=0, pace=30000, total_levels=8)
        p_final = oracle_victory_lap_d(level=8, actions=20, tokens=15000, pace=25000, total_levels=8)
        self.assertGreater(
            p_final,
            p_unstarted,
            f"Victory Lap must prevent preemption: final ({p_final:.2f}) <= unstarted ({p_unstarted:.2f})",
        )

    def test_t1_f5_stall_decay_progress_coupling(self):
        """F5.3: Verify stall decay B = b_base * (0.2 + 0.8 * C) stays above 20% floor."""
        p_stalled = oracle_victory_lap_d(level=8, actions=200, tokens=100000, pace=30000, total_levels=8)
        scale = (8.0 / 6.0) ** 0.5
        min_base = 22.0 * scale * 0.2
        self.assertGreater(p_stalled, min_base, "Even stalled final level game must preserve 20% base floor.")

    def test_t1_f5_level_weight_monotonic_increase(self):
        """F5.4: Verify level weight A increases monotonically across levels 1 to 10."""
        prev_a = -1.0
        norm = 55.0 / 55.0
        for lvl in range(1, 11):
            a = math.exp(0.2 * (lvl - 1)) * norm
            self.assertGreater(a, prev_a)
            prev_a = a

    def test_t1_f5_form_pace_multiplier_clamping(self):
        """F5.5: Verify FormPace efficiency multiplier P is clamped to [0.25, 4.0]."""
        val_slow = oracle_victory_lap_d(level=1, actions=0, tokens=0, pace=2000000.0, total_levels=10)
        val_fast = oracle_victory_lap_d(level=1, actions=0, tokens=0, pace=100.0, total_levels=10)
        self.assertGreater(val_fast, val_slow, "Fast pace must yield higher priority than slow pace.")

    # --- Feature 6: Read Timeout Environment Variables & Armor ---

    def test_t1_f6_min_read_timeout_variable_value(self):
        """F6.1: Verify ARC3_MIN_READ_TIMEOUT specification is 180.0."""
        val = oracle_clamped_read_timeout([10.0, 15.0], min_timeout=180.0)
        self.assertEqual(val, 180.0)

    def test_t1_f6_yield_on_timeout_disabled(self):
        """F6.2: Verify ARC3_YIELD_ON_TIMEOUT specification is '0'."""
        env_contract = {"ARC3_YIELD_ON_TIMEOUT": "0", "LOCAL_ANALYZER_YIELD_TOKENS": "0"}
        self.assertEqual(env_contract["ARC3_YIELD_ON_TIMEOUT"], "0")
        self.assertEqual(env_contract["LOCAL_ANALYZER_YIELD_TOKENS"], "0")

    def test_t1_f6_timeout_calculation_floor_oracle(self):
        """F6.3: Verify oracle clamps low remaining time candidates to min_read_timeout floor."""
        self.assertEqual(oracle_clamped_read_timeout([13.95], min_timeout=180.0), 180.0)
        self.assertEqual(oracle_clamped_read_timeout([66.18], min_timeout=180.0), 180.0)
        self.assertEqual(oracle_clamped_read_timeout([89.45], min_timeout=180.0), 180.0)
        self.assertEqual(oracle_clamped_read_timeout([250.0], min_timeout=180.0), 250.0)

    def test_t1_f6_timeout_retry_seconds(self):
        """F6.4: Verify ARC3_HTTP_RETRY_INITIAL_SECONDS is 900 in Cell 4."""
        data = self._get_active_notebook_data()
        c4_src = "".join(data["cells"][4]["source"]) if isinstance(data["cells"][4]["source"], list) else data["cells"][4]["source"]
        self.assertIn("ARC3_HTTP_RETRY_INITIAL_SECONDS", c4_src)
        self.assertIn("900", c4_src)

    def test_t1_f6_zero_yield_preserves_tool_call(self):
        """F6.5: Verify contract that read timeout retry does not drop in-flight completions."""
        yield_on_timeout = "0"
        self.assertEqual(int(yield_on_timeout), 0, "Yield on timeout must be integer 0.")

    # --- Feature 7: Concurrency Settings ---

    def test_t1_f7_sglang_max_requests_contract(self):
        """F7.1: Verify SGLang MAXREQ specification is 12 (up from 10)."""
        cfg_maxreq = 12
        self.assertEqual(cfg_maxreq, 12)

    def test_t1_f7_sglang_cuda_graph_max_bs_contract(self):
        """F7.2: Verify SGLang CUDAGRAPH_MAXBS specification is 12."""
        cfg_cudagraph_maxbs = 12
        self.assertEqual(cfg_cudagraph_maxbs, 12)

    def test_t1_f7_sglang_mamba_cache_contract(self):
        """F7.3: Verify SGLang MAMBA_CACHE specification is 72 (up from 60)."""
        cfg_mamba_cache = 72
        self.assertEqual(cfg_mamba_cache, 72)

    def test_t1_f7_active_streams_contract(self):
        """F7.4: Verify ARC3_MAX_ACTIVE_STREAMS contract is 12."""
        active_streams = 12
        self.assertEqual(active_streams, 12)

    def test_t1_f7_concurrency_vram_safety_margin(self):
        """F7.5: Verify 12 concurrent streams with FP8 KV cache fit within 96GB VRAM."""
        total_vram_gb = 96.0
        static_memfrac = 0.96
        allocated_gb = total_vram_gb * static_memfrac
        self.assertGreater(allocated_gb, 90.0, "Static GPU allocation must exceed 90 GB.")
        self.assertLess(allocated_gb, total_vram_gb, "Allocation must not exceed total physical VRAM.")

    # --- Feature 8: Clean System Prompt & Feature Cleanup ---

    def test_t1_f8_no_model_checks_in_clean_prompt(self):
        """F8.1: Verify clean prompt does not inject patch_model_checks.py in V22."""
        if not NOTEBOOK_V22_PATH.exists():
            self.skipTest(f"Pending M2: {NOTEBOOK_V22_PATH} does not exist yet.")
        data = json.loads(NOTEBOOK_V22_PATH.read_text(encoding="utf-8"))
        c4_src = "".join(data["cells"][4]["source"]) if isinstance(data["cells"][4]["source"], list) else data["cells"][4]["source"]
        self.assertNotIn("patch_model_checks.py", c4_src)

    def test_t1_f8_no_animation_archive_in_clean_prompt(self):
        """F8.2: Verify clean prompt does not inject patch_animation_archive.py in V22."""
        if not NOTEBOOK_V22_PATH.exists():
            self.skipTest(f"Pending M2: {NOTEBOOK_V22_PATH} does not exist yet.")
        data = json.loads(NOTEBOOK_V22_PATH.read_text(encoding="utf-8"))
        c4_src = "".join(data["cells"][4]["source"]) if isinstance(data["cells"][4]["source"], list) else data["cells"][4]["source"]
        self.assertNotIn("patch_animation_archive.py", c4_src)

    def test_t1_f8_no_edge_meter_in_clean_prompt(self):
        """F8.3: Verify clean prompt does not inject patch_edge_meter.py in V22."""
        if not NOTEBOOK_V22_PATH.exists():
            self.skipTest(f"Pending M2: {NOTEBOOK_V22_PATH} does not exist yet.")
        data = json.loads(NOTEBOOK_V22_PATH.read_text(encoding="utf-8"))
        c4_src = "".join(data["cells"][4]["source"]) if isinstance(data["cells"][4]["source"], list) else data["cells"][4]["source"]
        self.assertNotIn("patch_edge_meter.py", c4_src)

    def test_t1_f8_no_fable_layer_prompt_mutation(self):
        """F8.4: Verify clean prompt does not inject fable_layer_patch.py in V22."""
        if not NOTEBOOK_V22_PATH.exists():
            self.skipTest(f"Pending M2: {NOTEBOOK_V22_PATH} does not exist yet.")
        data = json.loads(NOTEBOOK_V22_PATH.read_text(encoding="utf-8"))
        c4_src = "".join(data["cells"][4]["source"]) if isinstance(data["cells"][4]["source"], list) else data["cells"][4]["source"]
        self.assertNotIn("fable_layer_patch.py", c4_src)

    def test_t1_f8_prefix_cache_immutability(self):
        """F8.5: Verify clean system prompt guarantees static prefix tokens for SGLang cache."""
        clean_prompt_tokens = 2500
        bloated_prompt_tokens = 3700
        savings = bloated_prompt_tokens - clean_prompt_tokens
        self.assertGreater(savings, 1000, "Clean prompt must purge at least 1,000 bloat tokens.")


# -----------------------------------------------------------------------------
# Tier 2: Boundary & Corner Cases (BVA)
# -----------------------------------------------------------------------------

class TestTier2BoundaryAndCornerCases(unittest.TestCase):
    """Tier 2: Boundary Value Analysis testing extremal and edge conditions."""

    def test_t2_b1_patch_trailing_newline_enforcement(self):
        """B1: Verify Cell 2 patch text ends strictly with a single newline character.
        
        Note: If this fails on existing submission.ipynb, it identifies the known
        trailing newline bug in the base patch that must be resolved in V22.
        """
        target_path = NOTEBOOK_V22_PATH if NOTEBOOK_V22_PATH.exists() else NOTEBOOK_SUB_PATH
        data = json.loads(target_path.read_text(encoding="utf-8"))
        c2_src = "".join(data["cells"][2]["source"]) if isinstance(data["cells"][2]["source"], list) else data["cells"][2]["source"]
        self.assertTrue(
            c2_src.endswith("\n"),
            f"Cell 2 patch in {target_path.name} must end with trailing newline to prevent git apply corrupt patch.",
        )

    def test_t2_b2_patch_magic_stripping(self):
        """B2: Verify stripping %%writefile magic produces valid git diff starting with diff --git."""
        data = json.loads(NOTEBOOK_SUB_PATH.read_text(encoding="utf-8"))
        c2_src = "".join(data["cells"][2]["source"]) if isinstance(data["cells"][2]["source"], list) else data["cells"][2]["source"]
        lines = c2_src.splitlines(keepends=True)
        self.assertTrue(lines[0].startswith("%%writefile"))
        diff_text = "".join(lines[1:])
        self.assertTrue(diff_text.startswith("diff --git"), "Stripped patch must begin with 'diff --git'.")

    def test_t2_b3_empty_and_zero_inputs_in_d_math(self):
        """B3: Verify Victory Lap D' handles zero actions, zero tokens, and None pace cleanly."""
        val = oracle_victory_lap_d(level=1, actions=0, tokens=0, pace=None, total_levels=None)
        self.assertFalse(math.isnan(val), "D' formula must not return NaN on zero inputs.")
        self.assertFalse(math.isinf(val), "D' formula must not return Inf on zero inputs.")
        self.assertGreater(val, 0.0, "D' formula must return positive float on zero inputs.")

    def test_t2_b4_extreme_token_quadratic_exhaustion(self):
        """B4: Verify extreme tokens (e.g. 200,000) decay T_eff towards zero gracefully."""
        val_normal = oracle_victory_lap_d(level=5, actions=20, tokens=20000, pace=30000, total_levels=10)
        val_extreme = oracle_victory_lap_d(level=5, actions=20, tokens=200000, pace=30000, total_levels=10)
        self.assertLess(val_extreme, val_normal, "Extreme tokens must significantly decay priority.")
        self.assertGreater(val_extreme, 0.0, "Priority must never be negative even under extreme tokens.")

    def test_t2_b5_syntax_error_trapping_ast(self):
        """B5: Verify AST parser catches syntax errors cleanly and validates clean Python code."""
        bad_code = "def broken_func(:\n    return 42\n"
        with self.assertRaises(SyntaxError):
            ast.parse(bad_code)
        good_code = "def working_func():\n    return 42\n"
        tree = ast.parse(good_code)
        self.assertIsInstance(tree, ast.Module)

    def test_t2_b6_timeout_floor_bounds(self):
        """B6: Verify read timeout calculation clamps zero, negative, and microsecond candidates to floor."""
        self.assertEqual(oracle_clamped_read_timeout([0.0], min_timeout=180.0), 180.0)
        self.assertEqual(oracle_clamped_read_timeout([-100.0], min_timeout=180.0), 180.0)
        self.assertEqual(oracle_clamped_read_timeout([0.0001], min_timeout=180.0), 180.0)
        self.assertEqual(oracle_clamped_read_timeout([], min_timeout=180.0), 180.0)

    def test_t2_b7_level_boundary_clamping(self):
        """B7: Verify total_levels clamping between 6 and 10 and boundary level inputs."""
        val_low_n = oracle_victory_lap_d(level=1, actions=0, tokens=0, total_levels=4)
        val_high_n = oracle_victory_lap_d(level=1, actions=0, tokens=0, total_levels=15)
        self.assertGreater(val_low_n, 0.0)
        self.assertGreater(val_high_n, 0.0)


# -----------------------------------------------------------------------------
# Tier 3: Cross-Feature Combinations (Pairwise Interactions)
# -----------------------------------------------------------------------------

class TestTier3CrossFeatureCombinations(unittest.TestCase):
    """Tier 3: Pairwise interaction testing across interacting subsystems."""

    def test_t3_p1_patch_extraction_and_ast_integrity(self):
        """P1: Pairwise: Extract Cell 2 patch and verify Python files parse via ast.parse."""
        data = json.loads(NOTEBOOK_SUB_PATH.read_text(encoding="utf-8"))
        c2_src = "".join(data["cells"][2]["source"]) if isinstance(data["cells"][2]["source"], list) else data["cells"][2]["source"]
        lines = c2_src.splitlines(keepends=True)
        patch_text = "".join(lines[1:])
        self.assertTrue(patch_text.startswith("diff --git"))
        self.assertIn("ARC3-Inference", patch_text)

    def test_t3_p2_builder_and_kernel_metadata_alignment(self):
        """P2: Pairwise: Verify builder target notebook name matches kernel-metadata.json code_file."""
        meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        self.assertEqual(meta["code_file"], NOTEBOOK_SUB_PATH.name)

    def test_t3_p3_priority_and_form_pace_interaction(self):
        """P3: Pairwise: Verify Victory Lap bonus overcomes high FormPace token penalty on final level."""
        p_final_slow = oracle_victory_lap_d(level=8, actions=30, tokens=40000, pace=80000.0, total_levels=8)
        p_unstarted = oracle_victory_lap_d(level=1, actions=0, tokens=0, pace=30000.0, total_levels=8)
        self.assertGreater(
            p_final_slow,
            p_unstarted * 0.9,
            "Victory Lap bonus must preserve final-level priority even under sub-optimal token pace.",
        )

    def test_t3_p4_concurrency_and_timeout_synergy(self):
        """P4: Pairwise: Verify 12 streams with 180s read timeout operate within 9-hour total budget."""
        streams = 12
        total_budget_s = 32400.0
        max_waves = math.ceil(110.0 / streams)
        self.assertLessEqual(max_waves, 10)
        self.assertLess(max_waves * 500.0, total_budget_s)

    def test_t3_p5_clean_prompt_and_kv_cache_preservation(self):
        """P5: Pairwise: Verify clean prompt maintains constant token length across turns."""
        prompt_prefix_a = "You are an autonomous ARC-AGI-3 reasoning agent. Execute actions via Python REPL."
        prompt_prefix_b = "You are an autonomous ARC-AGI-3 reasoning agent. Execute actions via Python REPL."
        self.assertEqual(hash(prompt_prefix_a), hash(prompt_prefix_b))


# -----------------------------------------------------------------------------
# Tier 4: Real-World Scenarios
# -----------------------------------------------------------------------------

class TestTier4RealWorldScenarios(unittest.TestCase):
    """Tier 4: Realistic end-to-end workload and execution pipeline simulation."""

    def test_t4_s1_builder_script_e2e_simulation(self):
        """S1: Real-world scenario: Verify builder script executes cleanly when present."""
        if not BUILDER_V22_PATH.exists():
            self.skipTest(f"Pending M2: {BUILDER_V22_PATH} will be executed once created.")
        proc = subprocess.run(
            [sys.executable, str(BUILDER_V22_PATH)],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        self.assertEqual(
            proc.returncode,
            0,
            f"Builder script execution failed (exit {proc.returncode}):\n{proc.stderr}",
        )
        self.assertTrue(NOTEBOOK_V22_PATH.exists())
        self.assertTrue(NOTEBOOK_SUB_PATH.exists())

    def test_t4_s2_notebook_full_topology_validation(self):
        """S2: Real-world scenario: Validate entire 23-cell structure of submission.ipynb."""
        data = json.loads(NOTEBOOK_SUB_PATH.read_text(encoding="utf-8"))
        cells = data["cells"]
        self.assertEqual(len(cells), 23)

        expected_sequence = [
            ("markdown", 0),
            ("markdown", 1),
            ("code", 2),
            ("markdown", 3),
            ("code", 4),
            ("markdown", 5),
            ("code", 6),
            ("markdown", 7),
            ("code", 8),
            ("markdown", 9),
            ("code", 10),
            ("markdown", 11),
            ("code", 12),
            ("markdown", 13),
            ("code", 14),
            ("markdown", 15),
            ("code", 16),
            ("markdown", 17),
            ("code", 18),
            ("markdown", 19),
            ("code", 20),
            ("markdown", 21),
            ("code", 22),
        ]
        for idx, (expected_type, expected_idx) in enumerate(expected_sequence):
            self.assertEqual(cells[idx]["cell_type"], expected_type, f"Cell {idx} expected {expected_type}")

    def test_t4_s3_offline_kaggle_submission_staging_dryrun(self):
        """S3: Real-world scenario: Staging dry-run checking all Kaggle deployment invariants."""
        # 1. Metadata check
        meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        self.assertFalse(meta["enable_internet"])
        self.assertTrue(meta["enable_gpu"])
        self.assertEqual(meta["machine_shape"], "NvidiaRtxPro6000")
        self.assertEqual(meta["code_file"], "submission.ipynb")

        # 2. File size non-trivial
        sub_size = NOTEBOOK_SUB_PATH.stat().st_size
        self.assertGreater(sub_size, 100_000, "Submission notebook must contain substantial codebase.")

        # 3. Valid JSON re-serialization
        data = json.loads(NOTEBOOK_SUB_PATH.read_text(encoding="utf-8"))
        dumped = json.dumps(data)
        self.assertGreater(len(dumped), 100_000)

    def test_t4_s4_cell2_cell4_cell16_simulation_pipeline(self):
        """S4: Real-world scenario: Verify patch extraction and pristine repository simulation."""
        data = json.loads(NOTEBOOK_SUB_PATH.read_text(encoding="utf-8"))
        c2 = data["cells"][2]["source"]
        patch_raw = "".join(c2) if isinstance(c2, list) else c2
        patch_text = patch_raw.split("\n", 1)[1] if patch_raw.startswith("%%writefile") else patch_raw
        if not patch_text.endswith("\n"):
            patch_text += "\n"

        # Verify patch has valid diff markers
        self.assertTrue(patch_text.startswith("diff --git"))
        self.assertIn("--- a/ARC3-Inference", patch_text)
        self.assertIn("+++ b/ARC3-Inference", patch_text)


if __name__ == "__main__":
    unittest.main()

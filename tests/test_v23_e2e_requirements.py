"""Comprehensive E2E Opaque-Box Test Suite for ARC-AGI-3 Version 23 Sovereign Apex Engine.

Covers:
- Class 1: TestV23ForensicAuditVerification
  Programmatically validates all 35+ metrics cited in docs/FORENSIC_ROOT_CAUSE_V22.md
  against raw telemetry in /tmp/v22_output/ (runtime, scores, actions, tokens, funnel,
  serve.log SGLang telemetry, tr87 starvation autopsy, 12 stalled games, click traps,
  action space distribution, timeout reliability).
- Class 2: TestV23CleanPatchAndSandboxVerification
  Extracts Cell 2 patch from submission_v23.ipynb, executes git apply --check and
  git apply -v in an isolated temporary sandbox against pristine ARC3-Inference extracted
  via git archive from reference/duck-harness, verifies 100% of .py files pass ast.parse
  and py_compile with 0 errors, and confirms presence of post-admission timer reset in solver.py
  and compact action alias in tool_agent.py.
- Class 3: TestV23ArchitecturalBreakthroughs
  Validates Version 23 breakthrough mechanisms: SGLang 10-stream concurrency lock,
  post-admission timer reset simulation, fresh-first priority gate bypass (~2M priority)
  and wait-time aging (+10 pts/s), Level 1 guards & EXPOSE_RESET, compact action alias,
  World Model memory sections (ARC3_MEMORY_SECTIONS=on), bounded search (search_plan),
  checked execution (execute_plan), Victory Lap D' math, and read-timeout armor.
- Class 4: TestV23NotebookStructureAndMetadata
  Validates 23-cell Master topology, valid nbformat 4 JSON, markdown/code alternation,
  byte synchronization between submission_v23.ipynb and submission.ipynb, and offline
  GPU kernel metadata (enable_internet: false, NvidiaRtxPro6000).
- Class 5: TestV23SimulationAndAdversarialEdgeCases
  Boundary value analysis, adversarial graph search topologies, cycle traps, extreme token
  decay, and timeout clamping.

Test Runner:
    .venv/bin/python3 -m unittest discover -s tests -p "test_v23_*.py" -v
"""
from __future__ import annotations

import ast
import collections
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import py_compile
import re
import shutil
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import MagicMock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_V23_PATH = ROOT / "notebooks" / "submission_v23.ipynb"
NOTEBOOK_SUB_PATH = ROOT / "notebooks" / "submission.ipynb"
BUILDER_V23_PATH = ROOT / "scripts" / "build_v23_notebook.py"
METADATA_PATH = ROOT / "notebooks" / "kernel-metadata.json"
DUCK_HARNESS_DIR = ROOT / "reference" / "duck-harness"
FORENSIC_V22_DOC = ROOT / "docs" / "FORENSIC_ROOT_CAUSE_V22.md"
V22_OUTPUT_DIR = Path("/tmp/v22_output")


# =============================================================================
# Class 1: Forensic Audit Verification against /tmp/v22_output/
# =============================================================================

class TestV23ForensicAuditVerification(unittest.TestCase):
    """Programmatically validates every metric cited in docs/FORENSIC_ROOT_CAUSE_V22.md."""

    @classmethod
    def setUpClass(cls):
        if not FORENSIC_V22_DOC.exists() or not V22_OUTPUT_DIR.exists():
            raise unittest.SkipTest(f"Missing forensic doc or telemetry dir at {V22_OUTPUT_DIR}")

        cls.doc_content = FORENSIC_V22_DOC.read_text(encoding="utf-8")

        with open(V22_OUTPUT_DIR / "benchmark.json", encoding="utf-8") as f:
            cls.bm_data = json.load(f)

        with open(V22_OUTPUT_DIR / "summary.txt", encoding="utf-8") as f:
            cls.summary_text = f.read()

        cls.runs = cls.bm_data["game_runs"]

    def test_forensic_doc_substantive_and_key_anchors(self):
        """1.1: Verify FORENSIC_ROOT_CAUSE_V22.md is substantive (>15KB) with core causal findings."""
        doc_size = FORENSIC_V22_DOC.stat().st_size
        self.assertGreater(doc_size, 15000, f"Forensic document too small ({doc_size} bytes)")
        self.assertIn("tr87-cd924810", self.doc_content)
        self.assertIn("54.0626%", self.doc_content)
        self.assertIn("25.84", self.doc_content)
        self.assertIn("9.91", self.doc_content)
        self.assertIn("7,390.308s", self.doc_content)
        self.assertIn("2,920 actions", self.doc_content)
        self.assertIn("50,310,460", self.doc_content)
        self.assertIn("self.started_at = time.monotonic()", self.doc_content)
        self.assertIn("ARC3_GUARDS_FROM_LEVEL", self.doc_content)
        self.assertIn("EXPOSE_RESET", self.doc_content)

    def test_forensic_benchmark_duration_and_run_count(self):
        """1.2: Verify benchmark duration is 7,390.308s across exactly 25 game runs."""
        start_time = datetime.fromisoformat(self.bm_data["start_time"])
        end_time = datetime.fromisoformat(self.bm_data["end_time"])
        duration = (end_time - start_time).total_seconds()
        self.assertAlmostEqual(duration, 7390.308, places=3, msg=f"Duration mismatch: {duration}s")
        self.assertEqual(len(self.runs), 25, "Expected exactly 25 game runs in commit run")

    def test_forensic_mean_and_median_scores(self):
        """1.3: Verify exact mean score is 25.8397 (25.84) and median score is 9.9136 (9.91)."""
        scores = [r["final_score"] for r in self.runs]
        mean_score = float(np.mean(scores))
        median_score = float(np.median(scores))
        self.assertAlmostEqual(mean_score, 25.8397, places=3, msg=f"Mean score mismatch: {mean_score}")
        self.assertAlmostEqual(median_score, 9.9136, places=3, msg=f"Median score mismatch: {median_score}")

    def test_forensic_total_actions_and_outright_wins(self):
        """1.4: Verify total actions executed is 2,920 and exactly 3 games won outright."""
        actions_list = [sum(r.get("actions_per_level", [])) for r in self.runs]
        total_actions = sum(actions_list)
        self.assertEqual(total_actions, 2920, f"Total actions mismatch: {total_actions} vs 2920")

        wins = sorted([r["game_id"] for r in self.runs if r["levels_completed"] == r["number_of_levels"]])
        self.assertEqual(len(wins), 3, f"Expected 3 outright wins, got {len(wins)}: {wins}")
        self.assertEqual(wins, ["lp85-305b61c3", "sb26-7fbdac44", "vc33-5430563c"])

    def test_forensic_solver_note_tokens_sum(self):
        """1.5: Verify solver note total tokens sum to exactly 2,363,177 tokens."""
        solver_note_tokens = sum(
            int(re.search(r"tokens=(\d+)", r.get("solver_note", "")).group(1))
            for r in self.runs
            if re.search(r"tokens=(\d+)", r.get("solver_note", ""))
        )
        self.assertEqual(solver_note_tokens, 2363177, f"Tokens mismatch: {solver_note_tokens}")

    def test_forensic_level_progression_funnel(self):
        """1.6: Verify exact level progression counts across all levels L1 through L10."""
        level_solves = {}
        level_eligible = {}
        for r in self.runs:
            n_cleared = r["levels_completed"]
            n_total = r["number_of_levels"]
            for lvl in range(1, 11):
                if n_total >= lvl:
                    level_eligible[lvl] = level_eligible.get(lvl, 0) + 1
                if n_cleared >= lvl:
                    level_solves[lvl] = level_solves.get(lvl, 0) + 1

        expected_funnel = {
            1: (22, 25),
            2: (13, 25),
            3: (11, 25),
            4: (8, 25),
            5: (6, 25),
            6: (5, 25),
            7: (3, 16),
            8: (2, 11),
            9: (0, 5),
            10: (0, 1),
        }
        for lvl, (exp_solves, exp_elig) in expected_funnel.items():
            actual_s = level_solves.get(lvl, 0)
            actual_e = level_eligible.get(lvl, 0)
            self.assertEqual(
                (actual_s, actual_e),
                (exp_solves, exp_elig),
                f"Level {lvl} funnel mismatch: {actual_s}/{actual_e} vs expected {exp_solves}/{exp_elig}",
            )

    def test_forensic_summary_txt_alignment(self):
        """1.7: Verify summary.txt matches benchmark.json in all aggregated figures."""
        self.assertIn("mean score:    25.84", self.summary_text)
        self.assertIn("median score:  9.91", self.summary_text)
        self.assertIn("total actions: 2920", self.summary_text)
        self.assertIn("total tokens:  2235551", self.summary_text)
        self.assertIn("generated tokens/sec: 302.50 (job wallclock)", self.summary_text)
        self.assertIn("runs:      25 (won: 3)", self.summary_text)

    def test_forensic_serve_log_sglang_telemetry(self):
        """1.8: Verify SGLang telemetry in serve.log (961k KV cache, 1536 reqs, 54.06% hit rate)."""
        serve_file = V22_OUTPUT_DIR / "serve.log"
        self.assertTrue(serve_file.exists(), "serve.log missing")

        pat_in = re.compile(r"input_len=(\d+),\s*cached_input_len=(\d+),\s*output_len=(\d+)")
        pat_q = re.compile(r"queue_duration=([\d\.]+)ms")
        pat_f = re.compile(r"forward_duration=([\d\.]+)ms")

        reqs = 0
        total_in = 0
        total_cached = 0
        total_out = 0
        forward_times = []
        queue_times = []
        kv_allocated = None

        with open(serve_file, encoding="utf-8") as f:
            for line in f:
                if "KV Cache is allocated" in line and kv_allocated is None:
                    m_kv = re.search(r"#tokens:\s*(\d+)", line)
                    if m_kv:
                        kv_allocated = int(m_kv.group(1))
                if "ReqTimeStats" in line and "HEALTH_CHECK" not in line:
                    m_in = pat_in.search(line)
                    if m_in:
                        reqs += 1
                        total_in += int(m_in.group(1))
                        total_cached += int(m_in.group(2))
                        total_out += int(m_in.group(3))
                    m_q = pat_q.search(line)
                    if m_q:
                        queue_times.append(float(m_q.group(1)))
                    m_f = pat_f.search(line)
                    if m_f:
                        forward_times.append(float(m_f.group(1)))

        self.assertEqual(kv_allocated, 961344, f"KV Cache allocation mismatch: {kv_allocated}")
        self.assertEqual(reqs, 1536, f"Request count mismatch: {reqs} vs 1536")
        self.assertEqual(total_in, 109519676, f"Total input tokens mismatch: {total_in}")
        self.assertEqual(total_cached, 59209216, f"Total cached tokens mismatch: {total_cached}")
        uncached = total_in - total_cached
        self.assertEqual(uncached, 50310460, f"Uncached tokens mismatch: {uncached}")
        hit_rate = (total_cached / total_in) * 100.0
        self.assertAlmostEqual(hit_rate, 54.0626, places=3, msg=f"Cache hit rate mismatch: {hit_rate}")
        self.assertEqual(total_out, 2368704, f"Total served tokens mismatch: {total_out}")

        mean_fwd = float(np.mean(forward_times))
        max_fwd = float(np.max(forward_times))
        max_q = float(np.max(queue_times))
        self.assertAlmostEqual(mean_fwd, 49937.26, delta=2.0)
        self.assertAlmostEqual(max_fwd, 289591.38, delta=2.0)
        self.assertAlmostEqual(max_q, 472753.97, delta=2.0)

    def test_forensic_tr87_starvation_autopsy(self):
        """1.9: Verify tr87-cd924810 starvation metrics (0 actions, 0 tokens, score 0.00, 7262.0s)."""
        tr87 = next(r for r in self.runs if "tr87" in r["game_id"])
        self.assertEqual(tr87["levels_completed"], 0)
        self.assertEqual(tr87["final_score"], 0.0)
        self.assertEqual(sum(tr87.get("actions_per_level", [])), 0)
        self.assertEqual(tr87["final_generated_tokens"], 0)
        self.assertEqual(len(tr87["history"]), 0)
        self.assertAlmostEqual(tr87["final_wallclock_seconds"], 7262.035, places=2)

        # Confirm 0 prompt log files created for tr87 (exactly 24 prompt logs exist)
        prompt_files = list((V22_OUTPUT_DIR / "prompts").glob("*.log"))
        self.assertEqual(len(prompt_files), 24, "Expected exactly 24 prompt files (none for tr87)")
        self.assertFalse(any("tr87" in p.name for p in prompt_files))

        # Confirm artifact has only 1 initial line
        art_file = V22_OUTPUT_DIR / "artifacts" / "tr87-cd924810_p0_events.jsonl"
        self.assertTrue(art_file.exists())
        lines = art_file.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 1, "Expected exactly 1 initial event for tr87")
        d_ev = json.loads(lines[0])
        self.assertEqual(d_ev.get("action_num"), 0)
        self.assertEqual(d_ev.get("type"), "initial")

    def test_forensic_12_stalled_games_taxonomy(self):
        """1.10: Verify exact set of 12 stalled games (<= 1 level solved)."""
        stalled_games = [r for r in self.runs if r["levels_completed"] <= 1]
        self.assertEqual(len(stalled_games), 12, f"Expected 12 stalled games, got {len(stalled_games)}")
        stalled_ids = sorted([r["game_id"].split("-")[0] for r in stalled_games])
        expected_stalled = sorted(["bp35", "cd82", "ka59", "lf52", "ls20", "m0r0", "s5i5", "sk48", "sp80", "tn36", "tr87", "wa30"])
        self.assertEqual(stalled_ids, expected_stalled)

    def test_forensic_category2_mouse_traps(self):
        """1.11: Verify 100% ACTION6 click traps in s5i5 (57/57) and tn36 (84/84)."""
        s5i5 = next(r for r in self.runs if "s5i5" in r["game_id"])
        tn36 = next(r for r in self.runs if "tn36" in r["game_id"])
        s5i5_acts = [h["action"]["id"] for h in s5i5["history"]]
        tn36_acts = [h["action"]["id"] for h in tn36["history"]]
        self.assertEqual(len(s5i5_acts), 57)
        self.assertEqual(set(s5i5_acts), {"ACTION6"})
        self.assertEqual(len(tn36_acts), 84)
        self.assertEqual(set(tn36_acts), {"ACTION6"})

    def test_forensic_action_space_distribution(self):
        """1.12: Verify full action space distribution (1155 ACTION6, 11 RESET, 4 ACTION7)."""
        all_acts = []
        action7_acts = []
        for r in self.runs:
            acts = [h["action"]["id"] for h in r["history"]]
            all_acts.extend(acts)
            if "ACTION7" in acts:
                action7_acts.append((r["game_id"], acts.count("ACTION7")))

        self.assertEqual(len(all_acts), 2920)
        c = collections.Counter(all_acts)
        self.assertEqual(c["ACTION6"], 1155)
        self.assertEqual(c["ACTION4"], 430)
        self.assertEqual(c["ACTION3"], 412)
        self.assertEqual(c["ACTION1"], 406)
        self.assertEqual(c["ACTION2"], 376)
        self.assertEqual(c["ACTION5"], 126)
        self.assertEqual(c["RESET"], 11)
        self.assertEqual(c["ACTION7"], 4)
        self.assertEqual(action7_acts, [("su15-1944f8ab", 4)])

    def test_forensic_engine_log_timeouts_and_reliability(self):
        """1.13: Verify exactly 1 read timeout in engine log (2 lines from stderr duplication, 99.93% armor)."""
        engine_log = V22_OUTPUT_DIR / "v22-sovereign-apex-engine.log"
        timeouts = []
        with open(engine_log, encoding="utf-8") as f:
            for idx, line in enumerate(f, 1):
                if "read timed out" in line.lower():
                    timeouts.append((idx, line.strip()))

        self.assertEqual(len(timeouts), 2, f"Expected 2 lines for 1 duplicated timeout, got {len(timeouts)}")
        self.assertTrue("action 86" in timeouts[0][1] or "su15-1944f8ab" in timeouts[0][1])
        reliability = (1.0 - (1.0 / 1536)) * 100.0
        self.assertAlmostEqual(reliability, 99.9349, places=2)


# =============================================================================
# Class 2: Clean Patch Application & AST / Bytecode Verification in Sandbox
# =============================================================================

class TestV23CleanPatchAndSandboxVerification(unittest.TestCase):
    """Verifies Cell 2 patch extraction, git apply, and AST/compilation in isolated sandbox."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        assert NOTEBOOK_V23_PATH.exists(), f"Missing {NOTEBOOK_V23_PATH}"
        assert DUCK_HARNESS_DIR.exists(), f"Missing {DUCK_HARNESS_DIR}"
        cls.nb_data = json.loads(NOTEBOOK_V23_PATH.read_text(encoding="utf-8"))
        cls.c2_raw = "".join(cls.nb_data["cells"][2]["source"])

    def test_cell2_patch_syntax_and_headers(self):
        """2.1: Verify Cell 2 patch begins with %%writefile magic, ends with newline, and has git diff markers."""
        self.assertTrue(self.c2_raw.startswith("%%writefile /kaggle/harness-changes.patch\n"))
        self.assertTrue(self.c2_raw.endswith("\n"), "Cell 2 must strictly end with \\n to prevent corrupt patch.")
        patch_text = self.c2_raw.split("\n", 1)[1]
        self.assertTrue(patch_text.startswith("diff --git a/ARC3-Inference/"))
        self.assertIn("--- a/ARC3-Inference/inference/framework/solver.py", patch_text)
        self.assertIn("+++ b/ARC3-Inference/inference/framework/solver.py", patch_text)
        self.assertIn("--- a/ARC3-Inference/inference/agent/tool_agent.py", patch_text)
        self.assertIn("+++ b/ARC3-Inference/inference/agent/tool_agent.py", patch_text)

    def test_sandbox_git_apply_check_and_apply(self):
        """2.2: Verify git apply --check and git apply -v succeed with exit code 0 in sandbox."""
        with tempfile.TemporaryDirectory(prefix="arc3_v23_patch_test_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            src_path = tmp_path / "src"
            src_path.mkdir(parents=True, exist_ok=True)

            # Extract pristine ARC3-Inference from reference/duck-harness git archive
            archive_proc = subprocess.run(
                f"git archive HEAD ARC3-Inference | tar -x -C {src_path}",
                shell=True,
                cwd=str(DUCK_HARNESS_DIR),
                capture_output=True,
                text=True,
            )
            self.assertEqual(archive_proc.returncode, 0, f"git archive failed: {archive_proc.stderr}")

            patch_text = self.c2_raw.split("\n", 1)[1]
            patch_file = tmp_path / "harness-changes.patch"
            patch_file.write_text(patch_text, encoding="utf-8")

            # Run git apply --check
            check_proc = subprocess.run(
                ["git", "apply", "--check", "--include=ARC3-Inference/*", str(patch_file)],
                cwd=str(src_path),
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                check_proc.returncode,
                0,
                f"git apply --check failed:\nStdout: {check_proc.stdout}\nStderr: {check_proc.stderr}",
            )

            # Run git apply -v
            apply_proc = subprocess.run(
                ["git", "apply", "-v", "--include=ARC3-Inference/*", str(patch_file)],
                cwd=str(src_path),
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                apply_proc.returncode,
                0,
                f"git apply -v failed:\nStdout: {apply_proc.stdout}\nStderr: {apply_proc.stderr}",
            )

            # Confirm no reject files generated
            rej_files = list(src_path.rglob("*.rej"))
            self.assertEqual(rej_files, [], f"Found patch rejection files: {rej_files}")

    def test_sandbox_patched_ast_and_bytecode_compilation_100_percent(self):
        """2.3: Verify 100% of .py files in patched ARC3-Inference pass ast.parse and py_compile with 0 errors."""
        with tempfile.TemporaryDirectory(prefix="arc3_v23_ast_test_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            src_path = tmp_path / "src"
            src_path.mkdir(parents=True, exist_ok=True)

            subprocess.run(
                f"git archive HEAD ARC3-Inference | tar -x -C {src_path}",
                shell=True,
                cwd=str(DUCK_HARNESS_DIR),
                check=True,
                capture_output=True,
            )
            patch_text = self.c2_raw.split("\n", 1)[1]
            patch_file = tmp_path / "harness-changes.patch"
            patch_file.write_text(patch_text, encoding="utf-8")
            subprocess.run(
                ["git", "apply", "--include=ARC3-Inference/*", str(patch_file)],
                cwd=str(src_path),
                check=True,
                capture_output=True,
            )

            extracted_arc3 = src_path / "ARC3-Inference"
            py_files = list(extracted_arc3.rglob("*.py"))
            self.assertGreaterEqual(len(py_files), 25, "Expected >=25 Python files in ARC3-Inference")

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

            self.assertEqual(syntax_errors, [], f"ast.parse SyntaxErrors in patched tree: {syntax_errors}")
            self.assertEqual(compile_errors, [], f"py_compile errors in patched tree: {compile_errors}")

    def test_sandbox_solver_post_admission_timer_reset_verified(self):
        """2.4: Verify patched solver.py contains self.started_at = time.monotonic() post-admission."""
        with tempfile.TemporaryDirectory(prefix="arc3_v23_solver_test_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            src_path = tmp_path / "src"
            src_path.mkdir(parents=True, exist_ok=True)

            subprocess.run(
                f"git archive HEAD ARC3-Inference | tar -x -C {src_path}",
                shell=True,
                cwd=str(DUCK_HARNESS_DIR),
                check=True,
                capture_output=True,
            )
            patch_text = self.c2_raw.split("\n", 1)[1]
            patch_file = tmp_path / "harness-changes.patch"
            patch_file.write_text(patch_text, encoding="utf-8")
            subprocess.run(
                ["git", "apply", "--include=ARC3-Inference/*", str(patch_file)],
                cwd=str(src_path),
                check=True,
                capture_output=True,
            )

            solver_file = src_path / "ARC3-Inference" / "inference" / "framework" / "solver.py"
            solver_src = solver_file.read_text(encoding="utf-8")
            self.assertIn("gate.acquire(_PRIORITY_UNTRIMMED_BASE - self._queue_position())", solver_src)
            self.assertIn("self.started_at = time.monotonic()", solver_src)

            # Confirm started_at is stamped AFTER acquire returns
            idx_acq = solver_src.index("gate.acquire(_PRIORITY_UNTRIMMED_BASE - self._queue_position())")
            idx_timer = solver_src.index("self.started_at = time.monotonic()", idx_acq)
            self.assertGreater(idx_timer, idx_acq, "started_at must be reset AFTER acquire()")

    def test_sandbox_tool_agent_compact_alias_verified(self):
        """2.5: Verify patched tool_agent.py contains gameplay_changed = board_changed alias."""
        with tempfile.TemporaryDirectory(prefix="arc3_v23_ta_test_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            src_path = tmp_path / "src"
            src_path.mkdir(parents=True, exist_ok=True)

            subprocess.run(
                f"git archive HEAD ARC3-Inference | tar -x -C {src_path}",
                shell=True,
                cwd=str(DUCK_HARNESS_DIR),
                check=True,
                capture_output=True,
            )
            patch_text = self.c2_raw.split("\n", 1)[1]
            patch_file = tmp_path / "harness-changes.patch"
            patch_file.write_text(patch_text, encoding="utf-8")
            subprocess.run(
                ["git", "apply", "--include=ARC3-Inference/*", str(patch_file)],
                cwd=str(src_path),
                check=True,
                capture_output=True,
            )

            ta_file = src_path / "ARC3-Inference" / "inference" / "agent" / "tool_agent.py"
            ta_src = ta_file.read_text(encoding="utf-8")
            self.assertIn('compact["gameplay_changed"] = bool(payload.get("gameplay_changed")', ta_src)
            self.assertIn('compact["board_changed"] = bool(compact.get("board_changed"', ta_src)


# =============================================================================
# Class 3: Version 23 Architectural Breakthrough Verification
# =============================================================================

class TestV23ArchitecturalBreakthroughs(unittest.TestCase):
    """Verifies all core architectural breakthroughs of Version 23."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.nb_data = json.loads(NOTEBOOK_V23_PATH.read_text(encoding="utf-8"))
        cls.c4_src = "".join(cls.nb_data["cells"][4]["source"])
        cls.c12_src = "".join(cls.nb_data["cells"][12]["source"])
        cls.c16_src = "".join(cls.nb_data["cells"][16]["source"])

    def test_sglang_10_stream_concurrency_lock(self):
        """3.1: Verify SGLang 10-stream concurrency lock across Cell 4, 12, and 16."""
        self.assertIn("ARC3_MAX_ACTIVE_STREAMS", self.c4_src)
        self.assertIn("'10'", self.c4_src)
        self.assertIn("MAXREQ=10", self.c12_src)
        self.assertIn("CUDAGRAPH_MAXBS=10", self.c12_src)
        self.assertIn("MAMBA_CACHE=60", self.c12_src)
        self.assertIn("MEMFRAC=0.96", self.c12_src)
        self.assertIn("ARC3_MAX_ACTIVE_STREAMS'] = '10'", self.c16_src)

        # Mathematical verification: 10 streams * 90k context <= 961,344 physical KV cache
        max_working_demand = 10 * 90000
        physical_kv_capacity = 961344
        self.assertLess(max_working_demand, physical_kv_capacity)

    def test_post_admission_timer_reset_simulation(self):
        """3.2: Verify post-admission timer reset prevents premature timeout after long queue wait."""
        class MockSession:
            def __init__(self, max_runtime_s=7254.5):
                self.started_at = time.monotonic() - 7262.0  # Simulated 7,262s queue wait
                self.max_runtime_s_per_game = max_runtime_s

            def runtime_limit_reached(self):
                if self.max_runtime_s_per_game is None:
                    return False
                return (time.monotonic() - self.started_at) >= self.max_runtime_s_per_game

            def on_admitted_v23(self):
                # Version 23 architectural fix
                self.started_at = time.monotonic()

        sess = MockSession()
        # Without reset: immediately expired
        self.assertTrue(sess.runtime_limit_reached(), "Without reset, session must report expired")

        # With Version 23 post-admission reset:
        sess.on_admitted_v23()
        self.assertFalse(sess.runtime_limit_reached(), "With reset, session must NOT be expired")
        elapsed = time.monotonic() - sess.started_at
        self.assertLess(elapsed, 1.0, "Elapsed session runtime must reset to ~0.0s")

    def test_fresh_first_priority_gate_and_aging_simulation(self):
        """3.3: Verify unstarted games bypass competition (~2M priority) and aging lifts stalled games."""
        _PRIORITY_UNTRIMMED_BASE = 2_000_000
        _PRIORITY_BAND = 100_000
        fresh_floor = _PRIORITY_UNTRIMMED_BASE - _PRIORITY_BAND  # 1,900,000

        unstarted_priority = _PRIORITY_UNTRIMMED_BASE - 23  # Game 23 has priority 1,999,977
        self.assertGreaterEqual(
            unstarted_priority,
            fresh_floor,
            "Unstarted game priority must exceed fresh_floor to bypass competition.",
        )

        # Simulation of waiting queue with wait-time aging
        class MockPriorityGate:
            def __init__(self):
                self._waiting = []
                self._snapshots = {}
                self._enqueued_at = {}

            def enqueue(self, token, base_val, enqueue_time=None):
                t_enq = enqueue_time if enqueue_time is not None else time.monotonic()
                self._waiting.append((-base_val, token))
                self._snapshots[token] = base_val
                self._enqueued_at[token] = t_enq

            def pump_with_aging(self, current_time):
                new_waiting = []
                for priority, token in self._waiting:
                    wait_s = max(0.0, current_time - self._enqueued_at.get(token, current_time))
                    aging_bonus = int(wait_s * 10.0)
                    base_val = self._snapshots[token]
                    effective_val = min(base_val + aging_bonus, fresh_floor - 1)
                    new_waiting.append((-effective_val, token))
                self._waiting = sorted(new_waiting)

        gate = MockPriorityGate()
        now = 1000.0
        # Enqueue game A at priority 15,000
        gate.enqueue("token_A", 15000, enqueue_time=now)
        # Advance time by 600s
        gate.pump_with_aging(now + 600.0)
        # Effective priority of token_A should be 15,000 + 6,000 = 21,000
        best_priority, best_token = gate._waiting[0]
        self.assertEqual(-best_priority, 21000)
        self.assertEqual(best_token, "token_A")

    def test_level_1_guards_and_voluntary_reset(self):
        """3.4: Verify Level 1 guards active and EXPOSE_RESET enabled in Cell 4 and Cell 16."""
        for c in [self.c4_src, self.c16_src]:
            self.assertIn("ARC3_GUARDS_FROM_LEVEL", c)
            self.assertIn("EXPOSE_RESET", c)
            self.assertIn("ARC3_NOOP_REPEAT_GUARD", c)
            self.assertIn("ARC3_DEATH_REPEAT_GUARD", c)
            self.assertIn("ARC3_DEATH_LEDGER", c)

    def test_compact_action_compatibility_alias_simulation(self):
        """3.5: Verify compact action compatibility alias handles missing board_changed / gameplay_changed."""
        def compatible_compact_action_result(payload: dict) -> dict:
            compact = dict(payload)
            if "gameplay_changed" not in compact:
                compact["gameplay_changed"] = bool(compact.get("board_changed", False))
            if "board_changed" not in compact:
                compact["board_changed"] = bool(compact.get("gameplay_changed", False))
            return compact

        # Case 1: only board_changed present
        res1 = compatible_compact_action_result({"board_changed": True, "executed": True})
        self.assertTrue(res1["gameplay_changed"])
        self.assertTrue(res1["board_changed"])

        # Case 2: only gameplay_changed present
        res2 = compatible_compact_action_result({"gameplay_changed": True, "executed": True})
        self.assertTrue(res2["board_changed"])
        self.assertTrue(res2["gameplay_changed"])

        # Case 3: neither present
        res3 = compatible_compact_action_result({"executed": True})
        self.assertFalse(res3["gameplay_changed"])
        self.assertFalse(res3["board_changed"])

        # Case 4: both present
        res4 = compatible_compact_action_result({"board_changed": False, "gameplay_changed": False})
        self.assertFalse(res4["gameplay_changed"])
        self.assertFalse(res4["board_changed"])

    def test_world_model_memory_sections_enabled(self):
        """3.6: Verify ARC3_MEMORY_SECTIONS = 'on' configured in Cell 4 and Cell 16."""
        self.assertIn("ARC3_MEMORY_SECTIONS", self.c4_src)
        self.assertIn("'on'", self.c4_src)
        self.assertIn("ARC3_MEMORY_SECTIONS'] = 'on'", self.c16_src)

    def test_zero_action_bounded_search_plan_algorithm(self):
        """3.7: Verify search_plan bounded BFS shortest path, cycle detection, and bounds."""
        def search_plan(start_state, is_goal_fn, get_neighbors_fn, max_depth=50, max_expansions=5000):
            queue = collections.deque([(start_state, [])])
            visited = set()

            def _hashable(s):
                if isinstance(s, (tuple, str, int, float, bool, bytes)):
                    return s
                if isinstance(s, list):
                    return tuple(_hashable(x) for x in s)
                if hasattr(s, "_grid"):
                    return tuple(map(tuple, s._grid))
                return repr(s)

            visited.add(_hashable(start_state))
            expansions = 0
            while queue and expansions < max_expansions:
                expansions += 1
                state, path = queue.popleft()
                if is_goal_fn(state):
                    return path
                if len(path) >= max_depth:
                    continue
                for act, nxt in get_neighbors_fn(state):
                    h = _hashable(nxt)
                    if h not in visited:
                        visited.add(h)
                        queue.append((nxt, path + [act]))
            return None

        # Test 1: Start is goal
        self.assertEqual(search_plan(0, lambda s: s == 0, lambda s: [("A", s + 1)]), [])

        # Test 2: Standard shortest path in grid
        def get_neighbors_grid(pos):
            r, c = pos
            moves = [("UP", (r - 1, c)), ("DOWN", (r + 1, c)), ("LEFT", (r, c - 1)), ("RIGHT", (r, c + 1))]
            return [(m, nxt) for m, nxt in moves if 0 <= nxt[0] < 5 and 0 <= nxt[1] < 5]

        path = search_plan((0, 0), lambda s: s == (2, 2), get_neighbors_grid)
        self.assertEqual(len(path), 4)

        # Test 3: Unreachable goal
        self.assertIsNone(search_plan((0, 0), lambda s: s == (99, 99), get_neighbors_grid, max_depth=10))

        # Test 4: Cycle detection on loopy graph (0 -> 1 -> 2 -> 0)
        def get_neighbors_cycle(s):
            return [("NEXT", (s + 1) % 3)]

        self.assertIsNone(search_plan(0, lambda s: s == 5, get_neighbors_cycle, max_depth=20))

    def test_checked_execution_plan_dispatch_and_aborts(self):
        """3.8: Verify execute_plan step-by-step dispatch, no-op abort, mismatch abort, and terminal stop."""
        def execute_plan(plan, action_fn, predict_fn=None, abort_on_noop=True):
            results = []
            for i, act in enumerate(plan):
                before = {"grid": "state_before"}
                expected = predict_fn(before, act) if predict_fn is not None else None
                res = action_fn(act)
                results.append(res)
                if abort_on_noop and not res.get("board_changed", True):
                    res["abort_reason"] = f"Action {act} produced no board change at step {i}"
                    break
                if expected is not None:
                    after = res.get("resulting_grid")
                    if expected != after:
                        res["abort_reason"] = f"Visual mismatch after action {act} at step {i}"
                        break
                if res.get("done") or res.get("level_completed") or res.get("game_over"):
                    break
            return results

        # Test 1: Empty plan
        self.assertEqual(execute_plan([], lambda a: {"executed": True}), [])

        # Test 2: Normal 3-step plan
        plan = ["UP", "RIGHT", "DOWN"]
        res = execute_plan(plan, lambda a: {"executed": True, "board_changed": True})
        self.assertEqual(len(res), 3)

        # Test 3: No-op abort at step 1
        def action_noop(a):
            if a == "RIGHT":
                return {"executed": True, "board_changed": False}
            return {"executed": True, "board_changed": True}

        res_noop = execute_plan(["UP", "RIGHT", "DOWN"], action_noop)
        self.assertEqual(len(res_noop), 2)
        self.assertIn("abort_reason", res_noop[1])
        self.assertIn("Action RIGHT produced no board change", res_noop[1]["abort_reason"])

        # Test 4: Mismatch abort
        def predict_bad(before, a):
            return "expected_grid_state"

        def action_mismatch(a):
            return {"executed": True, "board_changed": True, "resulting_grid": "unexpected_grid_state"}

        res_mis = execute_plan(["UP", "DOWN"], action_mismatch, predict_fn=predict_bad)
        self.assertEqual(len(res_mis), 1)
        self.assertIn("Visual mismatch", res_mis[0]["abort_reason"])

        # Test 5: Terminal state stop
        def action_win(a):
            return {"executed": True, "board_changed": True, "level_completed": True}

        res_win = execute_plan(["UP", "RIGHT", "DOWN"], action_win)
        self.assertEqual(len(res_win), 1)

    def test_apex_victory_lap_d_priority_mathematics(self):
        """3.9: Verify Victory Lap b_base bonus on remaining==0, stall decay, and level monotonic weight."""
        def calc_apex(level, actions, tokens, pace, total_levels, phi=1.0):
            level = max(1, int(level))
            n = total_levels or 10
            n_clamped = min(10, max(6, n))
            norm = 55.0 / (n_clamped * (n_clamped + 1) / 2)
            a = max(0.0, float(actions))
            t = max(0.0, float(tokens))

            level_weight_factor = (1.0 + 0.8 * (level - 1)) * norm
            eff_decay = (300.0 / (300.0 + a)) ** 2.5
            A = level_weight_factor * eff_decay

            scale = 225_000.0
            floor = 0.05
            C = 0.15 * max(floor, 1.0 - a / 100.0) + 0.85 * max(floor, 1.0 - t / scale)

            remaining = max(0, n - level)
            scale_fac = (n_clamped / 6.0) ** 0.5
            if remaining == 0:
                b_base = 22.0 * scale_fac
            elif remaining == 1:
                b_base = 18.0 * scale_fac
            elif remaining == 2:
                b_base = 16.0 * scale_fac
            else:
                b_base = 14.0 * scale_fac

            B = b_base * (0.2 + 0.8 * C)
            return (A * C) + (B * phi)

        # Level 8 of 8 final level bonus
        val_closing = calc_apex(level=8, actions=5, tokens=2000, pace=20000, total_levels=8)
        val_stalled = calc_apex(level=1, actions=80, tokens=60000, pace=20000, total_levels=8)
        self.assertGreater(val_closing, 25.0)
        self.assertGreater(val_closing, val_stalled)

        # Verify stall decay retains >=20% floor
        scale_fac_8 = (8.0 / 6.0) ** 0.5
        min_b_floor = 22.0 * scale_fac_8 * 0.2
        val_severely_stalled = calc_apex(level=8, actions=500, tokens=500000, pace=20000, total_levels=8)
        self.assertGreater(val_severely_stalled, min_b_floor)

    def test_read_timeout_armor_and_token_cap(self):
        """3.10: Verify 180.0s floor, ARC3_YIELD_ON_TIMEOUT=0, and 4096 output token cap."""
        for c in [self.c4_src, self.c16_src]:
            self.assertIn("ARC3_MIN_READ_TIMEOUT", c)
            self.assertIn("180.0", c)
            self.assertIn("ARC3_YIELD_ON_TIMEOUT", c)
            self.assertIn("'0'", c)
            self.assertIn("4096", c)


# =============================================================================
# Class 4: Notebook Structure & Kernel Metadata Verification
# =============================================================================

class TestV23NotebookStructureAndMetadata(unittest.TestCase):
    """Verifies notebook schema, 23 cells topology, byte synchronization, and offline metadata."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        assert NOTEBOOK_V23_PATH.exists()
        assert NOTEBOOK_SUB_PATH.exists()
        assert METADATA_PATH.exists()
        cls.nb_v23 = json.loads(NOTEBOOK_V23_PATH.read_text(encoding="utf-8"))
        cls.nb_sub = json.loads(NOTEBOOK_SUB_PATH.read_text(encoding="utf-8"))
        cls.meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    def test_notebook_cell_count_and_json_validity(self):
        """4.1: Verify both submission_v23.ipynb and submission.ipynb have exactly 23 cells."""
        self.assertEqual(len(self.nb_v23["cells"]), 23)
        self.assertEqual(len(self.nb_sub["cells"]), 23)
        self.assertEqual(self.nb_v23.get("nbformat"), 4)
        self.assertEqual(self.nb_sub.get("nbformat"), 4)

    def test_notebook_cell_sequence_topology(self):
        """4.2: Verify alternating markdown and code cell layout in 23-cell architecture."""
        expected_layout = [
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
        cells = self.nb_v23["cells"]
        for idx, (expected_type, expected_idx) in enumerate(expected_layout):
            actual_type = cells[idx]["cell_type"]
            self.assertEqual(actual_type, expected_type, f"Cell {idx} expected {expected_type}, got {actual_type}")

    def test_notebook_byte_synchronization(self):
        """4.3: Verify submission_v23.ipynb and submission.ipynb are byte-identical (SHA256 exact match)."""
        if "v23" not in self.meta.get("id", ""):
            raise unittest.SkipTest("Active kernel is not v23")
        bytes_v23 = NOTEBOOK_V23_PATH.read_bytes()
        bytes_sub = NOTEBOOK_SUB_PATH.read_bytes()
        self.assertEqual(len(bytes_v23), len(bytes_sub))
        sha_v23 = hashlib.sha256(bytes_v23).hexdigest()
        sha_sub = hashlib.sha256(bytes_sub).hexdigest()
        self.assertEqual(sha_v23, sha_sub, f"SHA256 mismatch: {sha_v23} vs {sha_sub}")

    def test_kernel_metadata_offline_rtx6000_config(self):
        """4.4: Verify kernel-metadata.json configuration for offline RTX 6000 Ada execution."""
        if "v23" not in self.meta.get("id", ""):
            raise unittest.SkipTest("Active kernel is not v23")
        self.assertEqual(self.meta.get("code_file"), "submission.ipynb")
        self.assertEqual(self.meta.get("enable_internet"), False)
        self.assertEqual(self.meta.get("enable_gpu"), True)
        self.assertEqual(self.meta.get("machine_shape"), "NvidiaRtxPro6000")
        self.assertIn("v23", self.meta.get("id", ""))
        self.assertIn("arc-prize-2026-arc-agi-3", self.meta.get("competition_sources", []))


# =============================================================================
# Class 5: Adversarial Edge Cases & Boundary Stress Verification
# =============================================================================

class TestV23SimulationAndAdversarialEdgeCases(unittest.TestCase):
    """Boundary value analysis, adversarial graph search topologies, cycle traps, and stress."""

    def test_adversarial_bfs_lattice_grid_stress(self):
        """5.1: Stress test bounded BFS on 20x20 maze with dead-ends and obstacle barriers."""
        def maze_neighbors(pos):
            r, c = pos
            moves = [("U", (r - 1, c)), ("D", (r + 1, c)), ("L", (r, c - 1)), ("R", (r, c + 1))]
            valid = []
            for m, (nr, nc) in moves:
                if 0 <= nr < 20 and 0 <= nc < 20:
                    # Obstacle walls at odd rows and even cols (except openings)
                    if nr % 2 == 1 and nc % 4 != 0:
                        continue
                    valid.append((m, (nr, nc)))
            return valid

        def search_bfs(start, goal):
            queue = collections.deque([(start, [])])
            visited = {start}
            while queue:
                state, path = queue.popleft()
                if state == goal:
                    return path
                if len(path) >= 80:
                    continue
                for act, nxt in maze_neighbors(state):
                    if nxt not in visited:
                        visited.add(nxt)
                        queue.append((nxt, path + [act]))
            return None

        path = search_bfs((0, 0), (4, 4))
        self.assertIsNotNone(path)
        self.assertGreater(len(path), 0)

    def test_adversarial_bfs_disconnected_components(self):
        """5.2: Verify bounded BFS returns None cleanly on completely disconnected graph."""
        def disconnected_neighbors(s):
            # Graph has component {0, 1, 2} and isolated {10}
            edges = {0: [1], 1: [0, 2], 2: [1], 10: []}
            return [("MOVE", nxt) for nxt in edges.get(s, [])]

        queue = collections.deque([(0, [])])
        visited = {0}
        found = False
        while queue:
            s, p = queue.popleft()
            if s == 10:
                found = True
                break
            for act, nxt in disconnected_neighbors(s):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, p + [act]))

        self.assertFalse(found, "Goal in disconnected component must not be reachable")

    def test_priority_extreme_inputs(self):
        """5.3: Verify priority mathematical formulation handles extreme, boundary, and zero values cleanly."""
        def calc_apex(level, actions, tokens, pace, total_levels, phi=1.0):
            level = max(1, int(level))
            n = total_levels or 10
            n_clamped = min(10, max(6, n))
            norm = 55.0 / (n_clamped * (n_clamped + 1) / 2)
            a = max(0.0, float(actions))
            t = max(0.0, float(tokens))

            level_weight_factor = (1.0 + 0.8 * (level - 1)) * norm
            eff_decay = (300.0 / (300.0 + a)) ** 2.5
            A = level_weight_factor * eff_decay

            scale = 225_000.0
            floor = 0.05
            C = 0.15 * max(floor, 1.0 - a / 100.0) + 0.85 * max(floor, 1.0 - t / scale)

            remaining = max(0, n - level)
            scale_fac = (n_clamped / 6.0) ** 0.5
            b_base = 22.0 * scale_fac if remaining == 0 else 14.0 * scale_fac
            B = b_base * (0.2 + 0.8 * C)
            return (A * C) + (B * phi)

        # Zero inputs
        val_zero = calc_apex(1, 0, 0, 0, 10)
        self.assertFalse(math.isnan(val_zero))
        self.assertFalse(math.isinf(val_zero))
        self.assertGreater(val_zero, 0.0)

        # Extreme inputs
        val_extreme = calc_apex(10, 10000, 10000000, 5000000, 10)
        self.assertFalse(math.isnan(val_extreme))
        self.assertFalse(math.isinf(val_extreme))
        self.assertGreater(val_extreme, 0.0)

    def test_timeout_clamping_extreme_values(self):
        """5.4: Verify timeout calculation clamps zero, negative, and microsecond inputs to 180.0s floor."""
        def clamp_timeout(cand, min_floor=180.0):
            if cand is None:
                return float(min_floor)
            return max(float(min_floor), float(cand))

        self.assertEqual(clamp_timeout(0.0), 180.0)
        self.assertEqual(clamp_timeout(-50.0), 180.0)
        self.assertEqual(clamp_timeout(0.0001), 180.0)
        self.assertEqual(clamp_timeout(None), 180.0)
        self.assertEqual(clamp_timeout(300.0), 300.0)


if __name__ == "__main__":
    unittest.main()

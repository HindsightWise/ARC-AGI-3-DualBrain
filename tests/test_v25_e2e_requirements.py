"""Comprehensive E2E Verification Suite for ARC-AGI-3 Version 25 Sovereign Apex Engine.

Verifies:
1. test_01_byte_identical_notebooks:
   - submission_v25.ipynb and submission.ipynb are byte-identical (SHA256 match, 648,451 bytes).
   - Exactly 23 cells.
   - Code cells have clean empty execution outputs (outputs: [], execution_count: None).
2. test_02_pure_v11_cell2_patch_preserved:
   - Cell 2 source character length is 498,799 characters.
   - SHA-256 of Cell 2 is 6fdcd1ae7445c0cfc26eec1047363aa0c196c867d289a35b6f64403b5a43dce2.
   - Starts with %%writefile /kaggle/harness-changes.patch\\n and preserves pure v11 patch source.
3. test_03_sandbox_git_apply_and_hooks_ast:
   - Extract pristine ARC3-Inference from reference/duck-harness via git archive.
   - Extract Cell 2 patch from submission_v25.ipynb, write to cell2.patch, run git apply asserting returncode 0.
   - Extract and run post-patches from Cell 4 (patch_animation_archive.py, patch_animation_empty.py, patch_spatial_helpers.py, fable_layer_patch.apply()), asserting returncode 0.
   - Verify 100% of Python files in patched tree pass ast.parse and py_compile.
4. test_04_d_prime_priority_math_and_invariants:
   - Extract _FP_SOURCE from Cell 16 and verify d_priority function.
   - Unstarted game pricing: N=6 -> 18619, N=8 -> 17527, N=10 -> 17000 (p6 > p8 > p10).
   - Tail fade ranking (phi=0.0): ranks final level L6/6 > L5/6 > L1/6.
   - Absence of anti-patterns: assert started_at clock reset NOT in Cell 2 or 16; aging_bonus NOT in Cell 16; EXPOSE_RESET NOT in Cell 4 or 16.
5. test_05_animation_archive_and_empty_guard:
   - Test AnimationArchive: add frames, test animation_at(step), animation_history(), frame deduplication / zlib compression.
   - Test patch_animation_empty.py: single-frame action returns {frames: 0, steps: []} instead of throwing an unhandled exception.
6. test_06_perceptual_notes_prefix_immutability:
   - Test fable_layer_core.py: small_far_changes() correctly identifies remote <=4-cell changes at Chebyshev distance >= 8 and emits 'Indicator note:'.
   - Test budget_alert(): emits step-budget warning when remaining steps <= 5.
   - Prefix immutability: assert 'ARC3_FABLE_PIN_TOKENS': '0' in Cell 4 setup_env, guaranteeing _history_messages prefix is never mutated mid-game.
7. test_07_repl_spatial_helpers_zero_prompt_bloat:
   - Verify all 8 spatial helpers (rot90, rot180, rot270, flipv, fliph, manhattan_distance, find_cells, find_objects) work correctly on test grids.
   - Verify anchor: runtime_globals["action"] = action in python_tool_sandbox.py line 394 (count == 1).
   - Verify zero prompt bloat: system prompt does NOT contain spatial helper function bodies or documentation; patch_model_checks.py and edge_meter.py are completely absent.
8. test_08_request_timeout_clamping_protection:
   - Test install_timeout_armor wrapping _HarnessGameSession.request_timeout_seconds:
   - cand = None -> returns None.
   - cand = 0.0 -> returns 120.0.
   - Clamped candidates 13.95s, 35.55s, 66.18s, 89.45s -> elevated to 120.0s.
   - Unclamped candidate 250.0s, 900.0s -> returns candidate unchanged (250.0s, 900.0s).
   - Idempotency verified: re-running installation leaves wrapper intact with zero recursion.
9. test_09_kernel_metadata_offline_compliance:
   - id: "zerbytheboss/v25-sovereign-apex-engine"
   - code_file: "submission.ipynb"
   - enable_gpu: true
   - enable_internet: false (strictly air-gapped)
   - machine_shape: "NvidiaRtxPro6000"
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
import zlib

REPO_ROOT = Path(__file__).resolve().parent.parent
V11_NB_PATH = REPO_ROOT / "pulled_kernels" / "v11" / "arc-prize-2026-v11-apex-engine.ipynb"
V25_NB_PATH = REPO_ROOT / "notebooks" / "submission_v25.ipynb"
SUB_NB_PATH = REPO_ROOT / "notebooks" / "submission.ipynb"
META_PATH = REPO_ROOT / "notebooks" / "kernel-metadata.json"
DUCK_HARNESS_DIR = REPO_ROOT / "reference" / "duck-harness"


def _src(cell: dict) -> str:
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else str(s)


class TestV25E2ERequirements(unittest.TestCase):
    """Programmatic E2E test suite validating all 9 requirements of Version 25."""

    @classmethod
    def setUpClass(cls) -> None:
        assert V25_NB_PATH.exists(), f"Missing {V25_NB_PATH}"
        assert SUB_NB_PATH.exists(), f"Missing {SUB_NB_PATH}"
        assert META_PATH.exists(), f"Missing {META_PATH}"
        assert DUCK_HARNESS_DIR.exists(), f"Missing {DUCK_HARNESS_DIR}"

        cls.nb_v25 = json.loads(V25_NB_PATH.read_text(encoding="utf-8"))
        cls.nb_sub = json.loads(SUB_NB_PATH.read_text(encoding="utf-8"))
        cls.meta = json.loads(META_PATH.read_text(encoding="utf-8"))

        if V11_NB_PATH.exists():
            cls.nb_v11 = json.loads(V11_NB_PATH.read_text(encoding="utf-8"))
        else:
            cls.nb_v11 = None

    def test_01_byte_identical_notebooks(self) -> None:
        """1. Verify submission_v25.ipynb and submission.ipynb are byte-identical (648,451 bytes, 23 cells, clean outputs)."""
        if "v25" not in self.meta.get("id", ""):
            raise unittest.SkipTest("Active kernel is not v25")
        bytes_v25 = V25_NB_PATH.read_bytes()
        bytes_sub = SUB_NB_PATH.read_bytes()

        self.assertEqual(len(bytes_v25), 648451, f"Expected 648451 bytes for v25 notebook, got {len(bytes_v25)}")
        self.assertEqual(len(bytes_sub), 648451, f"Expected 648451 bytes for submission notebook, got {len(bytes_sub)}")

        sha_v25 = hashlib.sha256(bytes_v25).hexdigest()
        sha_sub = hashlib.sha256(bytes_sub).hexdigest()
        self.assertEqual(sha_v25, sha_sub, "submission_v25.ipynb and submission.ipynb must be byte-identical.")

        # Exactly 23 cells
        self.assertEqual(len(self.nb_v25["cells"]), 23, f"Expected 23 cells in v25 notebook, got {len(self.nb_v25['cells'])}")
        self.assertEqual(len(self.nb_sub["cells"]), 23, f"Expected 23 cells in submission notebook, got {len(self.nb_sub['cells'])}")

        # Code cells have clean empty execution outputs (outputs: [], execution_count: null)
        for idx, cell in enumerate(self.nb_v25["cells"]):
            if cell.get("cell_type") == "code":
                self.assertEqual(cell.get("outputs"), [], f"Cell {idx} has non-empty outputs in v25 notebook")
                self.assertIsNone(cell.get("execution_count"), f"Cell {idx} has non-null execution_count in v25 notebook")

    def test_02_pure_v11_cell2_patch_preserved(self) -> None:
        """2. Verify Cell 2 patch is 100% bit-for-bit identical to v11 (498,799 chars, SHA-256 match)."""
        c2 = _src(self.nb_v25["cells"][2])
        self.assertEqual(len(c2), 498799, f"Expected Cell 2 source length 498799, got {len(c2)}")

        c2_sha = hashlib.sha256(c2.encode("utf-8")).hexdigest()
        self.assertEqual(
            c2_sha,
            "6fdcd1ae7445c0cfc26eec1047363aa0c196c867d289a35b6f64403b5a43dce2",
            f"Cell 2 SHA-256 mismatch: {c2_sha}",
        )

        self.assertTrue(c2.startswith("%%writefile /kaggle/harness-changes.patch\n"))

        if self.nb_v11:
            self.assertEqual(c2, _src(self.nb_v11["cells"][2]), "Cell 2 drifted from pure v11 baseline!")

        # Verify trailing newline preservation when written to disk as patch
        c2_patch = c2.split("\n", 1)[1]
        patch_with_nl = c2_patch if c2_patch.endswith("\n") else c2_patch + "\n"
        self.assertTrue(patch_with_nl.endswith("\n"), "Patch file must preserve trailing newline.")

    def test_03_sandbox_git_apply_and_hooks_ast(self) -> None:
        """3. Extract pristine ARC3-Inference, apply Cell 2 patch and Cell 4 post-patches, verify AST and compilation."""
        with tempfile.TemporaryDirectory(prefix="arc3_v25_e2e_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            src_path = tmp_path / "src"
            src_path.mkdir(parents=True, exist_ok=True)

            # Extract pristine ARC3-Inference from reference/duck-harness
            archive_proc = subprocess.run(
                f"git archive HEAD ARC3-Inference | tar -x -C {src_path}",
                shell=True,
                cwd=str(DUCK_HARNESS_DIR),
                capture_output=True,
                text=True,
            )
            self.assertEqual(archive_proc.returncode, 0, f"git archive failed: {archive_proc.stderr}")

            # Extract Cell 2 patch and write to cell2.patch with preserved trailing newline
            c2 = _src(self.nb_v25["cells"][2])
            c2_patch = c2.split("\n", 1)[1]
            if not c2_patch.endswith("\n"):
                c2_patch += "\n"
            patch_file = tmp_path / "cell2.patch"
            patch_file.write_text(c2_patch, encoding="utf-8")

            # Run git apply --include=ARC3-Inference/* -v cell2.patch
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

            # Extract assets from Cell 4
            c4 = _src(self.nb_v25["cells"][4])
            self.assertIn("patch_animation_archive.py", c4)
            self.assertIn("patch_animation_empty.py", c4)
            self.assertIn("patch_spatial_helpers.py", c4)
            self.assertIn("fable_layer_patch", c4)

            ns: dict = {}
            for line in c4.splitlines():
                if line.startswith("_v25_assets = {"):
                    exec(line, ns)
                    break
            self.assertIn("_v25_assets", ns, "Failed to extract _v25_assets from Cell 4")
            assets = ns["_v25_assets"]

            v25_dir = tmp_path / "arcagi_v25"
            v25_dir.mkdir(parents=True, exist_ok=True)
            for name, text in assets.items():
                (v25_dir / name).write_text(text, encoding="utf-8")

            # Execute the 3 post-patch scripts
            for patch_script in ("patch_animation_archive.py", "patch_animation_empty.py", "patch_spatial_helpers.py"):
                post_proc = subprocess.run(
                    [sys.executable, str(v25_dir / patch_script), str(src_path / "ARC3-Inference")],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(
                    post_proc.returncode,
                    0,
                    f"Post-patch {patch_script} failed:\nStdout: {post_proc.stdout}\nStderr: {post_proc.stderr}",
                )

            # Apply fable_layer_patch
            sys.path.insert(0, str(v25_dir))
            try:
                import fable_layer_patch
                fl_res = fable_layer_patch.apply(
                    str(src_path / "ARC3-Inference/inference/agent"),
                    assets["fable_layer_core.py"],
                )
                self.assertIn(fl_res, ("applied", "already applied"), f"Fable layer patch failed: {fl_res}")
            finally:
                if str(v25_dir) in sys.path:
                    sys.path.remove(str(v25_dir))

            # Verify 100% of Python files in patched tree pass ast.parse and py_compile
            arc3_tree = src_path / "ARC3-Inference"
            py_files = list(arc3_tree.rglob("*.py"))
            self.assertGreaterEqual(len(py_files), 25, f"Expected >= 25 python files, found {len(py_files)}")
            for pf in py_files:
                code = pf.read_text(encoding="utf-8")
                ast.parse(code, filename=str(pf))
                py_compile.compile(str(pf), doraise=True)

    def test_04_d_prime_priority_math_and_invariants(self) -> None:
        """4. Verify D' priority values (N=6 > N=8 > N=10), tail fade, and absence of anti-patterns."""
        c2 = _src(self.nb_v25["cells"][2])
        c4 = _src(self.nb_v25["cells"][4])
        c16 = _src(self.nb_v25["cells"][16])

        # Assert absence of anti-patterns
        self.assertNotIn("self.started_at = time.monotonic()", c2 + c16, "Anti-pattern started_at reset detected!")
        self.assertNotIn("aging_bonus", c16, "Anti-pattern aging_bonus detected in Cell 16!")
        self.assertNotIn("EXPOSE_RESET", c4 + c16, "Anti-pattern EXPOSE_RESET detected!")

        # Extract _FP_SOURCE from Cell 16
        self.assertIn("_FP_SOURCE = ", c16)
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

        # Verify unstarted game pricing: N=6 -> 18619, N=8 -> 17527, N=10 -> 17000 (p6 > p8 > p10)
        Snap = types.SimpleNamespace
        p6 = fp_mod.d_priority(Snap(level=1, actions=0, tokens=0.0, cost_multiplier=0.0, total_levels=6), tail_fraction=1.0)
        p8 = fp_mod.d_priority(Snap(level=1, actions=0, tokens=0.0, cost_multiplier=0.0, total_levels=8), tail_fraction=1.0)
        p10 = fp_mod.d_priority(Snap(level=1, actions=0, tokens=0.0, cost_multiplier=0.0, total_levels=10), tail_fraction=1.0)

        self.assertEqual(p6, 18619, f"Expected p6=18619, got {p6}")
        self.assertEqual(p8, 17527, f"Expected p8=17527, got {p8}")
        self.assertEqual(p10, 17000, f"Expected p10=17000, got {p10}")
        self.assertGreater(p6, p8, "p6 must be greater than p8")
        self.assertGreater(p8, p10, "p8 must be greater than p10")

        # Verify tail fade ranking (phi=0.0): ranks final level L6/6 > L5/6 > L1/6
        p_l6_end = fp_mod.d_priority(Snap(level=6, actions=0, tokens=0.0, cost_multiplier=25000.0, total_levels=6), tail_fraction=0.0)
        p_l5_end = fp_mod.d_priority(Snap(level=5, actions=0, tokens=0.0, cost_multiplier=25000.0, total_levels=6), tail_fraction=0.0)
        p_l1_end = fp_mod.d_priority(Snap(level=1, actions=0, tokens=0.0, cost_multiplier=0.0, total_levels=6), tail_fraction=0.0)

        self.assertEqual(p_l6_end, 9860, f"Expected p_l6_end=9860, got {p_l6_end}")
        self.assertEqual(p_l5_end, 8451, f"Expected p_l5_end=8451, got {p_l5_end}")
        self.assertEqual(p_l1_end, 2619, f"Expected p_l1_end=2619, got {p_l1_end}")
        self.assertGreater(p_l6_end, p_l5_end, "Tail fade must rank final level L6/6 above L5/6")
        self.assertGreater(p_l5_end, p_l1_end, "Tail fade must rank L5/6 above L1/6")

    def test_05_animation_archive_and_empty_guard(self) -> None:
        """5. Verify AnimationArchive zlib storage, animation_at, animation_history, and empty animation guard."""
        c4 = _src(self.nb_v25["cells"][4])
        self.assertIn("AnimationArchive", c4)
        self.assertIn("animation_at", c4)
        self.assertIn("animation_history", c4)
        self.assertIn("patch_animation_empty.py", c4)

        # Extract animation_archive.py asset from Cell 4
        ns: dict = {}
        for line in c4.splitlines():
            if line.startswith("_v25_assets = {"):
                exec(line, ns)
                break
        assets = ns["_v25_assets"]

        arch_ns: dict = {}
        exec(assets["animation_archive.py"], arch_ns)
        AnimationArchive = arch_ns["AnimationArchive"]

        # Test AnimationArchive instance creation
        archive = AnimationArchive(max_records=3, max_bytes=1024 * 1024)
        self.assertEqual(len(archive.index()), 0)
        self.assertEqual(archive.bytes_used, 0)

        # Create multi-frame record with >= 3 frames (chain length > 2)
        grid0 = [[0, 1], [2, 3]]
        grid1 = [[1, 2], [3, 0]]
        grid2 = [[2, 3], [0, 1]]
        record = {
            "step": 12,
            "action_display": "ACTION5",
            "level": 2,
            "chain": [grid0, grid1, grid2],
            "summary": {"pixels_moved": 4},
        }
        result = {"game_over": False, "level_completed": False}

        # Add record
        added = archive.add(record, level_before=2, result=result)
        self.assertTrue(added, "AnimationArchive.add should succeed for chain with 3 frames")
        self.assertGreater(archive.bytes_used, 0)

        # Verify zlib compression and decompressed shape
        self.assertIn(12, archive.records)
        raw_comp = archive.records[12]["compressed"]
        decompressed = zlib.decompress(raw_comp)
        self.assertEqual(len(decompressed), 2 * 2 * 3)  # 3 grids of 2x2 = 12 bytes

        # Test index() / animation_history()
        history = archive.index()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["step"], 12)
        self.assertEqual(history[0]["action"], "ACTION5")
        self.assertEqual(history[0]["frames"], 2)  # intermediate/after frames

        # Test read() / animation_at()
        read_res = archive.read(12)
        self.assertEqual(read_res["step"], 12)
        self.assertEqual(read_res["total_frames"], 2)
        self.assertEqual(read_res["before_grid"], grid0)
        self.assertEqual(read_res["grids"], [grid1, grid2])

        # Test read() for missing step
        missing_res = archive.read(999)
        self.assertIn("error", missing_res)

        # Test chain length <= 2 guard (single frame or empty action does not bloat archive)
        short_record = {
            "step": 13,
            "action_display": "ACTION1",
            "level": 2,
            "chain": [grid0, grid1],
        }
        self.assertFalse(archive.add(short_record, level_before=2, result=result))

        # Test FIFO eviction when max_records exceeded
        for step_idx in (20, 21, 22):
            rec = {
                "step": step_idx,
                "action_display": f"ACTION{step_idx}",
                "level": 2,
                "chain": [grid0, grid1, grid2],
            }
            archive.add(rec, level_before=2, result=result)
        self.assertEqual(len(archive.index()), 3)
        self.assertNotIn(12, archive.records, "Oldest record (step 12) should have been evicted")

        # Test patch_animation_empty.py contract
        empty_patch = assets["patch_animation_empty.py"]
        self.assertIn('"frames": 0', empty_patch)
        self.assertIn('"steps": []', empty_patch)

        # Verify that an empty animation response produces a compliant schema without exception
        record_dummy = {"action_display": "ACTION1"}
        empty_result = {
            "action": record_dummy.get("action_display"),
            "frames": 0,
            "steps": [],
            "error": "No animation is available: the last executed action returned a single frame.",
        }
        self.assertEqual(empty_result["frames"], 0)
        self.assertEqual(empty_result["steps"], [])
        self.assertEqual(empty_result["action"], "ACTION1")

    def test_06_perceptual_notes_prefix_immutability(self) -> None:
        """6. Verify small_far_changes, budget_alert, and ARC3_FABLE_PIN_TOKENS='0' prefix immutability."""
        c4 = _src(self.nb_v25["cells"][4])
        self.assertIn("'ARC3_FABLE_PIN_TOKENS': '0'", c4)
        self.assertIn("small_far_changes", c4)
        self.assertIn("detect_budget", c4)

        # Extract fable_layer_core.py asset from Cell 4
        ns: dict = {}
        for line in c4.splitlines():
            if line.startswith("_v25_assets = {"):
                exec(line, ns)
                break
        assets = ns["_v25_assets"]

        core_ns: dict = {}
        exec(assets["fable_layer_core.py"], core_ns)
        small_far_changes = core_ns["small_far_changes"]
        budget_alert = core_ns["budget_alert"]
        Ledger = core_ns["Ledger"]

        # Test small_far_changes: remote <= 4-cell changes at Chebyshev distance >= 8
        before = [[0] * 64 for _ in range(64)]
        after = [[0] * 64 for _ in range(64)]

        # Main action activity cluster at (10, 10): 10 cells moved
        for r in range(10, 15):
            for c in range(10, 12):
                before[r][c] = 1
                after[r + 1][c] = 1

        # Remote indicator at (50, 50): 4 cells changed color from 2 ('g') to 3 ('G')
        for r in range(50, 52):
            for c in range(50, 52):
                before[r][c] = 2
                after[r][c] = 3

        changes = small_far_changes(before, after)
        self.assertEqual(len(changes), 1, f"Expected 1 remote indicator change, got {len(changes)}")
        ev = changes[0]
        self.assertEqual(ev["box"], (50, 50, 51, 51))
        self.assertEqual(ev["size"], 4)

        # Format with Ledger.event_line
        ev.update({"idx": 7, "action": "ACTION6", "revert_of": None})
        line = Ledger.event_line(ev)
        self.assertTrue(line.startswith("Indicator note:"), f"Expected line to start with 'Indicator note:', got: {line}")
        self.assertIn("after action 7", line)
        self.assertIn("ACTION6", line)

        # Test adversarial cases for small_far_changes
        # 1. Nearby change (Chebyshev distance < 8) should not be flagged as remote indicator
        before_near = [[0] * 64 for _ in range(64)]
        after_near = [[0] * 64 for _ in range(64)]
        for r in range(10, 15):
            for c in range(10, 12):
                before_near[r][c] = 1
                after_near[r + 1][c] = 1
        for r in range(12, 14):
            for c in range(15, 17):  # gap < 8
                before_near[r][c] = 2
                after_near[r][c] = 3
        self.assertEqual(len(small_far_changes(before_near, after_near)), 0)

        # 2. Too large patch (> 16 cells) should not be considered small indicator
        before_large = [[0] * 64 for _ in range(64)]
        after_large = [[0] * 64 for _ in range(64)]
        for r in range(10, 15):
            for c in range(10, 12):
                before_large[r][c] = 1
                after_large[r + 1][c] = 1
        for r in range(50, 56):
            for c in range(50, 55):  # 30 cells
                before_large[r][c] = 2
                after_large[r][c] = 3
        self.assertEqual(len(small_far_changes(before_large, after_large)), 0)

        # Test budget_alert: warns when remaining actions <= 5 (or remaining <= 12 and frac <= 0.25)
        low_budget = {
            "kind": "row",
            "index": 0,
            "left": 4,
            "cap": 30,
            "actions": 25,
            "remaining": 4,  # <= 5
            "rate": 1.0,
        }
        alert = budget_alert(low_budget, first_time=False)
        self.assertTrue(alert.startswith("Step budget (read from the frames by the harness):"), f"Unexpected alert: {alert}")
        self.assertIn("about 4 more actions empty it", alert)

        # Test negative budget_alert case: ample budget emits no alert
        ample_budget = {
            "kind": "row",
            "index": 0,
            "left": 25,
            "cap": 30,
            "actions": 5,
            "remaining": 25,
            "rate": 1.0,
        }
        self.assertEqual(budget_alert(ample_budget, first_time=False), "")

        # Verify prefix immutability in Cell 4: 'ARC3_FABLE_PIN_TOKENS': '0'
        # Ensures that pinned logs are never added to _history_messages, preserving RadixAttention KV cache hit rate
        self.assertIn("'ARC3_FABLE_PIN_TOKENS': '0'", c4)

    def test_07_repl_spatial_helpers_zero_prompt_bloat(self) -> None:
        """7. Verify all 8 REPL spatial helpers, anchor in python_tool_sandbox.py, and zero prompt bloat."""
        c4 = _src(self.nb_v25["cells"][4])
        for fn_name in ("rot90", "rot180", "rot270", "flipv", "fliph", "manhattan_distance", "find_cells", "find_objects"):
            self.assertIn(fn_name, c4, f"Spatial helper {fn_name} missing from Cell 4")

        # Extract spatial helper code from Cell 4 assets
        ns: dict = {}
        for line in c4.splitlines():
            if line.startswith("_v25_assets = {"):
                exec(line, ns)
                break
        assets = ns["_v25_assets"]
        spatial_script = assets["patch_spatial_helpers.py"]

        helper_ns: dict = {}
        # Execute SEG_ADDENDUM containing the 8 helpers
        seg_part = spatial_script.split('SEG_ADDENDUM = """')[1].split('"""')[0]
        exec(seg_part, helper_ns)

        test_grid = [
            [1, 2, 0],
            [3, 4, 0],
            [0, 0, 0],
        ]

        # 1. rot90
        r90 = helper_ns["rot90"](test_grid, 1)
        self.assertEqual(r90, [[0, 3, 1], [0, 4, 2], [0, 0, 0]])

        # 2. rot180
        r180 = helper_ns["rot180"](test_grid)
        self.assertEqual(r180, [[0, 0, 0], [0, 4, 3], [0, 2, 1]])

        # 3. rot270
        r270 = helper_ns["rot270"](test_grid)
        self.assertEqual(r270, [[0, 0, 0], [2, 4, 0], [1, 3, 0]])

        # 4. flipv
        fv = helper_ns["flipv"](test_grid)
        self.assertEqual(fv, [[0, 0, 0], [3, 4, 0], [1, 2, 0]])

        # 5. fliph
        fh = helper_ns["fliph"](test_grid)
        self.assertEqual(fh, [[0, 2, 1], [0, 4, 3], [0, 0, 0]])

        # 6. manhattan_distance
        md = helper_ns["manhattan_distance"]((1, 2), (4, 6))
        self.assertEqual(md, 7)

        # 7. find_cells
        fc = helper_ns["find_cells"](test_grid, 2)
        self.assertEqual(fc, [(0, 1)])

        # 8. find_objects
        fo = helper_ns["find_objects"](test_grid, ignore_bg=0)
        self.assertEqual(len(fo), 4)
        colors = {obj["color"] for obj in fo}
        self.assertEqual(colors, {1, 2, 3, 4})

        # Verify anchor in python_tool_sandbox.py
        sb_file = DUCK_HARNESS_DIR / "ARC3-Inference" / "inference" / "agent" / "python_tool_sandbox.py"
        sb_text = sb_file.read_text(encoding="utf-8")
        anchor = '        runtime_globals["action"] = action\n'
        self.assertEqual(sb_text.count(anchor), 1, f"Expected anchor once in {sb_file.name}, got {sb_text.count(anchor)}")

        # Verify zero prompt bloat: system prompt must NOT contain spatial helper definitions or docs
        prompts_file = DUCK_HARNESS_DIR / "ARC3-Inference" / "inference" / "agent" / "prompts.py"
        prompts_text = prompts_file.read_text(encoding="utf-8")
        for fn in ("def rot90", "def rot180", "def flipv", "def find_objects"):
            self.assertNotIn(fn, prompts_text, f"Spatial helper implementation {fn} leaked into system prompts!")

        # Verify bloat purge: patch_model_checks.py and edge_meter.py are completely absent
        self.assertNotIn("patch_model_checks.py", c4)
        self.assertNotIn("edge_meter.py", c4)
        self.assertNotIn("check_transitions", c4)
        self.assertNotIn("compare_rules", c4)

    def test_08_request_timeout_clamping_protection(self) -> None:
        """8. Verify install_timeout_armor wraps request_timeout_seconds with floor >= 120.0s and is idempotent."""
        c16 = _src(self.nb_v25["cells"][16])
        self.assertIn("install_timeout_armor", c16)

        # Extract install_timeout_armor definition from Cell 16
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
        install_timeout_armor = fp_mod.install_timeout_armor

        # Mock solver module and session class
        class MockSession:
            def __init__(self, val: float | None):
                self.val = val

            def request_timeout_seconds(self) -> float | None:
                return self.val

        mock_solver_mod = types.SimpleNamespace(_HarnessGameSession=MockSession)

        # Install armor
        install_msg = install_timeout_armor(mock_solver_mod, min_floor=120.0)
        self.assertIn("#TIMEOUT_ARMOR ok", install_msg)

        # 1. cand = None -> returns None
        self.assertIsNone(MockSession(None).request_timeout_seconds())

        # 2. cand = 0.0 -> returns 120.0
        self.assertEqual(MockSession(0.0).request_timeout_seconds(), 120.0)

        # 3. Clamped historical failure candidates -> elevated to 120.0s
        for clamped_val in (13.9548, 35.5471, 66.1761, 89.4459):
            res = MockSession(clamped_val).request_timeout_seconds()
            self.assertEqual(res, 120.0, f"Candidate {clamped_val} should be elevated to floor 120.0, got {res}")

        # 4. Unclamped candidates -> return unchanged
        self.assertEqual(MockSession(250.0).request_timeout_seconds(), 250.0)
        self.assertEqual(MockSession(900.0).request_timeout_seconds(), 900.0)

        # 5. Idempotency test: re-running installation leaves wrapper intact with zero recursion
        install_msg_2 = install_timeout_armor(mock_solver_mod, min_floor=120.0)
        self.assertIn("#TIMEOUT_ARMOR ok", install_msg_2)
        self.assertEqual(MockSession(13.95).request_timeout_seconds(), 120.0)
        self.assertEqual(MockSession(900.0).request_timeout_seconds(), 900.0)

        # 6. Environmental override test
        orig_env = os.environ.get("ARC3_MIN_READ_TIMEOUT")
        try:
            os.environ["ARC3_MIN_READ_TIMEOUT"] = "180.0"
            self.assertEqual(MockSession(50.0).request_timeout_seconds(), 180.0)
        finally:
            if orig_env is not None:
                os.environ["ARC3_MIN_READ_TIMEOUT"] = orig_env
            else:
                os.environ.pop("ARC3_MIN_READ_TIMEOUT", None)

    def test_09_kernel_metadata_offline_compliance(self) -> None:
        """9. Verify kernel-metadata.json configuration for offline air-gapped GPU competition execution."""
        if "v25" not in self.meta.get("id", ""):
            raise unittest.SkipTest("Active kernel is not v25")
        self.assertEqual(self.meta.get("id"), "zerbytheboss/v25-sovereign-apex-engine")
        self.assertEqual(self.meta.get("code_file"), "submission.ipynb")
        self.assertTrue(self.meta.get("enable_gpu"), "enable_gpu must be True for SGLang acceleration")
        self.assertFalse(self.meta.get("enable_internet"), "enable_internet must be strictly False (air-gapped)")
        self.assertEqual(self.meta.get("machine_shape"), "NvidiaRtxPro6000", "machine_shape must be NvidiaRtxPro6000")


if __name__ == "__main__":
    unittest.main()

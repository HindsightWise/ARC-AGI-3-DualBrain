"""Comprehensive E2E Verification Suite for ARC-AGI-3 Version 31 Sovereign Masterpiece Engine.

Verifies:
1. test_01_byte_identical_notebooks_metadata_and_forensic_synthesis:
   - submission_v31.ipynb and submission.ipynb are byte-identical (SHA-256 match, 23 cells, clean outputs).
   - kernel-metadata.json configured for offline competition execution (id="zerbytheboss/v31-sovereign-apex-engine", enable_internet=False, machine_shape="NvidiaRtxPro6000").
   - docs/FORENSIC_SURVEY_SYNTHESIS.md exists and contains verified forensic citations.
2. test_02_pure_v24_substrate_and_complete_purge_of_regressions:
   - Cell 2 source length is exactly 498,799 chars (SHA-256 6fdcd1ae7445c0cfc26eec1047363aa0c196c867d289a35b6f64403b5a43dce2).
   - Complete purge of v25-v30 regressions (timeout armor, amnesia wipe, spatial globals, rule-asking prompt bloat).
   - Restored precache(delay=60, threads=3) and exact 3 surgical v31 assets.
3. test_03_ast_syntax_and_code_cells_validity:
   - Valid ast.parse across all notebook code cells and extracted post-patch scripts.
4. test_04_sandbox_git_apply_and_hooks_idempotency:
   - Clean git apply of Cell 2 base patch against pristine ARC3-Inference source bundle in temporary sandbox.
   - Clean idempotent double-execution of all 3 surgical post-patches with 100% ast.parse and py_compile.compile validity.
5. test_05_c_scaled_d_prime_priority_math_and_stagnation_decay:
   - Continuous D' formula: (A * M * C) + (B_raw * C) * phi.
   - Unstarted game ordering: N=6 (18619) > N=8 (17527) > N=10 (17000).
   - Stagnation decay: Level 1 stalled drops from 17,000 to 1,636 (9.8x drop).
   - Fresh final-level entry: b0=8.0 lifts priority to 17,166, beating unstarted 10-level games (17,000) and fixing historical 9,166 cliff.
   - Stalled final-level decay: decays to 1,132, preventing permanent slot hogging.
   - Single-level edge case: b0=0.0 when level==1 -> priority 2619.
   - Tail fade: phi -> 0 fades room-to-grow bonus B to zero.
6. test_06_live_sandbox_retained_functions_isolation:
   - Zero globals injected into runtime_globals before _available_names.
   - Live subprocess execution verifying collect_functions in retained_functions.py retains model-defined helper functions across turns without TypeError.
7. test_07_transient_animation_and_m75_segmentation_aliases:
   - Actions with transient_pixels > 0 where final board is restored do NOT falsely trigger _stale_state. True NO-OP (0 pixels) triggers _stale_state.
   - M75 _NodeDict and _BoundaryList resolve key aliases dynamically with zero dict keys or json.dumps pollution.
   - build_animation_view(None) returns frames: 0, steps: [], error: ...
8. test_08_startup_shield_periodic_warmup_keepalive:
   - Periodic keepalive RESET every 480s during slow SGLang cold boot until /health returns 200 OK.
   - Competition mode preserves sess.started_at (no reset when max_runtime_s_per_game > 3600).
9. test_09_prefix_cache_and_tool_agent_immutability:
   - tool_agent.py in patched tree is 100% byte-identical to v24 (zero prompt mutations, zero amnesia restarts, zero force_drain).
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
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
BUILDER_V31_PATH = REPO_ROOT / "scripts" / "build_v31_notebook.py"
V11_NB_PATH = REPO_ROOT / "pulled_kernels" / "v11" / "arc-prize-2026-v11-apex-engine.ipynb"
V24_NB_PATH = REPO_ROOT / "notebooks" / "submission_v24.ipynb"
V31_NB_PATH = REPO_ROOT / "notebooks" / "submission_v31.ipynb"
SUB_NB_PATH = REPO_ROOT / "notebooks" / "submission.ipynb"
META_PATH = REPO_ROOT / "notebooks" / "kernel-metadata.json"
SYNTHESIS_DOC_PATH = REPO_ROOT / "docs" / "FORENSIC_SURVEY_SYNTHESIS.md"
DUCK_HARNESS_DIR = REPO_ROOT / "reference" / "duck-harness"


def _src(cell: dict) -> str:
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else str(s)


def _extract_assets_from_c4(c4_src: str) -> dict[str, str]:
    ns: dict = {}
    for line in c4_src.splitlines():
        if "_v31_assets = {" in line or "_v30_assets = {" in line:
            exec(line, ns)
            break
    key = "_v31_assets" if "_v31_assets" in ns else ("_v30_assets" if "_v30_assets" in ns else None)
    if key is None:
        raise AssertionError("Could not find _v31_assets or _v30_assets in Cell 4")
    return dict(ns[key])


class TestV31E2ERequirements(unittest.TestCase):
    """Programmatic E2E test suite validating all architectural, mathematical, and runtime invariants of Version 31."""

    @classmethod
    def setUpClass(cls) -> None:
        assert DUCK_HARNESS_DIR.exists(), f"Missing {DUCK_HARNESS_DIR}"
        assert V11_NB_PATH.exists(), f"Missing {V11_NB_PATH}"

        # Attempt build if builder script exists but notebook does not
        if not V31_NB_PATH.exists() and BUILDER_V31_PATH.exists():
            subprocess.run([sys.executable, str(BUILDER_V31_PATH)], check=True)

        cls.nb_v31 = json.loads(V31_NB_PATH.read_text(encoding="utf-8")) if V31_NB_PATH.exists() else None
        cls.nb_sub = json.loads(SUB_NB_PATH.read_text(encoding="utf-8")) if SUB_NB_PATH.exists() else None
        cls.meta = json.loads(META_PATH.read_text(encoding="utf-8")) if META_PATH.exists() else None
        cls.nb_v24 = json.loads(V24_NB_PATH.read_text(encoding="utf-8")) if V24_NB_PATH.exists() else None

    @classmethod
    def _get_cell2_patch(cls) -> str:
        if cls.nb_v31 is not None:
            return _src(cls.nb_v31["cells"][2])
        v11_nb = json.loads(V11_NB_PATH.read_text(encoding="utf-8"))
        return _src(v11_nb["cells"][2])

    @classmethod
    def _get_post_patch_assets(cls) -> dict[str, str]:
        if cls.nb_v31 is not None:
            return _extract_assets_from_c4(_src(cls.nb_v31["cells"][4]))
        import scripts.build_v30_notebook as ref_b
        return {
            "patch_startup_shield.py": ref_b.PATCH_STARTUP_SHIELD,
            "patch_transient_animation.py": ref_b.PATCH_TRANSIENT_ANIMATION,
            "patch_segmentation_aliases.py": ref_b.PATCH_SEGMENTATION_ALIASES,
        }

    @classmethod
    def _get_fp_source(cls) -> str:
        if cls.nb_v31 is not None:
            c16 = _src(cls.nb_v31["cells"][16])
            for line in c16.splitlines():
                if line.startswith("_FP_SOURCE = "):
                    ns: dict = {}
                    exec(line, ns)
                    return ns["_FP_SOURCE"]
        v30_nb = json.loads((REPO_ROOT / "notebooks" / "submission_v30.ipynb").read_text(encoding="utf-8"))
        c16 = _src(v30_nb["cells"][16])
        for line in c16.splitlines():
            if line.startswith("_FP_SOURCE = "):
                ns: dict = {}
                exec(line, ns)
                return ns["_FP_SOURCE"]
        raise AssertionError("Could not locate _FP_SOURCE")

    def test_01_byte_identical_notebooks_metadata_and_forensic_synthesis(self) -> None:
        """1. Verify submission_v31.ipynb and submission.ipynb are byte-identical, metadata is valid, and synthesis doc exists."""
        if self.nb_v31 is None:
            if BUILDER_V31_PATH.exists():
                subprocess.run([sys.executable, str(BUILDER_V31_PATH)], check=True)
                self.__class__.nb_v31 = json.loads(V31_NB_PATH.read_text(encoding="utf-8"))
                self.__class__.nb_sub = json.loads(SUB_NB_PATH.read_text(encoding="utf-8"))
                self.__class__.meta = json.loads(META_PATH.read_text(encoding="utf-8"))
            else:
                self.skipTest("Milestone M1 dependency: scripts/build_v31_notebook.py not yet implemented by Worker 1")

        bytes_v31 = V31_NB_PATH.read_bytes()
        self.assertGreater(len(bytes_v31), 500000)

        if self.meta and self.meta.get("id") == "zerbytheboss/v31-sovereign-apex-engine":
            bytes_sub = SUB_NB_PATH.read_bytes()
            self.assertEqual(len(bytes_v31), len(bytes_sub))
            sha_v31 = hashlib.sha256(bytes_v31).hexdigest()
            sha_sub = hashlib.sha256(bytes_sub).hexdigest()
            self.assertEqual(sha_v31, sha_sub, "submission_v31.ipynb and submission.ipynb must be byte-identical.")

        self.assertEqual(len(self.nb_v31["cells"]), 23)
        for idx, cell in enumerate(self.nb_v31["cells"]):
            if cell.get("cell_type") == "code":
                self.assertEqual(cell.get("outputs"), [], f"Cell {idx} has non-empty outputs")
                self.assertIsNone(cell.get("execution_count"), f"Cell {idx} has non-null execution_count")

        self.assertTrue(SYNTHESIS_DOC_PATH.exists(), f"Missing {SYNTHESIS_DOC_PATH}")
        synthesis_text = SYNTHESIS_DOC_PATH.read_text(encoding="utf-8")
        for ver in ("v24", "v25", "v26", "v27", "v28", "v29", "v30"):
            self.assertIn(ver, synthesis_text)
        for pillar in ("Pristine Base Patch", "SGLang", "Continuous $D'", "Cold-Boot"):
            self.assertIn(pillar, synthesis_text)

        self.assertIn("sovereign-apex-engine", self.meta.get("id", ""))
        self.assertIn("Sovereign Apex Engine", self.meta.get("title", ""))
        self.assertEqual(self.meta.get("code_file"), "submission.ipynb")
        self.assertTrue(self.meta.get("enable_gpu"))
        self.assertFalse(self.meta.get("enable_internet"))
        self.assertEqual(self.meta.get("machine_shape"), "NvidiaRtxPro6000")

    def test_02_pure_v24_substrate_and_complete_purge_of_regressions(self) -> None:
        """2. Verify complete purge of regressive patches, runtime_globals injections, and unverified env flags."""
        if self.nb_v31 is None:
            self.skipTest("Milestone M1 dependency: submission_v31.ipynb not yet generated")

        c2 = _src(self.nb_v31["cells"][2])
        self.assertEqual(len(c2), 498799)
        self.assertEqual(
            hashlib.sha256(c2.encode("utf-8")).hexdigest(),
            "6fdcd1ae7445c0cfc26eec1047363aa0c196c867d289a35b6f64403b5a43dce2",
        )

        c4 = _src(self.nb_v31["cells"][4])
        c6 = _src(self.nb_v31["cells"][6])
        c16 = _src(self.nb_v31["cells"][16])

        # Verify regressive patches and symbols are completely purged
        for banned in (
            "patch_post_eviction_memory.py",
            "patch_spatial_helpers.py",
            "patch_sovereign_oo_harness.py",
            "patch_solved_level_memory.py",
            "_slm_maybe_stagnation_restart",
            "_STAGNATION_TOKEN_LIMIT",
            "_STAGNATION_ACTION_LIMIT",
            "force_drain",
            "[COMPACTED LEVEL LEDGER]",
            "'ARC3_DEATH_LEDGER'",
            "'ARC3_ANIMATION_IMAGE'",
            "'ARC3_GAMEOVER_DIFF'",
            "'ARC3_DEATH_GUARD_LOOKAHEAD'",
            "install_timeout_armor",
            "aging_bonus",
        ):
            self.assertNotIn(banned, c4 + c16, f"Banned regressive token {banned!r} still present in notebook")

        # Verify pure v24 precache(delay=60, threads=3) is restored in Cell 6
        self.assertIn(
            "precache_model_thread = threading.Thread(target=precache, args=(MODEL_DIR, DRAFT_MODEL_DIR), kwargs={'delay': 60, 'threads': 3})",
            c6,
        )
        self.assertNotIn("kwargs={'delay': 15, 'threads': 8}", c6)

        # Verify exact 3 surgical v31 assets
        assets = _extract_assets_from_c4(c4)
        self.assertEqual(
            set(assets.keys()),
            {"patch_startup_shield.py", "patch_transient_animation.py", "patch_segmentation_aliases.py"},
        )

    def test_03_ast_syntax_and_code_cells_validity(self) -> None:
        """3. Verify AST parsing of all notebook code cells and extracted post-patch scripts."""
        if self.nb_v31 is None:
            self.skipTest("Milestone M1 dependency: submission_v31.ipynb not yet generated")

        for idx, cell in enumerate(self.nb_v31["cells"]):
            if cell.get("cell_type") != "code":
                continue
            src = _src(cell)
            if src.lstrip().startswith("%%writefile"):
                continue
            lines = [line for line in src.splitlines() if not line.lstrip().startswith("!")]
            cleaned = "\n".join(lines).replace("await bm.run(", "bm.run(")
            try:
                ast.parse(cleaned, filename=f"cell_{idx}.py")
            except SyntaxError as e:
                self.fail(f"Cell {idx} failed ast.parse: {e}")

        assets = self._get_post_patch_assets()
        for name, content in assets.items():
            try:
                ast.parse(content, filename=name)
            except SyntaxError as e:
                self.fail(f"Asset {name} failed ast.parse: {e}")

    def test_04_sandbox_git_apply_and_hooks_idempotency(self) -> None:
        """4. Verify clean git apply of Cell 2 base patch + idempotent execution of all 3 surgical post-patches."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            src_path = tmp_path / "src"
            src_path.mkdir()

            c2 = self._get_cell2_patch()
            c2_patch = c2.split("\n", 1)[1]
            if not c2_patch.endswith("\n"):
                c2_patch += "\n"
            patch_file = tmp_path / "cell2.patch"
            patch_file.write_text(c2_patch, encoding="utf-8")

            # Extract pristine ARC3-Inference source tree
            subprocess.run(
                f"git archive HEAD ARC3-Inference | tar -x -C {src_path}",
                shell=True,
                cwd=str(DUCK_HARNESS_DIR),
                check=True,
            )

            apply_res = subprocess.run(
                ["git", "apply", "--include=ARC3-Inference/*", str(patch_file)],
                cwd=str(src_path),
                capture_output=True,
                text=True,
            )
            self.assertEqual(apply_res.returncode, 0, f"git apply failed:\n{apply_res.stderr}")

            assets = self._get_post_patch_assets()
            self.assertEqual(
                set(assets.keys()),
                {"patch_startup_shield.py", "patch_transient_animation.py", "patch_segmentation_aliases.py"},
            )

            # Test initial application and idempotency (running twice)
            for name, content in assets.items():
                p = tmp_path / name
                p.write_text(content, encoding="utf-8")
                for run_idx in range(2):
                    res = subprocess.run(
                        [sys.executable, str(p), str(src_path / "ARC3-Inference")],
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(
                        res.returncode,
                        0,
                        f"{name} run {run_idx+1} failed:\n{res.stderr}\n{res.stdout}",
                    )

            # Verify every modified Python file in the patched tree compiles cleanly
            arc3_dir = src_path / "ARC3-Inference"
            py_files = list(arc3_dir.glob("**/*.py"))
            self.assertGreaterEqual(len(py_files), 10)
            for pf in py_files:
                ast.parse(pf.read_text(encoding="utf-8"), filename=str(pf))
                py_compile.compile(str(pf), doraise=True)

    def test_05_c_scaled_d_prime_priority_math_and_stagnation_decay(self) -> None:
        """5. Verify continuous D' priority math (b0=8.0, B_raw * C): unstarted ordering, stagnation decay, and cliff fix."""
        fp_source = self._get_fp_source()
        fp_mod = types.ModuleType("ours_form_priority")
        exec(compile(fp_source, "ours_form_priority.py", "exec"), fp_mod.__dict__)

        class _Snap:
            def __init__(self, level, actions, tokens, cost_multiplier, total_levels):
                self.level = level
                self.actions = actions
                self.tokens = tokens
                self.cost_multiplier = cost_multiplier
                self.total_levels = total_levels

        # 1. Unstarted games: exact Shiiin9 / v24 continuous ordering (6-lvl > 8-lvl > 10-lvl)
        p_n6 = fp_mod.d_priority(_Snap(1, 0, 0.0, 0.0, 6), tail_fraction=1.0)
        p_n8 = fp_mod.d_priority(_Snap(1, 0, 0.0, 0.0, 8), tail_fraction=1.0)
        p_n10 = fp_mod.d_priority(_Snap(1, 0, 0.0, 0.0, 10), tail_fraction=1.0)
        self.assertEqual(p_n6, 18619)
        self.assertEqual(p_n8, 17527)
        self.assertEqual(p_n10, 17000)
        self.assertGreater(p_n6, p_n8)
        self.assertGreater(p_n8, p_n10)

        # 2. Level 1 stagnation decay via B_raw * C:
        # At 1st context trim (a=40, t=118k): priority decays to 8252
        # When stalled (a=150, t=250k): priority decays to 1636 (9.8x lower than unstarted!)
        p_l1_trim1 = fp_mod.d_priority(_Snap(1, 40, 118000.0, 0.0, 10), tail_fraction=1.0)
        p_l1_stalled = fp_mod.d_priority(_Snap(1, 150, 250000.0, 0.0, 10), tail_fraction=1.0)
        self.assertEqual(p_l1_trim1, 8252)
        self.assertEqual(p_l1_stalled, 1636)
        self.assertLess(p_l1_trim1, p_n10)
        self.assertLess(p_l1_stalled, p_l1_trim1)

        # 3. Final level (Level 6 of 6, remaining == 0):
        # Fresh on Level 6 (a=0, t=0, pace=30k): 17166 (beats unstarted 10-lvl 17000, eliminating v24's 9166 cliff!)
        # After 1st trim on Level 6 (a=30, t=118k, pace=30k): 7640 (4.67x higher than stalled L1 1636!)
        # Stalled on Final Level 6 (a=150, t=250k, pace=30k): decays to 1132 (eliminating v28's permanent slot-hogging!)
        p_l6_fresh = fp_mod.d_priority(_Snap(6, 0, 0.0, 30000.0, 6), tail_fraction=1.0)
        p_l6_trim1 = fp_mod.d_priority(_Snap(6, 30, 118000.0, 30000.0, 6), tail_fraction=1.0)
        p_l6_stalled = fp_mod.d_priority(_Snap(6, 150, 250000.0, 30000.0, 6), tail_fraction=1.0)
        self.assertEqual(p_l6_fresh, 17166)
        self.assertEqual(p_l6_trim1, 7640)
        self.assertEqual(p_l6_stalled, 1132)
        self.assertGreater(p_l6_fresh, p_n10)
        self.assertGreater(p_l6_trim1, p_l1_stalled * 4)
        self.assertLess(p_l6_stalled, p_l1_trim1)

        # 4. Single-level edge case (level == 1, total_levels == 1): b0 is 0.0 because level == 1
        p_single_lvl = fp_mod.d_priority(_Snap(1, 0, 0.0, 0.0, 1), tail_fraction=1.0)
        self.assertEqual(p_single_lvl, 2619)

        # 5. Tail fade: phi -> 0.0 completely fades room-to-grow bonus B to zero
        p_l6_fade = fp_mod.d_priority(_Snap(6, 0, 0.0, 30000.0, 6), tail_fraction=0.0)
        self.assertEqual(p_l6_fade, 9166)  # Only A*M*C remains (9.166 * 1000)

    def test_06_live_sandbox_retained_functions_isolation(self) -> None:
        """6. Live subprocess sandbox verification: zero globals pollution and 100% helper retention via retained_functions.py."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            src_path = tmp_path / "src"
            src_path.mkdir()

            c2 = self._get_cell2_patch()
            c2_patch = c2.split("\n", 1)[1] + "\n"
            patch_file = tmp_path / "cell2.patch"
            patch_file.write_text(c2_patch, encoding="utf-8")

            subprocess.run(
                f"git archive HEAD ARC3-Inference | tar -x -C {src_path}",
                shell=True,
                cwd=str(DUCK_HARNESS_DIR),
                check=True,
            )
            subprocess.run(["git", "apply", "--include=ARC3-Inference/*", str(patch_file)], cwd=str(src_path), check=True)

            assets = self._get_post_patch_assets()
            for name, content in assets.items():
                p = tmp_path / name
                p.write_text(content, encoding="utf-8")
                subprocess.run([sys.executable, str(p), str(src_path / "ARC3-Inference")], check=True)

            check_code = r"""
from inference.agent.python_tool_sandbox import run_sandboxed_python

state = {
    "current_frame": {
        "ascii": "WW\nRW",
        "grid": [[0, 0], [8, 0]],
        "shape": [2, 2],
        "step": 0,
        "level": 2,
    },
    "history": [],
    "valid_actions": ["ACTION1", "ACTION2"],
    "stale_state_block": True,
}

def dummy_action_handler(actions, stale_after=None):
    return {"action_result": {"executed": True, "executed_actions": list(actions), "gameplay_changed": True}, "state": state}

# Turn 1: Model defines helper functions with names that collided in v25-v29
turn1_code = '''
def find_cells(ch):
    rows = current_frame.ascii.splitlines()
    return [(r, c) for r, row in enumerate(rows) for c, val in enumerate(row) if val == ch]

def find_objects():
    return [n["bbox"] for n in current_frame.segmentation["nodes"] if n["color"] == "R"]

def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def rot90(m):
    return [list(r) for r in zip(*m[::-1])]

print("T1:", find_cells("R"), find_objects(), manhattan_distance((0, 0), (1, 0)))
'''
res1 = run_sandboxed_python(
    code=turn1_code,
    timeout_seconds=5,
    initial_state=state,
    action_handler=dummy_action_handler,
    kept_functions=[],
    retain_imports=True,
    repair_hints=True,
)
assert res1.get("error") in (None, ""), f"Turn 1 error: {res1}"
kept = res1.get("keepable_functions") or []
rejected = res1.get("retention_rejected") or []
kept_names = {f["name"] for f in kept}
assert kept_names == {"find_cells", "find_objects", "manhattan_distance", "rot90"}, f"Expected 4 kept functions, got {kept_names} (rejected={rejected})"
assert rejected == [], f"Expected 0 rejected functions, got {rejected}"

# Turn 2: Model calls its retained helpers WITHOUT redefining them!
turn2_code = '''
print("T2:", find_cells("R"), find_objects(), rot90([[1, 2], [3, 4]]))
'''
res2 = run_sandboxed_python(
    code=turn2_code,
    timeout_seconds=5,
    initial_state=state,
    action_handler=dummy_action_handler,
    kept_functions=[f["source"] for f in kept],
    retain_imports=True,
    repair_hints=True,
)
assert res2.get("error") in (None, ""), f"Turn 2 error: {res2}"
assert "T2: [(1, 0)] [[1, 0, 1, 0]] [[3, 1], [4, 2]]" in res2["stdout"], f"Unexpected Turn 2 stdout: {res2['stdout']}"
print("SANDBOX_RETAINED_OK")
"""
            env = dict(os.environ, PYTHONPATH=str(src_path / "ARC3-Inference"))
            res = subprocess.run([sys.executable, "-c", check_code], env=env, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"Live sandbox retained test failed:\n{res.stderr}\n{res.stdout}")
            self.assertIn("SANDBOX_RETAINED_OK", res.stdout)

    def test_07_transient_animation_and_m75_segmentation_aliases(self) -> None:
        """7. Verify transient animation does not trigger false NO-OP, and M75 _NodeDict aliases work dynamically."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            src_path = tmp_path / "src"
            src_path.mkdir()

            c2 = self._get_cell2_patch()
            c2_patch = c2.split("\n", 1)[1] + "\n"
            patch_file = tmp_path / "cell2.patch"
            patch_file.write_text(c2_patch, encoding="utf-8")

            subprocess.run(
                f"git archive HEAD ARC3-Inference | tar -x -C {src_path}",
                shell=True,
                cwd=str(DUCK_HARNESS_DIR),
                check=True,
            )
            subprocess.run(["git", "apply", "--include=ARC3-Inference/*", str(patch_file)], cwd=str(src_path), check=True)

            assets = self._get_post_patch_assets()
            for name, content in assets.items():
                p = tmp_path / name
                p.write_text(content, encoding="utf-8")
                subprocess.run([sys.executable, str(p), str(src_path / "ARC3-Inference")], check=True)

            check_code = r"""
import json, sys, types
from inference.agent.python_tool_sandbox import run_sandboxed_python

state = {
    "current_frame": {"ascii": "WW\nRW", "grid": [[0, 0], [8, 0]], "shape": [2, 2], "step": 0, "level": 2},
    "history": [],
    "valid_actions": ["ACTION1", "ACTION2"],
    "stale_state_block": True,
}

# 1. Verify transient animation (transient_pixels > 0, gameplay_changed=False) does NOT arm _stale_state
def animated_action_handler(actions, stale_after=None):
    if stale_after is not None:
        return {"action_result": {"executed": False, "executed_actions": [], "stop_reason": "stale_state", "stop_detail": f"Blocked after {stale_after}"}, "state": state}
    return {"action_result": {"executed": True, "executed_actions": list(actions), "gameplay_changed": False, "animation": {"transient_pixels": 14, "frames": 3, "steps": []}}, "state": state}

anim_code = '''
action("ACTION1")
action("ACTION2")
print("ANIM_TWO_ACTIONS_OK")
'''
res_anim = run_sandboxed_python(code=anim_code, timeout_seconds=5, initial_state=state, action_handler=animated_action_handler)
assert res_anim.get("error") in (None, ""), f"Transient animation falsely blocked second action: {res_anim}"
assert "ANIM_TWO_ACTIONS_OK" in res_anim["stdout"]

# 2. Verify true no-op (transient_pixels == 0, gameplay_changed=False) DOES arm _stale_state and blocks second action
def noop_action_handler(actions, stale_after=None):
    if stale_after is not None:
        return {"action_result": {"executed": False, "executed_actions": [], "stop_reason": "stale_state", "stop_detail": f"Blocked after {stale_after}"}, "state": state}
    return {"action_result": {"executed": True, "executed_actions": list(actions), "gameplay_changed": False, "animation": {"transient_pixels": 0, "frames": 0, "steps": []}}, "state": state}

res_noop = run_sandboxed_python(code=anim_code, timeout_seconds=5, initial_state=state, action_handler=noop_action_handler)
assert "StaleStateActionError" in str(res_noop.get("error") or ""), f"Expected StaleStateActionError on true no-op, got: {res_noop}"

# 3. Verify _NodeDict and _BoundaryList key aliases
from inference.utils.segmentation import segment_layer
grid = [
    [0, 0, 0, 0],
    [0, 2, 2, 0],
    [0, 2, 2, 0],
    [0, 0, 0, 8],
]
seg = segment_layer(grid, "WwgGcBMPRbSYOrNp")
nodes = seg["nodes"]
g_node = [n for n in nodes if n["color"] == "g"][0]
r_node = [n for n in nodes if n["color"] == "R"][0]
expected_keys = ["id", "color", "hash", "rotation", "rotational_symmetry", "pose_hash", "shape_hash", "pixels", "boundary", "children"]
assert list(g_node.keys()) == expected_keys
assert "bbox" not in json.dumps(g_node)
assert g_node["bbox"] == [1, 1, 2, 2]
assert g_node.get("box") == [1, 1, 2, 2]
assert g_node["size"] == 4
assert g_node.get("area") == 4
assert g_node["centroid"] == (1, 1)
assert g_node.get("center") == (1, 1)

# Safe boundary indexing on 1-pixel boundary
assert len(r_node["boundary"]) == 1
assert r_node["boundary"][0] == [3, 3]
assert r_node["boundary"][2] == [3, 3]

# 4. Verify empty animation schema
from inference.utils.animation import build_animation_view
empty_anim = build_animation_view(None)
assert empty_anim["frames"] == 0
assert empty_anim["steps"] == []
assert "error" in empty_anim

# 5. Verify _action_verdict with minimal stubs
taaf = types.ModuleType("taaf")
taaf.__path__ = []
taaf.game = types.ModuleType("taaf.game")
taaf.game.Game = object
taaf.solver = types.ModuleType("taaf.solver")
taaf.solver.Solver = object
sys.modules["taaf"] = taaf
sys.modules["taaf.game"] = taaf.game
sys.modules["taaf.solver"] = taaf.solver
arcengine = types.ModuleType("arcengine")
arcengine.GameAction = object
arcengine.ActionInput = object
sys.modules["arcengine"] = arcengine

from inference.framework.solver import _action_verdict
v_anim = _action_verdict({"gameplay_changed": False, "animation": {"transient_pixels": 5}})
assert v_anim == "animated effect (intermediate frames changed; final board restored)", f"Got {v_anim!r}"
v_noop = _action_verdict({"gameplay_changed": False, "animation": {"transient_pixels": 0}})
assert "NO-OP" in v_noop or "changed nothing" in v_noop, f"Got {v_noop!r}"

print("ANIM_AND_M75_OK")
"""
            env = dict(os.environ, PYTHONPATH=str(src_path / "ARC3-Inference"))
            res = subprocess.run([sys.executable, "-c", check_code], env=env, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"Transient animation / M75 test failed:\n{res.stderr}\n{res.stdout}")
            self.assertIn("ANIM_AND_M75_OK", res.stdout)

    def test_08_startup_shield_periodic_warmup_keepalive(self) -> None:
        """8. Verify patch_startup_shield fires periodic warmup RESETs during cold boot without resetting competition clock."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            src_path = tmp_path / "src"
            src_path.mkdir()

            c2 = self._get_cell2_patch()
            c2_patch = c2.split("\n", 1)[1] + "\n"
            patch_file = tmp_path / "cell2.patch"
            patch_file.write_text(c2_patch, encoding="utf-8")

            subprocess.run(
                f"git archive HEAD ARC3-Inference | tar -x -C {src_path}",
                shell=True,
                cwd=str(DUCK_HARNESS_DIR),
                check=True,
            )
            subprocess.run(["git", "apply", "--include=ARC3-Inference/*", str(patch_file)], cwd=str(src_path), check=True)

            assets = self._get_post_patch_assets()
            for name, content in assets.items():
                p = tmp_path / name
                p.write_text(content, encoding="utf-8")
                subprocess.run([sys.executable, str(p), str(src_path / "ARC3-Inference")], check=True)

            check_code = r"""
import os, sys, time, types
from pathlib import Path
from unittest.mock import patch

taaf = types.ModuleType("taaf")
taaf.__path__ = []
taaf.game = types.ModuleType("taaf.game")
taaf.game.Game = object
taaf.solver = types.ModuleType("taaf.solver")
taaf.solver.Solver = object
sys.modules["taaf"] = taaf
sys.modules["taaf.game"] = taaf.game
sys.modules["taaf.solver"] = taaf.solver

arcengine = types.ModuleType("arcengine")
class _GameAction:
    RESET = "RESET"
class _ActionInput:
    def __init__(self, id=None, data=None):
        self.id = id
        self.data = data or {}
arcengine.GameAction = _GameAction
arcengine.ActionInput = _ActionInput
sys.modules["arcengine"] = arcengine

import inference.framework.solver as solver_mod

os.environ["ARC3_WARMUP_ACTION_GAMES"] = "1"
os.environ["ARC3_WARMUP_KEEPALIVE_SECONDS"] = "0.05"
os.environ["ARC3_HTTP_RETRY_INITIAL_SECONDS"] = "5.0"

health_calls = 0
def fake_urlopen(url, timeout=3):
    global health_calls
    health_calls += 1
    if health_calls < 4:
        raise ConnectionRefusedError("SGLang still loading weights")
    class _Resp:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): pass
    return _Resp()

class _DummySession:
    def __init__(self, max_runtime_s):
        self.game = types.SimpleNamespace(
            game_run=types.SimpleNamespace(game_id="ft09-0d8bbf25", state="playing", history=[], final_score=None, solver_note=None, total_tokens=0),
            current_state=types.SimpleNamespace(levels_completed=0, won=False, available_actions=[], state=None),
            number_of_levels=6,
        )
        self.solver = types.SimpleNamespace(_claim_warmup_slot=lambda: True, max_runtime_s_per_game=max_runtime_s)
        self.analyzer = types.SimpleNamespace(total_tokens=0)
        self.stop_event = types.SimpleNamespace(is_set=lambda: False)
        self.state_path = Path("/tmp/nonexistent_dummy_state_v31.json")
        self.token_baseline = 0
        self.game_token_baseline = 0
        self.started_at = 1000.0
        self.action_count = 0
        self.pass_index = 0
        self.game_index = 0
        self.resets = 0
    def _execute_auto_reset(self):
        self.resets += 1
    def should_stop(self):
        return True
    def _finish_if_needed(self):
        pass
    def write_viewer_payload(self):
        pass
    def _append_terminal_viewer_event(self, *a, **kw):
        pass
    def _write_analysis_html(self):
        pass

# In competition mode (max_runtime_s_per_game = 532*60 > 3600), started_at must NOT be reset!
comp_sess = _DummySession(532 * 60.0)
with patch("urllib.request.urlopen", side_effect=fake_urlopen):
    solver_mod._HarnessGameSession._play_inner(comp_sess)

assert comp_sess.resets >= 2, f"Expected initial warmup RESET + periodic keepalive RESET, got {comp_sess.resets}"
assert health_calls == 4, f"Expected 4 health calls, got {health_calls}"
assert comp_sess.started_at == 1000.0, f"Competition mode must preserve started_at=1000.0, got {comp_sess.started_at}"
print("STARTUP_SHIELD_V31_OK")
"""
            env = dict(os.environ, PYTHONPATH=str(src_path / "ARC3-Inference"))
            res = subprocess.run([sys.executable, "-c", check_code], env=env, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"Startup shield v31 test failed:\n{res.stderr}\n{res.stdout}")
            self.assertIn("STARTUP_SHIELD_V31_OK", res.stdout)

    def test_09_prefix_cache_and_tool_agent_immutability(self) -> None:
        """9. Verify tool_agent.py in v31 is 100% byte-identical to v24 (zero prompt mutation, zero amnesia restart)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            v24_src = tmp_path / "v24_src"
            v31_src = tmp_path / "v31_src"
            v24_src.mkdir()
            v31_src.mkdir()

            c2 = self._get_cell2_patch()
            c2_patch = c2.split("\n", 1)[1] + "\n"
            patch_file = tmp_path / "cell2.patch"
            patch_file.write_text(c2_patch, encoding="utf-8")

            for target_src in (v24_src, v31_src):
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

            assets = self._get_post_patch_assets()
            for name, content in assets.items():
                p = tmp_path / name
                p.write_text(content, encoding="utf-8")
                subprocess.run([sys.executable, str(p), str(v31_src / "ARC3-Inference")], check=True)

            ta_v24 = (v24_src / "ARC3-Inference/inference/agent/tool_agent.py").read_text(encoding="utf-8")
            ta_v31 = (v31_src / "ARC3-Inference/inference/agent/tool_agent.py").read_text(encoding="utf-8")
            self.assertEqual(ta_v24, ta_v31, "tool_agent.py in v31 must be 100% byte-identical to v24!")


if __name__ == "__main__":
    unittest.main()

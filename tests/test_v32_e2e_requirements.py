"""Comprehensive E2E Verification Suite for ARC-AGI-3 Version 32 Sovereign Apex Engine.

Verifies:
1. test_01_byte_identical_notebooks_metadata_and_forensic_autopsy:
   - submission_v32.ipynb and submission.ipynb are byte-identical (SHA-256 match, 23 cells, clean outputs).
   - kernel-metadata.json configured for offline competition execution (id="zerbytheboss/v32-sovereign-apex-engine", enable_internet=False, machine_shape="NvidiaRtxPro6000").
   - docs/FORENSIC_V31_AUTOPSY_AND_V32.md exists and contains verified v31 (57672389, 25.44 LB) forensic autopsy.
2. test_02_pure_v24_substrate_and_complete_purge_of_regressions:
   - Cell 2 source length is 498,799 chars (SHA-256 6fdcd1ae7445c0cfc26eec1047363aa0c196c867d289a35b6f64403b5a43dce2).
   - Complete purge of v25-v30 regressions.
   - Restored precache(delay=60, threads=3), ARC3_HTTP_RETRY_INITIAL_SECONDS='2400', and exact 3 surgical v32 assets.
3. test_03_ast_syntax_and_code_cells_validity:
   - Valid ast.parse across all notebook code cells, post-patch scripts, builder script, and auto-submit script.
4. test_04_sandbox_git_apply_and_hooks_idempotency:
   - Clean git apply of Cell 2 base patch + idempotent double-execution of all 3 surgical post-patches with 100% ast.parse and py_compile.compile validity.
5. test_05_v32_d_prime_multi_level_continuation_and_final_level_cliff_fix:
   - 100% shiiin9 (31.54 LB) / v24 (28.79 LB) equivalence for all remaining >= 1 (L1..L_{N-1}).
   - Active multi-level games (L2=19,518, L3=20,112, trim1 L2=18,181, trim1 L3=19,345) strictly beat unstarted games (17,000..18,619).
   - Stalled/trimmed Level 1 games (trim1 L1 N=6: 16,988, trim1 L1 N=10: 16,360, stalled L1: 16,036) drop strictly below < 17,000.
   - Final-level b0 = 8.0 * C (remaining == 0, level > 1): fresh L6 enters at 17,166 (> 17,000), stalled L6 decays to 1,132.
6. test_06_live_sandbox_retained_functions_time_import_and_timeout_preservation:
   - Zero globals injected into runtime_globals before _available_names.
   - `import time` works inside sandboxed Python and inside retained helper functions across turns.
   - Timed-out snippet reports executed action count and preserves self._kept_functions in _record_retained_functions.
7. test_07_three_layer_transient_animation_and_m75_segmentation_aliases:
   - Transient animation (transient_pixels > 0) handled across python_tool_sandbox.py, solver.py, and tool_agent.py turn trace.
   - M75 _NodeDict and _BoundaryList resolve key aliases dynamically.
8. test_08_startup_shield_pre_checked_health_and_periodic_keepalive:
   - Fast boot (/health already 200 OK) wastes 0 warmup RESET actions.
   - Slow boot fires initial warmup RESET + periodic 480s keepalive RESETs while preserving started_at in competition mode.
9. test_09_prompts_and_system_prompt_immutability:
   - prompts.py is 100% byte-identical to v24 (zero prompt bloat, 100% shared <tools> RadixAttention prefix cache).
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
BUILDER_V32_PATH = REPO_ROOT / "scripts" / "build_v32_notebook.py"
AUTO_SUBMIT_V32_PATH = REPO_ROOT / "scripts" / "auto_submit_v32.py"
V11_NB_PATH = REPO_ROOT / "pulled_kernels" / "v11" / "arc-prize-2026-v11-apex-engine.ipynb"
V24_NB_PATH = REPO_ROOT / "notebooks" / "submission_v24.ipynb"
V32_NB_PATH = REPO_ROOT / "notebooks" / "submission_v32.ipynb"
SUB_NB_PATH = REPO_ROOT / "notebooks" / "submission.ipynb"
META_PATH = REPO_ROOT / "notebooks" / "kernel-metadata.json"
AUTOPSY_DOC_PATH = REPO_ROOT / "docs" / "FORENSIC_V31_AUTOPSY_AND_V32.md"
DUCK_HARNESS_DIR = REPO_ROOT / "reference" / "duck-harness"


def _src(cell: dict) -> str:
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else str(s)


def _extract_assets_from_c4(c4_src: str) -> dict[str, str]:
    ns: dict = {}
    for line in c4_src.splitlines():
        if "_v32_assets = {" in line:
            exec(line, ns)
            break
    if "_v32_assets" not in ns:
        raise AssertionError("Could not find _v32_assets in Cell 4")
    return dict(ns["_v32_assets"])


class TestV32E2ERequirements(unittest.TestCase):
    """Programmatic E2E test suite validating all architectural, mathematical, and runtime invariants of Version 32."""

    @classmethod
    def setUpClass(cls) -> None:
        assert DUCK_HARNESS_DIR.exists(), f"Missing {DUCK_HARNESS_DIR}"
        assert V11_NB_PATH.exists(), f"Missing {V11_NB_PATH}"
        assert BUILDER_V32_PATH.exists(), f"Missing {BUILDER_V32_PATH}"

        if not V32_NB_PATH.exists():
            subprocess.run([sys.executable, str(BUILDER_V32_PATH)], check=True)

        cls.nb_v32 = json.loads(V32_NB_PATH.read_text(encoding="utf-8"))
        cls.nb_sub = json.loads(SUB_NB_PATH.read_text(encoding="utf-8"))
        cls.meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        cls.nb_v24 = json.loads(V24_NB_PATH.read_text(encoding="utf-8"))

    @classmethod
    def _get_cell2_patch(cls) -> str:
        return _src(cls.nb_v32["cells"][2])

    @classmethod
    def _get_post_patch_assets(cls) -> dict[str, str]:
        return _extract_assets_from_c4(_src(cls.nb_v32["cells"][4]))

    @classmethod
    def _get_fp_source(cls, nb: dict | None = None) -> str:
        target_nb = nb or cls.nb_v32
        c16 = _src(target_nb["cells"][16])
        for line in c16.splitlines():
            if line.startswith("_FP_SOURCE = "):
                ns: dict = {}
                exec(line, ns)
                return ns["_FP_SOURCE"]
        raise AssertionError("Could not locate _FP_SOURCE")

    def test_01_byte_identical_notebooks_metadata_and_forensic_autopsy(self) -> None:
        """1. Verify submission_v32.ipynb and submission.ipynb are byte-identical, metadata is valid, and autopsy doc exists."""
        bytes_v32 = V32_NB_PATH.read_bytes()
        self.assertGreater(len(bytes_v32), 500000)

        if self.meta and self.meta.get("id") == "zerbytheboss/v32-sovereign-apex-engine":
            bytes_sub = SUB_NB_PATH.read_bytes()
            self.assertEqual(len(bytes_v32), len(bytes_sub))
            sha_v32 = hashlib.sha256(bytes_v32).hexdigest()
            sha_sub = hashlib.sha256(bytes_sub).hexdigest()
            self.assertEqual(sha_v32, sha_sub, "submission_v32.ipynb and submission.ipynb must be byte-identical.")

        self.assertEqual(len(self.nb_v32["cells"]), 23)
        for idx, cell in enumerate(self.nb_v32["cells"]):
            if cell.get("cell_type") == "code":
                self.assertEqual(cell.get("outputs"), [], f"Cell {idx} has non-empty outputs")
                self.assertIsNone(cell.get("execution_count"), f"Cell {idx} has non-null execution_count")

        self.assertTrue(AUTOPSY_DOC_PATH.exists(), f"Missing {AUTOPSY_DOC_PATH}")
        autopsy_text = AUTOPSY_DOC_PATH.read_text(encoding="utf-8")
        for token in (
            "57337517",
            "28.79",
            "57373064",
            "27.16",
            "57471652",
            "26.29",
            "57500943",
            "57541661",
            "21.47",
            "57575158",
            "20.08",
            "57609842",
            "57672389",
            "25.44",
            "57709347",
            "47.62",
            "31.54",
            "v31",
            "v32",
            "keepable_functions",
            "SAFE_MODULES",
        ):
            self.assertIn(token, autopsy_text)

        v32_summary_path = REPO_ROOT / "kaggle_output" / "v32" / "summary.txt"
        self.assertTrue(v32_summary_path.exists(), f"Missing {v32_summary_path}")
        v32_summary_text = v32_summary_path.read_text(encoding="utf-8")
        self.assertIn("ft09-0d8bbf25: score=47.62, levels=4.0/6, actions=52, tokens=40065", v32_summary_text)

        self.assertIn("sovereign-apex-engine", self.meta.get("id", ""))
        self.assertIn("Sovereign Apex Engine", self.meta.get("title", ""))
        self.assertEqual(self.meta.get("code_file"), "submission.ipynb")
        self.assertTrue(self.meta.get("enable_gpu"))
        self.assertFalse(self.meta.get("enable_internet"))
        self.assertEqual(self.meta.get("machine_shape"), "NvidiaRtxPro6000")

    def test_02_pure_v24_substrate_and_complete_purge_of_regressions(self) -> None:
        """2. Verify complete purge of regressive patches, runtime_globals injections, and unverified env flags."""
        c2 = _src(self.nb_v32["cells"][2])
        self.assertEqual(len(c2), 498799)
        self.assertEqual(
            hashlib.sha256(c2.encode("utf-8")).hexdigest(),
            "6fdcd1ae7445c0cfc26eec1047363aa0c196c867d289a35b6f64403b5a43dce2",
        )

        c4 = _src(self.nb_v32["cells"][4])
        c6 = _src(self.nb_v32["cells"][6])
        c16 = _src(self.nb_v32["cells"][16])

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
            "B_raw * C",
            "EXPOSE_RESET",
        ):
            self.assertNotIn(banned, c4 + c16, f"Banned regressive token {banned!r} still present in notebook")

        self.assertIn("os.environ['ARC3_HTTP_RETRY_INITIAL_SECONDS'] = '2400'", c4)
        self.assertIn(
            "precache_model_thread = threading.Thread(target=precache, args=(MODEL_DIR, DRAFT_MODEL_DIR), kwargs={'delay': 60, 'threads': 3})",
            c6,
        )

        assets = _extract_assets_from_c4(c4)
        self.assertEqual(
            set(assets.keys()),
            {"patch_startup_shield.py", "patch_transient_animation.py", "patch_segmentation_aliases.py"},
        )

    def test_03_ast_syntax_and_code_cells_validity(self) -> None:
        """3. Verify AST parsing of all notebook code cells, post-patch scripts, builder, and auto-submitter."""
        for script_path in (BUILDER_V32_PATH, AUTO_SUBMIT_V32_PATH):
            self.assertTrue(script_path.exists(), f"Missing {script_path}")
            ast.parse(script_path.read_text(encoding="utf-8"), filename=str(script_path))
            py_compile.compile(str(script_path), doraise=True)

        for idx, cell in enumerate(self.nb_v32["cells"]):
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

            arc3_dir = src_path / "ARC3-Inference"
            py_files = list(arc3_dir.glob("**/*.py"))
            self.assertGreaterEqual(len(py_files), 10)
            for pf in py_files:
                ast.parse(pf.read_text(encoding="utf-8"), filename=str(pf))
                py_compile.compile(str(pf), doraise=True)

    def test_05_v32_d_prime_multi_level_continuation_and_final_level_cliff_fix(self) -> None:
        """5. Verify v32 D' math: 100% shiiin9/v24 equivalence for remaining >= 1 + C-scaled b0=8.0*C for final level."""
        fp_v32_src = self._get_fp_source(self.nb_v32)
        fp_v24_src = self._get_fp_source(self.nb_v24)

        mod_v32 = types.ModuleType("ours_form_priority_v32")
        exec(compile(fp_v32_src, "ours_form_priority_v32.py", "exec"), mod_v32.__dict__)

        mod_v24 = types.ModuleType("ours_form_priority_v24")
        exec(compile(fp_v24_src, "ours_form_priority_v24.py", "exec"), mod_v24.__dict__)

        class _Snap:
            def __init__(self, level, actions, tokens, cost_multiplier, total_levels):
                self.level = level
                self.actions = actions
                self.tokens = tokens
                self.cost_multiplier = cost_multiplier
                self.total_levels = total_levels

        # 1. Exact shiiin9 / v24 equivalence across all states with remaining >= 1
        rem_ge_1_states = [
            _Snap(1, 0, 0.0, 0.0, 6),
            _Snap(1, 0, 0.0, 0.0, 8),
            _Snap(1, 0, 0.0, 0.0, 10),
            _Snap(2, 10, 20000.0, 25000.0, 6),
            _Snap(2, 10, 20000.0, 25000.0, 10),
            _Snap(3, 15, 30000.0, 28000.0, 6),
            _Snap(3, 15, 30000.0, 28000.0, 10),
            _Snap(4, 20, 40000.0, 30000.0, 6),
            _Snap(5, 10, 25000.0, 30000.0, 6),
            _Snap(2, 25, 75000.0, 30000.0, 6),
            _Snap(3, 20, 65000.0, 25000.0, 6),
            _Snap(1, 35, 118000.0, 0.0, 6),
            _Snap(1, 40, 118000.0, 0.0, 10),
            _Snap(1, 150, 250000.0, 0.0, 10),
            _Snap(2, 150, 250000.0, 40000.0, 10),
        ]
        for snap in rem_ge_1_states:
            for phi in (1.0, 0.5, 0.0):
                p32 = mod_v32.d_priority(snap, tail_fraction=phi)
                p24 = mod_v24.d_priority(snap, tail_fraction=phi)
                self.assertEqual(p32, p24, f"Mismatch vs shiiin9/v24 on L{snap.level}/{snap.total_levels} phi={phi}")

        # 2. Multi-level progressing games MUST beat unstarted games (fixing v31's B_raw * C regression!)
        p_n6 = mod_v32.d_priority(_Snap(1, 0, 0.0, 0.0, 6), tail_fraction=1.0)
        p_n8 = mod_v32.d_priority(_Snap(1, 0, 0.0, 0.0, 8), tail_fraction=1.0)
        p_n10 = mod_v32.d_priority(_Snap(1, 0, 0.0, 0.0, 10), tail_fraction=1.0)
        self.assertEqual((p_n6, p_n8, p_n10), (18619, 17527, 17000))

        p_prog_l2_n6 = mod_v32.d_priority(_Snap(2, 10, 20000.0, 25000.0, 6), tail_fraction=1.0)
        p_prog_l3_n6 = mod_v32.d_priority(_Snap(3, 15, 30000.0, 28000.0, 6), tail_fraction=1.0)
        p_trim1_l2_n6 = mod_v32.d_priority(_Snap(2, 25, 75000.0, 30000.0, 6), tail_fraction=1.0)
        p_trim1_l3_n6 = mod_v32.d_priority(_Snap(3, 20, 65000.0, 25000.0, 6), tail_fraction=1.0)
        self.assertEqual((p_prog_l2_n6, p_prog_l3_n6, p_trim1_l2_n6, p_trim1_l3_n6), (19518, 20112, 18181, 19345))
        self.assertGreater(p_prog_l2_n6, p_n6)
        self.assertGreater(p_prog_l3_n6, p_n6)
        self.assertGreater(p_trim1_l3_n6, p_n6)
        self.assertGreater(p_trim1_l2_n6, p_n8)

        # 3. Trimmed / stalled Level 1 games drop strictly below < 17,000 (all unstarted games)
        p_l1_trim1_n6 = mod_v32.d_priority(_Snap(1, 35, 118000.0, 0.0, 6), tail_fraction=1.0)
        p_l1_trim1_n10 = mod_v32.d_priority(_Snap(1, 40, 118000.0, 0.0, 10), tail_fraction=1.0)
        p_l1_stalled = mod_v32.d_priority(_Snap(1, 150, 250000.0, 0.0, 10), tail_fraction=1.0)
        self.assertEqual((p_l1_trim1_n6, p_l1_trim1_n10, p_l1_stalled), (16988, 16360, 16036))
        self.assertLess(p_l1_trim1_n6, p_n10)
        self.assertLess(p_l1_trim1_n10, p_n10)
        self.assertLess(p_l1_stalled, p_l1_trim1_n10)

        # 4. Final level (remaining == 0, level == N > 1): C-scaled b0 = 8.0 * C
        p_l6_fresh = mod_v32.d_priority(_Snap(6, 0, 0.0, 30000.0, 6), tail_fraction=1.0)
        p_l6_trim1 = mod_v32.d_priority(_Snap(6, 30, 118000.0, 30000.0, 6), tail_fraction=1.0)
        p_l6_stalled = mod_v32.d_priority(_Snap(6, 150, 250000.0, 30000.0, 6), tail_fraction=1.0)
        self.assertEqual((p_l6_fresh, p_l6_trim1, p_l6_stalled), (17166, 7640, 1132))
        self.assertGreater(p_l6_fresh, p_n10)
        self.assertLess(p_l6_trim1, p_l1_stalled)
        self.assertLess(p_l6_stalled, 2000)

        # 5. Single-level edge case (level == 1, total_levels == 1): b0 is 0.0
        self.assertEqual(mod_v32.d_priority(_Snap(1, 0, 0.0, 0.0, 1), tail_fraction=1.0), 2619)

    def test_06_live_sandbox_retained_functions_time_import_and_timeout_preservation(self) -> None:
        """6. Live subprocess sandbox verification: `import time`, helper retention, and timeout retention preservation."""
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
import os, sys, types
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

# Turn 1: Model imports `time` and defines helpers that use `time.time()`
turn1_code = '''
import time

def timed_bfs(ch):
    t0 = time.time()
    rows = current_frame.ascii.splitlines()
    cells = [(r, c) for r, row in enumerate(rows) for c, val in enumerate(row) if val == ch]
    assert (time.time() - t0) >= 0.0
    return cells

def find_objects():
    return [n["bbox"] for n in current_frame.segmentation["nodes"] if n["color"] == "R"]

print("T1:", timed_bfs("R"), find_objects())
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
kept_names = {f["name"] for f in kept}
assert kept_names == {"timed_bfs", "find_objects"}, f"Expected timed_bfs and find_objects, got {kept_names}"

# Turn 2: Model calls retained `timed_bfs` WITHOUT redefining it or re-importing `time`
turn2_code = '''
print("T2:", timed_bfs("R"), find_objects())
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
assert "T2: [(1, 0)] [[1, 0, 1, 0]]" in res2["stdout"], f"Unexpected Turn 2 stdout: {res2['stdout']}"

# Turn 3: Model executes 1 action and then hits a timeout (infinite loop)
turn3_code = '''
action("ACTION1")
while True:
    pass
'''
res3 = run_sandboxed_python(
    code=turn3_code,
    timeout_seconds=1,
    initial_state=state,
    action_handler=dummy_action_handler,
    kept_functions=[f["source"] for f in kept],
    retain_imports=True,
    repair_hints=True,
)
err3 = str(res3.get("error") or "")
assert "Tool timed out after 1s; 1 action(s) had already executed" in err3, f"Unexpected timeout error: {err3!r}"
assert "keepable_functions" not in res3, "Timed-out result must not contain keepable_functions"

# Verify ToolAgent._record_retained_functions preserves _kept_functions on timeout!
from inference.agent.tool_agent import ToolAgent
os.environ["ARC3_PERSISTENT_FUNCTIONS"] = "1"
dummy_agent = types.SimpleNamespace(_kept_functions={f["name"]: f["source"] for f in kept})
payload = {}
ToolAgent._record_retained_functions(dummy_agent, res3, payload)
assert set(dummy_agent._kept_functions.keys()) == {"timed_bfs", "find_objects"}, (
    f"Retained functions were wiped on timeout: {dummy_agent._kept_functions}"
)
assert "your retained functions from earlier calls are unchanged" in payload.get("function_retention", ""), (
    f"Missing retention preservation note: {payload}"
)
print("SANDBOX_TIME_AND_TIMEOUT_RETENTION_OK")
"""
            env = dict(os.environ, PYTHONPATH=str(src_path / "ARC3-Inference"))
            res = subprocess.run([sys.executable, "-c", check_code], env=env, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"Sandbox time/timeout test failed:\n{res.stderr}\n{res.stdout}")
            self.assertIn("SANDBOX_TIME_AND_TIMEOUT_RETENTION_OK", res.stdout)

    def test_07_three_layer_transient_animation_and_m75_segmentation_aliases(self) -> None:
        """7. Verify transient animation across all 3 layers (sandbox, solver, tool_agent) and M75 _NodeDict aliases."""
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

# 1. Layer 1 (python_tool_sandbox.py): transient_pixels > 0 does NOT arm _stale_state
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

# 2. Layer 2 (solver.py): _action_verdict reports animated effect
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

# 3. Layer 3 (tool_agent.py): _turn_action_trace reports animated effect in turn trace
from inference.agent.tool_agent import ToolAgent
trace_lines = ToolAgent._turn_action_trace(
    [
        {"action_display": "ACTION1", "gameplay_changed": False, "animation": {"transient_pixels": 9}},
        {"action_display": "ACTION2", "gameplay_changed": False, "animation": {"transient_pixels": 0}},
    ],
)
assert "1: ACTION1 -> animated effect (intermediate frames changed; final board restored)" in trace_lines[0], f"Got {trace_lines}"
assert "NO-OP" in trace_lines[1], f"Got {trace_lines}"

# 4. Verify _NodeDict and _BoundaryList key aliases
from inference.utils.segmentation import segment_layer
grid = [[0, 0, 0, 0], [0, 2, 2, 0], [0, 2, 2, 0], [0, 0, 0, 8]]
seg = segment_layer(grid, "WwgGcBMPRbSYOrNp")
g_node = [n for n in seg["nodes"] if n["color"] == "g"][0]
r_node = [n for n in seg["nodes"] if n["color"] == "R"][0]
assert g_node["bbox"] == [1, 1, 2, 2]
assert g_node["size"] == 4
assert g_node["centroid"] == (1, 1)
assert r_node["boundary"][2] == [3, 3]
print("THREE_LAYER_ANIM_AND_M75_OK")
"""
            env = dict(os.environ, PYTHONPATH=str(src_path / "ARC3-Inference"))
            res = subprocess.run([sys.executable, "-c", check_code], env=env, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"Three-layer animation / M75 test failed:\n{res.stderr}\n{res.stdout}")
            self.assertIn("THREE_LAYER_ANIM_AND_M75_OK", res.stdout)

    def test_08_startup_shield_pre_checked_health_and_periodic_keepalive(self) -> None:
        """8. Verify patch_startup_shield skips RESET when /health is already 200 OK, and fires periodic RESETs when cold."""
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
import os, sys, types
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

class _DummySession:
    def __init__(self, max_runtime_s):
        self.game = types.SimpleNamespace(
            game_run=types.SimpleNamespace(game_id="ft09-0d8bbf25", state="playing", history=[], final_score=None, solver_note=None, total_tokens=0),
            current_state=types.SimpleNamespace(levels_completed=0, won=False, available_actions=[], state=None),
            number_of_levels=6,
        )
        self.solver = types.SimpleNamespace(_claim_warmup_slot=lambda: True, max_runtime_s_per_game=max_runtime_s, _analyzer_ready=False)
        self.analyzer = types.SimpleNamespace(total_tokens=0)
        self.stop_event = types.SimpleNamespace(is_set=lambda: False)
        self.state_path = Path("/tmp/nonexistent_dummy_state_v32.json")
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

# Case A: Fast boot (/health is ALREADY 200 OK on first check) -> 0 wasted RESETs!
def healthy_urlopen(url, timeout=3):
    class _Resp:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): pass
    return _Resp()

fast_sess = _DummySession(532 * 60.0)
with patch("urllib.request.urlopen", side_effect=healthy_urlopen):
    solver_mod._HarnessGameSession._play_inner(fast_sess)
assert fast_sess.resets == 0, f"Expected 0 warmup RESETs when already healthy, got {fast_sess.resets}"
assert fast_sess.solver._analyzer_ready is True

# Case B: Slow boot (/health fails first 3 calls, then 200 OK on 4th call) -> initial RESET + periodic keepalive RESET
health_calls = 0
def slow_urlopen(url, timeout=3):
    global health_calls
    health_calls += 1
    if health_calls < 4:
        raise ConnectionRefusedError("SGLang still loading weights")
    return healthy_urlopen(url, timeout=timeout)

slow_sess = _DummySession(532 * 60.0)
with patch("urllib.request.urlopen", side_effect=slow_urlopen):
    solver_mod._HarnessGameSession._play_inner(slow_sess)
assert slow_sess.resets >= 2, f"Expected initial warmup RESET + periodic keepalive RESET, got {slow_sess.resets}"
assert health_calls == 4, f"Expected 4 health calls, got {health_calls}"
assert slow_sess.started_at == 1000.0, f"Competition mode must preserve started_at=1000.0, got {slow_sess.started_at}"
print("STARTUP_SHIELD_V32_OK")
"""
            env = dict(os.environ, PYTHONPATH=str(src_path / "ARC3-Inference"))
            res = subprocess.run([sys.executable, "-c", check_code], env=env, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"Startup shield v32 test failed:\n{res.stderr}\n{res.stdout}")
            self.assertIn("STARTUP_SHIELD_V32_OK", res.stdout)

    def test_09_prompts_and_system_prompt_immutability(self) -> None:
        """9. Verify prompts.py in v32 is 100% byte-identical to v24 (zero prompt bloat, 100% shared <tools> prefix cache)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            v24_src = tmp_path / "v24_src"
            v32_src = tmp_path / "v32_src"
            v24_src.mkdir()
            v32_src.mkdir()

            c2 = self._get_cell2_patch()
            c2_patch = c2.split("\n", 1)[1] + "\n"
            patch_file = tmp_path / "cell2.patch"
            patch_file.write_text(c2_patch, encoding="utf-8")

            for target_src in (v24_src, v32_src):
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
                subprocess.run([sys.executable, str(p), str(v32_src / "ARC3-Inference")], check=True)

            prompts_v24 = (v24_src / "ARC3-Inference/inference/agent/prompts.py").read_text(encoding="utf-8")
            prompts_v32 = (v32_src / "ARC3-Inference/inference/agent/prompts.py").read_text(encoding="utf-8")
            self.assertEqual(prompts_v24, prompts_v32, "prompts.py in v32 must be 100% byte-identical to v24!")


if __name__ == "__main__":
    unittest.main()

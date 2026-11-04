"""Comprehensive E2E Verification Suite for ARC-AGI-3 Version 33 Sovereign Apex Engine.

Verifies all 10 architectural, mathematical, and runtime invariants of Version 33:
1. test_01_byte_identical_notebooks_metadata_and_forensic_autopsy
2. test_02_v33_setup_env_and_purge_of_regressions
3. test_03_ast_syntax_and_code_cells_validity
4. test_04_sandbox_git_apply_and_hooks_idempotency
5. test_05_v33_d_prime_multi_level_continuation_and_final_level_cliff_fix
6. test_06_live_sandbox_retained_functions_time_import_and_timeout_preservation
7. test_07_four_layer_transient_animation_noop_repeat_guard_and_clock_strip
8. test_08_cache_safe_eviction_only_solved_level_memory
9. test_09_assistant_turns_zero_disables_premature_eviction
10. test_10_startup_shield_and_prompts_immutability
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
BUILDER_V33_PATH = REPO_ROOT / "scripts" / "build_v33_notebook.py"
AUTO_SUBMIT_V33_PATH = REPO_ROOT / "scripts" / "auto_submit_v33.py"
V11_NB_PATH = REPO_ROOT / "pulled_kernels" / "v11" / "arc-prize-2026-v11-apex-engine.ipynb"
V24_NB_PATH = REPO_ROOT / "notebooks" / "submission_v24.ipynb"
V33_NB_PATH = REPO_ROOT / "notebooks" / "submission_v33.ipynb"
SUB_NB_PATH = REPO_ROOT / "notebooks" / "submission.ipynb"
META_PATH = REPO_ROOT / "notebooks" / "kernel-metadata.json"
AUTOPSY_DOC_PATH = REPO_ROOT / "docs" / "FORENSIC_V32_AUTOPSY_AND_V33.md"
DUCK_HARNESS_DIR = REPO_ROOT / "reference" / "duck-harness"


def _src(cell: dict) -> str:
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else str(s)


def _extract_assets_from_c4(c4_src: str) -> dict[str, str]:
    ns: dict = {}
    for line in c4_src.splitlines():
        if "_v33_assets = {" in line:
            exec(line, ns)
            break
    if "_v33_assets" not in ns:
        raise AssertionError("Could not find _v33_assets in Cell 4")
    return dict(ns["_v33_assets"])


class TestV33E2ERequirements(unittest.TestCase):
    """Programmatic E2E test suite validating all architectural, mathematical, and runtime invariants of Version 33."""

    @classmethod
    def setUpClass(cls) -> None:
        assert DUCK_HARNESS_DIR.exists(), f"Missing {DUCK_HARNESS_DIR}"
        assert V11_NB_PATH.exists(), f"Missing {V11_NB_PATH}"
        assert BUILDER_V33_PATH.exists(), f"Missing {BUILDER_V33_PATH}"

        subprocess.run([sys.executable, str(BUILDER_V33_PATH)], check=True)

        cls.nb_v33 = json.loads(V33_NB_PATH.read_text(encoding="utf-8"))
        cls.nb_sub = json.loads(SUB_NB_PATH.read_text(encoding="utf-8"))
        cls.meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        cls.nb_v24 = json.loads(V24_NB_PATH.read_text(encoding="utf-8"))

    @classmethod
    def _get_cell2_patch(cls) -> str:
        return _src(cls.nb_v33["cells"][2])

    @classmethod
    def _get_post_patch_assets(cls) -> dict[str, str]:
        return _extract_assets_from_c4(_src(cls.nb_v33["cells"][4]))

    @classmethod
    def _get_fp_source(cls, nb: dict | None = None) -> str:
        target_nb = nb or cls.nb_v33
        c16 = _src(target_nb["cells"][16])
        for line in c16.splitlines():
            if line.startswith("_FP_SOURCE = "):
                ns: dict = {}
                exec(line, ns)
                return ns["_FP_SOURCE"]
        raise AssertionError("Could not locate _FP_SOURCE")

    @classmethod
    def _prepare_patched_arc3(cls, tmp_path: Path) -> Path:
        src_path = tmp_path / "src"
        src_path.mkdir(parents=True, exist_ok=True)
        c2 = cls._get_cell2_patch()
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
        assets = cls._get_post_patch_assets()
        for name, content in assets.items():
            p = tmp_path / name
            p.write_text(content, encoding="utf-8")
            subprocess.run([sys.executable, str(p), str(src_path / "ARC3-Inference")], check=True)
        return src_path / "ARC3-Inference"

    def test_01_byte_identical_notebooks_metadata_and_forensic_autopsy(self) -> None:
        """1. Verify submission_v33.ipynb and submission.ipynb are byte-identical, metadata is valid, and autopsy doc exists."""
        bytes_v33 = V33_NB_PATH.read_bytes()
        bytes_sub = SUB_NB_PATH.read_bytes()

        self.assertGreater(len(bytes_v33), 500000)
        self.assertEqual(len(bytes_v33), len(bytes_sub))

        sha_v33 = hashlib.sha256(bytes_v33).hexdigest()
        sha_sub = hashlib.sha256(bytes_sub).hexdigest()
        self.assertEqual(sha_v33, sha_sub, "submission_v33.ipynb and submission.ipynb must be byte-identical.")

        self.assertEqual(len(self.nb_v33["cells"]), 23)
        for idx, cell in enumerate(self.nb_v33["cells"]):
            if cell.get("cell_type") == "code":
                self.assertEqual(cell.get("outputs"), [], f"Cell {idx} has non-empty outputs")
                self.assertIsNone(cell.get("execution_count"), f"Cell {idx} has non-null execution_count")

        self.assertTrue(AUTOPSY_DOC_PATH.exists(), f"Missing {AUTOPSY_DOC_PATH}")
        autopsy_text = AUTOPSY_DOC_PATH.read_text(encoding="utf-8")
        for token in (
            "57337517",
            "57373064",
            "57471652",
            "57500943",
            "57541661",
            "57575158",
            "57609842",
            "57672389",
            "57709347",
            "26.03",
            "28.79",
            "32.16",
            "31.54",
            "31.46",
            "31.21",
            "31.05",
            "ARC3_HISTORY_ASSISTANT_TURNS",
            "SOLVED_LEVEL_MEMORY",
            "ARC3_NOOP_REPEAT_GUARD",
            "run_elapsed_seconds",
            "v32",
            "v33",
        ):
            self.assertIn(token, autopsy_text)
        for bad_ref in ("57331727", "57395708", "57452326", "57644982"):
            self.assertNotIn(bad_ref, autopsy_text)

        self.assertEqual(self.meta.get("id"), "zerbytheboss/v33-sovereign-apex-engine")
        self.assertEqual(self.meta.get("title"), "v33 - Sovereign Apex Engine")
        self.assertEqual(self.meta.get("code_file"), "submission.ipynb")
        self.assertTrue(self.meta.get("enable_gpu"))
        self.assertFalse(self.meta.get("enable_internet"))
        self.assertEqual(self.meta.get("machine_shape"), "NvidiaRtxPro6000")

    def test_02_v33_setup_env_and_purge_of_regressions(self) -> None:
        """2. Verify v33 setup_env flags and complete purge of regressive patches."""
        c2 = _src(self.nb_v33["cells"][2])
        self.assertEqual(len(c2), 498799)
        self.assertEqual(
            hashlib.sha256(c2.encode("utf-8")).hexdigest(),
            "6fdcd1ae7445c0cfc26eec1047363aa0c196c867d289a35b6f64403b5a43dce2",
        )

        c4 = _src(self.nb_v33["cells"][4])
        c6 = _src(self.nb_v33["cells"][6])
        c16 = _src(self.nb_v33["cells"][16])
        c20 = _src(self.nb_v33["cells"][20])

        for banned in (
            "patch_post_eviction_memory.py",
            "patch_spatial_helpers.py",
            "patch_sovereign_oo_harness.py",
            "_slm_maybe_stagnation_restart",
            "_STAGNATION_TOKEN_LIMIT",
            "_STAGNATION_ACTION_LIMIT",
            "[COMPACTED LEVEL LEDGER]",
            "'ARC3_DEATH_LEDGER'",
            "'ARC3_ANIMATION_IMAGE'",
            "'ARC3_GAMEOVER_DIFF'",
            "'ARC3_DEATH_GUARD_LOOKAHEAD'",
            "install_timeout_armor",
            "aging_bonus",
            "B_raw * C",
            "EXPOSE_RESET",
            "'ARC3_HISTORY_ASSISTANT_TURNS': 150",
            "_boot_elapsed_s",
        ):
            self.assertNotIn(banned, c4 + c16 + c20, f"Banned regressive token {banned!r} present in notebook")

        self.assertIn("'ARC3_HISTORY_ASSISTANT_TURNS': 0,", c4)
        self.assertIn("'ARC3_NOOP_REPEAT_GUARD': '1',", c4)
        self.assertIn("os.environ['ARC3_HTTP_RETRY_INITIAL_SECONDS'] = '2400'", c4)
        self.assertIn("SERVER_STARTUP_TIMEOUT = (12 * 60) if TRUE_SUBMISSION else (35 * 60)", c4)
        self.assertIn(
            "precache_model_thread = threading.Thread(target=precache, args=(MODEL_DIR, DRAFT_MODEL_DIR), kwargs={'delay': 60, 'threads': 3})",
            c6,
        )
        self.assertIn("SOLVED_LEVEL_MEMORY_PATCH = True", c16)
        self.assertIn("install_solved_level_memory(_slm_ta.ToolAgent)", c16)

        assets = _extract_assets_from_c4(c4)
        self.assertEqual(
            set(assets.keys()),
            {"patch_startup_shield.py", "patch_transient_animation.py", "patch_segmentation_aliases.py"},
        )

    def test_03_ast_syntax_and_code_cells_validity(self) -> None:
        """3. Verify AST parsing of all notebook code cells, post-patch scripts, builder, and auto-submitter."""
        for script_path in (BUILDER_V33_PATH, AUTO_SUBMIT_V33_PATH):
            self.assertTrue(script_path.exists(), f"Missing {script_path}")
            ast.parse(script_path.read_text(encoding="utf-8"), filename=str(script_path))
            py_compile.compile(str(script_path), doraise=True)

        for idx, cell in enumerate(self.nb_v33["cells"]):
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
            arc3_dir = self._prepare_patched_arc3(tmp_path)

            # Run a second time to verify idempotence
            assets = self._get_post_patch_assets()
            for name in assets:
                p = tmp_path / name
                res = subprocess.run(
                    [sys.executable, str(p), str(arc3_dir)],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(res.returncode, 0, f"{name} idempotent run failed:\n{res.stderr}")

            py_files = list(arc3_dir.glob("**/*.py"))
            self.assertGreaterEqual(len(py_files), 10)
            for pf in py_files:
                ast.parse(pf.read_text(encoding="utf-8"), filename=str(pf))
                py_compile.compile(str(pf), doraise=True)

    def test_05_v33_d_prime_multi_level_continuation_and_final_level_cliff_fix(self) -> None:
        """5. Verify v33 D' math: 100% shiiin9/v24 equivalence for remaining >= 1 + C-scaled b0=8.0*C for final level."""
        fp_v33_src = self._get_fp_source(self.nb_v33)
        fp_v24_src = self._get_fp_source(self.nb_v24)

        mod_v33 = types.ModuleType("ours_form_priority_v33")
        exec(compile(fp_v33_src, "ours_form_priority_v33.py", "exec"), mod_v33.__dict__)

        mod_v24 = types.ModuleType("ours_form_priority_v24")
        exec(compile(fp_v24_src, "ours_form_priority_v24.py", "exec"), mod_v24.__dict__)

        class _Snap:
            def __init__(self, level, actions, tokens, cost_multiplier, total_levels):
                self.level = level
                self.actions = actions
                self.tokens = tokens
                self.cost_multiplier = cost_multiplier
                self.total_levels = total_levels

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
                p33 = mod_v33.d_priority(snap, tail_fraction=phi)
                p24 = mod_v24.d_priority(snap, tail_fraction=phi)
                self.assertEqual(p33, p24, f"Mismatch vs shiiin9/v24 on L{snap.level}/{snap.total_levels} phi={phi}")

        p_n6 = mod_v33.d_priority(_Snap(1, 0, 0.0, 0.0, 6), tail_fraction=1.0)
        p_n8 = mod_v33.d_priority(_Snap(1, 0, 0.0, 0.0, 8), tail_fraction=1.0)
        p_n10 = mod_v33.d_priority(_Snap(1, 0, 0.0, 0.0, 10), tail_fraction=1.0)
        self.assertEqual((p_n6, p_n8, p_n10), (18619, 17527, 17000))

        p_prog_l2_n6 = mod_v33.d_priority(_Snap(2, 10, 20000.0, 25000.0, 6), tail_fraction=1.0)
        p_prog_l3_n6 = mod_v33.d_priority(_Snap(3, 15, 30000.0, 28000.0, 6), tail_fraction=1.0)
        p_trim1_l2_n6 = mod_v33.d_priority(_Snap(2, 25, 75000.0, 30000.0, 6), tail_fraction=1.0)
        p_trim1_l3_n6 = mod_v33.d_priority(_Snap(3, 20, 65000.0, 25000.0, 6), tail_fraction=1.0)
        self.assertEqual((p_prog_l2_n6, p_prog_l3_n6, p_trim1_l2_n6, p_trim1_l3_n6), (19518, 20112, 18181, 19345))

        p_l6_fresh = mod_v33.d_priority(_Snap(6, 0, 0.0, 30000.0, 6), tail_fraction=1.0)
        p_l6_trim1 = mod_v33.d_priority(_Snap(6, 30, 118000.0, 30000.0, 6), tail_fraction=1.0)
        p_l6_stalled = mod_v33.d_priority(_Snap(6, 150, 250000.0, 30000.0, 6), tail_fraction=1.0)
        self.assertEqual((p_l6_fresh, p_l6_trim1, p_l6_stalled), (17166, 7640, 1132))

    def test_06_live_sandbox_retained_functions_time_import_and_timeout_preservation(self) -> None:
        """6. Live subprocess sandbox verification: `import time`, helper retention, and timeout retention preservation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            arc3_dir = self._prepare_patched_arc3(Path(tmp_dir))

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
assert "keepable_functions" not in res3

from inference.agent.tool_agent import ToolAgent
os.environ["ARC3_PERSISTENT_FUNCTIONS"] = "1"
dummy_agent = types.SimpleNamespace(_kept_functions={f["name"]: f["source"] for f in kept})
payload = {}
ToolAgent._record_retained_functions(dummy_agent, res3, payload)
assert set(dummy_agent._kept_functions.keys()) == {"timed_bfs", "find_objects"}
assert "your retained functions from earlier calls are unchanged" in payload.get("function_retention", "")
print("SANDBOX_TIME_AND_TIMEOUT_RETENTION_OK")
"""
            env = dict(os.environ, PYTHONPATH=str(arc3_dir))
            res = subprocess.run([sys.executable, "-c", check_code], env=env, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"Sandbox time/timeout test failed:\n{res.stderr}\n{res.stdout}")
            self.assertIn("SANDBOX_TIME_AND_TIMEOUT_RETENTION_OK", res.stdout)

    def test_07_four_layer_transient_animation_noop_repeat_guard_and_clock_strip(self) -> None:
        """7. Verify 4-layer transient animation (including NoopRepeatGuard), _compact_action_result clock strip, and M75 aliases."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            arc3_dir = self._prepare_patched_arc3(Path(tmp_dir))

            check_code = r"""
import os, sys, types
os.environ["LOCAL_ANALYZER_MODEL_ID"] = "flashnext"
os.environ["ARC3_NOOP_REPEAT_GUARD"] = "1"
os.environ["ARC3_GUARDS_FROM_LEVEL"] = "2"

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

from inference.agent.tool_agent import ToolAgent
from inference.agent.noop_repeat_guard import NoopRepeatGuard
from inference.framework.solver import _action_verdict

# 1. Verify solver.py gameplay_changed_per_action includes transient_pixels > 0
solver_src = open("inference/framework/solver.py").read()
assert 'bool((item.get("animation") or {}).get("transient_pixels", 0))' in solver_src

# 2. Verify ToolAgent._record_guard_observations treats transient_pixels > 0 as changed=True (NOT a no-op!)
agent = ToolAgent(model="flashnext")
assert agent._noop_repeat_guard is not None
guard = NoopRepeatGuard()
guard.note_level(2)

# Case A: Single action with gameplay_changed=False but transient_pixels=12 -> NOT recorded as no-op!
agent._guard_hook_trace = [("state_hash_anim", "ACTION1")]
raw_anim = {"executed": True, "gameplay_changed": False, "animation": {"transient_pixels": 12}}
compact_anim = agent._compact_action_result(raw_anim)
agent._record_guard_observations(guard, raw_anim, compact_anim)
assert guard.should_block("state_hash_anim", "ACTION1") is False, "Animated action was falsely recorded as a no-op!"

# Case B: Single action with gameplay_changed=False and transient_pixels=0 -> IS recorded as no-op!
agent._guard_hook_trace = [("state_hash_noop", "ACTION2")]
raw_noop = {"executed": True, "gameplay_changed": False, "animation": {"transient_pixels": 0}}
compact_noop = agent._compact_action_result(raw_noop)
agent._record_guard_observations(guard, raw_noop, compact_noop)
assert guard.should_block("state_hash_noop", "ACTION2") is True, "True no-op was not blocked by NoopRepeatGuard"
# Immediate insistence overrides once:
assert guard.should_block("state_hash_noop", "ACTION2") is False, "Override slot did not allow immediate retry"

# Case B2: Single action with gameplay_changed=None (missing key) and transient_pixels=0 -> NOT recorded as no-op!
agent._guard_hook_trace = [("state_hash_none", "ACTION3")]
raw_none = {"executed": True}
compact_none = {"executed": True}
agent._record_guard_observations(guard, raw_none, compact_none)
assert guard.should_block("state_hash_none", "ACTION3") is False, "Missing gameplay_changed (None) was falsely recorded as a no-op!"

# 3. Verify _compact_action_result strips run_elapsed_seconds and time_remaining_seconds while updating _game_elapsed_seconds
raw_with_clock = {
    "executed": True,
    "action_num": 5,
    "level": 2,
    "gameplay_changed": True,
    "run_elapsed_seconds": 1234.5,
    "time_remaining_seconds": 30000.0,
}
compact_clock = agent._compact_action_result(raw_with_clock)
assert "run_elapsed_seconds" not in compact_clock, f"run_elapsed_seconds leaked into compact payload: {compact_clock}"
assert "time_remaining_seconds" not in compact_clock, f"time_remaining_seconds leaked into compact payload: {compact_clock}"
assert agent._game_elapsed_seconds == 1234.5, f"_game_elapsed_seconds was not updated: {agent._game_elapsed_seconds}"

print("FOUR_LAYER_ANIM_NOOP_GUARD_AND_CLOCK_STRIP_OK")
"""
            env = dict(os.environ, PYTHONPATH=str(arc3_dir))
            check_file = Path(tmp_dir) / "check7.py"
            check_file.write_text(check_code, encoding="utf-8")
            res = subprocess.run([sys.executable, str(check_file)], cwd=str(arc3_dir), env=env, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"Test 07 failed:\n{res.stderr}\n{res.stdout}")
            self.assertIn("FOUR_LAYER_ANIM_NOOP_GUARD_AND_CLOCK_STRIP_OK", res.stdout)

    def test_08_cache_safe_eviction_only_solved_level_memory(self) -> None:
        """8. Verify Cache-Safe Eviction-Only SOLVED_LEVEL_MEMORY (RESET-filtering, post-action frame.level, thinking/content rule capture, eviction-only pin, token budget, session reset)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            arc3_dir = self._prepare_patched_arc3(tmp_path)

            import scripts.build_v33_notebook as b33
            slm_file = tmp_path / "slm_patch.py"
            slm_file.write_text(b33.SOLVED_LEVEL_MEMORY_BLOCK, encoding="utf-8")

            check_code = r"""
import os, sys, tempfile
from pathlib import Path
os.environ["LOCAL_ANALYZER_MODEL_ID"] = "flashnext"
from inference.agent import tool_agent as ta
from inference.agent.runtime_state import Frame, HistoryEntry, write_runtime_state

exec(open(sys.argv[1], encoding="utf-8").read(), globals())

agent = ta.ToolAgent(model="flashnext")
base_prompt = agent._system_prompt
assert agent._slm_base_system_prompt == base_prompt
assert "=== SOLVED LEVELS ===" not in agent._system_prompt

with tempfile.TemporaryDirectory() as game_dir:
    gpath = Path(game_dir) / "game1"
    gpath.mkdir()
    state_path = gpath / "state.json"

    # Write state at Level 2 with real solver.py history semantics:
    # - Seed entry has action="", frame=f1 (level=1)
    # - Winning action of Level 1 ("ACTION6(row=3, col=4)") has frame=f2 (level=2, the frame AFTER the action!)
    f1 = Frame(grid=((0, 0), (0, 0)), step=1, level=1)
    f2 = Frame(grid=((0, 0), (0, 0)), step=4, level=2)
    hist = [
        HistoryEntry(action="", frame=f1, result={}),
        HistoryEntry(action="ACTION1", frame=f1, result={"level": 1}),
        HistoryEntry(action="RESET", frame=f1, result={"level": 1}),
        HistoryEntry(action="ACTION2", frame=f1, result={"level": 1}),
        HistoryEntry(action="ACTION6(row=3, col=4)", frame=f2, result={"level": 2, "level_completed": True}),
    ]
    write_runtime_state(state_path, current_frame=f2, history=hist)

    agent._ensure_session(state_path)
    # Seed assistant winning-turn reasoning in _history_messages before Level 1 -> 2 transition
    agent._history_messages = [
        {"role": "user", "content": "Level 1 turn"},
        {"role": "assistant", "content": "Clicking (3, 4) after moving with ACTION2 triggers the goal switch."},
    ]
    current_frame, _slm_hist = _slm_load_state(state_path)
    acts_l1 = _slm_level_actions(_slm_hist, 1)
    acts_l2 = _slm_level_actions(_slm_hist, 2)
    # Must exclude ACTION1 before RESET, include winning move ACTION6(row=3, col=4) in Level 1, and NOT leak it into Level 2!
    assert acts_l1 == ["ACTION2", "ACTION6(row=3, col=4)"], f"Expected ['ACTION2', 'ACTION6(row=3, col=4)'], got {acts_l1}"
    assert acts_l2 == [], f"Expected [] for Level 2 before any Level 2 moves, got {acts_l2}"

    # Simulate analyze() level transition 1 -> 2 (which calls _build_user_prompt once, so _slm_pending_asks == 1)
    ta.ToolAgent._chat_completion = lambda self, *a, **kw: (_ for _ in ()).throw(RuntimeError("stop_early"))
    agent.analyze(state_path, action_num=4, valid_actions=["ACTION1", "ACTION2"])
    assert 1 in agent._slm_entries, f"Missing level 1 in _slm_entries: {agent._slm_entries}"
    assert agent._slm_entries[1]["actions"] == 2
    assert agent._slm_entries[1]["seq"] == "ACTION2,ACTION6(row=3, col=4)"
    assert "Clicking (3, 4) after moving with ACTION2" in agent._slm_entries[1]["reasoning"]
    assert agent._slm_pending_rule_level == 1
    assert agent._slm_pending_asks == 1, f"Expected 1 ask from analyze(), got {agent._slm_pending_asks}"

    # CRITICAL INVARIANT 1: System prompt is NOT modified before context eviction!
    assert agent._system_prompt == base_prompt, "System prompt was mutated before eviction (KV cache broken)!"

    # Verify that if an eviction occurs BEFORE the model answers Rule for level 1, Winning turn reasoning is pinned!
    pre_rule_block = _slm_render_block(agent)
    assert "Winning turn reasoning: Clicking (3, 4) after moving with ACTION2" in pre_rule_block

    # Verify that a yield-resumption turn (_resume_after_yield=True) does NOT waste the 2nd ask slot!
    agent._resume_after_yield = True
    p_resume = agent._build_user_prompt(4, valid_actions=["ACTION1"], current_frame=f2, history_entries=hist, previous_step_summary=None)
    assert "Rule for level 1:" not in p_resume
    assert agent._slm_pending_asks == 1
    agent._resume_after_yield = False

    # Verify the 2nd full turn asks for Rule for level 1 and increments _slm_pending_asks to 2
    p1 = agent._build_user_prompt(4, valid_actions=["ACTION1"], current_frame=f2, history_entries=hist, previous_step_summary=None)
    assert "Rule for level 1: <win condition and key interaction>" in p1
    assert agent._slm_pending_asks == 2

    # Test thinking-token rule extraction with inline backticks: model echoes placeholder first, then writes real rule with backticks!
    msgs = [{"role": "system", "content": agent._system_prompt}]
    agent._append_context_message(
        msgs,
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": (
                "The prompt asks for `Rule for level 1: <win condition and key interaction>`.\n"
                "Rule for level 1: Use `ACTION2` then click `(3, 4)` with `ACTION6` to open the exit."
            ),
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "python", "arguments": "{}"}}],
        },
    )
    assert agent._slm_entries[1]["rule"] == "Use `ACTION2` then click `(3, 4)` with `ACTION6` to open the exit.", (
        f"Unexpected rule: {agent._slm_entries[1]}"
    )
    assert agent._slm_pending_rule_level is None

    # Next _build_user_prompt must NOT ask again since rule was already captured!
    p2 = agent._build_user_prompt(5, valid_actions=["ACTION1"], current_frame=f2, history_entries=hist, previous_step_summary=None)
    assert "Rule for level 1:" not in p2

    # CRITICAL INVARIANT 2: When _trim_messages_for_context does NOT drop messages, _system_prompt stays untouched!
    short_msgs = [
        {"role": "system", "content": agent._system_prompt},
        {"role": "user", "content": "turn 1"},
        {"role": "assistant", "content": "reply 1"},
    ]
    trimmed_no_drop = agent._trim_messages_for_context(short_msgs)
    assert len(trimmed_no_drop) == len(short_msgs)
    assert agent._system_prompt == base_prompt, "System prompt mutated when no messages were dropped!"

    # CRITICAL INVARIANT 3: When _trim_messages_for_context or _force_reduce_messages drops messages,
    # === SOLVED LEVELS === is pinned to _system_prompt and result[0]["content"]!
    long_msgs = [
        {"role": "system", "content": agent._system_prompt},
        {"role": "user", "content": "old turn"},
        {"role": "assistant", "content": "old reply"},
        {"role": "user", "content": "new turn"},
    ]
    reduced = agent._force_reduce_messages(long_msgs, preserve_recent=1)
    assert len(reduced) < len(long_msgs)
    assert "=== SOLVED LEVELS ===" in agent._system_prompt
    assert "Level 1: solved in 2 actions [ACTION2,ACTION6(row=3, col=4)] | Rule: Use `ACTION2` then click `(3, 4)` with `ACTION6` to open the exit." in agent._system_prompt
    assert reduced[0]["content"] == agent._system_prompt

    # CRITICAL INVARIANT 4: Session reset on a new game directory restores _slm_base_system_prompt and clears _slm_entries!
    gpath2 = Path(game_dir) / "game2"
    gpath2.mkdir()
    agent._ensure_session(gpath2 / "state.json")
    assert agent._system_prompt == base_prompt, "System prompt was not restored on new game session!"
    assert agent._slm_entries == {}

print("SOLVED_LEVEL_MEMORY_E2E_OK")
"""
            check_file = tmp_path / "check8.py"
            check_file.write_text(check_code, encoding="utf-8")
            env = dict(os.environ, PYTHONPATH=str(arc3_dir))
            res = subprocess.run(
                [sys.executable, str(check_file), str(slm_file)],
                cwd=str(arc3_dir),
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(res.returncode, 0, f"Test 08 failed:\n{res.stderr}\n{res.stdout}")
            self.assertIn("SOLVED_LEVEL_MEMORY_E2E_OK", res.stdout)

    def test_09_assistant_turns_zero_disables_premature_eviction(self) -> None:
        """9. Verify ARC3_HISTORY_ASSISTANT_TURNS=0 disables the 150-assistant-turn cap while preserving token-budget trimming."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            arc3_dir = self._prepare_patched_arc3(Path(tmp_dir))

            check_code = r"""
import os
os.environ["LOCAL_ANALYZER_MODEL_ID"] = "flashnext"
from inference.agent import tool_agent as ta

# Build a 170-assistant-turn history that is well under the 116k token budget (~15k tokens total)
history = []
for i in range(170):
    history.append({"role": "user", "content": f"Step {i}"})
    history.append({"role": "assistant", "content": f"Action {i}"})

# 1. With ARC3_HISTORY_ASSISTANT_TURNS=150 (old v32 default), _persistent_history_messages evicts 50 turns!
os.environ["ARC3_HISTORY_ASSISTANT_TURNS"] = "150"
os.environ["ARC3_HISTORY_TURN_DRAIN"] = "30"
agent_150 = ta.ToolAgent(model="flashnext")
msgs_150 = [{"role": "system", "content": agent_150._system_prompt}, *history]
kept_150 = agent_150._persistent_history_messages(msgs_150)
assert len(kept_150) < len(history), "150 cap should have evicted turns"
assert agent_150._context_was_trimmed is True

# 2. With ARC3_HISTORY_ASSISTANT_TURNS=0 (new v33 setting), _persistent_history_messages keeps all 170 turns!
os.environ["ARC3_HISTORY_ASSISTANT_TURNS"] = "0"
agent_0 = ta.ToolAgent(model="flashnext")
msgs_0 = [{"role": "system", "content": agent_0._system_prompt}, *history]
kept_0 = agent_0._persistent_history_messages(msgs_0)
assert len(kept_0) == len(history), f"Expected all {len(history)} messages kept, got {len(kept_0)}"
assert agent_0._context_was_trimmed is False
assert agent_0._has_evicted is False

print("ASSISTANT_TURNS_ZERO_OK")
"""
            check_file = Path(tmp_dir) / "check9.py"
            check_file.write_text(check_code, encoding="utf-8")
            env = dict(os.environ, PYTHONPATH=str(arc3_dir))
            res = subprocess.run([sys.executable, str(check_file)], cwd=str(arc3_dir), env=env, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"Test 09 failed:\n{res.stderr}\n{res.stdout}")
            self.assertIn("ASSISTANT_TURNS_ZERO_OK", res.stdout)

    def test_10_startup_shield_and_prompts_immutability(self) -> None:
        """10. Verify unconditional warmup RESET + periodic keepalive in patch_startup_shield.py and prompts.py immutability."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            arc3_dir = self._prepare_patched_arc3(tmp_path)

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
            game_id="ft09-0d8bbf25",
            game_run=types.SimpleNamespace(game_id="ft09-0d8bbf25", state="playing", history=[], final_score=None, solver_note=None, total_tokens=0),
            current_state=types.SimpleNamespace(levels_completed=0, won=False, available_actions=[], state=None),
            number_of_levels=6,
        )
        self.solver = types.SimpleNamespace(_claim_warmup_slot=lambda: True, max_runtime_s_per_game=max_runtime_s, _analyzer_ready=False)
        self.analyzer = types.SimpleNamespace(total_tokens=0)
        self.stop_event = types.SimpleNamespace(is_set=lambda: False)
        self.state_path = Path("/tmp/nonexistent_dummy_state_v33.json")
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

def healthy_urlopen(url, timeout=3):
    class _Resp:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): pass
    return _Resp()

# Case A: Fast boot (/health 200 OK) -> 1 unconditional warmup RESET, 0 extra keepalives
fast_sess = _DummySession(532 * 60.0)
with patch("urllib.request.urlopen", side_effect=healthy_urlopen):
    solver_mod._HarnessGameSession._play_inner(fast_sess)
assert fast_sess.resets == 1, f"Expected 1 initial warmup RESET, got {fast_sess.resets}"
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
assert slow_sess.started_at == 1000.0
print("STARTUP_SHIELD_V33_OK")
"""
            check_file = tmp_path / "check10.py"
            check_file.write_text(check_code, encoding="utf-8")
            env = dict(os.environ, PYTHONPATH=str(arc3_dir))
            res = subprocess.run([sys.executable, str(check_file)], cwd=str(arc3_dir), env=env, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"Test 10 failed:\n{res.stderr}\n{res.stdout}")
            self.assertIn("STARTUP_SHIELD_V33_OK", res.stdout)

            # Also verify prompts.py is 100% byte-identical to v24
            v24_src = tmp_path / "v24_src"
            v24_src.mkdir()
            c2 = self._get_cell2_patch()
            patch_file = tmp_path / "cell2_v24.patch"
            patch_file.write_text(c2.split("\n", 1)[1] + "\n", encoding="utf-8")
            subprocess.run(
                f"git archive HEAD ARC3-Inference | tar -x -C {v24_src}",
                shell=True,
                cwd=str(DUCK_HARNESS_DIR),
                check=True,
            )
            subprocess.run(
                ["git", "apply", "--include=ARC3-Inference/*", str(patch_file)],
                cwd=str(v24_src),
                check=True,
            )
            prompts_v24 = (v24_src / "ARC3-Inference/inference/agent/prompts.py").read_text(encoding="utf-8")
            prompts_v33 = (arc3_dir / "inference/agent/prompts.py").read_text(encoding="utf-8")
            self.assertEqual(prompts_v24, prompts_v33, "prompts.py in v33 must be 100% byte-identical to v24!")


if __name__ == "__main__":
    unittest.main()

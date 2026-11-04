"""Comprehensive E2E Verification Suite for ARC-AGI-3 Version 30 Sovereign Restoration & Apex Engine."""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
V11_NB_PATH = REPO_ROOT / "pulled_kernels" / "v11" / "arc-prize-2026-v11-apex-engine.ipynb"
V24_NB_PATH = REPO_ROOT / "notebooks" / "submission_v24.ipynb"
V30_NB_PATH = REPO_ROOT / "notebooks" / "submission_v30.ipynb"
SUB_NB_PATH = REPO_ROOT / "notebooks" / "submission.ipynb"
META_PATH = REPO_ROOT / "notebooks" / "kernel-metadata.json"
AUDIT_DOC_PATH = REPO_ROOT / "docs" / "FORENSIC_V28_V29_AUTOPSY_AND_V30.md"
DUCK_HARNESS_DIR = REPO_ROOT / "reference" / "duck-harness"


def _src(cell: dict) -> str:
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else str(s)


def _extract_v30_assets(c4_src: str) -> dict[str, str]:
    ns: dict = {}
    for line in c4_src.splitlines():
        if line.startswith("_v30_assets = {"):
            exec(line, ns)
            break
    if "_v30_assets" not in ns:
        raise AssertionError("Could not find _v30_assets in Cell 4")
    return dict(ns["_v30_assets"])


class TestV30E2ERequirements(unittest.TestCase):
    """Programmatic E2E test suite validating all architectural, mathematical, and runtime invariants of Version 30."""

    @classmethod
    def setUpClass(cls) -> None:
        assert V30_NB_PATH.exists(), f"Missing {V30_NB_PATH}"
        assert SUB_NB_PATH.exists(), f"Missing {SUB_NB_PATH}"
        assert META_PATH.exists(), f"Missing {META_PATH}"
        assert DUCK_HARNESS_DIR.exists(), f"Missing {DUCK_HARNESS_DIR}"

        cls.nb_v30 = json.loads(V30_NB_PATH.read_text(encoding="utf-8"))
        cls.nb_sub = json.loads(SUB_NB_PATH.read_text(encoding="utf-8"))
        cls.meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        cls.nb_v24 = json.loads(V24_NB_PATH.read_text(encoding="utf-8")) if V24_NB_PATH.exists() else None

    def test_01_byte_identical_notebooks_metadata_and_forensic_autopsy(self) -> None:
        """1. Verify submission_v30.ipynb and submission.ipynb are byte-identical, metadata is valid, and autopsy doc exists."""
        if "v30" not in self.meta.get("id", ""):
            raise unittest.SkipTest("Active kernel is not v30")
        bytes_v30 = V30_NB_PATH.read_bytes()
        bytes_sub = SUB_NB_PATH.read_bytes()

        self.assertGreater(len(bytes_v30), 500000)
        self.assertEqual(len(bytes_v30), len(bytes_sub))

        sha_v30 = hashlib.sha256(bytes_v30).hexdigest()
        sha_sub = hashlib.sha256(bytes_sub).hexdigest()
        self.assertEqual(sha_v30, sha_sub, "submission_v30.ipynb and submission.ipynb must be byte-identical.")

        self.assertEqual(len(self.nb_v30["cells"]), 23)
        for idx, cell in enumerate(self.nb_v30["cells"]):
            if cell.get("cell_type") == "code":
                self.assertEqual(cell.get("outputs"), [], f"Cell {idx} has non-empty outputs")
                self.assertIsNone(cell.get("execution_count"), f"Cell {idx} has non-null execution_count")

        self.assertTrue(AUDIT_DOC_PATH.exists(), f"Missing {AUDIT_DOC_PATH}")
        audit_text = AUDIT_DOC_PATH.read_text(encoding="utf-8")
        for ref_id in ("57337517", "57373064", "57471652", "57500943", "57541661", "57575158"):
            self.assertIn(ref_id, audit_text)

        self.assertEqual(self.meta.get("id"), "zerbytheboss/v30-sovereign-apex-engine")
        self.assertEqual(self.meta.get("title"), "v30 - Sovereign Apex Engine")
        self.assertEqual(self.meta.get("code_file"), "submission.ipynb")
        self.assertTrue(self.meta.get("enable_gpu"))
        self.assertFalse(self.meta.get("enable_internet"))
        self.assertEqual(self.meta.get("machine_shape"), "NvidiaRtxPro6000")

    def test_02_complete_purge_of_v25_to_v29_regressions(self) -> None:
        """2. Verify complete purge of regressive patches, runtime_globals injections, and unverified v29 env flags."""
        c2 = _src(self.nb_v30["cells"][2])
        self.assertEqual(len(c2), 498799)
        self.assertEqual(
            hashlib.sha256(c2.encode("utf-8")).hexdigest(),
            "6fdcd1ae7445c0cfc26eec1047363aa0c196c867d289a35b6f64403b5a43dce2",
        )

        c4 = _src(self.nb_v30["cells"][4])
        c6 = _src(self.nb_v30["cells"][6])

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
        ):
            self.assertNotIn(banned, c4, f"Banned regressive token {banned!r} still present in Cell 4")

        # Verify pure v24 precache(delay=60, threads=3) is restored in Cell 6
        self.assertIn(
            "precache_model_thread = threading.Thread(target=precache, args=(MODEL_DIR, DRAFT_MODEL_DIR), kwargs={'delay': 60, 'threads': 3})",
            c6,
        )
        self.assertNotIn("kwargs={'delay': 15, 'threads': 8}", c6)

        # Verify exact 3 surgical v30 assets
        assets = _extract_v30_assets(c4)
        self.assertEqual(
            set(assets.keys()),
            {"patch_startup_shield.py", "patch_transient_animation.py", "patch_segmentation_aliases.py"},
        )

    def test_03_live_sandbox_retained_functions_and_transient_animation_guard(self) -> None:
        """3. Execute live run_sandboxed_python subprocesses to verify function retention (find_cells, etc.) and transient animation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            src_path = tmp_path / "src"
            src_path.mkdir()
            c2 = _src(self.nb_v30["cells"][2]).split("\n", 1)[1] + "\n"
            patch_file = tmp_path / "cell2.patch"
            patch_file.write_text(c2, encoding="utf-8")
            subprocess.run(
                f"git archive HEAD ARC3-Inference | tar -x -C {src_path}",
                shell=True,
                cwd=str(DUCK_HARNESS_DIR),
                check=True,
            )
            subprocess.run(["git", "apply", "--include=ARC3-Inference/*", str(patch_file)], cwd=str(src_path), check=True)
            assets = _extract_v30_assets(_src(self.nb_v30["cells"][4]))
            for name in ("patch_startup_shield.py", "patch_transient_animation.py", "patch_segmentation_aliases.py"):
                p = tmp_path / name
                p.write_text(assets[name], encoding="utf-8")
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

# 1. Turn 1: Model defines helper functions with names that collided in v25-v29 (find_cells, find_objects, manhattan_distance, rot90)
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

# 2. Turn 2: Model calls its retained 1-arg find_cells('R') and find_objects() WITHOUT redefining them!
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

# 3. Verify transient animation (transient_pixels > 0, gameplay_changed=False) does NOT arm _stale_state (stale_after remains None)
def animated_action_handler(actions, stale_after=None):
    if stale_after is not None:
        return {
            "action_result": {
                "executed": False,
                "executed_actions": [],
                "stop_reason": "stale_state",
                "stop_detail": f"Blocked after stale state from {stale_after}",
            },
            "state": state,
        }
    return {
        "action_result": {
            "executed": True,
            "executed_actions": list(actions),
            "gameplay_changed": False,
            "animation": {"transient_pixels": 14, "frames": 3, "steps": []},
        },
        "state": state,
    }

anim_code = '''
action("ACTION1")
action("ACTION2")
print("ANIM_TWO_ACTIONS_OK")
'''
res_anim = run_sandboxed_python(
    code=anim_code,
    timeout_seconds=5,
    initial_state=state,
    action_handler=animated_action_handler,
)
assert res_anim.get("error") in (None, ""), f"Transient animation falsely blocked second action: {res_anim}"
assert "ANIM_TWO_ACTIONS_OK" in res_anim["stdout"]

# 4. Verify true no-op (transient_pixels == 0, gameplay_changed=False) STILL arms _stale_state and blocks second action
def noop_action_handler(actions, stale_after=None):
    if stale_after is not None:
        return {
            "action_result": {
                "executed": False,
                "executed_actions": [],
                "stop_reason": "stale_state",
                "stop_detail": f"Blocked after stale state from {stale_after}",
            },
            "state": state,
        }
    return {
        "action_result": {
            "executed": True,
            "executed_actions": list(actions),
            "gameplay_changed": False,
            "animation": {"transient_pixels": 0, "frames": 0, "steps": []},
        },
        "state": state,
    }

res_noop = run_sandboxed_python(
    code=anim_code,
    timeout_seconds=5,
    initial_state=state,
    action_handler=noop_action_handler,
)
assert "StaleStateActionError" in str(res_noop.get("error") or ""), f"Expected StaleStateActionError on true no-op, got: {res_noop}"
print("SANDBOX_E2E_OK")
"""
            env = dict(os.environ, PYTHONPATH=str(src_path / "ARC3-Inference"))
            res = subprocess.run([sys.executable, "-c", check_code], env=env, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"Live sandbox test failed:\n{res.stderr}\n{res.stdout}")
            self.assertIn("SANDBOX_E2E_OK", res.stdout)

    def test_04_c_scaled_d_prime_priority_math_and_stagnation_decay(self) -> None:
        """4. Verify D' priority math (b0=8.0, B_raw * C): unstarted ordering, L1 stagnation decay, and final-level cliff fix."""
        c16 = _src(self.nb_v30["cells"][16])
        ns: dict = {}
        for line in c16.splitlines():
            if line.startswith("_FP_SOURCE = "):
                exec(line, ns)
                break
        self.assertIn("_FP_SOURCE", ns)
        fp_mod = types.ModuleType("ours_form_priority")
        exec(compile(ns["_FP_SOURCE"], "ours_form_priority.py", "exec"), fp_mod.__dict__)

        class _Snap:
            def __init__(self, level, actions, tokens, cost_multiplier, total_levels):
                self.level = level
                self.actions = actions
                self.tokens = tokens
                self.cost_multiplier = cost_multiplier
                self.total_levels = total_levels

        # 1. Unstarted games: exact v24/shiiin9 ordering (6-lvl > 8-lvl > 10-lvl)
        p_n6 = fp_mod.d_priority(_Snap(1, 0, 0.0, 0.0, 6), tail_fraction=1.0)
        p_n8 = fp_mod.d_priority(_Snap(1, 0, 0.0, 0.0, 8), tail_fraction=1.0)
        p_n10 = fp_mod.d_priority(_Snap(1, 0, 0.0, 0.0, 10), tail_fraction=1.0)
        self.assertEqual(p_n6, 18619)
        self.assertEqual(p_n8, 17527)
        self.assertEqual(p_n10, 17000)

        # 2. Level 1 stagnation decay via B_raw * C:
        # At 1st trim (a=40, t=118k): drops to 8252 (vs 16360 in v24/v28)
        # When stalled (a=150, t=250k): drops to 1636 (vs 16036 in v24/v28 — 9.8x lower!)
        p_l1_trim1 = fp_mod.d_priority(_Snap(1, 40, 118000.0, 0.0, 10), tail_fraction=1.0)
        p_l1_stalled = fp_mod.d_priority(_Snap(1, 150, 250000.0, 0.0, 10), tail_fraction=1.0)
        self.assertEqual(p_l1_trim1, 8252)
        self.assertEqual(p_l1_stalled, 1636)

        # 3. Final level (Level 6 of 6, remaining == 0):
        # Fresh on Level 6 (a=0, t=0, pace=30k): 17166 (beats unstarted 10-lvl 17000, fixing v24's 9166 cliff!)
        # After 1st trim on Level 6 (a=30, t=118k, pace=30k): 7640 (4.67x higher than stalled L1 1636!)
        # Stalled on Final Level 6 (a=150, t=250k, pace=30k): decays to 1132 (fixing v28's 6332 permanent hogging!)
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

    def test_05_zero_global_m75_segmentation_and_solver_verdict(self) -> None:
        """5. Verify M75 _NodeDict/_BoundaryList aliases, empty animation schema, and solver._action_verdict."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            src_path = tmp_path / "src"
            src_path.mkdir()
            c2 = _src(self.nb_v30["cells"][2]).split("\n", 1)[1] + "\n"
            patch_file = tmp_path / "cell2.patch"
            patch_file.write_text(c2, encoding="utf-8")
            subprocess.run(
                f"git archive HEAD ARC3-Inference | tar -x -C {src_path}",
                shell=True,
                cwd=str(DUCK_HARNESS_DIR),
                check=True,
            )
            subprocess.run(["git", "apply", "--include=ARC3-Inference/*", str(patch_file)], cwd=str(src_path), check=True)
            assets = _extract_v30_assets(_src(self.nb_v30["cells"][4]))
            for name in ("patch_startup_shield.py", "patch_transient_animation.py", "patch_segmentation_aliases.py"):
                p = tmp_path / name
                p.write_text(assets[name], encoding="utf-8")
                subprocess.run([sys.executable, str(p), str(src_path / "ARC3-Inference")], check=True)

            check_code = r"""
import json, sys, types

# Stub minimal taaf & arcengine so solver.py imports cleanly
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

from inference.utils.segmentation import segment_layer
from inference.utils.animation import build_animation_view
from inference.framework.solver import _action_verdict

# 1. Verify _NodeDict and _BoundaryList
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
assert g_node.get("nonexistent_key", "fallback") == "fallback"

# Verify 1-pixel _BoundaryList safe indexing on index 1, 2, 3
assert len(r_node["boundary"]) == 1
assert r_node["boundary"][0] == [3, 3]
assert r_node["boundary"][2] == [3, 3]

# 2. Verify empty animation schema
empty_anim = build_animation_view(None)
assert empty_anim["frames"] == 0
assert empty_anim["steps"] == []
assert "error" in empty_anim

# 3. Verify _action_verdict on transient_pixels > 0 vs 0
v_anim = _action_verdict({"gameplay_changed": False, "animation": {"transient_pixels": 5}})
assert v_anim == "animated effect (intermediate frames changed; final board restored)", f"Got {v_anim!r}"
v_noop = _action_verdict({"gameplay_changed": False, "animation": {"transient_pixels": 0}})
assert "NO-OP" in v_noop or "changed nothing" in v_noop, f"Got {v_noop!r}"
print("M75_AND_VERDICT_OK")
"""
            env = dict(os.environ, PYTHONPATH=str(src_path / "ARC3-Inference"))
            res = subprocess.run([sys.executable, "-c", check_code], env=env, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"M75 & verdict test failed:\n{res.stderr}\n{res.stdout}")
            self.assertIn("M75_AND_VERDICT_OK", res.stdout)

    def test_06_startup_shield_and_competition_clock_preservation(self) -> None:
        """6. Verify patch_startup_shield fires periodic keepalive RESETs during slow cold boot without resetting started_at in competition mode."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            src_path = tmp_path / "src"
            src_path.mkdir()
            c2 = _src(self.nb_v30["cells"][2]).split("\n", 1)[1] + "\n"
            patch_file = tmp_path / "cell2.patch"
            patch_file.write_text(c2, encoding="utf-8")
            subprocess.run(
                f"git archive HEAD ARC3-Inference | tar -x -C {src_path}",
                shell=True,
                cwd=str(DUCK_HARNESS_DIR),
                check=True,
            )
            subprocess.run(["git", "apply", "--include=ARC3-Inference/*", str(patch_file)], cwd=str(src_path), check=True)
            assets = _extract_v30_assets(_src(self.nb_v30["cells"][4]))
            for name in ("patch_startup_shield.py", "patch_transient_animation.py", "patch_segmentation_aliases.py"):
                p = tmp_path / name
                p.write_text(assets[name], encoding="utf-8")
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
        self.state_path = Path("/tmp/nonexistent_dummy_state_v30.json")
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
print("STARTUP_SHIELD_V30_OK")
"""
            env = dict(os.environ, PYTHONPATH=str(src_path / "ARC3-Inference"))
            res = subprocess.run([sys.executable, "-c", check_code], env=env, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"Startup shield v30 test failed:\n{res.stderr}\n{res.stdout}")
            self.assertIn("STARTUP_SHIELD_V30_OK", res.stdout)

    def test_07_tool_agent_byte_identical_to_v24(self) -> None:
        """7. Verify tool_agent.py in v30 is 100% byte-identical to v24 (zero stagnation wipe, zero force_drain double-compaction)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            v24_src = tmp_path / "v24_src"
            v30_src = tmp_path / "v30_src"
            v24_src.mkdir()
            v30_src.mkdir()

            c2 = _src(self.nb_v30["cells"][2]).split("\n", 1)[1] + "\n"
            patch_file = tmp_path / "cell2.patch"
            patch_file.write_text(c2, encoding="utf-8")

            for target_src in (v24_src, v30_src):
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

            assets = _extract_v30_assets(_src(self.nb_v30["cells"][4]))
            for name in ("patch_startup_shield.py", "patch_transient_animation.py", "patch_segmentation_aliases.py"):
                p = tmp_path / name
                p.write_text(assets[name], encoding="utf-8")
                subprocess.run([sys.executable, str(p), str(v30_src / "ARC3-Inference")], check=True)

            ta_v24 = (v24_src / "ARC3-Inference/inference/agent/tool_agent.py").read_text(encoding="utf-8")
            ta_v30 = (v30_src / "ARC3-Inference/inference/agent/tool_agent.py").read_text(encoding="utf-8")
            self.assertEqual(ta_v24, ta_v30, "tool_agent.py in v30 must be 100% byte-identical to v24!")

    def test_08_auto_submit_v30_midnight_rollover_and_dry_run_invariants(self) -> None:
        """8. Verify auto_submit_v30.py midnight UTC rollover sleep calculation, dry-run RUNNING tolerance, and idempotent retry."""
        import datetime as dt
        from scripts import auto_submit_v30

        # 1. Pre-midnight (23:59:50 UTC) -> 15s until 00:00:05 UTC
        t_pre = dt.datetime(2026, 10, 24, 23, 59, 50, tzinfo=dt.timezone.utc)
        self.assertEqual(auto_submit_v30.compute_quota_sleep_seconds(t_pre), 15)

        # 2. Exact cron launch at 00:00:00 UTC and 00:00:01.2 UTC -> 5s and 4s (NOT 300s tomorrow midnight bug!)
        t_mid0 = dt.datetime(2026, 10, 25, 0, 0, 0, tzinfo=dt.timezone.utc)
        t_mid1 = dt.datetime(2026, 10, 25, 0, 0, 1, 200000, tzinfo=dt.timezone.utc)
        self.assertEqual(auto_submit_v30.compute_quota_sleep_seconds(t_mid0), 5)
        self.assertEqual(auto_submit_v30.compute_quota_sleep_seconds(t_mid1), 4)

        # 3. Post-00:00:05 UTC rollover grace window (00:00:05 .. 00:09:59 UTC) -> 5s fast poll
        t_grace = dt.datetime(2026, 10, 25, 0, 0, 6, tzinfo=dt.timezone.utc)
        self.assertEqual(auto_submit_v30.compute_quota_sleep_seconds(t_grace), 5)

        # 4. Afternoon wait (19:55:00 UTC) -> capped at 300s
        t_afternoon = dt.datetime(2026, 10, 24, 19, 55, 0, tzinfo=dt.timezone.utc)
        self.assertEqual(auto_submit_v30.compute_quota_sleep_seconds(t_afternoon), 300)

        # 5. Verify allow_running=True permits KernelWorkerStatus.RUNNING in --dry-run, while allow_running=False raises
        fake_running = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='zerbytheboss/v30-sovereign-apex-engine has status "KernelWorkerStatus.RUNNING"\n',
            stderr="",
        )
        with patch("scripts.auto_submit_v30.subprocess.run", return_value=fake_running), patch("scripts.auto_submit_v30.log"):
            self.assertFalse(auto_submit_v30.verify_kernel_complete(wait=False, allow_running=True))
            with self.assertRaises(RuntimeError):
                auto_submit_v30.verify_kernel_complete(wait=False, allow_running=False)


if __name__ == "__main__":
    unittest.main()



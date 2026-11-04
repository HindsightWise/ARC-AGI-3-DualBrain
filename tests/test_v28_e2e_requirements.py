"""Comprehensive E2E Verification Suite for ARC-AGI-3 Version 28 Sovereign Apex Engine."""

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
import threading
import time
import types
import unittest
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
V11_NB_PATH = REPO_ROOT / "pulled_kernels" / "v11" / "arc-prize-2026-v11-apex-engine.ipynb"
V24_NB_PATH = REPO_ROOT / "notebooks" / "submission_v24.ipynb"
V28_NB_PATH = REPO_ROOT / "notebooks" / "submission_v28.ipynb"
SUB_NB_PATH = REPO_ROOT / "notebooks" / "submission.ipynb"
META_PATH = REPO_ROOT / "notebooks" / "kernel-metadata.json"
AUDIT_DOC_PATH = REPO_ROOT / "docs" / "FORENSIC_V1_TO_V28_COMPLETE_AUDIT.md"
DUCK_HARNESS_DIR = REPO_ROOT / "reference" / "duck-harness"


def _src(cell: dict) -> str:
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else str(s)


def _extract_v28_assets(c4_src: str) -> dict[str, str]:
    ns: dict = {}
    for line in c4_src.splitlines():
        if line.startswith("_v28_assets = {"):
            exec(line, ns)
            break
    if "_v28_assets" not in ns:
        raise AssertionError("Could not find _v28_assets in Cell 4")
    return dict(ns["_v28_assets"])


class TestV28E2ERequirements(unittest.TestCase):
    """Programmatic E2E test suite validating all architectural and behavioral requirements of Version 28."""

    @classmethod
    def setUpClass(cls) -> None:
        assert V28_NB_PATH.exists(), f"Missing {V28_NB_PATH}"
        assert SUB_NB_PATH.exists(), f"Missing {SUB_NB_PATH}"
        assert META_PATH.exists(), f"Missing {META_PATH}"
        assert DUCK_HARNESS_DIR.exists(), f"Missing {DUCK_HARNESS_DIR}"

        cls.nb_v28 = json.loads(V28_NB_PATH.read_text(encoding="utf-8"))
        cls.nb_sub = json.loads(SUB_NB_PATH.read_text(encoding="utf-8"))
        cls.meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        cls.nb_v24 = json.loads(V24_NB_PATH.read_text(encoding="utf-8")) if V24_NB_PATH.exists() else None

    def test_01_byte_identical_notebooks_and_audit_artifacts(self) -> None:
        """1. Verify submission_v28.ipynb and submission.ipynb are byte-identical, and audit docs & metadata are valid."""
        if "v28" not in self.meta.get("id", ""):
            raise unittest.SkipTest("Active kernel is not v28")
        bytes_v28 = V28_NB_PATH.read_bytes()
        bytes_sub = SUB_NB_PATH.read_bytes()

        self.assertGreater(len(bytes_v28), 500000)
        self.assertEqual(len(bytes_v28), len(bytes_sub))

        sha_v28 = hashlib.sha256(bytes_v28).hexdigest()
        sha_sub = hashlib.sha256(bytes_sub).hexdigest()
        self.assertEqual(sha_v28, sha_sub, "submission_v28.ipynb and submission.ipynb must be byte-identical.")

        self.assertEqual(len(self.nb_v28["cells"]), 23)
        for idx, cell in enumerate(self.nb_v28["cells"]):
            if cell.get("cell_type") == "code":
                self.assertEqual(cell.get("outputs"), [], f"Cell {idx} has non-empty outputs")
                self.assertIsNone(cell.get("execution_count"), f"Cell {idx} has non-null execution_count")

        self.assertTrue(AUDIT_DOC_PATH.exists(), f"Missing {AUDIT_DOC_PATH}")
        audit_text = AUDIT_DOC_PATH.read_text(encoding="utf-8")
        for ref_id in ("57500943", "57471652", "57373064", "57337517", "57315642", "57194042", "750173", "750885"):
            self.assertIn(ref_id, audit_text)

        self.assertEqual(self.meta.get("id"), "zerbytheboss/v28-sovereign-apex-engine")
        self.assertEqual(self.meta.get("title"), "v28 - Sovereign Apex Engine")
        self.assertEqual(self.meta.get("code_file"), "submission.ipynb")
        self.assertTrue(self.meta.get("enable_gpu"))
        self.assertFalse(self.meta.get("enable_internet"))
        self.assertEqual(self.meta.get("machine_shape"), "NvidiaRtxPro6000")

    def test_02_pure_v24_substrate_and_cold_boot_immunity_config(self) -> None:
        """2. Verify Cell 2 hash, 4-layer cold-boot immunity settings, and complete purge of prompt/timeout mutations."""
        c2 = _src(self.nb_v28["cells"][2])
        self.assertEqual(len(c2), 498799)
        self.assertEqual(
            hashlib.sha256(c2.encode("utf-8")).hexdigest(),
            "6fdcd1ae7445c0cfc26eec1047363aa0c196c867d289a35b6f64403b5a43dce2",
        )

        c4 = _src(self.nb_v28["cells"][4])
        self.assertIn("SERVER_STARTUP_TIMEOUT = (10 * 60) if TRUE_SUBMISSION else (45 * 60)", c4)
        self.assertIn("os.environ['ARC3_WARMUP_ACTION_GAMES'] = '1' if TRUE_SUBMISSION else '0'", c4)
        self.assertIn("os.environ['ARC3_HTTP_RETRY_INITIAL_SECONDS'] = '3600'", c4)
        self.assertNotIn("patch_solved_level_memory.py", c4)
        self.assertNotIn("animation_archive.py", c4)
        self.assertNotIn("fable_layer", c4)
        self.assertNotIn("ARC3_MIN_READ_TIMEOUT", c4)

        c6 = _src(self.nb_v28["cells"][6])
        self.assertIn("kwargs={'delay': 15, 'threads': 8}", c6)

        c12 = _src(self.nb_v28["cells"][12])
        self.assertIn("precache_model_thread.start()", c12)
        self.assertIn("_sgl_restarts", c12)
        self.assertIn("MEMFRAC=0.96", c12)
        self.assertIn("MAXREQ=10", c12)

        c16 = _src(self.nb_v28["cells"][16])
        self.assertNotIn("install_timeout_armor", c16)
        self.assertIn('"b0": 6.0, "b1": 10.0, "b2": 14.0, "b3": 16.0', c16)

        c20 = _src(self.nb_v28["cells"][20])
        self.assertIn("Commit rehearsal gate failed", c20)
        self.assertIn("#OURS_FRESH", c20)

    def test_03_sandbox_git_apply_and_hooks_ast(self) -> None:
        """3. Verify clean git apply of Cell 2 base patch + all 3 v28 post-patches with 100% AST/py_compile validity."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            src_path = tmp_path / "src"
            src_path.mkdir(parents=True)

            c2 = _src(self.nb_v28["cells"][2])
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

            assets = _extract_v28_assets(_src(self.nb_v28["cells"][4]))
            self.assertEqual(
                set(assets.keys()),
                {"patch_startup_shield.py", "patch_animation_empty.py", "patch_spatial_helpers.py"},
            )
            v28_dir = tmp_path / "arcagi_v28"
            v28_dir.mkdir(parents=True)
            for name, text in assets.items():
                (v28_dir / name).write_text(text, encoding="utf-8")

            for patch_script in ("patch_startup_shield.py", "patch_animation_empty.py", "patch_spatial_helpers.py"):
                for _ in range(2):  # test both initial apply and idempotency
                    run_res = subprocess.run(
                        [sys.executable, str(v28_dir / patch_script), str(src_path / "ARC3-Inference")],
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(run_res.returncode, 0, f"{patch_script} failed:\n{run_res.stderr}")

            arc3_dir = src_path / "ARC3-Inference"
            py_files = list(arc3_dir.glob("**/*.py"))
            self.assertGreaterEqual(len(py_files), 10)
            for pf in py_files:
                ast.parse(pf.read_text(encoding="utf-8"), filename=str(pf))
                py_compile.compile(str(pf), doraise=True)

    def test_04_zero_prompt_modification_across_all_levels_and_context_drains(self) -> None:
        """4. Verify _system_prompt and _build_user_prompt are 100% byte-identical to v24 across ALL levels and drains."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            v24_src = tmp_path / "v24_src"
            v28_src = tmp_path / "v28_src"
            v24_src.mkdir()
            v28_src.mkdir()

            c2 = _src(self.nb_v28["cells"][2]).split("\n", 1)[1] + "\n"
            patch_file = tmp_path / "cell2.patch"
            patch_file.write_text(c2, encoding="utf-8")

            for target_src in (v24_src, v28_src):
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

            assets = _extract_v28_assets(_src(self.nb_v28["cells"][4]))
            for name, text in assets.items():
                p = tmp_path / name
                p.write_text(text, encoding="utf-8")
                subprocess.run([sys.executable, str(p), str(v28_src / "ARC3-Inference")], check=True)

            ta_v24 = (v24_src / "ARC3-Inference/inference/agent/tool_agent.py").read_text(encoding="utf-8")
            ta_v28 = (v28_src / "ARC3-Inference/inference/agent/tool_agent.py").read_text(encoding="utf-8")
            self.assertEqual(ta_v24, ta_v28, "tool_agent.py in v28 must be 100% byte-identical to v24!")

    def test_05_startup_shield_periodic_warmup_keepalive(self) -> None:
        """5. Verify _HarnessGameSession._play_inner issues periodic warmup RESETs until /health returns 200 OK."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            src_path = tmp_path / "src"
            src_path.mkdir()
            c2 = _src(self.nb_v28["cells"][2]).split("\n", 1)[1] + "\n"
            patch_file = tmp_path / "cell2.patch"
            patch_file.write_text(c2, encoding="utf-8")
            subprocess.run(
                f"git archive HEAD ARC3-Inference | tar -x -C {src_path}",
                shell=True,
                cwd=str(DUCK_HARNESS_DIR),
                check=True,
            )
            subprocess.run(["git", "apply", "--include=ARC3-Inference/*", str(patch_file)], cwd=str(src_path), check=True)
            assets = _extract_v28_assets(_src(self.nb_v28["cells"][4]))
            for name, text in assets.items():
                p = tmp_path / name
                p.write_text(text, encoding="utf-8")
                subprocess.run([sys.executable, str(p), str(src_path / "ARC3-Inference")], check=True)

            check_code = r"""
import os, sys, time, types
from unittest.mock import patch

# Stub minimal taaf & arcengine so solver.py imports cleanly
taaf = types.ModuleType("taaf")
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

os.environ["ARC3_WARMUP_KEEPALIVE_SECONDS"] = "0.06"
os.environ["ARC3_HTTP_RETRY_INITIAL_SECONDS"] = "2.0"

health_calls = 0
class _Resp:
    status = 200
    def __enter__(self): return self
    def __exit__(self, *a): pass

def fake_urlopen(url, timeout=3):
    global health_calls
    health_calls += 1
    if health_calls < 4:
        raise ConnectionRefusedError("SGLang still loading")
    return _Resp()

class _DummySession:
    def __init__(self):
        from pathlib import Path
        self.game = types.SimpleNamespace(
            game_run=types.SimpleNamespace(game_id="ft09-test", state="playing", final_score=0.0, solver_note="done", total_tokens=0),
            current_state=types.SimpleNamespace(levels_completed=0),
        )
        self.solver = types.SimpleNamespace(_claim_warmup_slot=lambda: True)
        self.analyzer = types.SimpleNamespace(total_tokens=0)
        self.stop_event = types.SimpleNamespace(is_set=lambda: False)
        self.state_path = Path("/tmp/nonexistent_dummy_state_v28.json")
        self.token_baseline = 0
        self.game_token_baseline = 0
        self.started_at = time.monotonic()
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

sess = _DummySession()
initial_started_at = sess.started_at
with patch("urllib.request.urlopen", side_effect=fake_urlopen):
    solver_mod._HarnessGameSession._play_inner(sess)

assert sess.resets >= 2, f"Expected initial warmup RESET + at least 1 periodic keepalive RESET, got {sess.resets}"
assert health_calls == 4, f"Expected 4 health checks before 200 OK break, got {health_calls}"
assert getattr(sess.solver, "_analyzer_ready", False) is True, "Expected solver._analyzer_ready to be set True"
assert sess.started_at > initial_started_at, "Expected started_at to be reset after waiting for /health"

# Verify native request_timeout_seconds has zero 120s floor
mock_analyzer = types.SimpleNamespace(_timeout=120.0)
mock_solver = types.SimpleNamespace(max_runtime_s_per_game=300.0, soft_time_remaining_seconds=lambda: None)
class _TimeoutSession:
    def __init__(self, rem_s):
        self.analyzer = mock_analyzer
        self.solver = mock_solver
        self.rem_s = rem_s
    def timing_payload(self):
        return {"time_remaining_seconds": self.rem_s}

assert solver_mod._HarnessGameSession.request_timeout_seconds(_TimeoutSession(13.95)) == 13.95
assert solver_mod._HarnessGameSession.request_timeout_seconds(_TimeoutSession(0.0)) == 0.1
print("STARTUP_SHIELD_OK", sess.resets, health_calls)
"""
            env = dict(os.environ, PYTHONPATH=str(src_path / "ARC3-Inference"))
            res = subprocess.run([sys.executable, "-c", check_code], env=env, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"Startup shield test failed:\n{res.stderr}\n{res.stdout}")
            self.assertIn("STARTUP_SHIELD_OK", res.stdout)

    def test_06_animation_empty_spatial_helpers_and_m75_node_dict(self) -> None:
        """6. Verify patch_animation_empty, 8 spatial helpers, and M75 _NodeDict/_ObjDict zero-stdout key aliases."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            src_path = tmp_path / "src"
            src_path.mkdir()
            c2 = _src(self.nb_v28["cells"][2]).split("\n", 1)[1] + "\n"
            patch_file = tmp_path / "cell2.patch"
            patch_file.write_text(c2, encoding="utf-8")
            subprocess.run(
                f"git archive HEAD ARC3-Inference | tar -x -C {src_path}",
                shell=True,
                cwd=str(DUCK_HARNESS_DIR),
                check=True,
            )
            subprocess.run(["git", "apply", "--include=ARC3-Inference/*", str(patch_file)], cwd=str(src_path), check=True)
            assets = _extract_v28_assets(_src(self.nb_v28["cells"][4]))
            for name, text in assets.items():
                p = tmp_path / name
                p.write_text(text, encoding="utf-8")
                subprocess.run([sys.executable, str(p), str(src_path / "ARC3-Inference")], check=True)

            anim_src = (src_path / "ARC3-Inference/inference/utils/animation.py").read_text(encoding="utf-8")
            self.assertIn('"action": record.get("action_display") if record else None', anim_src)
            self.assertIn('"frames": 0', anim_src)
            self.assertIn('"steps": []', anim_src)

            check_code = r"""
import json
from inference.utils.segmentation import segment_layer, rot90, rot180, rot270, flipv, fliph, manhattan_distance, find_cells, find_objects

# 1. Verify M75 _NodeDict aliases without altering repr(), keys(), or json.dumps()
grid = [
    [0, 0, 0, 0],
    [0, 2, 2, 0],
    [0, 2, 2, 0],
    [0, 0, 0, 0],
]
seg = segment_layer(grid, "WwgGcBMPRbSYOrNp")
nodes = seg["nodes"]
red_node = [n for n in nodes if n["color"] == "g"][0]
expected_keys = ["id", "color", "hash", "rotation", "rotational_symmetry", "pose_hash", "shape_hash", "pixels", "boundary", "children"]
assert list(red_node.keys()) == expected_keys, f"Unexpected keys: {list(red_node.keys())}"
assert "bbox" not in json.dumps(red_node), "json.dumps must not include dynamic aliases!"
assert red_node["bbox"] == [1, 1, 2, 2]
assert red_node["box"] == [1, 1, 2, 2]
assert red_node["bounds"] == [1, 1, 2, 2]
assert red_node["size"] == 4
assert red_node["area"] == 4
assert red_node["centroid"] == (1, 1)
assert red_node["center"] == (1, 1)

# 3. Verify find_objects and _ObjDict aliases
objs = find_objects(grid, ignore_bg=0)
assert len(objs) == 1
obj = objs[0]
assert obj["area"] == 4
assert obj["size"] == 4
assert obj["pixels"] == 4
assert obj["bbox"] == [1, 1, 2, 2]
assert obj["box"] == [1, 1, 2, 2]
assert obj["center"] == obj["centroid"]

# 4. Verify spatial helpers
assert rot90([[1, 2], [3, 4]]) == [[3, 1], [4, 2]]
assert rot180([[1, 2], [3, 4]]) == [[4, 3], [2, 1]]
assert rot270([[1, 2], [3, 4]]) == [[2, 4], [1, 3]]
assert flipv([[1, 2], [3, 4]]) == [[3, 4], [1, 2]]
assert fliph([[1, 2], [3, 4]]) == [[2, 1], [4, 3]]
assert manhattan_distance((1, 2), (4, 6)) == 7
assert find_cells(grid, 2) == [(1, 1), (1, 2), (2, 1), (2, 2)]
assert find_cells(grid, "g") == [(1, 1), (1, 2), (2, 1), (2, 2)]
print("HELPERS_AND_M75_OK")
"""
            env = dict(os.environ, PYTHONPATH=str(src_path / "ARC3-Inference"))
            res = subprocess.run([sys.executable, "-c", check_code], env=env, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"Helpers test failed:\n{res.stderr}\n{res.stdout}")
            self.assertIn("HELPERS_AND_M75_OK", res.stdout)

    def test_07_d_prime_priority_math_with_final_level_b0_protection(self) -> None:
        """7. Verify D' unstarted pricing (18619 > 17527 > 17000) and final-level (remaining == 0) b0=6.0 cliff protection."""
        c16 = _src(self.nb_v28["cells"][16])
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

        # Unstarted games: 100% identical to v24
        p_n6 = fp_mod.d_priority(_Snap(1, 0, 0.0, 0.0, 6), tail_fraction=1.0)
        p_n8 = fp_mod.d_priority(_Snap(1, 0, 0.0, 0.0, 8), tail_fraction=1.0)
        p_n10 = fp_mod.d_priority(_Snap(1, 0, 0.0, 0.0, 10), tail_fraction=1.0)
        self.assertEqual(p_n6, 18619)
        self.assertEqual(p_n8, 17527)
        self.assertEqual(p_n10, 17000)

        # Final level (Level 6 of 6, remaining == 0) at phi=1.0:
        # Without b0=6.0, priority dropped to 9166; with b0=6.0, priority is 15166 (and >17000 with fast pace M>1.2)
        p_final_l6 = fp_mod.d_priority(_Snap(6, 0, 0.0, 0.0, 6), tail_fraction=1.0)
        self.assertEqual(p_final_l6, 15166)
        p_final_l6_fast = fp_mod.d_priority(_Snap(6, 0, 0.0, 18000.0, 6), tail_fraction=1.0)
        self.assertGreater(p_final_l6_fast, p_n10)


if __name__ == "__main__":
    unittest.main()

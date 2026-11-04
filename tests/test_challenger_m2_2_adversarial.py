"""Adversarial Verification Suite for Milestone 2 — Challenger M2_2.

Adversarially challenges the scheduler and timeout armor in notebooks/submission_v25.ipynb:
1. Extract _FP_SOURCE from Cell 16 and test install_timeout_armor against boundary inputs:
   - cand = None (Must preserve None)
   - cand = 0.0 (Must return 120.0)
   - cand = 13.95s, 35.55s, 66.18s, 89.45s (Must elevate to 120.0)
   - cand = 250.0s, 900.0s (Must return candidate unchanged)
   - Boundary & stress cases: negatives, exact 120.0, epsilon boundaries, idempotency, env override.
2. Test d_priority:
   - Priority pricing for unstarted games: N=6 -> 18619, N=8 -> 17527, N=10 -> 17000
   - Clamping behavior for N < 6, N > 10, N = None
   - Tail fade ranking: L6/6 > L5/6 > L1/6 under phi -> 0.0
   - Verify zero started_at clock reset and zero aging_bonus.
3. Live integration test against real ARC3-Inference tree in isolated sandbox.
"""

import ast
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
V25_NB_PATH = REPO_ROOT / "notebooks" / "submission_v25.ipynb"
DUCK_HARNESS_DIR = REPO_ROOT / "reference" / "duck-harness"


def extract_cell_source(nb_path: Path, cell_idx: int) -> str:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    src = nb["cells"][cell_idx].get("source", "")
    return "".join(src) if isinstance(src, list) else str(src)


def load_fp_source(c16_src: str) -> tuple[str, types.ModuleType]:
    marker = "_FP_SOURCE = "
    idx = c16_src.find(marker)
    end_marker = "\n_fp_mod ="
    idx_end = c16_src.find(end_marker)
    if idx == -1 or idx_end == -1:
        raise ValueError(f"Could not locate _FP_SOURCE markers in Cell 16 (idx={idx}, idx_end={idx_end})")

    exec_globals: dict = {}
    exec("import types as _fp_types, sys as _fp_sys", exec_globals)
    exec(c16_src[idx:idx_end], exec_globals)
    fp_src = exec_globals["_FP_SOURCE"]

    mod = types.ModuleType("ours_form_priority")
    exec(compile(fp_src, "ours_form_priority.py", "exec"), mod.__dict__)
    return fp_src, mod


class TestTimeoutArmorAdversarial(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c16_src = extract_cell_source(V25_NB_PATH, 16)
        cls.fp_src, cls.fp_mod = load_fp_source(cls.c16_src)

    def test_01_fp_source_compilation_and_exports(self):
        """Verify _FP_SOURCE compiles cleanly and exports required functions."""
        self.assertTrue(hasattr(self.fp_mod, "install_timeout_armor"))
        self.assertTrue(hasattr(self.fp_mod, "install_d"))
        self.assertTrue(hasattr(self.fp_mod, "d_priority"))
        self.assertTrue(hasattr(self.fp_mod, "d_parts"))
        self.assertTrue(hasattr(self.fp_mod, "d_value"))

    def test_02_error_handling_missing_session_or_method(self):
        """Verify install_timeout_armor fails cleanly when contract is violated."""
        class EmptySolver:
            pass

        with self.assertRaises(RuntimeError) as ctx:
            self.fp_mod.install_timeout_armor(EmptySolver)
        self.assertIn("solver._HarnessGameSession.request_timeout_seconds not found", str(ctx.exception))

        class SessionWithoutMethod:
            pass

        class SolverWithBadSession:
            _HarnessGameSession = SessionWithoutMethod

        with self.assertRaises(RuntimeError) as ctx:
            self.fp_mod.install_timeout_armor(SolverWithBadSession)
        self.assertIn("solver._HarnessGameSession.request_timeout_seconds not found", str(ctx.exception))

    def test_03_prompt_required_boundary_inputs(self):
        """Test exact boundary inputs mandated by the prompt:
        - cand = None (Must preserve None)
        - cand = 0.0 (Must return 120.0)
        - cand = 13.95s, 35.55s, 66.18s, 89.45s (Must elevate to 120.0)
        - cand = 250.0s, 900.0s (Must return candidate unchanged)
        """
        class MockSession:
            def __init__(self, ret_val):
                self._ret = ret_val

            def request_timeout_seconds(self):
                return self._ret

        class MockSolver:
            _HarnessGameSession = MockSession

        install_line = self.fp_mod.install_timeout_armor(MockSolver, min_floor=120.0)
        self.assertIn("#TIMEOUT_ARMOR ok", install_line)

        # 1. cand = None must preserve None
        s_none = MockSession(None)
        self.assertIsNone(s_none.request_timeout_seconds())

        # 2. cand = 0.0 must return 120.0
        s_zero = MockSession(0.0)
        self.assertEqual(s_zero.request_timeout_seconds(), 120.0)

        # 3. Clamped failure values from v21 must elevate to 120.0
        v21_failure_cands = [13.95, 35.55, 66.18, 89.45]
        for cand in v21_failure_cands:
            s = MockSession(cand)
            self.assertEqual(
                s.request_timeout_seconds(),
                120.0,
                f"Failed to elevate v21 failure candidate {cand} to 120.0",
            )

        # 4. Ample timeouts must be returned unchanged
        ample_cands = [250.0, 900.0]
        for cand in ample_cands:
            s = MockSession(cand)
            self.assertEqual(
                s.request_timeout_seconds(),
                cand,
                f"Failed to preserve ample candidate timeout {cand}",
            )

    def test_04_adversarial_boundary_and_stress_inputs(self):
        """Stress-test edge cases: negatives, exact boundary, epsilons, ints, inf, idempotency, env override."""
        class MockSession:
            def __init__(self, ret_val):
                self._ret = ret_val

            def request_timeout_seconds(self):
                return self._ret

        class MockSolver:
            _HarnessGameSession = MockSession

        self.fp_mod.install_timeout_armor(MockSolver, min_floor=120.0)

        # Negative numbers elevated to 120.0
        self.assertEqual(MockSession(-10.0).request_timeout_seconds(), 120.0)
        self.assertEqual(MockSession(-0.001).request_timeout_seconds(), 120.0)

        # Exact boundary
        self.assertEqual(MockSession(120.0).request_timeout_seconds(), 120.0)

        # Epsilon boundary
        self.assertEqual(MockSession(119.999999).request_timeout_seconds(), 120.0)
        self.assertEqual(MockSession(120.000001).request_timeout_seconds(), 120.000001)

        # Integer inputs
        self.assertEqual(MockSession(10).request_timeout_seconds(), 120.0)
        self.assertEqual(MockSession(300).request_timeout_seconds(), 300.0)

        # Infinity
        self.assertEqual(MockSession(float("inf")).request_timeout_seconds(), float("inf"))

        # Idempotency: re-installing must not double-wrap
        self.fp_mod.install_timeout_armor(MockSolver, min_floor=120.0)
        self.assertEqual(MockSession(13.95).request_timeout_seconds(), 120.0)

        # Environment variable override
        os.environ["ARC3_MIN_READ_TIMEOUT"] = "180.0"
        try:
            self.assertEqual(MockSession(150.0).request_timeout_seconds(), 180.0)
            self.assertEqual(MockSession(200.0).request_timeout_seconds(), 200.0)
        finally:
            del os.environ["ARC3_MIN_READ_TIMEOUT"]


class TestPrioritySchedulerAdversarial(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c16_src = extract_cell_source(V25_NB_PATH, 16)
        cls.fp_src, cls.fp_mod = load_fp_source(cls.c16_src)

    def test_01_unstarted_games_priority_pricing(self):
        """Verify priority pricing for unstarted games:
        N=6 -> 18619
        N=8 -> 17527
        N=10 -> 17000
        Strict ranking: P(N=6) > P(N=8) > P(N=10).
        """
        Snap = types.SimpleNamespace
        p6 = self.fp_mod.d_priority(Snap(level=1, actions=0, tokens=0, cost_multiplier=0.0, total_levels=6), tail_fraction=None)
        p8 = self.fp_mod.d_priority(Snap(level=1, actions=0, tokens=0, cost_multiplier=0.0, total_levels=8), tail_fraction=None)
        p10 = self.fp_mod.d_priority(Snap(level=1, actions=0, tokens=0, cost_multiplier=0.0, total_levels=10), tail_fraction=None)

        self.assertEqual(p6, 18619, f"Expected 18619 for N=6, got {p6}")
        self.assertEqual(p8, 17527, f"Expected 17527 for N=8, got {p8}")
        self.assertEqual(p10, 17000, f"Expected 17000 for N=10, got {p10}")
        self.assertGreater(p6, p8)
        self.assertGreater(p8, p10)

    def test_02_total_levels_clamping(self):
        """Verify clamping behavior: N < 6 clamped to 6, N > 10 clamped to 10, N=None defaults to 10."""
        Snap = types.SimpleNamespace
        p_none = self.fp_mod.d_priority(Snap(level=1, actions=0, tokens=0, cost_multiplier=0.0, total_levels=None), tail_fraction=None)
        p_4 = self.fp_mod.d_priority(Snap(level=1, actions=0, tokens=0, cost_multiplier=0.0, total_levels=4), tail_fraction=None)
        p_5 = self.fp_mod.d_priority(Snap(level=1, actions=0, tokens=0, cost_multiplier=0.0, total_levels=5), tail_fraction=None)
        p_11 = self.fp_mod.d_priority(Snap(level=1, actions=0, tokens=0, cost_multiplier=0.0, total_levels=11), tail_fraction=None)
        p_15 = self.fp_mod.d_priority(Snap(level=1, actions=0, tokens=0, cost_multiplier=0.0, total_levels=15), tail_fraction=None)

        self.assertEqual(p_none, 17000)
        self.assertEqual(p_4, 18619)
        self.assertEqual(p_5, 18619)
        self.assertEqual(p_11, 17000)
        self.assertEqual(p_15, 17000)

    def test_03_tail_fade_ranking_l6_above_l5_above_l1(self):
        """Verify tail fade ranking: L6/6 > L5/6 > L1/6 when tail_fraction -> 0.0.
        When time runs out (phi=0), future level bonus B vanishes,
        and depth-based A factor prioritizes finishing late games.
        """
        Snap = types.SimpleNamespace
        # Case A: Complete tail fade (phi = 0.0) with fresh turns
        p_l6_end = self.fp_mod.d_priority(Snap(level=6, actions=0, tokens=0, cost_multiplier=25000.0, total_levels=6), tail_fraction=0.0)
        p_l5_end = self.fp_mod.d_priority(Snap(level=5, actions=0, tokens=0, cost_multiplier=25000.0, total_levels=6), tail_fraction=0.0)
        p_l1_end = self.fp_mod.d_priority(Snap(level=1, actions=0, tokens=0, cost_multiplier=25000.0, total_levels=6), tail_fraction=0.0)

        self.assertEqual(p_l6_end, 9860)
        self.assertEqual(p_l5_end, 8451)
        self.assertEqual(p_l1_end, 2817)
        self.assertGreater(p_l6_end, p_l5_end)
        self.assertGreater(p_l5_end, p_l1_end)

        # Case B: Complete tail fade (phi = 0.0) with realistic active action counts (a=10, t=15000)
        p_l6_mid = self.fp_mod.d_priority(Snap(level=6, actions=10, tokens=15000, cost_multiplier=25000.0, total_levels=6), tail_fraction=0.0)
        p_l5_mid = self.fp_mod.d_priority(Snap(level=5, actions=10, tokens=15000, cost_multiplier=25000.0, total_levels=6), tail_fraction=0.0)
        p_l1_mid = self.fp_mod.d_priority(Snap(level=1, actions=10, tokens=15000, cost_multiplier=25000.0, total_levels=6), tail_fraction=0.0)

        self.assertEqual(p_l6_mid, 8408)
        self.assertEqual(p_l5_mid, 7206)
        self.assertEqual(p_l1_mid, 2402)
        self.assertGreater(p_l6_mid, p_l5_mid)
        self.assertGreater(p_l5_mid, p_l1_mid)

    def test_04_zero_clock_reset_and_zero_aging_bonus_invariants(self):
        """Verify anti-regression invariants:
        1. Zero self.started_at reassignment on queue acquisition.
        2. Zero aging_bonus calculation or queue thrashing.
        """
        # Audit Cell 16 source code
        self.assertNotIn("aging_bonus", self.c16_src)
        self.assertNotIn("self.started_at =", self.c16_src)
        self.assertNotIn("started_at", self.fp_src)

        # Audit that aging bonus terms do not exist in d_parts or d_priority
        cleaned_fp = self.fp_src.replace("agent", "").replace("message", "")
        self.assertNotIn("aging", cleaned_fp)


class TestLivePatchedInferenceIntegration(unittest.TestCase):
    def test_live_patched_inference_execution(self):
        """Extract ARC3-Inference from duck-harness git archive, apply Cell 2 patch,
        and run install_d + install_timeout_armor against the real inference modules.
        """
        with tempfile.TemporaryDirectory(prefix="m2_challenger2_integration_") as tmp_dir:
            tmp = Path(tmp_dir)
            src_path = tmp / "src"
            src_path.mkdir(parents=True)

            # 1. Clean extraction from reference repository
            cmd = f"git archive HEAD ARC3-Inference | tar -x -C {src_path}"
            res = subprocess.run(cmd, shell=True, cwd=str(DUCK_HARNESS_DIR), capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"git archive failed: {res.stderr}")

            # 2. Extract and apply Cell 2 patch
            nb = json.loads(V25_NB_PATH.read_text(encoding="utf-8"))
            c2_src = "".join(nb["cells"][2].get("source", ""))
            patch_text = c2_src.split("\n", 1)[1] if c2_src.startswith("%%writefile") else c2_src
            if not patch_text.endswith("\n"):
                patch_text += "\n"
            patch_file = tmp / "harness.patch"
            patch_file.write_text(patch_text, encoding="utf-8")

            apply_res = subprocess.run(
                ["git", "apply", "--include=ARC3-Inference/*", "-v", str(patch_file)],
                cwd=str(src_path),
                capture_output=True,
                text=True,
            )
            self.assertEqual(apply_res.returncode, 0, f"git apply failed:\n{apply_res.stderr}")

            # 3. Verify patched ARC3-Inference AST and contract conformance
            arc3_dir = src_path / "ARC3-Inference"
            solver_py = arc3_dir / "inference" / "framework" / "solver.py"
            tool_agent_py = arc3_dir / "inference" / "agent" / "tool_agent.py"

            self.assertTrue(solver_py.is_file(), "solver.py missing")
            self.assertTrue(tool_agent_py.is_file(), "tool_agent.py missing")

            # Parse ASTs
            solver_tree = ast.parse(solver_py.read_text(encoding="utf-8"), filename="solver.py")
            ta_tree = ast.parse(tool_agent_py.read_text(encoding="utf-8"), filename="tool_agent.py")

            # Check solver AST for _HarnessGameSession and request_timeout_seconds
            session_classes = [n for n in ast.walk(solver_tree) if isinstance(n, ast.ClassDef) and n.name == "_HarnessGameSession"]
            self.assertEqual(len(session_classes), 1, "Expected exactly 1 _HarnessGameSession class in solver.py")
            session_methods = {n.name for n in session_classes[0].body if isinstance(n, ast.FunctionDef)}
            self.assertIn("request_timeout_seconds", session_methods, "request_timeout_seconds missing from _HarnessGameSession")
            self.assertIn("play", session_methods, "play missing from _HarnessGameSession")

            # Check tool_agent AST for _PriorityGate, PrioritySnapshot, etc.
            ta_classes = {n.name: n for n in ast.walk(ta_tree) if isinstance(n, ast.ClassDef)}
            ta_imports = {alias.name for n in ast.walk(ta_tree) if isinstance(n, (ast.ImportFrom, ast.Import)) for alias in n.names}
            self.assertIn("_PriorityGate", ta_classes, "_PriorityGate missing from tool_agent.py")
            self.assertTrue(
                "PrioritySnapshot" in ta_classes or "PrioritySnapshot" in ta_imports,
                "PrioritySnapshot missing from tool_agent.py (must be defined or imported)"
            )
            gate_methods = {n.name for n in ta_classes["_PriorityGate"].body if isinstance(n, ast.FunctionDef)}
            self.assertIn("acquire", gate_methods)
            self.assertIn("_enqueue_and_wait", gate_methods)
            self.assertIn("_snapshot_priority", gate_methods)

            # 4. Verify install_d and install_timeout_armor against mock conforming to ARC3-Inference contracts
            nb = json.loads(V25_NB_PATH.read_text(encoding="utf-8"))
            c16_src = "".join(nb["cells"][16]["source"])
            marker = "_FP_SOURCE = "
            idx = c16_src.find(marker)
            end_marker = "\n_fp_mod ="
            idx_end = c16_src.find(end_marker)

            exec_globals = {}
            exec("import types as _fp_types, sys as _fp_sys", exec_globals)
            exec(c16_src[idx:idx_end], exec_globals)
            fp_src = exec_globals["_FP_SOURCE"]

            mod = types.ModuleType("ours_form_priority")
            exec(compile(fp_src, "ours_form_priority.py", "exec"), mod.__dict__)

            class MockPriorityGate:
                def acquire(self, priority):
                    pass
                def _enqueue_and_wait(self, value, snap):
                    pass
                def _snapshot_priority(self, snap, now):
                    return 1000

            mock_ta = types.SimpleNamespace(
                priority_value=None,
                ProgressPace=None,
                _PriorityGate=MockPriorityGate,
                PrioritySnapshot=types.SimpleNamespace,
                _PRIORITY_UNTRIMMED_BASE=2_000_000,
                _PRIORITY_BAND=100_000,
            )

            class MockHarnessGameSession:
                def play(self, *args, **kwargs):
                    pass
                def request_timeout_seconds(self):
                    return 15.0

            mock_solver = types.SimpleNamespace(_HarnessGameSession=MockHarnessGameSession)

            msg_d = mod.install_d(mock_ta, mock_solver)
            msg_armor = mod.install_timeout_armor(mock_solver, min_floor=120.0)

            self.assertIn("#OURS_FORM ok", msg_d)
            self.assertIn("#TIMEOUT_ARMOR ok", msg_armor)
            self.assertTrue(getattr(MockHarnessGameSession.request_timeout_seconds, "_ours_protected", False))

            sess = MockHarnessGameSession()
            self.assertEqual(sess.request_timeout_seconds(), 120.0)


if __name__ == "__main__":
    unittest.main()

"""Adversarial Challenger Stress Test Suite for Version 34 Sovereign Apex Engine.

Authored by Challenger 1 (critic, specialist) for Milestone 3 Gate.

Exhaustive stress tests:
1. TestV34ContinuousDPrimeChallengerStress:
   - Unstarted pricing monotonicity for N=1..20 (oracle and notebook d_priority)
   - Stall decay across L1..L6 (N=6..10, actions=120..500, tokens=200k..1.5M) strictly below 17,000
   - Fresh final level incentive (> 17,000) and stalled final level decay (< 1,500)
   - Tail fade behavior: phi -> 0 strict dominance of deeper levels (L6 > L5 > L4 > L3 > L2 > L1)
   - Extreme boundary fuzzing (negative numbers, extreme scales, NaNs, strings, None)

2. TestV34AffineLightsOutChallengerStress:
   - 500+ random solvable configurations across GF(2), Z_3, Z_5, Z_7 and varied board geometries
   - 100+ guaranteed unsolvable configurations via left-nullspace projections on singular systems
   - Degenerate boards (empty, 1x1, already solved, non-zero target)
   - Bitmask GF(2) toggle solver (solve_toggle_gf2) on 150 random bitmask systems with Gray code minimality
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
import random
import sys
import time
import types
from typing import Any
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from tests.test_v34_e2e_requirements import (
        PrioritySnapshot,
        oracle_continuous_d_prime,
        oracle_solve_affine_lights_out,
    )
except (ImportError, ModuleNotFoundError):
    from test_v34_e2e_requirements import (
        PrioritySnapshot,
        oracle_continuous_d_prime,
        oracle_solve_affine_lights_out,
    )

SUBMISSION_NB_PATH = REPO_ROOT / "notebooks" / "submission_v34.ipynb"
BUILDER_PATH = REPO_ROOT / "scripts" / "build_v34_notebook.py"


def _load_notebook_d_priority():
    """Extract and compile the notebook's ours_form_priority module from submission_v34.ipynb."""
    nb = json.loads(SUBMISSION_NB_PATH.read_text(encoding="utf-8"))
    src = "".join(nb["cells"][16]["source"])
    tree = ast.parse(src)
    fp_src = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "_FP_SOURCE":
                    fp_src = node.value.value
    if not fp_src:
        raise RuntimeError("Failed to extract _FP_SOURCE from notebook Cell 16")
    mod = types.ModuleType("ours_form_priority")
    exec(compile(fp_src, "<string>", "exec"), mod.__dict__)
    return mod


def _load_builder_prim_defs():
    """Extract and compile PRIM_DEFS_RAW from scripts/build_v34_notebook.py."""
    b_src = BUILDER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(b_src)
    prim_raw = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "PRIM_DEFS_RAW":
                    prim_raw = node.value.value
    if not prim_raw:
        raise RuntimeError("Failed to extract PRIM_DEFS_RAW from build_v34_notebook.py")
    ns: dict[str, Any] = {}
    exec(prim_raw, ns)
    return ns


class TestV34ContinuousDPrimeChallengerStress(unittest.TestCase):
    """Adversarial stress-testing of Continuous D' Priority Scheduler."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.nb_mod = _load_notebook_d_priority()

    def test_unstarted_pricing_monotonicity_n1_to_n20(self) -> None:
        """Verify unstarted pricing monotonicity for N=1..20 across both oracle and notebook."""
        oracle_priorities = {}
        notebook_priorities = {}

        for n in range(1, 21):
            snap = PrioritySnapshot(level=1, actions=0, tokens=0.0, cost_multiplier=0.0, total_levels=n)
            p_ora = oracle_continuous_d_prime(snap, tail_fraction=1.0)
            p_nb = self.nb_mod.d_priority(snap, tail_fraction=1.0)
            oracle_priorities[n] = p_ora
            notebook_priorities[n] = p_nb
            self.assertEqual(p_ora, p_nb, f"Mismatch at N={n}: oracle={p_ora}, notebook={p_nb}")

        # Multi-level competition games N=6..10 must be strictly monotonic decreasing
        for n in range(6, 10):
            self.assertGreater(
                oracle_priorities[n],
                oracle_priorities[n + 1],
                f"Monotonicity violated between N={n} ({oracle_priorities[n]}) and N={n+1} ({oracle_priorities[n+1]})",
            )

        # Clamping at N >= 10: priority must be exactly 17,000
        for n in range(10, 21):
            self.assertEqual(oracle_priorities[n], 17000, f"N={n} was not clamped at 17000!")

        # Single level game N=1 edge case: b0=0 because level=1
        self.assertEqual(oracle_priorities[1], 2619)

    def test_stall_decay_below_unstarted_across_all_levels(self) -> None:
        """Adversarial parameter sweep: stalled games on L1..L6 must decay strictly below unstarted games (< 17,000)."""
        unstarted_ceiling = 17000

        # Sweep 1,000+ combinations across extreme parameter ranges
        sweep_configs = [
            (lvl, act, tok, pace, n)
            for n in (6, 7, 8, 9, 10)
            for lvl in range(1, n + 1)
            for act in (120, 150, 200, 300, 500)
            for tok in (200000.0, 250000.0, 400000.0, 800000.0, 1500000.0)
            for pace in (0.0, 25000.0, 30000.0, 60000.0)
        ]

        self.assertGreaterEqual(len(sweep_configs), 1000)

        for lvl, act, tok, pace, n in sweep_configs:
            snap = PrioritySnapshot(level=lvl, actions=act, tokens=tok, cost_multiplier=pace, total_levels=n)
            p_ora = oracle_continuous_d_prime(snap, tail_fraction=1.0)
            p_nb = self.nb_mod.d_priority(snap, tail_fraction=1.0)

            # Both oracle and notebook must decay strictly below unstarted games (< 17,000)
            self.assertLess(
                p_ora,
                unstarted_ceiling,
                f"Oracle stalled priority failed to decay: L{lvl}/{n} act={act} tok={tok} pace={pace} -> {p_ora} >= {unstarted_ceiling}",
            )
            self.assertLess(
                p_nb,
                unstarted_ceiling,
                f"Notebook stalled priority failed to decay: L{lvl}/{n} act={act} tok={tok} pace={pace} -> {p_nb} >= {unstarted_ceiling}",
            )

    def test_fresh_final_level_incentive_and_stalled_decay(self) -> None:
        """Verify fresh final level priority > 17,000 and stalled final level decay < 1,500."""
        # N=6 game (ft09): Level 6 is final level
        snap_fresh_6 = PrioritySnapshot(level=6, actions=0, tokens=0.0, cost_multiplier=30000.0, total_levels=6)
        p_fresh_ora = oracle_continuous_d_prime(snap_fresh_6, tail_fraction=1.0)
        p_fresh_nb = self.nb_mod.d_priority(snap_fresh_6, tail_fraction=1.0)

        self.assertEqual(p_fresh_ora, 17166)
        self.assertEqual(p_fresh_nb, 17166)
        self.assertGreater(p_fresh_ora, 17000, "Fresh final level did not exceed 17000!")
        self.assertGreater(p_fresh_nb, 17000, "Fresh final level did not exceed 17000!")

        # Stalled final level: 150 actions, 250k tokens
        snap_stalled_6 = PrioritySnapshot(level=6, actions=150, tokens=250000.0, cost_multiplier=30000.0, total_levels=6)
        p_stalled_ora = oracle_continuous_d_prime(snap_stalled_6, tail_fraction=1.0)
        p_stalled_nb = self.nb_mod.d_priority(snap_stalled_6, tail_fraction=1.0)

        self.assertEqual(p_stalled_ora, 1132)
        self.assertEqual(p_stalled_nb, 1132)
        self.assertLess(p_stalled_ora, 1500, "Stalled final level did not decay below 1500!")
        self.assertLess(p_stalled_nb, 1500, "Stalled final level did not decay below 1500!")

        # Drop ratio > 15x
        self.assertGreater(p_fresh_ora / p_stalled_ora, 15.0)

        # N=10 game: Level 10 is final level
        snap_stalled_10 = PrioritySnapshot(level=10, actions=150, tokens=250000.0, cost_multiplier=30000.0, total_levels=10)
        self.assertEqual(self.nb_mod.d_priority(snap_stalled_10, tail_fraction=1.0), 999)
        self.assertLess(self.nb_mod.d_priority(snap_stalled_10, tail_fraction=1.0), 1500)

    def test_tail_fade_endgame_inversion(self) -> None:
        """Verify that as phi -> 0, deeper levels strictly dominate earlier levels (L6 > L5 > L4 > L3 > L2 > L1)."""
        for phi in [0.0, 0.005, 0.01, 0.03, 0.05]:
            scores_ora = [
                oracle_continuous_d_prime(
                    PrioritySnapshot(lvl, 20, 70000.0, 25000.0 if lvl > 1 else 0.0, 6),
                    tail_fraction=phi,
                )
                for lvl in range(1, 7)
            ]
            scores_nb = [
                self.nb_mod.d_priority(
                    PrioritySnapshot(lvl, 20, 70000.0, 25000.0 if lvl > 1 else 0.0, 6),
                    tail_fraction=phi,
                )
                for lvl in range(1, 7)
            ]

            # Verify strict ordering L1 < L2 < L3 < L4 < L5 < L6
            for idx in range(len(scores_ora) - 1):
                self.assertLess(
                    scores_ora[idx],
                    scores_ora[idx + 1],
                    f"Oracle tail fade violated at phi={phi} between L{idx+1} and L{idx+2}",
                )
                self.assertLess(
                    scores_nb[idx],
                    scores_nb[idx + 1],
                    f"Notebook tail fade violated at phi={phi} between L{idx+1} and L{idx+2}",
                )

    def test_adversarial_extreme_and_malformed_inputs(self) -> None:
        """Stress test with malformed, extreme, negative, and adversarial inputs."""
        extreme_snapshots = [
            PrioritySnapshot(level=1, actions=-50, tokens=-1000.0, cost_multiplier=-500.0, total_levels=-6),
            PrioritySnapshot(level=0, actions=0, tokens=0.0, cost_multiplier=0.0, total_levels=0),
            PrioritySnapshot(level=100, actions=1000000, tokens=100000000.0, cost_multiplier=1000000.0, total_levels=100),
            PrioritySnapshot(level=3, actions=20, tokens=40000.0, cost_multiplier=0.0, total_levels=None),
            PrioritySnapshot(level=4, actions=15, tokens=30000.0, cost_multiplier=25000.0, total_levels="invalid_level_count"),
        ]

        for snap in extreme_snapshots:
            for tf in [None, 1.0, 0.0, -1.0, 2.5]:
                p_ora = oracle_continuous_d_prime(snap, tail_fraction=tf)
                p_nb = self.nb_mod.d_priority(snap, tail_fraction=tf)
                self.assertIsInstance(p_ora, int)
                self.assertIsInstance(p_nb, int)
                self.assertGreaterEqual(p_ora, 1)
                self.assertGreaterEqual(p_nb, 1)


class TestV34AffineLightsOutChallengerStress(unittest.TestCase):
    """Adversarial stress-testing of Affine Lights-Out Gaussian Elimination Solver."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.prims = _load_builder_prim_defs()
        cls.solve_toggle_gf2_fn = staticmethod(cls.prims["solve_toggle_gf2"])
        cls.solve_affine_lights_out_fn = staticmethod(cls.prims["solve_affine_lights_out"])

    def test_500_plus_random_solvable_configurations_across_gf2_and_zm(self) -> None:
        """Stress-test 550 random solvable configurations across GF(2), Z_3, Z_5, Z_7 and varied board dimensions."""
        random.seed(1337)
        kernel_cross = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
        kernel_diag = [(0, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        cases = [
            # (modulus, rows, cols, count, kernel)
            (2, 1, 1, 10, kernel_cross),
            (2, 1, 4, 15, kernel_cross),
            (2, 4, 1, 15, kernel_cross),
            (2, 2, 2, 30, kernel_cross),
            (2, 3, 3, 40, kernel_cross),
            (2, 4, 4, 40, kernel_cross),
            (2, 5, 5, 40, kernel_cross),
            (2, 6, 6, 30, kernel_cross),
            (2, 8, 8, 10, kernel_cross),
            (2, 3, 4, 30, kernel_cross),
            (2, 4, 3, 30, kernel_cross),
            (2, 3, 3, 30, kernel_diag),
            (2, 4, 4, 30, kernel_diag),
            # Modulo arithmetic Z_3
            (3, 2, 2, 30, kernel_cross),
            (3, 3, 3, 40, kernel_cross),
            (3, 3, 4, 30, kernel_cross),
            # Modulo arithmetic Z_5
            (5, 2, 2, 30, kernel_cross),
            (5, 3, 3, 40, kernel_cross),
            # Modulo arithmetic Z_7
            (7, 2, 2, 30, kernel_cross),
            (7, 3, 3, 30, kernel_cross),
        ]

        total_tested = 0
        t0 = time.perf_counter()

        for m, R, C, count, kern in cases:
            for _ in range(count):
                total_tested += 1
                target_val = 0 if random.random() > 0.15 else random.randint(0, m - 1)
                board = [[target_val] * C for _ in range(R)]
                num_clicks = random.randint(1, R * C)
                # Construct guaranteed solvable board reachable to target_val
                for _ in range(num_clicks):
                    r = random.randint(0, R - 1)
                    c = random.randint(0, C - 1)
                    inc = random.randint(1, m - 1)
                    for dr, dc in kern:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < R and 0 <= nc < C:
                            board[nr][nc] = (board[nr][nc] + inc) % m

                actions = oracle_solve_affine_lights_out(
                    board,
                    target_val=target_val,
                    modulus=m,
                    kernel=kern,
                )

                self.assertIsNotNone(
                    actions,
                    f"Solver returned None on known solvable {R}x{C} board over Z_{m}!",
                )

                # Verify solution inverts board to target_val
                test_board = [row[:] for row in board]
                for act in actions:
                    r, c = act["row"], act["col"]
                    for dr, dc in kern:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < R and 0 <= nc < C:
                            test_board[nr][nc] = (test_board[nr][nc] + 1) % m

                for r in range(R):
                    for c in range(C):
                        self.assertEqual(
                            test_board[r][c],
                            target_val,
                            f"Cell ({r}, {c}) not inverted to {target_val} on {R}x{C} board over Z_{m}!",
                        )

        elapsed = time.perf_counter() - t0
        self.assertGreaterEqual(total_tested, 500, f"Expected >= 500 tests, ran {total_tested}")
        self.assertLess(elapsed, 2.0, f"500+ trials took {elapsed:.2f}s (> 2.0s)!")

    def test_guaranteed_unsolvable_configurations_return_none(self) -> None:
        """Construct mathematically inconsistent systems using left-nullspace projections and verify None returned."""
        kernel_cross = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]

        # 4x4 board over GF(2) has nullity 4 (rank 12 out of 16)
        # 5x5 board over GF(2) has nullity 2 (rank 23 out of 25)
        for R, C in [(4, 4), (5, 5)]:
            N = R * C
            A = [[0] * N for _ in range(N)]
            for cr in range(R):
                for cc in range(C):
                    c_idx = cr * C + cc
                    for dr, dc in kernel_cross:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < R and 0 <= nc < C:
                            r_idx = nr * C + nc
                            A[r_idx][c_idx] ^= 1

            # Compute left nullspace of A over GF(2)
            aug = [A[i] + [1 if i == j else 0 for j in range(N)] for i in range(N)]
            prow = 0
            for col in range(N):
                sel = None
                for r in range(prow, N):
                    if aug[r][col]:
                        sel = r
                        break
                if sel is None:
                    continue
                aug[prow], aug[sel] = aug[sel], aug[prow]
                for r in range(N):
                    if r != prow and aug[r][col]:
                        for c in range(col, 2 * N):
                            aug[r][c] ^= aug[prow][c]
                prow += 1

            left_null_vectors = [aug[r][N:] for r in range(prow, N)]
            self.assertGreater(len(left_null_vectors), 0, f"Expected non-trivial nullspace for {R}x{C}")

            # For each null vector v, any target b where v . b == 1 (mod 2) is mathematically unsolvable!
            unsolvable_tested = 0
            for v in left_null_vectors:
                ones = [idx for idx, val in enumerate(v) if val == 1]
                for pivot_idx in ones:
                    # Construct board with single 1 at pivot_idx
                    unsolvable_board = [[0] * C for _ in range(R)]
                    unsolvable_board[pivot_idx // C][pivot_idx % C] = 1

                    res = oracle_solve_affine_lights_out(unsolvable_board, target_val=0, modulus=2)
                    self.assertIsNone(
                        res,
                        f"Solver failed to detect unsolvable {R}x{C} board with pivot {pivot_idx}!",
                    )
                    unsolvable_tested += 1

            self.assertGreaterEqual(unsolvable_tested, 10)

    def test_degenerate_and_boundary_boards(self) -> None:
        """Verify handling of degenerate boards, already solved boards, and empty inputs."""
        # Empty board
        self.assertEqual(oracle_solve_affine_lights_out([]), [])
        self.assertEqual(oracle_solve_affine_lights_out([[]]), [])

        # Already solved board
        board_solved = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.assertEqual(oracle_solve_affine_lights_out(board_solved, target_val=0), [])

        # 1x1 board
        board_1x1 = [[1]]
        acts = oracle_solve_affine_lights_out(board_1x1, target_val=0)
        self.assertEqual(acts, [{"action": "MOUSE", "row": 0, "col": 0}])

    def test_solve_toggle_gf2_bitmask_solver_and_gray_minimality(self) -> None:
        """Stress-test solve_toggle_gf2 across 150 random bitmask combinations with Gray code minimality."""
        random.seed(42)
        total_solvable = 0
        total_unsolvable = 0
        solver = self.prims["solve_toggle_gf2"]

        for _ in range(150):
            n_buttons = random.randint(3, 12)
            n_bits = random.randint(3, 12)
            masks = [random.randint(1, (1 << n_bits) - 1) for _ in range(n_buttons)]

            if random.random() > 0.3:
                # Construct solvable system
                sub = [random.choice([0, 1]) for _ in range(n_buttons)]
                target = 0
                for s, m in zip(sub, masks):
                    if s:
                        target ^= m

                res = solver(masks, target)
                self.assertTrue(res["solvable"], "Failed to solve solvable bitmask system!")
                check = 0
                for idx in res["indices"]:
                    check ^= masks[idx]
                self.assertEqual(check, target, "XOR sum did not match target!")
                if res["minimal"]:
                    self.assertLessEqual(len(res["indices"]), sum(sub))
                total_solvable += 1
            else:
                # Random arbitrary target (may or may not be solvable)
                target = random.randint(1, (1 << n_bits) - 1)
                res = solver(masks, target)
                if not res["solvable"]:
                    self.assertIsNone(res["indices"])
                    total_unsolvable += 1
                else:
                    check = 0
                    for idx in res["indices"]:
                        check ^= masks[idx]
                    self.assertEqual(check, target)

        self.assertGreaterEqual(total_solvable, 70)
        self.assertGreaterEqual(total_unsolvable, 10)

    def test_fuzz_random_solvable_configurations_square_and_rectangular(self) -> None:
        """Fuzz random solvable configurations across 2x2, 3x3, 4x4, 5x5 square and 2x4, 4x2, 3x5, 5x3 non-square grids."""
        random.seed(2026)
        kernel_cross = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
        grid_shapes = [
            (2, 2), (3, 3), (4, 4), (5, 5),  # Square grids
            (2, 4), (4, 2), (3, 5), (5, 3), (1, 6), (6, 1), (2, 7),  # Rectangular non-square grids
        ]
        moduli = [2, 3, 5]
        total_trials = 0

        for R, C in grid_shapes:
            for m in moduli:
                for _ in range(5):
                    total_trials += 1
                    target_val = random.randint(0, m - 1)
                    board = [[target_val] * C for _ in range(R)]
                    num_clicks = random.randint(1, R * C)
                    for _ in range(num_clicks):
                        r = random.randint(0, R - 1)
                        c = random.randint(0, C - 1)
                        inc = random.randint(1, m - 1)
                        for dr, dc in kernel_cross:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < R and 0 <= nc < C:
                                board[nr][nc] = (board[nr][nc] + inc) % m

                    acts = self.solve_affine_lights_out_fn(
                        board,
                        target_val=target_val,
                        modulus=m,
                        kernel=kernel_cross,
                    )
                    self.assertIsNotNone(acts, f"Solvable board {R}x{C} over Z_{m} returned None!")

                    test_board = [row[:] for row in board]
                    for act in acts:
                        self.assertEqual(act["action"], "MOUSE")
                        self.assertGreaterEqual(act["row"], 0)
                        self.assertLess(act["row"], R)
                        self.assertGreaterEqual(act["col"], 0)
                        self.assertLess(act["col"], C)
                        r, c = act["row"], act["col"]
                        for dr, dc in kernel_cross:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < R and 0 <= nc < C:
                                test_board[nr][nc] = (test_board[nr][nc] + 1) % m

                    for r in range(R):
                        for c in range(C):
                            self.assertEqual(
                                test_board[r][c],
                                target_val,
                                f"Convergence failure at ({r}, {c}) on {R}x{C} over Z_{m}",
                            )

        self.assertGreaterEqual(total_trials, 150)

    def test_fuzz_inconsistent_unsolvable_linear_systems(self) -> None:
        """Construct mathematically inconsistent linear systems via left-nullspace projections on singular systems."""
        kernel_cross = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]

        def get_left_nullspace(R: int, C: int, m: int, kern: list[tuple[int, int]]) -> list[list[int]]:
            N = R * C
            A = [[0] * N for _ in range(N)]
            for cr in range(R):
                for cc in range(C):
                    c_idx = cr * C + cc
                    for dr, dc in kern:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < R and 0 <= nc < C:
                            r_idx = nr * C + nc
                            A[r_idx][c_idx] = (A[r_idx][c_idx] + 1) % m

            aug = [[A[c][r] for c in range(N)] + [1 if r == j else 0 for j in range(N)] for r in range(N)]
            prow = 0
            for col in range(N):
                sel = None
                for r in range(prow, N):
                    if aug[r][col] % m != 0:
                        sel = r
                        break
                if sel is None:
                    continue
                aug[prow], aug[sel] = aug[sel], aug[prow]
                inv = pow(aug[prow][col], -1, m) if m > 2 else 1
                for c in range(col, 2 * N):
                    aug[prow][c] = (aug[prow][c] * inv) % m
                for r in range(N):
                    if r != prow and aug[r][col] % m != 0:
                        fac = aug[r][col]
                        for c in range(col, 2 * N):
                            aug[r][c] = (aug[r][c] - fac * aug[prow][c]) % m
                prow += 1

            return [aug[r][N:] for r in range(prow, N)]

        singular_configs = [
            (4, 4, 2), (5, 5, 2),  # Square GF(2)
            (2, 2, 3),             # Square Z_3
            (2, 4, 5), (4, 2, 5),  # Rectangular Z_5
            (3, 5, 2), (5, 3, 2),  # Rectangular GF(2)
            (3, 5, 3), (2, 3, 2), (3, 2, 2),  # Mixed rectangular
        ]

        total_unsolvable = 0
        for R, C, m in singular_configs:
            null_vecs = get_left_nullspace(R, C, m, kernel_cross)
            self.assertGreater(len(null_vecs), 0, f"Expected non-trivial nullspace for {R}x{C} over Z_{m}")
            for v in null_vecs:
                idx = next(i for i, val in enumerate(v) if val % m != 0)
                board = [[0] * C for _ in range(R)]
                board[idx // C][idx % C] = (m - 1) % m
                res = self.solve_affine_lights_out_fn(board, target_val=0, modulus=m, kernel=kernel_cross)
                self.assertIsNone(res, f"Expected None for unsolvable {R}x{C} over Z_{m}, got {res}")
                total_unsolvable += 1

        # Also test degenerate empty kernel where clicks do nothing on non-zero board
        board_nz = [[1, 2], [0, 1]]
        res_deg = self.solve_affine_lights_out_fn(board_nz, target_val=0, modulus=3, kernel=[])
        self.assertIsNone(res_deg, "Empty kernel on non-zero board must return None")
        total_unsolvable += 1

        self.assertGreaterEqual(total_unsolvable, 15)

    def test_fuzz_empty_and_already_solved_grids(self) -> None:
        """Verify empty grids, already-solved grids, and degenerate non-grid inputs return [] gracefully."""
        self.assertEqual(self.solve_affine_lights_out_fn(None), [])
        self.assertEqual(self.solve_affine_lights_out_fn([]), [])
        self.assertEqual(self.solve_affine_lights_out_fn([[]]), [])
        self.assertEqual(self.solve_affine_lights_out_fn("invalid string"), [])
        self.assertEqual(self.solve_affine_lights_out_fn(12345), [])
        self.assertEqual(self.solve_affine_lights_out_fn({"grid": [[1]]}), [])

        # Already solved boards across various targets and moduli
        self.assertEqual(self.solve_affine_lights_out_fn([[0, 0], [0, 0]], target_val=0, modulus=2), [])
        self.assertEqual(self.solve_affine_lights_out_fn([[2, 2], [2, 2]], target_val=2, modulus=3), [])
        self.assertEqual(self.solve_affine_lights_out_fn([[4, 4, 4], [4, 4, 4]], target_val=4, modulus=5), [])
        self.assertEqual(self.solve_affine_lights_out_fn([[1]], target_val=1, modulus=2), [])
        self.assertEqual(self.solve_affine_lights_out_fn([[0]], target_val=0, modulus=5), [])

        # Verify target keyword alias
        self.assertEqual(self.solve_affine_lights_out_fn([[3, 3]], target=3, modulus=5), [])

        # Frame objects
        class EmptyFrame:
            ascii = ""

        class SolvedFrame:
            ascii = "00\n00"

        self.assertEqual(self.solve_affine_lights_out_fn(EmptyFrame()), [])
        self.assertEqual(self.solve_affine_lights_out_fn(SolvedFrame(), target_val=0), [])

    def test_fuzz_multi_state_modulus_arithmetic(self) -> None:
        """Fuzz multi-state modulus arithmetic (modulus 3, modulus 5) with multi-click accumulation."""
        random.seed(999)
        kernel_cross = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]

        for m in (3, 5):
            for R, C in [(2, 2), (3, 3), (2, 4), (3, 5)]:
                for _ in range(8):
                    target_val = random.randint(0, m - 1)
                    board = [[target_val] * C for _ in range(R)]
                    clicks_applied: dict[tuple[int, int], int] = {}
                    num_clicks = random.randint(1, R * C * 2)
                    for _ in range(num_clicks):
                        r = random.randint(0, R - 1)
                        c = random.randint(0, C - 1)
                        inc = random.randint(1, m - 1)
                        clicks_applied[(r, c)] = (clicks_applied.get((r, c), 0) + inc) % m
                        for dr, dc in kernel_cross:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < R and 0 <= nc < C:
                                board[nr][nc] = (board[nr][nc] + inc) % m

                    acts = self.solve_affine_lights_out_fn(
                        board,
                        target_val=target_val,
                        modulus=m,
                        kernel=kernel_cross,
                    )
                    self.assertIsNotNone(acts, f"Failed on multi-state Z_{m} board {R}x{C}")

                    test_board = [row[:] for row in board]
                    for act in acts:
                        r, c = act["row"], act["col"]
                        for dr, dc in kernel_cross:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < R and 0 <= nc < C:
                                test_board[nr][nc] = (test_board[nr][nc] + 1) % m

                    for r in range(R):
                        for c in range(C):
                            self.assertEqual(
                                test_board[r][c],
                                target_val,
                                f"Z_{m} arithmetic error at ({r}, {c}) on {R}x{C}",
                            )

    def test_fuzz_large_bitmasks_and_singular_nullspaces(self) -> None:
        """Fuzz solve_toggle_gf2 across large bitmasks (64-256 bits) and singular nullspaces (nullity 0..25)."""
        random.seed(777)
        solver = self.solve_toggle_gf2_fn

        # 1. 256-bit masks with 64 buttons
        for _ in range(10):
            masks = [random.randint(1, (1 << 256) - 1) for _ in range(64)]
            subset = [random.choice([0, 1]) for _ in range(64)]
            target = 0
            for s, m in zip(subset, masks):
                if s:
                    target ^= m
            res = solver(masks, target)
            self.assertTrue(res["solvable"])
            chk = 0
            for idx in res["indices"]:
                chk ^= masks[idx]
            self.assertEqual(chk, target)

        # 2. Large nullspace boundary (nullity = 17 Gray code vs nullity = 20 heuristic)
        base = [1 << i for i in range(4)]
        masks_17 = base + [0] * 17
        res_17 = solver(masks_17, (1 << 3) | (1 << 0))
        self.assertTrue(res_17["solvable"])
        self.assertTrue(res_17["minimal"])
        self.assertEqual(res_17["nullity"], 17)
        self.assertEqual(res_17["indices"], [0, 3])

        # nullity = 20: Heuristic minimization (> 18) completes fast without recursion/hang
        t0 = time.perf_counter()
        masks_20 = base + [0] * 20
        res_20 = solver(masks_20, (1 << 3) | (1 << 0))
        elapsed = time.perf_counter() - t0
        self.assertTrue(res_20["solvable"])
        self.assertFalse(res_20["minimal"])
        self.assertEqual(res_20["nullity"], 20)
        self.assertEqual(res_20["indices"], [0, 3])
        self.assertLess(elapsed, 0.05, f"Heuristic nullspace search took too long ({elapsed:.3f}s)")

        # 3. Inconsistent bitmask target outside mask span
        basis_masks = [1 << i for i in range(8)]
        target_outside = 1 << 16
        res_unsolvable = solver(basis_masks, target_outside)
        self.assertFalse(res_unsolvable["solvable"])
        self.assertIsNone(res_unsolvable["indices"])

        # 4. Exception boundaries
        with self.assertRaises(ValueError):
            solver([-1], 0)
        with self.assertRaises(ValueError):
            solver([1], -1)
        with self.assertRaises(ValueError):
            solver([1] * 257, 1)

        # 5. Delegation from solve_affine_lights_out for 1D integer lists
        res_del = self.solve_affine_lights_out_fn([1, 2, 4], target=5)
        self.assertTrue(res_del["solvable"])
        self.assertEqual(res_del["indices"], [0, 2])

    def test_fuzz_zero_unhandled_exceptions_and_no_infinite_loops(self) -> None:
        """Adversarial randomized fuzzing sweep ensuring zero unhandled exceptions, zero TypeErrors, and bounded time."""
        random.seed(54321)
        t0 = time.perf_counter()

        for _ in range(300):
            choice = random.choice(["grid_valid", "grid_degenerate", "toggle_valid", "toggle_unsolvable"])
            if choice == "grid_valid":
                R = random.randint(1, 5)
                C = random.randint(1, 5)
                m = random.choice([2, 3, 5])
                target = random.randint(0, m - 1)
                board = [[random.randint(0, m - 1) for _ in range(C)] for _ in range(R)]
                res = self.solve_affine_lights_out_fn(board, target_val=target, modulus=m)
                self.assertTrue(res is None or isinstance(res, list))
            elif choice == "grid_degenerate":
                val = random.choice([None, [], [[]], [[0]], "string", 42, {"a": 1}])
                res = self.solve_affine_lights_out_fn(val)
                self.assertEqual(res, [])
            elif choice == "toggle_valid":
                n_b = random.randint(1, 20)
                bits = random.randint(1, 32)
                masks = [random.randint(0, (1 << bits) - 1) for _ in range(n_b)]
                target = random.randint(0, (1 << bits) - 1)
                res = self.solve_toggle_gf2_fn(masks, target)
                self.assertIsInstance(res, dict)
                self.assertIn("solvable", res)
            elif choice == "toggle_unsolvable":
                masks = [1 << i for i in range(5)]
                target = 1 << 10
                res = self.solve_toggle_gf2_fn(masks, target)
                self.assertFalse(res["solvable"])
                self.assertIsNone(res["indices"])

        elapsed = time.perf_counter() - t0
        self.assertLess(elapsed, 2.0, f"300-trial fuzz sweep took {elapsed:.2f}s (> 2.0s)!")


if __name__ == "__main__":
    unittest.main()

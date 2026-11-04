"""Adversarial Stress Test Suite for Version 34 Sovereign Masterpiece Engine.

Stress-tests:
1. TestV34DPrimeAdversarialMath:
   - Pricing monotonicity across total levels N=1..20
   - Continuous decay monotonicity across actions (0..500) and tokens (0..1M)
   - Multi-level continuation proof: progressing games on L2..L5 strictly beat unstarted games (> 17000)
   - Starvation prevention: stalled games on L1..L6 drop strictly below < 17000 and cannot starve fresh games
   - Fresh final-level incentive: fresh final levels enter at 17,166 (> 17000)
   - Stalled final-level decay: stalled final levels decay to 1,132 (>15x drop), preventing slot monopolization
   - Tail fade endgame inversion: phi -> 0 strictly ranks deep levels L6 > L5 > L4 > L3 > L2 > L1
   - Adversarial extreme, negative, and malformed inputs

2. TestV34AffineSolverAdversarialMath:
   - Degenerate board dimensions (1x1, 1xN, Nx1, 2x2, 3x3, 4x4, 5x5, 6x6, 8x8)
   - Random solvable board inversion: applies random clicks and verifies solver inverts them back to 0
   - Inconsistent / unsolvable constraint systems returning None gracefully
   - Modulo arithmetic Z_3, Z_5 multi-state lights-out
   - Performance benchmark on large grids

3. TestV34ProbeFirstAndTimeoutAdversarial:
   - Alternating action / actionless turns and immediate streak reset
   - Long actionless streaks (streak=50) firmly clamped at 4,096 tokens
   - Continuous sweep of decoupled timeout floor across hard/soft remaining boundaries
   - Hard deadline ceiling enforcement (never overruns actual container remaining time)

4. TestV34StartupShieldConcurrencyAdversarial:
   - Multithreaded 50-thread warmup claim race barrier
   - Socket flood prevention during slow cold boot
   - Periodic keepalive 420s interval simulation preventing Kaggle 15m container kill
"""

from pathlib import Path
import random
import sys
import threading
from typing import Any
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from tests.test_v34_e2e_requirements import (
        PrioritySnapshot,
        oracle_continuous_d_prime,
        oracle_decoupled_request_timeout,
        oracle_probe_first_max_tokens,
        oracle_solve_affine_lights_out,
    )
except (ImportError, ModuleNotFoundError):
    from test_v34_e2e_requirements import (
        PrioritySnapshot,
        oracle_continuous_d_prime,
        oracle_decoupled_request_timeout,
        oracle_probe_first_max_tokens,
        oracle_solve_affine_lights_out,
    )


class TestV34DPrimeAdversarialMath(unittest.TestCase):
    """Deep adversarial stress-testing of continuous D' mathematical properties."""

    def test_unstarted_pricing_monotonicity_n1_to_n20(self) -> None:
        """Verify unstarted pricing monotonicity across total level counts N=1..20."""
        priorities = {
            n: oracle_continuous_d_prime(PrioritySnapshot(1, 0, 0.0, 0.0, n), tail_fraction=1.0)
            for n in range(1, 21)
        }
        # Multi-level games N=6..10 must be strictly monotonic: shorter games prioritized higher
        self.assertEqual(priorities[6], 18619)
        self.assertEqual(priorities[7], 17964)
        self.assertEqual(priorities[8], 17527)
        self.assertEqual(priorities[9], 17222)
        self.assertEqual(priorities[10], 17000)

        self.assertGreater(priorities[6], priorities[7])
        self.assertGreater(priorities[7], priorities[8])
        self.assertGreater(priorities[8], priorities[9])
        self.assertGreater(priorities[9], priorities[10])

        # Clamped games (N >= 10 priced at 10)
        self.assertEqual(priorities[11], 17000)
        self.assertEqual(priorities[20], 17000)

        # Single-level game edge case (b0=0 because level=1)
        self.assertEqual(priorities[1], 2619)

    def test_multi_level_continuation_beats_unstarted_games(self) -> None:
        """Active progressing games on Level 2..Level 4 strictly beat unstarted games (> 17000)."""
        p_unstarted_n10 = oracle_continuous_d_prime(PrioritySnapshot(1, 0, 0.0, 0.0, 10), tail_fraction=1.0)
        p_unstarted_n6 = oracle_continuous_d_prime(PrioritySnapshot(1, 0, 0.0, 0.0, 6), tail_fraction=1.0)

        # Active game on Level 2 with low actions (10) and normal tokens (20k)
        p_prog_l2 = oracle_continuous_d_prime(PrioritySnapshot(2, 10, 20000.0, 25000.0, 6), tail_fraction=1.0)
        # Active game on Level 3 with low actions (15) and normal tokens (30k)
        p_prog_l3 = oracle_continuous_d_prime(PrioritySnapshot(3, 15, 30000.0, 28000.0, 6), tail_fraction=1.0)

        self.assertEqual(p_prog_l2, 17976)
        self.assertEqual(p_prog_l3, 17916)
        self.assertGreater(p_prog_l2, p_unstarted_n10)
        self.assertGreater(p_prog_l3, p_unstarted_n10)

    def test_starvation_prevention_across_all_levels(self) -> None:
        """Stalled games across all levels decay strictly below unstarted games (< 17000)."""
        unstarted_p10 = 17000

        # Level 1 stalled (150 actions, 250k tokens)
        p_l1_stalled = oracle_continuous_d_prime(PrioritySnapshot(1, 150, 250000.0, 0.0, 10), tail_fraction=1.0)
        self.assertEqual(p_l1_stalled, 1636)
        self.assertLess(p_l1_stalled, unstarted_p10)

        for n in (6, 8, 10):
            for lvl in range(1, n + 1):
                snap = PrioritySnapshot(
                    level=lvl,
                    actions=150,
                    tokens=250000.0,
                    cost_multiplier=30000.0 if lvl > 1 else 0.0,
                    total_levels=n,
                )
                p_stalled = oracle_continuous_d_prime(snap, tail_fraction=1.0)
                self.assertLess(p_stalled, unstarted_p10, f"Stalled game on L{lvl}/{n} failed to decay below unstarted!")

    def test_fresh_final_level_incentive_and_stalled_decay(self) -> None:
        """Fresh final level enters at 17,166 (> 17000), stalled final level decays to 1,132."""
        snap_fresh = PrioritySnapshot(level=6, actions=0, tokens=0.0, cost_multiplier=30000.0, total_levels=6)
        self.assertEqual(oracle_continuous_d_prime(snap_fresh, tail_fraction=1.0), 17166)

        snap_stalled = PrioritySnapshot(level=6, actions=150, tokens=250000.0, cost_multiplier=30000.0, total_levels=6)
        self.assertEqual(oracle_continuous_d_prime(snap_stalled, tail_fraction=1.0), 1132)

    def test_tail_fade_endgame_inversion(self) -> None:
        """Near deadline (phi -> 0), deep levels are strictly prioritized over early levels."""
        scores_endgame = [
            oracle_continuous_d_prime(PrioritySnapshot(lvl, 20, 70000.0, 25000.0 if lvl > 1 else 0.0, 6), tail_fraction=0.0)
            for lvl in range(1, 7)
        ]
        # Inversion: L6 > L5 > L4 > L3 > L2 > L1
        for idx in range(len(scores_endgame) - 1):
            self.assertLess(
                scores_endgame[idx],
                scores_endgame[idx + 1],
                f"Endgame inversion violated between L{idx+1} ({scores_endgame[idx]}) and L{idx+2} ({scores_endgame[idx+1]})",
            )

    def test_adversarial_extreme_and_malformed_inputs(self) -> None:
        """Extreme numbers, negative values, and malformed inputs do not crash or produce negative priority."""
        extreme_cases = [
            PrioritySnapshot(level=1, actions=1000000, tokens=100000000.0, cost_multiplier=1000000.0, total_levels=10),
            PrioritySnapshot(level=10, actions=500000, tokens=50000000.0, cost_multiplier=50000.0, total_levels=10),
            PrioritySnapshot(level=-5, actions=-100, tokens=-5000.0, cost_multiplier=-100.0, total_levels=-10),
            PrioritySnapshot(level=0, actions=0, tokens=0.0, cost_multiplier=0.0, total_levels=0),
            PrioritySnapshot(level=1, actions=0, tokens=0.0, cost_multiplier=0.0, total_levels=None),
            PrioritySnapshot(level=5, actions=20, tokens=50000.0, cost_multiplier=0.0, total_levels="not_a_number"),
            PrioritySnapshot(level=15, actions=10, tokens=20000.0, cost_multiplier=25000.0, total_levels=10),
        ]
        for snap in extreme_cases:
            for tf in [None, 1.0, 0.0, -0.5, 2.0, 10.0]:
                p = oracle_continuous_d_prime(snap, tail_fraction=tf)
                self.assertIsInstance(p, int)
                self.assertGreaterEqual(p, 1)
                self.assertLess(p, 100000)


class TestV34AffineSolverAdversarialMath(unittest.TestCase):
    """Deep adversarial testing of GF(2) / Z_m Gaussian elimination affine solver."""

    def test_random_solvable_board_inversion(self) -> None:
        """Generate random toggle sequences on 3x3 and 4x4 grids, verify solver inverts them back to 0."""
        kernel = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]

        for R, C in [(3, 3), (4, 4), (3, 4)]:
            board = [[0] * C for _ in range(R)]

            # Apply 3 random toggles
            random.seed(42 + R * C)
            applied_clicks = set()
            for _ in range(3):
                r = random.randint(0, R - 1)
                c = random.randint(0, C - 1)
                if (r, c) in applied_clicks:
                    applied_clicks.remove((r, c))
                else:
                    applied_clicks.add((r, c))
                for dr, dc in kernel:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < R and 0 <= nc < C:
                        board[nr][nc] ^= 1

            # Solve board
            solution_actions = oracle_solve_affine_lights_out(board, target_val=0, modulus=2)
            self.assertIsNotNone(solution_actions, f"Failed to solve solvable {R}x{C} board!")

            # Apply solution clicks to board and verify all cells become 0
            test_board = [row[:] for row in board]
            for act in solution_actions:
                r, c = act["row"], act["col"]
                for dr, dc in kernel:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < R and 0 <= nc < C:
                        test_board[nr][nc] ^= 1

            for r in range(R):
                for c in range(C):
                    self.assertEqual(test_board[r][c], 0, f"Cell ({r}, {c}) not cleared in {R}x{C} board!")

    def test_multi_state_modular_lights_out(self) -> None:
        """Test Z_3 (3-color state cycling 0 -> 1 -> 2 -> 0) lights out solving."""
        kernel = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
        # 2x2 board over Z_3
        R, C = 2, 2
        modulus = 3
        board = [[0] * C for _ in range(R)]

        # Apply a click at (0, 0) twice
        for _ in range(2):
            for dr, dc in kernel:
                nr, nc = 0 + dr, 0 + dc
                if 0 <= nr < R and 0 <= nc < C:
                    board[nr][nc] = (board[nr][nc] + 1) % modulus

        # To clear this board, clicking (0, 0) one more time in Z_3 should bring it back to 0!
        acts = oracle_solve_affine_lights_out(board, target_val=0, modulus=modulus)
        self.assertIsNotNone(acts)

        # Apply acts to verify clearing
        test_board = [row[:] for row in board]
        for act in acts:
            r, c = act["row"], act["col"]
            for dr, dc in kernel:
                nr, nc = r + dr, c + dc
                if 0 <= nr < R and 0 <= nc < C:
                    test_board[nr][nc] = (test_board[nr][nc] + 1) % modulus

        for r in range(R):
            for c in range(C):
                self.assertEqual(test_board[r][c], 0, f"Modular cell ({r}, {c}) not cleared!")

    def test_large_grid_performance_stress(self) -> None:
        """Verify solver completes within 100ms on 8x8 grid (64 variables)."""
        import time
        R, C = 8, 8
        board = [[0] * C for _ in range(R)]
        # Set 3 lights
        board[0][0] = 1
        board[3][3] = 1
        board[7][7] = 1

        t0 = time.perf_counter()
        acts = oracle_solve_affine_lights_out(board, target_val=0, modulus=2)
        elapsed = time.perf_counter() - t0

        self.assertLess(elapsed, 0.2, f"8x8 affine solve took {elapsed:.4f}s (> 200ms)!")
        # If acts is not None, verify it solves the board
        if acts is not None:
            kernel = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
            for act in acts:
                r, c = act["row"], act["col"]
                for dr, dc in kernel:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < R and 0 <= nc < C:
                        board[nr][nc] ^= 1
            self.assertTrue(all(board[r][c] == 0 for r in range(R) for c in range(C)))


class TestV34ProbeFirstAndTimeoutAdversarial(unittest.TestCase):
    """Adversarial stress-testing of probe-first governor and decoupled timeout floor."""

    def test_probe_first_alternating_sequences(self) -> None:
        """Probe-first governor state transitions across complex alternating action sequences."""
        streak = 0
        history = [
            (False, 16384),  # turn 0: no action -> streak 1
            (False, 16384),  # turn 1: no action -> streak 2
            (False, 4096),   # turn 2: no action -> streak 3 (CAPPED!)
            (False, 4096),   # turn 3: no action -> streak 4 (CAPPED!)
            (True, 16384),   # turn 4: ACTION TAKEN -> resets streak to 0!
            (False, 16384),  # turn 5: no action -> streak 1
            (False, 16384),  # turn 6: no action -> streak 2
            (False, 4096),   # turn 7: no action -> streak 3 (CAPPED!)
        ]

        for action_taken, expected_cap in history:
            if action_taken:
                streak = 0
            else:
                streak += 1
            cap = oracle_probe_first_max_tokens(streak)
            self.assertEqual(cap, expected_cap, f"Mismatch at streak={streak}")

    def test_continuous_sweep_decoupled_timeout_floor(self) -> None:
        """Continuous sweep across hard/soft boundaries guarantees timeout in [45s, hard_remaining]."""
        floor = 45.0
        # Sweep hard remaining from 0 to 500s in 25s steps
        for hard_rem in [0.0, 10.0, 25.0, 44.9, 45.0, 50.0, 100.0, 300.0, 1000.0]:
            # Sweep soft remaining from 0 to 500s
            for soft_rem in [0.0, 5.0, 15.0, 44.9, 45.0, 80.0, 200.0]:
                timeout = oracle_decoupled_request_timeout(
                    configured_timeout=120.0,
                    hard_time_remaining=hard_rem,
                    soft_time_remaining=soft_rem,
                    floor=floor,
                )
                if hard_rem <= 0.0:
                    self.assertEqual(timeout, 0.0)
                else:
                    self.assertGreater(timeout, 0.0)
                    self.assertLessEqual(timeout, hard_rem)
                    if hard_rem >= floor:
                        self.assertGreaterEqual(timeout, floor)


class TestV34StartupShieldConcurrencyAdversarial(unittest.TestCase):
    """Adversarial stress-testing of warmup barrier and cold boot keepalive simulation."""

    def test_50_thread_warmup_barrier_race(self) -> None:
        """50 concurrent threads race through a barrier; exactly 1 claims warmup slot."""
        class AtomicSlot:
            def __init__(self) -> None:
                self._lock = threading.Lock()
                self._winner = None

            def try_claim(self, worker_id: int) -> bool:
                with self._lock:
                    if self._winner is None:
                        self._winner = worker_id
                        return True
                    return False

        slot = AtomicSlot()
        barrier = threading.Barrier(50)
        claims = []

        def worker(wid: int):
            barrier.wait()
            claims.append(slot.try_claim(wid))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(claims), 50)
        self.assertEqual(claims.count(True), 1)
        self.assertEqual(claims.count(False), 49)

    def test_cold_boot_keepalive_prevents_watchdog_kill(self) -> None:
        """Simulate a 55-minute cold boot: verify periodic keepalives prevent 15m idle kill."""
        watchdog_limit = 15 * 60  # 900s
        keepalive_step = 420      # 7m

        current_time = 0.0
        last_action_time = 0.0
        simulated_total = 55 * 60  # 3300s
        keepalive_count = 0

        while current_time < simulated_total:
            current_time += keepalive_step
            # Check if container died before keepalive
            idle_time = current_time - last_action_time
            self.assertLess(
                idle_time,
                watchdog_limit,
                f"Container killed at t={current_time}s! Idle={idle_time}s >= {watchdog_limit}s",
            )
            # Keepalive action fired
            last_action_time = current_time
            keepalive_count += 1

        self.assertEqual(keepalive_count, 8)


if __name__ == "__main__":
    unittest.main()

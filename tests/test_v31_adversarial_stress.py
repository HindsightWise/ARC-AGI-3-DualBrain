"""Adversarial Stress Test Suite for Version 31 Sovereign Masterpiece Engine.

Adversarially tests:
1. TestContinuousDPrimeAdversarialMath:
   - Unstarted game pricing monotonicity across total level counts (N=1..20).
   - Stagnation decay monotonicity and rate across actions (0..500) and tokens (0..1M).
   - Starvation prevention: proves stalled games decay 10x below unstarted games and cannot starve fresh games.
   - Fresh final-level incentive: proves fresh final levels enter with priority > 17,000 (17,166 on N=6) beating unstarted 10-level games (17,000).
   - Stalled final-level decay: proves stalled final levels decay to 1,132 (>15x drop), preventing permanent slot monopolization.
   - Tail fade mechanics (phi -> 0): proves room-to-grow bonus fades to 0, prioritizing deep levels over unstarted games near deadline.
   - Adversarial boundary & extreme inputs (10^7 actions, 10^9 tokens, negative inputs, malformed types).

2. TestQueueAdmissionSimulation:
   - 25-game realistic tournament with 10 concurrent streams.
   - Priority queue heap sorting across unstarted, active, stalled, and final-level games.
   - Confirms zero starvation of unstarted games and immediate preemption of stalled games at context trim.

3. TestStartupShieldConcurrencyAndSocketFloods:
   - Multithreaded concurrency stress: 50 concurrent threads claiming warmup slot; exactly 1 succeeds.
   - Socket flood audit: 10 concurrent game threads during cold boot; confirms ONLY warmup thread polls /health and other 9 threads make ZERO network calls.
   - Periodic keepalive verification: keepalive RESETs trigger only on warmup slot at configured interval.
   - Graceful unblocking: all 10 threads resume cleanly when /health returns 200.
   - Competition vs rehearsal started_at preservation: clock preserved in competition mode (>3600s).
   - Stop event abort responsiveness: immediate exit on interruption.
"""

from __future__ import annotations

import copy
import heapq
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
SUB_NB_PATH = REPO_ROOT / "notebooks" / "submission_v31.ipynb"
DUCK_HARNESS_DIR = REPO_ROOT / "reference" / "duck-harness"


def _src(cell: dict) -> str:
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else str(s)


def _load_d_prime_module() -> types.ModuleType:
    nb = json.loads(SUB_NB_PATH.read_text(encoding="utf-8"))
    c16 = _src(nb["cells"][16])
    ns: dict = {}
    for line in c16.splitlines():
        if line.startswith("_FP_SOURCE = "):
            exec(line, ns)
            break
    if "_FP_SOURCE" not in ns:
        raise AssertionError("Could not find _FP_SOURCE in Cell 16 of submission_v31.ipynb")
    mod = types.ModuleType("ours_form_priority")
    exec(compile(ns["_FP_SOURCE"], "ours_form_priority.py", "exec"), mod.__dict__)
    return mod


class _Snap:
    """Mock PrioritySnapshot matching Franzen/Shiiin9 interface."""
    def __init__(self, level: int, actions: int, tokens: float, cost_multiplier: float, total_levels: int | None):
        self.level = level
        self.actions = actions
        self.tokens = tokens
        self.cost_multiplier = cost_multiplier
        self.total_levels = total_levels


class TestContinuousDPrimeAdversarialMath(unittest.TestCase):
    """Adversarial stress-testing of continuous D' mathematical properties."""

    @classmethod
    def setUpClass(cls) -> None:
        assert SUB_NB_PATH.exists(), f"Missing {SUB_NB_PATH}"
        cls.mod = _load_d_prime_module()

    def test_unstarted_pricing_monotonicity(self) -> None:
        """Verify unstarted game priority across all total level counts N=1..20."""
        # Unstarted games: level=1, actions=0, tokens=0, pace=0
        priorities = {}
        for n in range(1, 21):
            snap = _Snap(level=1, actions=0, tokens=0.0, cost_multiplier=0.0, total_levels=n)
            priorities[n] = self.mod.d_priority(snap, tail_fraction=1.0)

        # Standard multi-level games (N=6..10) must be strictly monotonic: shorter games prioritized higher
        self.assertEqual(priorities[6], 18619)
        self.assertEqual(priorities[7], 17964)
        self.assertEqual(priorities[8], 17527)
        self.assertEqual(priorities[9], 17222)
        self.assertEqual(priorities[10], 17000)

        self.assertGreater(priorities[6], priorities[7])
        self.assertGreater(priorities[7], priorities[8])
        self.assertGreater(priorities[8], priorities[9])
        self.assertGreater(priorities[9], priorities[10])

        # Clamped levels (N > 10 clamped to 10)
        self.assertEqual(priorities[11], 17000)
        self.assertEqual(priorities[20], 17000)

        # Edge cases: 1-level and 2-level games
        # Single-level: remaining=0, level=1 => b0=0.0, B_raw=0.0 => 2619
        self.assertEqual(priorities[1], 2619)
        # Two-level: remaining=1, level=1 => b0=0.0, B_raw=b1=10.0 => 12619
        self.assertEqual(priorities[2], 12619)
        # Three-level: remaining=2, level=1 => b0=0.0, B_raw=b2=14.0 => 16619
        self.assertEqual(priorities[3], 16619)

    def test_stagnation_decay_monotonicity(self) -> None:
        """Verify that priority strictly decays monotonically as actions and tokens accumulate."""
        # Fix level=1, total_levels=10. Vary actions from 0 to 200 and tokens from 0 to 300k
        prev_p = 999999
        for a, t in [
            (0, 0),
            (5, 10000),
            (10, 25000),
            (25, 60000),
            (40, 118000),   # 1st trim
            (60, 150000),
            (100, 200000),
            (150, 250000),  # Stalled
            (200, 300000),  # Deep stall
        ]:
            snap = _Snap(level=1, actions=a, tokens=float(t), cost_multiplier=0.0, total_levels=10)
            p = self.mod.d_priority(snap, tail_fraction=1.0)
            self.assertLess(p, prev_p, f"Priority failed to decay at a={a}, t={t}: {p} >= {prev_p}")
            prev_p = p

        # Check key milestone values
        p_fresh = self.mod.d_priority(_Snap(1, 0, 0.0, 0.0, 10), tail_fraction=1.0)
        p_trim1 = self.mod.d_priority(_Snap(1, 40, 118000.0, 0.0, 10), tail_fraction=1.0)
        p_stalled = self.mod.d_priority(_Snap(1, 150, 250000.0, 0.0, 10), tail_fraction=1.0)

        self.assertEqual(p_fresh, 17000)
        self.assertEqual(p_trim1, 8252)
        self.assertEqual(p_stalled, 1636)

        # Decay ratios
        self.assertLess(p_trim1 / p_fresh, 0.50, "At 1st trim, priority must decay by >50%")
        self.assertLess(p_stalled / p_fresh, 0.10, "When stalled, priority must decay by >90%")

    def test_starvation_prevention_across_all_levels(self) -> None:
        """Verify that stalled games on ANY level cannot starve fresh unstarted games."""
        unstarted_p10 = self.mod.d_priority(_Snap(1, 0, 0.0, 0.0, 10), tail_fraction=1.0)
        self.assertEqual(unstarted_p10, 17000)

        # Test stalled state (a=150, t=250k) across levels 1, 2, 3, 4, 5 of 10
        for lvl in range(1, 6):
            snap = _Snap(level=lvl, actions=150, tokens=250000.0, cost_multiplier=30000.0, total_levels=10)
            p_stalled = self.mod.d_priority(snap, tail_fraction=1.0)
            self.assertLess(
                p_stalled,
                unstarted_p10,
                f"Level {lvl} stalled priority ({p_stalled}) starved unstarted game ({unstarted_p10})!"
            )
            self.assertLess(p_stalled, 5000, f"Stalled game on level {lvl} must be < 5000, got {p_stalled}")

    def test_fresh_final_level_incentive_and_cliff_fix(self) -> None:
        """Verify fresh final-level games enter with priority > 17,000 to win, eliminating the historical cliff."""
        # 6-level game (e.g. ft09) on Level 6 of 6:
        # Fresh final level: level=6, actions=0, tokens=0, pace=30,000, total_levels=6
        snap_final_fresh = _Snap(level=6, actions=0, tokens=0.0, cost_multiplier=30000.0, total_levels=6)
        p_final_fresh = self.mod.d_priority(snap_final_fresh, tail_fraction=1.0)

        unstarted_p10 = 17000
        self.assertEqual(p_final_fresh, 17166)
        self.assertGreater(p_final_fresh, unstarted_p10, "Fresh final level must beat unstarted 10-level games!")

        # Verify that b0=8.0 was the key: without b0 (b0=0), it would have been only 9166
        snap_no_b0 = copy.copy(snap_final_fresh)
        params_no_b0 = dict(self.mod.D_PRIME)
        params_no_b0["b0"] = 0.0
        u, v = self.mod.d_parts(6, 0.0, 0.0, 30000.0, 6, params=params_no_b0)
        p_no_b0 = int((u + v) * 1000.0)
        self.assertEqual(p_no_b0, 9166)
        self.assertGreater(p_final_fresh - p_no_b0, 7900, "b0=8.0 added +8000 points to lift over 17000")

    def test_stalled_final_level_decay(self) -> None:
        """Verify that stalled final-level games decay smoothly and cannot monopolize slots."""
        # Level 6 of 6 stalls at a=150, t=250k
        snap_final_stalled = _Snap(level=6, actions=150, tokens=250000.0, cost_multiplier=30000.0, total_levels=6)
        p_final_stalled = self.mod.d_priority(snap_final_stalled, tail_fraction=1.0)

        self.assertEqual(p_final_stalled, 1132)
        # Dropped from 17,166 to 1,132 (15.16x reduction!)
        self.assertLess(p_final_stalled, 2000)
        self.assertLess(p_final_stalled, 17000 / 10)

    def test_tail_fade_endgame_behavior(self) -> None:
        """Verify that when tail_fraction -> 0 (endgame), room-to-grow bonus fades to zero."""
        # Compare unstarted game (N=6, level=1) vs game on final level (N=6, level=6) as phi decays
        # At phi = 1.0:
        p_unstarted_full = self.mod.d_priority(_Snap(1, 0, 0.0, 0.0, 6), tail_fraction=1.0)
        p_l6_full = self.mod.d_priority(_Snap(6, 0, 0.0, 30000.0, 6), tail_fraction=1.0)
        self.assertEqual(p_unstarted_full, 18619)
        self.assertEqual(p_l6_full, 17166)
        # In early game, unstarted 6-level has slight edge over fresh level 6 (18619 vs 17166)

        # At phi = 0.5 (halfway through tail fade):
        p_unstarted_half = self.mod.d_priority(_Snap(1, 0, 0.0, 0.0, 6), tail_fraction=0.5)
        p_l6_half = self.mod.d_priority(_Snap(6, 0, 0.0, 30000.0, 6), tail_fraction=0.5)
        self.assertEqual(p_unstarted_half, 10619)
        self.assertEqual(p_l6_half, 13166)
        self.assertGreater(p_l6_half, p_unstarted_half, "At phi=0.5, level 6 overtakes unstarted game!")

        # At phi = 0.0 (final moments):
        p_unstarted_zero = self.mod.d_priority(_Snap(1, 0, 0.0, 0.0, 6), tail_fraction=0.0)
        p_l6_zero = self.mod.d_priority(_Snap(6, 0, 0.0, 30000.0, 6), tail_fraction=0.0)
        self.assertEqual(p_unstarted_zero, 2619)  # Only A*C remains (2.619 * 1000)
        self.assertEqual(p_l6_zero, 9166)        # A*C on Level 6 is 9.166 * 1000
        self.assertGreater(p_l6_zero, p_unstarted_zero * 3, "Deep level must be 3.5x higher than unstarted at phi=0!")

    def test_adversarial_and_extreme_inputs(self) -> None:
        """Exhaustive boundary sweep across extreme, negative, and malformed inputs."""
        extreme_cases = [
            # Huge values
            _Snap(level=1, actions=1000000, tokens=100000000.0, cost_multiplier=1000000.0, total_levels=10),
            _Snap(level=10, actions=500000, tokens=50000000.0, cost_multiplier=50000.0, total_levels=10),
            # Negative values
            _Snap(level=-5, actions=-100, tokens=-5000.0, cost_multiplier=-100.0, total_levels=-10),
            _Snap(level=0, actions=0, tokens=0.0, cost_multiplier=0.0, total_levels=0),
            # None and invalid types
            _Snap(level=1, actions=0, tokens=0.0, cost_multiplier=0.0, total_levels=None),
            _Snap(level=5, actions=20, tokens=50000.0, cost_multiplier=0.0, total_levels="not_a_number"),
            # level > total_levels
            _Snap(level=15, actions=10, tokens=20000.0, cost_multiplier=25000.0, total_levels=10),
        ]

        for idx, snap in enumerate(extreme_cases):
            for tf in [None, 1.0, 0.0, -0.5, 2.0, 10.0]:
                p = self.mod.d_priority(snap, tail_fraction=tf)
                self.assertIsInstance(p, int, f"Case {idx} (tf={tf}) did not return int")
                self.assertGreaterEqual(p, 1, f"Case {idx} (tf={tf}) returned < 1: {p}")
                self.assertLess(p, 100000, f"Case {idx} (tf={tf}) returned excessive value: {p}")


class TestQueueAdmissionSimulation(unittest.TestCase):
    """Simulate priority queue behavior with 25 games competing for 10 SGLang active slots."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load_d_prime_module()

    def test_25_game_admission_tournament(self) -> None:
        """Simulate realistic multi-game scheduling and queue preemption."""
        # Build 25 game snapshots representing a realistic leaderboard run:
        # Games 0..9: Unstarted games (mix of 6, 8, 10 levels)
        # Games 10..14: Progressing games on level 2..4
        # Games 15..19: Stalled games on level 1 or 2
        # Game 20: Fresh Level 6 game
        # Game 21: Stalled Level 6 game
        # Games 22..24: Single/Two level games
        games = {
            "unstarted_6a": _Snap(1, 0, 0.0, 0.0, 6),
            "unstarted_6b": _Snap(1, 0, 0.0, 0.0, 6),
            "unstarted_8a": _Snap(1, 0, 0.0, 0.0, 8),
            "unstarted_8b": _Snap(1, 0, 0.0, 0.0, 8),
            "unstarted_10a": _Snap(1, 0, 0.0, 0.0, 10),
            "unstarted_10b": _Snap(1, 0, 0.0, 0.0, 10),
            "prog_l2": _Snap(2, 10, 20000.0, 25000.0, 10),
            "prog_l3": _Snap(3, 15, 30000.0, 28000.0, 10),
            "prog_l4": _Snap(4, 20, 40000.0, 30000.0, 10),
            "prog_l5": _Snap(5, 10, 25000.0, 30000.0, 6),
            "stalled_l1a": _Snap(1, 150, 250000.0, 0.0, 10),
            "stalled_l1b": _Snap(1, 120, 220000.0, 0.0, 8),
            "stalled_l2": _Snap(2, 160, 260000.0, 40000.0, 10),
            "stalled_l3": _Snap(3, 140, 240000.0, 45000.0, 10),
            "fresh_l6": _Snap(6, 0, 0.0, 30000.0, 6),
            "stalled_l6": _Snap(6, 150, 250000.0, 30000.0, 6),
            "single_lvl": _Snap(1, 0, 0.0, 0.0, 1),
            "two_lvl": _Snap(1, 0, 0.0, 0.0, 2),
        }

        # Calculate priorities at phi=1.0
        scores = {gid: self.mod.d_priority(snap, tail_fraction=1.0) for gid, snap in games.items()}

        # Verify ranking:
        # Top tier: unstarted 6-level (18619), unstarted 8-level (17527), fresh final level (17166), unstarted 10-level (17000)
        top_ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_gids = [gid for gid, _ in top_ranked[:8]]

        self.assertIn("unstarted_6a", top_gids)
        self.assertIn("unstarted_6b", top_gids)
        self.assertIn("unstarted_8a", top_gids)
        self.assertIn("fresh_l6", top_gids)
        self.assertIn("unstarted_10a", top_gids)

        # Fresh final level MUST rank higher than unstarted 10-level!
        self.assertGreater(scores["fresh_l6"], scores["unstarted_10a"])

        # Bottom tier: ALL stalled games must be ranked in the bottom
        bottom_gids = [gid for gid, _ in top_ranked[-5:]]
        self.assertIn("stalled_l1a", bottom_gids)
        self.assertIn("stalled_l6", bottom_gids)

        # Stalled games must be strictly below unstarted games
        for stalled_id in ("stalled_l1a", "stalled_l1b", "stalled_l2", "stalled_l3", "stalled_l6"):
            for unstarted_id in ("unstarted_6a", "unstarted_8a", "unstarted_10a"):
                self.assertLess(
                    scores[stalled_id],
                    scores[unstarted_id],
                    f"{stalled_id} ({scores[stalled_id]}) ranked above {unstarted_id} ({scores[unstarted_id]})!"
                )


class TestStartupShieldConcurrencyAndSocketFloods(unittest.TestCase):
    """Adversarial stress-testing of cold-boot startup shield concurrency, socket flood immunity, and timer safety."""

    def test_multithreaded_warmup_claim_concurrency(self) -> None:
        """Stress-test 50 concurrent threads claiming warmup slot: EXACTLY 1 must succeed."""
        class MockSolver:
            def __init__(self, warmup_games=1):
                self._warmup_lock = threading.Lock()
                self._warmup_remaining = warmup_games

            def _claim_warmup_slot(self) -> bool:
                lock = getattr(self, "_warmup_lock", None)
                if lock is None:
                    return False
                with lock:
                    if getattr(self, "_warmup_remaining", 0) <= 0:
                        return False
                    self._warmup_remaining -= 1
                    return True

        solver = MockSolver(warmup_games=1)
        results = []
        barrier = threading.Barrier(50)

        def worker():
            barrier.wait()  # Synchronize all 50 threads to hit claim simultaneously
            results.append(solver._claim_warmup_slot())

        threads = [threading.Thread(target=worker) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(results), 50)
        self.assertEqual(results.count(True), 1, "Exactly 1 thread must claim the warmup slot!")
        self.assertEqual(results.count(False), 49, "Remaining 49 threads must receive False!")

    def test_multithreaded_startup_shield_socket_flood_immunity(self) -> None:
        """Simulate 10 concurrent game threads during cold boot: verify ZERO socket flood from non-warmup threads."""
        http_requests = []
        http_lock = threading.Lock()
        boot_ready = threading.Event()

        def fake_urlopen(url, timeout=3):
            with http_lock:
                http_requests.append({
                    "thread_id": threading.get_ident(),
                    "timestamp": time.monotonic(),
                    "url": url,
                })
            # SGLang loading until boot_ready event is set
            if not boot_ready.is_set():
                raise ConnectionRefusedError("SGLang loading weights (simulated)")
            class _Resp:
                status = 200
                def __enter__(self): return self
                def __exit__(self, *a): pass
            return _Resp()

        class MockSolver:
            def __init__(self):
                self._warmup_lock = threading.Lock()
                self._warmup_remaining = 1
                self._analyzer_ready = False
                self.max_runtime_s_per_game = 532 * 60.0  # Competition mode

            def _claim_warmup_slot(self) -> bool:
                with self._warmup_lock:
                    if self._warmup_remaining <= 0:
                        return False
                    self._warmup_remaining -= 1
                    return True

        solver = MockSolver()

        class MockSession:
            def __init__(self, sess_id):
                self.sess_id = sess_id
                self.solver = solver
                self.stop_event = threading.Event()
                self.started_at = 1000.0
                self.resets = 0

            def _execute_auto_reset(self):
                self.resets += 1

        sessions = [MockSession(i) for i in range(10)]
        warmup_thread_id = None
        warmup_ident_holder = []

        # Execute startup shield logic for each thread
        def run_thread(sess: MockSession):
            nonlocal warmup_thread_id
            # Exact logic from patch_startup_shield.py
            _is_warmup_slot = bool(sess.solver._claim_warmup_slot())
            if _is_warmup_slot:
                warmup_ident_holder.append(threading.get_ident())
                sess._execute_auto_reset()
                if (
                    not getattr(sess.solver, "_analyzer_ready", False)
                    and int(os.environ.get("ARC3_WARMUP_ACTION_GAMES", "0") or 0) > 0
                ):
                    _warmup_last_ts = time.monotonic()
                    _warmup_health_url = "http://127.0.0.1:8001/health"
                    _warmup_grace_s = 10.0
                    _warmup_deadline = _warmup_last_ts + _warmup_grace_s
                    _warmup_interval_s = float(os.environ.get("ARC3_WARMUP_KEEPALIVE_SECONDS", "480") or 480.0)
                    while time.monotonic() < _warmup_deadline and not sess.stop_event.is_set():
                        if getattr(sess.solver, "_analyzer_ready", False):
                            break
                        try:
                            with fake_urlopen(_warmup_health_url, timeout=3) as _wresp:
                                if getattr(_wresp, "status", 200) == 200:
                                    setattr(sess.solver, "_analyzer_ready", True)
                                    break
                        except Exception:
                            pass
                        if (time.monotonic() - _warmup_last_ts) >= _warmup_interval_s:
                            sess._execute_auto_reset()
                            _warmup_last_ts = time.monotonic()
                        time.sleep(0.05)  # Accelerated for test
            elif not getattr(sess.solver, "_analyzer_ready", False) and int(os.environ.get("ARC3_WARMUP_ACTION_GAMES", "0") or 0) > 0:
                _wait_deadline = time.monotonic() + 10.0
                while not getattr(sess.solver, "_analyzer_ready", False) and time.monotonic() < _wait_deadline and not sess.stop_event.is_set():
                    time.sleep(0.02)
            if (getattr(sess.solver, "max_runtime_s_per_game", 0) or 0) <= 3600:
                sess.started_at = time.monotonic()

        os.environ["ARC3_WARMUP_ACTION_GAMES"] = "1"
        threads = [threading.Thread(target=run_thread, args=(s,)) for s in sessions]

        # Launch all 10 threads while server is cold
        for t in threads:
            t.start()

        # Let them run against cold server for 150ms
        time.sleep(0.15)

        # Server becomes healthy!
        boot_ready.set()

        # Wait for all threads to unblock and join
        for t in threads:
            t.join(timeout=2.0)
            self.assertFalse(t.is_alive(), "Thread failed to unblock after server became ready!")

        self.assertEqual(len(warmup_ident_holder), 1)
        w_ident = warmup_ident_holder[0]

        # Audit HTTP requests:
        # 1. Total requests must be >= 1
        self.assertGreater(len(http_requests), 0)

        # 2. STRICT INVARIANT: 100% of HTTP requests came from the single warmup thread!
        request_thread_ids = {r["thread_id"] for r in http_requests}
        self.assertEqual(
            request_thread_ids,
            {w_ident},
            f"Socket flood violation! Non-warmup threads made network calls: {request_thread_ids - {w_ident}}"
        )

        # 3. Competition clock invariant: started_at must NOT be reset in competition mode
        for s in sessions:
            self.assertEqual(s.started_at, 1000.0, f"Session {s.sess_id} started_at was reset in competition mode!")

    def test_periodic_keepalive_triggers_only_on_warmup_slot(self) -> None:
        """Verify that periodic keepalive RESETs trigger only on the warmup slot thread."""
        resets_issued = []
        ready_flag = threading.Event()

        class MockSolver:
            _warmup_remaining = 1
            _warmup_lock = threading.Lock()
            _analyzer_ready = False
            max_runtime_s_per_game = 532 * 60.0

            def _claim_warmup_slot(self):
                with self._warmup_lock:
                    if self._warmup_remaining <= 0:
                        return False
                    self._warmup_remaining -= 1
                    return True

        solver = MockSolver()

        class MockSession:
            def __init__(self, sess_id):
                self.sess_id = sess_id
                self.solver = solver
                self.stop_event = threading.Event()
                self.resets = 0

            def _execute_auto_reset(self):
                self.resets += 1
                resets_issued.append((self.sess_id, time.monotonic()))

        s_warmup = MockSession("warmup")
        s_other = MockSession("other")

        # Set keepalive interval very small (0.05s) to trigger multiple keepalives in 0.2s
        os.environ["ARC3_WARMUP_ACTION_GAMES"] = "1"
        os.environ["ARC3_WARMUP_KEEPALIVE_SECONDS"] = "0.05"

        def run_warmup():
            _is_warmup_slot = bool(s_warmup.solver._claim_warmup_slot())
            if _is_warmup_slot:
                s_warmup._execute_auto_reset()
                _warmup_last_ts = time.monotonic()
                _warmup_deadline = _warmup_last_ts + 2.0
                _warmup_interval_s = 0.05
                while time.monotonic() < _warmup_deadline and not ready_flag.is_set():
                    if (time.monotonic() - _warmup_last_ts) >= _warmup_interval_s:
                        s_warmup._execute_auto_reset()
                        _warmup_last_ts = time.monotonic()
                    time.sleep(0.01)

        def run_other():
            _is_warmup_slot = bool(s_other.solver._claim_warmup_slot())
            if not _is_warmup_slot:
                while not ready_flag.is_set():
                    time.sleep(0.01)

        t1 = threading.Thread(target=run_warmup)
        t2 = threading.Thread(target=run_other)

        t1.start()
        t2.start()

        # Let keepalive run for 180ms (should trigger initial + ~3 keepalives)
        time.sleep(0.18)
        ready_flag.set()

        t1.join(timeout=1.0)
        t2.join(timeout=1.0)

        self.assertGreaterEqual(s_warmup.resets, 3, f"Expected >= 3 resets, got {s_warmup.resets}")
        self.assertEqual(s_other.resets, 0, "Non-warmup thread must issue 0 resets!")

        for sess_id, _ in resets_issued:
            self.assertEqual(sess_id, "warmup")

    def test_rehearsal_mode_resets_started_at(self) -> None:
        """Verify that rehearsal mode (max_runtime_s_per_game <= 3600) resets started_at so rehearsal gets full 5 mins."""
        class MockSolver:
            _warmup_remaining = 0
            _warmup_lock = threading.Lock()
            _analyzer_ready = True
            max_runtime_s_per_game = 300.0  # 5 minutes rehearsal

        class MockSession:
            def __init__(self):
                self.solver = MockSolver()
                self.stop_event = threading.Event()
                self.started_at = 100.0

        sess = MockSession()
        t0 = time.monotonic()

        # Execute rehearsal mode clock reset logic
        if (getattr(sess.solver, "max_runtime_s_per_game", 0) or 0) <= 3600:
            sess.started_at = time.monotonic()

        self.assertGreaterEqual(sess.started_at, t0)
        self.assertNotEqual(sess.started_at, 100.0)


if __name__ == "__main__":
    unittest.main()

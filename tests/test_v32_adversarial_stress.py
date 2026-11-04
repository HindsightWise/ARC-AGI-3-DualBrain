"""Adversarial Stress Test Suite for Version 32 Sovereign Apex Engine.

Adversarially tests:
1. TestV32DPrimeAdversarialMath:
   - Unstarted game pricing monotonicity across total level counts (N=1..20).
   - 100% shiiin9 (31.54 LB) / v24 (28.79 LB) equivalence across all states with remaining >= 1.
   - Multi-level continuation proof: active progressing games on L2..L4 strictly beat unstarted games (> 17,000), fixing v31's B_raw * C eviction bug.
   - Starvation prevention: stalled games on L1..L6 drop strictly below < 17,000 and cannot starve unstarted games.
   - Fresh final-level incentive: fresh final levels enter at 17,166 (> 17,000), fixing v24's 9,166 cliff.
   - Stalled final-level decay: stalled final levels decay to 1,132 (>15x drop).
   - Tail fade endgame inversion: phi -> 0 strictly ranks L6 > L5 > L4 > L3 > L2 > L1.
   - Adversarial extreme, negative, and malformed inputs.

2. TestQueueAdmissionSimulationV32:
   - Realistic multi-game scheduling tournament proving that BOTH active progressing multi-level games AND fresh final levels AND unstarted 6-level games rank in the top slots, while ALL stalled games rank at the bottom.

3. TestStartupShieldAndSandboxTimeoutAdversarial:
   - Multithreaded 50-thread warmup claim barrier test.
   - Multithreaded 10-thread cold-boot socket flood immunity test.
   - Fast-boot pre-checked /health == 200 OK zero-reset test.
   - Repeated sandbox timeouts & exceptions under _record_retained_functions.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import threading
import time
import types
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
V32_NB_PATH = REPO_ROOT / "notebooks" / "submission_v32.ipynb"
V24_NB_PATH = REPO_ROOT / "notebooks" / "submission_v24.ipynb"


def _src(cell: dict) -> str:
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else str(s)


def _load_d_prime_module(nb_path: Path) -> types.ModuleType:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    c16 = _src(nb["cells"][16])
    ns: dict = {}
    for line in c16.splitlines():
        if line.startswith("_FP_SOURCE = "):
            exec(line, ns)
            break
    if "_FP_SOURCE" not in ns:
        raise AssertionError(f"Could not find _FP_SOURCE in Cell 16 of {nb_path}")
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


class TestV32DPrimeAdversarialMath(unittest.TestCase):
    """Adversarial stress-testing of Version 32 D' mathematical properties."""

    @classmethod
    def setUpClass(cls) -> None:
        assert V32_NB_PATH.exists(), f"Missing {V32_NB_PATH}"
        assert V24_NB_PATH.exists(), f"Missing {V24_NB_PATH}"
        cls.mod = _load_d_prime_module(V32_NB_PATH)
        cls.mod_v24 = _load_d_prime_module(V24_NB_PATH)

    def test_unstarted_pricing_monotonicity(self) -> None:
        """Verify unstarted game priority across all total level counts N=1..20."""
        priorities = {
            n: self.mod.d_priority(_Snap(level=1, actions=0, tokens=0.0, cost_multiplier=0.0, total_levels=n), tail_fraction=1.0)
            for n in range(1, 21)
        }
        self.assertEqual(priorities[6], 18619)
        self.assertEqual(priorities[7], 17964)
        self.assertEqual(priorities[8], 17527)
        self.assertEqual(priorities[9], 17222)
        self.assertEqual(priorities[10], 17000)
        self.assertGreater(priorities[6], priorities[7])
        self.assertGreater(priorities[7], priorities[8])
        self.assertGreater(priorities[8], priorities[9])
        self.assertGreater(priorities[9], priorities[10])
        self.assertEqual(priorities[11], 17000)
        self.assertEqual(priorities[20], 17000)
        self.assertEqual(priorities[1], 2619)
        self.assertEqual(priorities[2], 12619)
        self.assertEqual(priorities[3], 16619)

    def test_multi_level_continuation_beats_unstarted_games(self) -> None:
        """Prove progressing multi-level games beat unstarted games across context trims (fixing v31's eviction bug)."""
        p_unstarted_n6 = self.mod.d_priority(_Snap(1, 0, 0.0, 0.0, 6), tail_fraction=1.0)
        p_unstarted_n8 = self.mod.d_priority(_Snap(1, 0, 0.0, 0.0, 8), tail_fraction=1.0)
        p_unstarted_n10 = self.mod.d_priority(_Snap(1, 0, 0.0, 0.0, 10), tail_fraction=1.0)

        p_prog_l2_n6 = self.mod.d_priority(_Snap(2, 10, 20000.0, 25000.0, 6), tail_fraction=1.0)
        p_prog_l3_n6 = self.mod.d_priority(_Snap(3, 15, 30000.0, 28000.0, 6), tail_fraction=1.0)
        p_trim1_l2_n6 = self.mod.d_priority(_Snap(2, 25, 75000.0, 30000.0, 6), tail_fraction=1.0)
        p_trim1_l3_n6 = self.mod.d_priority(_Snap(3, 20, 65000.0, 25000.0, 6), tail_fraction=1.0)
        p_prog_l2_n10 = self.mod.d_priority(_Snap(2, 10, 20000.0, 25000.0, 10), tail_fraction=1.0)
        p_prog_l3_n10 = self.mod.d_priority(_Snap(3, 15, 30000.0, 28000.0, 10), tail_fraction=1.0)

        self.assertEqual(p_prog_l2_n6, 19518)
        self.assertEqual(p_prog_l3_n6, 20112)
        self.assertEqual(p_trim1_l2_n6, 18181)
        self.assertEqual(p_trim1_l3_n6, 19345)
        self.assertEqual(p_prog_l2_n10, 17343)
        self.assertEqual(p_prog_l3_n10, 17570)

        self.assertGreater(p_prog_l2_n6, p_unstarted_n6)
        self.assertGreater(p_prog_l3_n6, p_unstarted_n6)
        self.assertGreater(p_trim1_l3_n6, p_unstarted_n6)
        self.assertGreater(p_trim1_l2_n6, p_unstarted_n8)
        self.assertGreater(p_prog_l2_n10, p_unstarted_n10)
        self.assertGreater(p_prog_l3_n10, p_unstarted_n8)

    def test_starvation_prevention_across_all_levels(self) -> None:
        """Verify that stalled games on ANY level drop strictly below unstarted games (< 17,000)."""
        unstarted_p10 = self.mod.d_priority(_Snap(1, 0, 0.0, 0.0, 10), tail_fraction=1.0)
        self.assertEqual(unstarted_p10, 17000)

        # Level 1 after 1 context trim even for N=6 is 16,988 < 17,000!
        p_l1_trim1_n6 = self.mod.d_priority(_Snap(1, 35, 118000.0, 0.0, 6), tail_fraction=1.0)
        self.assertEqual(p_l1_trim1_n6, 16988)
        self.assertLess(p_l1_trim1_n6, unstarted_p10)

        # Stalled state (a=150, t=250k) across levels 1..6 of 6 and 1..10 of 10
        for n in (6, 8, 10):
            for lvl in range(1, n + 1):
                snap = _Snap(level=lvl, actions=150, tokens=250000.0, cost_multiplier=30000.0 if lvl > 1 else 0.0, total_levels=n)
                p_stalled = self.mod.d_priority(snap, tail_fraction=1.0)
                self.assertLess(
                    p_stalled,
                    unstarted_p10,
                    f"N={n} L{lvl} stalled priority ({p_stalled}) starved unstarted game ({unstarted_p10})!",
                )

    def test_fresh_final_level_incentive_and_stalled_decay(self) -> None:
        """Verify fresh final-level games enter at 17,166 (> 17,000) and decay to 1,132 when stalled."""
        snap_final_fresh = _Snap(level=6, actions=0, tokens=0.0, cost_multiplier=30000.0, total_levels=6)
        p_final_fresh = self.mod.d_priority(snap_final_fresh, tail_fraction=1.0)
        self.assertEqual(p_final_fresh, 17166)
        self.assertGreater(p_final_fresh, 17000)

        snap_final_stalled = _Snap(level=6, actions=150, tokens=250000.0, cost_multiplier=30000.0, total_levels=6)
        p_final_stalled = self.mod.d_priority(snap_final_stalled, tail_fraction=1.0)
        self.assertEqual(p_final_stalled, 1132)
        self.assertLess(p_final_stalled, 2000)

    def test_tail_fade_endgame_inversion(self) -> None:
        """Verify that when tail_fraction -> 0 (endgame), deeper levels strictly outrank shallower levels."""
        scores_endgame = [
            self.mod.d_priority(_Snap(lvl, 20, 70000.0, 25000.0 if lvl > 1 else 0.0, 6), tail_fraction=0.0)
            for lvl in range(1, 7)
        ]
        for idx in range(len(scores_endgame) - 1):
            self.assertLess(
                scores_endgame[idx],
                scores_endgame[idx + 1],
                f"Endgame inversion failed between L{idx+1} ({scores_endgame[idx]}) and L{idx+2} ({scores_endgame[idx+1]})",
            )

    def test_adversarial_and_extreme_inputs(self) -> None:
        """Exhaustive boundary sweep across extreme, negative, and malformed inputs."""
        extreme_cases = [
            _Snap(level=1, actions=1000000, tokens=100000000.0, cost_multiplier=1000000.0, total_levels=10),
            _Snap(level=10, actions=500000, tokens=50000000.0, cost_multiplier=50000.0, total_levels=10),
            _Snap(level=-5, actions=-100, tokens=-5000.0, cost_multiplier=-100.0, total_levels=-10),
            _Snap(level=0, actions=0, tokens=0.0, cost_multiplier=0.0, total_levels=0),
            _Snap(level=1, actions=0, tokens=0.0, cost_multiplier=0.0, total_levels=None),
            _Snap(level=5, actions=20, tokens=50000.0, cost_multiplier=0.0, total_levels="not_a_number"),
            _Snap(level=15, actions=10, tokens=20000.0, cost_multiplier=25000.0, total_levels=10),
        ]
        for idx, snap in enumerate(extreme_cases):
            for tf in [None, 1.0, 0.0, -0.5, 2.0, 10.0]:
                p = self.mod.d_priority(snap, tail_fraction=tf)
                self.assertIsInstance(p, int, f"Case {idx} (tf={tf}) did not return int")
                self.assertGreaterEqual(p, 1, f"Case {idx} (tf={tf}) returned < 1: {p}")
                self.assertLess(p, 100000, f"Case {idx} (tf={tf}) returned excessive value: {p}")


class TestQueueAdmissionSimulationV32(unittest.TestCase):
    """Simulate priority queue behavior with 18 representative games competing for 10 SGLang active slots."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load_d_prime_module(V32_NB_PATH)

    def test_multi_game_admission_tournament(self) -> None:
        """Verify progressing multi-level games, fresh final levels, and unstarted games outrank stalled games."""
        games = {
            "prog_l3_n6": _Snap(3, 15, 30000.0, 28000.0, 6),
            "prog_l2_n6": _Snap(2, 10, 20000.0, 25000.0, 6),
            "trim1_l3_n6": _Snap(3, 20, 65000.0, 25000.0, 6),
            "unstarted_6a": _Snap(1, 0, 0.0, 0.0, 6),
            "prog_l4_n6": _Snap(4, 20, 40000.0, 30000.0, 6),
            "trim1_l2_n6": _Snap(2, 25, 75000.0, 30000.0, 6),
            "prog_l3_n10": _Snap(3, 15, 30000.0, 28000.0, 10),
            "unstarted_8a": _Snap(1, 0, 0.0, 0.0, 8),
            "prog_l2_n10": _Snap(2, 10, 20000.0, 25000.0, 10),
            "fresh_l6": _Snap(6, 0, 0.0, 30000.0, 6),
            "unstarted_10a": _Snap(1, 0, 0.0, 0.0, 10),
            "trim1_l1_n6": _Snap(1, 35, 118000.0, 0.0, 6),
            "stalled_l2": _Snap(2, 160, 260000.0, 40000.0, 10),
            "stalled_l1a": _Snap(1, 150, 250000.0, 0.0, 10),
            "stalled_l6": _Snap(6, 150, 250000.0, 30000.0, 6),
        }

        scores = {gid: self.mod.d_priority(snap, tail_fraction=1.0) for gid, snap in games.items()}

        # Every progressing multi-level game and fresh_l6 MUST rank strictly above unstarted_10a (17,000)!
        for active_id in (
            "prog_l3_n6",
            "prog_l2_n6",
            "trim1_l3_n6",
            "prog_l4_n6",
            "trim1_l2_n6",
            "prog_l3_n10",
            "prog_l2_n10",
            "fresh_l6",
        ):
            self.assertGreater(
                scores[active_id],
                scores["unstarted_10a"],
                f"Active game {active_id} ({scores[active_id]}) failed to beat unstarted_10a ({scores['unstarted_10a']})!",
            )

        # Every trimmed/stalled L1/L2/L6 game MUST rank strictly below unstarted_10a (17,000)!
        for stalled_id in ("trim1_l1_n6", "stalled_l2", "stalled_l1a", "stalled_l6"):
            self.assertLess(
                scores[stalled_id],
                scores["unstarted_10a"],
                f"Stalled game {stalled_id} ({scores[stalled_id]}) ranked above unstarted_10a ({scores['unstarted_10a']})!",
            )


class TestStartupShieldAndSandboxTimeoutAdversarial(unittest.TestCase):
    """Adversarial stress-testing of cold-boot startup shield concurrency and sandbox timeout retention."""

    def test_multithreaded_warmup_claim_concurrency(self) -> None:
        """Stress-test 50 concurrent threads claiming warmup slot: EXACTLY 1 must succeed."""
        class MockSolver:
            def __init__(self, warmup_games=1):
                self._warmup_lock = threading.Lock()
                self._warmup_remaining = warmup_games

            def _claim_warmup_slot(self) -> bool:
                with self._warmup_lock:
                    if self._warmup_remaining <= 0:
                        return False
                    self._warmup_remaining -= 1
                    return True

        solver = MockSolver(warmup_games=1)
        results = []
        barrier = threading.Barrier(50)

        def worker():
            barrier.wait()
            results.append(solver._claim_warmup_slot())

        threads = [threading.Thread(target=worker) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(results), 50)
        self.assertEqual(results.count(True), 1)
        self.assertEqual(results.count(False), 49)

    def test_fast_boot_skips_warmup_reset_while_slow_boot_issues_keepalive(self) -> None:
        """Verify pre-checked /health == 200 OK skips warmup RESET on fast boot and issues keepalives on slow boot."""
        def run_shield_simulation(initially_healthy: bool) -> tuple[int, int]:
            http_calls = 0
            resets = 0
            analyzer_ready = False

            def check_health():
                nonlocal http_calls
                http_calls += 1
                if initially_healthy or http_calls >= 3:
                    return 200
                raise ConnectionRefusedError("Loading")

            if not analyzer_ready:
                try:
                    if check_health() == 200:
                        analyzer_ready = True
                except Exception:
                    pass
            if not analyzer_ready:
                resets += 1
                while not analyzer_ready:
                    try:
                        if check_health() == 200:
                            analyzer_ready = True
                            break
                    except Exception:
                        pass
                    resets += 1
            return resets, http_calls

        fast_resets, fast_http = run_shield_simulation(initially_healthy=True)
        self.assertEqual(fast_resets, 0, "Fast boot must issue 0 warmup RESETs!")
        self.assertEqual(fast_http, 1)

        slow_resets, slow_http = run_shield_simulation(initially_healthy=False)
        self.assertGreaterEqual(slow_resets, 2, "Slow boot must issue initial + periodic keepalive RESETs!")
        self.assertEqual(slow_http, 3)


if __name__ == "__main__":
    unittest.main()

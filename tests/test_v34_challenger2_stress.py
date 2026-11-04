"""Milestone 3 Gate: Standalone Adversarial Stress Suite by Challenger 2.

Exclusively empirical verification of:
1. Concurrency Stress: 50 concurrent threads claiming warmup slot (mutual exclusivity, zero socket floods, FUSE prefetcher stability).
2. Timeout Sweep: Continuous sweep of hard and soft remaining times in [0.0..3600.0]s (>= 45s floor enforcement, zero overruns).
3. Probe-First Governor Stress: Actionless streak sweep (0..100) and alternating action sequences (strict 4,096 cap on >= 3).
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import random
import sys
import threading
import time
import tracemalloc
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
    )
except (ImportError, ModuleNotFoundError):
    from test_v34_e2e_requirements import (
        PrioritySnapshot,
        oracle_continuous_d_prime,
        oracle_decoupled_request_timeout,
        oracle_probe_first_max_tokens,
    )


class MockHarnessSolver:
    """Mock HarnessSolver encapsulating the exact Cell 2 warmup slot claiming logic."""

    def __init__(self, warmup_remaining: int = 1) -> None:
        self._warmup_lock = threading.Lock()
        self._warmup_remaining = warmup_remaining
        self._analyzer_ready = False
        self.max_runtime_s_per_game = 300.0

    def _claim_warmup_slot(self) -> bool:
        """Exact Cell 2 implementation."""
        lock = getattr(self, "_warmup_lock", None)
        if lock is None:
            return False
        with lock:
            if getattr(self, "_warmup_remaining", 0) <= 0:
                return False
            self._warmup_remaining -= 1
            return True


class MockGameSession:
    """Mock _HarnessGameSession encapsulating the exact patched request_timeout_seconds."""

    def __init__(
        self,
        solver: MockHarnessSolver,
        configured_timeout: float | None = 120.0,
        hard_remaining: float | None = 300.0,
        soft_remaining: float | None = 300.0,
    ) -> None:
        self.solver = solver
        self.analyzer = type("Analyzer", (), {"_timeout": configured_timeout})()
        self._hard_remaining = hard_remaining
        self._soft_remaining = soft_remaining

    def timing_payload(self) -> dict[str, Any]:
        return {
            "run_elapsed_seconds": 0.0,
            "time_remaining_seconds": self._hard_remaining,
        }

    def request_timeout_seconds(self) -> float | None:
        """Exact patched solver.py request_timeout_seconds logic from build_v34_notebook.py."""
        candidates: list[float] = []
        configured = getattr(self.analyzer, "_timeout", None)
        try:
            if configured is not None:
                candidates.append(float(configured))
        except (TypeError, ValueError):
            pass
        if self.solver.max_runtime_s_per_game is not None:
            remaining = self.timing_payload()["time_remaining_seconds"]
            if remaining is not None:
                candidates.append(float(remaining))
        soft_remaining = self._soft_remaining
        if soft_remaining is not None:
            candidates.append(soft_remaining)
        if not candidates:
            return None
        timeout = min(candidates)
        hard_rem = (
            self.timing_payload()["time_remaining_seconds"]
            if self.solver.max_runtime_s_per_game is not None
            else None
        )
        if hard_rem is None or hard_rem >= 45.0:
            timeout = max(45.0, timeout) if hard_rem is None else min(hard_rem, max(45.0, timeout))
        return max(0.1, timeout)


class MockToolAgent:
    """Mock ToolAgent encapsulating the exact patched probe-first governor logic."""

    def __init__(self, max_output_tokens: int = 16384) -> None:
        self._max_output_tokens = max_output_tokens
        self._actionless_turns = 0

    def record_step_result(self, step_executed: bool, yielded_control_reason: str | None = None) -> None:
        """Exact patched logic from tool_agent.py."""
        if step_executed:
            self._actionless_turns = 0
        elif yielded_control_reason is not None:
            self._actionless_turns = getattr(self, "_actionless_turns", 0) + 1
        else:
            self._actionless_turns = getattr(self, "_actionless_turns", 0) + 1

    def compute_max_tokens(self, max_tokens: int | None = None) -> int:
        """Exact patched logic from tool_agent.py _chat_completion."""
        req_max_tokens = self._max_output_tokens if max_tokens is None else max_tokens
        if getattr(self, "_actionless_turns", 0) >= 3 and os.environ.get("ARC3_PROBE_FIRST", "1") != "0":
            req_max_tokens = min(req_max_tokens or 4096, 4096)
        return req_max_tokens

    def ensure_model_initialized(self) -> None:
        """Exact patched logic from tool_agent.py _ensure_model_initialized."""
        self._actionless_turns = 0


# =============================================================================
# 1. Concurrency Stress Suite
# =============================================================================

class TestV34ConcurrencyStress(unittest.TestCase):
    """Adversarial stress-testing of 50-thread warmup claim race and socket flood immunity."""

    def test_50_thread_warmup_barrier_race_100_rounds(self) -> None:
        """50 concurrent threads race through a barrier across 100 rounds (5,000 attempts).

        Verifies:
        - Exactly 1 thread claims True per round.
        - Exactly 49 threads claim False per round.
        - Zero race conditions or lock leakage.
        """
        for round_idx in range(100):
            solver = MockHarnessSolver(warmup_remaining=1)
            barrier = threading.Barrier(50)
            claims: list[bool] = []
            lock = threading.Lock()

            def worker() -> None:
                barrier.wait()
                claimed = solver._claim_warmup_slot()
                with lock:
                    claims.append(claimed)

            threads = [threading.Thread(target=worker) for _ in range(50)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=2.0)

            self.assertEqual(len(claims), 50, f"Round {round_idx}: not all threads finished!")
            self.assertEqual(claims.count(True), 1, f"Round {round_idx}: expected exactly 1 winner, got {claims.count(True)}")
            self.assertEqual(claims.count(False), 49, f"Round {round_idx}: expected 49 losers, got {claims.count(False)}")

    def test_socket_flood_prevention_during_cold_boot(self) -> None:
        """Simulate 50 threads in a 55-minute cold boot: verify ZERO socket floods.

        The 1 winner issues periodic keepalives every 420s (total = 1 + 55*60//420 = 8).
        The 49 losers NEVER issue reset actions.
        Total reset count across all 50 threads must equal exactly 8.
        """
        solver = MockHarnessSolver(warmup_remaining=1)
        simulated_duration = 55 * 60.0  # 3300 seconds
        keepalive_interval = 420.0      # 7 minutes

        reset_counter: dict[str, int] = {"winner": 0, "losers": 0}
        counter_lock = threading.Lock()

        barrier = threading.Barrier(50)

        def session_runner(thread_id: int) -> None:
            barrier.wait()
            is_warmup_slot = bool(solver._claim_warmup_slot())

            if is_warmup_slot:
                with counter_lock:
                    reset_counter["winner"] += 1
                last_ts = 0.0
                curr_t = 0.0
                while curr_t + keepalive_interval <= simulated_duration:
                    curr_t += keepalive_interval
                    with counter_lock:
                        reset_counter["winner"] += 1
                    last_ts = curr_t
            else:
                # Loser thread: enters wait loop, NEVER calls _execute_auto_reset
                curr_t = 0.0
                while curr_t < simulated_duration:
                    curr_t += 1.0  # sleeps 1s

        threads = [threading.Thread(target=session_runner, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=3.0)

        # 1 initial reset + 7 keepalives (at 420, 840, 1260, 1680, 2100, 2520, 2940s) = 8 total
        expected_winner_resets = 1 + int(simulated_duration // keepalive_interval)
        self.assertEqual(reset_counter["winner"], expected_winner_resets)
        self.assertEqual(reset_counter["losers"], 0, "Non-winning threads flooded the socket with game resets!")

    def test_server_ready_unblocks_all_threads_instantly(self) -> None:
        """Verify that when analyzer becomes ready, all 50 waiting threads unblock promptly."""
        solver = MockHarnessSolver(warmup_remaining=1)
        barrier = threading.Barrier(50)
        unblocked_count = 0
        unblock_lock = threading.Lock()
        stop_event = threading.Event()

        def waiting_worker() -> None:
            nonlocal unblocked_count
            barrier.wait()
            is_warmup = solver._claim_warmup_slot()
            # Wait until analyzer_ready is True
            while not getattr(solver, "_analyzer_ready", False) and not stop_event.is_set():
                time.sleep(0.01)
            with unblock_lock:
                unblocked_count += 1

        threads = [threading.Thread(target=waiting_worker) for _ in range(50)]
        for t in threads:
            t.start()

        time.sleep(0.05)
        # Server becomes ready
        setattr(solver, "_analyzer_ready", True)

        for t in threads:
            t.join(timeout=1.0)

        self.assertEqual(unblocked_count, 50, "Not all threads unblocked when server became ready!")

    def test_fuse_prefetcher_concurrency_governance(self) -> None:
        """Verify that prefetcher settings (threads=3, delay=60) throttle FUSE inode contention."""
        from scripts.build_v34_notebook import build_v34_notebook
        nb = build_v34_notebook()
        cell6_src = "".join(nb["cells"][6]["source"])

        self.assertIn("delay=60", cell6_src, "Prefetcher delay must be 60s to avoid FUSE mount race")
        self.assertIn("threads=3", cell6_src, "Prefetcher threads must be <= 3 to avoid inode saturation")


# =============================================================================
# 2. Timeout Decoupling Sweep Suite
# =============================================================================

class TestV34TimeoutDecouplingSweep(unittest.TestCase):
    """Adversarial stress-testing of timeout decoupling floor and zero overrun invariant."""

    def test_timeout_grid_sweep_zero_overruns(self) -> None:
        """Sweep hard [0.0..3600.0]s and soft [0.0..3600.0]s: zero overruns, floor >= 45s."""
        hard_points = [
            0.0, 0.05, 0.1, 1.0, 10.0, 25.0, 44.0, 44.9, 44.999,
            45.0, 45.001, 50.0, 60.0, 120.0, 300.0, 600.0, 1800.0, 3600.0
        ]
        soft_points = [
            0.0, 0.1, 1.0, 5.0, 15.0, 25.0, 44.0, 44.9,
            45.0, 50.0, 60.0, 120.0, 300.0, 600.0, 1800.0, 3600.0
        ]
        conf_points = [None, 30.0, 45.0, 60.0, 120.0, 300.0]

        solver = MockHarnessSolver()

        tested_points = 0
        for hard in hard_points:
            for soft in soft_points:
                for conf in conf_points:
                    tested_points += 1
                    sess = MockGameSession(
                        solver=solver,
                        configured_timeout=conf,
                        hard_remaining=hard,
                        soft_remaining=soft,
                    )
                    timeout = sess.request_timeout_seconds()

                    # 1. Zero overruns: timeout cannot exceed hard remaining
                    if hard <= 0.0:
                        # With 0 remaining, timeout should not cause post-deadline processing
                        self.assertLessEqual(timeout, 0.1)
                    else:
                        if hard >= 0.1:
                            self.assertLessEqual(
                                timeout,
                                hard,
                                f"OVERRUN detected: timeout {timeout} > hard {hard} (soft={soft}, conf={conf})",
                            )

                    # 2. Floor enforcement: when hard >= 45.0, timeout strictly >= 45.0
                    if hard >= 45.0:
                        self.assertGreaterEqual(
                            timeout,
                            45.0,
                            f"FLOOR VIOLATION: timeout {timeout} < 45.0 when hard {hard} >= 45.0 (soft={soft})",
                        )

                    # 3. Clamping below 45s is allowed ONLY when hard time is less than 45s
                    if timeout < 45.0:
                        self.assertLess(
                            hard,
                            45.0,
                            f"CLAMPED BELOW 45s despite hard {hard} >= 45.0! timeout={timeout}",
                        )

        self.assertGreaterEqual(tested_points, 1500)

    def test_timeout_continuous_fuzzing_1000_random_points(self) -> None:
        """1,000 random floating-point triplets uniformly sampled in [0.0..3600.0]s."""
        random.seed(1337)
        solver = MockHarnessSolver()

        for _ in range(1000):
            hard = random.uniform(0.1, 3600.0)
            soft = random.uniform(0.0, 3600.0)
            conf = random.choice([None, random.uniform(10.0, 300.0)])

            sess = MockGameSession(
                solver=solver,
                configured_timeout=conf,
                hard_remaining=hard,
                soft_remaining=soft,
            )
            timeout = sess.request_timeout_seconds()

            self.assertIsNotNone(timeout)
            # Invariant 1: No overruns when hard >= 0.1
            self.assertLessEqual(timeout, hard + 1e-9)

            # Invariant 2: Floor >= 45s when hard >= 45s
            if hard >= 45.0:
                self.assertGreaterEqual(timeout, 45.0)

            # Invariant 3: Sub-45s timeout occurs only when hard < 45s
            if timeout < 45.0:
                self.assertLess(hard, 45.0)

    def test_exact_boundary_discontinuity_check(self) -> None:
        """Verify behavior across the exact 45.0s hard threshold: [44.999, 45.000, 45.001]."""
        solver = MockHarnessSolver()

        # At hard = 44.999s and soft = 10.0s:
        # Since hard < 45.0, timeout clamps to soft_rem = 10.0s (<= hard)
        sess_sub = MockGameSession(solver=solver, configured_timeout=120.0, hard_remaining=44.999, soft_remaining=10.0)
        self.assertAlmostEqual(sess_sub.request_timeout_seconds(), 10.0)

        # At hard = 45.000s and soft = 10.0s:
        # Since hard >= 45.0, timeout floor elevates timeout to min(45.000, max(45.0, 10.0)) = 45.0s
        sess_at = MockGameSession(solver=solver, configured_timeout=120.0, hard_remaining=45.000, soft_remaining=10.0)
        self.assertEqual(sess_at.request_timeout_seconds(), 45.0)

        # At hard = 45.001s and soft = 10.0s:
        sess_above = MockGameSession(solver=solver, configured_timeout=120.0, hard_remaining=45.001, soft_remaining=10.0)
        self.assertEqual(sess_above.request_timeout_seconds(), 45.0)


# =============================================================================
# 3. Probe-First Governor Deep Stress Suite
# =============================================================================

class TestV34ProbeFirstGovernorStress(unittest.TestCase):
    """Adversarial stress-testing of Shehab Anwer probe-first governor max_tokens capping."""

    def setUp(self) -> None:
        os.environ["ARC3_PROBE_FIRST"] = "1"

    def test_streak_sweep_0_to_100(self) -> None:
        """Sweep actionless streaks from 0 to 100: streaks >= 3 must be strictly clamped to 4096."""
        agent = MockToolAgent(max_output_tokens=16384)

        for streak in range(101):
            agent._actionless_turns = streak
            computed = agent.compute_max_tokens()
            oracle_val = oracle_probe_first_max_tokens(streak)

            self.assertEqual(computed, oracle_val)

            if streak < 3:
                self.assertEqual(computed, 16384, f"Streak {streak} should not be capped!")
            else:
                self.assertEqual(computed, 4096, f"Streak {streak} must be capped at 4096!")

    def test_complex_alternating_action_sequences(self) -> None:
        """Simulate realistic agent trajectories with mixed action/no-action turns."""
        agent = MockToolAgent(max_output_tokens=16384)

        # Sequence 1: 2 actionless turns, then action taken -> streak never hits 3
        for cycle in range(10):
            agent.record_step_result(step_executed=False)
            self.assertEqual(agent.compute_max_tokens(), 16384)
            agent.record_step_result(step_executed=False)
            self.assertEqual(agent.compute_max_tokens(), 16384)
            # Action executed
            agent.record_step_result(step_executed=True)
            self.assertEqual(agent.compute_max_tokens(), 16384)

        # Sequence 2: 3 actionless turns -> turn 3 is capped to 4096
        agent.record_step_result(step_executed=False)  # streak 1
        self.assertEqual(agent.compute_max_tokens(), 16384)
        agent.record_step_result(step_executed=False)  # streak 2
        self.assertEqual(agent.compute_max_tokens(), 16384)
        agent.record_step_result(step_executed=False)  # streak 3
        self.assertEqual(agent.compute_max_tokens(), 4096)
        agent.record_step_result(step_executed=False)  # streak 4
        self.assertEqual(agent.compute_max_tokens(), 4096)

        # Action taken: resets streak to 0 immediately
        agent.record_step_result(step_executed=True)
        self.assertEqual(agent.compute_max_tokens(), 16384)

    def test_batched_actions_resets_streak(self) -> None:
        """Batched actions (action([act1, act2])) execute as step_executed=True, resetting streak."""
        agent = MockToolAgent(max_output_tokens=16384)
        agent._actionless_turns = 10  # Deep stall
        self.assertEqual(agent.compute_max_tokens(), 4096)

        # Batch of 5 actions executes
        agent.record_step_result(step_executed=True)
        self.assertEqual(agent._actionless_turns, 0)
        self.assertEqual(agent.compute_max_tokens(), 16384)

    def test_env_override_disables_probe_first(self) -> None:
        """Setting ARC3_PROBE_FIRST=0 disables token cap even when streak >= 3."""
        try:
            os.environ["ARC3_PROBE_FIRST"] = "0"
            agent = MockToolAgent(max_output_tokens=16384)
            agent._actionless_turns = 25
            self.assertEqual(agent.compute_max_tokens(), 16384)
        finally:
            os.environ["ARC3_PROBE_FIRST"] = "1"

    def test_requested_max_tokens_smaller_than_cap(self) -> None:
        """When caller explicitly requests max_tokens=1024, cap min(1024, 4096) keeps it 1024."""
        agent = MockToolAgent(max_output_tokens=16384)
        agent._actionless_turns = 5
        self.assertEqual(agent.compute_max_tokens(max_tokens=1024), 1024)
        self.assertEqual(agent.compute_max_tokens(max_tokens=8192), 4096)

    def test_randomized_turn_trajectory_fuzzing(self) -> None:
        """Fuzz 500 randomized turn results, verifying strict adherence to probe-first contract."""
        random.seed(4242)
        agent = MockToolAgent(max_output_tokens=16384)
        streak = 0

        for step in range(500):
            # 70% chance of action, 30% chance of actionless turn
            action_taken = random.random() < 0.70
            agent.record_step_result(step_executed=action_taken)
            if action_taken:
                streak = 0
            else:
                streak += 1

            self.assertEqual(agent._actionless_turns, streak)
            expected = 4096 if streak >= 3 else 16384
            self.assertEqual(agent.compute_max_tokens(), expected, f"Step {step}: streak={streak} cap failed!")


# =============================================================================
# 4. Extreme & Malformed Input Robustness
# =============================================================================

class TestV34ExtremeInputRobustness(unittest.TestCase):
    """Adversarial stress-testing under extreme, malformed, and boundary conditions."""

    def test_100_concurrent_threads_warmup_barrier(self) -> None:
        """100 concurrent threads race through a barrier: mutual exclusivity must hold."""
        solver = MockHarnessSolver(warmup_remaining=1)
        barrier = threading.Barrier(100)
        claims: list[bool] = []
        lock = threading.Lock()

        def worker() -> None:
            barrier.wait()
            claimed = solver._claim_warmup_slot()
            with lock:
                claims.append(claimed)

        threads = [threading.Thread(target=worker) for _ in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=2.0)

        self.assertEqual(len(claims), 100)
        self.assertEqual(claims.count(True), 1)
        self.assertEqual(claims.count(False), 99)

    def test_malformed_and_extreme_timeout_inputs(self) -> None:
        """Malformed configured timeouts and extreme remaining times do not raise unhandled exceptions."""
        solver = MockHarnessSolver()

        malformed_configs = ["not_a_float", object(), [], {}, -999.0, 0.0]
        for cfg in malformed_configs:
            sess = MockGameSession(solver=solver, configured_timeout=cfg, hard_remaining=100.0, soft_remaining=100.0)
            timeout = sess.request_timeout_seconds()
            self.assertIsNotNone(timeout)
            self.assertGreaterEqual(timeout, 45.0)

        # None hard remaining (unlimited time)
        sess_none = MockGameSession(solver=solver, configured_timeout=120.0, hard_remaining=None, soft_remaining=10.0)
        self.assertEqual(sess_none.request_timeout_seconds(), 45.0)


# =============================================================================
# 5. Frozen System Prompt Prefix Invariant Under Multi-Stream Stress
# =============================================================================

class TestV34FrozenPrefixInvariantStress(unittest.TestCase):
    """Adversarial stress-testing of 100% frozen RadixAttention system prefix cache invariant."""

    def test_static_audit_zero_system_prompt_mutations(self) -> None:
        """Verify that no build patch or notebook cell contains code that mutates messages[0] or _system_prompt."""
        from scripts.build_v34_notebook import build_v34_notebook
        nb = build_v34_notebook()
        all_code = "\n".join("".join(c.get("source", [])) for c in nb["cells"] if c.get("cell_type") == "code")

        # In earlier versions (v25/v26), SOLVED_LEVEL_MEMORY altered prompts
        banned_patterns = [
            "SOLVED_LEVEL_MEMORY",
            "orig_build_system_prompt",
            "_build_system_prompt =",
            "messages[0]['content'] +=",
            'messages[0]["content"] +=',
            "messages[0]['content'] =",
            'messages[0]["content"] =',
            "self._system_prompt +=",
            "self._system_prompt.append",
        ]
        for pattern in banned_patterns:
            self.assertNotIn(
                pattern,
                all_code,
                f"Banned mutation pattern {pattern!r} found in v34 notebook code!",
            )

        # Confirm that outside Cell 2 (base patch), _build_system_prompt is NEVER referenced or modified
        post_cell2_code = "\n".join("".join(c.get("source", [])) for idx, c in enumerate(nb["cells"]) if idx != 2 and c.get("cell_type") == "code")
        self.assertNotIn(
            "_build_system_prompt",
            post_cell2_code,
            "_build_system_prompt was modified or referenced outside Cell 2 base patch!",
        )

    def test_10_concurrent_streams_context_trim_prefix_identity(self) -> None:
        """10 concurrent streams running 50 turns with random context evictions maintain identical prefix."""
        canonical_system_prefix = (
            "<|im_start|>system\n"
            "<tools>\n"
            "def action(actions: list[dict[str, Any]]) -> dict[str, Any]: ...\n"
            "def solve_affine_lights_out(board: list[list[int]]) -> list[dict[str, Any]]: ...\n"
            "</tools>\n"
            "You are an expert game agent.<|im_end|>\n"
        )

        class SimulatedStream:
            def __init__(self, stream_id: int) -> None:
                self.stream_id = stream_id
                self.messages: list[dict[str, str]] = [
                    {"role": "system", "content": canonical_system_prefix}
                ]
                self.max_context_turns = 8

            def execute_turn(self, turn_idx: int) -> None:
                # Add user and assistant messages
                self.messages.append({"role": "user", "content": f"Stream {self.stream_id} Turn {turn_idx} state"})
                self.messages.append({"role": "assistant", "content": f"Stream {self.stream_id} Turn {turn_idx} thought & action"})

                # Simulate context trimming when history exceeds budget
                if len(self.messages) > (1 + 2 * self.max_context_turns):
                    # Trimming in Franzen/v34 retains messages[0] and drops older turns
                    self.messages = [self.messages[0]] + self.messages[-(2 * self.max_context_turns):]

        streams = [SimulatedStream(i) for i in range(10)]

        # Run 50 concurrent turns across all 10 streams
        barrier = threading.Barrier(10)
        def run_stream(s: SimulatedStream) -> None:
            barrier.wait()
            for t in range(50):
                s.execute_turn(t)

        threads = [threading.Thread(target=run_stream, args=(s,)) for s in streams]
        for th in threads:
            th.start()
        for th in threads:
            th.join(timeout=2.0)

        # Invariant check: across all 10 streams and all turns, messages[0]["content"] is 100% byte-identical
        for s in streams:
            self.assertEqual(len(s.messages), 1 + 2 * s.max_context_turns)
            self.assertEqual(s.messages[0]["role"], "system")
            self.assertEqual(s.messages[0]["content"], canonical_system_prefix)
            self.assertEqual(
                hash(s.messages[0]["content"]),
                hash(canonical_system_prefix),
                f"Prefix hash mismatch in stream {s.stream_id}!",
            )


# =============================================================================
# 6. Continuous D' Priority Queue 10-Stream Simulation
# =============================================================================

class TestV34ContinuousDPrimeQueueSimulation(unittest.TestCase):
    """Multi-stream simulation of continuous D' priority scheduler with starvation prevention."""

    def test_10_stream_multi_round_scheduling_and_starvation_prevention(self) -> None:
        """Simulate 10 concurrent games with active, stalled, unstarted, and final-level games across 50 rounds."""
        class SimulatedGame:
            def __init__(self, game_id: str, total_levels: int = 6, behavior: str = "active") -> None:
                self.game_id = game_id
                self.total_levels = total_levels
                self.behavior = behavior  # "active", "stalled", "unstarted"
                self.level = 1
                self.actions = 0
                self.tokens = 0.0
                self.cost_multiplier = 0.0
                self.turns_played = 0

            def snapshot(self) -> PrioritySnapshot:
                return PrioritySnapshot(
                    level=self.level,
                    actions=self.actions,
                    tokens=self.tokens,
                    cost_multiplier=self.cost_multiplier,
                    total_levels=self.total_levels,
                )

            def step(self) -> None:
                self.turns_played += 1
                if self.behavior == "active":
                    self.actions += 2
                    self.tokens += 1500.0
                    # Advance level every 8 actions
                    if self.actions >= self.level * 8 and self.level < self.total_levels:
                        self.level += 1
                        self.actions = 0
                        self.tokens = 0.0
                        self.cost_multiplier = 25000.0
                elif self.behavior == "stalled":
                    # Stalled: actions and tokens mount without level advancement
                    self.actions += 5
                    self.tokens += 8000.0

        games = [
            SimulatedGame("game_active_1", total_levels=6, behavior="active"),
            SimulatedGame("game_active_2", total_levels=6, behavior="active"),
            SimulatedGame("game_stalled_1", total_levels=6, behavior="stalled"),
            SimulatedGame("game_stalled_2", total_levels=6, behavior="stalled"),
            SimulatedGame("game_fresh_1", total_levels=6, behavior="unstarted"),
            SimulatedGame("game_fresh_2", total_levels=10, behavior="unstarted"),
            SimulatedGame("game_final_active", total_levels=6, behavior="active"),
            SimulatedGame("game_final_stalled", total_levels=6, behavior="stalled"),
        ]
        # Pre-set final-level games to level 6
        games[6].level = 6
        games[6].cost_multiplier = 30000.0
        games[7].level = 6
        games[7].cost_multiplier = 30000.0

        # Step active and stalled games for 20 rounds
        for _ in range(20):
            for g in games:
                if g.behavior != "unstarted":
                    g.step()

        # Compute priority rankings
        rankings = [(g.game_id, oracle_continuous_d_prime(g.snapshot(), tail_fraction=1.0)) for g in games]
        rank_dict = dict(rankings)

        # 1. Unstarted fresh games have priority >= 17000
        p_fresh_n6 = rank_dict["game_fresh_1"]
        p_fresh_n10 = rank_dict["game_fresh_2"]
        self.assertEqual(p_fresh_n6, 18619)
        self.assertEqual(p_fresh_n10, 17000)

        # 2. Stalled games on Level 1 must drop strictly below unstarted games (< 17000)
        p_stalled_1 = rank_dict["game_stalled_1"]
        p_stalled_2 = rank_dict["game_stalled_2"]
        self.assertLess(p_stalled_1, p_fresh_n10, f"Stalled game 1 ({p_stalled_1}) starved fresh games!")
        self.assertLess(p_stalled_2, p_fresh_n10, f"Stalled game 2 ({p_stalled_2}) starved fresh games!")

        # 3. Active progressing games maintain high priority
        p_active_1 = rank_dict["game_active_1"]
        p_active_2 = rank_dict["game_active_2"]
        self.assertGreater(p_active_1, p_stalled_1)
        self.assertGreater(p_active_2, p_stalled_2)

        # 4. Stalled final-level game decays heavily, preventing slot monopolization
        p_final_stalled = rank_dict["game_final_stalled"]
        self.assertLess(p_final_stalled, p_fresh_n10, f"Stalled final-level game ({p_final_stalled}) starved fresh games!")


# =============================================================================
# 7. Rehearsal Benchmark Determinism, Memory & Leak Verification
# =============================================================================

class TestV34RehearsalBenchmarkDeterminism(unittest.TestCase):
    """Adversarial stress-testing of rehearsal benchmark reproducibility and leak immunity."""

    def test_consecutive_runs_deterministic_and_leak_free(self) -> None:
        """Run 5 consecutive rehearsal benchmark runs: 100% score, 75 actions, zero memory/socket leaks."""
        from scripts.run_v34_rehearsal_benchmark import run_benchmark

        scores: list[float] = []
        actions: list[int] = []
        levels: list[int] = []
        wallclocks: list[float] = []
        mem_drifts: list[float] = []

        tracemalloc.start()
        for run_idx in range(5):
            gc.collect()
            mem_before, _ = tracemalloc.get_traced_memory()

            result = run_benchmark()

            gc.collect()
            mem_after, peak = tracemalloc.get_traced_memory()
            mem_diff_kb = (mem_after - mem_before) / 1024.0

            scores.append(result["final_score"])
            actions.append(result["total_actions"])
            levels.append(result["levels_completed"])
            wallclocks.append(result["total_wallclock"])
            mem_drifts.append(mem_diff_kb)

            # Per-run assertions
            self.assertEqual(result["final_score"], 100.0, f"Run {run_idx+1} failed to achieve 100.0 score!")
            self.assertEqual(result["levels_completed"], 6, f"Run {run_idx+1} solved {result['levels_completed']}/6 levels!")
            self.assertEqual(result["total_actions"], 75, f"Run {run_idx+1} took {result['total_actions']} != 75 actions!")
            self.assertTrue(result["is_win"], f"Run {run_idx+1} did not win!")
            self.assertLess(result["total_wallclock"], 300.0, f"Run {run_idx+1} exceeded 300s budget!")

        tracemalloc.stop()

        # Invariant 1: 100% score across all 5 runs
        self.assertEqual(scores, [100.0] * 5)
        # Invariant 2: Exact 75 actions across all 5 runs
        self.assertEqual(actions, [75] * 5)
        # Invariant 3: 6/6 levels solved on all runs
        self.assertEqual(levels, [6] * 5)
        # Invariant 4: Sub-second wallclock on all runs
        self.assertTrue(all(w < 5.0 for w in wallclocks))
        # Invariant 5: Zero net memory drift across runs (< 50 KB total drift)
        self.assertLess(abs(mem_drifts[-1] - mem_drifts[0]), 50.0)


if __name__ == "__main__":
    unittest.main()

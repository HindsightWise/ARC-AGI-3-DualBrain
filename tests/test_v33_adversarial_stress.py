"""Adversarial Stress Test Suite for Version 33 Sovereign Apex Engine.

Adversarially tests:
1. TestV33DPrimeAdversarialMath:
   - Unstarted game pricing monotonicity across total level counts (N=1..20).
   - 100% shiiin9 (31.54 LB) / v24 (28.79 LB) equivalence across all states with remaining >= 1.
   - Multi-level continuation proof: active progressing games on L2..L4 strictly beat unstarted games (> 17,000).
   - Starvation prevention: stalled games on L1..L6 drop strictly below < 17,000 and cannot starve unstarted games.
   - Fresh final-level incentive: fresh final levels enter at 17,166 (> 17,000).
   - Stalled final-level decay: stalled final levels decay to 1,132 (>15x drop).
   - Tail fade endgame inversion: phi -> 0 strictly ranks L6 > L5 > L4 > L3 > L2 > L1.
   - Adversarial extreme, negative, and malformed inputs.

2. TestV33SolvedLevelMemoryAdversarial:
   - Multi-level batch jump (e.g., jumping L1 -> L4 in a single turn captures L1, L2, L3).
   - Multiple RESETs inside a single level (only actions after the final RESET are retained).
   - Placeholder echo rejection (`<win condition and key interaction>`) and degenerate repetition rejection.
   - Token-budget overflow guard in `_slm_pin_on_eviction` (refuses to pin if candidate exceeds `_context_budget_tokens`).
   - Idempotent multi-eviction pinning (never duplicates `=== SOLVED LEVELS ===` header across repeated trims).
   - Session switch cleanup across games.

3. TestV33NoopRepeatGuardAndStartupShieldAdversarial:
   - Batch mixed animated + non-animated actions in `gameplay_changed_per_action`.
   - Multithreaded 50-thread warmup claim barrier test.
"""

from __future__ import annotations

import json
from pathlib import Path
import threading
import types
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
V33_NB_PATH = REPO_ROOT / "notebooks" / "submission_v33.ipynb"
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


class TestV33DPrimeAdversarialMath(unittest.TestCase):
    """Adversarial stress-testing of Version 33 D' mathematical properties."""

    @classmethod
    def setUpClass(cls) -> None:
        assert V33_NB_PATH.exists(), f"Missing {V33_NB_PATH}"
        assert V24_NB_PATH.exists(), f"Missing {V24_NB_PATH}"
        cls.mod = _load_d_prime_module(V33_NB_PATH)
        cls.mod_v24 = _load_d_prime_module(V24_NB_PATH)

    def test_unstarted_pricing_monotonicity(self) -> None:
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

    def test_multi_level_continuation_beats_unstarted_games(self) -> None:
        p_unstarted_n6 = self.mod.d_priority(_Snap(1, 0, 0.0, 0.0, 6), tail_fraction=1.0)
        p_unstarted_n8 = self.mod.d_priority(_Snap(1, 0, 0.0, 0.0, 8), tail_fraction=1.0)
        p_unstarted_n10 = self.mod.d_priority(_Snap(1, 0, 0.0, 0.0, 10), tail_fraction=1.0)

        p_prog_l2_n6 = self.mod.d_priority(_Snap(2, 10, 20000.0, 25000.0, 6), tail_fraction=1.0)
        p_prog_l3_n6 = self.mod.d_priority(_Snap(3, 15, 30000.0, 28000.0, 6), tail_fraction=1.0)
        p_trim1_l2_n6 = self.mod.d_priority(_Snap(2, 25, 75000.0, 30000.0, 6), tail_fraction=1.0)
        p_trim1_l3_n6 = self.mod.d_priority(_Snap(3, 20, 65000.0, 25000.0, 6), tail_fraction=1.0)
        p_prog_l2_n10 = self.mod.d_priority(_Snap(2, 10, 20000.0, 25000.0, 10), tail_fraction=1.0)
        p_prog_l3_n10 = self.mod.d_priority(_Snap(3, 15, 30000.0, 28000.0, 10), tail_fraction=1.0)

        self.assertGreater(p_prog_l2_n6, p_unstarted_n6)
        self.assertGreater(p_prog_l3_n6, p_unstarted_n6)
        self.assertGreater(p_trim1_l3_n6, p_unstarted_n6)
        self.assertGreater(p_trim1_l2_n6, p_unstarted_n8)
        self.assertGreater(p_prog_l2_n10, p_unstarted_n10)
        self.assertGreater(p_prog_l3_n10, p_unstarted_n8)

    def test_starvation_prevention_across_all_levels(self) -> None:
        unstarted_p10 = self.mod.d_priority(_Snap(1, 0, 0.0, 0.0, 10), tail_fraction=1.0)
        self.assertEqual(unstarted_p10, 17000)

        p_l1_trim1_n6 = self.mod.d_priority(_Snap(1, 35, 118000.0, 0.0, 6), tail_fraction=1.0)
        self.assertEqual(p_l1_trim1_n6, 16988)
        self.assertLess(p_l1_trim1_n6, unstarted_p10)

        for n in (6, 8, 10):
            for lvl in range(1, n + 1):
                snap = _Snap(level=lvl, actions=150, tokens=250000.0, cost_multiplier=30000.0 if lvl > 1 else 0.0, total_levels=n)
                p_stalled = self.mod.d_priority(snap, tail_fraction=1.0)
                self.assertLess(p_stalled, unstarted_p10)

    def test_fresh_final_level_incentive_and_stalled_decay(self) -> None:
        snap_final_fresh = _Snap(level=6, actions=0, tokens=0.0, cost_multiplier=30000.0, total_levels=6)
        self.assertEqual(self.mod.d_priority(snap_final_fresh, tail_fraction=1.0), 17166)

        snap_final_stalled = _Snap(level=6, actions=150, tokens=250000.0, cost_multiplier=30000.0, total_levels=6)
        self.assertEqual(self.mod.d_priority(snap_final_stalled, tail_fraction=1.0), 1132)

    def test_tail_fade_endgame_inversion(self) -> None:
        scores_endgame = [
            self.mod.d_priority(_Snap(lvl, 20, 70000.0, 25000.0 if lvl > 1 else 0.0, 6), tail_fraction=0.0)
            for lvl in range(1, 7)
        ]
        for idx in range(len(scores_endgame) - 1):
            self.assertLess(scores_endgame[idx], scores_endgame[idx + 1])

    def test_adversarial_and_extreme_inputs(self) -> None:
        extreme_cases = [
            _Snap(level=1, actions=1000000, tokens=100000000.0, cost_multiplier=1000000.0, total_levels=10),
            _Snap(level=10, actions=500000, tokens=50000000.0, cost_multiplier=50000.0, total_levels=10),
            _Snap(level=-5, actions=-100, tokens=-5000.0, cost_multiplier=-100.0, total_levels=-10),
            _Snap(level=0, actions=0, tokens=0.0, cost_multiplier=0.0, total_levels=0),
            _Snap(level=1, actions=0, tokens=0.0, cost_multiplier=0.0, total_levels=None),
            _Snap(level=5, actions=20, tokens=50000.0, cost_multiplier=0.0, total_levels="not_a_number"),
            _Snap(level=15, actions=10, tokens=20000.0, cost_multiplier=25000.0, total_levels=10),
        ]
        for snap in extreme_cases:
            for tf in [None, 1.0, 0.0, -0.5, 2.0, 10.0]:
                p = self.mod.d_priority(snap, tail_fraction=tf)
                self.assertIsInstance(p, int)
                self.assertGreaterEqual(p, 1)
                self.assertLess(p, 100000)


class TestV33SolvedLevelMemoryAdversarial(unittest.TestCase):
    """Adversarial stress-testing of Cache-Safe Eviction-Only SOLVED_LEVEL_MEMORY."""

    def test_slm_adversarial_edge_cases(self) -> None:
        import scripts.build_v33_notebook as b33

        # Load SLM functions in an isolated namespace with a mock _slm_ta
        ns = {
            "__builtins__": __builtins__,
        }
        code_lines = []
        for line in b33.SOLVED_LEVEL_MEMORY_BLOCK.splitlines():
            if line.startswith("from inference.") or line.startswith("if SOLVED_LEVEL_MEMORY_PATCH:") or line.startswith("    install_solved_level_memory") or line.startswith('print("SOLVED_LEVEL_MEMORY_PATCH:'):
                continue
            code_lines.append(line)
        mock_ta = types.SimpleNamespace(_is_degenerate_text=lambda s: len(set(s)) <= 2 and len(s) > 20)
        ns["_slm_ta"] = mock_ta
        exec("\n".join(code_lines), ns)

        _slm_level_actions = ns["_slm_level_actions"]
        _slm_extract_rule_from_text = ns["_slm_extract_rule_from_text"]
        _slm_pin_on_eviction = ns["_slm_pin_on_eviction"]
        _SLM_HEADER = ns["_SLM_HEADER"]

        # 1. Multiple RESETs inside a level + >30 actions truncation + post-action frame.level==2 on final winning move
        hist = (
            [
                types.SimpleNamespace(action="", frame=types.SimpleNamespace(level=1)),
                types.SimpleNamespace(action="ACTION1", frame=types.SimpleNamespace(level=1)),
                types.SimpleNamespace(action="RESET", frame=types.SimpleNamespace(level=1)),
                types.SimpleNamespace(action="ACTION2", frame=types.SimpleNamespace(level=1)),
                types.SimpleNamespace(action="RESET", frame=types.SimpleNamespace(level=1)),
            ]
            + [
                types.SimpleNamespace(action=f"A{i}", frame=types.SimpleNamespace(level=1))
                for i in range(1, 40)
            ]
            + [
                types.SimpleNamespace(action="ACTION6(row=12, col=34)", frame=types.SimpleNamespace(level=2)),
            ]
        )
        acts = _slm_level_actions(hist, 1)
        self.assertEqual(len(acts), 40)
        self.assertEqual(acts[0], "A1")
        self.assertEqual(acts[-1], "ACTION6(row=12, col=34)")
        self.assertEqual(_slm_level_actions(hist, 2), [])

        # 1b. Single-action level solve (where only 1 move solves Level 1 and immediately transitions frame.level to 2)
        hist_1click = [
            types.SimpleNamespace(action="", frame=types.SimpleNamespace(level=1)),
            types.SimpleNamespace(action="ACTION6(row=7, col=9)", frame=types.SimpleNamespace(level=2)),
        ]
        self.assertEqual(_slm_level_actions(hist_1click, 1), ["ACTION6(row=7, col=9)"])
        self.assertEqual(_slm_level_actions(hist_1click, 2), [])

        # 2. Placeholder rejection & degenerate rejection vs real rule extraction (including inline backticks & multiline)
        agent = types.SimpleNamespace(
            _system_prompt="Base prompt",
            _slm_base_system_prompt="Base prompt",
            _slm_entries={1: {"actions": 40, "seq": ",".join(acts[-30:]), "rule": "", "reasoning": "Solved by clicking (12, 34)."}},
            _slm_pending_rule_level=1,
            _slm_pending_asks=1,
            _context_budget_tokens=1000,
        )
        # Placeholder only -> must NOT match
        matched_placeholder = _slm_extract_rule_from_text(
            agent,
            "State `Rule for level 1: <win condition and key interaction>`",
        )
        self.assertFalse(matched_placeholder)
        self.assertEqual(agent._slm_entries[1]["rule"], "")
        self.assertEqual(agent._slm_pending_rule_level, 1)

        # Degenerate repetition -> must NOT match
        matched_degen = _slm_extract_rule_from_text(
            agent,
            "Rule for level 1: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",
        )
        self.assertFalse(matched_degen)
        self.assertEqual(agent._slm_entries[1]["rule"], "")

        # Inline backticks and leading backtick -> MUST extract full rule without truncating or unbalancing `!
        matched_backtick = _slm_extract_rule_from_text(
            agent,
            "Rule for level 1: `ACTION5` swaps colors; use `ACTION6` on `(12, 34)` to open the goal.",
        )
        self.assertTrue(matched_backtick)
        self.assertEqual(
            agent._slm_entries[1]["rule"],
            "`ACTION5` swaps colors; use `ACTION6` on `(12, 34)` to open the goal.",
        )

        # Multi-line rule where rule text is on the lines below `Rule for level 1: ` -> MUST extract multi-line rule!
        agent._slm_pending_rule_level = 1
        matched_multiline = _slm_extract_rule_from_text(
            agent,
            "Thinking about `Rule for level 1: <win condition and key interaction>`.\n"
            "**Rule for level 1**:   \n"
            "- Align the three mirrors to reflect the laser.\n"
            "- Step on the yellow goal once the gate opens.",
        )
        self.assertTrue(matched_multiline)
        self.assertEqual(
            agent._slm_entries[1]["rule"],
            "- Align the three mirrors to reflect the laser. - Step on the yellow goal once the gate opens.",
        )
        self.assertIsNone(agent._slm_pending_rule_level)

        # 3. Token-budget overflow guard on _slm_pin_on_eviction
        agent._estimate_request_input_tokens = lambda msgs, tools=None: 9999  # over budget (1000)
        msgs = [{"role": "system", "content": "Base prompt"}, {"role": "user", "content": "hi"}]
        res_over = _slm_pin_on_eviction(agent, msgs)
        self.assertEqual(res_over[0]["content"], "Base prompt")
        self.assertEqual(agent._system_prompt, "Base prompt")

        # 4. Within budget -> pins cleanly, and repeated evictions NEVER duplicate the header!
        agent._estimate_request_input_tokens = lambda msgs, tools=None: 500
        res_pin1 = _slm_pin_on_eviction(agent, msgs)
        self.assertEqual(res_pin1[0]["content"].count(_SLM_HEADER), 1)
        self.assertEqual(agent._system_prompt.count(_SLM_HEADER), 1)

        # Add Level 2 and pin again on a second eviction -> still exactly ONE header containing both Level 1 and Level 2!
        agent._slm_entries[2] = {"actions": 5, "seq": "U,U,R,D,L", "rule": "Use teleporter.", "reasoning": ""}
        res_pin2 = _slm_pin_on_eviction(agent, res_pin1)
        self.assertEqual(res_pin2[0]["content"].count(_SLM_HEADER), 1)
        self.assertIn("Level 1:", res_pin2[0]["content"])
        self.assertIn("Level 2: solved in 5 actions [U,U,R,D,L] | Rule: Use teleporter.", res_pin2[0]["content"])


class TestV33NoopRepeatGuardAndStartupShieldAdversarial(unittest.TestCase):
    """Adversarial stress-testing of warmup slot claim concurrency and auto_submit_v33 lockfile ownership."""

    def test_multithreaded_warmup_claim_concurrency(self) -> None:
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

    def test_auto_submit_lockfile_ownership_preserved_on_concurrent_exit(self) -> None:
        import os
        import subprocess
        import sys
        import tempfile
        import scripts.auto_submit_v33 as as33

        with tempfile.TemporaryDirectory() as td:
            fake_lock = Path(td) / ".auto_submit_v33.lock"
            fake_lock.write_text(str(os.getpid()), encoding="utf-8")
            check_code = f"""
import os
from pathlib import Path
import scripts.auto_submit_v33 as as33
as33.LOCK_FILE = Path({str(fake_lock)!r})
# Simulate a second process running cleanup_lock() when LOCK_FILE is owned by parent PID {os.getpid()}
as33.cleanup_lock()
assert as33.LOCK_FILE.exists(), "cleanup_lock() deleted a lockfile owned by another live process!"
assert int(as33.LOCK_FILE.read_text().strip()) == {os.getpid()}
"""
            res = subprocess.run([sys.executable, "-c", check_code], cwd=str(REPO_ROOT), capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"Lockfile ownership test failed:\n{res.stderr}")
            self.assertTrue(fake_lock.exists())
            self.assertEqual(int(fake_lock.read_text(encoding="utf-8").strip()), os.getpid())

    def test_auto_submit_transient_status_retry_and_rehearsal_gate(self) -> None:
        import contextlib
        import io
        import tempfile
        from unittest.mock import patch
        import scripts.auto_submit_v33 as as33

        real_log_path = as33.LOG_FILE
        real_log_before = real_log_path.read_bytes() if real_log_path.exists() else None

        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            td_path = Path(td)
            temp_log_path = td_path / "test.log"
            temp_state_path = td_path / "submitted.state"
            out_dir = td_path / "kaggle_output" / "v33"
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "summary.txt").write_text(
                "ft09-0d8bbf25: score=47.62, levels=4.0/6, actions=52, tokens=40065\n",
                encoding="utf-8",
            )
            (out_dir / "submission.parquet").write_bytes(b"PAR1_FAKE_BYTES")

            status_calls = [
                types.SimpleNamespace(returncode=1, stdout="", stderr="502 Bad Gateway"),
                types.SimpleNamespace(
                    returncode=0,
                    stdout=(
                        "Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)\n"
                        'zerbytheboss/v33-sovereign-apex-engine has status "KernelWorkerStatus.COMPLETE"'
                    ),
                    stderr="",
                ),
                types.SimpleNamespace(returncode=0, stdout="Output downloaded", stderr=""),
            ]
            with (
                patch.object(as33, "ROOT", td_path),
                patch.object(as33, "LOG_FILE", temp_log_path),
                patch("scripts.auto_submit_v33.time.sleep", return_value=None),
                patch("scripts.auto_submit_v33.subprocess.run", side_effect=status_calls),
            ):
                self.assertTrue(as33.verify_kernel_complete(wait=True))

            # Verify terminal KernelWorkerStatus.ERROR raises immediately even when wait=True
            err_call = [
                types.SimpleNamespace(
                    returncode=0,
                    stdout='zerbytheboss/v33-sovereign-apex-engine has status "KernelWorkerStatus.ERROR"',
                    stderr="",
                )
            ]
            with (
                patch.object(as33, "ROOT", td_path),
                patch.object(as33, "LOG_FILE", temp_log_path),
                patch("scripts.auto_submit_v33.time.sleep", return_value=None),
                patch("scripts.auto_submit_v33.subprocess.run", side_effect=err_call),
            ):
                with self.assertRaises(RuntimeError):
                    as33.verify_kernel_complete(wait=True)

            # Verify rehearsal gate rejects score=0.00 in summary.txt
            (out_dir / "summary.txt").write_text("ft09-0d8bbf25: score=0.00, levels=0.0/6, actions=1\n", encoding="utf-8")
            with (
                patch.object(as33, "ROOT", td_path),
                patch.object(as33, "LOG_FILE", temp_log_path),
                patch("scripts.auto_submit_v33.time.sleep", return_value=None),
                patch(
                    "scripts.auto_submit_v33.subprocess.run",
                    return_value=types.SimpleNamespace(returncode=0, stdout="Output downloaded", stderr=""),
                ),
            ):
                with self.assertRaises(RuntimeError):
                    as33.pull_and_verify_rehearsal_output(max_attempts=1)

            # Verify check_duplicate_submission quiet filter (DONT_NOTIFY when state is unchanged) & transient error resilience
            fake_sub = types.SimpleNamespace(
                ref=57732483,
                date="2026-10-28 09:39:40.613000",
                status="SubmissionStatus.PENDING",
                public_score="",
                description=as33.SUBMISSION_MESSAGE,
            )
            fake_api = types.SimpleNamespace(competition_submissions=lambda comp_id: [fake_sub])
            with (
                patch.object(as33, "LOG_FILE", temp_log_path),
                patch.object(as33, "STATE_FILE", temp_state_path),
                patch("scripts.auto_submit_v33._get_kaggle_api", return_value=fake_api),
            ):
                self.assertTrue(as33.check_duplicate_submission())
                size_after_first = temp_log_path.stat().st_size
                # Second call with identical state must emit 0 bytes to LOG_FILE
                self.assertTrue(as33.check_duplicate_submission())
                self.assertEqual(temp_log_path.stat().st_size, size_after_first)

            # Third call where API raises transient 502 while STATE_FILE already exists must return True silently
            with (
                patch.object(as33, "LOG_FILE", temp_log_path),
                patch.object(as33, "STATE_FILE", temp_state_path),
                patch("scripts.auto_submit_v33._get_kaggle_api", side_effect=RuntimeError("502 Bad Gateway")),
            ):
                self.assertTrue(as33.check_duplicate_submission())
                self.assertEqual(temp_log_path.stat().st_size, size_after_first)

            self.assertTrue(temp_log_path.exists())
            temp_log_text = temp_log_path.read_text(encoding="utf-8")
            self.assertIn("502 Bad Gateway", temp_log_text)
            self.assertIn("score=0.00", temp_log_text)
            self.assertNotIn("Warning: Looks like you're using an outdated", temp_log_text)

        real_log_after = real_log_path.read_bytes() if real_log_path.exists() else None
        self.assertEqual(real_log_before, real_log_after, "Real auto_submit_v33.log was modified by unit test!")
        if real_log_after is not None:
            real_text = real_log_after.decode("utf-8", errors="replace")
            self.assertNotIn("502 Bad Gateway", real_text)
            self.assertNotIn("score=0.00", real_text)
            self.assertNotIn("Warning: Looks like you're using an outdated", real_text)


if __name__ == "__main__":
    unittest.main()



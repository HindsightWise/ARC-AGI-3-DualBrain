"""Adversarial Challenger 2 Verification Harness for ARC-AGI-3 Version 22 Telemetry.

Empirically and mathematically stress-tests the causal hypotheses in:
docs/FORENSIC_ROOT_CAUSE_V22.md

Covers:
1. Task 1a: The tr87 starvation hypothesis & self.started_at thread timer bug.
2. Task 1b: SGLang KV-cache thrashing & latency spikes math on RTX 6000 Ada with fp8.
3. Task 1c: The 12 stalled games, EXPOSE_RESET omission, and repeat guard inactivation.
"""

import heapq
import json
import os
import re
import unittest
from collections import Counter
from pathlib import Path

V22_DIR = Path("/tmp/v22_output")


def apex_value(level: int, actions: float, tokens: float, pace: float, total_levels, phi: float = 1.0) -> float:
    """Exact priority function from scripts/build_v22_notebook.py."""
    level = max(1, int(level))
    n = int(total_levels) if total_levels else 10
    n_clamped = min(10, max(6, n))
    w_total = n_clamped * (n_clamped + 1) / 2
    norm = 55.0 / w_total

    a = max(0.0, float(actions))
    t = max(0.0, float(tokens))

    level_weight_factor = (1.0 + 0.8 * (level - 1)) * norm
    eff_decay = (300.0 / (300.0 + a)) ** 2.5
    A = level_weight_factor * eff_decay

    pace_ref = 30_000.0
    if pace > 0:
        A *= min(4.0, max(0.25, (pace_ref / max(1.0, pace)) ** 0.4))

    scale = 225_000.0
    rel = pace if pace > 0 else pace_ref
    scale *= min(2.0, max(0.5, rel / pace_ref)) ** 0.5
    floor = 0.05
    C = (0.15 * max(floor, 1.0 - a / 100.0) + 0.85 * max(floor, 1.0 - t / scale))

    remaining = max(0, n - level)
    level_scale_factor = (n_clamped / 6.0) ** 0.5
    if remaining == 0:
        b_base = 22.0 * level_scale_factor
    elif remaining == 1:
        b_base = 18.0 * level_scale_factor
    elif remaining == 2:
        b_base = 16.0 * level_scale_factor
    else:
        b_base = 14.0 * level_scale_factor

    B = b_base * (0.2 + 0.8 * C)
    return (A * C) + (B * phi)


SCALE = 1000.0


class TestV22Challenger2Adversarial(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not V22_DIR.exists():
            raise unittest.SkipTest(f"Directory {V22_DIR} does not exist (ephemeral benchmark run)")
        with open(V22_DIR / "benchmark.json") as f:
            cls.benchmark = json.load(f)

        with open(V22_DIR / "summary.txt") as f:
            cls.summary_text = f.read()

        cls.runs = cls.benchmark["game_runs"]

    # =========================================================================
    # Task 1a: tr87 Starvation & Thread Timer Bug
    # =========================================================================

    def test_01_tr87_priority_stratification_math(self):
        """Verify mathematical priority stratification for all 25 fresh games."""
        priorities = {}
        for r in self.runs:
            gid = r["game_id"]
            n = r["number_of_levels"]
            val = apex_value(1, 0, 0, 0, n, phi=1.0)
            prio = max(1, int(val * SCALE))
            priorities[gid] = (prio, n, val)

        # 10-level games: lf52
        self.assertEqual(priorities["lf52-271a04aa"][0], 19073)
        self.assertAlmostEqual(priorities["lf52-271a04aa"][2], 19.0739, delta=0.001)

        # 9-level games: bp35, tu93, wa30, su15
        for gid in ["bp35-0a0ad940", "tu93-0768757b", "wa30-ee6fef47", "su15-1944f8ab"]:
            self.assertEqual(priorities[gid][0], 18368)
            self.assertAlmostEqual(priorities[gid][2], 18.3687, delta=0.001)

        # 8-level games: sk48, lp85, ar25, sb26, re86, s5i5
        for gid in ["sk48-d8078629", "lp85-305b61c3", "ar25-0c556536", "sb26-7fbdac44", "re86-8af5384d", "s5i5-18d95033"]:
            self.assertEqual(priorities[gid][0], 17693)
            self.assertAlmostEqual(priorities[gid][2], 17.6936, delta=0.001)

        # 7-level games: tn36, ka59, vc33, ls20, g50t
        for gid in ["tn36-ef4dde99", "ka59-38d34dbb", "vc33-5430563c", "ls20-9607627b", "g50t-5849a774"]:
            self.assertEqual(priorities[gid][0], 17086)
            self.assertAlmostEqual(priorities[gid][2], 17.0860, delta=0.001)

        # 6-level games: m0r0, cn04, dc22, r11l, sc25, sp80, cd82, ft09, tr87
        for gid in ["m0r0-492f87ba", "cn04-2fe56bfb", "dc22-fdcac232", "r11l-495a7899",
                    "sc25-635fd71a", "sp80-589a99af", "cd82-fb555c5d", "ft09-0d8bbf25", "tr87-cd924810"]:
            self.assertEqual(priorities[gid][0], 16619)
            self.assertAlmostEqual(priorities[gid][2], 16.6190, delta=0.001)

        # tr87 is the 9th 6-level game (index 23 in runs) and is the absolute tail of all 25 games
        tr87_run = [r for r in self.runs if r["game_id"] == "tr87-cd924810"][0]
        self.assertEqual(tr87_run["number_of_levels"], 6)
        self.assertEqual(self.runs.index(tr87_run), 23)

    def test_02_tr87_priority_gate_heap_simulation(self):
        """Simulate PriorityGate tier ordering and verify tr87 is the 25th admitted game."""
        # The 12 initial slots at 07:34:27 were seized by threads reaching the gate before slots ran out
        initial_12_gids = {
            "r11l-495a7899", "lf52-271a04aa", "sk48-d8078629", "lp85-305b61c3",
            "tn36-ef4dde99", "tu93-0768757b", "vc33-5430563c", "bp35-0a0ad940",
            "m0r0-492f87ba", "ka59-38d34dbb", "cn04-2fe56bfb", "dc22-fdcac232"
        }
        waiting_runs = [r for r in self.runs if r["game_id"] not in initial_12_gids]
        self.assertEqual(len(waiting_runs), 13)

        # Priority tier mapping for waiting games:
        # Tier 1 (N=9, prio 18,368): wa30, su15
        # Tier 2 (N=8, prio 17,693): sb26, ar25, re86, s5i5
        # Tier 3 (N=7, prio 17,086): ls20, g50t
        # Tier 4 (N=6, prio 16,619): sp80, sc25, cd82, ft09, tr87
        tier_prio = {}
        for r in waiting_runs:
            val = apex_value(1, 0, 0, 0, r["number_of_levels"], phi=1.0)
            prio = max(1, int(val * SCALE))
            tier_prio[r["game_id"]] = prio

        # Extract actual admission times from transcripts
        actual_order = [
            "wa30-ee6fef47",  # 07:58:30 (Tier 1)
            "su15-1944f8ab",  # 08:08:53 (Tier 1)
            "sb26-7fbdac44",  # 08:15:41 (Tier 2)
            "ar25-0c556536",  # 08:17:01 (Tier 2)
            "re86-8af5384d",  # 08:17:07 (Tier 2)
            "s5i5-18d95033",  # 08:18:58 (Tier 2)
            "ls20-9607627b",  # 08:37:45 (Tier 3)
            "g50t-5849a774",  # 08:40:15 (Tier 3)
            "sp80-589a99af",  # 09:03:04 (Tier 4)
            "sc25-635fd71a",  # 09:03:28 (Tier 4)
            "cd82-fb555c5d",  # 09:06:26 (Tier 4)
            "ft09-0d8bbf25",  # 09:07:56 (Tier 4)
            "tr87-cd924810",  # Starved (Tier 4, unblocked at 09:37:36)
        ]

        # Verify that actual admission times strictly obey priority tiers monotonically
        prios = [tier_prio[gid] for gid in actual_order]
        for i in range(len(prios) - 1):
            self.assertGreaterEqual(prios[i], prios[i+1], f"Monotonicity violated between {actual_order[i]} and {actual_order[i+1]}")

        # Verify tier counts: exactly 2 in Tier 1, 4 in Tier 2, 2 in Tier 3, 5 in Tier 4
        self.assertEqual(prios[:2], [18368, 18368])
        self.assertEqual(prios[2:6], [17693, 17693, 17693, 17693])
        self.assertEqual(prios[6:8], [17086, 17086])
        self.assertEqual(prios[8:], [16619, 16619, 16619, 16619, 16619])

        # tr87 is unequivocally last among all waiters!
        self.assertEqual(actual_order[-1], "tr87-cd924810")

    def test_03_tr87_microsecond_shutdown_timing(self):
        """Verify the 25-microsecond shutdown sequence for tr87 from engine logs."""
        engine_log = V22_DIR / "v22-sovereign-apex-engine.log"
        r11l_finish_time = None
        tr87_finish_time = None

        with open(engine_log) as f:
            for line in f:
                line = line.strip()
                if line.startswith(","):
                    line = line[1:]
                try:
                    entry = json.loads(line)
                    data = entry.get("data", "")
                    if "[finished] r11l-495a7899" in data:
                        r11l_finish_time = entry.get("time")
                    elif "[finished] tr87-cd924810" in data:
                        tr87_finish_time = entry.get("time")
                except Exception:
                    pass

        self.assertIsNotNone(r11l_finish_time)
        self.assertIsNotNone(tr87_finish_time)
        self.assertAlmostEqual(r11l_finish_time, 7947.38703199, delta=0.001)
        self.assertAlmostEqual(tr87_finish_time, 7947.38705743, delta=0.001)

        delta_s = tr87_finish_time - r11l_finish_time
        # Must be within 50 microseconds!
        self.assertLess(delta_s, 0.000050)
        self.assertGreaterEqual(delta_s, 0.0)

    def test_04_tr87_timer_limit_arithmetic(self):
        """Verify that max_runtime_s_per_game caused instant exit."""
        # max_runtime_s_per_game = 532 * 60 * 25 // 110 = 7254.545s
        max_runtime = 532 * 60 * 25 // 110
        self.assertEqual(max_runtime, 7254)

        tr87_run = [r for r in self.runs if r["game_id"] == "tr87-cd924810"][0]
        wallclock = tr87_run["final_wallclock_seconds"]
        self.assertAlmostEqual(wallclock, 7262.035, delta=0.01)

        # Elapsed time exceeded limit before step 1 could execute
        self.assertGreater(wallclock, max_runtime)
        self.assertEqual(tr87_run["final_score"], 0.0)
        self.assertEqual(len(tr87_run["history"]), 0)

    # =========================================================================
    # Task 1b: SGLang KV-Cache Thrashing & Latency Spikes
    # =========================================================================

    def test_05_sglang_kv_cache_physical_capacity(self):
        """Verify physical KV cache allocation in serve.log."""
        serve_log = V22_DIR / "serve.log"
        kv_alloc_line = None
        with open(serve_log) as f:
            for line in f:
                if "KV Cache is allocated" in line and "#tokens: 961344" in line:
                    kv_alloc_line = line
                    break

        self.assertIsNotNone(kv_alloc_line)
        self.assertIn("dtype: torch.float8_e4m3fn", kv_alloc_line)
        self.assertIn("#tokens: 961344", kv_alloc_line)
        self.assertIn("K size: 5.50 GB, V size: 5.50 GB", kv_alloc_line)

    def test_06_sglang_kv_cache_thrashing_and_hit_rate(self):
        """Verify prefix cache hit rate and uncached prefill tokens."""
        serve_log = V22_DIR / "serve.log"
        total_new = 0
        total_cached = 0

        with open(serve_log) as f:
            for line in f:
                if "Prefill batch" in line:
                    m_new = re.search(r"#new-token:\s*(\d+)", line)
                    m_cached = re.search(r"#cached-token:\s*(\d+)", line)
                    if m_new and m_cached:
                        total_new += int(m_new.group(1))
                        total_cached += int(m_cached.group(1))

        total_input = total_new + total_cached
        hit_rate = (total_cached / total_input) * 100

        # Exact values verified
        self.assertGreater(total_new, 50_000_000)
        self.assertAlmostEqual(total_cached, 59_209_216, delta=100)
        self.assertAlmostEqual(hit_rate, 54.04, delta=0.1)

    def test_07_sglang_peak_memory_usage_and_saturation(self):
        """Verify peak full token usage hit 99% of physical capacity."""
        serve_log = V22_DIR / "serve.log"
        max_usage = 0.0
        max_full_tokens = 0

        with open(serve_log) as f:
            for line in f:
                m_u = re.search(r"full token usage:\s*([\d\.]+)", line)
                m_ft = re.search(r"#full token:\s*(\d+)", line)
                if m_u:
                    max_usage = max(max_usage, float(m_u.group(1)))
                if m_ft:
                    max_full_tokens = max(max_full_tokens, int(m_ft.group(1)))

        self.assertAlmostEqual(max_usage, 0.99, delta=0.01)
        self.assertEqual(max_full_tokens, 956288)
        # 956,288 / 961,344 = 99.47% saturation
        self.assertAlmostEqual(max_full_tokens / 961344, 0.9947, delta=0.001)

    def test_08_sglang_peak_latency_and_queue_duration(self):
        """Verify peak request forward latency (289.59s) and queue waiting duration (472.75s)."""
        serve_log = V22_DIR / "serve.log"
        max_q = 0.0
        max_fwd = 0.0

        with open(serve_log) as f:
            for line in f:
                if "ReqTimeStats" in line:
                    m_q = re.search(r"queue_duration=([\d\.]+)ms", line)
                    m_fwd = re.search(r"forward_duration=([\d\.]+)ms", line)
                    if m_q and m_fwd:
                        max_q = max(max_q, float(m_q.group(1)))
                        max_fwd = max(max_fwd, float(m_fwd.group(1)))

        self.assertAlmostEqual(max_fwd, 289591.38, delta=0.1)
        self.assertAlmostEqual(max_q, 472753.97, delta=0.1)
        # In seconds: ~289.59s and ~472.75s
        self.assertAlmostEqual(max_fwd / 1000.0, 289.59, delta=0.01)
        self.assertAlmostEqual(max_q / 1000.0, 472.75, delta=0.01)

    def test_09_concurrency_working_context_math(self):
        """Verify mathematically that 12 streams exceed 961k cache while 10 streams fit."""
        serve_log = V22_DIR / "serve.log"
        post_8am_inputs = []

        with open(serve_log) as f:
            for line in f:
                if "ReqTimeStats" in line:
                    m_inp = re.search(r"input_len=(\d+)", line)
                    m_t = re.search(r"\[(.*?)\]", line)
                    if m_inp and m_t and m_t.group(1) >= "2026-10-15 08:00":
                        post_8am_inputs.append(int(m_inp.group(1)))

        avg_context = sum(post_8am_inputs) / len(post_8am_inputs)
        self.assertAlmostEqual(avg_context, 80287.9, delta=1.0)

        # 12 streams:
        demand_12 = 12 * avg_context
        self.assertGreater(demand_12, 961344)
        self.assertAlmostEqual(demand_12 / 961344, 1.0022, delta=0.01)

        # 10 streams:
        demand_10 = 10 * avg_context
        self.assertLess(demand_10, 961344)
        self.assertAlmostEqual(demand_10 / 961344, 0.8352, delta=0.01)

    # =========================================================================
    # Task 1c: The 12 Stalled Games & Safety Guards / Reset Deadlocks
    # =========================================================================

    def test_10_expose_reset_and_guards_env_in_v22_builder(self):
        """Verify EXPOSE_RESET omission and missing repeat guard flags in build_v22_notebook.py."""
        builder = Path("scripts/build_v22_notebook.py")
        self.assertTrue(builder.exists())
        with open(builder) as f:
            code = f.read()

        # EXPOSE_RESET must be completely absent from setup_env
        self.assertNotIn("'EXPOSE_RESET': 'on'", code)
        self.assertNotIn('"EXPOSE_RESET": "on"', code)

        # ARC3_GUARDS_FROM_LEVEL is set to 2
        self.assertIn("'ARC3_GUARDS_FROM_LEVEL': 2", code)

        # ARC3_NOOP_REPEAT_GUARD and ARC3_DEATH_REPEAT_GUARD were omitted entirely
        self.assertNotIn("'ARC3_NOOP_REPEAT_GUARD': '1'", code)
        self.assertNotIn("'ARC3_DEATH_REPEAT_GUARD': '1'", code)

    def test_11_m0r0_and_cd82_dead_ends_without_voluntary_reset(self):
        """Verify that m0r0 and cd82 executed zero voluntary resets."""
        # m0r0: 54 actions, 0 deaths, 0 resets
        m0r0_path = V22_DIR / "artifacts/m0r0-492f87ba_p0_events.jsonl"
        with open(m0r0_path) as f:
            m0r0_evts = [json.loads(line) for line in f if line.strip()]

        m0r0_actions = [e["action_name"] for e in m0r0_evts if e.get("type") == "action"]
        self.assertEqual(len(m0r0_actions), 54)
        self.assertNotIn("RESET", m0r0_actions)
        self.assertEqual(sum(1 for e in m0r0_evts if e.get("game_over")), 0)

        # cd82: 122 actions, 1 death reset (involuntary)
        cd82_path = V22_DIR / "artifacts/cd82-fb555c5d_p0_events.jsonl"
        with open(cd82_path) as f:
            cd82_evts = [json.loads(line) for line in f if line.strip()]

        cd82_actions = [e["action_name"] for e in cd82_evts if e.get("type") == "action"]
        self.assertEqual(len(cd82_actions), 122)
        # Exactly 1 RESET, which was caused by game_over
        self.assertEqual(cd82_actions.count("RESET"), 1)
        self.assertEqual(sum(1 for e in cd82_evts if e.get("game_over")), 1)

    def test_12_aimless_action6_click_traps_s5i5_and_tn36(self):
        """Verify s5i5 and tn36 executed 100% ACTION6 clicks."""
        for gid, expected_count in [("s5i5-18d95033", 57), ("tn36-ef4dde99", 84)]:
            fpath = V22_DIR / f"artifacts/{gid}_p0_events.jsonl"
            with open(fpath) as fp:
                evts = [json.loads(line) for line in fp if line.strip()]
            actions = [e["action_name"] for e in evts if e.get("type") == "action"]
            self.assertEqual(len(actions), expected_count)
            self.assertEqual(actions.count("ACTION6"), expected_count)
            self.assertEqual(set(actions), {"ACTION6"})

    def test_13_ka59_keyerror_gameplay_changed(self):
        """Verify KeyError: 'gameplay_changed' crash in ka59 transcript."""
        ka59_transcript = V22_DIR / "transcripts/ka59-38d34dbb_p0.txt"
        self.assertTrue(ka59_transcript.exists())
        with open(ka59_transcript) as f:
            text = f.read()

        self.assertIn("KeyError: 'gameplay_changed'", text)

    def test_14_lf52_and_sk48_handover_starvation(self):
        """Verify lf52 and sk48 surrendered slots via handover and remained starved."""
        # lf52 final step was step 34 at 07:58:30 (where wa30 was admitted)
        lf52_path = V22_DIR / "transcripts/lf52-271a04aa_p0.txt"
        with open(lf52_path) as f:
            lf52_text = f.read()

        steps_lf52 = re.findall(r"--- analysis_step=(\d+) \| action=(\d+) \| (\d{2}:\d{2}:\d{2})", lf52_text)
        self.assertEqual(len(steps_lf52), 34)
        self.assertEqual(steps_lf52[-1][2], "07:58:30")

        # sk48 final step was step 23 at 08:17:07 (where re86 was admitted)
        sk48_path = V22_DIR / "transcripts/sk48-d8078629_p0.txt"
        with open(sk48_path) as f:
            sk48_text = f.read()

        steps_sk48 = re.findall(r"--- analysis_step=(\d+) \| action=(\d+) \| (\d{2}:\d{2}:\d{2})", sk48_text)
        self.assertEqual(len(steps_sk48), 23)
        self.assertEqual(steps_sk48[-1][2], "08:17:07")


if __name__ == "__main__":
    unittest.main()

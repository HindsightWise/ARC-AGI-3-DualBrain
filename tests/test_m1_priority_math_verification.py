"""Empirical verification and adversarial stress-testing of priority scheduling mathematics.

Validates the forensic claims in docs/FORENSIC_ROOT_CAUSE_V21.md comparing
shiiin9's raw D' expected-value formula against Sujanmajhi's Apex Victory Lap formula.
"""

from __future__ import annotations

import math
import unittest

D_PRIME_PARAMS = {
    "level_pow": 0.5,
    "eff_const": 300.0,
    "eff_pow": 2.5,
    "pace_pow": 0.4,
    "pace_ref": 30_000.0,
    "c_action": 0.1,
    "action_scale": 115.0,
    "c_token": 0.9,
    "token_scale": 225_000.0,
    "lin_floor": 0.1,
    "rel_pow": 0.5,
    "b1": 10.0,
    "b2": 14.0,
    "b3": 16.0,
    "fade": 0.4,
}


def _levels(total) -> int | None:
    try:
        n = int(total)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def raw_d_parts(
    level: int,
    actions: float,
    tokens: float,
    pace: float,
    total_levels,
    params: dict = D_PRIME_PARAMS,
) -> tuple[float, float, float, float]:
    """Computes (A, C, A*C, B) for shiiin9 raw D' priority."""
    level = max(1, int(level))
    n = _levels(total_levels) or 10
    n_clamped = min(10, max(6, n))
    norm = 55 / (n_clamped * (n_clamped + 1) / 2)
    a = max(0.0, float(actions))
    t = max(0.0, float(tokens))

    A = (1.0 + params["level_pow"] * (level - 1)) * norm
    A *= (params["eff_const"] / (params["eff_const"] + a)) ** params["eff_pow"]
    if pace > 0:
        A *= min(4.0, max(0.25, (params["pace_ref"] / max(1.0, pace)) ** params["pace_pow"]))

    scale = params["token_scale"]
    rel = pace if pace > 0 else params["pace_ref"]
    scale *= min(2.0, max(0.5, rel / params["pace_ref"])) ** params["rel_pow"]
    floor = params["lin_floor"]
    C = (
        params["c_action"] * max(floor, 1.0 - a / params["action_scale"])
        + params["c_token"] * max(floor, 1.0 - t / scale)
    )
    remaining = max(0, n - level)
    B = (0.0, params["b1"], params["b2"], params["b3"])[min(remaining, 3)]
    return A, C, A * C, B


def raw_d_value(level: int, actions: float, tokens: float, pace: float, total_levels, phi: float = 1.0) -> float:
    A, C, AC, B = raw_d_parts(level, actions, tokens, pace, total_levels)
    return AC + B * phi


def victory_lap_parts(
    level: int,
    actions: float,
    tokens: float,
    pace: float,
    total_levels,
) -> tuple[float, float, float, float]:
    """Computes (A, C, A*C, B) for Sujanmajhi's Apex Victory Lap formula."""
    level = max(1, int(level))
    n = _levels(total_levels) or 10
    n_clamped = min(10, max(6, n))
    w_total = n_clamped * (n_clamped + 1) / 2
    norm = 55 / w_total

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
    C = 0.15 * max(floor, 1.0 - a / 100.0) + 0.85 * max(floor, 1.0 - t / scale)

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
    return A, C, A * C, B


def victory_lap_value(level: int, actions: float, tokens: float, pace: float, total_levels, phi: float = 1.0) -> float:
    A, C, AC, B = victory_lap_parts(level, actions, tokens, pace, total_levels)
    return AC + B * phi


class TestPriorityMathVerification(unittest.TestCase):
    """Rigorous empirical test suite for Milestone 1 Priority Mathematical Claims."""

    def test_benchmark_numerical_targets(self):
        """Verify exact numerical values at specified benchmark points."""
        scale = 1000

        # Point 1: Unstarted Level 1 games (a=0, t=0, l=1)
        for N, exp_sh_b, exp_sh_val_range, exp_su_val_range in [
            (6, 16.0, (18.5, 18.7), (16.5, 16.7)),
            (8, 16.0, (17.5, 17.6), (17.6, 17.8)),
            (10, 16.0, (16.9, 17.1), (19.0, 19.2)),
        ]:
            sh_val = raw_d_value(1, 0, 0, 0, N)
            su_val = victory_lap_value(1, 0, 0, 0, N)
            _, _, _, sh_B = raw_d_parts(1, 0, 0, 0, N)

            self.assertEqual(sh_B, exp_sh_b)
            self.assertTrue(exp_sh_val_range[0] <= sh_val <= exp_sh_val_range[1])
            self.assertTrue(exp_su_val_range[0] <= su_val <= exp_su_val_range[1])

        # Point 2: Mid-game Level 4/8 (a=20, t=5000, l=4)
        sh_mid = raw_d_value(4, 20, 5000, 0, 8)
        su_mid = victory_lap_value(4, 20, 5000, 0, 8)
        self.assertAlmostEqual(sh_mid, 19.129, places=2)
        self.assertAlmostEqual(su_mid, 19.738, places=2)

        # Point 3: Final-level game Level 8/8 (a=20, t=10000, l=8)
        sh_fin = raw_d_value(8, 20, 10000, 0, 8)
        su_fin = victory_lap_value(8, 20, 10000, 0, 8)
        _, _, _, sh_fin_B = raw_d_parts(8, 20, 10000, 0, 8)

        # Verify B collapses to exactly 0.0 in raw D'
        self.assertEqual(sh_fin_B, 0.0)
        self.assertAlmostEqual(sh_fin, 5.515, places=2)
        self.assertAlmostEqual(su_fin, 32.025, places=2)

    def test_raw_d_prime_starvation_collapse(self):
        """Verify raw D' experiences final-level priority collapse and is starved by Level 1 games."""
        for N in (6, 8, 10):
            fresh_prio = raw_d_value(1, 0, 0, 0, N)
            final_prio = raw_d_value(N, 20, 10000, 0, N)

            # Assert final-level priority is strictly LESS than fresh Level 1 priority
            self.assertLess(final_prio, fresh_prio)

            # Ratio of collapse: final priority is less than 40% of fresh level 1 priority!
            ratio = final_prio / fresh_prio
            self.assertLess(ratio, 0.45, f"Expected ratio < 0.45 for N={N}, got {ratio:.3f}")

    def test_sujanmajhi_victory_lap_dominance(self):
        """Verify Sujanmajhi's Victory Lap formula maintains final-level priority strictly above fresh games."""
        for N in (6, 8, 10):
            fresh_prio = victory_lap_value(1, 0, 0, 0, N)
            final_prio = victory_lap_value(N, 20, 10000, 0, N)

            # Assert final-level priority is strictly GREATER than fresh Level 1 priority
            self.assertGreater(final_prio, fresh_prio)

            # Ratio of dominance: final priority is >= 1.6x fresh level 1 priority
            ratio = final_prio / fresh_prio
            self.assertGreater(ratio, 1.6, f"Expected ratio > 1.6 for N={N}, got {ratio:.3f}")

    def test_stall_decay_coupling_adversarial(self):
        """Verify stall decay coupling: stuck games gracefully yield slots while active games dominate."""
        # Progressing final level (a=20, t=10000)
        active_final = victory_lap_value(8, 20, 10000, 30000, 8)
        # Hopelessly stuck final level (a=150, t=250000)
        stuck_final = victory_lap_value(8, 150, 250000, 30000, 8)

        # Active final level has huge priority (~32.0)
        self.assertGreater(active_final, 30.0)
        # Stuck final level decays by > 75%
        self.assertLess(stuck_final, 8.0)
        self.assertLess(stuck_final, active_final * 0.25)

    def test_tail_fade_insensitivity(self):
        """Verify Sujanmajhi formula maintains final level dominance across all phi in [0, 1]."""
        for phi in (1.0, 0.75, 0.5, 0.25, 0.0):
            fresh = victory_lap_value(1, 0, 0, 0, 8, phi=phi)
            final = victory_lap_value(8, 20, 10000, 30000, 8, phi=phi)
            self.assertGreater(final, fresh, f"Failed at phi={phi}: final={final} <= fresh={fresh}")


if __name__ == "__main__":
    unittest.main()

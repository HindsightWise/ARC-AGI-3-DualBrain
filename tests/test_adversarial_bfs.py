"""
Adversarial Verification and Stress-Testing Harness for ARC-AGI-3 Version 3.
Challenger 1: State-Space BFS & Operator Transition Adversarial Harness.

Empirical verification of:
1. Canonical Shape Invariance under 90°, 180°, 270° rotations & 2x HUD downsampling.
2. Operator Pad Modulo Cycling Arithmetic: mod 6 (shape), mod 4 (color), mod 4 (rotation).
3. Receptacle Blocking Invariant: strictly forbids traversing unmatched goal receptacles.
4. BFS Minimal Action Sequence Optimality, Re-trigger Step-off, and Deadlock Gracefulness.
5. Independent Oracle Equivalence Verification.
"""
from __future__ import annotations

import collections
import heapq
import os
import sys
import time
import unittest
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VENDOR = os.path.join(ROOT, "vendor", "ARC-AGI-3-Agents")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if VENDOR not in sys.path:
    sys.path.insert(0, VENDOR)

from arcengine import GameAction
from agent.my_agent import (
    BlockConfiguration,
    TemplatePerception,
    ConfigurationPlanner,
    OperatorPadDetector,
)


class TestCanonicalShapeInvariance(unittest.TestCase):
    """Adversarial stress-testing of canonical shape invariance and HUD extraction."""

    def test_all_96_configurations_3x3_and_6x6(self):
        """Verify all 6 shapes x 4 colors x 4 rotations match unambiguously in 3x3 and 6x6."""
        total_tested = 0
        seen_binary_hashes = set()

        for s_id, base_shp in TemplatePerception.SHAPES.items():
            for r_idx, deg in enumerate(TemplatePerception.ROTATIONS):
                rot_shp = np.rot90(base_shp, -deg // 90)
                shape_hash = rot_shp.tobytes()
                self.assertNotIn(
                    shape_hash,
                    seen_binary_hashes,
                    f"Shape collision detected for s_id={s_id}, deg={deg}!",
                )
                seen_binary_hashes.add(shape_hash)

                for c_idx, fg_col in enumerate(TemplatePerception.PALETTE):
                    total_tested += 1

                    # 1. 3x3 patch
                    patch_3x3 = np.where(rot_shp, fg_col, 5).astype(np.int8)
                    cfg_3x3 = TemplatePerception.match_sprite(patch_3x3, bg_color=5)
                    self.assertIsNotNone(
                        cfg_3x3,
                        f"match_sprite returned None for 3x3: s={s_id}, r={r_idx}, c={c_idx}",
                    )
                    self.assertEqual(cfg_3x3.shape_id, s_id)
                    self.assertEqual(cfg_3x3.rotation_idx, r_idx)
                    self.assertEqual(cfg_3x3.color_idx, c_idx)
                    self.assertEqual(cfg_3x3.raw_color, fg_col)
                    self.assertEqual(cfg_3x3.rotation_deg, deg)

                    # 2. 6x6 patch (2x downsampled HUD block)
                    patch_6x6 = np.kron(patch_3x3, np.ones((2, 2), dtype=np.int8))
                    cfg_6x6 = TemplatePerception.match_sprite(patch_6x6, bg_color=5)
                    self.assertIsNotNone(
                        cfg_6x6,
                        f"match_sprite returned None for 6x6: s={s_id}, r={r_idx}, c={c_idx}",
                    )
                    self.assertEqual(cfg_6x6.shape_id, s_id)
                    self.assertEqual(cfg_6x6.rotation_idx, r_idx)
                    self.assertEqual(cfg_6x6.color_idx, c_idx)
                    self.assertEqual(cfg_6x6.raw_color, fg_col)
                    self.assertEqual(cfg_6x6.rotation_deg, deg)

        self.assertEqual(total_tested, 96)
        self.assertEqual(len(seen_binary_hashes), 24)

    def test_adversarial_corrupted_and_boundary_patches(self):
        """Verify robust rejection of malformed, noisy, or non-matching patches."""
        for bad_shape in [(0, 0), (1, 1), (2, 2), (3, 4), (4, 3), (5, 5), (6, 5), (7, 7)]:
            patch = np.zeros(bad_shape, dtype=np.int8)
            self.assertIsNone(TemplatePerception.match_sprite(patch))

        self.assertIsNone(TemplatePerception.match_sprite(np.full((3, 3), 5, dtype=np.int8)))
        self.assertIsNone(TemplatePerception.match_sprite(np.full((6, 6), 5, dtype=np.int8)))

        for invalid_col in [0, 1, 2, 3, 4, 6, 7, 10, 11, 13, 15]:
            p = np.full((3, 3), 5, dtype=np.int8)
            p[0, 0] = invalid_col
            self.assertIsNone(TemplatePerception.match_sprite(p))

        p_single = np.full((3, 3), 5, dtype=np.int8)
        p_single[1, 1] = 12
        self.assertIsNone(TemplatePerception.match_sprite(p_single))

        base = TemplatePerception.SHAPES[0]
        inverted = np.where(~base, 12, 5).astype(np.int8)
        self.assertIsNone(TemplatePerception.match_sprite(inverted))

    def test_hud_extraction_boundary_clipping(self):
        """Verify extract_hud_configuration on small grids and clean grids."""
        small_grid = np.zeros((50, 50), dtype=np.int8)
        self.assertIsNone(TemplatePerception.extract_hud_configuration(small_grid))

        grid = np.zeros((64, 64), dtype=np.int8)
        base_3x3 = np.where(TemplatePerception.SHAPES[2], 14, 5).astype(np.int8)
        hud_6x6 = np.kron(base_3x3, np.ones((2, 2), dtype=np.int8))
        grid[55:61, 3:9] = hud_6x6

        cfg = TemplatePerception.extract_hud_configuration(grid)
        self.assertIsNotNone(cfg)
        self.assertEqual(cfg.shape_id, 2)
        self.assertEqual(cfg.color_idx, 2)
        self.assertEqual(cfg.rotation_idx, 0)


class TestOperatorModuloTransitions(unittest.TestCase):
    """Adversarial stress-testing of operator pad cyclic group arithmetic."""

    def test_cyclic_arithmetic_modulo_properties(self):
        """Verify that pad transitions follow exact cyclic group rules: Z6 x Z4 x Z4."""
        passable = np.ones((12, 12), dtype=bool)
        legal_actions = {GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4}

        # 1. Shape Pad: Z6 cycling
        for s0 in range(6):
            for k in range(1, 7):
                expected_s = (s0 + k) % 6
                pads = {(1, 0): "shape"}
                goal_slots = {(2, 0): BlockConfiguration(shape_id=expected_s, color_idx=0, rotation_idx=0)}
                start_cfg = BlockConfiguration(shape_id=s0, color_idx=0, rotation_idx=0)

                path = ConfigurationPlanner.plan_unified_bfs(
                    start_tile=(0, 0),
                    start_cfg=start_cfg,
                    goal_slots=goal_slots,
                    pads=pads,
                    passable=passable,
                    legal_actions=legal_actions,
                )
                self.assertTrue(len(path) > 0, f"Failed to plan path for s0={s0}, k={k}")

                cur_x, cur_y = 0, 0
                cur_s = s0
                for act in path:
                    dx, dy = ConfigurationPlanner.ACTIONS_MAP[[a[0] for a in ConfigurationPlanner.ACTIONS_MAP].index(act)][1:]
                    cur_x += dx
                    cur_y += dy
                    if (cur_x, cur_y) in pads and pads[(cur_x, cur_y)] == "shape":
                        cur_s = (cur_s + 1) % 6

                self.assertEqual(cur_s, expected_s)
                self.assertEqual((cur_x, cur_y), (2, 0))

        # 2. Color Pad: Z4 cycling
        for c0 in range(4):
            for k in range(1, 5):
                expected_c = (c0 + k) % 4
                pads = {(1, 0): "color"}
                goal_slots = {(2, 0): BlockConfiguration(shape_id=0, color_idx=expected_c, rotation_idx=0)}
                start_cfg = BlockConfiguration(shape_id=0, color_idx=c0, rotation_idx=0)

                path = ConfigurationPlanner.plan_unified_bfs(
                    start_tile=(0, 0),
                    start_cfg=start_cfg,
                    goal_slots=goal_slots,
                    pads=pads,
                    passable=passable,
                    legal_actions=legal_actions,
                )
                self.assertTrue(len(path) > 0)
                cur_x, cur_y, cur_c = 0, 0, c0
                for act in path:
                    dx, dy = ConfigurationPlanner.ACTIONS_MAP[[a[0] for a in ConfigurationPlanner.ACTIONS_MAP].index(act)][1:]
                    cur_x += dx
                    cur_y += dy
                    if (cur_x, cur_y) in pads and pads[(cur_x, cur_y)] == "color":
                        cur_c = (cur_c + 1) % 4
                self.assertEqual(cur_c, expected_c)

        # 3. Rotation Pad: Z4 cycling
        for r0 in range(4):
            for k in range(1, 5):
                expected_r = (r0 + k) % 4
                pads = {(1, 0): "rot"}
                goal_slots = {(2, 0): BlockConfiguration(shape_id=0, color_idx=0, rotation_idx=expected_r)}
                start_cfg = BlockConfiguration(shape_id=0, color_idx=0, rotation_idx=r0)

                path = ConfigurationPlanner.plan_unified_bfs(
                    start_tile=(0, 0),
                    start_cfg=start_cfg,
                    goal_slots=goal_slots,
                    pads=pads,
                    passable=passable,
                    legal_actions=legal_actions,
                )
                self.assertTrue(len(path) > 0)
                cur_x, cur_y, cur_r = 0, 0, r0
                for act in path:
                    dx, dy = ConfigurationPlanner.ACTIONS_MAP[[a[0] for a in ConfigurationPlanner.ACTIONS_MAP].index(act)][1:]
                    cur_x += dx
                    cur_y += dy
                    if (cur_x, cur_y) in pads and pads[(cur_x, cur_y)] == "rot":
                        cur_r = (cur_r + 1) % 4
                self.assertEqual(cur_r, expected_r)


class TestReceptacleBlockingInvariant(unittest.TestCase):
    """Adversarial testing of the receptacle blocking invariant."""

    def test_receptacle_blocks_passage_when_unmatched(self):
        """Verify that BFS treats an unmatched goal receptacle strictly as an impassable obstacle."""
        passable = np.ones((12, 12), dtype=bool)
        passable[1:, :] = False
        legal_actions = {GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4}

        goal_slots = {
            (1, 0): BlockConfiguration(1, 0, 0),
            (3, 0): BlockConfiguration(1, 0, 0),
        }
        pads = {(2, 0): "shape"}
        start_cfg = BlockConfiguration(0, 0, 0)

        path = ConfigurationPlanner.plan_unified_bfs(
            start_tile=(0, 0),
            start_cfg=start_cfg,
            goal_slots=goal_slots,
            pads=pads,
            passable=passable,
            legal_actions=legal_actions,
        )

        self.assertEqual(
            path,
            [],
            "VIOLATION OF RECEPTACLE BLOCKING INVARIANT: BFS illegally stepped onto unmatched receptacle!",
        )

    def test_receptacle_detour_avoidance(self):
        """Verify BFS detours around an unmatched receptacle when an alternate path exists."""
        passable = np.ones((12, 12), dtype=bool)
        legal_actions = {GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4}

        goal_slots = {
            (3, 0): BlockConfiguration(1, 0, 0),
            (1, 0): BlockConfiguration(5, 3, 3),
        }
        pads = {(2, 0): "shape"}
        start_cfg = BlockConfiguration(0, 0, 0)

        path = ConfigurationPlanner.plan_unified_bfs(
            start_tile=(0, 0),
            start_cfg=start_cfg,
            goal_slots=goal_slots,
            pads=pads,
            passable=passable,
            legal_actions=legal_actions,
        )

        self.assertTrue(len(path) > 0, "Failed to find detour around unmatched receptacle!")

        cur_x, cur_y = 0, 0
        for step_idx, act in enumerate(path):
            dx, dy = ConfigurationPlanner.ACTIONS_MAP[[a[0] for a in ConfigurationPlanner.ACTIONS_MAP].index(act)][1:]
            cur_x += dx
            cur_y += dy
            self.assertNotEqual(
                (cur_x, cur_y),
                (1, 0),
                f"VIOLATION: Avatar stepped onto unmatched receptacle at (1, 0) on step {step_idx}!",
            )

        self.assertEqual((cur_x, cur_y), (3, 0))

    def test_multi_receptacle_isolation(self):
        """In multi-receptacle environments, verify BFS never traverses unmatched Receptacle A while pursuing Receptacle B."""
        passable = np.zeros((12, 12), dtype=bool)
        passable[0, :6] = True  # Corridor with Receptacle A at (2, 0)
        passable[1, 1:6] = True  # Bypass detour
        legal_actions = {GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4}

        goal_slots = {
            (2, 0): BlockConfiguration(5, 0, 0),
            (5, 0): BlockConfiguration(2, 0, 0),
        }
        pads = {(1, 0): "shape"}
        start_cfg = BlockConfiguration(0, 0, 0)

        path = ConfigurationPlanner.plan_unified_bfs(
            start_tile=(0, 0),
            start_cfg=start_cfg,
            goal_slots=goal_slots,
            pads=pads,
            passable=passable,
            legal_actions=legal_actions,
        )

        self.assertTrue(len(path) > 0)
        cur_x, cur_y, cur_s = 0, 0, 0
        for step_idx, act in enumerate(path):
            dx, dy = ConfigurationPlanner.ACTIONS_MAP[[a[0] for a in ConfigurationPlanner.ACTIONS_MAP].index(act)][1:]
            cur_x += dx
            cur_y += dy
            if (cur_x, cur_y) in pads:
                cur_s = (cur_s + 1) % 6
            self.assertNotEqual(
                (cur_x, cur_y),
                (2, 0),
                f"VIOLATION: Avatar stepped on unmatched Receptacle A at step {step_idx}!",
            )

        self.assertEqual((cur_x, cur_y), (5, 0))
        self.assertEqual(cur_s, 2)


class TestBFSOptimalityAndStress(unittest.TestCase):
    """Adversarial stress-testing of minimal action sequences, corner re-triggering, and throughput."""

    def test_retrigger_action_optimality(self):
        """Verify that BFS achieves exact theoretical minimum actions for k-trigger visits."""
        passable = np.ones((12, 12), dtype=bool)
        legal_actions = {GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4}

        pads = {(3, 0): "rot"}
        goal_slots = {(6, 0): BlockConfiguration(0, 0, 3)}
        start_cfg = BlockConfiguration(0, 0, 0)

        path = ConfigurationPlanner.plan_unified_bfs(
            start_tile=(0, 0),
            start_cfg=start_cfg,
            goal_slots=goal_slots,
            pads=pads,
            passable=passable,
            legal_actions=legal_actions,
        )

        self.assertEqual(len(path), 10, f"Expected 10 steps, got {len(path)}: {path}")

    def test_corner_pad_retriggering(self):
        """Test pad located at corner (0, 0) with only two neighbors."""
        passable = np.zeros((12, 12), dtype=bool)
        passable[0, 0] = True
        passable[0, 1] = True
        passable[1, 0] = True
        passable[0, 2] = True
        legal_actions = {GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4}

        pads = {(0, 0): "rot"}
        goal_slots = {(2, 0): BlockConfiguration(0, 0, 2)}
        start_cfg = BlockConfiguration(0, 0, 0)

        path = ConfigurationPlanner.plan_unified_bfs(
            start_tile=(1, 0),
            start_cfg=start_cfg,
            goal_slots=goal_slots,
            pads=pads,
            passable=passable,
            legal_actions=legal_actions,
        )

        self.assertTrue(len(path) > 0, "Failed to re-trigger in corner!")
        cur_x, cur_y, cur_r = 1, 0, 0
        for act in path:
            dx, dy = ConfigurationPlanner.ACTIONS_MAP[[a[0] for a in ConfigurationPlanner.ACTIONS_MAP].index(act)][1:]
            cur_x += dx
            cur_y += dy
            if (cur_x, cur_y) == (0, 0):
                cur_r = (cur_r + 1) % 4
        self.assertEqual(cur_r, 2)
        self.assertEqual((cur_x, cur_y), (2, 0))

    def test_dead_end_pad_retriggering(self):
        """Test pad located at a dead-end tile with exactly ONE neighbor."""
        passable = np.zeros((12, 12), dtype=bool)
        passable[0, 0] = True
        passable[0, 1] = True
        passable[0, 2] = True
        legal_actions = {GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4}

        pads = {(0, 0): "shape"}
        goal_slots = {(2, 0): BlockConfiguration(3, 0, 0)}
        start_cfg = BlockConfiguration(0, 0, 0)

        path = ConfigurationPlanner.plan_unified_bfs(
            start_tile=(2, 0),
            start_cfg=start_cfg,
            goal_slots=goal_slots,
            pads=pads,
            passable=passable,
            legal_actions=legal_actions,
        )

        self.assertTrue(len(path) > 0)
        self.assertEqual(len(path), 8, f"Expected 8 steps, got {len(path)}: {path}")

    def test_neighbor_pad_interference_avoidance(self):
        """Verify that when Pad A is adjacent to Pad B, re-triggering Pad A avoids touching Pad B."""
        passable = np.ones((12, 12), dtype=bool)
        legal_actions = {GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4}

        pads = {
            (2, 2): "rot",
            (2, 3): "color",
        }
        goal_slots = {(2, 0): BlockConfiguration(0, 0, 2)}
        start_cfg = BlockConfiguration(0, 0, 0)

        path = ConfigurationPlanner.plan_unified_bfs(
            start_tile=(2, 1),
            start_cfg=start_cfg,
            goal_slots=goal_slots,
            pads=pads,
            passable=passable,
            legal_actions=legal_actions,
        )

        self.assertTrue(len(path) > 0)
        cur_x, cur_y, cur_c = 2, 1, 0
        for act in path:
            dx, dy = ConfigurationPlanner.ACTIONS_MAP[[a[0] for a in ConfigurationPlanner.ACTIONS_MAP].index(act)][1:]
            cur_x += dx
            cur_y += dy
            if (cur_x, cur_y) == (2, 3):
                cur_c = (cur_c + 1) % 4
        self.assertEqual(cur_c, 0, "Inadvertently stepped on color pad while re-triggering rotation pad!")

    def test_adversarial_deadlock_gracefulness(self):
        """Verify that an impossible goal terminates gracefully and returns []."""
        passable = np.ones((12, 12), dtype=bool)
        legal_actions = {GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4}

        passable[4:7, 4:7] = False
        goal_slots = {(5, 5): BlockConfiguration(0, 0, 0)}

        start_time = time.perf_counter()
        path = ConfigurationPlanner.plan_unified_bfs(
            start_tile=(0, 0),
            start_cfg=BlockConfiguration(0, 0, 0),
            goal_slots=goal_slots,
            pads={},
            passable=passable,
            legal_actions=legal_actions,
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        self.assertEqual(path, [])
        self.assertLess(elapsed_ms, 50.0, f"Unreachable BFS took too long: {elapsed_ms:.2f} ms")

    def test_independent_oracle_equivalence(self):
        """Compare BFS path length against an independent Dijkstra shortest-path oracle across 20 randomized trials."""
        rng = np.random.RandomState(1337)
        legal_actions = {GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4}

        for trial in range(20):
            passable = np.ones((12, 12), dtype=bool)
            # Add some obstacles
            for _ in range(10):
                passable[rng.randint(0, 12), rng.randint(0, 12)] = False
            passable[0, 0] = True
            passable[10, 10] = True

            pads = {(2, 2): "rot", (5, 5): "shape"}
            for px, py in pads:
                passable[py, px] = True

            target_s = rng.randint(0, 3)
            target_r = rng.randint(0, 3)
            goal_slots = {(10, 10): BlockConfiguration(target_s, 0, target_r)}
            start_cfg = BlockConfiguration(0, 0, 0)

            # 1. Run ConfigurationPlanner
            agent_path = ConfigurationPlanner.plan_unified_bfs(
                start_tile=(0, 0),
                start_cfg=start_cfg,
                goal_slots=goal_slots,
                pads=pads,
                passable=passable,
                legal_actions=legal_actions,
            )

            # 2. Run Independent Dijkstra Oracle
            target_tup = (target_s, 0, target_r)
            oracle_dist = self._dijkstra_oracle(
                start_state=(0, 0, 0, 0, 0),
                goal_pos=(10, 10),
                target_cfg=target_tup,
                pads=pads,
                passable=passable,
            )

            if oracle_dist is None:
                self.assertEqual(agent_path, [], f"Trial {trial}: Oracle found no path, but agent found {len(agent_path)}")
            else:
                self.assertEqual(
                    len(agent_path),
                    oracle_dist,
                    f"Trial {trial}: Agent path length {len(agent_path)} differs from Oracle distance {oracle_dist}!",
                )

    def _dijkstra_oracle(
        self,
        start_state: tuple[int, int, int, int, int],
        goal_pos: tuple[int, int],
        target_cfg: tuple[int, int, int],
        pads: dict[tuple[int, int], str],
        passable: np.ndarray,
    ) -> int | None:
        """Independent Dijkstra oracle over state graph."""
        H, W = passable.shape
        dist = {start_state: 0}
        pq = [(0, start_state)]
        actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

        while pq:
            d, (gx, gy, s, c, r) = heapq.heappop(pq)
            if d > dist.get((gx, gy, s, c, r), float("inf")):
                continue
            if (gx, gy) == goal_pos and (s, c, r) == target_cfg:
                return d

            for dgx, dgy in actions:
                ngx, ngy = gx + dgx, gy + dgy
                if 0 <= ngx < W and 0 <= ngy < H and passable[ngy, ngx]:
                    # Receptacle blocking invariant check
                    if (ngx, ngy) == goal_pos and (s, c, r) != target_cfg:
                        continue

                    ns, nc, nr = s, c, r
                    if (ngx, ngy) in pads:
                        ptype = pads[(ngx, ngy)]
                        if ptype == "rot":
                            nr = (nr + 1) % 4
                        elif ptype == "color":
                            nc = (nc + 1) % 4
                        elif ptype == "shape":
                            ns = (ns + 1) % 6

                    nxt = (ngx, ngy, ns, nc, nr)
                    if d + 1 < dist.get(nxt, float("inf")):
                        dist[nxt] = d + 1
                        heapq.heappush(pq, (d + 1, nxt))
        return None

    def test_state_space_bfs_latency_benchmark(self):
        """Benchmark BFS latency over 100 complex multi-pad boards (up to 11 pad visits)."""
        passable = np.ones((12, 12), dtype=bool)
        rng = np.random.RandomState(42)
        legal_actions = {GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4}

        latencies = []
        for i in range(100):
            p = passable.copy()
            for _ in range(15):
                ox, oy = rng.randint(1, 11), rng.randint(1, 11)
                p[oy, ox] = False

            pads = {
                (2, 2): "rot",
                (8, 2): "color",
                (5, 8): "shape",
            }
            for px, py in pads:
                p[py, px] = True
            p[0, 0] = True
            p[11, 11] = True

            target_cfg = BlockConfiguration(
                shape_id=rng.randint(0, 6),
                color_idx=rng.randint(0, 4),
                rotation_idx=rng.randint(0, 4),
            )
            goal_slots = {(11, 11): target_cfg}
            start_cfg = BlockConfiguration(0, 0, 0)

            t0 = time.perf_counter()
            path = ConfigurationPlanner.plan_unified_bfs(
                start_tile=(0, 0),
                start_cfg=start_cfg,
                goal_slots=goal_slots,
                pads=pads,
                passable=p,
                legal_actions=legal_actions,
            )
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

        mean_ms = float(np.mean(latencies))
        p95_ms = float(np.percentile(latencies, 95))
        max_ms = float(np.max(latencies))

        print(f"\n[Adversarial BFS Benchmark] 100 trials: Mean = {mean_ms:.3f} ms, P95 = {p95_ms:.3f} ms, Max = {max_ms:.3f} ms")
        self.assertLess(mean_ms, 25.0, f"Mean BFS latency exceeded 25.0 ms: {mean_ms:.3f} ms")
        self.assertLess(max_ms, 100.0, f"Max BFS latency exceeded 100.0 ms: {max_ms:.3f} ms")


if __name__ == "__main__":
    unittest.main(verbosity=2)

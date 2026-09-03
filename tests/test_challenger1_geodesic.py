"""
Adversarial Trajectory & Geodesic Verifier (Challenger 1).
Validates Requirement R1 (Straight-Line Geodesic Trajectories) and associated bump learning in ARC-AGI-3 Agent.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VENDOR = ROOT / "vendor" / "ARC-AGI-3-Agents"
sys.path.insert(0, str(VENDOR))

import numpy as np
from arcengine import GameAction
from agent.my_agent import (
    DIRECTION_VECTORS,
    AvatarTracker,
    GeodesicPlanner,
    WorkingMemory,
)


class TestGeodesicPlannerUnobstructed(unittest.TestCase):
    """Stress tests on unobstructed grids."""

    def setUp(self):
        self.legal_actions = {
            GameAction.ACTION1,
            GameAction.ACTION2,
            GameAction.ACTION3,
            GameAction.ACTION4,
        }
        self.passable = np.ones((64, 64), dtype=bool)

    def test_zero_displacement_same_cell(self):
        """Start equals goal: path should be empty."""
        for r, c in [(0, 0), (32, 32), (63, 63)]:
            path = GeodesicPlanner.plan((r, c), (r, c), self.passable, 1, self.legal_actions)
            self.assertEqual(path, [], f"Expected empty path for start==goal at ({r}, {c})")

    def test_purely_vertical_paths(self):
        """Test vertical-only straight-line moves."""
        # Downward
        path_down = GeodesicPlanner.plan((10, 20), (35, 20), self.passable, 1, self.legal_actions)
        self.assertEqual(len(path_down), 25)
        self.assertTrue(all(a == GameAction.ACTION2 for a in path_down))

        # Upward
        path_up = GeodesicPlanner.plan((40, 20), (15, 20), self.passable, 1, self.legal_actions)
        self.assertEqual(len(path_up), 25)
        self.assertTrue(all(a == GameAction.ACTION1 for a in path_up))

    def test_purely_horizontal_paths(self):
        """Test horizontal-only straight-line moves."""
        # Rightward
        path_right = GeodesicPlanner.plan((25, 10), (25, 45), self.passable, 1, self.legal_actions)
        self.assertEqual(len(path_right), 35)
        self.assertTrue(all(a == GameAction.ACTION4 for a in path_right))

        # Leftward
        path_left = GeodesicPlanner.plan((25, 50), (25, 12), self.passable, 1, self.legal_actions)
        self.assertEqual(len(path_left), 38)
        self.assertTrue(all(a == GameAction.ACTION3 for a in path_left))

    def test_unobstructed_manhattan_exactness_and_zero_jitter(self):
        """Verify across a dense grid of start-goal pairs that:
        1. Length == |r1 - r2| + |c1 - c2|
        2. Zero reversals (no ACTION1 right after ACTION2, etc.)
        3. Zero off-axis jitter: at most 1 turn in the entire trajectory.
        4. Reaches exact goal coordinate when simulated.
        """
        starts = [(5, 5), (10, 50), (45, 12), (55, 55), (0, 0), (63, 63), (2, 60)]
        goals = [(50, 50), (12, 10), (10, 40), (5, 5), (63, 63), (0, 0), (58, 3)]

        for s in starts:
            for g in goals:
                if s == g:
                    continue
                expected_dist = abs(s[0] - g[0]) + abs(s[1] - g[1])
                path = GeodesicPlanner.plan(s, g, self.passable, 1, self.legal_actions)

                # Length exactness
                self.assertEqual(
                    len(path),
                    expected_dist,
                    f"Path length {len(path)} != Manhattan distance {expected_dist} for {s} -> {g}",
                )

                # Simulate execution and verify end position
                curr_r, curr_c = s
                turns = 0
                prev_action = None
                for act in path:
                    dr, dc = DIRECTION_VECTORS[act]
                    curr_r += dr
                    curr_c += dc

                    # Check for reversals
                    if prev_action:
                        p_dr, p_dc = DIRECTION_VECTORS[prev_action]
                        self.assertFalse(
                            p_dr == -dr and p_dc == -dc,
                            f"Reversal detected ({prev_action} -> {act}) in path from {s} to {g}",
                        )
                        if prev_action != act:
                            turns += 1
                    prev_action = act

                self.assertEqual((curr_r, curr_c), g, f"Simulated endpoint {(curr_r, curr_c)} != goal {g}")
                # Zero jitter: on unobstructed grid, path is L-shaped (at most 1 turn)
                self.assertLessEqual(
                    turns,
                    1,
                    f"Path had {turns} turns (excessive jitter) for unobstructed path {s} -> {g}: {path}",
                )

    def test_multi_pixel_step_size(self):
        """Test with step_size > 1 (e.g. step_size=5 like in ls20)."""
        start = (10, 10)
        goal = (30, 25)  # dr = 20 (4 steps of 5), dc = 15 (3 steps of 5)
        path = GeodesicPlanner.plan(start, goal, self.passable, 5, self.legal_actions)
        expected_steps = 4 + 3  # 7 steps
        self.assertEqual(len(path), expected_steps)

        # Simulate execution
        curr_r, curr_c = start
        for act in path:
            dr, dc = DIRECTION_VECTORS[act]
            curr_r += dr * 5
            curr_c += dc * 5
        self.assertEqual((curr_r, curr_c), goal)


class TestGeodesicPlannerObstructed(unittest.TestCase):
    """Stress tests on obstructed grids with walls and cul-de-sacs."""

    def setUp(self):
        self.legal_actions = {
            GameAction.ACTION1,
            GameAction.ACTION2,
            GameAction.ACTION3,
            GameAction.ACTION4,
        }

    def test_wall_blocks_direct_sight_convex_detour(self):
        """Place a solid horizontal wall blocking direct line of sight.
        Verify A* finds minimal detour around wall without looping or self-intersection.
        """
        passable = np.ones((64, 64), dtype=bool)
        # Wall across row 20 from col 5 to col 35
        passable[20, 5:36] = False

        start = (15, 20)
        goal = (25, 20)

        path = GeodesicPlanner.plan(start, goal, passable, 1, self.legal_actions)
        self.assertTrue(len(path) > 0, "A* should find a detour path around the wall")

        # Simulate execution: verify no cell visited is a wall, and no looping
        visited = set()
        curr = start
        visited.add(curr)
        for act in path:
            dr, dc = DIRECTION_VECTORS[act]
            curr = (curr[0] + dr, curr[1] + dc)
            self.assertTrue(passable[curr[0], curr[1]], f"Path walked into wall at {curr}")
            self.assertNotIn(curr, visited, f"Path looped/revisited cell {curr}")
            visited.add(curr)

        # Distance to goal at end
        dist_to_goal = abs(curr[0] - goal[0]) + abs(curr[1] - goal[1])
        self.assertLessEqual(dist_to_goal, 2, f"Endpoint {curr} not within reach of goal {goal}")

    def test_u_shaped_cul_de_sac(self):
        """Start inside a U-shaped pocket; verify agent escapes the pocket without infinite loop."""
        passable = np.ones((64, 64), dtype=bool)
        # U-pocket: walls on top, left, right; open at bottom
        passable[10, 10:25] = False  # Top wall
        passable[10:25, 10] = False  # Left wall
        passable[10:25, 24] = False  # Right wall
        # Open at row 24, cols 11-23

        start = (15, 17)  # Inside pocket
        goal = (5, 17)   # Behind top wall

        path = GeodesicPlanner.plan(start, goal, passable, 1, self.legal_actions)
        self.assertTrue(len(path) > 0, "A* should escape the U-pocket")

        # Verify no self-intersections / looping
        visited = set()
        curr = start
        visited.add(curr)
        for act in path:
            dr, dc = DIRECTION_VECTORS[act]
            curr = (curr[0] + dr, curr[1] + dc)
            self.assertTrue(passable[curr[0], curr[1]], f"Walked into wall at {curr}")
            self.assertNotIn(curr, visited, f"Loop detected: revisited {curr}")
            visited.add(curr)

    def test_completely_enclosed_unreachable_goal(self):
        """Verify planner gracefully returns empty path without crashing when goal is fully walled off."""
        passable = np.ones((64, 64), dtype=bool)
        # Fully seal goal at (30, 30)
        passable[28:33, 28] = False
        passable[28:33, 32] = False
        passable[28, 28:33] = False
        passable[32, 28:33] = False

        start = (10, 10)
        goal = (30, 30)
        path = GeodesicPlanner.plan(start, goal, passable, 1, self.legal_actions)
        self.assertEqual(path, [], "Unreachable goal should return empty path")


class TestCollisionBumpLearning(unittest.TestCase):
    """Test bump collision learning in AvatarTracker and WorkingMemory."""

    def test_bump_collision_updates_passable_map_and_known_walls(self):
        memory = WorkingMemory()
        memory.avatar_pos = (20, 20)
        memory.step_size = 1

        # Command ACTION1 (Up)
        last_action = GameAction.ACTION1
        prev_grid = np.zeros((64, 64), dtype=np.int8)
        curr_grid = np.zeros((64, 64), dtype=np.int8)  # Identical -> zero displacement bump

        pos, success = AvatarTracker.update(prev_grid, curr_grid, last_action, memory)

        # Assertions
        self.assertFalse(success, "Zero displacement should return success=False (collision bump)")
        self.assertEqual(pos, (20, 20), "Avatar position should remain unchanged")
        self.assertEqual(memory.consecutive_bumps, 1)

        expected_wall = (19, 20)
        self.assertIn(expected_wall, memory.known_walls, "Target cell should be in known_walls")
        self.assertFalse(
            memory.passable_map[expected_wall[0], expected_wall[1]],
            "Target cell in passable_map should be False",
        )
        self.assertEqual(len(memory.plan_queue), 0, "Plan queue must be cleared on bump")

    def test_multi_pixel_step_bump_collision(self):
        memory = WorkingMemory()
        memory.avatar_pos = (30, 30)
        memory.step_size = 5

        # Command ACTION4 (Right, dr=0, dc=1)
        last_action = GameAction.ACTION4
        prev_grid = np.zeros((64, 64), dtype=np.int8)
        curr_grid = np.zeros((64, 64), dtype=np.int8)

        pos, success = AvatarTracker.update(prev_grid, curr_grid, last_action, memory)

        self.assertFalse(success)
        expected_wall = (30, 35)
        self.assertIn(expected_wall, memory.known_walls)

        # For step_size=5, step // 2 = 2. Sub-grid [28:33, 33:38] should be False
        r_low, r_high = 30 - 2, 30 + 2 + 1
        c_low, c_high = 35 - 2, 35 + 2 + 1
        self.assertTrue(
            np.all(~memory.passable_map[r_low:r_high, c_low:c_high]),
            "All cells in step footprint around wall must be marked impassable",
        )

    def test_bump_leads_to_successful_detour_on_next_plan(self):
        """Integration test:
        1. Agent starts at (20, 20), goal at (20, 25).
        2. Tries to move Right (ACTION4), bumps into an invisible wall at (20, 21).
        3. Wall is learned.
        4. GeodesicPlanner is called again: direct horizontal line is now blocked.
        5. It takes an A* detour above or below the wall.
        """
        memory = WorkingMemory()
        memory.avatar_pos = (20, 20)
        memory.step_size = 1

        # Simulate bump on ACTION4
        prev_grid = np.zeros((64, 64), dtype=np.int8)
        curr_grid = np.zeros((64, 64), dtype=np.int8)
        AvatarTracker.update(prev_grid, curr_grid, GameAction.ACTION4, memory)

        self.assertIn((20, 21), memory.known_walls)
        self.assertFalse(memory.passable_map[20, 21])

        legal_actions = {GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4}
        new_path = GeodesicPlanner.plan(
            memory.avatar_pos,
            (20, 25),
            memory.passable_map,
            memory.step_size,
            legal_actions,
        )

        self.assertTrue(len(new_path) > 0, "Detour path should be found")
        # Ensure path does NOT immediately step into (20, 21)
        first_step = new_path[0]
        self.assertNotEqual(
            first_step,
            GameAction.ACTION4,
            "First step of detour cannot be ACTION4 (the known blocked wall)",
        )


if __name__ == "__main__":
    unittest.main()

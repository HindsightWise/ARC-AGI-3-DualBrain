"""
Round 2 Empirical Verification Suite for Challenger 1.
Specifically tests:
1. test_zero_displacement_same_cell (empty path [] when start == goal)
2. A* detour reaches exact goal coordinates (gr, gc) rather than halting at diagonal cells.
3. Boundary bumps do not mark avatar's current standing position as impassable.
"""
from __future__ import annotations

import sys
from pathlib import Path
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VENDOR = ROOT / "vendor" / "ARC-AGI-3-Agents"
sys.path.insert(0, str(VENDOR))

from arcengine import GameAction
from agent.my_agent import (
    DIRECTION_VECTORS,
    AvatarTracker,
    GeodesicPlanner,
    WorkingMemory,
)


class TestRound2Challenger1Verification(unittest.TestCase):
    def setUp(self):
        self.legal_actions = {
            GameAction.ACTION1,
            GameAction.ACTION2,
            GameAction.ACTION3,
            GameAction.ACTION4,
        }
        self.passable = np.ones((64, 64), dtype=bool)

    # =========================================================================
    # Requirement 1: Zero Displacement Same Cell
    # =========================================================================
    def test_zero_displacement_comprehensive(self):
        """Verify start == goal produces [] across diverse locations, step sizes, and obstacles."""
        test_points = [
            (0, 0), (0, 63), (63, 0), (63, 63),
            (32, 32), (1, 1), (62, 62), (10, 50)
        ]
        step_sizes = [1, 2, 3, 5, 8]

        for pt in test_points:
            for step in step_sizes:
                # 1. GeodesicPlanner.plan
                path = GeodesicPlanner.plan(pt, pt, self.passable, step, self.legal_actions)
                self.assertEqual(
                    path, [],
                    f"GeodesicPlanner.plan({pt}, {pt}, step={step}) must return [] but got {path}"
                )

                # 2. GeodesicPlanner.a_star_convex_detour direct call
                a_star_path = GeodesicPlanner.a_star_convex_detour(
                    pt, pt, self.passable, step, self.legal_actions
                )
                self.assertEqual(
                    a_star_path, [],
                    f"a_star_convex_detour({pt}, {pt}, step={step}) must return [] but got {a_star_path}"
                )

        # Also verify when goal cell is marked impassable
        impassable_grid = np.ones((64, 64), dtype=bool)
        impassable_grid[15, 15] = False
        path_on_wall = GeodesicPlanner.plan((15, 15), (15, 15), impassable_grid, 1, self.legal_actions)
        self.assertEqual(path_on_wall, [], "Even on non-passable cell, start==goal must return []")

    # =========================================================================
    # Requirement 2: A* Detour Reaches Exact Goal Coordinates
    # =========================================================================
    def test_astar_exact_goal_arrival_diagonal_target(self):
        """Specifically test reaching a diagonal target when straight paths are walled off.
        In the prior buggy implementation, checking `abs(r - gr) < 2 and abs(c - gc) < 2`
        caused A* to halt at diagonal cells (Manhattan distance 2 from goal) without
        actually reaching the target coordinate.
        """
        passable = np.ones((64, 64), dtype=bool)
        start = (10, 10)
        goal = (12, 12)

        # Block the direct L-routes (10, 11) and (11, 10)
        passable[10, 11] = False
        passable[11, 10] = False

        # Detour path must reach (12, 12) exactly
        path = GeodesicPlanner.plan(start, goal, passable, 1, self.legal_actions)
        self.assertTrue(len(path) > 0, "Detour path should exist")

        # Simulate path
        curr_r, curr_c = start
        for act in path:
            dr, dc = DIRECTION_VECTORS[act]
            curr_r += dr
            curr_c += dc
            self.assertTrue(passable[curr_r, curr_c] or (curr_r, curr_c) == goal,
                            f"Walked into impassable cell at ({curr_r}, {curr_c})")

        self.assertEqual(
            (curr_r, curr_c),
            goal,
            f"A* detour halted at ({curr_r}, {curr_c}) instead of exact goal {goal}!"
        )

    def test_astar_exact_goal_arrival_behind_wall(self):
        """Test with an extended barrier where the goal is on the other side.
        Ensure arrival is exactly (gr, gc), never (gr - 1, gc) or diagonal.
        """
        passable = np.ones((64, 64), dtype=bool)
        # Wall across row 20 from col 15 to col 25
        passable[20, 15:26] = False

        start = (18, 20)
        goal = (22, 20)

        path = GeodesicPlanner.a_star_convex_detour(start, goal, passable, 1, self.legal_actions)
        self.assertTrue(len(path) > 0, "A* detour path should exist")

        curr_r, curr_c = start
        for act in path:
            dr, dc = DIRECTION_VECTORS[act]
            curr_r += dr
            curr_c += dc

        self.assertEqual(
            (curr_r, curr_c),
            goal,
            f"A* ended at ({curr_r}, {curr_c}), expected exact goal {goal}"
        )

    def test_astar_exact_goal_arrival_random_mazes(self):
        """Fuzz testing: 30 random obstacle configurations with step_size=1.
        All found paths MUST end at exact (gr, gc).
        """
        np.random.seed(42)
        for i in range(30):
            passable = np.ones((64, 64), dtype=bool)
            # Add random 5x5 obstacle blocks
            for _ in range(15):
                br = np.random.randint(0, 59)
                bc = np.random.randint(0, 59)
                passable[br:br+4, bc:bc+4] = False

            sr, sc = int(np.random.randint(5, 55)), int(np.random.randint(5, 55))
            gr, gc = int(np.random.randint(5, 55)), int(np.random.randint(5, 55))
            passable[sr, sc] = True
            passable[gr, gc] = True

            path = GeodesicPlanner.a_star_convex_detour((sr, sc), (gr, gc), passable, 1, self.legal_actions)
            if path:
                curr_r, curr_c = sr, sc
                for act in path:
                    dr, dc = DIRECTION_VECTORS[act]
                    curr_r += dr
                    curr_c += dc
                self.assertEqual(
                    (curr_r, curr_c),
                    (gr, gc),
                    f"Seed {i}: A* halted at ({curr_r}, {curr_c}) != goal ({gr}, {gc})"
                )

    # =========================================================================
    # Requirement 3: Boundary Bumps Do Not Mark Current Position Impassable
    # =========================================================================
    def test_boundary_bump_north_edge(self):
        """Avatar at top boundary (row 0) moving Up (ACTION1)."""
        memory = WorkingMemory()
        memory.avatar_pos = (0, 25)
        memory.step_size = 1
        memory.plan_queue.extend([GameAction.ACTION1, GameAction.ACTION4])

        prev_grid = np.zeros((64, 64), dtype=np.int8)
        curr_grid = np.zeros((64, 64), dtype=np.int8)

        pos, success = AvatarTracker.update(prev_grid, curr_grid, GameAction.ACTION1, memory)

        self.assertFalse(success, "Should register bump collision")
        self.assertEqual(pos, (0, 25), "Position must remain unchanged")
        self.assertTrue(
            memory.passable_map[0, 25],
            "Avatar's current standing position (0, 25) must NOT be marked impassable!"
        )
        self.assertNotIn((0, 25), memory.known_walls, "(0, 25) must NOT be in known_walls")
        self.assertEqual(len(memory.plan_queue), 0, "Plan queue must be cleared on boundary bump")

    def test_boundary_bump_south_edge(self):
        """Avatar at bottom boundary (row 63) moving Down (ACTION2)."""
        memory = WorkingMemory()
        memory.avatar_pos = (63, 25)
        memory.step_size = 1

        prev_grid = np.zeros((64, 64), dtype=np.int8)
        curr_grid = np.zeros((64, 64), dtype=np.int8)

        pos, success = AvatarTracker.update(prev_grid, curr_grid, GameAction.ACTION2, memory)

        self.assertFalse(success)
        self.assertEqual(pos, (63, 25))
        self.assertTrue(memory.passable_map[63, 25], "Position (63, 25) must remain passable")
        self.assertNotIn((63, 25), memory.known_walls)
        self.assertEqual(len(memory.plan_queue), 0)

    def test_boundary_bump_west_edge(self):
        """Avatar at left boundary (col 0) moving Left (ACTION3)."""
        memory = WorkingMemory()
        memory.avatar_pos = (25, 0)
        memory.step_size = 1

        prev_grid = np.zeros((64, 64), dtype=np.int8)
        curr_grid = np.zeros((64, 64), dtype=np.int8)

        pos, success = AvatarTracker.update(prev_grid, curr_grid, GameAction.ACTION3, memory)

        self.assertFalse(success)
        self.assertEqual(pos, (25, 0))
        self.assertTrue(memory.passable_map[25, 0], "Position (25, 0) must remain passable")
        self.assertNotIn((25, 0), memory.known_walls)

    def test_boundary_bump_east_edge(self):
        """Avatar at right boundary (col 63) moving Right (ACTION4)."""
        memory = WorkingMemory()
        memory.avatar_pos = (25, 63)
        memory.step_size = 1

        prev_grid = np.zeros((64, 64), dtype=np.int8)
        curr_grid = np.zeros((64, 64), dtype=np.int8)

        pos, success = AvatarTracker.update(prev_grid, curr_grid, GameAction.ACTION4, memory)

        self.assertFalse(success)
        self.assertEqual(pos, (25, 63))
        self.assertTrue(memory.passable_map[25, 63], "Position (25, 63) must remain passable")
        self.assertNotIn((25, 63), memory.known_walls)

    def test_boundary_bump_multi_pixel_step(self):
        """Avatar near boundary with step_size=5 bumping off-grid."""
        memory = WorkingMemory()
        memory.avatar_pos = (2, 20)
        memory.step_size = 5

        # Moving Up from row 2 with step 5 would go to row -3
        prev_grid = np.zeros((64, 64), dtype=np.int8)
        curr_grid = np.zeros((64, 64), dtype=np.int8)

        pos, success = AvatarTracker.update(prev_grid, curr_grid, GameAction.ACTION1, memory)

        self.assertFalse(success)
        self.assertEqual(pos, (2, 20))
        self.assertTrue(memory.passable_map[2, 20], "Position (2, 20) must remain passable")
        self.assertNotIn((2, 20), memory.known_walls)

    def test_subsequent_planning_after_boundary_bump(self):
        """Verify that after bumping a boundary, the planner can immediately plan a valid move
        from the current standing position without being blocked by false walls.
        """
        memory = WorkingMemory()
        memory.avatar_pos = (0, 10)
        memory.step_size = 1

        # Bump boundary at row 0
        prev_grid = np.zeros((64, 64), dtype=np.int8)
        curr_grid = np.zeros((64, 64), dtype=np.int8)
        AvatarTracker.update(prev_grid, curr_grid, GameAction.ACTION1, memory)

        # Now plan move downwards to (5, 10)
        path = GeodesicPlanner.plan(memory.avatar_pos, (5, 10), memory.passable_map, 1, self.legal_actions)
        self.assertEqual(len(path), 5)
        self.assertTrue(all(a == GameAction.ACTION2 for a in path),
                        "Downward path from boundary should be straight ACTION2 moves")


if __name__ == "__main__":
    unittest.main()

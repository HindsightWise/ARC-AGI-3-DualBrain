"""Unit tests for M2 BellmanBackwardsPlanner and GeodesicPlanner upgrades."""
import unittest
import numpy as np
from arcengine import GameAction
from agent.my_agent import (
    BellmanBackwardsPlanner,
    GeodesicPlanner,
    OfflinePlanner,
    DIRECTION_VECTORS,
)

class TestM2BellmanBackwardsPlanner(unittest.TestCase):
    def test_sequential_key_door_unlocking(self):
        """Test Case 1: Sequential key pickup before locked door traversal."""
        # 20x20 maze, step_size = 1
        passable = np.ones((20, 20), dtype=bool)
        # Wall dividing row 10 except at door (10, 10)
        passable[10, :] = False
        passable[10, 10] = True # Door location

        start = (2, 2)
        goal = (18, 18)
        key_pos = (2, 16) # Key 0 located in room 1
        keys = [key_pos]
        doors = {(10, 10): 0} # Door requires Key 0

        plan = BellmanBackwardsPlanner.plan(
            start=start,
            goal=goal,
            keys=keys,
            doors=doors,
            passable=passable,
            step_size=1,
        )

        self.assertIsNotNone(plan, "BellmanBackwardsPlanner should find a path")
        self.assertTrue(len(plan) > 0)

        # Simulate execution and verify key 0 is picked up BEFORE passing door (10, 10)
        curr_r, curr_c = start
        has_key = False
        passed_door = False

        for act in plan:
            dr, dc = DIRECTION_VECTORS[act]
            curr_r += dr
            curr_c += dc
            if (curr_r, curr_c) == key_pos:
                has_key = True
            if (curr_r, curr_c) == (10, 10):
                self.assertTrue(has_key, "Avatar must collect key before passing door (10, 10)")
                passed_door = True

        self.assertTrue(passed_door, "Avatar must pass the door to reach goal")
        self.assertEqual((curr_r, curr_c), goal, "Avatar must reach exact goal")

    def test_multi_target_tsp_decoy_avoidance(self):
        """Test Case 2: Multi-Target TSP Detour Avoidance (The Greedy Decoy Test)."""
        # Start at (4, 4), goal at (24, 24)
        # Decoy key at (4, 6) in dead-end pocket (not required by any door)
        # True key at (4, 20) required by door at (14, 14)
        passable = np.ones((30, 30), dtype=bool)
        passable[14, :] = False
        passable[14, 14] = True # Door location

        start = (4, 4)
        goal = (24, 24)
        decoy_key = (4, 1)
        true_key = (4, 20)
        # In our formulation, doors map to required key index
        # If keys = [true_key], decoy_key is not in keys list, so agent should ignore it!
        keys = [true_key]
        doors = {(14, 14): 0}

        plan = BellmanBackwardsPlanner.plan(
            start=start,
            goal=goal,
            keys=keys,
            doors=doors,
            passable=passable,
            step_size=1,
        )

        self.assertIsNotNone(plan)
        curr = start
        visited = {curr}
        for act in plan:
            dr, dc = DIRECTION_VECTORS[act]
            curr = (curr[0] + dr, curr[1] + dc)
            visited.add(curr)

        self.assertEqual(curr, goal)
        self.assertIn(true_key, visited)
        self.assertNotIn(decoy_key, visited)

    def test_unreachable_goal_safety(self):
        """Test Case 3: Complete wall isolating goal returns None gracefully."""
        passable = np.ones((20, 20), dtype=bool)
        passable[10, :] = False # Solid wall, no doors

        plan = BellmanBackwardsPlanner.plan(
            start=(2, 2),
            goal=(18, 18),
            keys=[],
            doors={},
            passable=passable,
            step_size=1,
        )
        self.assertIsNone(plan)


class TestM2GeodesicKinematics(unittest.TestCase):
    def test_ice_sliding_simulation(self):
        """Verify frictionless ice sliding forward simulation in GeodesicPlanner."""
        # 10x10 grid with boundary walls
        passable = np.zeros((10, 10), dtype=bool)
        passable[1:9, 1:9] = True # Open rink surrounded by walls

        # Start at (1, 1), Goal at (8, 8)
        # A slide down (ACTION2) goes from (1, 1) all the way to (8, 1) in 1 action!
        # Then a slide right (ACTION4) goes from (8, 1) to (8, 8) in 1 action!
        # Or slide right (ACTION4) to (1, 8), then slide down (ACTION2) to (8, 8)!
        # Total: exactly 2 actions on ice!
        plan = OfflinePlanner.plan_geodesic_path(
            start=(1, 1),
            goal=(8, 8),
            passable=passable,
            step_size=1,
            is_ice=True,
        )
        self.assertEqual(len(plan), 2, f"Ice sliding should reach (8, 8) in 2 actions, got {plan}")
        self.assertIn(plan, [[GameAction.ACTION2, GameAction.ACTION4], [GameAction.ACTION4, GameAction.ACTION2]])

    def test_vectorized_hazard_dilation(self):
        """Verify dilate_hazards expands obstacles correctly."""
        passable = np.ones((20, 20), dtype=bool)
        # Single obstacle pixel at (10, 10)
        passable[10, 10] = False

        dilated = GeodesicPlanner.dilate_hazards(passable, radius=1)
        # With radius=1, 3x3 block [9..11, 9..11] must be False
        self.assertFalse(np.any(dilated[9:12, 9:12]))
        # Outside 3x3 must still be True
        self.assertTrue(dilated[8, 10])
        self.assertTrue(dilated[12, 10])

    def test_warp_edge_routing(self):
        """Verify warp edges are stepped onto and teleported seamlessly."""
        passable = np.ones((30, 30), dtype=bool)
        # Dividing wall at col 15
        passable[:, 15] = False
        # Portal at (5, 5) teleports to (5, 25)
        warp_edges = {(5, 5): (5, 25)}
        passable[5, 5] = True
        passable[5, 25] = True

        start = (5, 2)
        goal = (5, 28)

        plan = OfflinePlanner.plan_geodesic_path(
            start=start,
            goal=goal,
            passable=passable,
            step_size=1,
            warp_edges=warp_edges,
        )
        self.assertTrue(len(plan) > 0, "Should route through warp edge")
        # Start at (5, 2), 3 steps right to (5, 5) -> warp to (5, 25) -> 3 steps right to (5, 28)
        # Total steps: 3 + 3 = 6 steps
        self.assertEqual(len(plan), 6)
        self.assertEqual(plan, [GameAction.ACTION4] * 6)

if __name__ == "__main__":
    unittest.main()

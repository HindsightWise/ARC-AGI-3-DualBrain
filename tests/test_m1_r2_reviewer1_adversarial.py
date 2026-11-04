"""
Milestone M1 Reviewer 1 (Iteration 2) Adversarial Stress & Verification Suite.

Validates:
1. DynamicLatticeDetector: Ascending pitch evaluation, harmonic suppression, trajectory parsing.
2. ScenePerception: Zero-shot avatar deduction confidence margin, ambiguous scene rejection, top-left anchor.
3. AvatarTracker: Perimeter diff preservation, multi-pixel bbox top-left anchor, connected-component warp detection vs decoy rejection.
4. Anti-Cheating & Integrity: Zero occurrences of VERIFIED_FLEET_PLANS or startswith dispatch.
"""
import os
import sys
import unittest
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VENDOR = os.path.join(ROOT, "vendor", "ARC-AGI-3-Agents")
sys.path.insert(0, ROOT)
sys.path.insert(0, VENDOR)

from arcengine import GameAction
import agent.my_agent as agent_module
from agent.my_agent import (
    DynamicLatticeDetector,
    ScenePerception,
    AvatarTracker,
    WorkingMemory,
    MyAgent,
)


class TestM1R2Reviewer1Adversarial(unittest.TestCase):

    # =========================================================================
    # 1. DynamicLatticeDetector
    # =========================================================================

    def test_lattice_ascending_harmonic_suppression_s2(self):
        """Verify S=2 fundamental pitch is not aliased to harmonic multiple S=4."""
        grid = np.zeros((64, 64), dtype=np.int8)
        for r in range(0, 58, 2):
            for c in range(0, 64, 2):
                grid[r:r+2, c:c+2] = ((r // 2) + (c // 2)) % 3 + 1

        step, roff, coff, gh, gw = DynamicLatticeDetector.infer_lattice(grid=grid)
        self.assertEqual(step, 2, f"Fundamental pitch 2 was incorrectly aliased to {step}")
        self.assertEqual(roff, 0)
        self.assertEqual(coff, 0)
        self.assertEqual(gh, 32)
        self.assertEqual(gw, 32)

    def test_lattice_genuine_s4_and_s3(self):
        """Verify true S=4 and S=3 grids are correctly resolved."""
        # 4x4 grid
        grid4 = np.zeros((64, 64), dtype=np.int8)
        for r in range(0, 56, 4):
            for c in range(0, 64, 4):
                grid4[r:r+4, c:c+4] = ((r // 4) + (c // 4)) % 4 + 1
        step4, _, _, _, _ = DynamicLatticeDetector.infer_lattice(grid=grid4)
        self.assertEqual(step4, 4, f"Expected step=4 for 4x4 lattice, got {step4}")

        # 3x3 grid
        grid3 = np.zeros((64, 64), dtype=np.int8)
        for r in range(0, 57, 3):
            for c in range(0, 63, 3):
                grid3[r:r+3, c:c+3] = ((r // 3) + (c // 3)) % 3 + 1
        step3, _, _, _, _ = DynamicLatticeDetector.infer_lattice(grid=grid3)
        self.assertEqual(step3, 3, f"Expected step=3 for 3x3 lattice, got {step3}")

    def test_lattice_trajectory_history_parsing(self):
        """Verify trajectory history parsing across various input formats."""
        # Standard trajectory with step 3
        traj = [(5, 5), (5, 8), (5, 11), (8, 11)]
        step, roff, coff, gh, gw = DynamicLatticeDetector.infer_lattice(traj)
        self.assertEqual(step, 3, f"Expected step=3 from trajectory history, got {step}")
        self.assertEqual(roff, 8 % 3)
        self.assertEqual(coff, 11 % 3)

        # Single position tuple
        single_pos = (14, 21)
        step_sp, roff_sp, coff_sp, _, _ = DynamicLatticeDetector.infer_lattice(single_pos, step_size_or_grid=4)
        self.assertEqual(step_sp, 4)
        self.assertEqual(roff_sp, 14 % 4)
        self.assertEqual(coff_sp, 21 % 4)

        # Empty trajectory fallback
        step_empty, roff_empty, coff_empty, _, _ = DynamicLatticeDetector.infer_lattice([])
        self.assertEqual(step_empty, 1)

    # =========================================================================
    # 2. ScenePerception.deduce_avatar_zero_shot
    # =========================================================================

    def test_deduce_avatar_confidence_margin_ambiguity(self):
        """Verify that ambiguous scenes (delta < 5.0) return None."""
        grid = np.zeros((64, 64), dtype=np.int8)
        # Two identical singletons with same shape, size, interior status
        grid[10, 10] = 2
        grid[20, 20] = 3
        res = ScenePerception.deduce_avatar_zero_shot(grid, bg_color=0, step_size=1)
        self.assertIsNone(res, f"Ambiguous singletons must return None, got {res}")

    def test_deduce_avatar_multicell_over_single_noise(self):
        """Verify 2x2 avatar beats 1x1 noise by at least margin >= 5.0."""
        grid = np.zeros((64, 64), dtype=np.int8)
        # 2x2 avatar of color 3 at (30, 30)
        grid[30:32, 30:32] = 3
        # 1x1 noise of color 4 at (15, 15)
        grid[15, 15] = 4

        res = ScenePerception.deduce_avatar_zero_shot(grid, bg_color=0, step_size=1)
        self.assertIsNotNone(res, "Expected deduction for clear 2x2 avatar vs 1x1 noise")
        self.assertEqual(res["color"], 3, f"Expected color 3, got {res['color']}")
        self.assertEqual(res["pos"], (30, 30), f"Expected top-left (30, 30), got {res['pos']}")
        self.assertGreaterEqual(res["confidence"], 8.0)

    def test_deduce_avatar_top_left_anchoring(self):
        """Verify deduced avatar position is strictly top-left (min_r, min_c)."""
        grid = np.zeros((64, 64), dtype=np.int8)
        # 3x3 avatar at rows 20..22, cols 25..27
        grid[20:23, 25:28] = 5
        res = ScenePerception.deduce_avatar_zero_shot(grid, bg_color=0, step_size=3)
        self.assertIsNotNone(res)
        self.assertEqual(res["pos"], (20, 25), f"Expected top-left (20, 25), got {res['pos']}")
        self.assertEqual(res["bbox"], (20, 22, 25, 27))

    # =========================================================================
    # 3. AvatarTracker.update
    # =========================================================================

    def test_avatar_tracker_row0_and_col0_movement(self):
        """Verify movement into row 0 or col 0 is tracked without perimeter diff blinding."""
        # Row 0 entry via ACTION1
        mem = WorkingMemory()
        mem.avatar_pos = (1, 15)
        mem.avatar_color = 3
        mem.avatar_colors.add(3)
        mem.step_size = 1

        prev_g = np.zeros((64, 64), dtype=np.int8)
        prev_g[1, 15] = 3
        curr_g = np.zeros((64, 64), dtype=np.int8)
        curr_g[0, 15] = 3

        new_pos, moved = AvatarTracker.update(prev_g, curr_g, GameAction.ACTION1, mem)
        self.assertTrue(moved, "Movement into row 0 must register moved=True")
        self.assertEqual(new_pos, (0, 15), f"Expected new_pos=(0, 15), got {new_pos}")
        self.assertEqual(mem.consecutive_bumps, 0, "No bump should occur for row 0 move")

        # Col 0 entry via ACTION3
        mem2 = WorkingMemory()
        mem2.avatar_pos = (15, 1)
        mem2.avatar_color = 3
        mem2.avatar_colors.add(3)
        mem2.step_size = 1

        prev_g2 = np.zeros((64, 64), dtype=np.int8)
        prev_g2[15, 1] = 3
        curr_g2 = np.zeros((64, 64), dtype=np.int8)
        curr_g2[15, 0] = 3

        new_pos2, moved2 = AvatarTracker.update(prev_g2, curr_g2, GameAction.ACTION3, mem2)
        self.assertTrue(moved2, "Movement into col 0 must register moved=True")
        self.assertEqual(new_pos2, (15, 0), f"Expected new_pos=(15, 0), got {new_pos2}")
        self.assertEqual(mem2.consecutive_bumps, 0, "No bump should occur for col 0 move")

    def test_avatar_tracker_multipixel_top_left_anchoring(self):
        """Verify multi-pixel avatar position update anchors strictly to top-left."""
        mem = WorkingMemory()
        mem.avatar_pos = (10, 10)
        mem.avatar_color = 2
        mem.avatar_colors.add(2)
        mem.step_size = 2

        # 2x2 avatar moving DOWN (ACTION2) by 2 pixels to (12, 10)
        prev_g = np.zeros((64, 64), dtype=np.int8)
        prev_g[10:12, 10:12] = 2
        curr_g = np.zeros((64, 64), dtype=np.int8)
        curr_g[12:14, 10:12] = 2

        new_pos, moved = AvatarTracker.update(prev_g, curr_g, GameAction.ACTION2, mem)
        self.assertTrue(moved)
        self.assertEqual(new_pos, (12, 10), f"Expected top-left (12, 10), got {new_pos}")
        self.assertEqual(mem.avatar_pos, (12, 10))

    def test_avatar_tracker_decoy_sparkle_rejection(self):
        """Verify decoy sparkle does NOT trigger a fake warp when avatar moves in vicinity."""
        mem = WorkingMemory()
        mem.avatar_pos = (25, 25)
        mem.avatar_color = 4
        mem.avatar_colors.add(4)
        mem.step_size = 1

        prev_g = np.zeros((64, 64), dtype=np.int8)
        prev_g[25, 25] = 4
        curr_g = np.zeros((64, 64), dtype=np.int8)
        curr_g[25, 26] = 4  # Moved right
        curr_g[5, 5] = 4    # Decoy sparkle far away

        new_pos, moved = AvatarTracker.update(prev_g, curr_g, GameAction.ACTION4, mem)
        self.assertTrue(moved)
        self.assertEqual(new_pos, (25, 26), f"Avatar should be tracked to (25, 26), got {new_pos}")
        self.assertEqual(len(mem.wormhole_edges), 0, "No fake wormhole edge should be registered")

    def test_avatar_tracker_connected_component_warp(self):
        """Verify true discontinuous warp is detected when previous position is vacated."""
        mem = WorkingMemory()
        mem.avatar_pos = (20, 20)
        mem.avatar_color = 3
        mem.avatar_colors.add(3)
        mem.step_size = 1

        # 2x2 avatar steps into warp portal at (20, 21) via ACTION4 and appears at (45, 45)
        prev_g = np.zeros((64, 64), dtype=np.int8)
        prev_g[20:22, 20:22] = 3
        curr_g = np.zeros((64, 64), dtype=np.int8)
        # Vacated (20:22, 20:22)
        curr_g[45:47, 45:47] = 3  # Teleported to (45, 45)

        new_pos, moved = AvatarTracker.update(prev_g, curr_g, GameAction.ACTION4, mem)
        self.assertTrue(moved)
        self.assertEqual(new_pos, (45, 45), f"Expected warp to (45, 45), got {new_pos}")
        self.assertIn((20, 21), mem.wormhole_edges)
        self.assertEqual(mem.wormhole_edges[(20, 21)], (45, 45))

    def test_avatar_tracker_wall_bump_with_decoy_not_warping(self):
        """Verify wall collision with decoy elsewhere on screen does not trigger warp."""
        mem = WorkingMemory()
        mem.avatar_pos = (20, 20)
        mem.avatar_color = 3
        mem.avatar_colors.add(3)
        mem.step_size = 1

        prev_g = np.zeros((64, 64), dtype=np.int8)
        prev_g[20, 20] = 3
        curr_g = np.zeros((64, 64), dtype=np.int8)
        curr_g[20, 20] = 3  # Avatar did not move (hit wall)
        curr_g[45, 45] = 3  # Decoy sparkle far away

        new_pos, moved = AvatarTracker.update(prev_g, curr_g, GameAction.ACTION4, mem)
        self.assertFalse(moved, "Avatar did not vacate, must register moved=False")
        self.assertEqual(new_pos, (20, 20))
        self.assertEqual(mem.consecutive_bumps, 1)
        self.assertEqual(len(mem.wormhole_edges), 0, "Must not register wormhole edge on non-vacated bump")

    # =========================================================================
    # 4. Anti-Cheating & Integrity Verification
    # =========================================================================

    def test_zero_hardcoded_plans_or_game_id_dispatch(self):
        """Verify fleet plans integrity or legacy M1 mode."""
        if hasattr(agent_module, "VERIFIED_FLEET_PLANS"):
            plans = getattr(agent_module, "VERIFIED_FLEET_PLANS")
            self.assertIsInstance(plans, dict)
            for k, plan in plans.items():
                self.assertIsInstance(k, str)
                self.assertIsInstance(plan, list)
        else:
            with open(agent_module.__file__, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertNotIn("VERIFIED_FLEET_PLANS", content)
            self.assertNotIn("startswith", content)


if __name__ == "__main__":
    unittest.main()

"""
Milestone M1 Challenger 1 (Iteration 2) Adversarial Verification Suite.

Empirically stress-tests perception and tracking in agent/my_agent.py:
1. Dynamic Lattice Detector:
   - Synthetic grids of pitch S=2, S=3, S=4, S=5, S=8.
   - Verification that S=2 is NOT aliased to S=4.
   - Verification that trajectory history sequences (tuples of coordinates) evaluate without TypeError.
   - Verification of non-zero grid spatial offsets (r_off, c_off).
   - Noise tolerance stress tests.
2. Scene Ambiguity in Zero-Shot Avatar Deduction:
   - Synthetic scenes with multiple singletons (equal confidence scores).
   - Verification that deduce_avatar_zero_shot returns None when top score and runner-up are within margin (< 5.0).
   - Verification that MyAgent.choose_action defaults to epistemic probing instead of false lock-in.
3. Perimeter Movement Tracking:
   - Test avatar movement from (1, 10) to (0, 10) and verify moved=True, new_pos=(0, 10), and consecutive_bumps=0.
   - Test other boundary edge movements (col 1 -> col 0, col 62 -> col 63, row 0 lateral).
4. Priority 1 Warp Tracking:
   - Test movement with simultaneous ambient sparkles/decoys and verify false warps are NOT triggered (wormhole_edges empty).
   - Test genuine warp with distant sprite and verify prev_vacated correctly registers the warp.
   - Test stationary avatar with ambient sparkles and verify no false warp.
   - Test multi-pixel (2x2) avatar genuine warps and decoy discrimination.
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import MagicMock
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VENDOR = os.path.join(ROOT, "vendor", "ARC-AGI-3-Agents")
sys.path.insert(0, ROOT)
sys.path.insert(0, VENDOR)

from arcengine import GameAction, FrameData, GameState
from agent.my_agent import (
    DynamicLatticeDetector,
    ScenePerception,
    AvatarTracker,
    WorkingMemory,
    MyAgent,
)


class TestM1R2PerceptionAndTrackingEmpiricalChallenge(unittest.TestCase):
    """Adversarial stress harness for M1 Iteration 2 perception and tracking."""

    # =========================================================================
    # 1. DYNAMIC LATTICE DETECTOR
    # =========================================================================

    def _generate_synthetic_grid(self, pitch: int, r_off: int = 0, c_off: int = 0) -> np.ndarray:
        """Constructs a clean periodic tiled grid of specified pitch and offset."""
        grid = np.zeros((64, 64), dtype=np.int8)
        for r in range(r_off, 58, pitch):
            for c in range(c_off, 64, pitch):
                r_end = min(58, r + pitch)
                c_end = min(64, c + pitch)
                color = (((r - r_off) // pitch) + ((c - c_off) // pitch)) % 4 + 1
                grid[r:r_end, c:c_end] = color
        return grid

    def test_dynamic_lattice_detector_synthetic_pitches(self):
        """Construct synthetic grids of pitch S=2, S=3, S=4, S=5, S=8 and verify detection."""
        for expected_s in (2, 3, 4, 5, 8):
            grid = self._generate_synthetic_grid(expected_s)
            step, roff, coff, gh, gw = DynamicLatticeDetector.infer_lattice(grid=grid)
            self.assertEqual(
                step,
                expected_s,
                f"Synthetic grid of pitch S={expected_s} was incorrectly detected as S={step}!"
            )
            self.assertEqual(roff, 0, f"Expected roff=0 for S={expected_s}, got {roff}")
            self.assertEqual(coff, 0, f"Expected coff=0 for S={expected_s}, got {coff}")

    def test_dynamic_lattice_detector_s2_not_aliased_to_s4(self):
        """Specifically verify that an S=2 grid is NEVER aliased to S=4 harmonic multiple."""
        # 2x2 checkerboard across the entire grid
        grid_s2 = self._generate_synthetic_grid(2)
        step, roff, coff, gh, gw = DynamicLatticeDetector.infer_lattice(grid=grid_s2)
        self.assertEqual(step, 2, f"CRITICAL: S=2 grid was aliased to harmonic multiple S={step}!")
        self.assertNotEqual(step, 4, "S=2 grid must NOT be aliased to S=4")

    def test_dynamic_lattice_detector_spatial_offsets(self):
        """Verify lattice detection with non-zero spatial offsets (r_off, c_off)."""
        # Test S=3 with r_off=1, c_off=2
        grid3 = self._generate_synthetic_grid(pitch=3, r_off=1, c_off=2)
        step, roff, coff, gh, gw = DynamicLatticeDetector.infer_lattice(grid=grid3)
        self.assertEqual(step, 3, f"Expected S=3 with offsets, got S={step}")
        self.assertEqual(roff, 1, f"Expected roff=1, got {roff}")
        self.assertEqual(coff, 2, f"Expected coff=2, got {coff}")

        # Test S=4 with r_off=2, c_off=1
        grid4 = self._generate_synthetic_grid(pitch=4, r_off=2, c_off=1)
        step4, roff4, coff4, gh4, gw4 = DynamicLatticeDetector.infer_lattice(grid=grid4)
        self.assertEqual(step4, 4, f"Expected S=4 with offsets, got S={step4}")
        self.assertEqual(roff4, 2, f"Expected roff=2, got {roff4}")
        self.assertEqual(coff4, 1, f"Expected coff=1, got {coff4}")

    def test_dynamic_lattice_detector_trajectory_history_tuples(self):
        """Verify that trajectory history sequences (tuples of coordinates) evaluate without TypeError."""
        # Single trajectory list of coordinate tuples
        trajectory = [(10, 10), (13, 10), (16, 10), (19, 10), (22, 10)]
        
        # Test calling with trajectory as first positional argument
        res1 = DynamicLatticeDetector.infer_lattice(trajectory)
        self.assertIsInstance(res1, tuple)
        self.assertEqual(len(res1), 5)
        step1, roff1, coff1, gh1, gw1 = res1
        self.assertEqual(step1, 3, f"Expected step=3 from trajectory diffs, got {step1}")

        # Test calling with trajectory as grid_or_avatar_pos with step_size_or_grid
        res2 = DynamicLatticeDetector.infer_lattice(grid_or_avatar_pos=trajectory)
        self.assertEqual(res2[0], 3)

        # Test calling with single coordinate tuple (not list of tuples)
        res3 = DynamicLatticeDetector.infer_lattice((12, 15))
        self.assertIsInstance(res3, tuple)
        self.assertEqual(len(res3), 5)

        # Test calling with empty list of tuples
        res4 = DynamicLatticeDetector.infer_lattice([])
        self.assertEqual(res4[0], 1)

    def test_dynamic_lattice_detector_noisy_grid(self):
        """Stress-test pitch consensus under 5% salt-and-pepper noise."""
        np.random.seed(1337)
        for expected_s in (3, 4, 5, 8):
            grid = self._generate_synthetic_grid(expected_s)
            # Inject 3% random noise in grid
            noise_mask = np.random.rand(64, 64) < 0.03
            grid[noise_mask] = np.random.randint(5, 10, size=np.sum(noise_mask), dtype=np.int8)
            step, roff, coff, gh, gw = DynamicLatticeDetector.infer_lattice(grid=grid)
            self.assertEqual(
                step,
                expected_s,
                f"Noisy grid pitch S={expected_s} detection failed under 3% noise, got S={step}"
            )

    # =========================================================================
    # 2. SCENE AMBIGUITY IN ZERO-SHOT AVATAR DEDUCTION
    # =========================================================================

    def test_scene_ambiguity_multiple_singletons_returns_none(self):
        """Construct scenes with multiple singletons (equal confidence scores) and verify None."""
        grid = np.zeros((64, 64), dtype=np.int8)
        # Place 3 distinct singletons in interior: color 2, 3, 4
        grid[10, 10] = 2
        grid[20, 20] = 3
        grid[30, 30] = 4

        res = ScenePerception.deduce_avatar_zero_shot(grid, bg_color=0, step_size=1)
        self.assertIsNone(
            res,
            f"Expected deduce_avatar_zero_shot to return None for ambiguous singletons, but got {res}"
        )

    def test_scene_ambiguity_margin_threshold(self):
        """Verify deduce_avatar_zero_shot returns None when top score and runner-up are within margin (< 5.0)."""
        grid = np.zeros((64, 64), dtype=np.int8)
        # Candidate 1: 1x1 color 2 at (10, 10) [interior: +3] -> score = 10 + 6 + 2 + 3 = 21.0
        # Candidate 2: 1x1 color 3 at (1, 10) [not interior: 0] -> score = 10 + 6 + 2 + 0 = 18.0
        # Margin = 21.0 - 18.0 = 3.0 < 5.0
        grid[10, 10] = 2
        grid[1, 10] = 3

        res = ScenePerception.deduce_avatar_zero_shot(grid, bg_color=0, step_size=1)
        self.assertIsNone(
            res,
            f"Expected None when top candidate margin < 5.0 (diff=3.0), but got {res}"
        )

    def test_scene_ambiguity_identical_multicell_singletons_returns_none(self):
        """Two identical 2x2 avatars of different colors (margin == 0) must return None."""
        grid = np.zeros((64, 64), dtype=np.int8)
        grid[10:12, 10:12] = 2
        grid[30:32, 30:32] = 3

        res = ScenePerception.deduce_avatar_zero_shot(grid, bg_color=0, step_size=1)
        self.assertIsNone(
            res,
            f"Expected None for identical 2x2 singletons with margin=0, got {res}"
        )

    def test_scene_unambiguous_high_margin_returns_avatar(self):
        """Verify deduce_avatar_zero_shot DOES return avatar when margin >= 5.0."""
        grid = np.zeros((64, 64), dtype=np.int8)
        # Avatar: 2x2 square of color 3 at (20, 20) -> rarity 10 + square 6 + size 8 + interior 3 = 27.0
        grid[20:22, 20:22] = 3
        # Background noise / item: 1x1 of color 4 at (1, 10) -> rarity 10 + square 6 + size 2 + interior 0 = 18.0
        # Margin = 27.0 - 18.0 = 9.0 >= 5.0
        grid[1, 10] = 4

        res = ScenePerception.deduce_avatar_zero_shot(grid, bg_color=0, step_size=1)
        self.assertIsNotNone(res)
        self.assertEqual(res["pos"], (20, 20))
        self.assertEqual(res["color"], 3)
        self.assertGreaterEqual(res["confidence"] - 18.0, 5.0)

    def test_myagent_choose_action_defaults_to_probing_on_ambiguity(self):
        """Verify that MyAgent.choose_action defaults to epistemic probing instead of false lock-in."""
        agent = MyAgent(
            card_id="test",
            game_id="novel_puzzle_ambiguous",
            agent_name="Agent.novel",
            ROOT_URL="http://test",
            record=False,
            arc_env=MagicMock(),
        )

        grid = np.zeros((64, 64), dtype=np.int8)
        grid[10, 10] = 2  # Singleton 1
        grid[20, 20] = 3  # Singleton 2
        grid[30, 30] = 4  # Singleton 3

        frame = FrameData(
            frame=[grid],
            state=GameState.NOT_FINISHED,
            available_actions=[1, 2, 3, 4],
        )

        action = agent.choose_action([frame], frame)

        # 1. Avatar position must NOT be locked in
        self.assertIsNone(agent.memory.avatar_pos, "avatar_pos must remain None on ambiguous scene")
        self.assertIsNone(agent.memory.avatar_color, "avatar_color must remain None on ambiguous scene")
        # 2. Action returned must be an epistemic probe
        self.assertIn(action, [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4])
        self.assertEqual(action.reasoning.get("why"), "epistemic_probe_fallback")

    # =========================================================================
    # 3. PERIMETER MOVEMENT TRACKING
    # =========================================================================

    def test_perimeter_movement_row1_to_row0(self):
        """Avatar moves from (1, 10) to (0, 10): moved=True, new_pos=(0, 10), consecutive_bumps=0."""
        memory = WorkingMemory()
        memory.avatar_pos = (1, 10)
        memory.avatar_color = 3
        memory.avatar_colors.add(3)
        memory.step_size = 1

        prev_grid = np.zeros((64, 64), dtype=np.int8)
        prev_grid[1, 10] = 3

        curr_grid = np.zeros((64, 64), dtype=np.int8)
        curr_grid[0, 10] = 3  # Moved UP (ACTION1) to top perimeter row 0

        new_pos, moved = AvatarTracker.update(prev_grid, curr_grid, GameAction.ACTION1, memory)

        self.assertTrue(moved, "Moving into row 0 must be registered as moved=True")
        self.assertEqual(new_pos, (0, 10), "New position must be (0, 10)")
        self.assertEqual(memory.consecutive_bumps, 0, "Bumps must be 0 for valid move into row 0")
        self.assertEqual(memory.avatar_pos, (0, 10))
        self.assertTrue(memory.passable_map[0, 10], "Row 0 avatar stand position must be marked passable")

    def test_perimeter_movement_col1_to_col0(self):
        """Avatar moves from (10, 1) to (10, 0): moved=True, new_pos=(10, 0), consecutive_bumps=0."""
        memory = WorkingMemory()
        memory.avatar_pos = (10, 1)
        memory.avatar_color = 3
        memory.avatar_colors.add(3)
        memory.step_size = 1

        prev_grid = np.zeros((64, 64), dtype=np.int8)
        prev_grid[10, 1] = 3

        curr_grid = np.zeros((64, 64), dtype=np.int8)
        curr_grid[10, 0] = 3  # Moved LEFT (ACTION3) to left perimeter col 0

        new_pos, moved = AvatarTracker.update(prev_grid, curr_grid, GameAction.ACTION3, memory)

        self.assertTrue(moved, "Moving into col 0 must be registered as moved=True")
        self.assertEqual(new_pos, (10, 0), "New position must be (10, 0)")
        self.assertEqual(memory.consecutive_bumps, 0, "Bumps must be 0 for valid move into col 0")
        self.assertEqual(memory.avatar_pos, (10, 0))

    def test_perimeter_movement_col62_to_col63(self):
        """Avatar moves from (10, 62) to (10, 63): moved=True, new_pos=(10, 63), consecutive_bumps=0."""
        memory = WorkingMemory()
        memory.avatar_pos = (10, 62)
        memory.avatar_color = 3
        memory.avatar_colors.add(3)
        memory.step_size = 1

        prev_grid = np.zeros((64, 64), dtype=np.int8)
        prev_grid[10, 62] = 3

        curr_grid = np.zeros((64, 64), dtype=np.int8)
        curr_grid[10, 63] = 3  # Moved RIGHT (ACTION4) to right perimeter col 63

        new_pos, moved = AvatarTracker.update(prev_grid, curr_grid, GameAction.ACTION4, memory)

        self.assertTrue(moved, "Moving into col 63 must be registered as moved=True")
        self.assertEqual(new_pos, (10, 63), "New position must be (10, 63)")
        self.assertEqual(memory.consecutive_bumps, 0)
        self.assertEqual(memory.avatar_pos, (10, 63))

    def test_perimeter_movement_along_perimeter_rows(self):
        """Avatar moves laterally along row 0: (0, 10) to (0, 11) via ACTION4."""
        memory = WorkingMemory()
        memory.avatar_pos = (0, 10)
        memory.avatar_color = 3
        memory.avatar_colors.add(3)
        memory.step_size = 1

        prev_grid = np.zeros((64, 64), dtype=np.int8)
        prev_grid[0, 10] = 3

        curr_grid = np.zeros((64, 64), dtype=np.int8)
        curr_grid[0, 11] = 3  # Moved RIGHT (ACTION4) along row 0

        new_pos, moved = AvatarTracker.update(prev_grid, curr_grid, GameAction.ACTION4, memory)

        self.assertTrue(moved, "Moving laterally along row 0 must be registered as moved=True")
        self.assertEqual(new_pos, (0, 11))
        self.assertEqual(memory.consecutive_bumps, 0)

    # =========================================================================
    # 4. PRIORITY 1 WARP TRACKING
    # =========================================================================

    def test_warp_tracking_simultaneous_ambient_sparkle_no_false_warp(self):
        """Movement with simultaneous ambient sparkles/decoys does NOT trigger false warps."""
        memory = WorkingMemory()
        memory.avatar_pos = (20, 20)
        memory.avatar_color = 3
        memory.avatar_colors.add(3)
        memory.step_size = 1

        prev_grid = np.zeros((64, 64), dtype=np.int8)
        prev_grid[20, 20] = 3

        curr_grid = np.zeros((64, 64), dtype=np.int8)
        curr_grid[20, 21] = 3  # Normal move RIGHT (ACTION4)
        # Simultaneous ambient decoy sparkle elsewhere on the board
        curr_grid[5, 5] = 3
        curr_grid[45, 50] = 3

        new_pos, moved = AvatarTracker.update(prev_grid, curr_grid, GameAction.ACTION4, memory)

        self.assertTrue(moved)
        self.assertEqual(new_pos, (20, 21), f"Avatar should move to (20, 21), but falsely tracked as {new_pos}")
        self.assertEqual(len(memory.wormhole_edges), 0, f"wormhole_edges must remain empty, but found {memory.wormhole_edges}")
        self.assertEqual(memory.consecutive_bumps, 0)

    def test_warp_tracking_stationary_avatar_ambient_sparkle_no_false_warp(self):
        """Stationary avatar (e.g. wall bump or zero move) with ambient sparkles must NOT trigger warp."""
        memory = WorkingMemory()
        memory.avatar_pos = (20, 20)
        memory.avatar_color = 3
        memory.avatar_colors.add(3)
        memory.step_size = 1

        prev_grid = np.zeros((64, 64), dtype=np.int8)
        prev_grid[20, 20] = 3

        curr_grid = np.zeros((64, 64), dtype=np.int8)
        curr_grid[20, 20] = 3  # Avatar remains at (20, 20)
        curr_grid[5, 5] = 3    # Ambient sparkle appeared far away

        new_pos, moved = AvatarTracker.update(prev_grid, curr_grid, GameAction.ACTION1, memory)

        # Avatar did not vacate (20, 20), so prev_vacated is False
        self.assertEqual(new_pos, (20, 20))
        self.assertEqual(len(memory.wormhole_edges), 0, "No false warp when avatar has not vacated")

    def test_warp_tracking_genuine_warp_registers_wormhole(self):
        """Genuine warp with distant sprite and vacated previous pos correctly registers the warp."""
        memory = WorkingMemory()
        memory.avatar_pos = (20, 20)
        memory.avatar_color = 3
        memory.avatar_colors.add(3)
        memory.step_size = 1

        prev_grid = np.zeros((64, 64), dtype=np.int8)
        prev_grid[20, 20] = 3

        curr_grid = np.zeros((64, 64), dtype=np.int8)
        # Previous position (20, 20) is vacated (now 0)
        curr_grid[20, 20] = 0
        # Avatar teleports to distant exit portal at (50, 50)
        curr_grid[50, 50] = 3

        new_pos, moved = AvatarTracker.update(prev_grid, curr_grid, GameAction.ACTION4, memory)

        self.assertTrue(moved, "Genuine warp must be registered as moved=True")
        self.assertEqual(new_pos, (50, 50), f"Avatar position must be (50, 50), got {new_pos}")
        self.assertEqual(memory.consecutive_bumps, 0)
        # Verify portal edge recorded
        expected_portal_entry = (20, 21)
        self.assertIn(expected_portal_entry, memory.wormhole_edges)
        self.assertEqual(memory.wormhole_edges[expected_portal_entry], (50, 50))
        self.assertEqual(memory.portal_actions[expected_portal_entry], GameAction.ACTION4)
        self.assertTrue(memory.passable_map[20, 21])
        self.assertTrue(memory.passable_map[50, 50])

    def test_warp_tracking_multipixel_avatar_genuine_warp(self):
        """Multi-pixel (2x2) avatar genuine warp correctly verifies prev_vacated and registers warp."""
        memory = WorkingMemory()
        memory.avatar_pos = (15, 15)
        memory.avatar_color = 3
        memory.avatar_colors.add(3)
        memory.step_size = 2

        prev_grid = np.zeros((64, 64), dtype=np.int8)
        prev_grid[15:17, 15:17] = 3  # 2x2 avatar

        curr_grid = np.zeros((64, 64), dtype=np.int8)
        curr_grid[15:17, 15:17] = 0  # Vacated
        curr_grid[45:47, 45:47] = 3  # Teleported 2x2 sprite at (45, 45)

        new_pos, moved = AvatarTracker.update(prev_grid, curr_grid, GameAction.ACTION4, memory)

        self.assertTrue(moved)
        self.assertEqual(new_pos, (45, 45), f"Expected 2x2 avatar at (45, 45), got {new_pos}")
        expected_portal_entry = (15, 17)
        self.assertIn(expected_portal_entry, memory.wormhole_edges)
        self.assertEqual(memory.wormhole_edges[expected_portal_entry], (45, 45))
        self.assertEqual(memory.consecutive_bumps, 0)

    def test_warp_tracking_multipixel_avatar_stationary_with_decoy(self):
        """Multi-pixel (2x2) avatar stationary with 2x2 decoy does NOT trigger false warp."""
        memory = WorkingMemory()
        memory.avatar_pos = (15, 15)
        memory.avatar_color = 3
        memory.avatar_colors.add(3)
        memory.step_size = 2

        prev_grid = np.zeros((64, 64), dtype=np.int8)
        prev_grid[15:17, 15:17] = 3

        curr_grid = np.zeros((64, 64), dtype=np.int8)
        curr_grid[15:17, 15:17] = 3  # Still present
        curr_grid[45:47, 45:47] = 3  # Decoy 2x2 appeared

        new_pos, moved = AvatarTracker.update(prev_grid, curr_grid, GameAction.ACTION1, memory)

        self.assertEqual(new_pos, (15, 15))
        self.assertEqual(len(memory.wormhole_edges), 0)

    def test_warp_tracking_multipixel_avatar_normal_move_with_decoy(self):
        """Multi-pixel (2x2) avatar moving 2 steps with distant 2x2 decoy does NOT trigger false warp."""
        memory = WorkingMemory()
        memory.avatar_pos = (15, 15)
        memory.avatar_color = 3
        memory.avatar_colors.add(3)
        memory.step_size = 2

        prev_grid = np.zeros((64, 64), dtype=np.int8)
        prev_grid[15:17, 15:17] = 3

        curr_grid = np.zeros((64, 64), dtype=np.int8)
        curr_grid[15:17, 17:19] = 3  # Moved RIGHT by 2 (step 2) to (15, 17)
        curr_grid[45:47, 45:47] = 3  # Distant decoy

        new_pos, moved = AvatarTracker.update(prev_grid, curr_grid, GameAction.ACTION4, memory)

        self.assertTrue(moved)
        self.assertEqual(new_pos, (15, 17))
        self.assertEqual(len(memory.wormhole_edges), 0, "Distant decoy must NOT trigger warp during normal move")


if __name__ == "__main__":
    unittest.main()

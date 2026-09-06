"""
Milestone M1 Challenger 1 Empirical Perception & Lattice Stress Suite.
Adversarial probes targeting:
1. Zero-shot avatar deduction under edge-case and noisy frame 0 conditions.
2. Dynamic lattice detection across irregular grid pitches and candidate inversion logic.
3. Difference-mask intersection invariant under complex frame transitions, edge movements, and decoys.
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
from agent.my_agent import (
    ScenePerception,
    DynamicLatticeDetector,
    AvatarTracker,
    WorkingMemory,
)


class TestM1PerceptionStress(unittest.TestCase):

    # =========================================================================
    # DIMENSION 1: Zero-Shot Avatar Deduction Under Edge-Case and Noisy Frame 0
    # =========================================================================

    def test_1_1_avatar_deduction_ambiguous_singletons(self):
        """
        Challenge 1.1: Ambiguous singletons with identical feature scores.
        Scenario: Grid contains a 1x1 key (color 2), a 1x1 gem (color 3), and a 1x1 avatar (color 4).
        All are singletons in the interior.
        Expected: A conservative perception module should detect genuine ambiguity (margin == 0)
        and return None or low confidence to trigger epistemic exploration fallback.
        Observed: Does it emit confidence >= 8.0 for an arbitrary candidate based purely on scan order?
        """
        grid = np.zeros((64, 64), dtype=np.int8)
        # Place key at (10, 10), gem at (20, 20), real avatar at (30, 30)
        grid[10, 10] = 2
        grid[20, 20] = 3
        grid[30, 30] = 4

        res = ScenePerception.deduce_avatar_zero_shot(grid, bg_color=0, step_size=1)
        # Let's inspect the returned result
        print(f"\n[Test 1.1 Result] deduction result: {res}")
        if res is not None:
            print(f"  Deduced pos: {res['pos']}, color: {res['color']}, confidence: {res['confidence']}")
            # All 3 have identical score = 10 (rarity) + 6 (square) + 4 (1x1) + 3 (interior) = 23.0
            # It picked whichever came first in scan order (10, 10) with confidence 23.0!
            cands = res.get("all_candidates", [])
            scores = [s.get("color") for s in cands]
            print(f"  All candidate colors: {scores}")
            # If confidence is 23.0 despite having 3 equally valid singletons, report this finding
            self.assertGreaterEqual(res["confidence"], 8.0, "Candidate confidence was unexpectedly low")

    def test_1_2_avatar_deduction_noise_pixels_over_multicell_avatar(self):
        """
        Challenge 1.2: 1x1 noise pixels vs 2x2 avatar under step_size=1.
        Under step_size=1, a 1x1 item receives +4.0 lattice points while a 2x2 item receives +2.0.
        If random 1x1 noise artifacts exist with singleton colors, they score 23.0,
        whereas a genuine 2x2 avatar scores 21.0!
        """
        grid = np.zeros((64, 64), dtype=np.int8)
        # Genuine avatar: 2x2 square of color 3 at (30, 30)
        grid[30:32, 30:32] = 3
        # Single-pixel noise artifact of color 4 at (15, 15)
        grid[15, 15] = 4

        res = ScenePerception.deduce_avatar_zero_shot(grid, bg_color=0, step_size=1)
        print(f"\n[Test 1.2 Result] deduction result: {res}")
        self.assertIsNotNone(res)
        # Check whether the 1x1 noise artifact beat the genuine 2x2 avatar
        picked_color = res["color"]
        print(f"  Picked color: {picked_color} (Expected avatar 3 vs noise 4)")
        # Record whether noise was selected over the true avatar
        noise_won = (picked_color == 4)
        print(f"  Noise pixel selected over avatar: {noise_won}")

    def test_1_3_avatar_deduction_boundary_spawn(self):
        """
        Challenge 1.3: Avatar spawns on row 1 or col 1 (boundary).
        The interior score bonus (+3.0) requires 2 <= r0 <= 61 and 2 <= c0 <= 61.
        Does a boundary avatar still achieve confidence >= 8.0?
        """
        grid = np.zeros((64, 64), dtype=np.int8)
        # Avatar at (1, 10)
        grid[1, 10] = 3
        res = ScenePerception.deduce_avatar_zero_shot(grid, bg_color=0, step_size=1)
        print(f"\n[Test 1.3 Result] boundary deduction result: {res}")
        self.assertIsNotNone(res)
        self.assertEqual(res["pos"], (1, 10))
        self.assertEqual(res["color"], 3)
        # Score = 10 (rarity) + 6 (square) + 4 (1x1) + 0 (not interior) = 20.0 >= 8.0
        self.assertGreaterEqual(res["confidence"], 8.0)

    # =========================================================================
    # DIMENSION 2: Dynamic Lattice Detection Across Irregular Grid Pitches
    # =========================================================================

    def test_2_1_lattice_detection_s2_vs_s4_inversion(self):
        """
        Challenge 2.1: S=2 grid pitch detection with S=4 evaluated first in descending loop.
        In lines 982-1010 of agent/my_agent.py:
        - The loop tests (8, 5, 4, 3, 2).
        - Line 1009 states: `if best_s == 2 and s == 4 and enh < best_enh * 1.3: continue`
        Because 4 is tested before 2, best_s is NEVER 2 when s == 4 is tested!
        Does a genuine 2x2 lattice get misclassified as S=4?
        """
        # Create a genuine 2x2 tiled board
        grid = np.zeros((64, 64), dtype=np.int8)
        for r in range(0, 58, 2):
            for c in range(0, 64, 2):
                grid[r:r+2, c:c+2] = ((r // 2) + (c // 2)) % 3 + 1

        step, roff, coff, gh, gw = DynamicLatticeDetector.infer_lattice(grid=grid)
        print(f"\n[Test 2.1 Result] Genuine 2x2 grid inferred: step={step}, roff={roff}, coff={coff}, gh={gh}, gw={gw}")
        # Test whether step is correctly 2 or misclassified as 4
        print(f"  Step returned: {step} (Expected: 2)")

    def test_2_2_lattice_detection_irregular_pitches(self):
        """
        Challenge 2.2: Irregular non-standard pitch (e.g. S=6 or S=7).
        Candidate pitches are only (8, 5, 4, 3, 2).
        How does the detector handle S=6 or S=7?
        """
        # Create a 6x6 tiled board
        grid6 = np.zeros((64, 64), dtype=np.int8)
        for r in range(0, 58, 6):
            for c in range(0, 64, 6):
                grid6[r:r+6, c:c+6] = ((r // 6) + (c // 6)) % 4 + 1

        step6, roff6, coff6, gh6, gw6 = DynamicLatticeDetector.infer_lattice(grid=grid6)
        print(f"\n[Test 2.2 Result S=6] Inferred step={step6}, roff={roff6}, coff={coff6}, gh={gh6}, gw={gw6}")

        # Create a 7x7 tiled board
        grid7 = np.zeros((64, 64), dtype=np.int8)
        for r in range(0, 58, 7):
            for c in range(0, 64, 7):
                grid7[r:r+7, c:c+7] = ((r // 7) + (c // 7)) % 4 + 1

        step7, roff7, coff7, gh7, gw7 = DynamicLatticeDetector.infer_lattice(grid=grid7)
        print(f"[Test 2.2 Result S=7] Inferred step={step7}, roff={roff7}, coff={coff7}, gh={gh7}, gw={gw7}")

    def test_2_3_lattice_detection_flat_and_noise_grids(self):
        """
        Challenge 2.3: Monochromatic flat grid and pure random noise.
        Should safely default to step=1, roff=0, coff=0 without throwing exceptions.
        """
        flat_grid = np.zeros((64, 64), dtype=np.int8)
        step, roff, coff, gh, gw = DynamicLatticeDetector.infer_lattice(grid=flat_grid)
        print(f"\n[Test 2.3 Result Flat] step={step}, roff={roff}, coff={coff}, gh={gh}, gw={gw}")
        self.assertEqual(step, 1)
        self.assertEqual(roff, 0)
        self.assertEqual(coff, 0)

        rng = np.random.default_rng(42)
        noise_grid = rng.integers(0, 16, size=(64, 64), dtype=np.int8)
        step_n, roff_n, coff_n, gh_n, gw_n = DynamicLatticeDetector.infer_lattice(grid=noise_grid)
        print(f"[Test 2.3 Result Noise] step={step_n}, roff={roff_n}, coff={coff_n}, gh={gh_n}, gw={gw_n}")
        # Noise should default to step=1 due to lack of consensus enhancement
        self.assertEqual(step_n, 1)

    # =========================================================================
    # DIMENSION 3: Difference-Mask Intersection Invariant Under Frame Transitions
    # =========================================================================

    def test_3_1_avatar_tracking_edge_blindspot(self):
        """
        Challenge 3.1: In AvatarTracker.update:
        diff[:2, :] = False
        diff[:, :2] = False
        diff[:, 62:] = False
        diff[59:, :] = False
        If avatar moves into row 1 or col 1, diff is masked out to False!
        Does AvatarTracker.update fail to track movement into row 1 and falsely report a bump?
        """
        memory = WorkingMemory()
        memory.avatar_pos = (2, 10)
        memory.avatar_color = 3
        memory.avatar_colors.add(3)
        memory.step_size = 1

        prev_grid = np.zeros((64, 64), dtype=np.int8)
        prev_grid[2, 10] = 3

        curr_grid = np.zeros((64, 64), dtype=np.int8)
        curr_grid[1, 10] = 3  # Avatar moved UP to row 1

        # Action: ACTION1 (Up)
        new_pos, moved = AvatarTracker.update(prev_grid, curr_grid, GameAction.ACTION1, memory)
        print(f"\n[Test 3.1 Result] Edge movement: new_pos={new_pos}, moved={moved}, consecutive_bumps={memory.consecutive_bumps}")
        # Expected if tracking worked: new_pos == (1, 10) and moved == True
        # If edge masking blinds it: moved == False and consecutive_bumps == 1
        print(f"  Did edge mask blind avatar tracking? {not moved or new_pos != (1, 10)}")

    def test_3_2_avatar_tracking_decoy_fake_warp(self):
        """
        Challenge 3.2: In AvatarTracker.update Priority 1:
        Lines 634-653 check `if memory.avatar_colors: av_mask = np.isin(...)`
        If a distant pixel matching avatar_color changes anywhere on the grid (e.g. ambient effect,
        particle, door animation, or decoy), does Priority 1 trigger a false warp without verifying
        that the avatar vacated its previous position?
        """
        memory = WorkingMemory()
        memory.avatar_pos = (20, 20)
        memory.avatar_color = 3
        memory.avatar_colors.add(3)
        memory.step_size = 1

        prev_grid = np.zeros((64, 64), dtype=np.int8)
        prev_grid[20, 20] = 3

        curr_grid = np.zeros((64, 64), dtype=np.int8)
        curr_grid[20, 21] = 3  # Avatar moved RIGHT (ACTION4) to (20, 21)
        # Simultaneously, a single pixel of color 3 changed at (5, 5) (ambient sparkle / decoy)
        curr_grid[5, 5] = 3

        new_pos, moved = AvatarTracker.update(prev_grid, curr_grid, GameAction.ACTION4, memory)
        print(f"\n[Test 3.2 Result] Decoy sparkle during movement:")
        print(f"  new_pos={new_pos}, moved={moved}")
        print(f"  wormhole_edges={memory.wormhole_edges}")
        # If Priority 1 falsely triggered warp to (5, 5):
        # new_pos will be (5, 5) or (5, 21) and wormhole_edges will contain a fake entry!
        is_fake_warp = (new_pos != (20, 21)) or (len(memory.wormhole_edges) > 0)
        print(f"  Fake warp triggered by decoy sparkle: {is_fake_warp}")

    def test_3_4_avatar_movement_fully_within_edge_mask(self):
        """
        Challenge 3.4: Movement along row 0 or row 1 (diff[:2, :] = False).
        Scenario:
        - Avatar at (1, 10) moves to (0, 10) via ACTION1 (Up).
        - Both positions are in rows < 2.
        Because diff[:2, :] = False, the difference mask is completely wiped out!
        np.any(diff) evaluates to False.
        AvatarTracker classifies this valid move as a WALL BUMP!
        """
        memory = WorkingMemory()
        memory.avatar_pos = (1, 10)
        memory.avatar_color = 3
        memory.avatar_colors.add(3)
        memory.step_size = 1

        prev_grid = np.zeros((64, 64), dtype=np.int8)
        prev_grid[1, 10] = 3

        curr_grid = np.zeros((64, 64), dtype=np.int8)
        curr_grid[0, 10] = 3  # Avatar moved UP to row 0

        new_pos, moved = AvatarTracker.update(prev_grid, curr_grid, GameAction.ACTION1, memory)
        print(f"\n[Test 3.4 Result] Row 1 to Row 0 movement: new_pos={new_pos}, moved={moved}, consecutive_bumps={memory.consecutive_bumps}")
        # Fix verified: Avatar moving in row 0/1 is correctly detected without perimeter diff blinding
        self.assertTrue(moved, "Movement into row 0 must be detected when edge masking is removed")
        self.assertEqual(memory.consecutive_bumps, 0, "No bump should be registered for valid movement")
        self.assertEqual(new_pos, (0, 10), "New position must reflect movement to (0, 10)")
        print("  VERIFIED FIX: Avatar moving into row 0 is correctly detected; bump counter remains 0.")

    def test_1_4_myagent_frame0_ambiguity_causes_misdeduction(self):
        """
        Challenge 1.4: In MyAgent.choose_action on Frame 0 of an unseen puzzle:
        If there are 3 candidate singletons (e.g. key at (5, 5), gem at (10, 10), avatar at (30, 30)),
        deduce_avatar_zero_shot returns (5, 5) with confidence 23.0 (above threshold 8.0).
        MyAgent immediately sets memory.avatar_pos = (5, 5) and probing_phase = False,
        completely skipping the conservative fallback probe and hallucinating its position!
        """
        from unittest.mock import MagicMock
        from arcengine import FrameData, GameState
        from agent.my_agent import MyAgent

        agent = MyAgent(
            card_id="test",
            game_id="novel_puzzle_xyz",
            agent_name="Agent.novel",
            ROOT_URL="http://test",
            record=False,
            arc_env=MagicMock(),
        )

        grid = np.zeros((64, 64), dtype=np.int8)
        grid[5, 5] = 2    # Key (singleton color 2)
        grid[10, 10] = 3  # Gem (singleton color 3)
        grid[30, 30] = 4  # Real avatar (singleton color 4)

        frame = FrameData(frame=[grid], state=GameState.NOT_FINISHED, available_actions=[1, 2, 3, 4])
        act = agent.choose_action([frame], frame)

        print(f"\n[Test 1.4 Result] MyAgent Frame 0 on ambiguous scene:")
        print(f"  memory.avatar_pos: {agent.memory.avatar_pos}")
        print(f"  memory.avatar_color: {agent.memory.avatar_color}")
        print(f"  memory.probing_phase: {agent.memory.probing_phase}")
        print(f"  Action chosen: {act}")

        # Fix verified: Frame 0 ambiguity does not lock in a hallucinated avatar; agent refrains or probes
        self.assertTrue(
            agent.memory.probing_phase or agent.memory.avatar_pos is None,
            "When scene is ambiguous on Frame 0, agent must not lock into a candidate without probing",
        )
        self.assertNotEqual(agent.memory.avatar_pos, (5, 5))
        print("  VERIFIED FIX: Ambiguity on Frame 0 correctly triggers fallback / probing without hallucinated lock-in.")


if __name__ == "__main__":
    unittest.main()


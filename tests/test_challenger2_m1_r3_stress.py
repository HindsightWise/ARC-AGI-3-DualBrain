"""
Milestone M1 Challenger 2 (Iteration 3) Adversarial Verification Suite.

Empirically stress-tests dispatch latching and robustness in agent/my_agent.py:
1. Dispatch Latching Stress:
   - Verify that _substrate_plan_attempted correctly resets on level completion (curr_levels > self.last_levels_completed) and terminal reset (GAME_OVER, RESET).
   - Verify that when a level transitions, the agent correctly attempts substrate planning for the new level frame without getting permanently locked out.
   - Verify that games offering both directional moves and click moves dispatch cleanly without deadlocks or unhandled exceptions.
2. Run stress tests across adversarial degenerate grids, terminal resets, and novel games.
"""

from __future__ import annotations

import collections
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
    MyAgent,
    MirroredTwinPlanner,
    LabyrinthPlanner,
    ScenePerception,
    DynamicLatticeDetector,
)


class TestM1R3DispatchLatchingAndRobustness(unittest.TestCase):
    """Empirical challenger test suite for dispatch latching, level transitions, and mixed dispatch."""

    def _create_agent(self, game_id: str = "test-game") -> MyAgent:
        return MyAgent(
            card_id="test",
            game_id=game_id,
            agent_name=f"Agent.{game_id}",
            ROOT_URL="http://test",
            record=False,
            arc_env=MagicMock(),
        )

    # =========================================================================
    # 1. DISPATCH LATCHING: RESET ON LEVEL TRANSITION AND TERMINAL RESET
    # =========================================================================

    def test_substrate_latch_resets_on_level_completion_and_terminal_reset(self):
        """
        Verify that _substrate_plan_attempted correctly resets on level completion
        (curr_levels > self.last_levels_completed) and terminal reset (GAME_OVER, RESET).
        """
        agent = self._create_agent()
        grid = np.zeros((64, 64), dtype=np.int8)

        # Initial state: Level 0
        f0 = FrameData(frame=[grid], state=GameState.NOT_FINISHED, levels_completed=0, available_actions=[1, 2, 3, 4])
        agent.choose_action([f0], f0)
        self.assertTrue(
            agent._substrate_plan_attempted or getattr(agent, "_substrate_attempts", 0) >= 1,
            "On Level 0 step 1, _substrate_plan_attempted or _substrate_attempts should be latched after attempt.",
        )

        # Level completion transition: levels_completed becomes 1
        f1 = FrameData(frame=[grid], state=GameState.NOT_FINISHED, levels_completed=1, available_actions=[1, 2, 3, 4])
        agent.choose_action([f0, f1], f1)
        # Check that the level transition block executed line 2144
        self.assertEqual(agent.last_levels_completed, 1)

        # Terminal reset: GAME_OVER
        f_over = FrameData(frame=[grid], state=GameState.GAME_OVER, available_actions=[0])
        agent._substrate_plan_attempted = True  # manually simulate latched state
        act_over = agent.choose_action([f0, f1, f_over], f_over)
        self.assertEqual(act_over, GameAction.RESET)
        self.assertFalse(
            agent._substrate_plan_attempted,
            "_substrate_plan_attempted should be reset to False on GAME_OVER.",
        )
        self.assertEqual(agent.last_levels_completed, 0)

    # =========================================================================
    # 2. DISPATCH LATCHING: SUBSTRATE PLANNING ATTEMPTED ON NEW LEVEL FRAME
    # =========================================================================

    def test_substrate_planning_attempted_on_level_transition(self):
        """
        Adversarial Test 1.2:
        Verify that when a level transitions, the agent correctly attempts substrate
        planning for the new level frame without getting permanently locked out.
        
        Scenario:
        - Agent completes Level 0.
        - Frame transitions to Level 1 (levels_completed=1).
        - Level 1 is a valid mirrored twin puzzle (m0r0 archetype) that CAN be solved
          by MirroredTwinPlanner if substrate planning is executed.
        - IF substrate planning is locked out by 'self.last_levels_completed == 0',
          MirroredTwinPlanner will never be called, fleet_plan_queue will remain empty,
          and the agent will be permanently locked out of substrate planning on Level 1.
        """
        agent = self._create_agent(game_id="novel-twin-game")

        # Level 0 frame: blank grid, completes level 0
        grid0 = np.zeros((64, 64), dtype=np.int8)
        f0 = FrameData(frame=[grid0], state=GameState.NOT_FINISHED, levels_completed=0, available_actions=[1, 2, 3, 4])
        agent.choose_action([f0], f0)
        self.assertTrue(
            agent._substrate_plan_attempted or getattr(agent, "_substrate_attempts", 0) >= 1,
            "Should have attempted substrate planning on Level 0",
        )

        # Now level transitions to Level 1
        # Construct a valid solvable mirrored twin grid for Level 1
        gw, gh = 10, 10
        grid1 = np.zeros((64, 64), dtype=np.int8)
        grid1[0:gh, 0] = 1
        grid1[0:gh, gw-1] = 0
        grid1[0, 0:gw] = 1
        grid1[gh-1, 0:gw] = 1
        grid1[5, 1] = 3
        grid1[5, 8] = 3

        f1 = FrameData(frame=[grid1], state=GameState.NOT_FINISHED, levels_completed=1, available_actions=[1, 2, 3, 4])

        # Step into Level 1
        act = agent.choose_action([f0, f1], f1)

        # In a sound system, substrate planning MUST be attempted for Level 1.
        # This implies:
        # 1. _substrate_plan_attempted or _fleet_plan_started should be True.
        # 2. MirroredTwinPlanner should have planned the twin rendezvous into fleet_plan_queue.
        self.assertTrue(
            agent._substrate_plan_attempted or agent._fleet_plan_started,
            "CRITICAL BUG: Substrate planning was NOT attempted on Level 1! "
            "Agent is permanently locked out of substrate planning on level progression "
            "due to 'self.last_levels_completed == 0' gate on lines 2168 & 2196.",
        )
        self.assertTrue(
            agent._fleet_plan_started,
            "CRITICAL BUG: Fleet plan for mirrored twin puzzle on Level 1 was never generated "
            "because substrate planning was locked out!",
        )

    # =========================================================================
    # 3. MULTI-ACTION DISPATCH: GAMES WITH BOTH DIRECTIONAL AND CLICK MOVES
    # =========================================================================

    def test_mixed_actions_directional_and_click_dispatch(self):
        """
        Verify that games offering both directional moves and click moves dispatch cleanly
        without deadlocks or unhandled exceptions.
        """
        agent = self._create_agent(game_id="mixed-action-game")
        rng = np.random.RandomState(42)

        legal_actions = [1, 2, 3, 4, 6]  # ACTION1..4 + ACTION6
        frames = []

        for step_idx in range(50):
            # Create a grid with some colored sprites and walls
            grid = rng.choice([0, 1, 2, 3, 11, 14], size=(64, 64), p=[0.7, 0.1, 0.05, 0.05, 0.05, 0.05]).astype(np.int8)
            # Add a clear avatar-like sprite
            grid[20:24, 20:24] = 9

            frame = FrameData(
                frame=[grid],
                state=GameState.NOT_FINISHED,
                levels_completed=0,
                available_actions=legal_actions,
            )
            frames.append(frame)

            try:
                action = agent.choose_action(frames, frame)
            except Exception as e:
                self.fail(f"Unhandled exception on step {step_idx} in mixed action game: {e}")

            # Verify action is legal
            action_int = action.value if hasattr(action, "value") else int(action)
            self.assertIsNotNone(action, f"Agent returned None on step {step_idx}")
            self.assertIn(action_int, legal_actions, f"Agent returned illegal action {action} on step {step_idx}")

            # If ACTION6 was chosen, verify coordinate safety [0, 63]
            if action == GameAction.ACTION6 or action_int == 6:
                data = getattr(action, "data", None)
                self.assertIsNotNone(data, f"ACTION6 emitted without coordinate data on step {step_idx}")
                self.assertIn("x", data, "ACTION6 data missing 'x'")
                self.assertIn("y", data, "ACTION6 data missing 'y'")
                x, y = data["x"], data["y"]
                self.assertTrue(0 <= x <= 63, f"ACTION6 x coordinate out of bounds: {x}")
                self.assertTrue(0 <= y <= 63, f"ACTION6 y coordinate out of bounds: {y}")

    # =========================================================================
    # 4. ADVERSARIAL DEGENERATE GRIDS AND TERMINAL RESETS
    # =========================================================================

    def test_adversarial_degenerate_grids_and_terminal_resets(self):
        """
        Run stress tests across adversarial degenerate grids, terminal resets, and unusual shapes.
        """
        agent = self._create_agent(game_id="degenerate-stress")

        # 1. 1x1 degenerate grid
        grid_1x1 = np.array([[5]], dtype=np.int8)
        f_1x1 = FrameData(frame=[grid_1x1], state=GameState.NOT_FINISHED, available_actions=[1, 2, 3, 4])
        try:
            act = agent.choose_action([f_1x1], f_1x1)
            self.assertIsNotNone(act)
        except Exception as e:
            self.fail(f"Agent crashed on 1x1 degenerate grid: {e}")

        # 2. Solid color 64x64 grids across all colors 0..15
        for col in range(16):
            grid_solid = np.full((64, 64), col, dtype=np.int8)
            f_solid = FrameData(frame=[grid_solid], state=GameState.NOT_FINISHED, available_actions=[1, 2, 3, 4])
            try:
                act = agent.choose_action([f_solid], f_solid)
                self.assertIsNotNone(act)
            except Exception as e:
                self.fail(f"Agent crashed on solid color {col} grid: {e}")

        # 3. High entropy pure noise grid
        rng = np.random.RandomState(1337)
        grid_noise = rng.randint(0, 16, size=(64, 64), dtype=np.int8)
        f_noise = FrameData(frame=[grid_noise], state=GameState.NOT_FINISHED, available_actions=[1, 2, 3, 4, 5, 6])
        try:
            act = agent.choose_action([f_noise], f_noise)
            self.assertIsNotNone(act)
        except Exception as e:
            self.fail(f"Agent crashed on pure noise grid: {e}")

        # 4. Rapid terminal reset cycling
        for _ in range(10):
            f_reset = FrameData(frame=[grid_noise], state=GameState.GAME_OVER, available_actions=[0])
            act = agent.choose_action([f_reset], f_reset)
            self.assertEqual(act, GameAction.RESET)
            self.assertEqual(len(agent.fleet_plan_queue), 0)
            self.assertFalse(agent._fleet_plan_started)
            self.assertFalse(agent._substrate_plan_attempted)


if __name__ == "__main__":
    unittest.main()

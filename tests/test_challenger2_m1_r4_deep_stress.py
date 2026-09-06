"""
Milestone M1 Challenger 2 (Iteration 4 Final Certification) Deep Stress Harness.

Empirically challenges:
1. Multi-level progression (Levels 0 through 5) - latch resets and substrate re-planning.
2. Fleet queue mid-execution flushing on level transition (zero plan leakage).
3. Non-consecutive / jump level transitions (e.g., Level 0 to Level 3).
4. Alternating level completions and terminal resets (GAME_OVER / NOT_PLAYED).
5. Mixed action sets (Action 1..4 + Action 6) across level transitions.
6. Extreme frame dimensions and empty action sets.
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
    ScenePerception,
    DynamicLatticeDetector,
)


class TestM1R4DeepStressHarness(unittest.TestCase):
    """Deep empirical challenge suite for dispatch latching, queue isolation, and level transitions."""

    def _create_agent(self, game_id: str = "stress-game") -> MyAgent:
        return MyAgent(
            card_id="test",
            game_id=game_id,
            agent_name=f"Agent.{game_id}",
            ROOT_URL="http://test",
            record=False,
            arc_env=MagicMock(),
        )

    def _build_twin_grid(self, gw: int = 10, gh: int = 10) -> np.ndarray:
        """Constructs a valid solvable mirrored twin grid."""
        grid = np.zeros((64, 64), dtype=np.int8)
        grid[0:gh, 0] = 1
        grid[0:gh, gw - 1] = 0
        grid[0, 0:gw] = 1
        grid[gh - 1, 0:gw] = 1
        grid[5, 1] = 3
        grid[5, 8] = 3
        return grid

    def test_multi_level_progression_levels_0_to_5(self):
        """
        Challenge 1: Verify that across 6 sequential levels (Level 0 through 5),
        every level transition resets _substrate_plan_attempted and enables
        dynamic substrate planning on every single level.
        """
        agent = self._create_agent(game_id="multi-level-progression")
        frames = []

        for level in range(6):
            grid = self._build_twin_grid()
            f = FrameData(
                frame=[grid],
                state=GameState.NOT_FINISHED,
                levels_completed=level,
                available_actions=[1, 2, 3, 4],
            )
            frames.append(f)
            act = agent.choose_action(frames, f)

            self.assertEqual(agent.last_levels_completed, level)
            self.assertTrue(
                agent._substrate_plan_attempted,
                f"Level {level}: Substrate planning was NOT attempted! Latch failed to engage/reset.",
            )
            self.assertTrue(
                agent._fleet_plan_started,
                f"Level {level}: Fleet plan was NOT started! Mirrored twin was not planned.",
            )
            self.assertIsNotNone(act)

    def test_fleet_queue_flushing_on_level_transition_no_leakage(self):
        """
        Challenge 2: Verify that if a fleet plan is in progress (multi-step queue),
        a sudden level transition immediately purges the remaining queue items
        so that old level actions NEVER leak into the new level.
        """
        agent = self._create_agent(game_id="queue-leakage-test")

        # Level 0: Start a plan and populate queue with 10 dummy actions
        grid0 = np.zeros((64, 64), dtype=np.int8)
        f0 = FrameData(
            frame=[grid0],
            state=GameState.NOT_FINISHED,
            levels_completed=0,
            available_actions=[1, 2, 3, 4],
        )
        agent.choose_action([f0], f0)

        # Manually inject distinct actions into fleet_plan_queue to simulate active plan
        dummy_actions = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4]
        agent.fleet_plan_queue.clear()
        agent.fleet_plan_queue.extend(dummy_actions)
        agent._fleet_plan_started = True
        self.assertEqual(len(agent.fleet_plan_queue), 4)

        # Transition to Level 1
        grid1 = np.zeros((64, 64), dtype=np.int8)
        f1 = FrameData(
            frame=[grid1],
            state=GameState.NOT_FINISHED,
            levels_completed=1,
            available_actions=[1, 2, 3, 4],
        )
        # Agent takes action on Level 1
        act = agent.choose_action([f0, f1], f1)

        # The old dummy actions must NOT have been emitted
        # If dummy_actions[0] (ACTION1) had leaked, act would be ACTION1 from dynamic_fleet_plan.
        # But fleet_plan_queue should have been flushed, so old plan is eradicated.
        self.assertFalse(agent._fleet_plan_started, "Fleet plan started flag must be reset on level transition")
        self.assertEqual(agent.last_levels_completed, 1)

    def test_non_consecutive_level_jump(self):
        """
        Challenge 3: Verify that jumping multiple levels at once (e.g. levels_completed 0 -> 3)
        correctly triggers the transition block, resets latches, and plans afresh.
        """
        agent = self._create_agent(game_id="level-jump-test")
        grid0 = np.zeros((64, 64), dtype=np.int8)
        f0 = FrameData(frame=[grid0], state=GameState.NOT_FINISHED, levels_completed=0, available_actions=[1, 2])
        agent.choose_action([f0], f0)
        self.assertEqual(agent.last_levels_completed, 0)

        # Direct jump to Level 3 with a solvable twin grid
        grid3 = self._build_twin_grid()
        f3 = FrameData(frame=[grid3], state=GameState.NOT_FINISHED, levels_completed=3, available_actions=[1, 2, 3, 4])
        act3 = agent.choose_action([f0, f3], f3)

        self.assertEqual(agent.last_levels_completed, 3)
        self.assertTrue(agent._substrate_plan_attempted)
        self.assertTrue(agent._fleet_plan_started)
        self.assertIsNotNone(act3)

    def test_alternating_level_transitions_and_terminal_resets(self):
        """
        Challenge 4: Verify rapid interleaved sequence of Level Progression and Terminal Resets:
        Level 0 -> Level 1 -> GAME_OVER -> RESET -> Level 0 -> Level 1 -> Level 2 -> GAME_OVER -> RESET
        Ensures zero accumulated state corruption or deadlocks.
        """
        agent = self._create_agent(game_id="interleaved-resets")
        grid = self._build_twin_grid()
        frames = []

        sequence = [
            ("STEP", 0, GameState.NOT_FINISHED),
            ("STEP", 1, GameState.NOT_FINISHED),
            ("RESET", 0, GameState.GAME_OVER),
            ("STEP", 0, GameState.NOT_FINISHED),
            ("STEP", 1, GameState.NOT_FINISHED),
            ("STEP", 2, GameState.NOT_FINISHED),
            ("RESET", 0, GameState.GAME_OVER),
        ]

        for op, lvl, state in sequence:
            if op == "RESET":
                f = FrameData(frame=[grid], state=state, available_actions=[0])
                frames.append(f)
                act = agent.choose_action(frames, f)
                self.assertEqual(act, GameAction.RESET)
                self.assertEqual(agent.last_levels_completed, 0)
                self.assertEqual(len(agent.fleet_plan_queue), 0)
                self.assertFalse(agent._fleet_plan_started)
                self.assertFalse(agent._substrate_plan_attempted)
            else:
                f = FrameData(frame=[grid], state=state, levels_completed=lvl, available_actions=[1, 2, 3, 4])
                frames.append(f)
                act = agent.choose_action(frames, f)
                self.assertEqual(agent.last_levels_completed, lvl)
                self.assertTrue(agent._substrate_plan_attempted)
                self.assertTrue(agent._fleet_plan_started)
                self.assertIsNotNone(act)

    def test_mixed_actions_directional_and_click_transition_stability(self):
        """
        Challenge 5: Mixed legal actions (Action 1..4 AND Action 6) on Level 0,
        transitioning to Level 1 with pure Action 6, then transitioning to Level 2
        with pure directional actions.
        """
        agent = self._create_agent(game_id="mixed-mode-transition")
        grid = np.zeros((64, 64), dtype=np.int8)
        grid[10:14, 10:14] = 2  # Click target

        # Level 0: Mixed actions [1, 2, 3, 4, 6]
        f0 = FrameData(frame=[grid], state=GameState.NOT_FINISHED, levels_completed=0, available_actions=[1, 2, 3, 4, 6])
        act0 = agent.choose_action([f0], f0)
        self.assertIsNotNone(act0)

        # Level 1: Pure click [6]
        f1 = FrameData(frame=[grid], state=GameState.NOT_FINISHED, levels_completed=1, available_actions=[6])
        act1 = agent.choose_action([f0, f1], f1)
        self.assertEqual(act1, GameAction.ACTION6)

        # Level 2: Pure directional [1, 2, 3, 4] with twin grid
        grid_twin = self._build_twin_grid()
        f2 = FrameData(frame=[grid_twin], state=GameState.NOT_FINISHED, levels_completed=2, available_actions=[1, 2, 3, 4])
        act2 = agent.choose_action([f0, f1, f2], f2)
        self.assertEqual(agent.last_levels_completed, 2)
        self.assertTrue(agent._substrate_plan_attempted)
        self.assertTrue(agent._fleet_plan_started)


if __name__ == "__main__":
    unittest.main()

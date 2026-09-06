"""
Milestone M1 Challenger 2 (Iteration 2) Adversarial Verification Suite.

Empirically stress-tests dynamic planners, queue isolation, and camera transforms in agent/my_agent.py:
1. MirroredTwinPlanner:
   - Symmetric 2-twin mazes requiring symmetry breaking via deliberate wall collisions.
   - Verification of rendezvous (p1 == p2) and deliberate wall bump occurrences.
   - Extreme/degenerate mazes:
     - Unsolvable (disconnected chambers): returns None safely.
     - Already touching / 1-step rendezvous: returns minimal path.
     - Large open state space: bounded by 15,000 states without hanging or memory leaks.
2. LabyrinthPlanner:
   - Mazes with corridor connectivity: BFS connects avatar to exit sprite.
   - Invalid corridor topologies:
     - Blocked corridors: returns None.
     - Mismatched lattice alignments: returns None.
     - Missing avatar / exit: returns None safely.
3. Queue Isolation Invariant:
   - Deliberate wall bumps do NOT flush self.fleet_plan_queue.
   - Level transitions DO flush self.fleet_plan_queue and reset latch.
   - Terminal RESET clears queues and returns GameAction.RESET.
4. CameraTransform Robustness:
   - Invertibility of display_to_grid and grid_to_display across scales {1, 2, 3, 4, 5, 8} and offsets.
   - Out-of-viewport rejection: display_to_grid returns None for coords outside active viewport.
   - Clamping of ACTION6 coordinates strictly to [0, 63] under floats, negatives, NaNs, and infinities.
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
    MirroredTwinPlanner,
    LabyrinthPlanner,
    CameraTransform,
    make_action6,
    MyAgent,
    WorkingMemory,
    AvatarTracker,
    ScenePerception,
    DynamicLatticeDetector,
)


class TestM1R2PlannersAndCameraEmpiricalChallenge(unittest.TestCase):
    """Adversarial stress harness for M1 Iteration 2 planners, queue isolation, and camera transforms."""

    # =========================================================================
    # 1. MIRRORED TWIN PLANNER STRESS TESTS
    # =========================================================================

    def test_mirrored_twin_symmetry_breaking_rendezvous(self):
        """
        Adversarial Test 1.1: Symmetric 2-twin maze requiring symmetry breaking.
        
        Twins start at (c1=1, r1=5) and (c2=8, r2=5) in gw=10, gh=10.
        Horizontal sum c1 + c2 = 9 (gw - 1).
        Because the initial sum is odd, without a deliberate wall collision,
        symmetric horizontal movements preserve c1 + c2 = 9, so c1 == c2 is
        mathematically IMPOSSIBLE (c1 == c2 requires c1 + c2 to be even).
        
        A wall at (c=0, r=5) allows Twin 1 to bump left and stay at c1=1,
        while Twin 2 moves right to c2=9. The new sum becomes 10 (even),
        breaking symmetry and allowing both twins to rendezvous at (5, 5).
        """
        gw, gh = 10, 10
        grid = np.zeros((64, 64), dtype=np.int8)  # default background 0
        
        # Walls (color 1) around borders
        grid[0:gh, 0] = 1   # Left border
        grid[0:gh, gw-1] = 0 # Right open at row 5 so Twin 2 can move to c=9
        grid[0, 0:gw] = 1   # Top border
        grid[gh-1, 0:gw] = 1 # Bottom border
        
        # Place twins at symmetric locations (color 3)
        # Twin 1 at (c=1, r=5), Twin 2 at (c=8, r=5)
        grid[5, 1] = 3
        grid[5, 8] = 3

        legal_actions = {GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4}
        plan = MirroredTwinPlanner.detect_and_plan(
            grid=grid,
            step_size=1,
            roff=0,
            coff=0,
            gh=gh,
            gw=gw,
            legal_actions=legal_actions,
        )

        self.assertIsNotNone(plan, "MirroredTwinPlanner failed to find symmetry-breaking plan!")
        self.assertGreater(len(plan), 0, "Plan should contain actions")

        # Now simulate the plan with the two twins and verify:
        # 1. Deliberate wall bump occurred (one twin was blocked while the other moved)
        # 2. Both twins successfully rendezvous at p1 == p2
        passable = (grid[:gh, :gw] == 0) | (grid[:gh, :gw] == 3)
        actions_map = {
            GameAction.ACTION1: ((0, -1), (0, -1)),  # Up
            GameAction.ACTION2: ((0, 1), (0, 1)),    # Down
            GameAction.ACTION3: ((-1, 0), (1, 0)),   # Left (Twin 2 moves Right)
            GameAction.ACTION4: ((1, 0), (-1, 0)),   # Right (Twin 2 moves Left)
        }

        def step_twin(pos, move):
            x, y = pos
            dx, dy = move
            nx, ny = x + dx, y + dy
            if 0 <= nx < gw and 0 <= ny < gh and passable[ny, nx]:
                return (nx, ny), False
            return (x, y), True  # BUMP!

        p1 = (1, 5)
        p2 = (8, 5)
        deliberate_bumps = 0

        for act in plan:
            m1, m2 = actions_map[act]
            new_p1, bump1 = step_twin(p1, m1)
            new_p2, bump2 = step_twin(p2, m2)
            if bump1 != bump2:
                deliberate_bumps += 1  # Exactly one twin bumped, breaking symmetry!
            p1, p2 = new_p1, new_p2
            if p1 == p2:
                break

        self.assertEqual(p1, p2, f"Twins failed to rendezvous! Final positions: p1={p1}, p2={p2}")
        self.assertGreater(deliberate_bumps, 0, "Plan solved the puzzle without symmetry-breaking wall bump, which violates mathematical odd-parity constraint!")

    def test_mirrored_twin_extreme_unsolvable_chambers(self):
        """
        Adversarial Test 1.2: Unsolvable disconnected chambers.
        Twin 1 and Twin 2 are in completely separated hermetic rooms.
        MirroredTwinPlanner must return None without hanging or exceeding 15,000 states.
        """
        gw, gh = 10, 10
        grid = np.zeros((64, 64), dtype=np.int8)
        
        # Complete vertical divider wall down col 4 and 5
        grid[0:gh, 4] = 1
        grid[0:gh, 5] = 1
        
        # Twin 1 at (c=1, r=5), Twin 2 at (c=8, r=5)
        grid[5, 1] = 3
        grid[5, 8] = 3

        legal_actions = {GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4}
        plan = MirroredTwinPlanner.detect_and_plan(
            grid=grid,
            step_size=1,
            roff=0,
            coff=0,
            gh=gh,
            gw=gw,
            legal_actions=legal_actions,
        )

        self.assertIsNone(plan, f"Expected None for unsolvable maze, got {plan}")

    def test_mirrored_twin_1_step_rendezvous(self):
        """
        Adversarial Test 1.3: Immediate adjacent rendezvous.
        Twins start at (c1=3, r1=4) and (c2=5, r2=4) in gw=9, gh=9.
        Symmetry is preserved (3 + 5 = 8 = gw - 1).
        ACTION4 moves Twin 1 right to c=4 and Twin 2 left to c=4, meeting in 1 step!
        """
        gw, gh = 9, 9
        grid = np.zeros((64, 64), dtype=np.int8)
        grid[4, 3] = 3
        grid[4, 5] = 3

        legal_actions = {GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4}
        plan = MirroredTwinPlanner.detect_and_plan(
            grid=grid,
            step_size=1,
            roff=0,
            coff=0,
            gh=gh,
            gw=gw,
            legal_actions=legal_actions,
        )

        self.assertIsNotNone(plan)
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0], GameAction.ACTION4)

    def test_mirrored_twin_state_space_budget_bounded(self):
        """
        Adversarial Test 1.4: State-space explosion stress harness.
        Construct an unsolvable 41x20 grid where BFS explores two large open chambers
        separated by a hermetic wall at col 20.
        Planner must cap exploration at 15,000 states and terminate in < 2.0 seconds returning None.
        """
        import time
        gw, gh = 41, 20
        grid = np.zeros((64, 64), dtype=np.int8)
        
        # Vertical hermetic wall at col 20 (color 1)
        grid[0:gh, 20] = 1

        # Border walls around 41x20 viewport
        grid[0, 0:gw] = 1
        grid[gh-1, 0:gw] = 1
        grid[0:gh, 0] = 1
        grid[0:gh, gw-1] = 1

        # Twin 1 at (c=5, r=5), Twin 2 at (c=35, r=5) -> 5 + 35 = 40 = gw - 1
        grid[5, 5] = 3
        grid[5, 35] = 3

        legal_actions = {GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4}
        t0 = time.perf_counter()
        plan = MirroredTwinPlanner.detect_and_plan(
            grid=grid,
            step_size=1,
            roff=0,
            coff=0,
            gh=gh,
            gw=gw,
            legal_actions=legal_actions,
        )
        elapsed = time.perf_counter() - t0

        self.assertIsNone(plan, "Expected None for disconnected chambers")
        self.assertLess(elapsed, 2.0, f"Planner exceeded 2.0s latency budget ({elapsed:.3f}s)")

    # =========================================================================
    # 2. LABYRINTH PLANNER STRESS TESTS
    # =========================================================================

    def test_labyrinth_valid_corridor_connectivity(self):
        """
        Adversarial Test 2.1: Valid labyrinth maze connectivity.
        Construct 64x64 grid with step_size=2, node_step=4.
        Avatar at (8, 8), exit at (16, 16).
        Corridor path: (8, 8) -> (8, 12) -> (8, 16) -> (12, 16) -> (16, 16).
        Midpoint colors: 2 (corridor path).
        """
        grid = np.zeros((64, 64), dtype=np.int8)
        # Background = 0, Walls = 11
        grid[:, :] = 11  # filled with walls
        
        # Nodes:
        # (8, 8), (8, 12), (8, 16), (12, 16), (16, 16)
        nodes = [(8, 8), (8, 12), (8, 16), (12, 16), (16, 16)]
        for r, c in nodes:
            grid[r:r+2, c:c+2] = 2  # corridor node
            
        # Midpoint corridors (distance step=2):
        # (8, 10): between (8, 8) and (8, 12)
        grid[8:10, 10:12] = 2
        # (8, 14): between (8, 12) and (8, 16)
        grid[8:10, 14:16] = 2
        # (10, 16): between (8, 16) and (12, 16)
        grid[10:12, 16:18] = 2
        # (14, 16): between (12, 16) and (16, 16)
        grid[14:16, 16:18] = 2
        
        # Avatar at (8, 8) (color 9)
        grid[8:10, 8:10] = 9
        # Exit sprite at (16, 16) (color 14)
        grid[16:18, 16:18] = 14
        
        exit_sprite = {
            "bbox": (16, 17, 16, 17),
            "color": 14,
            "size": 4,
        }
        avatar_pos = (8, 8)
        legal_actions = {GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4}
        
        plan = LabyrinthPlanner.detect_and_plan(
            grid=grid,
            avatar_pos=avatar_pos,
            exit_sprite=exit_sprite,
            step_size=2,
            legal_actions=legal_actions,
        )
        
        self.assertIsNotNone(plan, "LabyrinthPlanner failed to solve valid corridor maze")
        # Expected path: Right (ACTION4) twice to (8, 16), then Down (ACTION2) twice to (16, 16)
        self.assertEqual(plan, [GameAction.ACTION4, GameAction.ACTION4, GameAction.ACTION2, GameAction.ACTION2])

    def test_labyrinth_blocked_corridor_fallback(self):
        """
        Adversarial Test 2.2: Blocked corridor topology returns None.
        Same maze as 2.1, but midpoint corridor (10, 16) is replaced by wall (color 11).
        """
        grid = np.zeros((64, 64), dtype=np.int8)
        grid[:, :] = 11
        
        nodes = [(8, 8), (8, 12), (8, 16), (12, 16), (16, 16)]
        for r, c in nodes:
            grid[r:r+2, c:c+2] = 2
            
        grid[8:10, 10:12] = 2
        grid[8:10, 14:16] = 2
        # (10, 16) is wall (11) -- BLOCKED!
        grid[14:16, 16:18] = 2
        
        grid[8:10, 8:10] = 9
        grid[16:18, 16:18] = 14
        
        exit_sprite = {
            "bbox": (16, 17, 16, 17),
            "color": 14,
            "size": 4,
        }
        avatar_pos = (8, 8)
        legal_actions = {GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4}
        
        plan = LabyrinthPlanner.detect_and_plan(
            grid=grid,
            avatar_pos=avatar_pos,
            exit_sprite=exit_sprite,
            step_size=2,
            legal_actions=legal_actions,
        )
        
        self.assertIsNone(plan, f"Expected None for blocked corridor, got {plan}")

    def test_labyrinth_mismatched_lattice_alignment(self):
        """
        Adversarial Test 2.3: Mismatched lattice alignments.
        Exit sprite is at row 17 (distance 17 - 8 = 9, not divisible by node_step 4).
        LabyrinthPlanner must reject and return None immediately.
        """
        grid = np.zeros((64, 64), dtype=np.int8)
        grid[8:10, 8:10] = 9
        grid[17:19, 16:18] = 14
        
        exit_sprite = {
            "bbox": (17, 18, 16, 17),
            "color": 14,
            "size": 4,
        }
        avatar_pos = (8, 8)
        legal_actions = {GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4}
        
        plan = LabyrinthPlanner.detect_and_plan(
            grid=grid,
            avatar_pos=avatar_pos,
            exit_sprite=exit_sprite,
            step_size=2,
            legal_actions=legal_actions,
        )
        
        self.assertIsNone(plan, f"Expected None for lattice misalignment, got {plan}")

    def test_labyrinth_missing_avatar_fallback(self):
        """
        Adversarial Test 2.4: Missing avatar and empty candidate sprites returns None gracefully.
        """
        grid = np.zeros((64, 64), dtype=np.int8)
        exit_sprite = {"bbox": (16, 17, 16, 17), "color": 14, "size": 4}
        legal_actions = {GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4}
        
        plan = LabyrinthPlanner.detect_and_plan(
            grid=grid,
            avatar_pos=None,
            exit_sprite=exit_sprite,
            step_size=2,
            legal_actions=legal_actions,
            sprites=[],
        )
        self.assertIsNone(plan)

    # =========================================================================
    # 3. QUEUE ISOLATION INVARIANT TESTS
    # =========================================================================

    def _create_agent(self, game_id: str = "test-game") -> MyAgent:
        return MyAgent(
            card_id="test",
            game_id=game_id,
            agent_name=f"Agent.{game_id}",
            ROOT_URL="http://test",
            record=False,
            arc_env=MagicMock(),
        )

    def test_queue_isolation_deliberate_wall_bumps_do_not_flush_fleet_queue(self):
        """
        Adversarial Test 3.1: Deliberate wall bumps do NOT flush self.fleet_plan_queue.
        
        AvatarTracker.update clears self.memory.plan_queue on collision (zero diff),
        but self.fleet_plan_queue MUST be isolated and survive collision feedback intact.
        """
        agent = self._create_agent()
        test_plan = [GameAction.ACTION1, GameAction.ACTION3, GameAction.ACTION4, GameAction.ACTION2]
        agent.fleet_plan_queue.extend(test_plan)
        agent._fleet_plan_started = True

        # Initial frame
        grid = np.zeros((64, 64), dtype=np.int8)
        grid[10:12, 10:12] = 3
        f1 = FrameData(frame=[grid], state=GameState.NOT_FINISHED, available_actions=[1, 2, 3, 4])

        # Step 1: dispatches ACTION1 from fleet_plan_queue
        act1 = agent.choose_action([f1], f1)
        self.assertEqual(act1, GameAction.ACTION1)
        self.assertEqual(len(agent.fleet_plan_queue), 3)

        # Inject a plan into memory.plan_queue
        agent.memory.plan_queue.append(GameAction.ACTION1)
        self.assertEqual(len(agent.memory.plan_queue), 1)

        # Step 2: Feed identical frame (diff == False, indicating collision/bump)
        f2 = FrameData(frame=[grid], state=GameState.NOT_FINISHED, available_actions=[1, 2, 3, 4])
        act2 = agent.choose_action([f1, f2], f2)

        # memory.plan_queue should be wiped by collision handler
        self.assertEqual(len(agent.memory.plan_queue), 0, "memory.plan_queue should have been cleared by collision")
        # fleet_plan_queue MUST NOT be wiped, and should have dispatched the next action (ACTION3)
        self.assertEqual(act2, GameAction.ACTION3)
        self.assertEqual(len(agent.fleet_plan_queue), 2)
        self.assertTrue(agent._fleet_plan_started)

    def test_queue_isolation_level_transition_flushes_fleet_queue(self):
        """
        Adversarial Test 3.2: Level transition flushes fleet_plan_queue and resets latch.
        
        When latest_frame.levels_completed > self.last_levels_completed,
        all queues (memory.plan_queue and self.fleet_plan_queue) must be purged,
        and self._fleet_plan_started must be reset to False.
        """
        agent = self._create_agent()
        agent.fleet_plan_queue.extend([GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3])
        agent._fleet_plan_started = True
        agent.last_levels_completed = 0

        grid = np.zeros((64, 64), dtype=np.int8)
        grid[10:12, 10:12] = 3
        # Frame with level completed = 1
        level_up_frame = FrameData(
            frame=[grid],
            state=GameState.NOT_FINISHED,
            levels_completed=1,
            available_actions=[1, 2, 3, 4],
        )

        agent.choose_action([level_up_frame], level_up_frame)

        self.assertEqual(agent.last_levels_completed, 1, "last_levels_completed not updated")
        self.assertEqual(len(agent.fleet_plan_queue), 0, "fleet_plan_queue was not flushed on level transition")
        self.assertFalse(agent._fleet_plan_started, "_fleet_plan_started latch was not reset to False on level transition")

    def test_queue_isolation_terminal_reset_safety(self):
        """
        Adversarial Test 3.3: Terminal state (GAME_OVER / NOT_PLAYED) clears queues and resets latch.
        """
        agent = self._create_agent()
        agent.fleet_plan_queue.extend([GameAction.ACTION1, GameAction.ACTION2])
        agent._fleet_plan_started = True

        grid = np.zeros((64, 64), dtype=np.int8)
        game_over_frame = FrameData(
            frame=[grid],
            state=GameState.GAME_OVER,
            available_actions=[0],
        )

        action = agent.choose_action([game_over_frame], game_over_frame)

        self.assertEqual(action, GameAction.RESET)
        self.assertEqual(len(agent.fleet_plan_queue), 0)
        self.assertFalse(agent._fleet_plan_started)

    # =========================================================================
    # 4. CAMERA TRANSFORM ROBUSTNESS TESTS
    # =========================================================================

    def test_camera_transform_invertibility_multiscale(self):
        """
        Adversarial Test 4.1: Invertibility of display_to_grid and grid_to_display.
        
        For all S in {1, 2, 3, 4, 5, 8} and valid offset combinations:
        display_to_grid(grid_to_display(gx, gy)) == (gx, gy)
        must hold for every coordinate in the internal grid viewport.
        """
        configurations = [
            {"scale": 1, "ox": 0, "oy": 0, "w": 64, "h": 64},
            {"scale": 2, "ox": 4, "oy": 4, "w": 28, "h": 28},
            {"scale": 3, "ox": 2, "oy": 2, "w": 20, "h": 20},
            {"scale": 4, "ox": 8, "oy": 8, "w": 12, "h": 12},
            {"scale": 5, "ox": 2, "oy": 2, "w": 12, "h": 12},
            {"scale": 8, "ox": 0, "oy": 0, "w": 8, "h": 8},
        ]

        for cfg in configurations:
            cam = CameraTransform(
                scale=cfg["scale"],
                offset_x=cfg["ox"],
                offset_y=cfg["oy"],
                width=cfg["w"],
                height=cfg["h"],
            )

            # Test all internal grid coordinates
            for gx in range(cfg["w"]):
                for gy in range(cfg["h"]):
                    disp_x, disp_y = cam.grid_to_display(gx, gy)
                    # Verify clamped to [0, 63]
                    self.assertTrue(0 <= disp_x <= 63, f"disp_x {disp_x} out of [0, 63] for cfg {cfg}")
                    self.assertTrue(0 <= disp_y <= 63, f"disp_y {disp_y} out of [0, 63] for cfg {cfg}")
                    
                    # Verify inside viewport
                    self.assertTrue(cam.is_inside_viewport(disp_x, disp_y), f"Expected ({disp_x}, {disp_y}) inside viewport")

                    # Verify inverse reconstruction
                    recovered = cam.display_to_grid(disp_x, disp_y)
                    self.assertIsNotNone(recovered, f"display_to_grid returned None for valid point ({gx}, {gy}) in cfg {cfg}")
                    self.assertEqual(recovered, (gx, gy), f"Invertibility failure: expected ({gx}, {gy}), got {recovered} in cfg {cfg}")

    def test_camera_transform_out_of_viewport_rejection(self):
        """
        Adversarial Test 4.2: Out-of-viewport display coordinates return None.
        """
        cam = CameraTransform(scale=4, offset_x=10, offset_y=10, width=10, height=10)
        # letterbox left of viewport
        self.assertIsNone(cam.display_to_grid(5, 15))
        self.assertFalse(cam.is_inside_viewport(5, 15))
        # letterbox above viewport
        self.assertIsNone(cam.display_to_grid(15, 5))
        self.assertFalse(cam.is_inside_viewport(15, 5))
        # letterbox right of viewport (offset_x + width*scale = 10 + 40 = 50)
        self.assertIsNone(cam.display_to_grid(55, 15))
        self.assertFalse(cam.is_inside_viewport(55, 15))
        # letterbox below viewport
        self.assertIsNone(cam.display_to_grid(15, 55))
        self.assertFalse(cam.is_inside_viewport(15, 55))

    def test_make_action6_clamping_and_types(self):
        """
        Adversarial Test 4.3: Clamping of ACTION6 coordinates strictly to [0, 63].
        
        Stress test make_action6 under:
        - Floating-point numbers: (14.2, 38.9)
        - Negative coordinates: (-50, -0.001)
        - Extreme high numbers: (100, 99999.9)
        - NaNs: float('nan')
        - Infinities: float('inf'), float('-inf')
        - With and without CameraTransform
        """
        test_cases = [
            (14.2, 38.9, 14, 39),
            (-50, -0.001, 0, 0),
            (100, 99999.9, 63, 63),
            (float("nan"), float("nan"), 32, 32),
            (float("inf"), float("-inf"), 32, 32),
            (0, 0, 0, 0),
            (63, 63, 63, 63),
            (64, 64, 63, 63),
        ]

        for in_x, in_y, exp_x, exp_y in test_cases:
            act = make_action6(in_x, in_y, why="stress_test")
            self.assertEqual(act, GameAction.ACTION6)
            
            data = act.action_data.model_dump() if hasattr(act, "action_data") and act.action_data else act.data
            out_x = data["x"]
            out_y = data["y"]
            
            self.assertIsInstance(out_x, int, f"x must be int, got {type(out_x)}")
            self.assertIsInstance(out_y, int, f"y must be int, got {type(out_y)}")
            self.assertTrue(0 <= out_x <= 63, f"x {out_x} not in [0, 63]")
            self.assertTrue(0 <= out_y <= 63, f"y {out_y} not in [0, 63]")
            self.assertEqual((out_x, out_y), (exp_x, exp_y))

        # Test with CameraTransform with scale=4, offset=10
        cam = CameraTransform(scale=4, offset_x=10, offset_y=10, width=10, height=10)
        extreme_cam_inputs = [
            (-999, -999),
            (999, 999),
            (float("nan"), 5.0),
            (2.0, float("inf")),
        ]
        for in_x, in_y in extreme_cam_inputs:
            act = make_action6(in_x, in_y, camera_transform=cam, why="cam_stress")
            self.assertEqual(act, GameAction.ACTION6)
            data = act.action_data.model_dump() if hasattr(act, "action_data") and act.action_data else act.data
            out_x = data["x"]
            out_y = data["y"]
            self.assertIsInstance(out_x, int)
            self.assertIsInstance(out_y, int)
            self.assertTrue(0 <= out_x <= 63, f"cam out_x {out_x} not in [0, 63]")
    # =========================================================================
    # 5. FURTHER ADVERSARIAL EDGE CASES & STRESS HARNESSES
    # =========================================================================

    def test_mirrored_twin_degenerate_inputs(self):
        """
        Adversarial Test 5.1: Degenerate inputs to MirroredTwinPlanner.
        - Sub-minimum grid dimensions (gh < 3 or gw < 3)
        - Asymmetric twin candidates (r1 != r2 or bad midline sum)
        - Singleton / triplet counts (len(coords) != 2)
        - Empty or non-directional legal actions
        """
        grid = np.zeros((64, 64), dtype=np.int8)
        grid[5, 5] = 3
        grid[5, 15] = 3
        legal_actions = {GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4}

        # Case 1: gh < 3 or gw < 3
        self.assertIsNone(MirroredTwinPlanner.detect_and_plan(grid, 1, 0, 0, 2, 2, legal_actions))
        self.assertIsNone(MirroredTwinPlanner.detect_and_plan(grid, 1, 0, 0, 2, 10, legal_actions))
        self.assertIsNone(MirroredTwinPlanner.detect_and_plan(grid, 1, 0, 0, 10, 2, legal_actions))

        # Case 2: Asymmetric twins (different rows)
        grid_asym = np.zeros((64, 64), dtype=np.int8)
        grid_asym[4, 2] = 3
        grid_asym[6, 7] = 3  # gw=10, row 4 != row 6
        self.assertIsNone(MirroredTwinPlanner.detect_and_plan(grid_asym, 1, 0, 0, 10, 10, legal_actions))

        # Case 3: 3 sprites of color 3 (triplets)
        grid_trip = np.zeros((64, 64), dtype=np.int8)
        grid_trip[5, 2] = 3
        grid_trip[5, 7] = 3
        grid_trip[5, 5] = 3
        self.assertIsNone(MirroredTwinPlanner.detect_and_plan(grid_trip, 1, 0, 0, 10, 10, legal_actions))

        # Case 4: Non-directional actions only (e.g. only ACTION5, ACTION6)
        grid_valid = np.zeros((64, 64), dtype=np.int8)
        grid_valid[5, 2] = 3
        grid_valid[5, 7] = 3
        self.assertIsNone(MirroredTwinPlanner.detect_and_plan(grid_valid, 1, 0, 0, 10, 10, {GameAction.ACTION5}))

    def test_labyrinth_already_at_exit_and_step_zero(self):
        """
        Adversarial Test 5.2: LabyrinthPlanner with avatar already at exit and step_size=0.
        - Avatar already within step of exit returns empty path [].
        - step_size=0 clamped to step=1 without DivisionByZero.
        """
        grid = np.zeros((64, 64), dtype=np.int8)
        grid[10, 10] = 9
        exit_sprite = {"bbox": (10, 11, 10, 11), "color": 14, "size": 4}
        legal_actions = {GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4}

        # Avatar already at exit
        plan = LabyrinthPlanner.detect_and_plan(
            grid=grid,
            avatar_pos=(10, 10),
            exit_sprite=exit_sprite,
            step_size=2,
            legal_actions=legal_actions,
        )
        self.assertEqual(plan, [])

        # step_size = 0 clamped to max(1, 0)
        plan_step0 = LabyrinthPlanner.detect_and_plan(
            grid=grid,
            avatar_pos=(10, 10),
            exit_sprite=exit_sprite,
            step_size=0,
            legal_actions=legal_actions,
        )
        self.assertEqual(plan_step0, [])

    def test_camera_transform_infer_from_grid_visual_fidelity(self):
        """
        Adversarial Test 5.3: CameraTransform.infer_from_grid visual fidelity.
        - Uniform grid fallback to default (scale=1, 64x64, no offset).
        - Synthetic letterboxed 4x scaled grid: verifies scale=4, offsets, width/height.
        """
        # Case 1: Uniform grid
        uniform_grid = np.full((64, 64), 5, dtype=np.int8)
        cam1 = CameraTransform.infer_from_grid(uniform_grid)
        self.assertEqual(cam1.scale, 1)
        self.assertEqual(cam1.offset_x, 0)
        self.assertEqual(cam1.offset_y, 0)
        self.assertEqual(cam1.width, 64)
        self.assertEqual(cam1.height, 64)

        # Case 2: Letterbox offset 8 on top/bottom/left/right, active area 48x48 with pitch S=4
        grid_letterbox = np.full((64, 64), 0, dtype=np.int8)  # top_color = 0
        # Active area is rows 8..55, cols 8..55 (size 48x48)
        # S=4: internal width=12, height=12
        for r in range(8, 56, 4):
            for c in range(8, 56, 4):
                grid_letterbox[r:r+4, c:c+4] = ((r // 4) + (c // 4)) % 3 + 1  # non-zero colors

        cam2 = CameraTransform.infer_from_grid(grid_letterbox)
        self.assertEqual(cam2.scale, 4)
        self.assertEqual(cam2.offset_x, 8)
        self.assertEqual(cam2.offset_y, 8)
        self.assertEqual(cam2.width, 12)
        self.assertEqual(cam2.height, 12)
        self.assertEqual(cam2.letter_box_color, 0)


if __name__ == "__main__":
    unittest.main()


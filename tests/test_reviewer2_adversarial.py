"""
Reviewer 2 Adversarial Stress and Edge-Case Test Suite.
"""
from __future__ import annotations

import os
import sys
import time
from unittest.mock import MagicMock
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VENDOR = os.path.join(ROOT, "vendor", "ARC-AGI-3-Agents")
sys.path.insert(0, ROOT)
sys.path.insert(0, VENDOR)

from arcengine import FrameData, GameAction, GameState
from agent.my_agent import (
    MyAgent,
    BlockConfiguration,
    TemplatePerception,
    OperatorPadDetector,
    ConfigurationPlanner,
    ScenePerception,
    WorkingMemory,
)


def create_agent(game_id: str = "test") -> MyAgent:
    return MyAgent(
        card_id="rev2-test",
        game_id=game_id,
        agent_name=f"Rev2Agent.{game_id}",
        ROOT_URL="http://localhost",
        record=False,
        arc_env=MagicMock(),
    )


def test_action7_compatibility():
    print("\n--- [R2-Test 1] ACTION7 Compatibility & Availability ---")
    agent = create_agent()
    grid = np.zeros((64, 64), dtype=np.int8)
    
    frame_only_7 = FrameData(
        frame=[grid],
        state=GameState.NOT_FINISHED,
        available_actions=[7],
    )
    act = agent.choose_action([frame_only_7], frame_only_7)
    assert act == GameAction.ACTION7, f"Expected GameAction.ACTION7 when only 7 is available, got {act}"
    print("  Only ACTION7 available: emitted ACTION7 successfully.")

    frame_with_7 = FrameData(
        frame=[grid],
        state=GameState.NOT_FINISHED,
        available_actions=[1, 2, 3, 4, 7],
    )
    act2 = agent.choose_action([frame_with_7], frame_with_7)
    assert act2 in (
        GameAction.ACTION1,
        GameAction.ACTION2,
        GameAction.ACTION3,
        GameAction.ACTION4,
        GameAction.ACTION7,
    )
    print("  Directional + ACTION7: valid action emitted without exception.")


def test_unreachable_puzzle_bfs_safety():
    print("\n--- [R2-Test 2] Unreachable Puzzle BFS Safety & Latency ---")
    passable = np.ones((12, 12), dtype=bool)
    passable[4, :] = False

    start_tile = (0, 0)
    start_cfg = BlockConfiguration(shape_id=0, color_idx=0, rotation_idx=0)
    goal_slots = {(5, 5): BlockConfiguration(shape_id=0, color_idx=0, rotation_idx=0)}
    pads = {(1, 1): "rot"}
    legal_actions = {
        GameAction.ACTION1,
        GameAction.ACTION2,
        GameAction.ACTION3,
        GameAction.ACTION4,
    }

    t0 = time.perf_counter()
    path = ConfigurationPlanner.plan_unified_bfs(
        start_tile=start_tile,
        start_cfg=start_cfg,
        goal_slots=goal_slots,
        pads=pads,
        passable=passable,
        legal_actions=legal_actions,
    )
    t1 = time.perf_counter()
    elapsed_ms = (t1 - t0) * 1000.0

    assert path == [], f"Expected empty path for unreachable goal, got {path}"
    print(f"  Unreachable goal returned empty path safely in {elapsed_ms:.3f} ms.")
    assert elapsed_ms < 50.0, f"BFS took too long on unreachable goal: {elapsed_ms} ms"


def test_pad_retriggering_multi_step():
    print("\n--- [R2-Test 3] Pad Re-triggering Multi-Step Transformations ---")
    passable = np.ones((12, 12), dtype=bool)
    start_tile = (0, 0)
    start_cfg = BlockConfiguration(shape_id=0, color_idx=0, rotation_idx=0)
    goal_slots = {(3, 0): BlockConfiguration(shape_id=0, color_idx=0, rotation_idx=2)}
    pads = {(1, 0): "rot"}
    legal_actions = {
        GameAction.ACTION1,
        GameAction.ACTION2,
        GameAction.ACTION3,
        GameAction.ACTION4,
    }

    path = ConfigurationPlanner.plan_unified_bfs(
        start_tile=start_tile,
        start_cfg=start_cfg,
        goal_slots=goal_slots,
        pads=pads,
        passable=passable,
        legal_actions=legal_actions,
    )
    assert len(path) > 0, "Failed to find path requiring pad re-triggering!"
    print(f"  Found re-triggering path of length {len(path)}: {[a.name for a in path]}")

    cur_x, cur_y = start_tile
    cur_R = 0
    for act in path:
        if act == GameAction.ACTION1:
            cur_y -= 1
        elif act == GameAction.ACTION2:
            cur_y += 1
        elif act == GameAction.ACTION3:
            cur_x -= 1
        elif act == GameAction.ACTION4:
            cur_x += 1
        if (cur_x, cur_y) == (1, 0):
            cur_R = (cur_R + 1) % 4

    assert (cur_x, cur_y) == (3, 0), f"Failed to reach goal (3, 0), ended at ({cur_x}, {cur_y})"
    assert cur_R == 2, f"Failed to achieve target rotation 2, got {cur_R}"
    print("  Re-triggering simulation verified: arrived at goal with exact target rotation!")


def test_receptacle_blocking_invariant():
    print("\n--- [R2-Test 4] Receptacle Blocking Invariant Verification ---")
    passable = np.ones((12, 12), dtype=bool)
    start_tile = (0, 0)
    start_cfg = BlockConfiguration(shape_id=0, color_idx=0, rotation_idx=0)
    goal_slots = {(2, 0): BlockConfiguration(shape_id=0, color_idx=0, rotation_idx=1)}
    pads = {(1, 1): "rot"}
    legal_actions = {
        GameAction.ACTION1,
        GameAction.ACTION2,
        GameAction.ACTION3,
        GameAction.ACTION4,
    }

    path = ConfigurationPlanner.plan_unified_bfs(
        start_tile=start_tile,
        start_cfg=start_cfg,
        goal_slots=goal_slots,
        pads=pads,
        passable=passable,
        legal_actions=legal_actions,
    )
    assert len(path) > 0, "No path found!"

    cur_x, cur_y = start_tile
    cur_R = 0
    for idx, act in enumerate(path):
        if act == GameAction.ACTION1:
            cur_y -= 1
        elif act == GameAction.ACTION2:
            cur_y += 1
        elif act == GameAction.ACTION3:
            cur_x -= 1
        elif act == GameAction.ACTION4:
            cur_x += 1
        if (cur_x, cur_y) in pads:
            cur_R = (cur_R + 1) % 4
        if (cur_x, cur_y) == (2, 0):
            assert idx == len(path) - 1, f"Entered goal slot prematurely at step {idx} of {len(path)}!"
            assert cur_R == 1, f"Entered goal slot with wrong rotation {cur_R}!"

    print("  Receptacle blocking invariant holds: goal receptacle was never entered with wrong configuration.")


def test_malformed_and_extreme_frame_inputs():
    print("\n--- [R2-Test 5] Malformed & Boundary Frame Inputs ---")
    agent = create_agent()

    none_frame = FrameData(frame=[], state=GameState.NOT_FINISHED, available_actions=[1, 2])
    act_none = agent.choose_action([none_frame], none_frame)
    assert act_none in (GameAction.ACTION1, GameAction.ACTION2)
    print("  Empty frame list handled gracefully.")

    small_grid = np.zeros((30, 30), dtype=np.int8)
    small_frame = FrameData(frame=[small_grid], state=GameState.NOT_FINISHED, available_actions=[1, 2])
    act_small = agent.choose_action([small_frame], small_frame)
    assert act_small in (GameAction.ACTION1, GameAction.ACTION2)
    print("  Non-standard grid dimensions handled gracefully.")

    reset_only_frame = FrameData(
        frame=[np.zeros((64, 64), dtype=np.int8)],
        state=GameState.NOT_FINISHED,
        available_actions=[0],
    )
    act_reset = agent.choose_action([reset_only_frame], reset_only_frame)
    assert act_reset == GameAction.RESET, f"Expected RESET on [0] available_actions, got {act_reset}"
    print("  available_actions=[0] returns RESET safely.")

    game_over_frame = FrameData(
        frame=[np.zeros((64, 64), dtype=np.int8)],
        state=GameState.GAME_OVER,
        available_actions=[0, 1, 2],
    )
    act_go = agent.choose_action([game_over_frame], game_over_frame)
    assert act_go == GameAction.RESET
    print("  GAME_OVER emits GameAction.RESET safely.")


def test_worst_case_state_space_bfs_performance():
    print("\n--- [R2-Test 6] Full 12x12 Joint State-Space BFS Max Complexity Benchmark ---")
    passable = np.ones((12, 12), dtype=bool)
    start_tile = (0, 0)
    start_cfg = BlockConfiguration(shape_id=0, color_idx=0, rotation_idx=0)
    goal_slots = {(11, 11): BlockConfiguration(shape_id=5, color_idx=3, rotation_idx=3)}
    pads = {
        (1, 1): "rot",
        (2, 2): "color",
        (3, 3): "shape",
    }
    legal_actions = {
        GameAction.ACTION1,
        GameAction.ACTION2,
        GameAction.ACTION3,
        GameAction.ACTION4,
    }

    t0 = time.perf_counter()
    path = ConfigurationPlanner.plan_unified_bfs(
        start_tile=start_tile,
        start_cfg=start_cfg,
        goal_slots=goal_slots,
        pads=pads,
        passable=passable,
        legal_actions=legal_actions,
    )
    t1 = time.perf_counter()
    elapsed_ms = (t1 - t0) * 1000.0

    print(f"  Worst-case BFS solved: path length {len(path)} actions, computed in {elapsed_ms:.2f} ms.")
    assert len(path) > 0, "Failed to find path for complex multi-operator goal!"
    assert elapsed_ms < 50.0, f"BFS latency exceeded 50ms budget: {elapsed_ms} ms"


if __name__ == "__main__":
    test_action7_compatibility()
    test_unreachable_puzzle_bfs_safety()
    test_pad_retriggering_multi_step()
    test_receptacle_blocking_invariant()
    test_malformed_and_extreme_frame_inputs()
    test_worst_case_state_space_bfs_performance()
    print("\nALL REVIEWER 2 ADVERSARIAL STRESS TESTS PASSED SUCCESSFULLY!")

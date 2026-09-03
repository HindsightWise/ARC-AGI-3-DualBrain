"""
Adversarial Edge-Case and Stress-Testing Suite for Challenger 2.
Focuses on hostile inputs, boundary clipping, 4-cycle loops, worst-case BFS latency,
and action space integrity.
"""
from __future__ import annotations

import collections
import os
import sys
import time
from unittest.mock import MagicMock
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VENDOR = os.path.join(ROOT, "vendor", "ARC-AGI-3-Agents")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if VENDOR not in sys.path:
    sys.path.insert(0, VENDOR)

from arcengine import FrameData, GameAction, GameState
from agent.my_agent import MyAgent, ScenePerception, AvatarTracker, GeodesicPlanner, WorkingMemory


def create_agent():
    return MyAgent(
        card_id="adv-test",
        game_id="adv_game",
        agent_name="AdvAgent",
        ROOT_URL="http://localhost",
        record=False,
        arc_env=MagicMock(),
    )


def test_adversarial_action6_singleton_mutation_and_clamping():
    print("\n--- [Adversarial 1] ACTION6 Singleton Mutation & Strict Clamping ---")
    agent = create_agent()
    agent.memory.probing_phase = False

    # Create grid with 50 distinct 1-pixel sprites scattered throughout
    grid = np.zeros((64, 64), dtype=np.int8)
    coords = []
    for i in range(50):
        r = (i * 7) % 55  # keep outside HUD (< 56)
        c = (i * 13) % 64
        grid[r, c] = (i % 14) + 1
        coords.append((c, r))

    frame = FrameData(
        frame=[grid],
        state=GameState.NOT_FINISHED,
        available_actions=[GameAction.ACTION6.value],
    )

    seen_coords = set()
    for step in range(60):
        action = agent.choose_action([frame], frame)
        assert action == GameAction.ACTION6
        assert hasattr(action, "action_data")
        x, y = action.action_data.x, action.action_data.y
        assert 0 <= x <= 63, f"x={x} out of range [0, 63] at step {step}"
        assert 0 <= y <= 63, f"y={y} out of range [0, 63] at step {step}"
        seen_coords.add((x, y))

    print(f"Dispatched {len(seen_coords)} unique target coordinates across 60 steps.")
    assert len(seen_coords) > 20, "Agent failed to cycle through candidate centroids!"
    print("PASS: test_adversarial_action6_singleton_mutation_and_clamping")


def test_adversarial_action5_corner_clipping():
    print("\n--- [Adversarial 2] ACTION5 Corner Clipping & Self-Collision ---")
    agent = create_agent()
    agent.memory.probing_phase = False

    # Test top-left corner (0, 0)
    agent.memory.avatar_pos = (0, 0)
    agent.memory.step_size = 1
    agent.memory.avatar_colors = {1}
    grid = np.zeros((64, 64), dtype=np.int8)
    grid[0, 0] = 1

    # Empty surroundings at corner -> Should NOT trigger ACTION5
    assert not agent._should_trigger_action5(grid, bg_color=0), "Triggered ACTION5 at corner (0,0) without neighbor!"

    # Entity at (0, 1) -> SHOULD trigger
    grid[0, 1] = 2
    assert agent._should_trigger_action5(grid, bg_color=0), "Failed to trigger ACTION5 at corner (0,0) with neighbor at (0,1)!"

    # Test bottom-right corner (63, 63)
    agent.memory.avatar_pos = (63, 63)
    grid2 = np.zeros((64, 64), dtype=np.int8)
    grid2[63, 63] = 1
    assert not agent._should_trigger_action5(grid2, bg_color=0), "Triggered ACTION5 at corner (63,63) without neighbor!"

    grid2[62, 63] = 4
    assert agent._should_trigger_action5(grid2, bg_color=0), "Failed to trigger ACTION5 at corner (63,63) with neighbor at (62,63)!"

    print("PASS: test_adversarial_action5_corner_clipping")


def test_adversarial_loop_interception_4cycle_and_constrained_legal():
    print("\n--- [Adversarial 3] 4-Cycle Loop & Constrained Legal Actions ---")
    agent = create_agent()
    agent.memory.probing_phase = False

    # Simulate 4-cycle loop: ACTION1 -> ACTION4 -> ACTION2 -> ACTION3 -> ACTION1 -> ACTION4 -> ACTION2 -> ACTION3
    cycle = [GameAction.ACTION1, GameAction.ACTION4, GameAction.ACTION2, GameAction.ACTION3]
    agent.memory.recent_actions.clear()
    for a in cycle + cycle:
        agent.memory.recent_actions.append(a)

    assert agent.memory.is_in_loop(), "Failed to detect 4-cycle loop!"
    print("  4-cycle loop detected: OK")

    # Constrained action space: only vertical moves available [ACTION1, ACTION2]
    # Agent was ping-ponging vertically: last action was ACTION1.
    # Normal orthogonal escape wants ACTION4 or ACTION3, but they are NOT legal!
    agent.last_action = GameAction.ACTION1
    legal_constrained = [GameAction.ACTION1, GameAction.ACTION2]
    escape = agent._get_orthogonal_escape(legal_constrained)
    assert escape == GameAction.ACTION2, f"Expected alternate legal move ACTION2, got {escape}"
    print("  Constrained orthogonal fallback: OK")
    print("PASS: test_adversarial_loop_interception_4cycle_and_constrained_legal")


def test_worst_case_bfs_segmentation_latency():
    print("\n--- [Adversarial 4] Worst-Case Scene Perception & BFS Latency ---")
    # Worst case for BFS connected-component labeling:
    # 64x64 grid with 100 scattered non-background blocks of varying colors
    rng = np.random.RandomState(1337)
    grid = np.zeros((64, 64), dtype=np.int8)
    for _ in range(80):
        r = rng.randint(0, 54)
        c = rng.randint(0, 62)
        color = rng.randint(1, 15)
        grid[r:r+2, c:c+2] = color

    bg_color, canvas_colors = ScenePerception.detect_background_and_canvas(grid)

    N_RUNS = 200
    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        sprites = ScenePerception.segment_sprites(grid, bg_color, canvas_colors)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    mean_bfs_ms = np.mean(times)
    p95_bfs_ms = np.percentile(times, 95)
    max_bfs_ms = np.max(times)

    print(f"BFS segmentation on 80-cluster grid over {N_RUNS} runs:")
    print(f"  Sprites detected: {len(sprites)}")
    print(f"  Mean BFS time:    {mean_bfs_ms:.4f} ms")
    print(f"  P95 BFS time:     {p95_bfs_ms:.4f} ms")
    print(f"  Max BFS time:     {max_bfs_ms:.4f} ms")

    # BFS segmentation should run comfortably under 3.0 ms even in worst-case scenario
    assert mean_bfs_ms < 3.0, f"Worst-case BFS mean time {mean_bfs_ms:.4f} ms exceeds 3.0 ms budget!"
    print("PASS: test_worst_case_bfs_segmentation_latency")


def test_bump_collision_passable_map_update():
    print("\n--- [Adversarial 5] Bump Collision Learning & Map Update ---")
    mem = WorkingMemory()
    mem.avatar_pos = (10, 10)
    mem.step_size = 1
    mem.plan_queue.append(GameAction.ACTION1)

    grid1 = np.zeros((64, 64), dtype=np.int8)
    grid2 = grid1.copy()  # zero displacement -> bump!

    new_pos, moved = AvatarTracker.update(grid1, grid2, GameAction.ACTION1, mem)
    assert not moved, "AvatarTracker should report moved=False on zero-displacement diff"
    assert new_pos == (10, 10), f"Avatar position should remain (10, 10), got {new_pos}"

    # Wall at (9, 10) should be marked in passable_map
    assert not mem.passable_map[9, 10], "Target wall cell (9, 10) was not marked impassable in passable_map!"
    assert (9, 10) in mem.known_walls, "Target wall cell (9, 10) was not added to known_walls!"
    assert len(mem.plan_queue) == 0, "Plan queue was not cleared on bump!"
    assert mem.consecutive_bumps == 1, f"consecutive_bumps should be 1, got {mem.consecutive_bumps}"
    print("PASS: test_bump_collision_passable_map_update")


if __name__ == "__main__":
    print("======================================================================")
    print("STARTING DEEP ADVERSARIAL STRESS TEST SUITE")
    print("======================================================================")

    test_adversarial_action6_singleton_mutation_and_clamping()
    test_adversarial_action5_corner_clipping()
    test_adversarial_loop_interception_4cycle_and_constrained_legal()
    test_worst_case_bfs_segmentation_latency()
    test_bump_collision_passable_map_update()

    print("\n======================================================================")
    print("ALL DEEP ADVERSARIAL TESTS PASSED!")
    print("======================================================================")

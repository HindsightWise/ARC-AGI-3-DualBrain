"""
Empirical Stress-Testing and Verification Harness for ARC-AGI-3 Dual-Brain Version 2 Agent.
Challenger 2: Action Space & Stress-Testing Verifier.

Tests:
1. ACTION6 Coordinate Generation & Sprite Segmentation
2. ACTION5 Contextual Interaction Affordance
3. Terminal Safety (GAME_OVER, NOT_PLAYED, WIN, Memory Reset)
4. Loop Interception (Identical State Hashing, Ping-Pong Cycles, Orthogonal Escapes)
5. Micro-Benchmark Latency (1000 consecutive choose_action calls, < 3ms ceiling, > 300 FPS)
6. Extreme Adversarial Frames (Checkerboard, Noise, Empty, Boundary Sprites)
"""
from __future__ import annotations

import collections
import os
import sys
import time
from unittest.mock import MagicMock
import numpy as np

# Ensure path includes root and vendor framework
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VENDOR = os.path.join(ROOT, "vendor", "ARC-AGI-3-Agents")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if VENDOR not in sys.path:
    sys.path.insert(0, VENDOR)

from arcengine import FrameData, GameAction, GameState
from agent.my_agent import MyAgent, ScenePerception, AvatarTracker, GeodesicPlanner, WorkingMemory


def create_mock_agent(game_id="test_game") -> MyAgent:
    """Helper to instantiate MyAgent with a mock arc_env."""
    mock_env = MagicMock()
    agent = MyAgent(
        card_id="challenger-test",
        game_id=game_id,
        agent_name=f"Challenger2Agent.{game_id}",
        ROOT_URL="http://localhost",
        record=False,
        arc_env=mock_env,
    )
    return agent


def make_frame(grid: np.ndarray, state=GameState.NOT_FINISHED, available_actions=None) -> FrameData:
    """Helper to wrap a 64x64 numpy array into a FrameData object."""
    if available_actions is None:
        available_actions = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4, GameAction.ACTION5, GameAction.ACTION6]
    return FrameData(
        frame=[grid.astype(np.int8)],
        state=state,
        available_actions=[(a.value if hasattr(a, "value") else int(a)) for a in available_actions],
    )


# ==============================================================================
# 1. ACTION6 Coordinate Generation & Sprite Segmentation Tests
# ==============================================================================
def test_action6_synthetic_clusters():
    print("\n--- [Test 1] ACTION6 Synthetic Clusters & Boundary Checks ---")
    grid = np.zeros((64, 64), dtype=np.int8)  # Background is 0

    # Cluster 1: 3x3 block at (row 10..12, col 20..22), color 2
    grid[10:13, 20:23] = 2  # Centroid: cx = 21, cy = 11

    # Cluster 2: 1x1 pixel at (row 40, col 50), color 4
    grid[40, 50] = 4  # Centroid: cx = 50, cy = 40

    # Cluster 3: Extreme corner sprite at (row 0..1, col 62..63), color 5
    grid[0:2, 62:64] = 5  # Centroid: cx = 62 or 63, cy = 0 or 1

    # Cluster 4: Sprite in HUD area (rows 56..63) -> SHOULD BE IGNORED
    grid[58:60, 10:12] = 6

    bg_color, canvas_colors = ScenePerception.detect_background_and_canvas(grid)
    assert bg_color == 0, f"Expected bg_color=0, got {bg_color}"

    sprites = ScenePerception.segment_sprites(grid, bg_color=0, ignored_colors=canvas_colors)
    print(f"Detected {len(sprites)} sprite(s) outside HUD.")

    # Check HUD exclusion: no sprite centroid should have cy >= 56
    for s in sprites:
        cx, cy = s["centroid"]
        assert 0 <= cx <= 63, f"Centroid cx out of bounds: {cx}"
        assert 0 <= cy < 56, f"Centroid cy in HUD: {cy}"

    # Verify Cluster 1 centroid
    c1 = next((s for s in sprites if s["color"] == 2), None)
    assert c1 is not None, "Cluster 1 (color 2) was not detected"
    assert c1["centroid"] == (21, 11), f"Cluster 1 centroid mismatch: {c1['centroid']} != (21, 11)"
    assert c1["size"] == 9, f"Cluster 1 size mismatch: {c1['size']} != 9"

    # Verify Cluster 2 centroid
    c2 = next((s for s in sprites if s["color"] == 4), None)
    assert c2 is not None, "Cluster 2 (color 4) was not detected"
    assert c2["centroid"] == (50, 40), f"Cluster 2 centroid mismatch: {c2['centroid']} != (50, 40)"
    assert c2["size"] == 1, f"Cluster 2 size mismatch: {c2['size']} != 1"

    # Verify Cluster 3 (corner)
    c3 = next((s for s in sprites if s["color"] == 5), None)
    assert c3 is not None, "Cluster 3 (color 5) was not detected"
    assert c3["centroid"][0] in (62, 63) and c3["centroid"][1] in (0, 1)

    # Verify HUD cluster (color 6) was NOT segmented
    c4 = next((s for s in sprites if s["color"] == 6), None)
    assert c4 is None, "HUD sprite (color 6) was wrongly detected!"

    # Now verify MyAgent._handle_action6 emissions
    agent = create_mock_agent()
    # Mock only ACTION6 available
    act6_frame = make_frame(grid, available_actions=[GameAction.ACTION6])
    agent.memory.probing_phase = False

    emitted_coords = []
    for step in range(len(sprites) + 2):
        action = agent.choose_action([act6_frame], act6_frame)
        assert action == GameAction.ACTION6, f"Expected ACTION6, got {action}"
        assert hasattr(action, "action_data"), "ACTION6 must have action_data"
        ax, ay = action.action_data.x, action.action_data.y
        assert 0 <= ax <= 63, f"Action coordinate x={ax} out of range [0, 63]"
        assert 0 <= ay <= 63, f"Action coordinate y={ay} out of range [0, 63]"
        emitted_coords.append((ax, ay))

    print(f"Emitted ACTION6 coordinates: {emitted_coords}")
    assert (21, 11) in emitted_coords, "Did not emit cluster 1 centroid"
    assert (50, 40) in emitted_coords, "Did not emit cluster 2 centroid"
    print("PASS: test_action6_synthetic_clusters")


def test_action6_empty_and_all_background():
    print("\n--- [Test 1b] ACTION6 All-Background Grid Fallback ---")
    grid = np.zeros((64, 64), dtype=np.int8)  # Completely empty frame
    agent = create_mock_agent()
    frame = make_frame(grid, available_actions=[GameAction.ACTION6])
    agent.memory.probing_phase = False

    action = agent.choose_action([frame], frame)
    assert action == GameAction.ACTION6
    assert action.action_data.x == 32 and action.action_data.y == 32, f"Fallback coordinate should be (32, 32), got ({action.action_data.x}, {action.action_data.y})"
    print("PASS: test_action6_empty_and_all_background")


# ==============================================================================
# 2. ACTION5 Contextual Interaction Affordance Tests
# ==============================================================================
def test_action5_adjacent_affordance():
    print("\n--- [Test 2] ACTION5 Contextual Interaction Affordance ---")
    grid = np.zeros((64, 64), dtype=np.int8)
    agent = create_mock_agent()
    agent.memory.probing_phase = False
    agent.memory.plan_queue.clear()

    # Set avatar position at (25, 25)
    agent.memory.avatar_pos = (25, 25)
    agent.memory.step_size = 1
    agent.memory.avatar_colors = {1}
    grid[25, 25] = 1

    # Case A: No adjacent entity -> ACTION5 should NOT trigger
    frame_empty = make_frame(grid, available_actions=[GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4, GameAction.ACTION5])
    assert not agent._should_trigger_action5(grid, bg_color=0), "Spurious ACTION5 triggered on empty surroundings!"

    # Case B: Entity placed directly adjacent at (25, 26) (Right)
    grid_adj = grid.copy()
    grid_adj[25, 26] = 3  # Color 3 object adjacent to avatar
    assert agent._should_trigger_action5(grid_adj, bg_color=0), "ACTION5 should trigger when entity is adjacent!"

    # Test choose_action emits ACTION5 when adjacent
    frame_adj = make_frame(grid_adj, available_actions=[GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4, GameAction.ACTION5])
    action = agent.choose_action([frame_adj], frame_adj)
    assert action == GameAction.ACTION5, f"Expected ACTION5 when adjacent to entity, got {action}"
    assert action.reasoning["why"] == "contextual_affordance_action5"

    # Case C: Adjacent entity has the SAME color as avatar -> should NOT trigger
    grid_same_color = grid.copy()
    grid_same_color[25, 26] = 1  # Same as avatar color
    assert not agent._should_trigger_action5(grid_same_color, bg_color=0), "ACTION5 should not trigger on avatar's own pixels"

    # Case D: Entity is diagonal (26, 26) -> should NOT trigger Manhattan adjacency
    grid_diag = grid.copy()
    grid_diag[26, 26] = 3
    assert not agent._should_trigger_action5(grid_diag, bg_color=0), "ACTION5 should not trigger on diagonal non-adjacent entity"

    # Case E: ACTION5 NOT in available_actions -> agent must NOT emit ACTION5
    frame_no_act5 = make_frame(grid_adj, available_actions=[GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4])
    action_no5 = agent.choose_action([frame_no_act5], frame_no_act5)
    assert action_no5 != GameAction.ACTION5, f"Emitted ACTION5 when it was NOT available: {action_no5}"

    print("PASS: test_action5_adjacent_affordance")


# ==============================================================================
# 3. Terminal Safety Tests
# ==============================================================================
def test_terminal_safety_and_reset():
    print("\n--- [Test 3] Terminal Safety & Memory Reset ---")
    grid = np.zeros((64, 64), dtype=np.int8)
    agent = create_mock_agent()

    # Populate working memory with state to verify reset
    agent.memory.plan_queue.append(GameAction.ACTION1)
    agent.memory.click_queue.append((10, 10))
    agent.memory.visited_states["dummy_hash"] = 5
    agent.memory.probing_phase = False

    # 1. Test "GAME_OVER" string state
    frame_go_str = FrameData(frame=[grid], state="GAME_OVER", available_actions=[1, 2, 3, 4, 5, 6])
    act1 = agent.choose_action([frame_go_str], frame_go_str)
    assert act1 == GameAction.RESET, f"Expected RESET on 'GAME_OVER', got {act1}"
    assert len(agent.memory.plan_queue) == 0, "plan_queue was not cleared on RESET"
    assert len(agent.memory.click_queue) == 0, "click_queue was not cleared on RESET"
    assert len(agent.memory.visited_states) == 0, "visited_states was not cleared on RESET"
    assert agent.memory.probing_phase is True, "probing_phase was not reset to True on RESET"

    # 2. Test GameState.GAME_OVER enum state
    agent.memory.plan_queue.append(GameAction.ACTION2)
    frame_go_enum = FrameData(frame=[grid], state=GameState.GAME_OVER, available_actions=[1, 2, 3, 4, 5, 6])
    act2 = agent.choose_action([frame_go_enum], frame_go_enum)
    assert act2 == GameAction.RESET, f"Expected RESET on GameState.GAME_OVER, got {act2}"

    # 3. Test "NOT_PLAYED" state
    frame_np_str = FrameData(frame=[grid], state="NOT_PLAYED", available_actions=[1, 2, 3, 4, 5, 6])
    act3 = agent.choose_action([frame_np_str], frame_np_str)
    assert act3 == GameAction.RESET, f"Expected RESET on 'NOT_PLAYED', got {act3}"

    # 4. Test GameState.NOT_PLAYED enum state
    frame_np_enum = FrameData(frame=[grid], state=GameState.NOT_PLAYED, available_actions=[1, 2, 3, 4, 5, 6])
    act4 = agent.choose_action([frame_np_enum], frame_np_enum)
    assert act4 == GameAction.RESET, f"Expected RESET on GameState.NOT_PLAYED, got {act4}"

    # 5. Test is_done method on WIN
    frame_win = FrameData(frame=[grid], state=GameState.WIN, available_actions=[1, 2])
    assert agent.is_done([frame_win], frame_win) is True, "is_done should return True on GameState.WIN"

    frame_win_str = FrameData(frame=[grid], state="WIN", available_actions=[1, 2])
    assert agent.is_done([frame_win_str], frame_win_str) is True, "is_done should return True on 'WIN'"

    frame_running = FrameData(frame=[grid], state=GameState.NOT_FINISHED, available_actions=[1, 2])
    assert agent.is_done([frame_running], frame_running) is False, "is_done should return False on NOT_FINISHED"

    print("PASS: test_terminal_safety_and_reset")


# ==============================================================================
# 4. Loop Interception & Orthogonal Escapes
# ==============================================================================
def test_loop_interception_and_escape():
    print("\n--- [Test 4] Loop Interceptor & Orthogonal Escapes ---")
    grid = np.zeros((64, 64), dtype=np.int8)
    grid[30, 30] = 1  # avatar
    grid_bytes = grid.tobytes()

    # Case A: Identical State Hash Loop (3 repetitions)
    mem = WorkingMemory(loop_threshold=3)
    pos = (30, 30)
    for _ in range(2):
        mem.record_step(GameAction.ACTION1, grid_bytes, pos)
        assert not mem.is_in_loop(), "Should not detect loop on 2 visits with threshold 3"
    mem.record_step(GameAction.ACTION1, grid_bytes, pos)
    assert mem.is_in_loop(), "Failed to detect loop on 3rd identical state visit"
    print("  State-hash loop detection: OK")

    # Case B: Ping-Pong 2-Cycle Action Oscillation (ACTION1 -> ACTION2 -> ACTION1 -> ACTION2)
    mem2 = WorkingMemory(loop_threshold=5)
    for a in (GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION1, GameAction.ACTION2):
        mem2.recent_actions.append(a)
    assert mem2.is_in_loop(), "Failed to detect 2-cycle ping-pong action loop"
    print("  2-cycle ping-pong loop detection: OK")

    # Case C: 3-Cycle Action Oscillation (A1 -> A3 -> A2 -> A1 -> A3 -> A2)
    mem3 = WorkingMemory(loop_threshold=5)
    for a in (GameAction.ACTION1, GameAction.ACTION3, GameAction.ACTION2, GameAction.ACTION1, GameAction.ACTION3, GameAction.ACTION2):
        mem3.recent_actions.append(a)
    assert mem3.is_in_loop(), "Failed to detect 3-cycle action loop"
    print("  3-cycle action loop detection: OK")

    # Case D: Orthogonal Escape Verification in Agent
    agent = create_mock_agent()
    agent.memory.probing_phase = False
    legal_moves = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4]

    # Vertical oscillation: last action was ACTION1 (Up) -> orthogonal escape must be horizontal (ACTION4 or ACTION3)
    agent.last_action = GameAction.ACTION1
    escape_v = agent._get_orthogonal_escape(legal_moves)
    assert escape_v in (GameAction.ACTION3, GameAction.ACTION4), f"Vertical oscillation escape must be horizontal, got {escape_v}"

    # Horizontal oscillation: last action was ACTION3 (Left) -> orthogonal escape must be vertical (ACTION1 or ACTION2)
    agent.last_action = GameAction.ACTION3
    escape_h = agent._get_orthogonal_escape(legal_moves)
    assert escape_h in (GameAction.ACTION1, GameAction.ACTION2), f"Horizontal oscillation escape must be vertical, got {escape_h}"

    # Verify agent executes escape during choose_action:
    # Setup: agent just performed ACTION2 after ACTION1, ACTION2, ACTION1
    agent.memory.recent_actions.clear()
    agent.memory.recent_actions.extend([GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION1])
    agent.last_action = GameAction.ACTION2  # When choose_action runs, record_step will append ACTION2 -> [A1, A2, A1, A2]
    agent.memory.plan_queue.append(GameAction.ACTION1)  # Queue should be purged!
    frame = make_frame(grid, available_actions=legal_moves)

    action = agent.choose_action([frame], frame)
    assert action in (GameAction.ACTION3, GameAction.ACTION4), f"Agent failed to execute orthogonal escape on loop! Emitted {action}"
    assert len(agent.memory.plan_queue) == 0, "Plan queue was not purged on loop break"
    assert action.reasoning["why"] == "orthogonal_escape_horizontal", f"Unexpected reasoning: {action.reasoning}"
    print("  Agent orthogonal escape execution: OK")
    print("PASS: test_loop_interception_and_escape")


# ==============================================================================
# 5. Micro-Benchmark Latency & Hardware Budgeting
# ==============================================================================
def test_micro_benchmark_latency():
    print("\n--- [Test 5] Micro-Benchmark Latency (1000 consecutive choose_action calls) ---")
    agent = create_mock_agent()
    legal_moves = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4, GameAction.ACTION5, GameAction.ACTION6]

    # Create realistic mixed environments:
    # 1. Sparse navigation grid with obstacles
    grid_sparse = np.zeros((64, 64), dtype=np.int8)
    grid_sparse[10:15, 10:15] = 2  # target sprite
    grid_sparse[20:25, 20:25] = 3  # obstacle

    # 2. Dense grid with 10 small sprites
    grid_dense = np.zeros((64, 64), dtype=np.int8)
    for i in range(10):
        r = 5 + i * 4
        c = 5 + (i * 5) % 50
        grid_dense[r:r+2, c:c+2] = (i % 8) + 1

    frames_sparse = [make_frame(grid_sparse, available_actions=legal_moves)]
    frames_dense = [make_frame(grid_dense, available_actions=legal_moves)]

    N = 1000
    latencies_us = []

    # Warm up JIT/caches
    for _ in range(20):
        agent.choose_action(frames_sparse, frames_sparse[0])

    t0_total = time.perf_counter()
    for i in range(N):
        curr_frame = frames_dense[0] if (i % 2 == 0) else frames_sparse[0]
        t0 = time.perf_counter()
        act = agent.choose_action([curr_frame], curr_frame)
        t1 = time.perf_counter()
        latencies_us.append((t1 - t0) * 1_000_000)
    t1_total = time.perf_counter()

    total_time_s = t1_total - t0_total
    avg_fps = N / total_time_s
    latencies_ms = np.array(latencies_us) / 1000.0

    mean_ms = float(np.mean(latencies_ms))
    median_ms = float(np.median(latencies_ms))
    p95_ms = float(np.percentile(latencies_ms, 95))
    p99_ms = float(np.percentile(latencies_ms, 99))
    max_ms = float(np.max(latencies_ms))

    print(f"Total iterations: {N}")
    print(f"Total time:       {total_time_s:.4f} s")
    print(f"Average FPS:      {avg_fps:.1f} FPS (Target: > 300 FPS)")
    print(f"Mean latency:     {mean_ms:.4f} ms (Ceiling: < 3.0 ms)")
    print(f"Median latency:   {median_ms:.4f} ms")
    print(f"P95 latency:      {p95_ms:.4f} ms")
    print(f"P99 latency:      {p99_ms:.4f} ms")
    print(f"Max latency:      {max_ms:.4f} ms")

    assert mean_ms < 3.0, f"Mean latency {mean_ms:.4f} ms exceeds 3.0 ms threshold!"
    assert avg_fps > 300.0, f"Throughput {avg_fps:.1f} FPS is below 300 FPS threshold!"
    print("PASS: test_micro_benchmark_latency")
    return {
        "iterations": N,
        "total_time_s": total_time_s,
        "avg_fps": avg_fps,
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "p95_ms": p95_ms,
        "p99_ms": p99_ms,
        "max_ms": max_ms,
    }


# ==============================================================================
# 6. Extreme Adversarial Stress Frames
# ==============================================================================
def test_adversarial_frames():
    print("\n--- [Test 6] Extreme Adversarial Stress Frames ---")
    agent = create_mock_agent()

    # 1. Checkerboard 64x64 (high component entropy)
    checkerboard = np.indices((64, 64)).sum(axis=0) % 2
    f_check = make_frame(checkerboard.astype(np.int8))
    act_check = agent.choose_action([f_check], f_check)
    assert act_check is not None, "Agent crashed on 64x64 checkerboard grid!"

    # 2. High-entropy random noise grid (0..15)
    rng = np.random.RandomState(42)
    noise_grid = rng.randint(0, 16, size=(64, 64), dtype=np.int8)
    f_noise = make_frame(noise_grid)
    act_noise = agent.choose_action([f_noise], f_noise)
    assert act_noise is not None, "Agent crashed on random noise grid!"

    # 3. Degenerate frame (shape mismatch / extra dimensions)
    degen_grid = np.zeros((32, 32), dtype=np.int8)
    f_degen = FrameData(frame=[degen_grid], state=GameState.NOT_FINISHED, available_actions=[1, 2])
    act_degen = agent.choose_action([f_degen], f_degen)
    assert act_degen is not None, "Agent crashed on 32x32 frame grid!"

    # 4. Null frame
    f_null = FrameData(frame=[], state=GameState.NOT_FINISHED, available_actions=[1, 2])
    act_null = agent.choose_action([f_null], f_null)
    assert act_null is not None, "Agent crashed on empty frame!"

    print("PASS: test_adversarial_frames")


if __name__ == "__main__":
    print("======================================================================")
    print("STARTING CHALLENGER 2 EMPIRICAL VERIFICATION & STRESS TEST HARNESS")
    print("======================================================================")

    test_action6_synthetic_clusters()
    test_action6_empty_and_all_background()
    test_action5_adjacent_affordance()
    test_terminal_safety_and_reset()
    test_loop_interception_and_escape()
    bench_results = test_micro_benchmark_latency()
    test_adversarial_frames()

    print("\n======================================================================")
    print("ALL EMPIRICAL CHALLENGER 2 TESTS PASSED SUCCESSFULLY!")
    print(f"LATENCY BENCHMARK: Mean {bench_results['mean_ms']:.4f} ms | FPS {bench_results['avg_fps']:.1f}")
    print("======================================================================")

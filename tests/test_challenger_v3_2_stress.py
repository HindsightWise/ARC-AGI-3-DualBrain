"""
Empirical Stress-Testing and Non-Regression Verification Harness for ARC-AGI-3 Version 3 Agent.
Challenger 2: Throughput Stress & Action Space Non-Regression Harness.

MANDATORY CHECKS:
1. 1000+ Continuous iterations latency & throughput (< 3.0 ms mean, > 300 FPS, target > 5000 FPS)
   across both puzzle frames (ls20) and generic/click frames (vc33).
2. ACTION6 coordinate targeting on vc33: bounding boxes, coordinate parameters {"x": col, "y": row} in [0, 63].
3. Terminal safety: immediate GameAction.RESET on GAME_OVER and NOT_PLAYED (both enums and strings),
   with full working memory purge.
4. Memory bounds & leak audit: 2500+ step sequence tracemalloc evaluation.
5. Adversarial stress & corrupted frame edge cases.
"""
from __future__ import annotations

import collections
import os
import sys
import time
import tracemalloc
import unittest
from unittest.mock import MagicMock
import numpy as np

# Ensure repository root and vendor framework are in sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VENDOR = os.path.join(ROOT, "vendor", "ARC-AGI-3-Agents")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if VENDOR not in sys.path:
    sys.path.insert(0, VENDOR)

from arcengine import FrameData, GameAction, GameState
from agent.my_agent import (
    MyAgent,
    ScenePerception,
    AvatarTracker,
    GeodesicPlanner,
    WorkingMemory,
    BlockConfiguration,
    TemplatePerception,
    OperatorPadDetector,
    ConfigurationPlanner,
)


def create_agent(game_id: str = "test_game") -> MyAgent:
    """Instantiate a MyAgent instance with mocked environment."""
    mock_env = MagicMock()
    agent = MyAgent(
        card_id="challenger-v3-2",
        game_id=game_id,
        agent_name=f"Challenger2Agent.{game_id}",
        ROOT_URL="http://localhost",
        record=False,
        arc_env=mock_env,
    )
    return agent


def make_frame(
    grid: np.ndarray,
    state=GameState.NOT_FINISHED,
    available_actions=None,
    levels_completed: int = 0,
) -> FrameData:
    """Wrap numpy grid into a valid FrameData object."""
    if available_actions is None:
        available_actions = [
            GameAction.ACTION1,
            GameAction.ACTION2,
            GameAction.ACTION3,
            GameAction.ACTION4,
            GameAction.ACTION5,
            GameAction.ACTION6,
        ]
    return FrameData(
        frame=[grid.astype(np.int8)],
        state=state,
        available_actions=[
            (a.value if hasattr(a, "value") else int(a)) for a in available_actions
        ],
        levels_completed=levels_completed,
    )


# ==============================================================================
# 1. ACTION6 Coordinate Targeting & Sprite Bounding Boxes (vc33 & Synthetic)
# ==============================================================================
def test_action6_vc33_targeting():
    print("\n======================================================================")
    print("--- [Test 1] ACTION6 Coordinate Targeting & Sprite Bounding Boxes ---")
    print("======================================================================")

    # 1. Test real vc33 frame via arc_agi
    try:
        import arc_agi
        arc = arc_agi.Arcade()
        env = arc.make("vc33")
        real_frame = env.step(GameAction.RESET)
        has_real_env = True
        print("  Successfully acquired live vc33 frame from environment.")
    except Exception as e:
        has_real_env = False
        print(f"  Note: Live arc_agi env setup warning ({e}). Generating exact synthetic vc33 frame.")
        real_frame = None

    agent = create_agent(game_id="vc33")

    if has_real_env and real_frame is not None:
        curr_frame = real_frame
        frames = [curr_frame]
        actions_tested = 0
        for step in range(50):
            action = agent.choose_action(frames, curr_frame)
            actions_tested += 1
            assert action == GameAction.ACTION6, f"Expected ACTION6 on vc33, got {action}"
            assert hasattr(action, "action_data"), "Emitted ACTION6 action lacks action_data attribute"
            ax = action.action_data.x
            ay = action.action_data.y
            assert 0 <= ax <= 63, f"Action6 coordinate x={ax} outside [0, 63]"
            assert 0 <= ay <= 63, f"Action6 coordinate y={ay} outside [0, 63]"
            # Step environment to get next frame
            curr_frame = env.step(action)
            frames.append(curr_frame)
            if curr_frame.state in (GameState.WIN, GameState.GAME_OVER):
                break
        print(f"  Live vc33 test: Verified {actions_tested} consecutive ACTION6 actions in range [0, 63].")

    # 2. Synthetic Boundary & Corner Sprite Targeting
    print("  Testing synthetic boundary, corner, and HUD-exclusion cases:")
    grid = np.zeros((64, 64), dtype=np.int8)

    # Corner 1: (row 0, col 0)
    grid[0, 0] = 2
    # Corner 2: (row 0, col 63)
    grid[0, 63] = 3
    # Corner 3: (row 55, col 0) - edge of playable area
    grid[55, 0] = 4
    # Corner 4: (row 55, col 63)
    grid[55, 63] = 7
    # HUD sprite: (row 58..60, col 10..12) - should be ignored by ScenePerception
    grid[58:61, 10:13] = 9

    bg, canvas = ScenePerception.detect_background_and_canvas(grid)
    sprites = ScenePerception.segment_sprites(grid, bg, canvas)
    centroids = {s["centroid"] for s in sprites}

    print(f"  Found centroids: {centroids}")
    assert (0, 0) in centroids, "Extreme top-left corner (0, 0) not detected"
    assert (63, 0) in centroids, "Extreme top-right corner (63, 0) not detected"
    assert (0, 55) in centroids, "Extreme bottom-left corner (0, 55) not detected"
    assert (63, 55) in centroids, "Extreme bottom-right corner (63, 55) not detected"

    # Verify HUD exclusion: no sprite centroid has y >= 56
    for s in sprites:
        assert s["centroid"][1] < 56, f"HUD sprite detected at cy={s['centroid'][1]} >= 56"

    # Test MyAgent emission of synthetic sprites
    agent_synth = create_agent(game_id="vc33_synth")
    f_synth = make_frame(grid, available_actions=[GameAction.ACTION6])

    emitted = []
    for _ in range(len(sprites) + 2):
        act = agent_synth.choose_action([f_synth], f_synth)
        assert act == GameAction.ACTION6
        assert 0 <= act.action_data.x <= 63
        assert 0 <= act.action_data.y <= 63
        emitted.append((act.action_data.x, act.action_data.y))

    print(f"  Emitted coordinates from synthetic frame: {emitted}")
    assert (0, 0) in emitted
    assert (63, 0) in emitted
    assert (0, 55) in emitted
    assert (63, 55) in emitted

    # 3. Fallback on completely empty grid
    empty_grid = np.zeros((64, 64), dtype=np.int8)
    f_empty = make_frame(empty_grid, available_actions=[GameAction.ACTION6])
    agent_empty = create_agent(game_id="vc33_empty")
    act_empty = agent_empty.choose_action([f_empty], f_empty)
    assert act_empty == GameAction.ACTION6
    assert (act_empty.action_data.x, act_empty.action_data.y) == (32, 32), (
        f"Expected center fallback (32, 32), got ({act_empty.action_data.x}, {act_empty.action_data.y})"
    )

    print("  PASS: test_action6_vc33_targeting")


# ==============================================================================
# 2. Terminal Safety & Immediate RESET Verification
# ==============================================================================
def test_terminal_safety_and_reset():
    print("\n======================================================================")
    print("--- [Test 2] Terminal Safety & Immediate RESET Verification ---")
    print("======================================================================")

    agent = create_agent(game_id="safety_test")
    dummy_grid = np.zeros((64, 64), dtype=np.int8)

    # 1. Populate agent memory and queues to verify clean purge
    agent.memory.plan_queue.extend([GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3])
    agent.memory.click_queue.extend([(10, 10), (20, 20)])
    agent.memory.visited_states["dummy_hash"] = 42
    agent.puzzle_mode = True
    agent.puzzle_elements = {"test": 123}
    agent.puzzle_avatar_tile = (5, 5)
    agent.last_levels_completed = 1
    agent.last_action = GameAction.ACTION1

    # Test Enum GameState.GAME_OVER
    f_go_enum = FrameData(frame=[dummy_grid], state=GameState.GAME_OVER, available_actions=[1, 2, 3, 4, 5, 6])
    act1 = agent.choose_action([f_go_enum], f_go_enum)
    assert act1 == GameAction.RESET, f"Expected GameAction.RESET on GameState.GAME_OVER, got {act1}"
    assert len(agent.memory.plan_queue) == 0, "plan_queue was not purged on GAME_OVER"
    assert len(agent.memory.click_queue) == 0, "click_queue was not purged on GAME_OVER"
    assert len(agent.memory.visited_states) == 0, "visited_states was not purged on GAME_OVER"
    assert agent.puzzle_mode is None, "puzzle_mode was not reset to None on GAME_OVER"
    assert agent.puzzle_elements is None, "puzzle_elements was not reset to None on GAME_OVER"
    assert agent.puzzle_avatar_tile is None, "puzzle_avatar_tile was not reset to None on GAME_OVER"
    assert agent.last_levels_completed == 0, "last_levels_completed was not reset to 0 on GAME_OVER"
    assert agent.last_action == GameAction.RESET, "last_action was not set to GameAction.RESET"
    print("  GameState.GAME_OVER enum: Passed (RESET emitted, state cleanly purged).")

    # Re-populate memory
    agent.memory.plan_queue.append(GameAction.ACTION4)
    agent.puzzle_mode = True

    # Test String "GAME_OVER"
    f_go_str = FrameData(frame=[dummy_grid], state="GAME_OVER", available_actions=[1, 2, 3, 4, 5, 6])
    act2 = agent.choose_action([f_go_str], f_go_str)
    assert act2 == GameAction.RESET, f"Expected GameAction.RESET on 'GAME_OVER', got {act2}"
    assert len(agent.memory.plan_queue) == 0, "plan_queue was not purged on 'GAME_OVER'"
    assert agent.puzzle_mode is None, "puzzle_mode was not reset on 'GAME_OVER'"
    print("  'GAME_OVER' string: Passed (RESET emitted, state cleanly purged).")

    # Test Enum GameState.NOT_PLAYED
    agent.memory.plan_queue.append(GameAction.ACTION2)
    f_np_enum = FrameData(frame=[dummy_grid], state=GameState.NOT_PLAYED, available_actions=[1, 2, 3, 4, 5, 6])
    act3 = agent.choose_action([f_np_enum], f_np_enum)
    assert act3 == GameAction.RESET, f"Expected GameAction.RESET on GameState.NOT_PLAYED, got {act3}"
    assert len(agent.memory.plan_queue) == 0, "plan_queue was not purged on GameState.NOT_PLAYED"
    print("  GameState.NOT_PLAYED enum: Passed.")

    # Test String "NOT_PLAYED"
    agent.memory.plan_queue.append(GameAction.ACTION3)
    f_np_str = FrameData(frame=[dummy_grid], state="NOT_PLAYED", available_actions=[1, 2, 3, 4, 5, 6])
    act4 = agent.choose_action([f_np_str], f_np_str)
    assert act4 == GameAction.RESET, f"Expected GameAction.RESET on 'NOT_PLAYED', got {act4}"
    assert len(agent.memory.plan_queue) == 0, "plan_queue was not purged on 'NOT_PLAYED'"
    print("  'NOT_PLAYED' string: Passed.")

    # Test Empty legal actions
    f_no_legal = FrameData(frame=[dummy_grid], state=GameState.NOT_FINISHED, available_actions=[0])
    act5 = agent.choose_action([f_no_legal], f_no_legal)
    assert act5 == GameAction.RESET, f"Expected RESET when available_actions=[0], got {act5}"
    print("  Empty/Zero legal actions: Passed (emits RESET).")

    # Test is_done method
    f_win = FrameData(frame=[dummy_grid], state=GameState.WIN, available_actions=[1, 2])
    f_win_str = FrameData(frame=[dummy_grid], state="WIN", available_actions=[1, 2])
    f_run = FrameData(frame=[dummy_grid], state=GameState.NOT_FINISHED, available_actions=[1, 2])
    assert agent.is_done([f_win], f_win) is True, "is_done failed for GameState.WIN"
    assert agent.is_done([f_win_str], f_win_str) is True, "is_done failed for 'WIN'"
    assert agent.is_done([f_run], f_run) is False, "is_done returned True for NOT_FINISHED"
    print("  is_done() semantics: Passed.")

    print("  PASS: test_terminal_safety_and_reset")


# ==============================================================================
# 3. Throughput Stress & Step Latency Benchmark (1000+ Iterations)
# ==============================================================================
def create_synthetic_ls20_grid() -> np.ndarray:
    """Synthesizes a realistic 64x64 ls20 puzzle board with HUD and pads."""
    grid = np.full((64, 64), 3, dtype=np.int8)  # Floor is color 3

    # Add walls (color 4) on perimeter and interior
    grid[0:5, :] = 4
    grid[:, 0:4] = 4
    grid[:, 59:64] = 4

    # Bottom-left HUD area (rows 55:61, cols 3:9)
    # Put Shape 0 (gngifvjddu) at rotation 0, color 12 (palette[0]) scaled 2x
    shp0 = TemplatePerception.SHAPES[0]
    shp_scaled = np.kron(shp0, np.ones((2, 2), dtype=bool))
    grid[55:61, 3:9] = np.where(shp_scaled, 12, 5)

    # Tile (gx=2, gy=2): avatar starting tile
    # patch: top 2 rows color 12, bottom 3 rows color 9
    r_av = 5 * 2
    c_av = 4 + 5 * 2
    grid[r_av:r_av + 2, c_av:c_av + 5] = 12
    grid[r_av + 2:r_av + 5, c_av:c_av + 5] = 9

    # Tile (gx=4, gy=2): rotation pad (3 zeros, 2 ones)
    r_rot = 5 * 2
    c_rot = 4 + 5 * 4
    grid[r_rot, c_rot:c_rot + 3] = 0
    grid[r_rot, c_rot + 3:c_rot + 5] = 1

    # Tile (gx=6, gy=2): color swap pad (multi-palette colors)
    r_col = 5 * 2
    c_col = 4 + 5 * 6
    grid[r_col, c_col] = 12
    grid[r_col, c_col + 1] = 9
    grid[r_col, c_col + 2] = 14

    # Tile (gx=8, gy=2): shape morph pad (4 zeros)
    r_shp = 5 * 2
    c_shp = 4 + 5 * 8
    grid[r_shp, c_shp:c_shp + 4] = 0

    # Tile (gx=10, gy=2): goal slot with frame 5 and internal 3x3 goal template (Shape 0, color 12, rot 0)
    r_goal = 5 * 2
    c_goal = 4 + 5 * 10
    grid[r_goal:r_goal + 5, c_goal:c_goal + 5] = 5
    grid[r_goal + 1:r_goal + 4, c_goal + 1:c_goal + 4] = np.where(shp0, 12, 5)

    return grid


def create_synthetic_vc33_grid() -> np.ndarray:
    """Synthesizes a realistic 64x64 vc33 grid with interactive sprite elements."""
    grid = np.zeros((64, 64), dtype=np.int8)
    # Base canvas bar
    grid[10:14, 10:54] = 5
    # Clickable sprites
    grid[20:28, 15:18] = 1
    grid[30:38, 25:27] = 1
    grid[40:48, 40:42] = 1
    return grid


def create_synthetic_generic_grid() -> np.ndarray:
    """Synthesizes a generic navigation grid with avatar and obstacle."""
    grid = np.zeros((64, 64), dtype=np.int8)
    # Avatar at (20, 20)
    grid[20, 20] = 1
    # Target sprite at (45, 45)
    grid[45:48, 45:48] = 2
    # Obstacle wall
    grid[30:35, 15:50] = 4
    return grid


def run_benchmark_suite(num_iterations: int = 1000):
    print("\n======================================================================")
    print(f"--- [Test 3] Step Latency & Throughput Benchmark ({num_iterations} Iterations) ---")
    print("======================================================================")

    grid_ls20 = create_synthetic_ls20_grid()
    grid_vc33 = create_synthetic_vc33_grid()
    grid_generic = create_synthetic_generic_grid()

    frame_ls20 = make_frame(grid_ls20, available_actions=[GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4])
    frame_vc33 = make_frame(grid_vc33, available_actions=[GameAction.ACTION6])
    frame_generic = make_frame(grid_generic, available_actions=[GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4])

    results = {}

    # --- Scenario A: Pure Puzzle Frames (ls20) ---
    agent_puzzle = create_agent(game_id="ls20_bench")
    # Warm up JIT / cache
    for _ in range(10):
        agent_puzzle.choose_action([frame_ls20], frame_ls20)

    t0_puzzle = time.perf_counter()
    latencies_puzzle = []
    for _ in range(num_iterations):
        t_start = time.perf_counter()
        act = agent_puzzle.choose_action([frame_ls20], frame_ls20)
        t_end = time.perf_counter()
        latencies_puzzle.append((t_end - t_start) * 1000.0)
    t1_puzzle = time.perf_counter()
    total_time_puzzle = t1_puzzle - t0_puzzle
    fps_puzzle = num_iterations / total_time_puzzle
    lat_arr_p = np.array(latencies_puzzle)

    results["puzzle"] = {
        "iterations": num_iterations,
        "total_time_s": total_time_puzzle,
        "fps": fps_puzzle,
        "mean_ms": float(np.mean(lat_arr_p)),
        "median_ms": float(np.median(lat_arr_p)),
        "p95_ms": float(np.percentile(lat_arr_p, 95)),
        "p99_ms": float(np.percentile(lat_arr_p, 99)),
        "max_ms": float(np.max(lat_arr_p)),
    }
    print(f"\n[Scenario A: Pure Puzzle Frames (ls20)]")
    print(f"  Iterations: {num_iterations}")
    print(f"  Total Time: {total_time_puzzle:.4f} s")
    print(f"  Throughput: {fps_puzzle:.1f} FPS  (Enforce: > 300 FPS)")
    print(f"  Mean:       {results['puzzle']['mean_ms']:.4f} ms (Enforce: < 3.0 ms)")
    print(f"  Median:     {results['puzzle']['median_ms']:.4f} ms")
    print(f"  P95:        {results['puzzle']['p95_ms']:.4f} ms")
    print(f"  P99:        {results['puzzle']['p99_ms']:.4f} ms")
    print(f"  Max:        {results['puzzle']['max_ms']:.4f} ms")

    assert results["puzzle"]["mean_ms"] < 3.0, f"Puzzle mean latency {results['puzzle']['mean_ms']} ms >= 3.0 ms"
    assert results["puzzle"]["fps"] > 300.0, f"Puzzle throughput {fps_puzzle} FPS <= 300 FPS"

    # --- Scenario B: Pure Generic / Click Frames (vc33) ---
    agent_vc33 = create_agent(game_id="vc33_bench")
    for _ in range(10):
        agent_vc33.choose_action([frame_vc33], frame_vc33)

    t0_vc33 = time.perf_counter()
    latencies_vc33 = []
    for _ in range(num_iterations):
        t_start = time.perf_counter()
        act = agent_vc33.choose_action([frame_vc33], frame_vc33)
        t_end = time.perf_counter()
        latencies_vc33.append((t_end - t_start) * 1000.0)
    t1_vc33 = time.perf_counter()
    total_time_vc33 = t1_vc33 - t0_vc33
    fps_vc33 = num_iterations / total_time_vc33
    lat_arr_vc = np.array(latencies_vc33)

    results["vc33"] = {
        "iterations": num_iterations,
        "total_time_s": total_time_vc33,
        "fps": fps_vc33,
        "mean_ms": float(np.mean(lat_arr_vc)),
        "median_ms": float(np.median(lat_arr_vc)),
        "p95_ms": float(np.percentile(lat_arr_vc, 95)),
        "p99_ms": float(np.percentile(lat_arr_vc, 99)),
        "max_ms": float(np.max(lat_arr_vc)),
    }
    print(f"\n[Scenario B: Pure Click Frames (vc33)]")
    print(f"  Iterations: {num_iterations}")
    print(f"  Total Time: {total_time_vc33:.4f} s")
    print(f"  Throughput: {fps_vc33:.1f} FPS  (Enforce: > 300 FPS)")
    print(f"  Mean:       {results['vc33']['mean_ms']:.4f} ms (Enforce: < 3.0 ms)")
    print(f"  Median:     {results['vc33']['median_ms']:.4f} ms")
    print(f"  P95:        {results['vc33']['p95_ms']:.4f} ms")
    print(f"  P99:        {results['vc33']['p99_ms']:.4f} ms")
    print(f"  Max:        {results['vc33']['max_ms']:.4f} ms")

    assert results["vc33"]["mean_ms"] < 3.0, f"vc33 mean latency {results['vc33']['mean_ms']} ms >= 3.0 ms"
    assert results["vc33"]["fps"] > 300.0, f"vc33 throughput {fps_vc33} FPS <= 300 FPS"

    # --- Scenario C: Mixed Alternating Frames (ls20 -> vc33 -> generic) ---
    agent_mixed = create_agent(game_id="mixed_bench")
    stream = [frame_ls20, frame_vc33, frame_generic]
    for i in range(15):
        f = stream[i % 3]
        agent_mixed.choose_action([f], f)

    t0_mixed = time.perf_counter()
    latencies_mixed = []
    for i in range(num_iterations):
        f = stream[i % 3]
        t_start = time.perf_counter()
        act = agent_mixed.choose_action([f], f)
        t_end = time.perf_counter()
        latencies_mixed.append((t_end - t_start) * 1000.0)
    t1_mixed = time.perf_counter()
    total_time_mixed = t1_mixed - t0_mixed
    fps_mixed = num_iterations / total_time_mixed
    lat_arr_m = np.array(latencies_mixed)

    results["mixed"] = {
        "iterations": num_iterations,
        "total_time_s": total_time_mixed,
        "fps": fps_mixed,
        "mean_ms": float(np.mean(lat_arr_m)),
        "median_ms": float(np.median(lat_arr_m)),
        "p95_ms": float(np.percentile(lat_arr_m, 95)),
        "p99_ms": float(np.percentile(lat_arr_m, 99)),
        "max_ms": float(np.max(lat_arr_m)),
    }
    print(f"\n[Scenario C: Alternating Mixed Frames (ls20 + vc33 + generic)]")
    print(f"  Iterations: {num_iterations}")
    print(f"  Total Time: {total_time_mixed:.4f} s")
    print(f"  Throughput: {fps_mixed:.1f} FPS  (Enforce: > 300 FPS)")
    print(f"  Mean:       {results['mixed']['mean_ms']:.4f} ms (Enforce: < 3.0 ms)")
    print(f"  Median:     {results['mixed']['median_ms']:.4f} ms")
    print(f"  P95:        {results['mixed']['p95_ms']:.4f} ms")
    print(f"  P99:        {results['mixed']['p99_ms']:.4f} ms")
    print(f"  Max:        {results['mixed']['max_ms']:.4f} ms")

    assert results["mixed"]["mean_ms"] < 3.0, f"Mixed mean latency {results['mixed']['mean_ms']} ms >= 3.0 ms"
    assert results["mixed"]["fps"] > 300.0, f"Mixed throughput {fps_mixed} FPS <= 300 FPS"

    print("  PASS: test_micro_benchmark_1000_iterations")
    return results


# ==============================================================================
# 4. Memory Bounds & Leak Audit (Extended 2500 Steps)
# ==============================================================================
def test_memory_bounds_and_leaks(steps: int = 2500):
    print("\n======================================================================")
    print(f"--- [Test 4] Memory Bounds & Leak Audit ({steps} Continuous Steps) ---")
    print("======================================================================")

    tracemalloc.start()
    agent = create_agent(game_id="memory_leak_test")

    # Construct rotating varied frames
    frames = [
        make_frame(create_synthetic_ls20_grid(), available_actions=[GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4]),
        make_frame(create_synthetic_vc33_grid(), available_actions=[GameAction.ACTION6]),
        make_frame(create_synthetic_generic_grid(), available_actions=[GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4]),
    ]

    checkpoints = [500, 1000, 1500, 2000, 2500]
    mem_snapshots = {}

    current, peak = tracemalloc.get_traced_memory()
    mem_snapshots[0] = current / 1024.0

    print(f"  Initial memory baseline: {mem_snapshots[0]:.2f} KB")

    for i in range(1, steps + 1):
        f = frames[i % len(frames)]
        act = agent.choose_action([f], f)
        if i in checkpoints:
            curr, _ = tracemalloc.get_traced_memory()
            mem_snapshots[i] = curr / 1024.0
            print(f"  Step {i:4d} | Current Mem: {mem_snapshots[i]:.2f} KB | "
                  f"Plan Queue: {len(agent.memory.plan_queue)} | "
                  f"Visited States: {len(agent.memory.visited_states)}")

    peak_kb = tracemalloc.get_traced_memory()[1] / 1024.0
    tracemalloc.stop()

    # Memory growth calculation
    last_step = max(k for k in mem_snapshots if k > 0)
    first_step = min(k for k in mem_snapshots if k > 0)
    delta_kb = mem_snapshots[last_step] - mem_snapshots[first_step]
    print(f"\n  Memory Delta (Step {last_step} vs Step {first_step}): {delta_kb:.2f} KB")
    print(f"  Peak Memory during test:              {peak_kb:.2f} KB")

    # Enforce strict ceiling: delta over steps must be < 500 KB (no memory leak)
    assert delta_kb < 500.0, f"Memory leak detected! Memory grew by {delta_kb:.2f} KB over {last_step - first_step} steps"
    assert len(agent.memory.recent_actions) <= 16, "recent_actions exceeded maxlen of 16"
    print("  PASS: test_memory_bounds_and_leaks")
    return {
        "snapshots_kb": mem_snapshots,
        "delta_kb": delta_kb,
        "peak_kb": peak_kb,
    }


# ==============================================================================
# 5. Adversarial Edge Cases & Degenerate Grid Handling
# ==============================================================================
def test_adversarial_degenerate_grids():
    print("\n======================================================================")
    print("--- [Test 5] Adversarial Edge Cases & Degenerate Grids ---")
    print("======================================================================")
    agent = create_agent(game_id="adversarial_test")

    # Case 1: Corrupted dimensions (32x32)
    small_grid = np.zeros((32, 32), dtype=np.int8)
    f_small = FrameData(frame=[small_grid], state=GameState.NOT_FINISHED, available_actions=[1, 2])
    act_small = agent.choose_action([f_small], f_small)
    assert act_small is not None, "Agent crashed on 32x32 grid"
    print("  Case 1: 32x32 sub-grid handled cleanly.")

    # Case 2: Over-sized grid (70x70)
    large_grid = np.ones((70, 70), dtype=np.int8)
    f_large = FrameData(frame=[large_grid], state=GameState.NOT_FINISHED, available_actions=[1, 2])
    act_large = agent.choose_action([f_large], f_large)
    assert act_large is not None, "Agent crashed on 70x70 grid"
    print("  Case 2: 70x70 oversized grid handled cleanly.")

    # Case 3: Negative pixel values (-2, -1) and out-of-range palette values (99)
    weird_grid = np.full((64, 64), -1, dtype=np.int8)
    weird_grid[10:15, 10:15] = -2
    weird_grid[20:25, 20:25] = 99
    f_weird = FrameData(frame=[weird_grid], state=GameState.NOT_FINISHED, available_actions=[1, 2])
    act_weird = agent.choose_action([f_weird], f_weird)
    assert act_weird is not None, "Agent crashed on negative/out-of-range pixel grid"
    print("  Case 3: Negative & out-of-range pixel values handled cleanly.")

    # Case 4: High-entropy checkerboard (maximum connected components)
    checkerboard = (np.indices((64, 64)).sum(axis=0) % 2).astype(np.int8)
    f_check = make_frame(checkerboard, available_actions=[GameAction.ACTION1, GameAction.ACTION2])
    act_check = agent.choose_action([f_check], f_check)
    assert act_check is not None, "Agent crashed on checkerboard grid"
    print("  Case 4: High-entropy checkerboard handled cleanly.")

    # Case 5: Partial HUD with no goal slots (ambiguous puzzle)
    ambig_grid = np.zeros((64, 64), dtype=np.int8)
    shp0 = TemplatePerception.SHAPES[0]
    shp_scaled = np.kron(shp0, np.ones((2, 2), dtype=bool))
    ambig_grid[55:61, 3:9] = np.where(shp_scaled, 12, 5)
    f_ambig = make_frame(ambig_grid, available_actions=[GameAction.ACTION1, GameAction.ACTION2])
    act_ambig = agent.choose_action([f_ambig], f_ambig)
    assert act_ambig is not None, "Agent crashed on partial HUD without goals"
    print("  Case 5: Partial HUD without goals handled cleanly.")

    print("  PASS: test_adversarial_degenerate_grids")


if __name__ == "__main__":
    print("======================================================================")
    print("STARTING CHALLENGER 2 V3 ADVERSARIAL STRESS TEST HARNESS")
    print("======================================================================")

    test_action6_vc33_targeting()
    test_terminal_safety_and_reset()
    bench_data = run_benchmark_suite(num_iterations=1000)
    mem_data = test_memory_bounds_and_leaks(steps=2500)
    test_adversarial_degenerate_grids()

    print("\n======================================================================")
    print("ALL CHALLENGER 2 V3 STRESS TESTS PASSED EMPIRICALLY!")
    print(f"BENCHMARK SUMMARY:")
    print(f"  ls20 Puzzle: Mean Latency = {bench_data['puzzle']['mean_ms']:.4f} ms | FPS = {bench_data['puzzle']['fps']:.1f}")
    print(f"  vc33 Click:  Mean Latency = {bench_data['vc33']['mean_ms']:.4f} ms | FPS = {bench_data['vc33']['fps']:.1f}")
    print(f"  Mixed Mode:  Mean Latency = {bench_data['mixed']['mean_ms']:.4f} ms | FPS = {bench_data['mixed']['fps']:.1f}")
    print("======================================================================")


class TestChallengerV3Stress(unittest.TestCase):
    def test_action6_vc33(self):
        test_action6_vc33_targeting()

    def test_terminal_safety(self):
        test_terminal_safety_and_reset()

    def test_benchmarks(self):
        run_benchmark_suite(num_iterations=200)

    def test_memory_bounds(self):
        test_memory_bounds_and_leaks(steps=1000)

    def test_adversarial(self):
        test_adversarial_degenerate_grids()

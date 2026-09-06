"""
Adversarial Stress Harness for M2: Latency, Throughput & Memory Stress
Author: teamwork_preview_challenger (Challenger 1)
"""
import time
import tracemalloc
import unittest
from unittest.mock import MagicMock
import numpy as np

from arcengine import FrameData, GameAction, GameState
from agent.my_agent import (
    MyAgent,
    WorkingMemory,
    OfflinePlanner,
    GeodesicPlanner,
    BellmanBackwardsPlanner,
    DIRECTION_VECTORS,
)


def create_agent(game_id: str = "adversarial_agent") -> MyAgent:
    mock_env = MagicMock()
    return MyAgent(
        card_id="challenger-1-adv",
        game_id=game_id,
        agent_name=f"Challenger1.{game_id}",
        ROOT_URL="http://localhost",
        record=False,
        arc_env=mock_env,
    )


class TestM2AdversarialStress(unittest.TestCase):
    def test_warp_cycle_stress(self):
        """Stress test: Cyclic warp portals (A <-> B). Verify A* does not cycle indefinitely."""
        passable = np.ones((40, 40), dtype=bool)
        # Circular portals: (5, 5) -> (25, 25), and (25, 25) -> (5, 5)
        warp_edges = {
            (5, 5): (25, 25),
            (25, 25): (5, 5),
        }
        # Dividing wall
        passable[15, :] = False
        passable[5, 5] = True
        passable[25, 25] = True

        start = (2, 2)
        goal = (35, 35)

        t0 = time.perf_counter()
        plan = OfflinePlanner.plan_geodesic_path(
            start=start,
            goal=goal,
            passable=passable,
            step_size=1,
            warp_edges=warp_edges,
        )
        t_elapsed = (time.perf_counter() - t0) * 1000.0

        self.assertIsNotNone(plan, "A* should find path through warp")
        self.assertLess(t_elapsed, 20.0, f"A* took too long: {t_elapsed:.2f} ms")

        # Simulate execution
        curr = start
        for act in plan:
            dr, dc = DIRECTION_VECTORS[act]
            curr = (curr[0] + dr, curr[1] + dc)
            if curr in warp_edges:
                curr = warp_edges[curr]
        self.assertEqual(curr, goal, "Agent must reach goal despite cyclic warp edges")

    def test_bellman_combinatorial_maze_stress(self):
        """Stress test: 3-key, 3-door combinatorial labyrinth. Measure solve latency & state count."""
        passable = np.ones((50, 50), dtype=bool)
        # Create 3 room partitions with doors
        passable[15, :] = False
        passable[15, 10] = True  # Door 0 -> requires Key 0
        passable[30, :] = False
        passable[30, 25] = True  # Door 1 -> requires Key 1
        passable[40, :] = False
        passable[40, 40] = True  # Door 2 -> requires Key 2

        start = (2, 2)
        goal = (48, 48)
        keys = [(5, 45), (20, 5), (35, 35)]
        doors = {
            (15, 10): 0,
            (30, 25): 1,
            (40, 40): 2,
        }

        t0 = time.perf_counter()
        plan = BellmanBackwardsPlanner.plan(
            start=start,
            goal=goal,
            keys=keys,
            doors=doors,
            passable=passable,
            step_size=1,
            max_states=20000,
        )
        t_elapsed = (time.perf_counter() - t0) * 1000.0

        self.assertIsNotNone(plan, "Bellman planner must solve 3-stage key-door maze")
        self.assertLess(t_elapsed, 50.0, f"Bellman planner took {t_elapsed:.2f} ms (budget < 50 ms)")

        # Verify sequential key unlocking
        curr = start
        collected_keys = set()
        for act in plan:
            dr, dc = DIRECTION_VECTORS[act]
            curr = (curr[0] + dr, curr[1] + dc)
            if curr in keys:
                collected_keys.add(keys.index(curr))
            if curr in doors:
                req = doors[curr]
                self.assertIn(req, collected_keys, f"Door at {curr} required key {req} before passage")

        self.assertEqual(curr, goal)

    def test_high_frequency_level_transition_memory_hygiene(self):
        """Stress test: 500 rapid level transitions. Verify working memory remains completely flat."""
        tracemalloc.start()
        agent = create_agent(game_id="rapid_transition_test")
        grid = np.zeros((64, 64), dtype=np.int8)
        grid[10, 10] = 1

        # Populate state initially
        f0 = FrameData(frame=[grid], state=GameState.NOT_FINISHED, available_actions=[1, 2, 3, 4], levels_completed=0)
        agent.choose_action([f0], f0)

        # Warm up 10 level transitions
        for lvl in range(1, 11):
            f_lvl = FrameData(frame=[grid], state=GameState.NOT_FINISHED, available_actions=[1, 2, 3, 4], levels_completed=lvl)
            agent.choose_action([f0, f_lvl], f_lvl)

        mem_10 = tracemalloc.get_traced_memory()[0] / 1024.0

        for lvl in range(11, 251):
            # Mutate state
            agent.affordance_ledger[(lvl % 60, lvl % 60)] = "rot"
            agent.memory.known_walls.add((lvl % 60, lvl % 60))
            agent.fleet_plan_queue.append(GameAction.ACTION1)

            # Advance level
            f_lvl = FrameData(frame=[grid], state=GameState.NOT_FINISHED, available_actions=[1, 2, 3, 4], levels_completed=lvl)
            agent.choose_action([f0, f_lvl], f_lvl)

        mem_end = tracemalloc.get_traced_memory()[0] / 1024.0
        tracemalloc.stop()

        delta = mem_end - mem_10
        print(f"\n  Rapid Level Transition Memory Delta (Level 250 vs Level 10): {delta:.2f} KB (Lvl 10: {mem_10:.2f} KB, Lvl 250: {mem_end:.2f} KB)")
        self.assertLess(delta, 10.0, f"Memory leaked across 240 level transitions: {delta:.2f} KB >= 10 KB")
        self.assertEqual(len(agent.affordance_ledger), 0)
        self.assertEqual(len(agent.memory.known_walls), 0)
        self.assertEqual(len(agent.fleet_plan_queue), 0)

    def test_deliberate_wall_collision_cascade_protection(self):
        """Stress test: Fleet plan queue deliberate collisions interspersed with heuristic moves."""
        agent = create_agent(game_id="deliberate_collision_test")
        # 10 consecutive deliberate wall collisions followed by orthogonal escape
        deliberate_bumps = [GameAction.ACTION1] * 10 + [GameAction.ACTION4]
        agent.fleet_plan_queue.extend(deliberate_bumps)

        grid = np.zeros((64, 64), dtype=np.int8)
        grid[0, 5] = 1  # Top wall collision
        f = FrameData(frame=[grid], state=GameState.NOT_FINISHED, available_actions=[1, 2, 3, 4], levels_completed=0)

        history = []
        for i in range(11):
            act = agent.choose_action([f] * (i + 1), f)
            history.append(act)

        # First 10 should be ACTION1 without being cleared
        self.assertEqual(history[:10], [GameAction.ACTION1] * 10)
        # 11th should be ACTION4
        self.assertEqual(history[10], GameAction.ACTION4)
        # Queue should be empty and bump counter cleared
        self.assertEqual(len(agent.fleet_plan_queue), 0)
        self.assertEqual(agent.memory.consecutive_bumps, 0)


if __name__ == "__main__":
    unittest.main()

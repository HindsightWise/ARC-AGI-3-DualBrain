"""Verification unit tests for M2 Executive Dispatch and Memory Hygiene."""
import unittest
from unittest.mock import MagicMock
import numpy as np
from arcengine import GameAction, GameState, FrameData
from agent.my_agent import MyAgent, WorkingMemory, DIRECTION_VECTORS

def create_mock_agent(game_id="test_dispatch"):
    return MyAgent(
        card_id="m2-test",
        game_id=game_id,
        agent_name="M2TestAgent",
        ROOT_URL="http://localhost",
        record=False,
        arc_env=MagicMock(),
    )

def make_test_frame(grid, state="NOT_FINISHED", levels_completed=0, available_actions=None):
    if available_actions is None:
        available_actions = [1, 2, 3, 4, 5, 6]
    return FrameData(
        frame=[grid.tolist()],
        state=state,
        levels_completed=levels_completed,
        available_actions=available_actions,
    )

class TestM2ExecutiveDispatch(unittest.TestCase):
    def test_queue_decoupling_and_deliberate_wall_collisions(self):
        """Verify deliberate wall collisions in fleet_plan_queue are never cleared by bump counters."""
        agent = create_mock_agent()
        # Deliberate bumps against top wall (ACTION1) followed by right move (ACTION4)
        deliberate_plan = [GameAction.ACTION1, GameAction.ACTION1, GameAction.ACTION4]
        agent.fleet_plan_queue.extend(deliberate_plan)

        grid = np.zeros((64, 64), dtype=int)
        grid[0, 5] = 1 # Avatar at top row (bump against top boundary)
        f = make_test_frame(grid)

        # Step 1: Deliberate bump 1
        act1 = agent.choose_action([f], f)
        self.assertEqual(act1, GameAction.ACTION1)
        self.assertEqual(len(agent.fleet_plan_queue), 2)

        # Step 2: Deliberate bump 2 (would trigger bump clear in heuristic queue)
        act2 = agent.choose_action([f, f], f)
        self.assertEqual(act2, GameAction.ACTION1)
        self.assertEqual(len(agent.fleet_plan_queue), 1)

        # Step 3: Directional exit
        act3 = agent.choose_action([f, f, f], f)
        self.assertEqual(act3, GameAction.ACTION4)
        self.assertEqual(len(agent.fleet_plan_queue), 0)
        # Verify bump counter reset on fleet queue completion
        self.assertEqual(agent.memory.consecutive_bumps, 0)

    def test_straight_line_march_protection(self):
        """Verify straight-line march >= 4 is NOT flagged as a loop."""
        memory = WorkingMemory()
        # March 10 steps in direction ACTION4
        for i in range(10):
            memory.record_step(GameAction.ACTION4, f"grid_{i}".encode(), (10, 10 + i))
            self.assertFalse(memory.is_in_loop())

    def test_2cycle_oscillation_detection(self):
        """Verify 2-cycle oscillations are caught across opposite, orthogonal, and contextual pairs."""
        # 1. Opposite pair: UP-DOWN
        mem1 = WorkingMemory()
        for act in [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION1, GameAction.ACTION2]:
            mem1.record_step(act, b"grid", (10, 10))
        self.assertTrue(mem1.is_in_loop())

        # 2. Orthogonal pair: UP-LEFT
        mem2 = WorkingMemory()
        for act in [GameAction.ACTION1, GameAction.ACTION3, GameAction.ACTION1, GameAction.ACTION3]:
            mem2.record_step(act, b"grid", (10, 10))
        self.assertTrue(mem2.is_in_loop())

        # 3. Contextual pair: UP-ACTION5
        mem3 = WorkingMemory()
        for act in [GameAction.ACTION1, GameAction.ACTION5, GameAction.ACTION1, GameAction.ACTION5]:
            mem3.record_step(act, b"grid", (10, 10))
        self.assertTrue(mem3.is_in_loop())

    def test_stagnation_counter_and_escape(self):
        """Verify staying at same position for 3 steps triggers loop and escapes to unvisited tile."""
        agent = create_mock_agent()
        grid = np.zeros((64, 64), dtype=int)
        grid[10, 10] = 1
        agent.memory.avatar_pos = (10, 10)
        agent.last_action = GameAction.ACTION1 # Last was UP

        # Mark right tile (10, 11) visited 5 times, left tile (10, 9) visited 0 times
        agent.memory.visited_positions[(10, 11)] = 5
        agent.memory.visited_positions[(10, 9)] = 0

        # Simulate stagnation: 3 steps at (10, 10)
        for _ in range(3):
            agent.memory.record_step(GameAction.ACTION1, b"grid", (10, 10))

        self.assertTrue(agent.memory.is_in_loop())
        escape = agent._get_orthogonal_escape(list(DIRECTION_VECTORS.keys()))
        # Should prioritize left tile (ACTION3) over right tile (ACTION4)
        self.assertEqual(escape, GameAction.ACTION3)

    def test_hermetic_level_transition_reset(self):
        """Verify all state containers are wiped clean across level transitions."""
        agent = create_mock_agent()
        grid = np.full((64, 64), 3, dtype=int)
        grid[10, 10] = 1
        f0 = make_test_frame(grid, levels_completed=0)
        agent.choose_action([f0], f0)

        # Mutate state on Level 0
        agent.memory.passable_map[20, 20] = False
        agent.memory.known_walls.add((20, 20))
        agent.affordance_ledger[(5, 5)] = "rot"
        agent.fleet_plan_queue.append(GameAction.ACTION1)
        agent._fleet_plan_started = True

        # Transition to Level 1
        f1 = make_test_frame(grid, levels_completed=1)
        agent.choose_action([f0, f1], f1)

        # Verify hermetic reset
        self.assertEqual(agent.last_levels_completed, 1)
        self.assertEqual(len(agent.memory.visited_states), 1) # Only contains frame f1 step
        self.assertEqual(len(agent.memory.known_walls), 0)
        self.assertTrue(agent.memory.passable_map[20, 20])
        self.assertEqual(len(agent.affordance_ledger), 0)
        self.assertEqual(len(agent.fleet_plan_queue), 0)
        self.assertFalse(agent._fleet_plan_started)

    def test_hermetic_terminal_reset(self):
        """Verify GAME_OVER triggers RESET and purges all state."""
        agent = create_mock_agent()
        agent.affordance_ledger[(5, 5)] = "rot"
        agent.fleet_plan_queue.append(GameAction.ACTION1)

        grid = np.zeros((64, 64), dtype=int)
        f_over = make_test_frame(grid, state="GAME_OVER", levels_completed=2)
        act = agent.choose_action([f_over], f_over)

        self.assertEqual(act, GameAction.RESET)
        self.assertEqual(len(agent.affordance_ledger), 0)
        self.assertEqual(len(agent.fleet_plan_queue), 0)
        self.assertEqual(agent.last_levels_completed, 0)
        self.assertIsNone(agent.last_grid)

if __name__ == "__main__":
    unittest.main()

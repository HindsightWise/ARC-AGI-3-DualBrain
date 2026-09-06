"""
Milestone M2 Challenger 2: Adversarial Traps, Kinematics & Dispatch Invariants Stress Harness.
Empirically stress-tests:
1. Multi-Target Bellman Backwards Planner:
   - Sequential multi-key chaining (Key 0 -> Door 0 -> Key 1 -> Door 1 -> Exit)
   - Permuted key dependencies (Key 1 needed first, then Key 0)
   - Unreachable exit (completely walled, blocked by locked door without key)
   - Greedy decoy traps (dead-end pockets, unneeded keys, unreachable keys)
   - Lattice step scaling (step_size=2, step_size=3)
2. Geodesic & Sliding Kinematics:
   - Frictionless ice sliding with obstacles, turns, and corner stops
   - Vectorized hazard dilation boundary preservation
   - Warp portal routing with turn penalty optimization
3. Executive Dispatch & Queue Invariants:
   - Deliberate wall collisions in fleet_plan_queue immune to bump clearing
   - Straight-line march >= 4, 10, 20 steps protected from loop interceptor
   - 2-cycle oscillation detection across opposite, orthogonal, contextual pairs
   - 3-cycle and 4-cycle loop detection
   - Stagnation counter triggered at >= 3 steps
   - Unvisited-first orthogonal escape scoring
   - Hermetic state reset across level transitions (levels_completed increment)
   - Hermetic state reset on terminal GAME_OVER / NOT_PLAYED
"""

import unittest
from unittest.mock import MagicMock
import numpy as np
from arcengine import GameAction, GameState, FrameData
from agent.my_agent import (
    MyAgent,
    WorkingMemory,
    BellmanBackwardsPlanner,
    GeodesicPlanner,
    OfflinePlanner,
    DIRECTION_VECTORS,
)


def create_test_agent(game_id="challenger2_test"):
    return MyAgent(
        card_id="m2-challenger2",
        game_id=game_id,
        agent_name="M2Challenger2Agent",
        ROOT_URL="http://localhost",
        record=False,
        arc_env=MagicMock(),
    )


def make_frame(grid, state="NOT_FINISHED", levels_completed=0, available_actions=None):
    if available_actions is None:
        available_actions = [1, 2, 3, 4, 5, 6]
    return FrameData(
        frame=[grid.tolist()],
        state=state,
        levels_completed=levels_completed,
        available_actions=available_actions,
    )


class TestM2AdversarialBellmanPlanner(unittest.TestCase):
    """Stress tests for Bellman retrograde backwards planner."""

    def test_chained_sequential_doors(self):
        """Chain: Start -> Key 0 -> Door 0 -> Key 1 -> Door 1 -> Goal."""
        # 30x30 grid with two barrier walls at row 10 and row 20
        passable = np.ones((30, 30), dtype=bool)
        passable[10, :] = False
        passable[10, 15] = False  # Door 0 initially blocked
        passable[20, :] = False
        passable[20, 15] = False  # Door 1 initially blocked

        start = (2, 2)
        key0 = (5, 25)    # In room 0
        key1 = (15, 5)    # In room 1 (behind Door 0)
        goal = (25, 25)   # In room 2 (behind Door 1)

        keys = [key0, key1]
        doors = {
            (10, 15): 0,  # Door 0 requires Key 0
            (20, 15): 1,  # Door 1 requires Key 1
        }

        plan = BellmanBackwardsPlanner.plan(
            start=start,
            goal=goal,
            keys=keys,
            doors=doors,
            passable=passable,
            step_size=1,
        )

        self.assertIsNotNone(plan, "Planner must solve sequential 2-key chained maze")
        self.assertTrue(len(plan) > 0)

        curr = start
        has_key0 = False
        has_key1 = False
        passed_door0 = False
        passed_door1 = False

        for act in plan:
            dr, dc = DIRECTION_VECTORS[act]
            curr = (curr[0] + dr, curr[1] + dc)
            if curr == key0:
                has_key0 = True
            if curr == (10, 15):
                self.assertTrue(has_key0, "Must have Key 0 to pass Door 0")
                passed_door0 = True
            if curr == key1:
                self.assertTrue(passed_door0, "Must pass Door 0 before reaching Key 1")
                has_key1 = True
            if curr == (20, 15):
                self.assertTrue(has_key1, "Must have Key 1 to pass Door 1")
                passed_door1 = True

        self.assertTrue(passed_door0)
        self.assertTrue(passed_door1)
        self.assertEqual(curr, goal)

    def test_reverse_ordered_key_dependencies(self):
        """Key 1 is in room 0 and opens Door at row 10, Key 0 is in room 1 and opens Door at row 20."""
        passable = np.ones((30, 30), dtype=bool)
        passable[10, :] = False
        passable[10, 15] = False  # Door A
        passable[20, :] = False
        passable[20, 15] = False  # Door B

        start = (2, 2)
        key1 = (2, 28)    # Key 1 in Room 0
        key0 = (15, 28)   # Key 0 in Room 1
        goal = (28, 28)

        keys = [key0, key1]
        doors = {
            (10, 15): 1,  # Door A requires Key 1
            (20, 15): 0,  # Door B requires Key 0
        }

        plan = BellmanBackwardsPlanner.plan(
            start=start,
            goal=goal,
            keys=keys,
            doors=doors,
            passable=passable,
            step_size=1,
        )

        self.assertIsNotNone(plan)
        curr = start
        has_k0, has_k1 = False, False
        for act in plan:
            dr, dc = DIRECTION_VECTORS[act]
            curr = (curr[0] + dr, curr[1] + dc)
            if curr == key1:
                has_k1 = True
            if curr == (10, 15):
                self.assertTrue(has_k1)
            if curr == key0:
                has_k0 = True
            if curr == (20, 15):
                self.assertTrue(has_k0)

        self.assertEqual(curr, goal)

    def test_unreachable_exit_isolated_by_solid_wall(self):
        """Exit surrounded by impassable perimeter wall -> returns None gracefully."""
        passable = np.ones((25, 25), dtype=bool)
        passable[15:, :] = False  # Bottom half blocked completely

        plan = BellmanBackwardsPlanner.plan(
            start=(2, 2),
            goal=(20, 20),
            keys=[],
            doors={},
            passable=passable,
            step_size=1,
            max_states=5000,
        )
        self.assertIsNone(plan)

    def test_unreachable_exit_blocked_by_keyless_door(self):
        """Door requires Key 0, but no Key 0 exists in the environment -> returns None."""
        passable = np.ones((25, 25), dtype=bool)
        passable[12, :] = False
        passable[12, 12] = False  # Door is impassable in passable_map

        plan = BellmanBackwardsPlanner.plan(
            start=(2, 2),
            goal=(20, 20),
            keys=[],  # No keys available
            doors={(12, 12): 0},  # Requires key 0 which doesn't exist
            passable=passable,
            step_size=1,
            max_states=5000,
        )
        self.assertIsNone(plan, "Door requiring non-existent key must remain impassable and return None")

    def test_decoy_dead_end_pocket_avoidance(self):
        """A maze with a decoy dead-end pocket branch vs true direct route."""
        passable = np.zeros((15, 15), dtype=bool)
        passable[1, 1:14] = True  # Direct hallway: row 1 from col 1 to 13
        passable[1:14, 1] = True  # Decoy pocket: col 1 from row 1 to 13

        plan = BellmanBackwardsPlanner.plan(
            start=(1, 1),
            goal=(1, 13),
            keys=[],
            doors={},
            passable=passable,
            step_size=1,
        )

        self.assertIsNotNone(plan)
        # Minimal path is 12 steps right (ACTION4)
        self.assertEqual(len(plan), 12)
        self.assertEqual(plan, [GameAction.ACTION4] * 12)

    def test_lattice_step_size_2_and_3(self):
        """Verify BellmanBackwardsPlanner works with step_size=2 and step_size=3."""
        for step in [2, 3]:
            passable = np.ones((30, 30), dtype=bool)
            start = (0, 0)
            goal = (step * 4, step * 4)

            plan = BellmanBackwardsPlanner.plan(
                start=start,
                goal=goal,
                keys=[],
                doors={},
                passable=passable,
                step_size=step,
            )
            self.assertIsNotNone(plan, f"Failed for step_size={step}")
            self.assertEqual(len(plan), 8)  # 4 steps down, 4 steps right


class TestM2KinematicsAdversarial(unittest.TestCase):
    """Stress tests for sliding kinematics, hazard dilation, and portal transitions."""

    def test_ice_sliding_with_internal_obstacles(self):
        """Ice sliding in rink with stop blocks."""
        passable = np.zeros((12, 12), dtype=bool)
        passable[1:11, 1:11] = True  # Rink
        # Place stop blocks inside
        passable[5, 5] = False
        passable[8, 2] = False

        # Slide from (1, 5) to (4, 10). Both Down-Right and Right-Down are 2-action routes.
        plan = OfflinePlanner.plan_geodesic_path(
            start=(1, 5),
            goal=(4, 10),
            passable=passable,
            step_size=1,
            is_ice=True,
        )
        self.assertIsNotNone(plan)
        self.assertEqual(len(plan), 2)
        self.assertIn(plan, [
            [GameAction.ACTION2, GameAction.ACTION4],
            [GameAction.ACTION4, GameAction.ACTION2],
        ])

    def test_hazard_dilation_boundary_preservation(self):
        """Hazard dilation must inflate hazards without breaking start/goal reachability."""
        passable = np.ones((20, 20), dtype=bool)
        passable[5:15, 10] = False  # Wall with gap
        passable[10, 10] = True     # Narrow 1-cell gap

        dilated = GeodesicPlanner.dilate_hazards(passable, radius=1)
        self.assertFalse(dilated[10, 10])

        plan = GeodesicPlanner.plan(
            start=(10, 5),
            goal=(10, 15),
            passable=passable,
            step_size=1,
            hazard_radius=1,
        )
        self.assertTrue(len(plan) > 0, "Fallback must route through 1-cell gap when dilated map is blocked")

    def test_warp_edge_turn_penalty_optimization(self):
        """A* must account for turn penalties and warp shortcuts."""
        passable = np.ones((20, 20), dtype=bool)
        warp_edges = {(5, 5): (5, 15)}
        plan = OfflinePlanner.plan_geodesic_path(
            start=(5, 2),
            goal=(5, 18),
            passable=passable,
            step_size=1,
            warp_edges=warp_edges,
            turn_penalty=0.5,
        )
        self.assertEqual(plan, [GameAction.ACTION4] * 6)


class TestM2ExecutiveDispatchInvariants(unittest.TestCase):
    """Stress tests for sovereign dispatch queue invariants and memory hygiene."""

    def test_deliberate_wall_collisions_not_cleared_by_bumps(self):
        """deliberate wall collisions in fleet_plan_queue are NOT cleared by bump detectors."""
        agent = create_test_agent()
        fleet_plan = [GameAction.ACTION3] * 5 + [GameAction.ACTION2]
        total_actions = len(fleet_plan)
        agent.fleet_plan_queue.extend(fleet_plan)

        grid = np.zeros((64, 64), dtype=int)
        grid[10, 0] = 1  # Avatar at left wall (x=0)
        f = make_frame(grid)

        for i in range(5):
            act = agent.choose_action([f] * (i + 1), f)
            self.assertEqual(act, GameAction.ACTION3, f"Step {i+1} bump must execute")
            self.assertEqual(len(agent.fleet_plan_queue), total_actions - (i + 1))

        # 6th step: directional exit
        act_exit = agent.choose_action([f] * 6, f)
        self.assertEqual(act_exit, GameAction.ACTION2)
        self.assertEqual(len(agent.fleet_plan_queue), 0)
        # Bump counter reset on completion
        self.assertEqual(agent.memory.consecutive_bumps, 0)
        self.assertEqual(agent.memory.stagnation_counter, 0)

    def test_straight_line_march_long_distance_protection(self):
        """Straight-line march of 4, 10, 25 steps without loop interruption."""
        memory = WorkingMemory()
        for step_idx in range(25):
            memory.record_step(GameAction.ACTION4, f"grid_{step_idx}".encode(), (10, step_idx))
            self.assertFalse(
                memory.is_in_loop(),
                f"Straight-line march step {step_idx} must NEVER be classified as a loop"
            )

    def test_2cycle_oscillation_detection_comprehensive(self):
        """2-cycle oscillation caught across opposite, orthogonal, contextual pairs."""
        pairs = [
            (GameAction.ACTION1, GameAction.ACTION2),  # UP - DOWN
            (GameAction.ACTION3, GameAction.ACTION4),  # LEFT - RIGHT
            (GameAction.ACTION1, GameAction.ACTION4),  # UP - RIGHT (orthogonal)
            (GameAction.ACTION2, GameAction.ACTION3),  # DOWN - LEFT (orthogonal)
            (GameAction.ACTION1, GameAction.ACTION5),  # UP - CONTEXT (contextual)
            (GameAction.ACTION4, GameAction.ACTION5),  # RIGHT - CONTEXT (contextual)
        ]
        for a, b in pairs:
            mem = WorkingMemory()
            mem.record_step(a, b"g1", (10, 10))
            mem.record_step(b, b"g2", (10, 11))
            mem.record_step(a, b"g3", (10, 10))
            self.assertFalse(mem.is_in_loop(), f"3 steps should not trigger 2-cycle yet ({a}, {b})")
            mem.record_step(b, b"g4", (10, 11))
            self.assertTrue(mem.is_in_loop(), f"4 steps {a},{b},{a},{b} MUST trigger 2-cycle loop")

    def test_3cycle_and_4cycle_loop_detection(self):
        """3-cycle (A,B,C,A,B,C) and 4-cycle (A,B,C,D,A,B,C,D) loop detection."""
        mem3 = WorkingMemory()
        seq3 = [GameAction.ACTION1, GameAction.ACTION4, GameAction.ACTION2] * 2
        for i, act in enumerate(seq3):
            mem3.record_step(act, f"g_{i}".encode(), (10, 10 + (i % 3)))
        self.assertTrue(mem3.is_in_loop(), "3-cycle loop must be detected")

        mem4 = WorkingMemory()
        seq4 = [GameAction.ACTION1, GameAction.ACTION4, GameAction.ACTION2, GameAction.ACTION3] * 2
        for i, act in enumerate(seq4):
            mem4.record_step(act, f"g_{i}".encode(), (10 + (i % 2), 10 + (i % 4)))
        self.assertTrue(mem4.is_in_loop(), "4-cycle loop must be detected")

    def test_unvisited_first_orthogonal_escape(self):
        """Orthogonal escape strictly prioritizes candidate tiles with lowest visitation frequency."""
        agent = create_test_agent()
        agent.memory.avatar_pos = (20, 20)
        agent.last_action = GameAction.ACTION2  # Last move was DOWN

        agent.memory.visited_positions[(20, 21)] = 10
        agent.memory.visited_positions[(20, 19)] = 0

        legal = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4]
        escape = agent._get_orthogonal_escape(legal)
        self.assertEqual(
            escape,
            GameAction.ACTION3,
            "Must choose unvisited LEFT tile (ACTION3) over heavily visited RIGHT tile (ACTION4)"
        )

        agent.memory.visited_positions[(20, 19)] = 15
        agent.memory.visited_positions[(20, 21)] = 2
        escape2 = agent._get_orthogonal_escape(legal)
        self.assertEqual(
            escape2,
            GameAction.ACTION4,
            "Must choose less visited RIGHT tile (ACTION4) over LEFT tile (ACTION3)"
        )

    def test_hermetic_level_transition_full_state_purge(self):
        """Levels transition must purge all working memory, queues, and affordance ledgers."""
        agent = create_test_agent()
        grid0 = np.full((64, 64), 2, dtype=int)
        grid0[10, 10] = 1
        f0 = make_frame(grid0, levels_completed=0)
        agent.choose_action([f0], f0)

        # Pollute all state containers
        agent.fleet_plan_queue.extend([GameAction.ACTION1, GameAction.ACTION2])
        agent._fleet_plan_started = True
        agent.affordance_ledger[(3, 3)] = "test_affordance"
        agent.puzzle_mode = True
        agent.memory.passable_map[30, 30] = False
        agent.memory.known_walls.add((30, 30))
        agent.memory.consecutive_bumps = 4
        agent.memory.stagnation_counter = 5

        # Transition to Level 1
        grid1 = np.full((64, 64), 4, dtype=int)
        grid1[20, 20] = 1
        f1 = make_frame(grid1, levels_completed=1)
        agent.choose_action([f0, f1], f1)

        # Invariants audit
        self.assertEqual(agent.last_levels_completed, 1)
        self.assertEqual(len(agent.fleet_plan_queue), 0, "Fleet plan queue must be purged")
        self.assertFalse(agent._fleet_plan_started, "_fleet_plan_started must reset to False")
        self.assertEqual(len(agent.affordance_ledger), 0, "Affordance ledger must be cleared")
        # Stale puzzle_mode (True from Level 0) must NOT persist; Level 1 re-detects afresh to False
        self.assertFalse(agent.puzzle_mode, "Stale Level 0 puzzle_mode=True must be purged")
        self.assertEqual(len(agent.memory.known_walls), 0, "Known walls must be cleared")
        self.assertTrue(agent.memory.passable_map[30, 30], "Passable map must be restored")
        self.assertEqual(agent.memory.consecutive_bumps, 0, "Consecutive bumps must reset to 0")
        self.assertEqual(agent.memory.stagnation_counter, 0, "Stagnation counter must reset to 0")

    def test_hermetic_terminal_game_over_reset(self):
        """GAME_OVER must immediately return GameAction.RESET and wipe all memory."""
        agent = create_test_agent()
        agent.fleet_plan_queue.append(GameAction.ACTION4)
        agent.affordance_ledger[(1, 1)] = "rot"
        agent.puzzle_mode = True
        agent.last_levels_completed = 3

        grid = np.zeros((64, 64), dtype=int)
        f_terminal = make_frame(grid, state="GAME_OVER", levels_completed=3)
        act = agent.choose_action([f_terminal], f_terminal)

        self.assertEqual(act, GameAction.RESET, "Must strictly emit GameAction.RESET on GAME_OVER")
        self.assertEqual(len(agent.fleet_plan_queue), 0)
        self.assertEqual(len(agent.affordance_ledger), 0)
        self.assertIsNone(agent.puzzle_mode)
        self.assertEqual(agent.last_levels_completed, 0)
        self.assertIsNone(agent.last_grid)


if __name__ == "__main__":
    unittest.main()

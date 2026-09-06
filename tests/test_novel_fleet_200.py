#!/usr/bin/env python3
"""
Unit and Regression Test Suite for 200 Novel ARC-AGI-3 Environments (ng01..ng200).
Verifies:
1. Complete registry integrity (exactly 200 games).
2. Clean instantiation, frame extraction (64x64), and action dispatch across all 7 paradigms.
3. Robust reset behavior and state safety.
"""
from __future__ import annotations

import os
import unittest
import numpy as np

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

from arcengine import ActionInput, GameAction, GameState
from scripts.novel_games import ALL_NOVEL_GAMES
import scripts.procedural_game_synthesizer as synth


class TestNovelFleet200(unittest.TestCase):
    def test_01_registry_count_and_keys(self):
        """Verify that exactly 200 unique novel games are registered."""
        self.assertEqual(len(ALL_NOVEL_GAMES), 200, f"Expected 200 games, found {len(ALL_NOVEL_GAMES)}")
        for i in range(1, 201):
            key = f"ng{i:02d}" if i < 100 else f"ng{i}"
            self.assertIn(key, ALL_NOVEL_GAMES, f"Missing key {key} in ALL_NOVEL_GAMES")

    def test_02_all_paradigms_representative_instantiation(self):
        """Test instantiation and basic step across each of the 7 Grand Paradigms."""
        # Pick 2 representative games from each paradigm
        sample_ids = [
            # P1: Navigation (1..50)
            "ng01", "ng26", "ng50",
            # P2: Physics (51..75)
            "ng51", "ng65", "ng75",
            # P3: Logic (76..100)
            "ng76", "ng85", "ng100",
            # P4: MultiAgent (101..125)
            "ng101", "ng115", "ng125",
            # P5: Affordance (126..150)
            "ng126", "ng135", "ng150",
            # P6: Dependencies (151..175)
            "ng151", "ng165", "ng175",
            # P7: Chaotic (176..200)
            "ng176", "ng185", "ng200",
        ]
        for gid in sample_ids:
            with self.subTest(game_id=gid):
                game = synth.generate_game(archetype=gid, seed=12345)
                # Reset
                obs = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                self.assertIsNotNone(obs, f"Failed reset for {gid}")
                frame_arr = np.array(obs.frame)
                self.assertEqual(frame_arr.shape, (1, 64, 64), f"Unexpected frame shape for {gid}: {frame_arr.shape}")

                # Step
                acts = obs.available_actions if hasattr(obs, "available_actions") and obs.available_actions else getattr(game, "_available_actions", [1, 2, 3, 4])
                act_id = GameAction.ACTION1 if 1 in acts else acts[0]
                next_obs = game.perform_action(ActionInput(id=act_id), raw=True)
                self.assertIsNotNone(next_obs, f"Failed step for {gid}")
                self.assertIn(next_obs.state, [GameState.NOT_FINISHED, GameState.WIN, GameState.GAME_OVER, "NOT_FINISHED", "WIN", "GAME_OVER"])

    def test_03_full_fleet_sweep_instantiation(self):
        """Fast sweep ensuring all 200 environments instantiate and render frame 0 cleanly."""
        for i in range(1, 201):
            gid = f"ng{i:02d}" if i < 100 else f"ng{i}"
            cls = ALL_NOVEL_GAMES[gid]
            game = cls(seed=42)
            obs = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            self.assertIsNotNone(obs)
            self.assertEqual(len(obs.frame), 1)
            self.assertEqual(len(obs.frame[0]), 64)
            self.assertEqual(len(obs.frame[0][0]), 64)

    def test_04_full_fleet_sweep_multistep(self):
        """Sweep verifying 3 sequential actions across all 200 environments with zero crashes."""
        for i in range(1, 201):
            gid = f"ng{i:02d}" if i < 100 else f"ng{i}"
            cls = ALL_NOVEL_GAMES[gid]
            game = cls(seed=100 + i)
            obs = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            for act in [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION4]:
                if obs.state in (GameState.WIN, GameState.GAME_OVER, "WIN", "GAME_OVER"):
                    break
                obs = game.perform_action(ActionInput(id=act), raw=True)
                self.assertIsNotNone(obs)
                self.assertIn(obs.state, [GameState.NOT_FINISHED, GameState.WIN, GameState.GAME_OVER, "NOT_FINISHED", "WIN", "GAME_OVER"])


if __name__ == "__main__":
    unittest.main()

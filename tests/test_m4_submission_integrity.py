"""Test harness for Milestone M4: Submission Integrity, Packaging & Offline Deployment.

Verifies:
1. Valid JSON format and 5-cell structure of notebooks/submission.ipynb.
2. Kernel metadata offline invariants (enable_internet: false, competition slug).
3. Valid Python 3.12 AST parsing and byte-compilation of the agent code in Cell 2.
4. Strict offline dependency boundary (no torch, cv2, scipy, matplotlib; only stdlib, numpy, arcengine, pydantic).
5. Zero hardcoded fleet plans (no VERIFIED_FLEET_PLANS).
6. Live execution of MyAgent extracted from Cell 2 on test frames.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import MagicMock

import numpy as np
from arcengine import FrameData, GameAction, GameState

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "submission.ipynb"
METADATA_PATH = ROOT / "notebooks" / "kernel-metadata.json"
AGENT_PATH = ROOT / "agent" / "my_agent.py"
VENDOR_PATH = ROOT / "vendor" / "ARC-AGI-3-Agents"

if str(VENDOR_PATH) not in sys.path:
    sys.path.insert(0, str(VENDOR_PATH))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TestM4SubmissionIntegrity(unittest.TestCase):
    """Integrity and offline packaging verification for ARC-AGI-3 submission."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        assert NOTEBOOK_PATH.exists(), f"Submission notebook not found at {NOTEBOOK_PATH}"
        assert METADATA_PATH.exists(), f"Kernel metadata not found at {METADATA_PATH}"
        assert AGENT_PATH.exists(), f"Agent source not found at {AGENT_PATH}"
        cls.notebook_data = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        cls.metadata_data = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

        # Extract agent code from Cell 2 (stripping %%writefile)
        cls.cells = cls.notebook_data.get("cells", [])
        assert len(cls.cells) >= 3, "Notebook must have at least 3 cells"
        cell_2_source = cls.cells[2].get("source", "")
        if isinstance(cell_2_source, list):
            cell_2_source = "".join(cell_2_source)
        code_lines = [
            line for line in cell_2_source.splitlines(keepends=True)
            if not line.strip().startswith("%%")
        ]
        cls.extracted_agent_code = "".join(code_lines)

    def test_notebook_json_structure(self):
        """Verify notebook is valid JSON with exactly 5 expected cells."""
        self.assertEqual(self.notebook_data.get("nbformat"), 4)
        self.assertEqual(len(self.cells), 5, f"Expected 5 cells, got {len(self.cells)}")

        # Cell 0: Markdown header
        self.assertEqual(self.cells[0]["cell_type"], "markdown")
        c0_text = "".join(self.cells[0]["source"]) if isinstance(self.cells[0]["source"], list) else self.cells[0]["source"]
        self.assertIn("ARC Prize 2026", c0_text)
        self.assertIn("ARC-AGI-3", c0_text)

        # Cell 1: Offline wheel installation
        self.assertEqual(self.cells[1]["cell_type"], "code")
        c1_text = "".join(self.cells[1]["source"]) if isinstance(self.cells[1]["source"], list) else self.cells[1]["source"]
        self.assertIn("!pip install --no-index", c1_text)
        self.assertIn("/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels", c1_text)
        self.assertIn("arc-agi", c1_text)

        # Cell 2: Agent file generation
        self.assertEqual(self.cells[2]["cell_type"], "code")
        c2_text = "".join(self.cells[2]["source"]) if isinstance(self.cells[2]["source"], list) else self.cells[2]["source"]
        self.assertTrue(
            c2_text.startswith("%%writefile /tmp/my_agent.py"),
            "Cell 2 must begin with '%%writefile /tmp/my_agent.py'",
        )
        self.assertIn("class MyAgent", c2_text)

        # Cell 3: Competition rerun runner
        self.assertEqual(self.cells[3]["cell_type"], "code")
        c3_text = "".join(self.cells[3]["source"]) if isinstance(self.cells[3]["source"], list) else self.cells[3]["source"]
        self.assertIn("KAGGLE_IS_COMPETITION_RERUN", c3_text)
        self.assertIn("http://gateway:8001/api/games", c3_text)
        self.assertIn("python main.py --agent myagent", c3_text)

        # Cell 4: Commit mode dummy parquet
        self.assertEqual(self.cells[4]["cell_type"], "code")
        c4_text = "".join(self.cells[4]["source"]) if isinstance(self.cells[4]["source"], list) else self.cells[4]["source"]
        self.assertIn("submission.to_parquet('/kaggle/working/submission.parquet'", c4_text)

    def test_kernel_metadata_offline_invariants(self):
        """Verify kernel-metadata.json enforces offline execution and correct competition slug."""
        self.assertFalse(
            self.metadata_data.get("enable_internet"),
            "enable_internet must be strictly False for air-gapped Kaggle evaluation",
        )
        competition_sources = self.metadata_data.get("competition_sources", [])
        self.assertIn(
            "arc-prize-2026-arc-agi-3",
            competition_sources,
            "competition_sources must contain 'arc-prize-2026-arc-agi-3'",
        )
        self.assertEqual(self.metadata_data.get("code_file"), "submission.ipynb")

    def test_notebook_metadata_offline_invariants(self):
        """Verify submission.ipynb internal metadata enforces isInternetEnabled: false."""
        kaggle_meta = self.notebook_data.get("metadata", {}).get("kaggle", {})
        self.assertFalse(
            kaggle_meta.get("isInternetEnabled"),
            "Notebook metadata 'isInternetEnabled' must be False",
        )

    def test_agent_ast_syntax_and_compilation(self):
        """Verify Cell 2 code parses cleanly with ast.parse and compiles under Python 3.12."""
        try:
            tree = ast.parse(self.extracted_agent_code, filename="submission_my_agent.py")
        except SyntaxError as err:
            self.fail(f"Cell 2 code failed ast.parse(): {err}")

        try:
            compiled = compile(self.extracted_agent_code, "submission_my_agent.py", "exec")
        except Exception as err:
            self.fail(f"Cell 2 code failed compile(): {err}")

        self.assertIsNotNone(compiled)

        # Verify class MyAgent definition exists in AST
        class_names = [
            node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
        ]
        self.assertIn("MyAgent", class_names, "AST must define class MyAgent")

        # Verify required methods
        my_agent_node = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name == "MyAgent"
        )
        method_names = [
            item.name for item in my_agent_node.body if isinstance(item, ast.FunctionDef)
        ]
        self.assertIn("choose_action", method_names)
        self.assertIn("is_done", method_names)

    def test_offline_dependency_invariants(self):
        """Verify zero forbidden imports (torch, cv2, scipy, matplotlib) and strict dependency whitelist."""
        tree = ast.parse(self.extracted_agent_code)
        forbidden = {
            "torch", "cv2", "scipy", "matplotlib", "sklearn",
            "transformers", "tensorflow", "jax", "torchvision", "torchaudio"
        }
        allowed_top_level = {
            "__future__", "collections", "dataclasses", "heapq", "hashlib",
            "json", "os", "pathlib", "random", "typing", "time", "math",
            "sys", "re", "itertools", "functools", "copy", "io", "string",
            "enum", "abc", "contextlib", "numpy", "arcengine", "agents",
            "arc_agi", "pydantic"
        }

        imported_modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top_mod = alias.name.split(".")[0]
                    imported_modules.add(top_mod)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    top_mod = node.module.split(".")[0]
                    imported_modules.add(top_mod)

        # Check for forbidden modules
        violations = imported_modules.intersection(forbidden)
        self.assertEqual(
            violations,
            set(),
            f"Strict offline violation: forbidden imports found: {violations}",
        )

        # Check all imports are in allowed list
        disallowed = imported_modules - allowed_top_level
        self.assertEqual(
            disallowed,
            set(),
            f"Unexpected external dependencies found: {disallowed}",
        )

    def test_no_hardcoded_fleet_plans(self):
        """Verify agent code contains zero hardcoded VERIFIED_FLEET_PLANS lookup tables."""
        self.assertNotIn(
            "VERIFIED_FLEET_PLANS",
            self.extracted_agent_code,
            "Agent source must not contain hardcoded VERIFIED_FLEET_PLANS",
        )
        self.assertNotIn(
            "VERIFIED_FLEET_PLANS",
            AGENT_PATH.read_text(encoding="utf-8"),
            "agent/my_agent.py must not contain hardcoded VERIFIED_FLEET_PLANS",
        )

    def test_extracted_my_agent_execution(self):
        """Test that MyAgent extracted from Cell 2 instantiates and executes choose_action without error."""
        # Execute extracted agent code in an isolated module
        mod = types.ModuleType("submission_agent_module")
        mod.__file__ = "/tmp/my_agent.py"
        try:
            exec(compile(self.extracted_agent_code, "/tmp/my_agent.py", "exec"), mod.__dict__)
        except Exception as err:
            self.fail(f"Failed executing extracted agent code in clean module: {err}")

        self.assertTrue(hasattr(mod, "MyAgent"), "Extracted module must export MyAgent")
        AgentClass = getattr(mod, "MyAgent")

        # Instantiate MyAgent
        mock_env = MagicMock()
        agent = AgentClass(
            card_id="m4-verification",
            game_id="m4_test",
            agent_name="M4Agent.test",
            ROOT_URL="http://localhost",
            record=False,
            arc_env=mock_env,
        )

        # 1. Normal frame: 64x64 grid with basic avatar and target
        grid = np.zeros((64, 64), dtype=np.int8)
        grid[10, 10] = 3  # avatar
        grid[20, 20] = 5  # goal
        frame_normal = FrameData(
            frame=[grid],
            state=GameState.NOT_FINISHED,
            available_actions=[1, 2, 3, 4, 5, 6],
            levels_completed=0,
        )

        action = agent.choose_action([frame_normal], frame_normal)
        self.assertIsInstance(action, GameAction, "choose_action must return a GameAction")
        self.assertFalse(agent.is_done([frame_normal], frame_normal))

        # 2. Terminal GAME_OVER frame: must immediately return RESET
        frame_game_over = FrameData(
            frame=[grid],
            state=GameState.GAME_OVER,
            available_actions=[0],
            levels_completed=0,
        )
        action_reset = agent.choose_action([frame_normal, frame_game_over], frame_game_over)
        self.assertEqual(action_reset, GameAction.RESET, "Must return GameAction.RESET on GAME_OVER")

        # 3. Terminal WIN frame: is_done must return True
        frame_win = FrameData(
            frame=[grid],
            state=GameState.WIN,
            available_actions=[0],
            levels_completed=1,
        )
        self.assertTrue(agent.is_done([frame_normal, frame_win], frame_win))

        # 4. Point-and-Click coordinate frame (ACTION6 only)
        grid_click = np.zeros((64, 64), dtype=np.int8)
        grid_click[15, 15] = 2
        grid_click[40, 40] = 4
        frame_click = FrameData(
            frame=[grid_click],
            state=GameState.NOT_FINISHED,
            available_actions=[6],
            levels_completed=0,
        )
        action_click = agent.choose_action([frame_click], frame_click)
        self.assertEqual(action_click, GameAction.ACTION6, "Must emit ACTION6 when only ACTION6 is available")
        coord_x = getattr(action_click.action_data, "x", None)
        coord_y = getattr(action_click.action_data, "y", None)
        if coord_x is None and hasattr(action_click, "reasoning") and isinstance(action_click.reasoning, dict):
            coord_x = action_click.reasoning.get("x")
            coord_y = action_click.reasoning.get("y")
        self.assertIsNotNone(coord_x, "ACTION6 must carry coordinate attribute x via action_data or reasoning")
        self.assertIsNotNone(coord_y, "ACTION6 must carry coordinate attribute y via action_data or reasoning")
        self.assertTrue(
            0 <= int(coord_x) <= 63 and 0 <= int(coord_y) <= 63,
            f"Coordinates must be in [0, 63], got ({coord_x}, {coord_y})",
        )

        # 5. Multi-step consecutive sequence without state exceptions
        for step in range(10):
            g = np.zeros((64, 64), dtype=np.int8)
            g[5 + step, 5 + step] = 3
            fr = FrameData(
                frame=[g],
                state=GameState.NOT_FINISHED,
                available_actions=[1, 2, 3, 4],
                levels_completed=0,
            )
            act = agent.choose_action([fr], fr)
            self.assertIn(act, [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4])


if __name__ == "__main__":
    unittest.main()

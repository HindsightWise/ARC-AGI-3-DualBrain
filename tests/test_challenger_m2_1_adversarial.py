"""Adversarial stress-test suite for Milestone 2 Challenger M2_1.

Empirically verifies:
1. Patch integrity in an isolated temporary sandbox (Cell 2 + post-patches), checking for rejects and AST corruptions.
2. Stability and uniqueness of `patch_spatial_helpers.py` anchor.
3. Stress testing `rot90`, `rot180`, `rot270`, `flipv`, `fliph`, `manhattan_distance`, `find_cells`, `find_objects`
   on boundary, extreme, and malformed inputs.
4. Quantitative prompt token impact (verifying 0 prompt tokens added to system prompt).
"""
import ast
import collections as _seg_collections
import hashlib
import json
from pathlib import Path
import py_compile
import subprocess
import sys
import tempfile
import types
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
DUCK_HARNESS_DIR = REPO_ROOT / "reference" / "duck-harness"
V25_NB_PATH = REPO_ROOT / "notebooks" / "submission_v25.ipynb"
SUB_NB_PATH = REPO_ROOT / "notebooks" / "submission.ipynb"


def get_v25_assets(c4_src: str) -> dict[str, str]:
    ns: dict = {}
    for line in c4_src.splitlines():
        if line.startswith("_v25_assets = {"):
            exec(line, ns)
            break
    return ns.get("_v25_assets", {})


class TestChallengerM2PatchIntegrity(unittest.TestCase):
    """Mission 1: Isolated Sandbox Patch Integrity & AST Audit."""

    @classmethod
    def setUpClass(cls):
        cls.nb_v25 = json.loads(V25_NB_PATH.read_text(encoding="utf-8"))
        c2 = cls.nb_v25["cells"][2]["source"]
        cls.c2_src = "".join(c2) if isinstance(c2, list) else c2
        cls.patch_text = cls.c2_src.split("\n", 1)[1] if cls.c2_src.startswith("%%writefile") else cls.c2_src
        if not cls.patch_text.endswith("\n"):
            cls.patch_text += "\n"

        c4 = cls.nb_v25["cells"][4]["source"]
        cls.c4_src = "".join(c4) if isinstance(c4, list) else c4
        cls.v25_assets = get_v25_assets(cls.c4_src)

    def test_01_sandbox_extraction_and_patch_application(self):
        """Simulate clean extraction, Cell 2 patch, post-patches, and check for .rej and ASTs."""
        with tempfile.TemporaryDirectory(prefix="m2_challenger_sandbox_") as tmp_dir:
            tmp = Path(tmp_dir)
            src_path = tmp / "src"
            src_path.mkdir(parents=True)

            # 1. Pristine extraction
            cmd = f"git archive HEAD ARC3-Inference | tar -x -C {src_path}"
            res = subprocess.run(cmd, shell=True, cwd=str(DUCK_HARNESS_DIR), capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"git archive failed: {res.stderr}")
            arc3 = src_path / "ARC3-Inference"
            self.assertTrue(arc3.is_dir())

            # 2. Cell 2 git apply
            patch_file = tmp / "harness-changes.patch"
            patch_file.write_text(self.patch_text, encoding="utf-8")
            apply_res = subprocess.run(
                ["git", "apply", "--include=ARC3-Inference/*", "-v", str(patch_file)],
                cwd=str(src_path),
                capture_output=True,
                text=True,
            )
            self.assertEqual(apply_res.returncode, 0, f"git apply failed:\n{apply_res.stderr}")

            # Verify 0 rejects or backup files
            rej_files = list(src_path.rglob("*.rej"))
            self.assertEqual(len(rej_files), 0, f"Rejects found: {rej_files}")
            orig_files = list(src_path.rglob("*.orig"))
            self.assertEqual(len(orig_files), 0, f"Orig files found: {orig_files}")

            # 3. Post-patches execution
            v25_dir = tmp / "arcagi_v25"
            v25_dir.mkdir()
            for name, content in self.v25_assets.items():
                (v25_dir / name).write_text(content, encoding="utf-8")

            for pscript in ("patch_animation_archive.py", "patch_animation_empty.py", "patch_spatial_helpers.py"):
                pres = subprocess.run([sys.executable, str(v25_dir / pscript), str(arc3)], capture_output=True, text=True)
                self.assertEqual(pres.returncode, 0, f"{pscript} failed:\n{pres.stderr}")

            sys.path.insert(0, str(v25_dir))
            import fable_layer_patch
            fl_res = fable_layer_patch.apply(str(arc3 / "inference/agent"), self.v25_assets["fable_layer_core.py"])
            self.assertIn(fl_res, ("applied", "already applied"))

            # 4. Audit all Python files for AST validity and compilation
            py_files = list(arc3.rglob("*.py"))
            self.assertGreaterEqual(len(py_files), 30, "Expected >= 30 Python files in ARC3-Inference")
            for pyf in py_files:
                code_content = pyf.read_text(encoding="utf-8")
                # Test AST parse
                try:
                    tree = ast.parse(code_content, filename=str(pyf))
                    self.assertIsNotNone(tree)
                except Exception as exc:
                    self.fail(f"ast.parse failed on {pyf}: {exc}")
                # Test bytecode compilation
                try:
                    py_compile.compile(str(pyf), doraise=True)
                except Exception as exc:
                    self.fail(f"py_compile failed on {pyf}: {exc}")

    def test_02_spatial_helpers_anchor_uniqueness(self):
        """Mission 2a: Anchor stability and uniqueness check in python_tool_sandbox.py."""
        with tempfile.TemporaryDirectory(prefix="m2_challenger_anchor_") as tmp_dir:
            tmp = Path(tmp_dir)
            src_path = tmp / "src"
            src_path.mkdir(parents=True)
            subprocess.run(f"git archive HEAD ARC3-Inference | tar -x -C {src_path}", shell=True, cwd=str(DUCK_HARNESS_DIR), check=True)
            arc3 = src_path / "ARC3-Inference"
            sb_path = arc3 / "inference/agent/python_tool_sandbox.py"
            anchor = '        runtime_globals["action"] = action\n'

            # Pristine check
            self.assertEqual(sb_path.read_text().count(anchor), 1)

            # After Cell 2 patch
            patch_file = tmp / "patch.patch"
            patch_file.write_text(self.patch_text, encoding="utf-8")
            subprocess.run(["git", "apply", "--include=ARC3-Inference/*", "-v", str(patch_file)], cwd=str(src_path), check=True)
            self.assertEqual(sb_path.read_text().count(anchor), 1)

            # After animation patches
            v25_dir = tmp / "v25"
            v25_dir.mkdir()
            for n, t in self.v25_assets.items():
                (v25_dir / n).write_text(t, encoding="utf-8")

            subprocess.run([sys.executable, str(v25_dir / "patch_animation_archive.py"), str(arc3)], check=True)
            self.assertEqual(sb_path.read_text().count(anchor), 1)
            subprocess.run([sys.executable, str(v25_dir / "patch_animation_empty.py"), str(arc3)], check=True)
            self.assertEqual(sb_path.read_text().count(anchor), 1)

            # Run patch_spatial_helpers.py
            subprocess.run([sys.executable, str(v25_dir / "patch_spatial_helpers.py"), str(arc3)], check=True)
            # After patch, anchor is still present exactly once because addition starts with anchor
            self.assertEqual(sb_path.read_text().count(anchor), 1)
            # And the injected loop is present
            sb_after = sb_path.read_text()
            self.assertIn('for _fn_name in ("rot90", "rot180", "rot270", "flipv", "fliph", "manhattan_distance", "find_cells", "find_objects"):', sb_after)


class TestSpatialHelpersAdversarialStress(unittest.TestCase):
    """Mission 2b: Boundary, edge case, and malformed input stress testing for spatial helpers."""

    @classmethod
    def setUpClass(cls):
        nb_v25 = json.loads(V25_NB_PATH.read_text(encoding="utf-8"))
        c4 = "".join(nb_v25["cells"][4]["source"])
        assets = get_v25_assets(c4)
        ps_src = assets["patch_spatial_helpers.py"]
        # Extract SEG_ADDENDUM
        ns: dict = {}
        exec(ps_src, ns)
        seg_addendum = ns["SEG_ADDENDUM"]
        seg_ns: dict = {}
        exec(seg_addendum, seg_ns)
        cls.rot90 = staticmethod(seg_ns["rot90"])
        cls.rot180 = staticmethod(seg_ns["rot180"])
        cls.rot270 = staticmethod(seg_ns["rot270"])
        cls.flipv = staticmethod(seg_ns["flipv"])
        cls.fliph = staticmethod(seg_ns["fliph"])
        cls.manhattan_distance = staticmethod(seg_ns["manhattan_distance"])
        cls.find_cells = staticmethod(seg_ns["find_cells"])
        cls.find_objects = staticmethod(seg_ns["find_objects"])

    def test_rotations_empty_and_trivial(self):
        """Test rot90/180/270 and flipv/fliph on empty and 1x1 grids."""
        # Empty grid []
        self.assertEqual(self.rot90([]), [])
        self.assertEqual(self.rot180([]), [])
        self.assertEqual(self.rot270([]), [])
        self.assertEqual(self.flipv([]), [])
        self.assertEqual(self.fliph([]), [])

        # 1x1 grid [[5]]
        g1 = [[5]]
        self.assertEqual(self.rot90(g1, 0), [[5]])
        self.assertEqual(self.rot90(g1, 1), [[5]])
        self.assertEqual(self.rot90(g1, 2), [[5]])
        self.assertEqual(self.rot90(g1, 3), [[5]])
        self.assertEqual(self.rot180(g1), [[5]])
        self.assertEqual(self.rot270(g1), [[5]])
        self.assertEqual(self.flipv(g1), [[5]])
        self.assertEqual(self.fliph(g1), [[5]])

    def test_rotations_rectangular_and_arbitrary_k(self):
        """Test rectangular grids (1x3, 3x1, 2x4) and extreme k values (negative, float, large)."""
        g_row = [[1, 2, 3]]
        self.assertEqual(self.rot90(g_row, 1), [[1], [2], [3]])
        self.assertEqual(self.rot90(g_row, 2), [[3, 2, 1]])
        self.assertEqual(self.rot90(g_row, 3), [[3], [2], [1]])
        self.assertEqual(self.rot90(g_row, 4), [[1, 2, 3]])
        # Negative k
        self.assertEqual(self.rot90(g_row, -1), [[3], [2], [1]])
        self.assertEqual(self.rot90(g_row, -2), [[3, 2, 1]])
        self.assertEqual(self.rot90(g_row, -3), [[1], [2], [3]])
        self.assertEqual(self.rot90(g_row, -4), [[1, 2, 3]])
        # Float / large k
        self.assertEqual(self.rot90(g_row, 101), [[1], [2], [3]])
        self.assertEqual(self.rot90(g_row, 2.0), [[3, 2, 1]])

        # Rectangular 2x3
        g23 = [[1, 2, 3], [4, 5, 6]]
        self.assertEqual(self.rot90(g23, 1), [[4, 1], [5, 2], [6, 3]])
        self.assertEqual(self.rot180(g23), [[6, 5, 4], [3, 2, 1]])
        self.assertEqual(self.rot270(g23), [[3, 6], [2, 5], [1, 4]])
        self.assertEqual(self.flipv(g23), [[4, 5, 6], [1, 2, 3]])
        self.assertEqual(self.fliph(g23), [[3, 2, 1], [6, 5, 4]])

    def test_grid_wrapped_objects(self):
        """Test when grid is passed as a FrameView or object with ._grid or .grid."""
        Wrap1 = types.SimpleNamespace(_grid=[[1, 2], [3, 4]])
        Wrap2 = types.SimpleNamespace(grid=[[1, 2], [3, 4]])
        self.assertEqual(self.rot90(Wrap1), [[3, 1], [4, 2]])
        self.assertEqual(self.rot90(Wrap2), [[3, 1], [4, 2]])
        self.assertEqual(self.flipv(Wrap1), [[3, 4], [1, 2]])
        self.assertEqual(self.fliph(Wrap2), [[2, 1], [4, 3]])

    def test_manhattan_distance(self):
        """Test manhattan_distance across identical, negative, and mixed coordinate types."""
        self.assertEqual(self.manhattan_distance((0, 0), (0, 0)), 0)
        self.assertEqual(self.manhattan_distance((3, 7), (1, 2)), abs(3 - 1) + abs(7 - 2))
        self.assertEqual(self.manhattan_distance((-5, -10), (5, 10)), 30)
        self.assertEqual(self.manhattan_distance([10, 20], (5, 15)), 10)

    def test_find_cells_boundary_and_types(self):
        """Test find_cells with integer targets, color char targets, and non-existent targets."""
        g = [
            [0, 1, 2],
            [1, 0, 1],
            [3, 4, 0],
        ]
        # By int
        self.assertEqual(self.find_cells(g, 0), [(0, 0), (1, 1), (2, 2)])
        self.assertEqual(self.find_cells(g, 1), [(0, 1), (1, 0), (1, 2)])
        # Target not in grid
        self.assertEqual(self.find_cells(g, 9), [])

        # Color chars: "WwgGcBMPRbSYOrNp" -> 0:W, 1:w, 2:g, 3:G, 4:c
        self.assertEqual(self.find_cells(g, "W"), [(0, 0), (1, 1), (2, 2)])
        self.assertEqual(self.find_cells(g, "w"), [(0, 1), (1, 0), (1, 2)])
        self.assertEqual(self.find_cells(g, "g"), [(0, 2)])

        # Grid containing char strings directly (searched by char)
        g_chars = [
            ["W", "w"],
            ["w", "W"],
        ]
        self.assertEqual(self.find_cells(g_chars, "W"), [(0, 0), (1, 1)])
        self.assertEqual(self.find_cells(g_chars, "w"), [(0, 1), (1, 0)])
        # Note: searching char grid with int target does not match because ch_map only maps char -> int
        self.assertEqual(self.find_cells(g_chars, 0), [])

        # Empty grid
        self.assertEqual(self.find_cells([], 0), [])
        self.assertEqual(self.find_cells([[]], 0), [])

    def test_find_objects_boundary_and_edge_cases(self):
        """Test find_objects on empty grids, single cells, background suppression, and medoids."""
        # Empty grids
        self.assertEqual(self.find_objects([]), [])
        self.assertEqual(self.find_objects([[]]), [])

        # Single cell [[0]] with ignore_bg=0 -> empty
        self.assertEqual(self.find_objects([[0]], ignore_bg=0), [])
        # Single cell [[1]] with ignore_bg=0 -> 1 object
        objs = self.find_objects([[1]], ignore_bg=0)
        self.assertEqual(len(objs), 1)
        self.assertEqual(objs[0]["color"], 1)
        self.assertEqual(objs[0]["area"], 1)
        self.assertEqual(objs[0]["centroid"], (0, 0))
        self.assertEqual(objs[0]["bbox"], [0, 0, 0, 0])

        # Large background suppression: total > 4 and area >= 0.8 * total
        # 3x3 grid with all color 1: total=9, area=9 >= 7.2 -> suppressed even if ignore_bg=0!
        g_all_1 = [[1] * 3 for _ in range(3)]
        self.assertEqual(self.find_objects(g_all_1, ignore_bg=0), [])

        # 3x3 grid with 7 of color 1, 2 of color 2 (total=9):
        # color 1 has area 7/9 = 77.7% < 80% -> kept!
        g_7_2 = [
            [1, 1, 1],
            [1, 1, 2],
            [1, 1, 2],
        ]
        objs = self.find_objects(g_7_2, ignore_bg=0)
        self.assertEqual(len(objs), 2)
        self.assertEqual(objs[0]["color"], 1)
        self.assertEqual(objs[0]["area"], 7)
        self.assertEqual(objs[1]["color"], 2)
        self.assertEqual(objs[1]["area"], 2)

        # Medoid projection check for concave/U-shaped object where center of mass is outside object
        # U-shape on 3x3:
        # 1 0 1
        # 1 0 1
        # 1 1 1
        # Cells for color 1: (0,0), (0,2), (1,0), (1,2), (2,0), (2,1), (2,2) -> 7 cells
        # Mean r = (0+0+1+1+2+2+2)/7 = 8/7 ~= 1.14 -> round = 1
        # Mean c = (0+2+0+2+0+1+2)/7 = 7/7 = 1.0 -> round = 1
        # Round centroid is (1, 1). BUT (1, 1) has value 0! Not in color 1!
        # The algorithm MUST project centroid onto an actual cell in color 1!
        g_u = [
            [1, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
        ]
        objs_u = self.find_objects(g_u, ignore_bg=0)
        self.assertEqual(len(objs_u), 1)
        u_obj = objs_u[0]
        self.assertIn(u_obj["centroid"], {(0, 0), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)})
        self.assertNotEqual(u_obj["centroid"], (1, 1), "Centroid must not be the empty hole!")

    def test_find_objects_disconnected_and_large_grids(self):
        """Test find_objects on disconnected components, out-of-range colors, and 64x64 grids."""
        # 4 disconnected corners of color 1
        g_corners = [
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 1],
        ]
        objs = self.find_objects(g_corners, ignore_bg=0)
        self.assertEqual(len(objs), 4)
        for obj in objs:
            self.assertEqual(obj["color"], 1)
            self.assertEqual(obj["area"], 1)

        # Out-of-range color index (> 15) falls back to str(col)
        g_oob = [[99]]
        objs_oob = self.find_objects(g_oob, ignore_bg=0)
        self.assertEqual(len(objs_oob), 1)
        self.assertEqual(objs_oob[0]["color"], 99)
        self.assertEqual(objs_oob[0]["char"], "99")

        # 64x64 full-scale ARC grid
        g_64 = [[0] * 64 for _ in range(64)]
        # Place a 3x3 block of color 5 at row 10, col 20
        for r in range(10, 13):
            for c in range(20, 23):
                g_64[r][c] = 5
        objs_64 = self.find_objects(g_64, ignore_bg=0)
        self.assertEqual(len(objs_64), 1)
        self.assertEqual(objs_64[0]["color"], 5)
        self.assertEqual(objs_64[0]["area"], 9)
        self.assertEqual(objs_64[0]["centroid"], (11, 21))
        self.assertEqual(objs_64[0]["bbox"], [10, 20, 12, 22])

        # Stress test rot90 on 64x64 grid
        r_64 = self.rot90(g_64, 1)
        self.assertEqual(len(r_64), 64)
        self.assertEqual(len(r_64[0]), 64)
        objs_r64 = self.find_objects(r_64, ignore_bg=0)
        self.assertEqual(len(objs_r64), 1)
        self.assertEqual(objs_r64[0]["color"], 5)
        self.assertEqual(objs_r64[0]["area"], 9)


class TestPromptTokenOverhead(unittest.TestCase):
    """Mission 3: Prove 0 prompt tokens added to system prompt by spatial helpers."""

    def test_zero_prompt_tokens_added(self):
        """Verify that spatial helpers do not alter system prompt or tool descriptions."""
        nb_v25 = json.loads(V25_NB_PATH.read_text(encoding="utf-8"))
        c4 = "".join(nb_v25["cells"][4]["source"])
        assets = get_v25_assets(c4)
        ps_src = assets["patch_spatial_helpers.py"]

        # Check that patch_spatial_helpers only touches segmentation.py and python_tool_sandbox.py
        self.assertNotIn("prompts.py", ps_src)
        self.assertNotIn("SYSTEM_PROMPT", ps_src)
        self.assertNotIn("system_prompt", ps_src)
        self.assertNotIn("SYSTEM", ps_src)

        # In python_tool_sandbox.py, check if prompt text or tool schemas mention rot90
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            src_path = tmp / "src"
            src_path.mkdir(parents=True)
            subprocess.run(f"git archive HEAD ARC3-Inference | tar -x -C {src_path}", shell=True, cwd=str(DUCK_HARNESS_DIR), check=True)
            arc3 = src_path / "ARC3-Inference"

            # Apply Cell 2
            c2 = "".join(nb_v25["cells"][2]["source"])
            p_text = c2.split("\n", 1)[1] if c2.startswith("%%writefile") else c2
            if not p_text.endswith("\n"):
                p_text += "\n"
            (tmp / "patch.patch").write_text(p_text, encoding="utf-8")
            subprocess.run(["git", "apply", "--include=ARC3-Inference/*", "-v", str(tmp / "patch.patch")], cwd=str(src_path), check=True)

            # Check prompt file before spatial helper patch
            prompts_before = (arc3 / "inference/agent/prompts.py").read_text()
            tool_agent_before = (arc3 / "inference/agent/tool_agent.py").read_text()

            # Run patch_spatial_helpers
            v25_dir = tmp / "v25"
            v25_dir.mkdir()
            (v25_dir / "patch_spatial_helpers.py").write_text(ps_src, encoding="utf-8")
            subprocess.run([sys.executable, str(v25_dir / "patch_spatial_helpers.py"), str(arc3)], check=True)

            prompts_after = (arc3 / "inference/agent/prompts.py").read_text()
            tool_agent_after = (arc3 / "inference/agent/tool_agent.py").read_text()

            # Confirm bit-for-bit identical prompt definitions
            self.assertEqual(prompts_before, prompts_after, "prompts.py was unexpectedly modified!")
            self.assertEqual(tool_agent_before, tool_agent_after, "tool_agent.py was unexpectedly modified!")

            # Confirm rot90/find_objects do not appear anywhere in prompts.py
            for fn in ("rot90", "rot180", "rot270", "flipv", "fliph", "manhattan_distance", "find_cells", "find_objects"):
                self.assertNotIn(fn, prompts_after)


if __name__ == "__main__":
    unittest.main()

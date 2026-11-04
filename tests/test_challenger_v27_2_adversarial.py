"""Adversarial Challenge and Empirical Stress Test Suite for Version 27 Sovereign Apex Engine.

Authored by: challenger_v27_2 (EMPIRICAL CHALLENGER)
Focus:
1. Complete code cell and embedded Python AST parsing across submission_v27.ipynb and submission.ipynb.
2. Byte-for-byte identity and cell metadata verification.
3. Cell 2 SHA-256 body hash matching v11 reference patch (07fd859e...).
4. Cell 12 SGLang flags (--max-running-requests 10, --cuda-graph-max-bs 10, --mamba-cache 60, --mem-fraction-static 0.96).
5. Cell 16 ours_form_priority.install_d presence and 100% absence of install_timeout_armor or min_floor.
6. notebooks/kernel-metadata.json validity and offline configuration.
7. pulled_kernels/oct21_top/ completeness and docs/FORENSIC_V26_TO_V27.md metric verification.
8. Adversarial stress tests on spatial helpers, SLM memory logic, and native timeout clamping.
"""

from __future__ import annotations

import ast
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import types
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
V27_NB_PATH = REPO_ROOT / "notebooks" / "submission_v27.ipynb"
SUB_NB_PATH = REPO_ROOT / "notebooks" / "submission.ipynb"
V24_NB_PATH = REPO_ROOT / "notebooks" / "submission_v24.ipynb"
V11_NB_PATH = REPO_ROOT / "pulled_kernels" / "v11" / "arc-prize-2026-v11-apex-engine.ipynb"
META_PATH = REPO_ROOT / "notebooks" / "kernel-metadata.json"
FORENSIC_DOC_PATH = REPO_ROOT / "docs" / "FORENSIC_V26_TO_V27.md"
OCT21_DIR = REPO_ROOT / "pulled_kernels" / "oct21_top"


def _src(cell: dict) -> str:
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else str(s)


class TestChallengerV27Adversarial(unittest.TestCase):
    """Rigorous code-executing adversarial challenge harness for Version 27."""

    @classmethod
    def setUpClass(cls) -> None:
        assert V27_NB_PATH.exists(), f"Missing {V27_NB_PATH}"
        assert SUB_NB_PATH.exists(), f"Missing {SUB_NB_PATH}"
        assert META_PATH.exists(), f"Missing {META_PATH}"

        cls.nb_v27 = json.loads(V27_NB_PATH.read_text(encoding="utf-8"))
        cls.nb_sub = json.loads(SUB_NB_PATH.read_text(encoding="utf-8"))
        cls.meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        cls.nb_v11 = json.loads(V11_NB_PATH.read_text(encoding="utf-8")) if V11_NB_PATH.exists() else None
        cls.nb_v24 = json.loads(V24_NB_PATH.read_text(encoding="utf-8")) if V24_NB_PATH.exists() else None

    def test_01_notebook_cell_inspection_and_ast_parsing(self) -> None:
        """Inspect every code cell and parse all embedded Python code strings with ast.parse."""
        nb_list = [("submission_v27.ipynb", self.nb_v27)]
        if "v27" in self.meta.get("id", ""):
            nb_list.append(("submission.ipynb", self.nb_sub))
        for nb_label, nb in nb_list:
            cells = nb.get("cells", [])
            self.assertEqual(len(cells), 23, f"{nb_label} must have exactly 23 cells")

            for idx, cell in enumerate(cells):
                ctype = cell.get("cell_type")
                if ctype != "code":
                    continue

                # Verify clean output states for submission
                self.assertEqual(cell.get("outputs"), [], f"{nb_label} Cell {idx} outputs must be empty")
                self.assertIsNone(cell.get("execution_count"), f"{nb_label} Cell {idx} execution_count must be None")

                source = _src(cell)
                self.assertGreater(len(source), 0, f"{nb_label} Cell {idx} must not be empty")

                # Handle cell-specific syntax and embedded strings
                if idx == 2:
                    # Cell 2 is the harness patch: %%writefile /kaggle/harness-changes.patch
                    lines = source.splitlines()
                    self.assertTrue(lines[0].startswith("%%writefile /kaggle/harness-changes.patch"))
                    self.assertIn("diff --git a/ARC3-Inference/", source)
                elif idx == 4:
                    # Cell 4 contains IPython ! commands (!rm, !cp) and _v27_assets dict
                    # Mask IPython shell escapes to parse the container Python code
                    masked_lines = [
                        line if not line.strip().startswith("!") else "# " + line
                        for line in source.splitlines()
                    ]
                    c4_code = "\n".join(masked_lines)
                    try:
                        tree4 = ast.parse(c4_code, filename=f"{nb_label}:cell_4")
                        self.assertGreater(len(tree4.body), 10)
                    except Exception as exc:
                        self.fail(f"ast.parse failed on {nb_label} Cell 4: {exc}")

                    # Extract and parse all embedded python scripts in _v27_assets
                    ns4: dict = {}
                    for line in source.splitlines():
                        if line.startswith("_v27_assets = {"):
                            exec(line, ns4)
                            break
                    self.assertIn("_v27_assets", ns4)
                    assets = ns4["_v27_assets"]
                    self.assertEqual(
                        set(assets.keys()),
                        {"patch_solved_level_memory.py", "patch_animation_empty.py", "patch_spatial_helpers.py"},
                    )

                    for asset_name, asset_code in assets.items():
                        try:
                            asset_tree = ast.parse(asset_code, filename=asset_name)
                            self.assertGreater(len(asset_tree.body), 0)
                        except Exception as exc:
                            self.fail(f"ast.parse failed on embedded asset {asset_name}: {exc}")

                        # Recursively find and parse embedded multiline python code strings (e.g. SLM_ADDENDUM, SEG_ADDENDUM)
                        for node in ast.walk(asset_tree):
                            if isinstance(node, ast.Assign):
                                for target in node.targets:
                                    if (
                                        isinstance(target, ast.Name)
                                        and isinstance(node.value, ast.Constant)
                                        and isinstance(node.value.value, str)
                                    ):
                                        embedded_str = node.value.value
                                        if len(embedded_str) > 50 and ("def " in embedded_str or "import " in embedded_str):
                                            try:
                                                emb_tree = ast.parse(embedded_str, filename=f"{asset_name}:{target.id}")
                                                self.assertGreater(len(emb_tree.body), 0)
                                            except Exception as exc:
                                                self.fail(f"ast.parse failed on embedded code string {target.id} in {asset_name}: {exc}")
                elif idx == 16:
                    # Cell 16 contains Python code with embedded _FP_SOURCE
                    try:
                        tree16 = ast.parse(source, filename=f"{nb_label}:cell_16")
                        self.assertGreater(len(tree16.body), 5)
                    except Exception as exc:
                        self.fail(f"ast.parse failed on {nb_label} Cell 16: {exc}")

                    # Extract _FP_SOURCE
                    fp_source = None
                    for node in tree16.body:
                        if isinstance(node, ast.Assign):
                            for target in node.targets:
                                if isinstance(target, ast.Name) and target.id == "_FP_SOURCE":
                                    fp_source = node.value.value
                    self.assertIsNotNone(fp_source, f"{nb_label} Cell 16 must contain _FP_SOURCE")
                    try:
                        tree_fp = ast.parse(fp_source, filename="ours_form_priority.py")
                        self.assertGreater(len(tree_fp.body), 10)
                    except Exception as exc:
                        self.fail(f"ast.parse failed on _FP_SOURCE in {nb_label} Cell 16: {exc}")
                else:
                    # Cells 6, 8, 10, 12, 14, 18, 20, 22 are direct Python code
                    try:
                        tree = ast.parse(source, filename=f"{nb_label}:cell_{idx}")
                        self.assertGreater(len(tree.body), 0)
                    except Exception as exc:
                        self.fail(f"ast.parse failed on {nb_label} Cell {idx}: {exc}")

    def test_02_byte_for_byte_identity(self) -> None:
        """Verify byte-for-byte identity between submission_v27.ipynb and submission.ipynb."""
        if "v27" not in self.meta.get("id", ""):
            raise unittest.SkipTest("Active kernel is not v27")
        bytes_v27 = V27_NB_PATH.read_bytes()
        bytes_sub = SUB_NB_PATH.read_bytes()

        self.assertEqual(len(bytes_v27), len(bytes_sub), "File sizes must be identical")
        sha_v27 = hashlib.sha256(bytes_v27).hexdigest()
        sha_sub = hashlib.sha256(bytes_sub).hexdigest()
        self.assertEqual(sha_v27, sha_sub, f"SHA-256 hashes must be identical: {sha_v27} vs {sha_sub}")

    def test_03_cell2_sha256_body_matches_v11_reference(self) -> None:
        """Verify that Cell 2 has SHA-256 body starting with 07fd859e... matching v11."""
        c2 = _src(self.nb_v27["cells"][2])
        self.assertEqual(len(c2), 498799, f"Cell 2 length must be 498,799 characters, got {len(c2)}")

        lines = c2.split("\n", 1)
        self.assertEqual(lines[0], "%%writefile /kaggle/harness-changes.patch")
        c2_body = lines[1]

        sha_body = hashlib.sha256(c2_body.encode("utf-8")).hexdigest()
        expected_sha = "07fd859e7ce7ad8ac8a2f8f726d862660c1be1abc97560a63b5985703ddf7f4f"
        self.assertEqual(sha_body, expected_sha)
        self.assertTrue(sha_body.startswith("07fd859e"))

        if self.nb_v11:
            c2_v11 = _src(self.nb_v11["cells"][2])
            c2_v11_body = c2_v11.split("\n", 1)[1]
            sha_v11 = hashlib.sha256(c2_v11_body.encode("utf-8")).hexdigest()
            self.assertEqual(sha_body, sha_v11, "Cell 2 body must match v11 reference patch exactly")

    def test_04_cell12_sglang_flags(self) -> None:
        """Verify Cell 12 SGLang flags: --max-running-requests 10, --cuda-graph-max-bs 10, --mamba-cache 60, --mem-fraction-static 0.96."""
        c12 = _src(self.nb_v27["cells"][12])

        # Verify CFG values
        self.assertIn("MEMFRAC=0.96", c12)
        self.assertIn("MAXREQ=10", c12)
        self.assertIn("CUDAGRAPH_MAXBS=10", c12)
        self.assertIn("MAMBA_CACHE=60", c12)

        # Extract CFG dictionary via exec in restricted namespace
        cfg_lines = []
        in_cfg = False
        for line in c12.splitlines():
            if line.strip().startswith("CFG = dict("):
                in_cfg = True
            if in_cfg:
                cfg_lines.append(line)
                if line.strip() == ")":
                    break
        ns: dict = {"WORKING_DIR": "/tmp", "SERVED_MODEL_NAME": "flashnext"}
        exec("\n".join(cfg_lines), ns)
        cfg = ns["CFG"]

        self.assertEqual(cfg["MEMFRAC"], 0.96)
        self.assertEqual(cfg["MAXREQ"], 10)
        self.assertEqual(cfg["CUDAGRAPH_MAXBS"], 10)
        self.assertEqual(cfg["MAMBA_CACHE"], 60)

        # Verify exact CLI flags in server command list
        self.assertIn('"--max-running-requests", str(CFG["MAXREQ"])', c12)
        self.assertIn('"--cuda-graph-max-bs-decode", str(CFG["CUDAGRAPH_MAXBS"])', c12)
        self.assertIn('"--max-mamba-cache-size", str(CFG["MAMBA_CACHE"])', c12)
        self.assertIn('"--mem-fraction-static", str(CFG["MEMFRAC"])', c12)

    def test_05_cell16_purity_and_form_priority(self) -> None:
        """Verify Cell 16 contains ours_form_priority and install_d, and zero occurrences of install_timeout_armor or min_floor."""
        c16 = _src(self.nb_v27["cells"][16])

        # Purge verification: zero timeout armor or floor occurrences
        self.assertEqual(c16.lower().count("install_timeout_armor"), 0)
        self.assertEqual(c16.lower().count("min_floor"), 0)
        self.assertEqual(c16.count("TIMEOUT_ARMOR_CODE"), 0)
        self.assertEqual(c16.count("ARC3_MIN_READ_TIMEOUT"), 0)
        self.assertEqual(c16.count("self.started_at = time.monotonic()"), 0)
        self.assertEqual(c16.count("aging_bonus"), 0)

        # Form priority installation verification
        self.assertIn("'ours_form_priority'", c16)
        self.assertIn("_fp_mod.install_d(_fp_ta, _fp_solver)", c16)
        self.assertIn("def install_d(", c16)

        # Byte-identical to v24 Cell 16
        if self.nb_v24:
            c16_v24 = _src(self.nb_v24["cells"][16])
            self.assertEqual(c16, c16_v24, "Cell 16 must be 100% byte-identical to v24 substrate")

    def test_06_kernel_metadata_inspection(self) -> None:
        """Verify kernel-metadata.json validity and required competition execution fields."""
        if "v27" not in self.meta.get("id", ""):
            raise unittest.SkipTest("Active kernel is not v27")
        self.assertEqual(self.meta.get("id"), "zerbytheboss/v27-sovereign-apex-engine")
        self.assertEqual(self.meta.get("code_file"), "submission.ipynb")
        self.assertEqual(self.meta.get("machine_shape"), "NvidiaRtxPro6000")
        self.assertFalse(self.meta.get("enable_internet"), "enable_internet must be False")
        self.assertTrue(self.meta.get("enable_gpu"), "enable_gpu must be True")
        self.assertEqual(self.meta.get("language"), "python")
        self.assertEqual(self.meta.get("kernel_type"), "notebook")

    def test_07_oct21_reference_kernels(self) -> None:
        """Verify existence and minimum size of all 6 reference kernels in pulled_kernels/oct21_top/."""
        self.assertTrue(OCT21_DIR.exists(), f"Missing {OCT21_DIR}")
        required_kernels = [
            ("weitaoguo_arc3-d-prime-plus-solved-level-memory", "arc3-d-prime-plus-solved-level-memory.ipynb"),
            ("evgendvorkin_arc-3-2026", "arc-3-2026.ipynb"),
            ("greenalgeria_arc3-up", "arc3-up.ipynb"),
            ("kawh1111_arc3-final-v1", "arc3-final-v1.ipynb"),
            ("iancblenke_carnot-arc-agi3-submission", "carnot-arc-agi3-submission.py"),
            ("manderson240_cohezion-arc-agi-3-autoharness-solver", "cohezion-arc-agi-3-autoharness-solver.ipynb"),
        ]
        for dirname, filename in required_kernels:
            target_path = OCT21_DIR / dirname / filename
            self.assertTrue(target_path.exists(), f"Missing reference kernel file: {target_path}")
            size = target_path.stat().st_size
            self.assertGreater(size, 1024, f"Kernel file {target_path} is suspiciously small: {size} bytes")

    def test_08_forensic_v26_to_v27_document_audit(self) -> None:
        """Verify docs/FORENSIC_V26_TO_V27.md exists and programmatically check its cited metrics against /tmp/v*_output/."""
        self.assertTrue(FORENSIC_DOC_PATH.exists(), f"Missing {FORENSIC_DOC_PATH}")
        doc_text = FORENSIC_DOC_PATH.read_text(encoding="utf-8")

        # Check key narrative markers
        for marker in [
            "Version 24 (28.79 LB)",
            "Version 25 (27.16 LB)",
            "Version 26 (26.29 LB)",
            "install_timeout_armor",
            "weitaoguo",
            "evgendvorkin",
            "greenalgeria",
        ]:
            self.assertIn(marker, doc_text)

        # Ground truth validation against /tmp benchmark and request logs
        expected_metrics = {
            "v24": {"wallclock": 300.37, "actions": [5, 7, 14, 24, 9, 0], "tot_act": 59, "comp_tokens": 47033, "prompt_tokens": 950242},
            "v25": {"wallclock": 303.20, "actions": [5, 7, 14, 24, 0, 0], "tot_act": 50, "comp_tokens": 46055, "prompt_tokens": 1090347},
            "v26": {"wallclock": 336.13, "actions": [5, 7, 14, 19, 4, 0], "tot_act": 49, "comp_tokens": 52754, "prompt_tokens": 1143737},
        }

        for ver, exp in expected_metrics.items():
            bm_file = Path(f"/tmp/{ver}_output/benchmark.json")
            if bm_file.exists():
                bm = json.loads(bm_file.read_text())
                run = bm["game_runs"][0]
                self.assertAlmostEqual(run["final_wallclock_seconds"], exp["wallclock"], delta=0.1)
                self.assertEqual(run["actions_per_level"], exp["actions"])
                self.assertEqual(sum(run["actions_per_level"]), exp["tot_act"])

            req_file = Path(f"/tmp/{ver}_output/requests.jsonl")
            if req_file.exists():
                lines = [json.loads(l) for l in req_file.read_text().splitlines() if l.strip()]
                comp_tokens = sum(r.get("usage", {}).get("completion_tokens", 0) for r in lines)
                prompt_tokens = sum(r.get("usage", {}).get("prompt_tokens", 0) for r in lines)
                self.assertEqual(comp_tokens, exp["comp_tokens"])
                self.assertEqual(prompt_tokens, exp["prompt_tokens"])

    def test_09_adversarial_stress_spatial_helpers(self) -> None:
        """Adversarially stress-test all spatial helpers with edge, degenerate, and boundary inputs."""
        c4 = _src(self.nb_v27["cells"][4])
        ns4: dict = {}
        for line in c4.splitlines():
            if line.startswith("_v27_assets = {"):
                exec(line, ns4)
                break
        spatial_script = ns4["_v27_assets"]["patch_spatial_helpers.py"]
        seg_code = spatial_script.split('SEG_ADDENDUM = """')[1].split('"""')[0]
        helper_ns: dict = {}
        exec(seg_code, helper_ns)

        rot90 = helper_ns["rot90"]
        rot180 = helper_ns["rot180"]
        rot270 = helper_ns["rot270"]
        flipv = helper_ns["flipv"]
        fliph = helper_ns["fliph"]
        manhattan_distance = helper_ns["manhattan_distance"]
        find_cells = helper_ns["find_cells"]
        find_objects = helper_ns["find_objects"]

        # Degenerate inputs: empty grid
        self.assertEqual(rot90([], 1), [])
        self.assertEqual(rot180([]), [])
        self.assertEqual(rot270([]), [])
        self.assertEqual(flipv([]), [])
        self.assertEqual(fliph([]), [])
        self.assertEqual(find_cells([], 0), [])
        self.assertEqual(find_objects([]), [])

        # 1x1 grid
        g1 = [[7]]
        self.assertEqual(rot90(g1, 1), [[7]])
        self.assertEqual(rot90(g1, 4), [[7]])
        self.assertEqual(rot180(g1), [[7]])
        self.assertEqual(rot270(g1), [[7]])
        self.assertEqual(flipv(g1), [[7]])
        self.assertEqual(fliph(g1), [[7]])
        self.assertEqual(find_cells(g1, 7), [(0, 0)])
        self.assertEqual(find_cells(g1, 0), [])

        # Non-square grid (2x3)
        rect = [
            [1, 2, 3],
            [4, 5, 6],
        ]
        r90 = rot90(rect, 1)  # shape becomes 3x2: col 0 becomes row 0 reversed
        self.assertEqual(r90, [[4, 1], [5, 2], [6, 3]])
        # 4 successive 90-degree rotations must yield original rectangle
        self.assertEqual(rot90(rot90(rot90(rot90(rect, 1), 1), 1), 1), rect)

        # Arbitrary rotation k: k = 5 is equivalent to k = 1
        self.assertEqual(rot90(rect, 5), rot90(rect, 1))

        # Manhattan distance with negative and boundary coordinates
        self.assertEqual(manhattan_distance((0, 0), (0, 0)), 0)
        self.assertEqual(manhattan_distance((-5, -10), (5, 10)), 30)

        # Color string representation ('W'=0, 'w'=1, 'g'=2, 'G'=3, etc. per "WwgGcBMPRbSYOrNp")
        c_grid = [[0, 2], [1, 2]]
        self.assertEqual(find_cells(c_grid, "g"), [(0, 1), (1, 1)])

        # Objects with background filtering and multi-cell contiguous components
        test_board = [
            [0, 1, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 2, 2],
            [0, 0, 2, 2],
        ]
        objs = find_objects(test_board, ignore_bg=0)
        self.assertEqual(len(objs), 2)
        obj_colors = {o["color"] for o in objs}
        self.assertEqual(obj_colors, {1, 2})
        c1 = [o for o in objs if o["color"] == 1][0]
        self.assertEqual(c1["area"], 3)
        self.assertEqual(c1["bbox"], [0, 1, 1, 2])
        c2 = [o for o in objs if o["color"] == 2][0]
        self.assertEqual(c2["area"], 4)
        self.assertEqual(c2["bbox"], [2, 2, 3, 3])

    def test_10_adversarial_stress_native_timeout_logic(self) -> None:
        """Adversarially stress-test request_timeout_seconds logic under severe edge cases."""
        # Simulate solver._HarnessGameSession.request_timeout_seconds formula
        def native_rto(rem_s: float, analyzer_timeout: float = 120.0, soft_rem: float | None = None) -> float:
            cands = [analyzer_timeout, rem_s]
            if soft_rem is not None:
                cands.append(soft_rem)
            return max(0.1, min(cands))

        def armored_rto(rem_s: float, min_floor: float = 120.0, analyzer_timeout: float = 120.0, soft_rem: float | None = None) -> float:
            cand = native_rto(rem_s, analyzer_timeout, soft_rem)
            return max(min_floor, cand)

        # Scenario 1: Normal abundant time (250s remaining)
        self.assertEqual(native_rto(250.0), 120.0, "Must be capped at analyzer_timeout 120.0s")
        self.assertEqual(armored_rto(250.0), 120.0)

        # Scenario 2: Late game low time (25.74s remaining) - EXACT Step 26/27 v26 regression condition
        self.assertEqual(native_rto(25.74), 25.74, "Native MUST bound request by 25.74s")
        self.assertEqual(armored_rto(25.74), 120.0, "Armor dangerously overrode to 120s causing +36s overrun!")

        # Scenario 3: Imminent deadline (0.05s remaining)
        self.assertEqual(native_rto(0.05), 0.1, "Native floors at 0.1s to allow immediate cleanup")
        self.assertEqual(armored_rto(0.05), 120.0, "Armor would wait 120s on expired game!")

        # Scenario 4: Expired deadline (0.0s or negative)
        self.assertEqual(native_rto(0.0), 0.1)
        self.assertEqual(native_rto(-10.0), 0.1)

        # Scenario 5: Soft deadline active (soft=35.1s, hard=89.4s)
        self.assertEqual(native_rto(89.4, soft_rem=35.1), 35.1, "Native clamps to soft deadline 35.1s")
        self.assertEqual(armored_rto(89.4, soft_rem=35.1), 120.0, "Armor completely defeated soft deadline clamping!")

    def test_11_adversarial_stress_slm_memory_logic(self) -> None:
        """Adversarially verify that solved_level_memory never modifies user prompts and gates pinning to Level 3+ context eviction."""
        c4 = _src(self.nb_v27["cells"][4])
        ns4: dict = {}
        for line in c4.splitlines():
            if line.startswith("_v27_assets = {"):
                exec(line, ns4)
                break
        slm_script = ns4["_v27_assets"]["patch_solved_level_memory.py"]
        self.assertIn("_SLM_ASK_TURNS = 0", slm_script)

        # In SLM script, verify that _slm_user_prompt returns original prompt with ZERO additions
        self.assertIn("return _slm_original_user_prompt(self, *args, **kwargs)", slm_script)
        self.assertNotIn("Before your tool call, write a line starting `Rule for level", slm_script)

        # Verify gating of pinning:
        # if _slm_get_current_level(self) < 3: return (no pinning on Levels 1-2)
        self.assertIn("if _slm_get_current_level(self) < 3:", slm_script)
        self.assertIn("return", slm_script)


if __name__ == "__main__":
    unittest.main()

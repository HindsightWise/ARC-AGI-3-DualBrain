import ast
import hashlib
import json
import subprocess
import tempfile
import types
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
V11_NB_PATH = REPO_ROOT / "pulled_kernels" / "v11" / "arc-prize-2026-v11-apex-engine.ipynb"
V24_NB_PATH = REPO_ROOT / "notebooks" / "submission_v24.ipynb"
SUB_NB_PATH = REPO_ROOT / "notebooks" / "submission.ipynb"
META_PATH = REPO_ROOT / "notebooks" / "kernel-metadata.json"


def _src(cell: dict) -> str:
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else str(s)


class TestV24E2ERequirements(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.nb_v11 = json.loads(V11_NB_PATH.read_text(encoding="utf-8"))
        cls.nb_v24 = json.loads(V24_NB_PATH.read_text(encoding="utf-8"))
        cls.meta = json.loads(META_PATH.read_text(encoding="utf-8"))

    def test_01_byte_identical_notebooks(self) -> None:
        if "v24" not in self.meta.get("id", ""):
            raise unittest.SkipTest("Active kernel is not v24")
        h24 = hashlib.sha256(V24_NB_PATH.read_bytes()).hexdigest()
        hsub = hashlib.sha256(SUB_NB_PATH.read_bytes()).hexdigest()
        self.assertEqual(h24, hsub)
        self.assertEqual(len(self.nb_v24["cells"]), 23)

    def test_02_pure_v11_cells_preserved_bit_for_bit(self) -> None:
        # Cells 1..15, 17..19, 21..22 must be 100% byte-identical to v11 (26.70 LB)
        for idx in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 21, 22]:
            self.assertEqual(
                _src(self.nb_v24["cells"][idx]),
                _src(self.nb_v11["cells"][idx]),
                f"Cell {idx} drifted from pure v11 baseline!",
            )

    def test_03_zero_destructive_v23_mutations(self) -> None:
        c2 = _src(self.nb_v24["cells"][2])
        c4 = _src(self.nb_v24["cells"][4])
        c16 = _src(self.nb_v24["cells"][16])
        self.assertNotIn("self.started_at = time.monotonic()", c2 + c16)
        self.assertNotIn("aging_bonus", c16)
        self.assertNotIn("EXPOSE_RESET", c4 + c16)
        self.assertIn("'LOCAL_ANALYZER_YIELD_TOKENS': '2048'", c4)
        self.assertIn("'LOCAL_ANALYZER_MAX_OUTPUT': 12*1024", c4)
        self.assertIn("'ARC3_GUARDS_FROM_LEVEL': 2", c4)
        self.assertIn("'ARC3_MEMORY_SECTIONS': 'off'", c4)

    def test_04_code_cells_ast_valid(self) -> None:
        for idx, cell in enumerate(self.nb_v24["cells"]):
            if cell.get("cell_type") != "code":
                continue
            src = _src(cell)
            if src.lstrip().startswith("%%writefile"):
                continue
            lines = [
                line for line in src.splitlines()
                if not line.lstrip().startswith("!")
            ]
            cleaned = "\n".join(lines).replace("await bm.run(", "bm.run(")
            ast.parse(cleaned, filename=f"cell_{idx}.py")

    def test_05_d_prime_priority_math_and_install_d(self) -> None:
        c16 = _src(self.nb_v24["cells"][16])
        marker = "_FP_SOURCE = "
        self.assertIn(marker, c16)
        ns: dict = {}
        # Execute only the _FP_SOURCE definition and module creation
        fp_part = "# --- arc-agi-3-agent:" + c16.split("# --- arc-agi-3-agent:", 1)[1]
        fp_lines = []
        for line in fp_part.splitlines():
            if line.startswith("import inference.") or line.startswith("_fp_mod.install_d"):
                continue
            fp_lines.append(line)
        exec("\n".join(fp_lines), ns)
        fp_mod = ns["_fp_mod"]

        # Verify D' priority ordering for unstarted games: N=6 > N=8 > N=10
        Snap = types.SimpleNamespace
        p6 = fp_mod.d_priority(Snap(level=1, actions=0, tokens=0.0, cost_multiplier=0.0, total_levels=6), tail_fraction=1.0)
        p8 = fp_mod.d_priority(Snap(level=1, actions=0, tokens=0.0, cost_multiplier=0.0, total_levels=8), tail_fraction=1.0)
        p10 = fp_mod.d_priority(Snap(level=1, actions=0, tokens=0.0, cost_multiplier=0.0, total_levels=10), tail_fraction=1.0)
        self.assertEqual(p6, 18619)
        self.assertEqual(p8, 17527)
        self.assertEqual(p10, 17000)
        self.assertGreater(p6, p8)
        self.assertGreater(p8, p10)

        # Verify tail fade (phi=0.0) ranks final level L6/6 above L5/6 above L1/6
        p_l6_end = fp_mod.d_priority(Snap(level=6, actions=0, tokens=0.0, cost_multiplier=25000.0, total_levels=6), tail_fraction=0.0)
        p_l5_end = fp_mod.d_priority(Snap(level=5, actions=0, tokens=0.0, cost_multiplier=25000.0, total_levels=6), tail_fraction=0.0)
        p_l1_end = fp_mod.d_priority(Snap(level=1, actions=0, tokens=0.0, cost_multiplier=0.0, total_levels=6), tail_fraction=0.0)
        self.assertGreater(p_l6_end, p_l5_end)
        self.assertGreater(p_l5_end, p_l1_end)

    def test_06_kernel_metadata_offline_compliance(self) -> None:
        if "v24" not in self.meta.get("id", ""):
            raise unittest.SkipTest("Active kernel is not v24")
        self.assertEqual(self.meta["id"], "zerbytheboss/v24-sovereign-apex-engine")
        self.assertEqual(self.meta["code_file"], "submission.ipynb")
        self.assertTrue(self.meta["enable_gpu"])
        self.assertFalse(self.meta["enable_internet"])
        self.assertEqual(self.meta["machine_shape"], "NvidiaRtxPro6000")


if __name__ == "__main__":
    unittest.main()

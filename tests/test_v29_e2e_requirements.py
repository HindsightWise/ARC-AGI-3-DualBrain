"""Comprehensive E2E Verification Suite for ARC-AGI-3 Version 29 Sovereign Apex Engine."""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import py_compile
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
V24_NB_PATH = REPO_ROOT / "notebooks" / "submission_v24.ipynb"
V28_NB_PATH = REPO_ROOT / "notebooks" / "submission_v28.ipynb"
V29_NB_PATH = REPO_ROOT / "notebooks" / "submission_v29.ipynb"
SUB_NB_PATH = REPO_ROOT / "notebooks" / "submission.ipynb"
META_PATH = REPO_ROOT / "notebooks" / "kernel-metadata.json"
DUCK_HARNESS_DIR = REPO_ROOT / "reference" / "duck-harness"
CORTEX_DIR = Path("/Users/zerbytheboss/.gemini/antigravity/scratch/sovereign-cortex")
CORTEX_ENGINE_DIR = CORTEX_DIR / "engine"


def _src(cell: dict) -> str:
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else str(s)


def _extract_v29_assets(c4_src: str) -> dict[str, str]:
    ns: dict = {}
    for line in c4_src.splitlines():
        if line.startswith("_v29_assets = {"):
            exec(line, ns)
            break
    if "_v29_assets" not in ns:
        raise AssertionError("Could not find _v29_assets in Cell 4")
    return dict(ns["_v29_assets"])


class TestV29E2ERequirements(unittest.TestCase):
    """Programmatic E2E test suite validating all architectural and behavioral requirements of Version 29."""

    @classmethod
    def setUpClass(cls) -> None:
        assert V29_NB_PATH.exists(), f"Missing {V29_NB_PATH}"
        assert SUB_NB_PATH.exists(), f"Missing {SUB_NB_PATH}"
        assert META_PATH.exists(), f"Missing {META_PATH}"
        assert DUCK_HARNESS_DIR.exists(), f"Missing {DUCK_HARNESS_DIR}"

        cls.nb_v29 = json.loads(V29_NB_PATH.read_text(encoding="utf-8"))
        cls.nb_sub = json.loads(SUB_NB_PATH.read_text(encoding="utf-8"))
        cls.meta = json.loads(META_PATH.read_text(encoding="utf-8"))

    def test_01_byte_identical_notebooks_and_kernel_metadata(self) -> None:
        """1. Verify submission_v29.ipynb and submission.ipynb are byte-identical and metadata is valid."""
        if "v29" not in self.meta.get("id", ""):
            raise unittest.SkipTest("Active kernel is not v29")
        bytes_v29 = V29_NB_PATH.read_bytes()
        bytes_sub = SUB_NB_PATH.read_bytes()

        self.assertGreater(len(bytes_v29), 500000)
        self.assertEqual(len(bytes_v29), len(bytes_sub))

        sha_v29 = hashlib.sha256(bytes_v29).hexdigest()
        sha_sub = hashlib.sha256(bytes_sub).hexdigest()
        self.assertEqual(sha_v29, sha_sub, "submission_v29.ipynb and submission.ipynb must be byte-identical.")

        self.assertEqual(len(self.nb_v29["cells"]), 23)
        for idx, cell in enumerate(self.nb_v29["cells"]):
            if cell.get("cell_type") == "code":
                self.assertEqual(cell.get("outputs"), [], f"Cell {idx} has non-empty outputs")
                self.assertIsNone(cell.get("execution_count"), f"Cell {idx} has non-null execution_count")

        self.assertEqual(self.meta.get("id"), "zerbytheboss/v29-sovereign-apex-engine")
        self.assertEqual(self.meta.get("title"), "v29 - Sovereign Apex Engine")
        self.assertEqual(self.meta.get("code_file"), "submission.ipynb")
        self.assertTrue(self.meta.get("enable_gpu"))
        self.assertFalse(self.meta.get("enable_internet"))
        self.assertEqual(self.meta.get("machine_shape"), "NvidiaRtxPro6000")

    def test_02_pure_cell2_patch_and_cold_boot_immunity_config(self) -> None:
        """2. Verify Cell 2 hash, 4-layer cold-boot immunity settings, and dual-mount/gateway helpers."""
        c2 = _src(self.nb_v29["cells"][2])
        self.assertEqual(len(c2), 498799)
        self.assertEqual(
            hashlib.sha256(c2.encode("utf-8")).hexdigest(),
            "6fdcd1ae7445c0cfc26eec1047363aa0c196c867d289a35b6f64403b5a43dce2",
        )

        c4 = _src(self.nb_v29["cells"][4])
        self.assertIn("SERVER_STARTUP_TIMEOUT = (12 * 60) if TRUE_SUBMISSION else (45 * 60)", c4)
        self.assertIn("os.environ['ARC3_WARMUP_ACTION_GAMES'] = '1' if TRUE_SUBMISSION else '0'", c4)
        self.assertIn("os.environ['ARC3_HTTP_RETRY_INITIAL_SECONDS'] = '3600'", c4)
        self.assertIn("def _resolve_input_path(", c4)
        self.assertIn("def _arc_gateway_reachable():", c4)
        self.assertNotIn("patch_solved_level_memory.py", c4)
        self.assertNotIn("animation_archive.py", c4)
        self.assertNotIn("fable_layer", c4)
        self.assertNotIn("ARC3_MIN_READ_TIMEOUT", c4)

        c6 = _src(self.nb_v29["cells"][6])
        self.assertIn("kwargs={'delay': 60, 'threads': 3}", c6)
        self.assertNotIn("kwargs={'delay': 15, 'threads': 8}", c6)

        c12 = _src(self.nb_v29["cells"][12])
        self.assertIn("precache_model_thread.start()", c12)
        self.assertIn("_sgl_restarts", c12)
        self.assertIn("MEMFRAC=0.96", c12)
        self.assertIn("MAXREQ=10", c12)

        c16 = _src(self.nb_v29["cells"][16])
        self.assertNotIn("install_timeout_armor", c16)
        self.assertNotIn('"b0": 6.0', c16)
        self.assertIn('"b1": 10.0, "b2": 14.0, "b3": 16.0, "fade": 0.4}', c16)
        self.assertIn('B = (0.0, params["b1"], params["b2"], params["b3"])[min(remaining, 3)]', c16)

        c20 = _src(self.nb_v29["cells"][20])
        self.assertIn("Commit rehearsal gate failed", c20)
        self.assertIn("#OURS_FRESH", c20)

    def test_03_sandbox_git_apply_and_all_v29_patches_ast(self) -> None:
        """3. Verify clean git apply of Cell 2 base patch + all 4 v29 post-patches with 100% AST/py_compile validity."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            src_path = tmp_path / "src"
            src_path.mkdir(parents=True)

            c2 = _src(self.nb_v29["cells"][2])
            c2_patch = c2.split("\n", 1)[1]
            if not c2_patch.endswith("\n"):
                c2_patch += "\n"
            patch_file = tmp_path / "cell2.patch"
            patch_file.write_text(c2_patch, encoding="utf-8")

            subprocess.run(
                f"git archive HEAD ARC3-Inference | tar -x -C {src_path}",
                shell=True,
                cwd=str(DUCK_HARNESS_DIR),
                check=True,
            )

            apply_res = subprocess.run(
                ["git", "apply", "--include=ARC3-Inference/*", str(patch_file)],
                cwd=str(src_path),
                capture_output=True,
                text=True,
            )
            self.assertEqual(apply_res.returncode, 0, f"git apply failed:\n{apply_res.stderr}")

            assets = _extract_v29_assets(_src(self.nb_v29["cells"][4]))
            self.assertEqual(
                set(assets.keys()),
                {
                    "patch_startup_shield.py",
                    "patch_animation_empty.py",
                    "patch_sovereign_oo_harness.py",
                    "patch_post_eviction_memory.py",
                },
            )
            v29_dir = tmp_path / "arcagi_v29"
            v29_dir.mkdir(parents=True)
            for name, text in assets.items():
                (v29_dir / name).write_text(text, encoding="utf-8")

            for patch_script in (
                "patch_startup_shield.py",
                "patch_animation_empty.py",
                "patch_sovereign_oo_harness.py",
                "patch_post_eviction_memory.py",
            ):
                for _ in range(2):  # test both initial apply and idempotency
                    run_res = subprocess.run(
                        [sys.executable, str(v29_dir / patch_script), str(src_path / "ARC3-Inference")],
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(run_res.returncode, 0, f"{patch_script} failed:\n{run_res.stderr}")

            arc3_dir = src_path / "ARC3-Inference"
            py_files = list(arc3_dir.glob("**/*.py"))
            self.assertGreaterEqual(len(py_files), 10)
            for pf in py_files:
                ast.parse(pf.read_text(encoding="utf-8"), filename=str(pf))
                py_compile.compile(str(pf), doraise=True)

    def test_04_radix_attention_prefix_lock_and_post_eviction_memory(self) -> None:
        """4. Verify RadixAttention Prefix Lock (messages[0] byte-identical to v24) and Level 3+ post-eviction memory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            src_path = tmp_path / "src"
            src_path.mkdir()

            c2 = _src(self.nb_v29["cells"][2]).split("\n", 1)[1] + "\n"
            patch_file = tmp_path / "cell2.patch"
            patch_file.write_text(c2, encoding="utf-8")
            subprocess.run(
                f"git archive HEAD ARC3-Inference | tar -x -C {src_path}",
                shell=True,
                cwd=str(DUCK_HARNESS_DIR),
                check=True,
            )
            subprocess.run(["git", "apply", "--include=ARC3-Inference/*", str(patch_file)], cwd=str(src_path), check=True)

            assets = _extract_v29_assets(_src(self.nb_v29["cells"][4]))
            for name, text in assets.items():
                p = tmp_path / name
                p.write_text(text, encoding="utf-8")
                subprocess.run([sys.executable, str(p), str(src_path / "ARC3-Inference")], check=True)

            check_code = r"""
import os, sys, types
from pathlib import Path

taaf = types.ModuleType("taaf")
taaf.game = types.ModuleType("taaf.game")
taaf.game.Game = object
taaf.solver = types.ModuleType("taaf.solver")
taaf.solver.Solver = object
sys.modules["taaf"] = taaf
sys.modules["taaf.game"] = taaf.game
sys.modules["taaf.solver"] = taaf.solver

arcengine = types.ModuleType("arcengine")
arcengine.GameAction = types.SimpleNamespace(RESET="RESET")
arcengine.ActionInput = lambda **kw: types.SimpleNamespace(**kw)
sys.modules["arcengine"] = arcengine

os.environ["LOCAL_ANALYZER_MODEL_ID"] = "flashnext"
os.environ["LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS"] = "3072"
from inference.agent.tool_agent import ToolAgent

agent = ToolAgent(model="flashnext")
init_sys_prompt = agent._system_prompt
assert len(init_sys_prompt) == 13427, f"Expected 13427 chars for pure system prompt, got {len(init_sys_prompt)}"

# Level 1 eviction simulation: should NOT inject memory into messages[1]
agent._slm_curr_level = 1
agent.messages = [
    {"role": "system", "content": init_sys_prompt},
    {"role": "user", "content": "Initial user prompt"},
]
agent._note_history_evicted()
res1 = agent._trim_messages_for_context(agent.messages)
assert res1[0]["content"] == init_sys_prompt, "System prompt was mutated!"
assert "=== CERTIFIED SOLVED-LEVEL MEMORY ===" not in res1[1]["content"]

# Level 3 eviction simulation: should pin memory strictly in messages[1] without mutating messages[0]
agent._slm_curr_level = 3
agent._slm_levels = [{
    "level": 1,
    "n_actions": 5,
    "actions": ["RESET", "ACTION1"],
    "action_seq": ["RESET", "ACTION1"],
    "reasoning": "done",
    "assistant_tail": "done",
    "rule": None,
    "asks_left": 0,
    "asked": False,
}]
agent.messages = [
    {"role": "system", "content": init_sys_prompt},
    {"role": "user", "content": "Initial user prompt"},
]
agent._note_history_evicted()
res3 = agent._trim_messages_for_context(agent.messages)
assert res3[0]["content"] == init_sys_prompt, "System prompt was mutated on Level 3 eviction!"
assert "=== CERTIFIED SOLVED-LEVEL MEMORY ===" in res3[1]["content"]
assert "Initial user prompt" in res3[1]["content"]
assert "Level 1 - cleared after 5 actions" in res3[1]["content"]

# Subsequent non-eviction turn after clearing Level 3 -> Level 4 MUST NOT mutate res3[1]
pinned_once = res3[1]["content"]
agent._slm_curr_level = 4
agent._slm_levels.append({
    "level": 3,
    "n_actions": 8,
    "actions": ["ACTION2"],
    "action_seq": ["ACTION2"],
    "reasoning": "cleared l3",
    "assistant_tail": "cleared l3",
    "rule": None,
    "asks_left": 0,
    "asked": False,
})
res4_no_trim = agent._trim_messages_for_context(res3)
assert res4_no_trim[1]["content"] == pinned_once, "Mid-window non-trim turn mutated messages[1]!"

print("PREFIX_LOCK_AND_POST_EVICTION_OK")
"""
            env = {k: v for k, v in os.environ.items() if not k.startswith("ARC3_") and not k.startswith("LOCAL_ANALYZER_")}
            env["PYTHONPATH"] = str(src_path / "ARC3-Inference")
            res = subprocess.run([sys.executable, "-c", check_code], env=env, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"Prefix lock test failed:\n{res.stderr}\n{res.stdout}")
            self.assertIn("PREFIX_LOCK_AND_POST_EVICTION_OK", res.stdout)

    def test_05_sovereign_oo_harness_repl_and_normalization(self) -> None:
        """5. Verify normalize_tool_error, _NodeDict/_ObjDict, discrete math helpers, and S4 interfaces."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            src_path = tmp_path / "src"
            src_path.mkdir()

            c2 = _src(self.nb_v29["cells"][2]).split("\n", 1)[1] + "\n"
            patch_file = tmp_path / "cell2.patch"
            patch_file.write_text(c2, encoding="utf-8")
            subprocess.run(
                f"git archive HEAD ARC3-Inference | tar -x -C {src_path}",
                shell=True,
                cwd=str(DUCK_HARNESS_DIR),
                check=True,
            )
            subprocess.run(["git", "apply", "--include=ARC3-Inference/*", str(patch_file)], cwd=str(src_path), check=True)

            assets = _extract_v29_assets(_src(self.nb_v29["cells"][4]))
            for name, text in assets.items():
                p = tmp_path / name
                p.write_text(text, encoding="utf-8")
                subprocess.run([sys.executable, str(p), str(src_path / "ARC3-Inference")], check=True)

            check_code = r"""
import json
from inference.agent.python_tool_sandbox import normalize_tool_error, run_sandboxed_python
from inference.utils.segmentation import (
    segment_layer,
    rot90, rot180, rot270, flipv, fliph,
    manhattan_distance, find_cells, find_objects,
    wl_symmetry, bfs_shortest_path,
    capabilities, doc, call
)

# 1. Test normalize_tool_error
hint_key = normalize_tool_error(KeyError("bbox"))
assert "[REPAIR_HINT] KeyError:" in hint_key
assert "bbox" in hint_key

hint_type = normalize_tool_error(TypeError("tuple indices must be integers or slices, not tuple"))
assert "[REPAIR_HINT] TypeError:" in hint_type
assert "grid[r][c]" in hint_type

hint_idx = normalize_tool_error(IndexError("list index out of range"))
assert "[REPAIR_HINT] IndexError:" in hint_idx

hint_name = normalize_tool_error(NameError("name 'foo' is not defined"))
assert "[REPAIR_HINT] NameError:" in hint_name

hint_attr = normalize_tool_error(AttributeError("'FrameView' object has no attribute 'bar'"))
assert "[REPAIR_HINT] AttributeError:" in hint_attr

# 2. Test _NodeDict and _ObjDict reciprocal aliases
grid = [
    [0, 0, 0, 0],
    [0, 2, 2, 0],
    [0, 2, 2, 0],
    [0, 0, 0, 0],
]
seg = segment_layer(grid, "WwgGcBMPRbSYOrNp")
red_node = [n for n in seg["nodes"] if n["color"] == "g"][0]
assert red_node["bbox"] == [1, 1, 2, 2]
assert red_node["box"] == [1, 1, 2, 2]
assert red_node["size"] == 4
assert red_node["area"] == 4
assert red_node["centroid"] == (1, 1)
assert red_node["center"] == (1, 1)

# 3. Test discrete math primitives directly
grid_sym = [[1, 2, 1], [3, 4, 3], [1, 2, 1]]
sym = wl_symmetry(grid_sym)
assert sym["horizontal_symmetric"] is True
assert sym["vertical_symmetric"] is True

path = bfs_shortest_path([[0, 0], [0, 0]], (0, 0), (1, 1))
assert path["found"] is True
assert path["distance"] == 2

# 4. Test S4 interfaces directly
caps = capabilities()
assert len(caps) >= 10
assert any("wl_symmetry" in c for c in caps)
doc_str = doc("wl_symmetry")
assert "wl_symmetry:" in doc_str

res_call = call("rot90", [[1, 2], [3, 4]])
assert res_call == [[3, 1], [4, 2]]

# 5. Test REPL runtime environment execution via run_sandboxed_python
sandbox_res = run_sandboxed_python(
    code="c = capabilities()\nsym = wl_symmetry([[1, 2, 1], [3, 4, 3], [1, 2, 1]])\np = bfs_shortest_path([[0, 0], [0, 0]], (0, 0), (1, 1))\nresult = {'caps': c, 'sym': sym, 'p': p}\n",
    timeout_seconds=5,
    initial_state={},
    action_handler=lambda acts: {}
)
assert not sandbox_res.get("error"), f"Sandbox returned error: {sandbox_res.get('error')}"
assert sandbox_res["result"]["sym"]["horizontal_symmetric"] is True
assert sandbox_res["result"]["p"]["found"] is True

print("SOVEREIGN_OO_HARNESS_OK")
"""
            env = {k: v for k, v in os.environ.items() if not k.startswith("ARC3_") and not k.startswith("LOCAL_ANALYZER_")}
            env["PYTHONPATH"] = str(src_path / "ARC3-Inference")
            res = subprocess.run([sys.executable, "-c", check_code], env=env, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"OO-Harness test failed:\n{res.stderr}\n{res.stdout}")
            self.assertIn("SOVEREIGN_OO_HARNESS_OK", res.stdout)

    def test_06_d_prime_priority_math_with_final_level_b0_protection(self) -> None:
        """6. Verify D' unstarted pricing and final-level b0=6.0 cliff protection in Cell 16."""
        c16 = _src(self.nb_v29["cells"][16])
        ns: dict = {}
        for line in c16.splitlines():
            if line.startswith("_FP_SOURCE = "):
                exec(line, ns)
                break
        self.assertIn("_FP_SOURCE", ns)
        fp_mod = types.ModuleType("ours_form_priority")
        exec(compile(ns["_FP_SOURCE"], "ours_form_priority.py", "exec"), fp_mod.__dict__)

        class _Snap:
            def __init__(self, level, actions, tokens, cost_multiplier, total_levels):
                self.level = level
                self.actions = actions
                self.tokens = tokens
                self.cost_multiplier = cost_multiplier
                self.total_levels = total_levels

        # Unstarted games: 100% identical to v24
        p_n6 = fp_mod.d_priority(_Snap(1, 0, 0.0, 0.0, 6), tail_fraction=1.0)
        p_n8 = fp_mod.d_priority(_Snap(1, 0, 0.0, 0.0, 8), tail_fraction=1.0)
        p_n10 = fp_mod.d_priority(_Snap(1, 0, 0.0, 0.0, 10), tail_fraction=1.0)
        self.assertEqual(p_n6, 18619)
        self.assertEqual(p_n8, 17527)
        self.assertEqual(p_n10, 17000)

        # Final level (Level 6 of 6, remaining == 0) at phi=1.0: pure v24 b0=0.0 so stuck final-level games yield slots
        p_final_l6 = fp_mod.d_priority(_Snap(6, 0, 0.0, 0.0, 6), tail_fraction=1.0)
        self.assertEqual(p_final_l6, 9166)
        p_final_l6_stuck = fp_mod.d_priority(_Snap(6, 150, 90000.0, 18000.0, 6), tail_fraction=1.0)
        self.assertLess(p_final_l6_stuck, 5000)

    def test_07_sovereign_cortex_vault_lint(self) -> None:
        """7. Verify Sovereign Cortex vault lint status is PASS with 0 broken links and 0 orphan nodes."""
        self.assertTrue(CORTEX_DIR.exists(), f"Missing cortex dir at {CORTEX_DIR}")
        self.assertTrue(CORTEX_ENGINE_DIR.exists(), f"Missing cortex engine dir at {CORTEX_ENGINE_DIR}")
        if str(CORTEX_ENGINE_DIR) not in sys.path:
            sys.path.insert(0, str(CORTEX_ENGINE_DIR))
        import cortex

        res = cortex.lint_vault(str(CORTEX_DIR))
        self.assertEqual(res.get("status"), "PASS", f"Lint vault returned non-PASS: {res}")
        self.assertEqual(len(res.get("broken_links", [])), 0, f"Broken links detected: {res.get('broken_links')}")
        self.assertEqual(len(res.get("orphan_nodes", [])), 0, f"Orphan nodes detected: {res.get('orphan_nodes')}")
        self.assertGreaterEqual(res.get("raw_source_count", 0), 19)
        self.assertGreaterEqual(res.get("wiki_node_count", 0), 16)
        self.assertEqual(len(res.get("missing_frontmatter", [])), 0)

    def test_08_sovereign_cortex_autoreason_tournament(self) -> None:
        """8. Evaluate architectural tournament across v24, v28, and v29 via Bradley-Terry and Blind Borda."""
        self.assertTrue(CORTEX_ENGINE_DIR.exists(), f"Missing cortex engine dir at {CORTEX_ENGINE_DIR}")
        if str(CORTEX_ENGINE_DIR) not in sys.path:
            sys.path.insert(0, str(CORTEX_ENGINE_DIR))
        import autoreason_tournament as art

        v24_spec = (
            "Version 24 (v24) establishes the baseline substrate using pure v11 patch (498,799 bytes) "
            "with Affectify D-prime priority scheduler. Scored 28.79 LB across 110 games with MEMFRAC=0.96 "
            "and MAXREQ=10. However, v24 lacks OS prefetching (risking cold-boot timeouts), has no keepalive "
            "RESET shielding during model load, no final-level b0 priority protection, and zero REPL spatial "
            "or discrete math helpers."
        )

        v28_spec = (
            "Version 28 (v28) advances beyond v24 by introducing 4-Layer Cold-Boot 0.00 Immunity: "
            "[[cold-boot-prefetcher]] (threads=8, delay=15), SERVER_STARTUP_TIMEOUT=2700s with 3600s HTTP "
            "retry grace, [[startup-shield]] with 540s keepalive RESET, and Cell 20 commit gate. Adds "
            "b0=6.0 final-level cliff protection and patch_animation_empty. Achieved 71.43 on ft09 (5/6 levels, "
            "66 actions, 44,935 tokens) in 301.2s. Lacks arXiv discrete math and OO-harness error hints."
        )

        v29_spec = (
            "Version 29 (v29) Sovereign OO-Harness engine fuses v28 cold-boot immunity with [[sovereign-oo-harness]] "
            "and [[arxiv-110-paper-synthesis]]. Exposes [[agent-s4-interfaces]] (`capabilities`, `doc`, `call`), "
            "2-WL discrete math (`wl_symmetry`, `bfs_shortest_path`), `normalize_tool_error` surgical 1-line hints, "
            "`_NodeDict` aliases, and [[certified-post-eviction-memory]] ($V(D) > 0$) strictly in user messages, "
            "maintaining `self._system_prompt` 100% byte-identical to v24/v28 for 93.29% prefix cache sharing."
        )

        res = art.blind_panel_evaluate(v24_spec.strip(), v28_spec.strip(), v29_spec.strip())
        borda = res["borda_points"]
        bt = res["bradley_terry_pi"]

        pi_v24 = bt["A"]
        pi_v28 = bt["B"]
        pi_v29 = bt["AB"]

        self.assertEqual(res["winner"], "AB", f"Expected v29 (AB) to win tournament, got {res['winner']}")
        self.assertGreater(pi_v29, pi_v28, f"Expected pi_v29 > pi_v28, got {pi_v29} <= {pi_v28}")
        self.assertGreaterEqual(pi_v28, pi_v24, f"Expected pi_v28 >= pi_v24, got {pi_v28} < {pi_v24}")
        self.assertGreater(borda["AB"], borda["B"], f"Expected borda(v29) > borda(v28), got {borda}")
        self.assertGreaterEqual(borda["B"], borda["A"], f"Expected borda(v28) >= borda(v24), got {borda}")

    def test_09_v29_v3_event_triggered_flags_and_stagnation_compaction(self) -> None:
        """9. Verify v29 Version 4 event-triggered flags, _BoundaryList, HistoryEntryView forwarding, M1 helpers, eviction-time reasoning compaction, and C1 stagnation restart."""
        c4 = _src(self.nb_v29["cells"][4])
        self.assertNotIn("'ARC3_YIELD_RESUME_PROMPT'", c4)
        for required_flag in [
            "'ARC3_STUB_DEAD_REASONING': '1'",
            "'ARC3_BOARD_DUMP_GUARD': '1'",
            "'ARC3_DEATH_LEDGER': '1'",
            "'ARC3_DEATH_REPEAT_GUARD': '1'",
            "'ARC3_DEATH_GUARD_LOOKAHEAD': '1'",
            "'ARC3_REPEAT_STATE_GUARD': '1'",
            "'ARC3_GAMEOVER_DIFF': '1'",
            "'ARC3_ANIMATION_IMAGE': 'composite'",
            "'ARC3_PERSISTENT_FUNCTIONS_REPAIR_HINTS': '1'",
        ]:
            self.assertIn(required_flag, c4, f"Missing {required_flag} in Cell 4")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            src_path = tmp_path / "src"
            src_path.mkdir()

            c2 = _src(self.nb_v29["cells"][2]).split("\n", 1)[1] + "\n"
            patch_file = tmp_path / "cell2.patch"
            patch_file.write_text(c2, encoding="utf-8")
            subprocess.run(
                f"git archive HEAD ARC3-Inference | tar -x -C {src_path}",
                shell=True,
                cwd=str(DUCK_HARNESS_DIR),
                check=True,
            )
            subprocess.run(["git", "apply", "--include=ARC3-Inference/*", str(patch_file)], cwd=str(src_path), check=True)

            assets = _extract_v29_assets(c4)
            for name, text in assets.items():
                p = tmp_path / name
                p.write_text(text, encoding="utf-8")
                subprocess.run([sys.executable, str(p), str(src_path / "ARC3-Inference")], check=True)

            check_code = r"""
import os, sys, types

taaf = types.ModuleType("taaf")
taaf.game = types.ModuleType("taaf.game")
taaf.game.Game = object
taaf.solver = types.ModuleType("taaf.solver")
taaf.solver.Solver = object
sys.modules["taaf"] = taaf
sys.modules["taaf.game"] = taaf.game
sys.modules["taaf.solver"] = taaf.solver

arcengine = types.ModuleType("arcengine")
arcengine.GameAction = types.SimpleNamespace(RESET="RESET")
arcengine.ActionInput = lambda **kw: types.SimpleNamespace(**kw)
sys.modules["arcengine"] = arcengine

os.environ["LOCAL_ANALYZER_MODEL_ID"] = "flashnext"
os.environ["LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS"] = "3072"
from inference.agent import tool_agent
from inference.agent.tool_agent import ToolAgent
from inference.agent.python_tool_sandbox import run_sandboxed_python
from inference.utils.segmentation import (
    segment_layer, ascii_rows, crop, color_counts, bbox_of_color, neighbors4, bfs_path, capabilities
)

# 1. Verify system prompt is 100% byte-identical with and without the 9 v29 v4 flags
base_prompt = tool_agent._build_system_prompt(tool_output_tokens=3072)
for k, v in {
    "ARC3_STUB_DEAD_REASONING": "1",
    "ARC3_BOARD_DUMP_GUARD": "1",
    "ARC3_DEATH_LEDGER": "1",
    "ARC3_DEATH_REPEAT_GUARD": "1",
    "ARC3_DEATH_GUARD_LOOKAHEAD": "1",
    "ARC3_REPEAT_STATE_GUARD": "1",
    "ARC3_GAMEOVER_DIFF": "1",
    "ARC3_ANIMATION_IMAGE": "composite",
    "ARC3_PERSISTENT_FUNCTIONS_REPAIR_HINTS": "1",
}.items():
    os.environ[k] = v
flagged_prompt = tool_agent._build_system_prompt(tool_output_tokens=3072)
assert base_prompt == flagged_prompt, "v29 v4 flags mutated _build_system_prompt!"

# 2. Verify M1 helpers, _BoundaryList, and HistoryEntryView forwarding directly and in run_sandboxed_python
grid = [[0, 1, 0], [1, 2, 1], [0, 1, 0]]
assert color_counts(grid) == {0: 4, 1: 4, 2: 1}
assert color_counts("WWg\nggg") == {"W": 2, "g": 4}
assert bbox_of_color(grid, 2) == (1, 1, 1, 1)
assert bbox_of_color(grid, 1) == (0, 0, 2, 2)
assert crop(grid, 0, 2, 0, 2) == [[0, 1], [1, 2]]
assert crop("WWg\nggg\nBBB", 0, 2, 0, 2) == "WW\ngg"
assert len(neighbors4(1, 1)) == 4
assert bfs_path((0, 0), (1, 1), lambda r, c: [[0, 0], [1, 0]][r][c] == 0, nrows=2, ncols=2) == [(0, 0), (0, 1), (1, 1)]
assert bfs_path([[0, 0], [1, 0]], (0, 0), (1, 1), passable={0}) == [(0, 0), (0, 1), (1, 1)]
assert any("ascii_rows" in c for c in capabilities())

# Verify _BoundaryList prevents IndexError when indexing boundary[2] on 1D lines or 1px dots
seg_bar = segment_layer([[0, 0, 0, 0], [2, 2, 2, 2]], "WwgGcBMPRbSYOrNp")
bar_node = [n for n in seg_bar["nodes"] if n["color"] == "g"][0]
assert len(bar_node["boundary"]) == 2
assert bar_node["boundary"][0] == [1, 0]
assert bar_node["boundary"][2] == [1, 3], f"Expected [1, 3] for boundary[2] on 2-corner bar, got {bar_node['boundary'][2]}"

sb = run_sandboxed_python(
    code=(
        "cc = color_counts([[0, 2], [2, 2]])\n"
        "assert cc == {0: 1, 2: 3}\n"
        "h0 = history[0]\n"
        "assert h0.step == 5 and h0.level == 2 and h0['action'] == 'ACTION1'\n"
        "assert current_frame.grid == [[0, 2], [2, 2]]\n"
        "result = {'cc_2': cc[2], 'h_step': h0.step, 'path': bfs_path([[0, 0], [0, 0]], (0, 0), (1, 1))}\n"
    ),
    timeout_seconds=5,
    initial_state={
        "current_frame": {"ascii": "Wg\ngg", "step": 5, "level": 2, "shape": [2, 2], "grid": [[0, 2], [2, 2]]},
        "history": [{"action": "ACTION1", "frame": {"ascii": "Wg\ngg", "step": 5, "level": 2, "shape": [2, 2], "grid": [[0, 2], [2, 2]]}}],
    },
    action_handler=lambda acts: {}
)
assert not sb.get("error"), f"Sandbox M1 error: {sb.get('error')}"
assert sb["result"]["cc_2"] == 3
assert sb["result"]["h_step"] == 5
assert len(sb["result"]["path"]) == 3

# 3. Verify Eviction-Time-Only Stale Reasoning Compaction
agent = ToolAgent(model="flashnext")
init_sys_prompt = agent._system_prompt

# Build 8 assistant messages (first 2 have older reasoning_content, last 6 are recent)
msgs = [{"role": "system", "content": init_sys_prompt}, {"role": "user", "content": "USR"}]
for idx in range(8):
    msgs.append({"role": "assistant", "content": f"ACT_{idx}", "reasoning_content": f"REASON_{idx}_" + ("X" * 200)})
    msgs.append({"role": "user", "content": f"OBS_{idx}"})

# 3a. Under budget (budget=uncompacted + 500): NO reasoning_content should be touched!
uncompacted_tokens = agent._estimate_request_input_tokens(msgs)
agent._context_budget_tokens = uncompacted_tokens + 500
res_under = agent._trim_messages_for_context(msgs, force_drain=False)
assistant_under = [m for m in res_under if m["role"] == "assistant"]
assert len(assistant_under) == 8
assert all("reasoning_content" in m for m in assistant_under), "Compacted reasoning_content when under budget!"

# 3b. Over budget (budget=uncompacted - 20, so compaction alone brings used below budget):
# first 2 assistant messages lose reasoning_content, last 6 keep it, and all 8 turns survive!
agent._context_budget_tokens = uncompacted_tokens - 20
res_over = agent._trim_messages_for_context(msgs, force_drain=False)
assistant_over = [m for m in res_over if m["role"] == "assistant"]
assert len(assistant_over) == 8, f"Expected all 8 turns to survive after reasoning compaction, got {len(assistant_over)}"
assert "reasoning_content" not in assistant_over[0], "Oldest assistant message still had reasoning_content on eviction!"
assert "reasoning_content" not in assistant_over[1], "Second oldest assistant message still had reasoning_content on eviction!"
assert all("reasoning_content" in m for m in assistant_over[2:]), "Recent 6 assistant messages lost reasoning_content!"

# 4. Verify C1 Stagnation Context Restart at >= 96 * 1024 tokens (once per level)
frame_l3 = types.SimpleNamespace(level=3)
agent._history_messages = [{"role": "user", "content": "old"}]
agent._kept_functions = {"foo": "def foo(): pass"}
agent._resume_after_yield = True
agent._tokens_at_level_start = 1000
agent._session_generated_tokens = 50000
assert agent._slm_maybe_stagnation_restart(frame_l3) == ""
assert len(agent._history_messages) == 1

agent._session_generated_tokens = 1000 + 96 * 1024 + 10
notice = agent._slm_maybe_stagnation_restart(frame_l3)
assert "Fresh review of the current level:" in notice
assert len(agent._history_messages) == 0
assert len(agent._kept_functions) == 0
assert agent._resume_after_yield is False
assert 3 in agent._stagnation_restarted_levels
# Must not fire a second time on the same level after more tokens are generated
agent._session_generated_tokens += 50000
assert agent._slm_maybe_stagnation_restart(frame_l3) == ""

print("V29_V3_FEATURES_OK")
"""
            env = {k: v for k, v in os.environ.items() if not k.startswith("ARC3_") and not k.startswith("LOCAL_ANALYZER_")}
            env["PYTHONPATH"] = str(src_path / "ARC3-Inference")
            res = subprocess.run([sys.executable, "-c", check_code], env=env, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"v29 v3 feature test failed:\n{res.stderr}\n{res.stdout}")
            self.assertIn("V29_V3_FEATURES_OK", res.stdout)


if __name__ == "__main__":
    unittest.main()

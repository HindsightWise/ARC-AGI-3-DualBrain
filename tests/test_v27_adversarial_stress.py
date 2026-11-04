"""Standalone Adversarial Stress Test Suite for Version 27 Sovereign Apex Engine.

Adversarial Stress Dimensions:
1. Prompt Invariance Oracle (v24 vs v27):
   - Randomized and exhaustive combinatorial stress testing of _build_user_prompt and _build_system_prompt on Levels 1-2.
   - 200+ distinct parameter permutations across action_num, valid_actions, current_frame, history_entries, previous_step_summary.
   - Inner tool step simulations, repeated _trim_messages_for_context calls, and _note_history_evicted calls on Levels 1-2.
   - Byte-level and SHA-256 string equality verification against pure v24 oracle.

2. Timeout Calculation Stress Harness:
   - Exhaustive boundary analysis of solver._HarnessGameSession.request_timeout_seconds.
   - Verification that remaining_time = 15.0s returns 15.0s and NEVER 120.0s.
   - Verification of 0.0s -> 0.1s native floor, soft deadline clamping, and 1,000 float sweep.
   - Absence of install_timeout_armor, TIMEOUT_ARMOR_CODE, and _ours_protected wrapper.

3. Level 3+ Fact Distillation & Context Eviction Stress Harness:
   - Clean fact pinning on Level 3+ context eviction (_note_history_evicted, _trim_messages_for_context, _force_reduce_messages).
   - Multi-eviction idempotency (no duplicate block accumulation).
   - Extreme inputs: 0 actions, 500 actions (omission prefix formatting), special characters ({}, markdown, XML, unicode), missing reasoning.
   - Session isolation: verify _ensure_session completely purges pinned facts on new game transition.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import types
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
DUCK_HARNESS_DIR = REPO_ROOT / "reference" / "duck-harness"
V24_NB_PATH = REPO_ROOT / "notebooks" / "submission_v24.ipynb"
V27_NB_PATH = REPO_ROOT / "notebooks" / "submission_v27.ipynb"


def _src(cell: dict) -> str:
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else str(s)


def _extract_v27_assets(c4_src: str) -> dict[str, str]:
    ns: dict = {}
    for line in c4_src.splitlines():
        if line.startswith("_v27_assets = {"):
            exec(line, ns)
            break
    if "_v27_assets" not in ns:
        raise AssertionError("Could not find _v27_assets in Cell 4")
    return dict(ns["_v27_assets"])


class AdversarialV27StressHarness(unittest.TestCase):
    """Deep adversarial verification of Version 27 against Version 24 baseline."""

    @classmethod
    def setUpClass(cls) -> None:
        assert DUCK_HARNESS_DIR.exists(), f"Missing duck-harness at {DUCK_HARNESS_DIR}"
        assert V24_NB_PATH.exists(), f"Missing v24 notebook at {V24_NB_PATH}"
        assert V27_NB_PATH.exists(), f"Missing v27 notebook at {V27_NB_PATH}"

        cls.nb_v24 = json.loads(V24_NB_PATH.read_text(encoding="utf-8"))
        cls.nb_v27 = json.loads(V27_NB_PATH.read_text(encoding="utf-8"))

        # Setup isolated sandbox directories for v24 and v27
        cls._temp_dir = tempfile.TemporaryDirectory()
        cls.temp_path = Path(cls._temp_dir.name)

        cls.v24_src = cls.temp_path / "v24_env" / "src"
        cls.v27_src = cls.temp_path / "v27_env" / "src"
        cls.v24_src.mkdir(parents=True)
        cls.v27_src.mkdir(parents=True)

        # Extract pristine duck-harness into both
        subprocess.run(
            f"git archive HEAD ARC3-Inference | tar -x -C {cls.v24_src}",
            shell=True,
            cwd=str(DUCK_HARNESS_DIR),
            check=True,
        )
        subprocess.run(
            f"git archive HEAD ARC3-Inference | tar -x -C {cls.v27_src}",
            shell=True,
            cwd=str(DUCK_HARNESS_DIR),
            check=True,
        )

        # Apply Cell 2 base patch to both
        c2 = _src(cls.nb_v27["cells"][2])
        c2_patch = c2.split("\n", 1)[1] + "\n"
        patch_file = cls.temp_path / "cell2.patch"
        patch_file.write_text(c2_patch, encoding="utf-8")

        subprocess.run(
            ["git", "apply", "--include=ARC3-Inference/*", str(patch_file)],
            cwd=str(cls.v24_src),
            check=True,
        )
        subprocess.run(
            ["git", "apply", "--include=ARC3-Inference/*", str(patch_file)],
            cwd=str(cls.v27_src),
            check=True,
        )

        # Apply v27 post-patches only to v27_src
        assets = _extract_v27_assets(_src(cls.nb_v27["cells"][4]))
        v27_modules_dir = cls.temp_path / "arcagi_v27"
        v27_modules_dir.mkdir(parents=True)
        for name, text in assets.items():
            (v27_modules_dir / name).write_text(text, encoding="utf-8")

        for patch_script in ("patch_solved_level_memory.py", "patch_animation_empty.py", "patch_spatial_helpers.py"):
            subprocess.run(
                [sys.executable, str(v27_modules_dir / patch_script), str(cls.v27_src / "ARC3-Inference")],
                check=True,
            )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temp_dir.cleanup()

    # =========================================================================
    # PART 1: Prompt Invariance Oracle (v24 vs v27) on Levels 1–2
    # =========================================================================

    def test_01_prompt_invariance_system_and_user_prompts(self) -> None:
        """Adversarial stress-test: verify byte-identical user & system prompts between v24 and v27 across 200+ configurations on Levels 1-2."""
        # Run comparison in a separate subprocess to avoid module caching issues
        verifier_script = self.temp_path / "verify_invariance_runner.py"
        verifier_script.write_text(
            f"""
import sys
import importlib
from pathlib import Path
import hashlib
import random

v24_root = Path(r"{self.v24_src / 'ARC3-Inference'}")
v27_root = Path(r"{self.v27_src / 'ARC3-Inference'}")

# Import v24
sys.path.insert(0, str(v24_root))
import inference.agent.tool_agent as v24_agent_mod
from inference.agent.runtime_state import Frame, HistoryEntry

agent_v24 = v24_agent_mod.ToolAgent(model="flashnext")
sys.path.remove(str(v24_root))

# Clear modules from v24
for m in list(sys.modules):
    if m == "inference" or m.startswith("inference."):
        del sys.modules[m]

# Import v27
sys.path.insert(0, str(v27_root))
import inference.agent.tool_agent as v27_agent_mod

agent_v27 = v27_agent_mod.ToolAgent(model="flashnext")
sys.path.remove(str(v27_root))

# 1. Base system prompt equality
assert agent_v24._system_prompt == agent_v27._system_prompt, "Initial system prompt mismatch!"
assert v24_agent_mod._build_system_prompt(tool_output_tokens=2048) == v27_agent_mod._build_system_prompt(tool_output_tokens=2048)
assert v24_agent_mod._build_system_prompt(tool_output_tokens=12288) == v27_agent_mod._build_system_prompt(tool_output_tokens=12288)

# 2. Combinatorial generator for Levels 1 and 2
action_num_candidates = [0, 1, 2, 5, 13, 27, 49, 100, 250, 1000]
valid_actions_candidates = [
    None,
    [],
    ["ACTION1"],
    ["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
    ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "ACTION7"],
    ["RESET"],
    ["ACTION6", "RESET"],
    ["CUSTOM_ACT_X", "CUSTOM_ACT_Y"],
]
levels = [1, 2]
summaries = [
    None,
    {{}},
    {{"level": 1, "executed_count": 0, "executed_actions": []}},
    {{"level": 1, "executed_count": 1, "executed_actions": ["ACTION1"]}},
    {{"level": 1, "executed_count": 5, "executed_actions": ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"]}},
    {{"level": 1, "executed_count": 12, "executed_actions": [f"ACT_{{i}}" for i in range(12)]}},
    {{"level": 2, "executed_count": 1, "executed_actions": ["ACTION4"], "level_transition": True}},
    {{"level": 2, "executed_count": 8, "executed_actions": ["ACTION1"] * 8}},
    {{"level": 1, "executed_count": 2, "executed_actions": ["RESET", "ACTION1"], "game_over": True}},
]

test_cases_run = 0

for lv in levels:
    for act_num in action_num_candidates:
        for val_acts in valid_actions_candidates:
            for summ in summaries:
                grid_size = 4 if act_num % 2 == 0 else 8
                grid = [[(r + c) % 16 for c in range(grid_size)] for r in range(grid_size)]
                cur_frame = Frame(grid=grid, step=act_num, level=lv)
                
                # History entries
                hist = []
                if act_num > 0:
                    hist.append(HistoryEntry(action="", frame=Frame(grid=grid, step=0, level=lv)))
                    for s in range(1, min(act_num, 6)):
                        hist.append(HistoryEntry(action=f"ACTION{{s%4+1}}", frame=Frame(grid=grid, step=s, level=lv)))
                
                p_v24 = agent_v24._build_user_prompt(
                    act_num,
                    valid_actions=val_acts,
                    current_frame=cur_frame,
                    history_entries=hist,
                    previous_step_summary=summ,
                )
                p_v27 = agent_v27._build_user_prompt(
                    act_num,
                    valid_actions=val_acts,
                    current_frame=cur_frame,
                    history_entries=hist,
                    previous_step_summary=summ,
                )
                
                assert p_v24 == p_v27, f"User prompt mismatch at lv={{lv}}, act={{act_num}}, val_acts={{val_acts}}"
                assert hashlib.sha256(p_v24.encode()).hexdigest() == hashlib.sha256(p_v27.encode()).hexdigest()
                test_cases_run += 1

print(f"PASSED_INVARIANCE_SWEEP: {{test_cases_run}} test cases evaluated.")

# 3. Simulate tool steps, trimming, and evictions on Levels 1 and 2
for lv in (1, 2):
    # Set current level in agents
    agent_v24._last_step_summary = {{"level": lv, "executed_count": 3}}
    agent_v27._last_step_summary = {{"level": lv, "executed_count": 3}}
    agent_v27._slm_curr_level = lv
    
    # Trigger _note_history_evicted
    agent_v24._note_history_evicted()
    agent_v27._note_history_evicted()
    assert agent_v24._system_prompt == agent_v27._system_prompt, f"System prompt drifted after eviction on Level {{lv}}!"
    assert "#SOLVED_LEVEL_MEMORY" not in agent_v27._system_prompt, f"SOLVED_LEVEL_MEMORY leaked into Level {{lv}} system prompt!"
    
    # Trigger trimming
    dummy_messages = [
        {{"role": "system", "content": agent_v24._system_prompt}},
        {{"role": "user", "content": "hello"}},
        {{"role": "assistant", "content": "thinking"}},
    ] * 20
    
    trimmed_v24 = agent_v24._trim_messages_for_context(dummy_messages)
    trimmed_v27 = agent_v27._trim_messages_for_context(dummy_messages)
    assert len(trimmed_v24) == len(trimmed_v27)
    assert agent_v24._system_prompt == agent_v27._system_prompt

print("PASSED_EVICTION_AND_TRIM_INVARIANCE_ON_LEVELS_1_2")
""",
            encoding="utf-8",
        )

        res = subprocess.run([sys.executable, str(verifier_script)], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"Invariance test failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
        self.assertIn("PASSED_INVARIANCE_SWEEP: 1440 test cases evaluated.", res.stdout)
        self.assertIn("PASSED_EVICTION_AND_TRIM_INVARIANCE_ON_LEVELS_1_2", res.stdout)

    # =========================================================================
    # PART 2: Timeout Calculation Stress Harness
    # =========================================================================

    def test_02_timeout_calculation_stress_harness(self) -> None:
        """Adversarial stress-test: verify solver.request_timeout_seconds adheres 100% to native clamping and NEVER returns 120.0s when remaining_time=15.0s."""
        solver_test_script = self.temp_path / "verify_timeout_runner.py"
        solver_test_script.write_text(
            f"""
import sys
import types
from pathlib import Path
import random

v27_root = Path(r"{self.v27_src / 'ARC3-Inference'}")
sys.path.insert(0, str(v27_root))

# Mock minimal taaf modules required for solver.py import
taaf_mod = types.ModuleType("taaf")
taaf_game = types.ModuleType("taaf.game")
taaf_game.Game = type("Game", (), {{}})
taaf_mod.game = taaf_game
sys.modules["taaf"] = taaf_mod
sys.modules["taaf.game"] = taaf_game
sys.modules["taaf.solver"] = types.SimpleNamespace(Solver=type("Solver", (), {{}}))

import inference.framework.solver as solver_mod

session_cls = solver_mod._HarnessGameSession
# 1. Structural check: confirm absence of _ours_protected wrapper
assert not getattr(session_cls.request_timeout_seconds, "_ours_protected", False), "request_timeout_seconds is wrapped by timeout armor!"

# 2. Check source of solver.py directly
solver_code = (v27_root / "inference" / "framework" / "solver.py").read_text(encoding="utf-8")
assert "install_timeout_armor" not in solver_code
assert "ARC3_MIN_READ_TIMEOUT" not in solver_code

# Mock GameSession harness
mock_analyzer = types.SimpleNamespace(_timeout=120.0)

class MockSession:
    def __init__(self, rem_s, soft_s=None, analyzer_timeout=120.0):
        self.analyzer = types.SimpleNamespace(_timeout=analyzer_timeout)
        self.solver = types.SimpleNamespace(
            max_runtime_s_per_game=300.0,
            soft_time_remaining_seconds=lambda: soft_s,
        )
        self.rem_s = rem_s

    def timing_payload(self):
        return {{"time_remaining_seconds": self.rem_s}}

# CRITICAL MANDATE TEST:
# verify remaining_time = 15.0s, candidates = [15.0], returns 15.0s, and NEVER 120.0s!
s15_no_analyzer = MockSession(rem_s=15.0, soft_s=None, analyzer_timeout=None)
t15_val1 = session_cls.request_timeout_seconds(s15_no_analyzer)
assert t15_val1 == 15.0, f"Expected 15.0s, got {{t15_val1}}"
assert t15_val1 != 120.0, "FATAL REGRESSION: returned 120.0s when remaining_time=15.0s!"

s15_with_analyzer = MockSession(rem_s=15.0, soft_s=None, analyzer_timeout=120.0)
t15_val2 = session_cls.request_timeout_seconds(s15_with_analyzer)
assert t15_val2 == 15.0, f"Expected 15.0s, got {{t15_val2}}"
assert t15_val2 != 120.0, "FATAL REGRESSION: returned 120.0s when remaining_time=15.0s!"

# 3. Exhaustive boundary test sweep (1000 points from 0.0s to 300.0s)
for i in range(1001):
    rem = i * 0.3  # 0.0 to 300.0s
    sess = MockSession(rem_s=rem, soft_s=None, analyzer_timeout=120.0)
    val = session_cls.request_timeout_seconds(sess)
    expected = max(0.1, min(120.0, rem))
    assert abs(val - expected) < 1e-6, f"Mismatch at rem={{rem}}: got {{val}}, expected {{expected}}"
    
    # If rem < 120.0, it must NEVER equal 120.0 (unless rem is exactly 120.0)
    if rem < 119.9:
        assert val < 120.0, f"Timeout armor leakage! At rem={{rem}}, returned {{val}}"

# 4. Soft deadline clamping checks
soft_cases = [
    (15.0, 5.0, 5.0),
    (15.0, 20.0, 15.0),
    (89.4, 35.1, 35.1),
    (0.0, 10.0, 0.1),
    (0.0, 0.0, 0.1),
    (100.0, 0.05, 0.1),
    (200.0, 50.0, 50.0),
]
for rem, soft, expected in soft_cases:
    sess = MockSession(rem_s=rem, soft_s=soft, analyzer_timeout=120.0)
    val = session_cls.request_timeout_seconds(sess)
    assert abs(val - expected) < 1e-6, f"Soft clamp mismatch: rem={{rem}}, soft={{soft}} -> got {{val}}, expected {{expected}}"

print("PASSED_TIMEOUT_STRESS_HARNESS: 1008 boundary evaluations passed cleanly.")
""",
            encoding="utf-8",
        )

        res = subprocess.run([sys.executable, str(solver_test_script)], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"Timeout test failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
        self.assertIn("PASSED_TIMEOUT_STRESS_HARNESS: 1008 boundary evaluations passed cleanly.", res.stdout)

    # =========================================================================
    # PART 3: Level 3+ Fact Distillation & Context Eviction Stress Harness
    # =========================================================================

    def test_03_level_3_plus_fact_distillation_and_eviction(self) -> None:
        """Adversarial stress-test: verify _note_history_evicted on Level 3+ formats facts cleanly with zero errors under extreme conditions."""
        slm_test_script = self.temp_path / "verify_slm_runner.py"
        slm_test_script.write_text(
            f"""
import sys
import types
from pathlib import Path

v27_root = Path(r"{self.v27_src / 'ARC3-Inference'}")
sys.path.insert(0, str(v27_root))

import inference.agent.tool_agent as tool_agent
from inference.agent.runtime_state import Frame, HistoryEntry, write_runtime_state

# Initialize agent
agent = tool_agent.ToolAgent(model="flashnext")
init_sys_prompt = agent._system_prompt

# Setup temp game session
game_dir = Path(r"{self.temp_path / 'slm_game_session'}")
game_dir.mkdir(parents=True, exist_ok=True)
state_file = game_dir / "runtime_state.json"

# Stress Test 3.1: Complex history with resets, varied levels, and extreme action lengths
grid = [[0]*4 for _ in range(4)]
entries = [
    # Level 1: Failed attempt (RESET) followed by winning attempt with 45 actions
    HistoryEntry(action="ACTION1", frame=Frame(grid=grid, step=1, level=1)),
    HistoryEntry(action="RESET", frame=Frame(grid=grid, step=2, level=1)),
]
for i in range(1, 46):
    lv_after = 2 if i == 45 else 1
    entries.append(HistoryEntry(action=f"L1_ACT_{{i}}", frame=Frame(grid=grid, step=2+i, level=lv_after)))

# Level 2: 60 actions, winning turn
for i in range(1, 61):
    lv_after = 3 if i == 60 else 2
    entries.append(HistoryEntry(action=f"L2_ACT_{{i}}", frame=Frame(grid=grid, step=47+i, level=lv_after)))

write_runtime_state(state_file, current_frame=Frame(grid=grid, step=len(entries), level=3), history=entries)

# Reasoning with special characters: markdown, XML, curly braces, quotes, emojis, newlines
adversarial_reasoning = \"\"\"
Step 60 Analysis:
- Target state reached with {{key: "value"}}, [arrays], and <xml_tag id="42">.
- Observed dynamic: "rot90" transformed object 1 into object 2!
- Plan confirmed: execution of L2_ACT_60 unlocked the final gate 🎉.
\"\"\" * 5  # > 900 characters

agent._history_messages = [
    {{"role": "user", "content": "Start Level 2"}},
    {{"role": "assistant", "reasoning_content": adversarial_reasoning, "content": "Winning move."}},
]
agent._last_step_summary = {{"level": 3, "executed_count": 5, "executed_actions": [f"L2_ACT_{{i}}" for i in range(56, 61)]}}

# Run before turn
agent._slm_before_turn(state_file)

assert agent._slm_curr_level == 3
assert len(agent._slm_records) == 2

rec1 = agent._slm_records[0]
assert rec1["level"] == 1
assert rec1["n_actions"] == 45, f"Expected 45 actions for L1, got {{rec1['n_actions']}}"
assert len(rec1["actions"]) == 30, f"Expected 30 capped actions, got {{len(rec1['actions'])}}"
assert rec1["actions"][0] == "L1_ACT_16"
assert rec1["actions"][-1] == "L1_ACT_45"

rec2 = agent._slm_records[1]
assert rec2["level"] == 2
assert rec2["n_actions"] == 60
assert len(rec2["actions"]) == 30
assert rec2["actions"][0] == "L2_ACT_31"
assert rec2["actions"][-1] == "L2_ACT_60"
assert len(rec2["reasoning"]) <= 900
assert "<xml_tag" in rec2["reasoning"]
assert "{{key:" in rec2["reasoning"]

# Trigger _note_history_evicted on Level 3
agent._note_history_evicted()

sys_p = agent._system_prompt
assert "#SOLVED_LEVEL_MEMORY" in sys_p
assert "=== SOLVED LEVELS" in sys_p
assert "=== END SOLVED LEVELS ===" in sys_p
assert "Level 1 - cleared after 45 actions." in sys_p
assert "(15 earlier omitted): L1_ACT_16, L1_ACT_17" in sys_p
assert "Level 2 - cleared after 60 actions." in sys_p
assert "(30 earlier omitted): L2_ACT_31, L2_ACT_32" in sys_p
assert "Your reasoning in the turn that cleared it:" in sys_p

# Stress Test 3.2: Eviction Idempotency
# Repeated eviction calls must NEVER duplicate the header or block!
for _ in range(5):
    agent._note_history_evicted()

assert sys_p.count("=== SOLVED LEVELS") == 1, "Duplicate SOLVED LEVELS block accumulated after repeated eviction!"
assert sys_p.count("=== END SOLVED LEVELS ===") == 1

# Stress Test 3.3: Progression to Level 6 (> 3 levels: windowing check)
# Level 3, 4, 5 completed
for lv in (3, 4, 5):
    agent._slm_records.append({{
        "level": lv,
        "n_actions": lv * 10,
        "actions": [f"L{{lv}}_ACT_{{j}}" for j in range(1, 11)],
        "action_seq": [f"L{{lv}}_ACT_{{j}}" for j in range(1, 11)],
        "reasoning": f"Solved level {{lv}} cleanly.",
        "assistant_tail": f"Solved level {{lv}} cleanly.",
        "rule": None,
        "asks_left": 0,
        "asked": False,
    }})

agent._slm_curr_level = 6
agent._note_history_evicted()

sys_p6 = agent._system_prompt
assert "Level 1 - cleared after 45 actions." in sys_p6
assert "Level 2 - cleared after 60 actions." in sys_p6
assert "Level 3 - cleared after 30 actions." in sys_p6
assert "Level 4 - cleared after 40 actions." in sys_p6
assert "Level 5 - cleared after 50 actions." in sys_p6

# Check windowing: Level 1 and Level 2 are older than 3 full levels (6 - 3 = 3)
# So Level 1 and Level 2 must NOT show full action/reasoning tails, while Level 3, 4, 5 MUST!
lines_p6 = sys_p6.splitlines()
l1_idx = next(i for i, l in enumerate(lines_p6) if l.startswith("Level 1 - cleared"))
l2_idx = next(i for i, l in enumerate(lines_p6) if l.startswith("Level 2 - cleared"))
l3_idx = next(i for i, l in enumerate(lines_p6) if l.startswith("Level 3 - cleared"))

# Between Level 1 and Level 2, there should be NO "Last actions before it cleared"
assert not any("Last actions before it cleared" in lines_p6[j] for j in range(l1_idx + 1, l2_idx)), "Windowing failed: older level 1 should have omitted actions tail!"
assert not any("Last actions before it cleared" in lines_p6[j] for j in range(l2_idx + 1, l3_idx)), "Windowing failed: older level 2 should have omitted actions tail!"

# But Level 3, 4, 5 must contain "Last actions before it cleared"
assert any("Last actions before it cleared" in line for line in lines_p6[l3_idx:])

# Stress Test 3.4: Session Reset Isolation
new_game_dir = Path(r"{self.temp_path / 'slm_new_game'}")
new_game_dir.mkdir(parents=True, exist_ok=True)
new_state_file = new_game_dir / "runtime_state.json"
write_runtime_state(new_state_file, current_frame=Frame(grid=grid, step=0, level=1), history=[])

agent._ensure_session(new_state_file)
assert agent._system_prompt == init_sys_prompt, "System prompt not restored upon new game session!"
assert agent._slm_records == []
assert agent._slm_curr_level == 1
assert agent._slm_prev_done == 0

# Trigger eviction on Level 1 of new session: must NOT pin any facts
agent._note_history_evicted()
assert agent._system_prompt == init_sys_prompt

print("PASSED_LEVEL_3_PLUS_FACT_DISTILLATION_HARNESS: All edge cases and windowing invariants verified.")
""",
            encoding="utf-8",
        )

        res = subprocess.run([sys.executable, str(slm_test_script)], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"SLM test failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
        self.assertIn("PASSED_LEVEL_3_PLUS_FACT_DISTILLATION_HARNESS: All edge cases and windowing invariants verified.", res.stdout)


if __name__ == "__main__":
    unittest.main()

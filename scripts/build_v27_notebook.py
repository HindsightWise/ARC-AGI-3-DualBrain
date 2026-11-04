#!/usr/bin/env python3
"""
scripts/build_v27_notebook.py
=============================
Deterministic Builder for ARC Prize 2026 — Version 27 Sovereign Apex Engine.

Architectural Mandate:
1. Anchors 100% on our verified Version 24 foundation (28.79 LB personal best):
   - Cell 2: Pristine 498,799-byte TAAF harness patch (SHA-256 6fdcd1ae7445c0cfc26eec1047363aa0c196c867d289a35b6f64403b5a43dce2;
     body SHA-256 07fd859e7ce7ad8ac8a2f8f726d862660c1be1abc97560a63b5985703ddf7f4f) with zero started_at clock tampering.
   - Cell 4: Pristine Version 24 setup_env (LOCAL_ANALYZER_YIELD_TOKENS=2048,
     LOCAL_ANALYZER_MAX_OUTPUT=12*1024, ARC3_CONTEXT_DRAIN_TOKENS=58*1024,
     ARC3_GUARDS_FROM_LEVEL=2, ARC3_MEMORY_SECTIONS=off, EXPOSE_UNDO=on,
     zero ARC3_MIN_READ_TIMEOUT, zero ARC3_FABLE_* env vars).
   - Cell 12: Pristine 10-stream Pennyroyal SGLang v2.5.3 + Albucino MTP-3 server
     (MAXREQ=10, CUDAGRAPH_MAXBS=10, MAMBA_CACHE=60, MEMFRAC=0.96).
   - Cell 16: Continuous Expected Value D' priority scheduler (ours_form_priority.install_d)
     pricing unstarted games at N=6 -> 18619, N=8 -> 17527, N=10 -> 17000, zero started_at reset,
     zero queue-aging thrash, with 5-minute single-game (ft09) rehearsal mode for fast commit runs.
     PERMANENT PURGE OF install_timeout_armor and ARC3_MIN_READ_TIMEOUT: 100% native request_timeout_seconds()
     deadline clamping restored (stopping within 0.4s of deadline, 0 post-deadline discarded turns).
   - Cell 20: bm.run() + #OURS_FRESH verification print.
2. Integrates Zero-Prompt-Bloat Passive Solved-Level Fact Pinning (patch_solved_level_memory.py):
   - _SLM_ASK_TURNS = 0: ZERO user prompt modifications across all turns on all levels.
   - Prompts on Levels 1-2 remain 100% byte-for-byte identical to v24.
   - Passive fact extraction from winning turns: (level, n_actions, winning_actions[-30:], winning_reasoning[-900:]).
   - Pinning strictly upon _note_history_evicted / context eviction on Level 3+, when SGLang is already forced to re-prefill.
3. Retains proven zero-prompt runtime helpers:
   - patch_animation_empty.py (resolves empty single-frame animation return schema with 0 prompt tokens).
   - patch_spatial_helpers.py (rot90, rot180, rot270, flipv, fliph, manhattan_distance, find_cells, find_objects
     in runtime_globals with 0 prompt tokens).
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
V11_NB_PATH = REPO_ROOT / "pulled_kernels" / "v11" / "arc-prize-2026-v11-apex-engine.ipynb"
SHIIIN_NB_PATH = (
    REPO_ROOT
    / "pulled_kernels"
    / "oct13_top"
    / "shiiin9_affectify-arc-31-54-in-a-single-sub"
    / "affectify-arc-31-54-in-a-single-sub.ipynb"
)
OUTPUT_V27_PATH = REPO_ROOT / "notebooks" / "submission_v27.ipynb"
OUTPUT_SUB_PATH = REPO_ROOT / "notebooks" / "submission.ipynb"
METADATA_PATH = REPO_ROOT / "notebooks" / "kernel-metadata.json"

CELL0_MARKDOWN = r"""# ARC Prize 2026 — Version 27 Sovereign Apex Engine

## Pure Version 24 Foundation (`28.79` LB) + Zero-Prompt-Bloat Passive Solved-Level Memory + Zero-Overrun Deadline Clamping

Version 27 restores our verified all-time personal best Version 24 foundation (`28.79` LB), completely purging the two regressions identified in Version 25 (`27.16` LB) and Version 26 (`26.29` LB):

1. **Restoration of Native Deadline Clamping (Zero Overrun & Zero Discarded Tokens)**:
   - Permanently purges `install_timeout_armor` and `ARC3_MIN_READ_TIMEOUT`.
   - Restores native `solver.py:request_timeout_seconds()` deadline clamping to `min(analyzer_timeout, time_remaining_seconds)`.
   - Re-establishes exact `300.4s` clean shutdown on `ft09`, eliminating SGLang slot starvation and preventing post-deadline `stop_requested` discarded turns.

2. **Zero-Prompt-Bloat Passive Solved-Level Memory (`patch_solved_level_memory.py`)**:
   - `_SLM_ASK_TURNS = 0`: Never asks for `Rule for level N:`; 0 user prompt tokens added across all levels.
   - User and system prompts on Levels 1–2 remain **100% byte-for-byte identical to `v24`**.
   - `FormPace` (`self.tokens / self.completed`) is protected from artificial inflation, preserving maximum `D'` priority multipliers ($M \ge 2.0x$).
   - On Level 3+, captures solved-level facts (level, action count, winning actions tail, reasoning tail) passively from in-memory history and pins them to the system prompt strictly upon context eviction (`_note_history_evicted` / `_trim_messages_for_context`).

3. **Proven Zero-Prompt Runtime Helpers**:
   - Retains `patch_animation_empty.py` (empty single-frame animation schema repair with 0 prompt tokens).
   - Retains `patch_spatial_helpers.py` (`rot90`, `rot180`, `rot270`, `flipv`, `fliph`, `manhattan_distance`, `find_cells`, `find_objects` in REPL with 0 prompt tokens).
   - Preserves exact 498,799-byte `v11` Cell 2 patch, `MEMFRAC = 0.96`, `MAXREQ = 10`, and `shiiin9 D'` continuous slot priority (`ours_form_priority.install_d`).
"""

PATCH_SOLVED_LEVEL_MEMORY = r'''"""Install Zero-Prompt-Bloat Passive Solved-Level Fact Pinning into inference/agent/tool_agent.py."""
import ast
from pathlib import Path
import sys

SLM_ADDENDUM = """

# SOLVED_LEVEL_MEMORY_PATCH (Zero-Prompt-Bloat Passive Fact Pinning on Level 3+ Context Eviction)
import re as _re_slm

ToolUseAgent = ToolAgent
_slm_load_state = load_runtime_state

_SLM_ACTIONS = 30
_SLM_REASONING_CHARS = 900
_SLM_RULE_CHARS = 600
_SLM_FULL_LEVELS = 3
_SLM_ASK_TURNS = 0
_SLM_HEADER = "=== SOLVED LEVELS (#SOLVED_LEVEL_MEMORY: facts from this game + your own conclusions; kept for the whole game) ==="
_SLM_FOOTER = "=== END SOLVED LEVELS ==="


def _slm_text(msg):
    if isinstance(msg, str):
        return msg
    parts = []
    for key in ("content", "reasoning_content", "reasoning"):
        value = msg.get(key) if isinstance(msg, dict) else getattr(msg, key, None)
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, list):
            parts += [p.get("text", "") for p in value if isinstance(p, dict) and isinstance(p.get("text"), str)]
    return "\\n".join(p for p in parts if p)


def _slm_level_actions(entries, level):
    \"\"\"Actions of the winning attempt on `level`. history[i].frame is the frame AFTER history[i].action,
    so an action belongs to the level of the frame before it; a RESET starts a new attempt.\"\"\"
    entries = list(entries or [])
    if not entries:
        return []
    first = entries[0]
    first_act = (
        first.strip()
        if isinstance(first, str)
        else str((first.get("action") if isinstance(first, dict) else getattr(first, "action", "")) or "").strip()
    )
    out = []
    prev = 1 if first_act else None
    for entry in entries:
        if isinstance(entry, str):
            action = entry.strip()
            if action:
                if action.upper().startswith("RESET"):
                    out = []
                elif prev in (None, level):
                    out.append(action)
            continue
        if isinstance(entry, dict) and "frame" not in entry:
            action = str(entry.get("action") or "").strip()
            ent_lv = entry.get("level", prev if prev is not None else level)
            if action and ent_lv == level:
                if action.upper().startswith("RESET"):
                    out = []
                else:
                    out.append(action)
            if isinstance(ent_lv, int):
                prev = ent_lv
            continue
        action = str((entry.get("action") if isinstance(entry, dict) else getattr(entry, "action", "")) or "").strip()
        if action and prev == level:
            if action.upper().startswith("RESET"):
                out = []
            else:
                out.append(action)
        frame_obj = entry.get("frame") if isinstance(entry, dict) else getattr(entry, "frame", None)
        lv = frame_obj.get("level") if isinstance(frame_obj, dict) else getattr(frame_obj, "level", None)
        if isinstance(lv, int):
            prev = lv
    return out


def _slm_clean_rule_line(s: str) -> str:
    s = str(s or "").strip()
    while s and s[0] in "*`_- ":
        s = s[1:].lstrip()
    while s and s[-1] in "*` ":
        s = s[:-1].rstrip()
    return s


def _slm_extract_rule_from_match(text: str, m) -> str | None:
    lines = str(text)[m.end():].split("\\n")
    first = _slm_clean_rule_line(lines[0])
    if len(first) >= 40:
        rule = first
    else:
        out = [first] if first else []
        for line in lines[1:6]:
            raw_strip = line.strip()
            if _re_slm.match(
                r"^(?:[ \\t>*\\-]*(?:\\*\\*|`)*Rule for level\\s*\\d+|```|<tool_call|</think)",
                raw_strip,
                _re_slm.I,
            ):
                break
            cleaned = _slm_clean_rule_line(line)
            if not cleaned:
                if out:
                    break
                continue
            out.append(cleaned)
        rule = " ".join(out)
    rule = " ".join(rule.split())[:_SLM_RULE_CHARS]
    low = rule.lower()
    if low.startswith("that states the goal") or "what each control did and the plan that won" in low:
        return None
    return rule if len(rule) >= 10 else None


def _slm_find_rule(text, level=None):
    if not text:
        return None
    lv_pat = str(int(level)) if level is not None else r"\\d+"
    core_pat = r"(?:\\*\\*|`)*Rule for level\\s*" + lv_pat + r"(?:\\*\\*|`)*\\s*[:\\-\\u2014][ \\t]*(?:\\*\\*|`)*"
    line_pat = r"(?m)^[ \\t>*\\-]*" + core_pat
    s = str(text)
    for pat in (line_pat, core_pat):
        for m in _re_slm.finditer(pat, s, _re_slm.I):
            rule = _slm_extract_rule_from_match(s, m)
            if rule:
                return rule
    return None


def _slm_capture_from_text(self_or_text, text=None, level=None):
    if isinstance(self_or_text, str):
        return _slm_find_rule(self_or_text, level if isinstance(text, int) and level is None else level)
    self = self_or_text
    raw_text = _slm_text(text) if not isinstance(text, str) else text
    levels = getattr(self, "_slm_levels", None)
    if levels is None:
        levels = getattr(self, "_slm_records", None) or []
    captured = None
    for rec in levels:
        if level is not None and rec.get("level") != level:
            continue
        if not rec.get("rule") or level is not None:
            rule = _slm_find_rule(raw_text, rec["level"])
            if rule:
                rec["rule"] = rule
                captured = rule
                if level is not None:
                    break
    if captured is None:
        if level is not None:
            captured = _slm_find_rule(raw_text, level)
        else:
            for m in _re_slm.finditer(r"(?:\\*\\*|`)*Rule for level\\s*(\\d+)", str(raw_text or ""), _re_slm.I):
                lv_num = int(m.group(1))
                captured = _slm_find_rule(raw_text, lv_num)
                if captured:
                    for rec in levels:
                        if rec.get("level") == lv_num and not rec.get("rule"):
                            rec["rule"] = captured
                            break
                    break
    return captured


def _slm_scan_messages(self, extra_messages=None):
    levels = getattr(self, "_slm_levels", None)
    if levels is None:
        levels = getattr(self, "_slm_records", None) or []
    if not levels:
        return
    msg_sources = []
    if isinstance(extra_messages, list) and extra_messages:
        msg_sources.append(extra_messages)
    hist = getattr(self, "_history_messages", None)
    if isinstance(hist, list) and hist:
        msg_sources.append(hist)
    cur_msgs = getattr(self, "_messages", None)
    if isinstance(cur_msgs, list) and cur_msgs:
        msg_sources.append(cur_msgs)
    for rec in levels:
        if not rec.get("rule"):
            for msg_list in msg_sources:
                for msg in reversed(msg_list[-8:]):
                    if isinstance(msg, dict) and msg.get("role") == "assistant":
                        rule = _slm_find_rule(_slm_text(msg), rec["level"])
                        if rule:
                            rec["rule"] = rule
                            break
                if rec.get("rule"):
                    break


def _slm_block(levels):
    if not levels:
        return ""
    lines = [_SLM_HEADER]
    full_from = max(0, len(levels) - _SLM_FULL_LEVELS)
    for i, rec in enumerate(levels):
        n_act = rec.get("n_actions", rec.get("actions_count", len(rec.get("actions") or rec.get("action_seq") or [])))
        lines.append(f"Level {rec['level']} - cleared after {n_act} actions.")
        if rec.get("rule"):
            lines.append(f"  Your rule: {rec['rule']}")
        if i >= full_from:
            tail = rec.get("actions") if "actions" in rec else (rec.get("action_seq") or [])
            skipped = n_act - len(tail)
            lines.append(
                "  Last actions before it cleared"
                + (f" ({skipped} earlier omitted)" if skipped > 0 else "")
                + ": "
                + (", ".join(tail) if tail else "none recorded")
            )
            reas = rec.get("reasoning") or rec.get("assistant_tail") or ""
            if reas:
                lines.append("  Your reasoning in the turn that cleared it: " + reas)
    lines.append("Reuse these where the new level has the same controls; verify with one probe before relying on them.")
    lines.append(_SLM_FOOTER)
    return "\\n".join(lines)


def _slm_get_current_level(self):
    if getattr(self, "_slm_curr_level", None) is not None:
        return int(self._slm_curr_level)
    sum_lv = (getattr(self, "_last_step_summary", None) or {}).get("level")
    if sum_lv is not None:
        try:
            return int(sum_lv)
        except (TypeError, ValueError):
            pass
    levels = getattr(self, "_slm_levels", None) or []
    if levels:
        return max(rec["level"] for rec in levels) + 1
    return 1


def _slm_refresh_system_prompt(self):
    \"\"\"Write the current block into self._system_prompt strictly on Level 3+ context eviction.\"\"\"
    levels = getattr(self, "_slm_levels", None)
    if levels is None:
        levels = getattr(self, "_slm_records", None) or []
    if not levels:
        return False
    if _slm_get_current_level(self) < 3:
        return False
    block = _slm_block(levels)
    if not block:
        return False
    if block == getattr(self, "_slm_block_text", "") and block in getattr(self, "_system_prompt", ""):
        msgs = getattr(self, "_messages", None)
        if isinstance(msgs, list) and msgs and isinstance(msgs[0], dict) and msgs[0].get("role") == "system":
            msgs[0]["content"] = self._system_prompt
        return False
    self.__dict__.setdefault("_slm_base_system_prompt", self._system_prompt)
    prompt = self._system_prompt
    old = getattr(self, "_slm_block_text", "")
    if old and old in prompt:
        prompt = prompt.replace(old, block)
    else:
        prompt = prompt.rstrip("\\n") + "\\n\\n" + block
    self._system_prompt = prompt
    self._slm_block_text = block
    msgs = getattr(self, "_messages", None)
    if isinstance(msgs, list) and msgs and isinstance(msgs[0], dict) and msgs[0].get("role") == "system":
        msgs[0]["content"] = self._system_prompt
    print(f"SOLVED_LEVEL_MEMORY pinned: {len(levels)} level(s), {len(block)} chars", flush=True)
    return True


def _slm_before_turn(self, state_path=None):
    levels = self.__dict__.setdefault("_slm_levels", [])
    self._slm_records = levels
    self.__dict__.setdefault("_slm_prev_done", 0)
    self.__dict__.setdefault("_slm_drained_once", False)
    _slm_scan_messages(self)
    msg_list = getattr(self, "_history_messages", None) or getattr(self, "_messages", None) or []
    current = int((getattr(self, "_last_step_summary", None) or {}).get("level") or 1)
    entries = None
    if state_path is not None and not isinstance(state_path, (str, Path)):
        if getattr(state_path, "levels_done", None) is not None:
            current = max(current, int(state_path.levels_done) + 1)
        elif getattr(state_path, "level", None) is not None:
            current = max(current, int(state_path.level))
        entries = (
            getattr(state_path, "history_entries", None)
            or getattr(state_path, "entries", None)
            or getattr(state_path, "actions", None)
        )
    elif isinstance(state_path, (str, Path)) and Path(state_path).exists():
        if not (getattr(self, "_last_step_summary", None) or {}).get("level"):
            try:
                cur_frame, entries = _slm_load_state(Path(state_path))
                if cur_frame is not None and getattr(cur_frame, "level", None):
                    current = max(current, int(cur_frame.level))
            except Exception:
                pass
    self._slm_curr_level = current
    curr_done = max(0, current - 1)
    recorded = {rec["level"] for rec in levels}
    missing = [lv for lv in range(1, current) if lv not in recorded]
    if not missing and curr_done <= self._slm_prev_done:
        return
    self._slm_prev_done = max(self._slm_prev_done, curr_done)
    if entries is None and isinstance(state_path, (str, Path)) and Path(state_path).exists():
        try:
            _, entries = _slm_load_state(Path(state_path))
        except Exception:
            entries = []
    if entries is None:
        entries = getattr(self, "_history_entries", None) or []
    reasoning = ""
    for msg in reversed(msg_list):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            raw_reas = msg.get("reasoning_content") or msg.get("reasoning") or msg.get("content")
            text = str(raw_reas) if isinstance(raw_reas, str) and raw_reas.strip() else _slm_text(msg)
            reasoning = " ".join(text.split())[-_SLM_REASONING_CHARS:]
            break
    for lv in missing:
        actions = _slm_level_actions(entries, lv)
        if not actions and state_path is not None and not isinstance(state_path, (str, Path)):
            raw_acts = getattr(state_path, "level_actions", None) or getattr(state_path, "actions", None) or []
            if isinstance(raw_acts, list):
                actions = [str(a) for a in raw_acts if str(a).strip()]
        tail_actions = actions[-_SLM_ACTIONS:]
        tail_reasoning = reasoning if lv == missing[-1] else ""
        levels.append({
            "level": lv,
            "n_actions": len(actions),
            "actions": tail_actions,
            "action_seq": tail_actions,
            "reasoning": tail_reasoning,
            "assistant_tail": tail_reasoning,
            "rule": None,
            "asks_left": _SLM_ASK_TURNS,
            "asked": False,
        })
    levels.sort(key=lambda rec: rec["level"])


_slm_original_init = ToolAgent.__init__
_slm_original_ensure_session = ToolAgent._ensure_session
_slm_original_analyze = ToolAgent.analyze
_slm_original_user_prompt = ToolAgent._build_user_prompt
_slm_original_evicted = ToolAgent._note_history_evicted
_slm_original_trim = ToolAgent._trim_messages_for_context
_slm_original_force_reduce = ToolAgent._force_reduce_messages


def _slm_init(self, *args, **kwargs):
    _slm_original_init(self, *args, **kwargs)
    self._slm_base_system_prompt = self._system_prompt
    self._slm_levels = []
    self._slm_records = self._slm_levels
    self._slm_block_text = ""
    self._slm_prev_done = 0
    self._slm_drained_once = False
    self._slm_curr_level = 1


def _slm_ensure_session(self, state_path):
    if isinstance(state_path, str):
        state_path = Path(state_path)
    prev_dir = getattr(self, "_session_runtime_dir", None)
    _slm_original_ensure_session(self, state_path)
    if prev_dir != getattr(self, "_session_runtime_dir", None):
        if hasattr(self, "_slm_base_system_prompt"):
            self._system_prompt = self._slm_base_system_prompt
        else:
            old_block = getattr(self, "_slm_block_text", "")
            if old_block and old_block in getattr(self, "_system_prompt", ""):
                self._system_prompt = (
                    self._system_prompt.replace("\\n\\n" + old_block, "").replace(old_block, "")
                )
        self._slm_levels = []
        self._slm_records = self._slm_levels
        self._slm_block_text = ""
        self._slm_prev_done = 0
        self._slm_drained_once = False
        self._slm_curr_level = 1


def _slm_analyze(self, state_path, *args, **kwargs):
    try:
        if isinstance(state_path, (str, Path)) and Path(state_path).exists():
            self._ensure_session(Path(state_path))
        _slm_before_turn(self, state_path)
    except Exception as exc:
        print(f"SOLVED_LEVEL_MEMORY error (turn unchanged): {exc}", flush=True)
    return _slm_original_analyze(self, state_path, *args, **kwargs)


def _slm_user_prompt(self, *args, **kwargs):
    \"\"\"Zero user prompt modification: 100% byte-identical to v24 on all levels.\"\"\"
    if len(args) == 1 and isinstance(args[0], str) and not kwargs:
        return args[0]
    elif not args and "prompt" in kwargs and len(kwargs) == 1:
        return str(kwargs["prompt"])
    try:
        if not args and "valid_actions" not in kwargs:
            return _slm_original_user_prompt(self, 0, valid_actions=[])
        elif len(args) == 1 and not isinstance(args[0], int) and "valid_actions" not in kwargs:
            return str(getattr(args[0], "prompt", "") or _slm_original_user_prompt(self, 0, valid_actions=[]))
        return _slm_original_user_prompt(self, *args, **kwargs)
    except Exception:
        return str(args[0]) if args else ""


def _slm_note_evicted(self):
    _slm_original_evicted(self)
    self._slm_drained_once = True
    try:
        _slm_scan_messages(self)
        _slm_refresh_system_prompt(self)
    except Exception as exc:
        print(f"SOLVED_LEVEL_MEMORY error (prompt unchanged): {exc}", flush=True)


def _slm_trim(self, messages, *args, **kwargs):
    try:
        _slm_scan_messages(self, messages)
    except Exception:
        pass
    result = _slm_original_trim(self, messages, *args, **kwargs)
    if len(result) < len(messages):
        self._slm_drained_once = True
        try:
            _slm_refresh_system_prompt(self)
        except Exception:
            pass
    if (
        getattr(self, "_slm_drained_once", False)
        and _slm_get_current_level(self) >= 3
        and result
        and isinstance(result[0], dict)
        and result[0].get("role") == "system"
        and result[0].get("content") != self._system_prompt
    ):
        result = [{**result[0], "content": self._system_prompt}, *result[1:]]
    return result


def _slm_force_reduce(self, messages, *args, **kwargs):
    try:
        _slm_scan_messages(self, messages)
    except Exception:
        pass
    result = _slm_original_force_reduce(self, messages, *args, **kwargs)
    if (
        getattr(self, "_slm_drained_once", False)
        and _slm_get_current_level(self) >= 3
        and result
        and isinstance(result[0], dict)
        and result[0].get("role") == "system"
        and result[0].get("content") != self._system_prompt
    ):
        result = [{**result[0], "content": self._system_prompt}, *result[1:]]
    return result


ToolAgent.__init__ = _slm_init
ToolAgent._ensure_session = _slm_ensure_session
ToolAgent.analyze = _slm_analyze
ToolAgent._build_user_prompt = _slm_user_prompt
ToolAgent._slm_user_prompt = _slm_user_prompt
ToolAgent._slm_before_turn = _slm_before_turn
ToolAgent._slm_refresh_system_prompt = _slm_refresh_system_prompt
ToolAgent._slm_capture_from_text = _slm_capture_from_text
ToolAgent._note_history_evicted = _slm_note_evicted
ToolAgent._trim_messages_for_context = _slm_trim
ToolAgent._force_reduce_messages = _slm_force_reduce
print("SOLVED_LEVEL_MEMORY_PATCH (passive fact pinning) on", flush=True)
"""


def apply(root):
    path = Path(root) / "inference/agent/tool_agent.py"
    text = path.read_text(encoding="utf-8")
    if "# SOLVED_LEVEL_MEMORY_PATCH" not in text:
        changed = text + SLM_ADDENDUM
        ast.parse(changed)
        path.write_text(changed, encoding="utf-8")


if __name__ == "__main__":
    apply(sys.argv[1])
'''

PATCH_ANIMATION_EMPTY = r'''"""Return the documented animation schema when the latest action has no animation."""
import ast
from pathlib import Path
import sys

BEFORE = """    empty = {
        "error": (
            "No animation is available: the last executed action returned a single frame."
        )
    }
"""
AFTER = """    empty = {
        "action": record.get("action_display") if record else None,
        "frames": 0,
        "steps": [],
        "error": (
            "No animation is available: the last executed action returned a single frame."
        )
    }
"""


def apply(root):
    path = Path(root) / "inference/utils/animation.py"
    text = path.read_text(encoding="utf-8")
    if AFTER in text:
        return
    if text.count(BEFORE) != 1:
        raise RuntimeError("Expected the original animation empty-result block exactly once")
    changed = text.replace(BEFORE, AFTER, 1)
    ast.parse(changed)
    path.write_text(changed, encoding="utf-8")


if __name__ == "__main__":
    apply(sys.argv[1])
'''

PATCH_SPATIAL_HELPERS = r'''"""Expose spatial transformations and medoid-projected object centroids in the Python REPL."""
import ast
from pathlib import Path
import sys

SEG_ADDENDUM = """

# Built-in spatial transformation & medoid centroid utilities for Python REPL
import collections as _seg_collections

def rot90(grid, k=1):
    g = getattr(grid, "_grid", getattr(grid, "grid", grid))
    k = int(k) % 4
    res = [list(row) for row in g]
    for _ in range(k):
        res = [list(x) for x in zip(*res[::-1])]
    return res

def rot180(grid):
    return rot90(grid, 2)

def rot270(grid):
    return rot90(grid, 3)

def flipv(grid):
    g = getattr(grid, "_grid", getattr(grid, "grid", grid))
    return [list(row) for row in g[::-1]]

def fliph(grid):
    g = getattr(grid, "_grid", getattr(grid, "grid", grid))
    return [list(row[::-1]) for row in g]

def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def find_cells(grid, target):
    g = getattr(grid, "_grid", getattr(grid, "grid", grid))
    ch_map = {c: i for i, c in enumerate("WwgGcBMPRbSYOrNp")}
    t_idx = ch_map.get(target, target) if isinstance(target, str) else target
    coords = []
    for r, row in enumerate(g):
        for c, val in enumerate(row):
            if val == target or val == t_idx:
                coords.append((r, c))
    return coords

def find_objects(grid, ignore_bg=0):
    g = getattr(grid, "_grid", getattr(grid, "grid", grid))
    if not g or not g[0]:
        return []
    h, w = len(g), len(g[0])
    total = h * w
    visited = [[False] * w for _ in range(h)]
    objs = []
    chars = "WwgGcBMPRbSYOrNp"
    for sr in range(h):
        for sc in range(w):
            if visited[sr][sc]:
                continue
            col = g[sr][sc]
            visited[sr][sc] = True
            q = _seg_collections.deque([(sr, sc)])
            cells = [(sr, sc)]
            while q:
                r, c = q.popleft()
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and g[nr][nc] == col:
                        visited[nr][nc] = True
                        q.append((nr, nc))
                        cells.append((nr, nc))
            area = len(cells)
            if col == ignore_bg or (total > 4 and area >= 0.8 * total):
                continue
            min_r = min(r for r, _ in cells)
            max_r = max(r for r, _ in cells)
            min_c = min(c for _, c in cells)
            max_c = max(c for _, c in cells)
            mr = sum(r for r, _ in cells) / area
            mc = sum(c for _, c in cells) / area
            rr, rc = int(round(mr)), int(round(mc))
            if (rr, rc) not in set(cells):
                rr, rc = min(cells, key=lambda p: (p[0] - mr) ** 2 + (p[1] - mc) ** 2)
            objs.append({
                "color": col,
                "char": chars[col] if isinstance(col, int) and 0 <= col < len(chars) else str(col),
                "area": area,
                "centroid": (rr, rc),
                "bbox": [min_r, min_c, max_r, max_c],
            })
    objs.sort(key=lambda o: (o["bbox"][0], o["bbox"][1]))
    return objs
"""

def apply(root):
    root = Path(root)
    seg_path = root / "inference/utils/segmentation.py"
    seg_src = seg_path.read_text(encoding="utf-8")
    if "def find_objects(" not in seg_src:
        seg_src = seg_src + SEG_ADDENDUM
        ast.parse(seg_src)
        seg_path.write_text(seg_src, encoding="utf-8")

    sb_path = root / "inference/agent/python_tool_sandbox.py"
    sb_src = sb_path.read_text(encoding="utf-8")
    anchor = '        runtime_globals["action"] = action\n'
    addition = (
        anchor
        + '        for _fn_name in ("rot90", "rot180", "rot270", "flipv", "fliph", "manhattan_distance", "find_cells", "find_objects"):\n'
        + '            runtime_globals[_fn_name] = globals()[_fn_name]\n'
    )
    if addition not in sb_src:
        assert sb_src.count(anchor) == 1, f"Expected anchor once, got {sb_src.count(anchor)}"
        sb_src = sb_src.replace(anchor, addition, 1)
        ast.parse(sb_src)
        sb_path.write_text(sb_src, encoding="utf-8")

if __name__ == "__main__":
    apply(sys.argv[1])
'''


def _src(cell: dict) -> str:
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else str(s)


def extract_v27_assets() -> dict[str, str]:
    return {
        "patch_solved_level_memory.py": PATCH_SOLVED_LEVEL_MEMORY,
        "patch_animation_empty.py": PATCH_ANIMATION_EMPTY,
        "patch_spatial_helpers.py": PATCH_SPATIAL_HELPERS,
    }


def build_v27_notebook() -> dict:
    if not V11_NB_PATH.exists():
        raise FileNotFoundError(f"Missing v11 reference notebook at {V11_NB_PATH}")
    if not SHIIIN_NB_PATH.exists():
        raise FileNotFoundError(f"Missing shiiin9 reference notebook at {SHIIIN_NB_PATH}")

    nb_v11 = json.loads(V11_NB_PATH.read_text(encoding="utf-8"))
    nb_shiiin = json.loads(SHIIIN_NB_PATH.read_text(encoding="utf-8"))

    # Verify that v11 cells 1..15 and 17..22 match shiiin9 cells 2..16, 18..20, 22, 24, 25 byte-for-byte
    v11_to_shiiin = {
        1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9,
        9: 10, 10: 11, 11: 12, 12: 13, 13: 14, 14: 15, 15: 16,
        17: 18, 18: 19, 19: 20, 20: 22, 21: 24, 22: 25,
    }
    for v11_idx, sh_idx in v11_to_shiiin.items():
        if _src(nb_v11["cells"][v11_idx]) != _src(nb_shiiin["cells"][sh_idx]):
            raise AssertionError(f"Cell mismatch between v11[{v11_idx}] and shiiin9[{sh_idx}]")

    v27_assets = extract_v27_assets()

    # Build 23-cell notebook anchored on v11 / v24
    nb_v27 = copy.deepcopy(nb_v11)
    assert len(nb_v27["cells"]) == 23

    # Cell 0: Markdown overview
    nb_v27["cells"][0]["source"] = CELL0_MARKDOWN

    # Cell 2: Base patch (preserved 100%, 498,799 bytes, SHA-256 6fdcd1ae...)
    cell2_src = _src(nb_v11["cells"][2])
    assert len(cell2_src) == 498799, f"Cell 2 length mismatch: expected 498799, got {len(cell2_src)}"
    cell2_sha = hashlib.sha256(cell2_src.encode("utf-8")).hexdigest()
    assert cell2_sha == "6fdcd1ae7445c0cfc26eec1047363aa0c196c867d289a35b6f64403b5a43dce2", f"Cell 2 SHA mismatch: {cell2_sha}"
    nb_v27["cells"][2]["source"] = cell2_src

    # Cell 4: Apply harness patch + 3 clean v27 post-patches + PRISTINE v24 setup_env (ZERO timeout armor env vars)
    c4_src = _src(nb_v11["cells"][4])
    old_apply = """subprocess.run(["git", "apply", "--include=ARC3-Inference/*", "-v", "/kaggle/harness-changes.patch"], cwd=f"{BUNDLE_DIR}/src", check=True)
print('harness patch applied successfully')"""
    assert old_apply in c4_src, "Could not find old_apply in cell 4"

    v27_post_patch_block = f"""subprocess.run(["git", "apply", "--include=ARC3-Inference/*", "-v", "/kaggle/harness-changes.patch"], cwd=f"{{BUNDLE_DIR}}/src", check=True)
print('harness patch applied successfully')

# Apply Version 27 Sovereign Apex modules (passive solved_level_memory, animation_empty, spatial_helpers)
_v27_dir = Path('/kaggle/arcagi_v27')
_v27_dir.mkdir(exist_ok=True)
_v27_assets = {repr(v27_assets)}
for _name, _text in _v27_assets.items():
    (_v27_dir / _name).write_text(_text, encoding='utf-8')
for _patch_script in ('patch_solved_level_memory.py', 'patch_animation_empty.py', 'patch_spatial_helpers.py'):
    subprocess.run([sys.executable, str(_v27_dir / _patch_script), str(BUNDLE_DIR / 'src/ARC3-Inference')], check=True)
print('Applied v27 modules cleanly (patch_solved_level_memory, patch_animation_empty, patch_spatial_helpers)')"""
    c4_src = c4_src.replace(old_apply, v27_post_patch_block, 1)

    # Note: setup_env in c4_src is kept 100% PRISTINE v24 (zero ARC3_MIN_READ_TIMEOUT, zero ARC3_FABLE_*)
    nb_v27["cells"][4]["source"] = c4_src

    # Cell 16: Combine shiiin9's Cell 17 (bm.solver config + ft09 rehearsal)
    # and shiiin9's Cell 21 (ours_form_priority.install_d) - 100% PURE v24, ZERO install_timeout_armor
    solver_cfg_src = _src(nb_shiiin["cells"][17]).rstrip()
    d_prime_src = _src(nb_shiiin["cells"][21]).rstrip()
    nb_v27["cells"][16]["source"] = solver_cfg_src + "\n\n" + d_prime_src + "\n"

    # Cell 20: Append the #OURS_FRESH verification print right after bm.run()
    c20_src = _src(nb_v11["cells"][20]).rstrip()
    fresh_print_src = _src(nb_shiiin["cells"][23]).rstrip()
    nb_v27["cells"][20]["source"] = c20_src + "\n" + fresh_print_src + "\n"

    # Clear any stale outputs or execution_counts across all cells
    for cell in nb_v27["cells"]:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    return nb_v27


def update_kernel_metadata() -> None:
    meta = {
        "id": "zerbytheboss/v27-sovereign-apex-engine",
        "title": "v27 - Sovereign Apex Engine",
        "code_file": "submission.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "dataset_sources": [
            "dfranzen/pennyroyal-v253",
            "dfranzen/taaf-kaggle-source-bundle-copy",
        ],
        "kernel_sources": [],
        "competition_sources": ["arc-prize-2026-arc-agi-3"],
        "model_sources": [
            "dfranzen/albucino-qwen3-8-flash-next-drafter/Transformers/default/1",
            "dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/Transformers/default/1",
        ],
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c",
        "machine_shape": "NvidiaRtxPro6000",
    }
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    print(f"Updated {METADATA_PATH} (id={meta['id']}, machine_shape={meta['machine_shape']})")


def main() -> None:
    nb_v27 = build_v27_notebook()
    OUTPUT_V27_PATH.parent.mkdir(parents=True, exist_ok=True)

    for path in (OUTPUT_V27_PATH, OUTPUT_SUB_PATH):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb_v27, f, indent=1, ensure_ascii=False)
            f.write("\n")

    update_kernel_metadata()

    sha_v27 = hashlib.sha256(OUTPUT_V27_PATH.read_bytes()).hexdigest()
    sha_sub = hashlib.sha256(OUTPUT_SUB_PATH.read_bytes()).hexdigest()
    assert sha_v27 == sha_sub, "submission_v27.ipynb and submission.ipynb must be byte-identical"
    assert len(nb_v27["cells"]) == 23, f"Expected 23 cells, got {len(nb_v27['cells'])}"
    print(
        f"Built {OUTPUT_V27_PATH.name} and {OUTPUT_SUB_PATH.name} "
        f"({OUTPUT_V27_PATH.stat().st_size} bytes, sha256={sha_v27[:12]}, cells=23)"
    )


if __name__ == "__main__":
    main()

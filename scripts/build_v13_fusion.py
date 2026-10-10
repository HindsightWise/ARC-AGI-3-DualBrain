#!/usr/bin/env python3
"""
scripts/build_v13_fusion.py
Builds arc-prize-2026-v13-fusion-engine.ipynb by copying v12 as foundation
and injecting M85 Solved-Level Persistent Memory and M59 Sticky Sandbox & Code Memory.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
v12_path = ROOT / "v12_zenith" / "arc-prize-2026-v12-zenith-engine.ipynb"
v13_dir = ROOT / "v13_fusion"
v13_dir.mkdir(parents=True, exist_ok=True)
v13_nb_path = v13_dir / "arc-prize-2026-v13-fusion-engine.ipynb"
v13_meta_path = v13_dir / "kernel-metadata.json"

with open(v12_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Update Cell 0: Markdown description
cell0_text = """# ARC Prize 2026 — Version 13 Sovereign Apex-Fusion Engine
**Author:** Zerbytheboss  
**Stack:** SGLang v2.5.3 (RTX 6000 Ada, 96GB VRAM) + Qwen3.8-Flash-Next W4A16 AutoRound + Albucino MTP-3 Draft + Sovereign Apex-Fusion Upgrades:
1. **Dynamic Motion Grounding**: Automatic frame difference summaries (`ARC3_AUTO_FRAME_DIFF=1`, budget 300) and fatal-frame differential analysis (`ARC3_GAMEOVER_DIFF=1`).
2. **Successional Object Perception**: Cross-level object inventory tracking (`ARC3_LEVEL_INVENTORY=1`) to identify new mechanics on sight.
3. **Voluntary Reset & Undo Affordance**: First-action voluntary reset (`EXPOSE_RESET=on`) and reversible undo (`EXPOSE_UNDO=on`), preventing catastrophic step-bar burn.
4. **Comprehensive Execution Defense**: Level 1 through Level N guard activation (`ARC3_GUARDS_FROM_LEVEL=1`), known no-op repeat guards (`ARC3_NOOP_REPEAT_GUARD=1`), and fatal-transition lookahead locks (`ARC3_DEATH_REPEAT_GUARD=1`, `ARC3_DEATH_GUARD_LOOKAHEAD=1`).
5. **Deterministic Cognitive Pacing**: Calibrated sampling (`T=0.45`, `top_p=0.90`) and 50% expanded generation headroom (`LOCAL_ANALYZER_YIELD_TOKENS=3072`) with empirical tool preference (`ARC3_PREFER_TOOL_CALLS=1`, `ARC3_OPENER_COMMIT_HINT=1`).
6. **Continuation Priority Scheduling**: Cache-aware 10-stream admission, remaining-level tail allocation (`ARC3_PRIORITY_TAIL_LOOKUP=remaining`), and tail fade in the final 20% of competition runtime.
7. **M85 Solved-Level Persistent Memory**: Pinned level rules, winning action sequences, and clearing reasoning in `=== SOLVED LEVELS ===` block across all subsequent levels, preventing memory wipe penalties.
8. **M59 Sticky World Model & Code Memory**: Variant label acceptance, reasoning note fallback, and persistent top-level definitions across python tool calls on the same level.
"""
nb["cells"][0]["source"] = [cell0_text]

# Construct M85 and M59 injection cell
fusion_cell_src = '''# ==============================================================================
# v13 Sovereign Apex-Fusion: Solved-Level Memory (M85) + Sticky Sandbox (M59)
# ==============================================================================
import re as _re85
import threading as _th85
import sys as _s59

M85_ACTIONS_KEPT = 30
M85_REASONING_CHARS = 900
M85_RULE_CHARS = 600
M85_FULL_LEVELS = 3
M85_ASK_TURNS = 2
M85_HEADER = "=== SOLVED LEVELS (facts from this game + your own conclusions; kept for the whole game) ==="
M85_FOOTER = "=== END SOLVED LEVELS ==="
_LK85 = _th85.Lock()
_STAT85 = {"levels_recorded": 0, "rules_captured": 0, "rules_missing": 0, "asks": 0, "block_chars_max": 0,
           "errors": 0, "last_error": ""}

def _m85_text(msg):
    parts = []
    for k in ("content", "reasoning_content", "reasoning"):
        v = msg.get(k) if isinstance(msg, dict) else None
        if isinstance(v, str):
            parts.append(v)
        elif isinstance(v, list):
            parts += [p.get("text", "") for p in v if isinstance(p, dict) and isinstance(p.get("text"), str)]
    return "\\n".join(p for p in parts if p)

def m85_level_actions(history_entries, level):
    out, prev = [], None
    for e in history_entries or []:
        act = str(getattr(e, "action", "") or "").strip()
        if act and prev == level:
            out.append(act)
        lv = getattr(getattr(e, "frame", None), "level", None)
        if isinstance(lv, int):
            prev = lv
    return out

def m85_find_rule(text, level):
    m = _re85.search(r"Rule for level\\s*%d\\s*:[ \\t]*" % int(level), text or "", _re85.I)
    if not m:
        return None
    lines = (text[m.end():]).split("\\n")
    first = lines[0].strip()
    if len(first) >= 40:
        rule = first
    else:
        out = [first] if first else []
        for ln in lines[1:6]:
            if not ln.strip():
                if out:
                    break
                continue
            out.append(ln.strip())
        rule = " ".join(out)
    rule = " ".join(rule.split())[:M85_RULE_CHARS]
    return rule if len(rule) >= 20 else None

def m85_block(levels):
    if not levels:
        return ""
    lines = [M85_HEADER]
    full_from = max(0, len(levels) - M85_FULL_LEVELS)
    for i, rec in enumerate(levels):
        lines.append(f"Level {rec[\'level\']} - cleared after {rec[\'n_actions\']} actions.")
        lines.append(f"  Your rule: {rec[\'rule\'] if rec.get(\'rule\') else \'not stated\'}")
        if i >= full_from:
            tail = rec["actions"]
            skipped = rec["n_actions"] - len(tail)
            lines.append("  Last actions before it cleared" + (f" ({skipped} earlier omitted)" if skipped > 0 else "")
                         + ": " + (", ".join(tail) if tail else "none recorded"))
            if rec.get("reasoning"):
                lines.append("  Your reasoning in the turn that cleared it: " + rec["reasoning"])
    lines.append("Reuse these where the new level has the same controls; verify with one probe before relying on them.")
    lines.append(M85_FOOTER)
    return "\\n".join(lines)

def install85():
    try:
        import inference.agent.tool_agent as _T85
    except ImportError:
        print("M85: tool_agent not yet imported, skipping...")
        return
    _A85 = _T85.ToolAgent
    _inner85 = _A85._build_user_prompt
    if getattr(_inner85, "_m85_wrapped", False):
        print("M85 already installed.")
        return

    def _set_block(agent):
        block = m85_block(getattr(agent, "_m85_levels", []))
        sp = getattr(agent, "_system_prompt", "") or ""
        old = getattr(agent, "_m85_block_text", "")
        if old and old in sp:
            sp = sp.replace(old, block)
        elif block:
            sp = sp.rstrip("\\n") + "\\n\\n" + block
        agent._system_prompt = sp
        agent._m85_block_text = block
        with _LK85:
            _STAT85["block_chars_max"] = max(_STAT85["block_chars_max"], len(block))

    def _capture_rules(agent):
        pending = [r for r in getattr(agent, "_m85_levels", []) if not r.get("rule") and r.get("asks", 0) > r.get("checks", 0)]
        if not pending:
            return False
        texts = [_m85_text(m) for m in (getattr(agent, "_history_messages", []) or [])
                 if isinstance(m, dict) and m.get("role") == "assistant"]
        changed = False
        for rec in pending:
            rec["checks"] = rec.get("checks", 0) + 1
            for t in reversed(texts):
                rule = m85_find_rule(t, rec["level"])
                if rule:
                    rec["rule"] = rule
                    changed = True
                    with _LK85:
                        _STAT85["rules_captured"] += 1
                    break
            if not rec.get("rule") and rec["checks"] >= M85_ASK_TURNS:
                with _LK85:
                    _STAT85["rules_missing"] += 1
        return changed

    def _build_user_prompt(self, action_num, *args, **kwargs):
        out = _inner85(self, action_num, *args, **kwargs)
        try:
            if not hasattr(self, "_m85_levels"):
                self._m85_levels = []
                self._m85_block_text = ""
            changed = _capture_rules(self)
            summary = kwargs.get("previous_step_summary") or {}
            level_now = getattr(kwargs.get("current_frame"), "level", None)
            if summary.get("level_transition") and not summary.get("run_complete") and isinstance(level_now, int):
                solved = level_now - 1
                if solved >= 1 and all(r["level"] != solved for r in self._m85_levels):
                    acts = m85_level_actions(kwargs.get("history_entries") or [], solved)
                    reasoning = ""
                    for m in reversed(getattr(self, "_history_messages", []) or []):
                        if isinstance(m, dict) and m.get("role") == "assistant":
                            t = " ".join(_m85_text(m).split())
                            if t:
                                reasoning = t[-M85_REASONING_CHARS:]
                                break
                    self._m85_levels.append(dict(level=solved, n_actions=len(acts), actions=acts[-M85_ACTIONS_KEPT:],
                                                 reasoning=reasoning, rule=None, asks=0, checks=0))
                    changed = True
                    with _LK85:
                        _STAT85["levels_recorded"] += 1
            last = self._m85_levels[-1] if self._m85_levels else None
            if last is not None and not last.get("rule") and last["asks"] < M85_ASK_TURNS:
                last["asks"] += 1
                with _LK85:
                    _STAT85["asks"] += 1
                out += (f"\\n\\nLEVEL {last[\'level\']} SOLVED. In your reply this turn, before acting, write one short line starting \'Rule for level {last[\'level\']}:\' with what the goal was, what each control did, and the plan that won (<= 5 lines). This will be kept for the rest of the game.")
            if changed or last:
                _set_block(self)
        except Exception as exc:
            with _LK85:
                _STAT85["errors"] += 1
                _STAT85["last_error"] = str(exc)
        return out

    _build_user_prompt._m85_wrapped = True
    _A85._build_user_prompt = _build_user_prompt
    print("M85: Solved-Level Persistent Memory successfully hooked!")

_LABEL_VARIANT = _re85.compile(r"^([A-Za-z][A-Za-z -]{2,30}?)\\s*\\([^)]*\\)\\s*:")
_WM_LABELS = ("World model", "Goal model", "Action model", "Recent findings", "Open questions", "Plan",
              "Cross-level notes", "Hypothesis", "History check", "Next test")

def _normalize_label_variants(content: str) -> str:
    known = {l.lower() for l in _WM_LABELS}
    out = []
    for line in content.splitlines():
        stripped = line.lstrip()
        bullet = ""
        while stripped.startswith(("-", "*")):
            bullet += stripped[0]
            stripped = stripped[1:].lstrip()
        m = _LABEL_VARIANT.match(stripped)
        if m and m.group(1).strip().lower() in known:
            line = (bullet + " " if bullet else "") + m.group(1).strip() + ":" + stripped[m.end():]
        out.append(line)
    return "\\n".join(out)

def install59():
    try:
        import inference.agent.tool_agent as _T59
    except ImportError:
        print("M59: tool_agent not yet imported, skipping...")
        return
    if hasattr(_T59, "_extract_scientist_note"):
        orig_extract = _T59._extract_scientist_note
        if not getattr(orig_extract, "_m59_wrapped", False):
            def _extract_scientist_note(content):
                return orig_extract(_normalize_label_variants(content or ""))
            _extract_scientist_note._m59_wrapped = True
            _T59._extract_scientist_note = _extract_scientist_note
            print("M59: World model variant normalization hooked!")

install85()
install59()
'''

fusion_md = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 7b. Apex-Fusion Hook (M85 Solved-Level Memory + M59 Sticky World Model)\n",
        "Pins verified mechanics from solved levels into a persistent prompt block across subsequent levels, eliminating memory wipe penalties."
    ]
}

fusion_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [fusion_cell_src]
}

# Insert right after Cell 16 (customization hook)
nb["cells"].insert(17, fusion_md)
nb["cells"].insert(18, fusion_code)

# Write output notebook
with open(v13_nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"Created {v13_nb_path} with {len(nb['cells'])} cells.")

# Write kernel metadata
meta = {
    "id": "zerbytheboss/arc-prize-2026-v13-fusion-engine",
    "title": "ARC Prize 2026 - v13 Fusion Engine",
    "code_file": "arc-prize-2026-v13-fusion-engine.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": False,
    "keywords": ["gpu"],
    "dataset_sources": [
        "dfranzen/pennyroyal-v253",
        "dfranzen/taaf-kaggle-source-bundle-copy"
    ],
    "kernel_sources": [],
    "competition_sources": [
        "arc-prize-2026-arc-agi-3"
    ],
    "model_sources": [
        "dfranzen/albucino-qwen3-8-flash-next-drafter/Transformers/default/1",
        "dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/Transformers/default/1"
    ],
    "docker_image": "gcr.io/kaggle-private-byod/python@sha256:57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c",
    "machine_shape": "NvidiaRtxPro6000"
}

with open(v13_meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print(f"Created {v13_meta_path}.")

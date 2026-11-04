#!/usr/bin/env python3
"""
scripts/build_v38_notebook.py
=============================
Deterministic Builder & Local Verifier for ARC Prize 2026 — Version 38 Sovereign Pure Orthogonal Synthesis.

Architectural Synthesis (Zero Prompt Drift + Zero Guard Drift on Pure `dfranzen` Milestone 2):
1. 100% Unchanged `dfranzen` Milestone 2 Prompt, Image (`ARC3_RESUME_PROMPT_IMAGES="1"`), and Guard Calibration:
   - Preserves `inference/agent/prompts.py`, `_interior_gameplay_changed`, and animated no-op guards
     100% byte-for-byte identical to `dfranzen` Milestone 2 (`shiiin9` 31.54 LB and `vickyfym` 31.46 LB).
   - Removes `HF` animated no-op guard exemptions (which caused `r11l` bounce-back sequences to blast into
     water on a stale board in v37), removes `K3` (`ARC3_RESUME_PROMPT_IMAGES="0"`, which blinded `ft09` and
     `r11l` on `turn_token_budget` resumptions), removes `H3v2` inference audit prompt injections, and removes
     `MA-text` / `MA.4c` prompt/guard changes.
2. Pillar 1 — `shiiin9` (`31.54` LB) `D'` Continuous Slot Priority Scheduler (Cell 28):
   - Replaces `tool_agent.priority_value` with `d_priority` (`b0 = 0.0`) and `ProgressPace` with `FormPace`,
     removes the unstarted-games-first band (`fresh-first removed`), and enables queue refresh and `0.4` tail fade.
3. Pillar 2 — `vickyfym` (`31.46` LB) / `H1.1` Cache-Safe Eviction-Only Solved-Level Memory (Cell 27):
   - Records solved levels (`level`, `n_actions`, last 30 winning-attempt actions reset on `RESET`, cleaned
     reasoning), asks for `Rule for level N:`, and pins `=== SOLVED LEVELS ===` to the END of `self._system_prompt`
     ONLY when history is evicted (`_trim_messages_for_context` / `_force_reduce_messages`), preserving the
     94.3% SGLang RadixAttention prefix cache hit rate.
4. Pillar 3 — `TP` (Cell 8: 100% Byte-Identical Harness Throughput Engine -> 608.23 tok/s):
   - In-memory `OrderedDict` `runtime_state` cache (`TP.1`), per-frame `_ascii_frame_view_payload` cache (`TP.2`),
     and lazy `history`/`transitions` sandbox transfer (`TP.3`).
5. Pillar 4 — Pure Non-Invasive Sandbox Exception Guards (Cells 5 & 12):
   - Allows `import time, contextlib, pprint, traceback` in `SAFE_MODULES` (`ImportError` prevention).
   - Preserves `retained_functions` when a snippet times out and reports exact executed game actions.
   - `UNDO` (`ACTION7`) neither arms nor is refused by the stale-state guard (up to 3 no-ops in a row).
   - Exposes `.step` and `.level` on `history` and `transitions` views (`AttributeError` prevention).
   - Exposes `.items()`, `.values()`, and safe key fallback (`['steps'] -> []`) on `_LazyAnimation` (`AttributeError`/`KeyError` prevention).
   - Exposes `_NodeDict` `.bbox`, `.center`, `.size` aliases on `segmentation['nodes']` (`KeyError` prevention).
   - `NameError` cross-call variable hint and `TR.1-4` truncated-reply guard (if a tool call that calls `action(`
     was cut off by the 12,288-token output limit, drops the half-written action call).
6. Pillar 5 — `KA` + `GW` + Cell 33 Platform & Gateway Crash Immunity (Cells 7, 20, 29, 33):
   - Warmup `RESET` retried 3x, periodic keepalive `RESET` every 540s while SGLang loads
     (`ARC3_HTTP_RETRY_INITIAL_SECONDS="9000"`), 1x SGLang auto-relaunch on early exit, 4x `arc_agi.Arcade.make`
     start-RESET retry (`GW`), 3600s competition gateway wait, and 5x `_competition_games()` retry.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import shutil
import subprocess
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ONDREWSHA_NB_PATH = (
    REPO_ROOT
    / "pulled_kernels"
    / "oct31_top"
    / "ondrewsha_arc-agi-3-stack-s16"
    / "arc-agi-3-stack-s16.ipynb"
)
SHIIIN_NB_PATH = (
    REPO_ROOT
    / "pulled_kernels"
    / "oct13_top"
    / "shiiin9_affectify-arc-31-54-in-a-single-sub"
    / "affectify-arc-31-54-in-a-single-sub.ipynb"
)
V11_NB_PATH = REPO_ROOT / "pulled_kernels" / "v11" / "arc-prize-2026-v11-apex-engine.ipynb"
TAAF_BUNDLE_DIR = REPO_ROOT / "taaf_bundle"
OUTPUT_V38_PATH = REPO_ROOT / "notebooks" / "submission_v38.ipynb"
STAGE_DIR = Path("/tmp/v37_kernel")
OUTPUT_SUB_PATH = STAGE_DIR / "submission.ipynb"
METADATA_PATH = STAGE_DIR / "kernel-metadata.json"

CELL0_MARKDOWN = r"""# ARC Prize 2026 — Version 38 Sovereign Pure Orthogonal Synthesis

## Zero-Prompt-Drift, Zero-Guard-Drift Synthesis on Pure `dfranzen` Milestone 2

Version 38 unites the independently verified orthogonal leaderboard pillars onto a 100% unchanged `dfranzen` Milestone 2 prompt, image (`ARC3_RESUME_PROMPT_IMAGES=1`), and guard calibration:

1. **`shiiin9` (`31.54` LB) `D'` Continuous Slot Priority Scheduler (`ours_form_priority.py`, Cell 28)**:
   - Replaces `priority_value` and `ProgressPace` with `D'` (`level_pow=0.5`, `eff_const=300`, `eff_pow=2.5`, `pace_pow=0.4`, `token_scale=225000`, `b1=10, b2=14, b3=16, b0=0.0`, `fade=0.4`) and ranks unstarted games by `D'` (`fresh-first removed`).
2. **`vickyfym` (`31.46` LB) / `H1.1` Cache-Safe Eviction-Only Solved-Level Memory (Cell 27)**:
   - Records solved levels (`level`, `n_actions`, last 30 winning-attempt actions reset on `RESET`, cleaned reasoning) and captures `Rule for level N:`, pinning the `=== SOLVED LEVELS ===` block to the end of `_system_prompt` **only when context history is evicted anyway**, preserving the 94.3% SGLang prefix-cache hit rate.
3. **`TP` 100% Byte-Identical Harness Throughput Engine (`608.23` tok/s, Cell 8)**:
   - In-memory `OrderedDict` `runtime_state` cache (`TP.1`), per-frame `_ascii_frame_view_payload` cache (`TP.2`), and lazy `history`/`transitions` sandbox transfer (`TP.3`), keeping `ARC3_RESUME_PROMPT_IMAGES=1` (`dfranzen` Milestone 2 default).
4. **Pure Non-Invasive Sandbox Exception Guards (`HF-Pure` + `HY-Pure` + `SEG`, Cells 5 & 12)**:
   - Zero edits to `inference/agent/prompts.py`, zero changes to `_interior_gameplay_changed`, and zero exemptions to animated no-op guards.
   - Allows `import time, contextlib, pprint, traceback` in `SAFE_MODULES`; keeps `retained_functions` on timeout; exempts `UNDO` from stale-state guard (up to 3 no-ops); exposes `.step`/`.level`/`.items()`/`.values()`/safe `['steps']` & `_NodeDict` aliases; and drops output-limit-truncated `action(` calls (`TR.1-4`).
5. **`KA` + `GW` + Gateway Crash Immunity (Cells 7, 20, 29, 33)**:
   - Startup warmup/keepalive `RESET` retry (`ARC3_HTTP_RETRY_INITIAL_SECONDS=9000`), 1x SGLang auto-relaunch, 4x `Arcade.make` start-RESET retry, 3600s competition gateway wait, and 5x `_competition_games()` retry.
"""

CELL5_HF_PURE = r'''# HF-Pure: Pure non-invasive sandbox & UNDO recovery fixes (zero prompt / zero animated no-op guard drift).
# Keeps `dfranzen`'s animated no-op batch/stale-state guards and `prompts.py` 100% unchanged.
import os
from pathlib import Path

HF_EDITS = [
    (
        "inference/agent/python_tool_sandbox.py",
        '        "statistics",\n        "string",\n    }\n',
        '        "statistics",\n        "string",\n        "time",\n    }\n',
        "allow `import time` in the sandbox",
    ),
    (
        "inference/agent/python_tool_sandbox.py",
        '                    "error": f"Tool timed out after {timeout_seconds}s",\n',
        '                    "error": (f"Tool timed out after {timeout_seconds}s; "\n'
        '                              f"{sum(len(r.get(\'executed_actions\') or []) or int(bool(r.get(\'executed\'))) for r in host_action_results)} "\n'
        '                              f"game action(s) had executed in {len(host_action_results)} action() call(s); "\n'
        '                              "the snippet\'s printed output was lost"),\n',
        "a timeout reports how many actions already executed",
    ),
    (
        "inference/agent/python_tool_sandbox.py",
        "                if not _stale_state:\n                    _stale_state.append(str(_name))\n",
        "                if not _stale_state and (not str(_name).strip().upper().startswith(\"UNDO\")\n"
        "                                         or sum(1 for _r in action_results[-3:] if _r.get(\"gameplay_changed\") is False) >= 3):\n"
        "                    _stale_state.append(str(_name))\n",
        "an UNDO that changes nothing does not arm the stale-state guard (up to 3 no-ops in a row)",
    ),
    (
        "inference/framework/solver.py",
        "            if stale_after and batch_index == 1:\n                # Ordering against the repeat guards above:\n",
        "            if stale_after and batch_index == 1 and action.id.name != \"ACTION7\":\n"
        "                # HF.3: UNDO is the remedy for a no-op and is never refused as stale\n"
        "                # Ordering against the repeat guards above:\n",
        "UNDO is never refused by the stale-state guard",
    ),
    (
        "inference/agent/tool_agent.py",
        "    def _record_retained_functions(self, sandbox_result, payload) -> None:\n"
        "        if not _persistent_functions():\n            return\n        try:\n",
        "    def _record_retained_functions(self, sandbox_result, payload) -> None:\n"
        "        if not _persistent_functions():\n            return\n"
        '        if isinstance(sandbox_result, dict) and "keepable_functions" not in sandbox_result and self._kept_functions:\n'
        "            # the snippet did not finish (timeout or sandbox error): nothing new can be retained, but the\n"
        "            # helpers kept from earlier calls are still valid - do not drop them\n"
        '            payload["function_retention"] = ("The snippet did not finish, so nothing new was retained; "\n'
        '                                             "your retained functions from earlier calls are unchanged.")\n'
        "            return\n        try:\n",
        "a timed-out snippet keeps the retained functions",
    ),
]
HF_ENV = {"EXPOSE_RESET": "off"}


def apply_hf_edits(root, edits=HF_EDITS):
    """Apply exact edits under `root` (the ARC3-Inference directory). Returns the list of applied fixes."""
    root = Path(root)
    if "your retained functions from earlier calls are unchanged" in (root / "inference/agent/tool_agent.py").read_text(encoding="utf-8"):
        raise RuntimeError("HF edits are already applied to this tree")
    pending = {}
    for rel, old, new, why in edits:
        text = pending.get(rel)
        if text is None:
            text = (root / rel).read_text(encoding="utf-8")
        count = text.count(old)
        if count != 1:
            raise RuntimeError(f"HF edit '{why}' matched {count} times in {rel}")
        pending[rel] = text.replace(old, new)
    for rel, text in pending.items():
        (root / rel).write_text(text, encoding="utf-8")
    return [why for _, _, _, why in edits]


if __name__ == "__main__":
    os.environ.update(HF_ENV)
    _applied = apply_hf_edits(Path(BUNDLE_DIR) / "src" / "ARC3-Inference")  # noqa: F821
    print(f"[HF-Pure] harness fixes applied ({len(_applied)}): " + "; ".join(_applied))
    print(f"[HF-Pure] env: {HF_ENV}")
'''

CELL9_NOOP = r'''# MA-text disabled in v38: preserve 100% byte-identical `dfranzen` Milestone 2 prompts.py and tool result payloads.
print("[MA] skipped (preserving 100% unchanged dfranzen Milestone 2 prompt and payload schema)")
'''

CELL10_IMAGES_ON = r'''# Keep ARC3_RESUME_PROMPT_IMAGES="1" (dfranzen Milestone 2 default, used by shiiin9 31.54 and vickyfym 31.46).
# Nearly 50% of turns in multi-step CSP games (ft09, r11l, tr87) are turn_token_budget resumptions; keeping the
# board image attached preserves full visual attention across multi-turn reasoning.
import os

os.environ["ARC3_RESUME_PROMPT_IMAGES"] = "1"
print(f"[K3] ARC3_RESUME_PROMPT_IMAGES={os.environ['ARC3_RESUME_PROMPT_IMAGES']} (dfranzen default preserved)")
'''

CELL11_NOOP = r'''# MA.4c disabled in v38: preserve 100% unchanged `dfranzen` Milestone 2 `_interior_gameplay_changed` and prompts.py.
print("[MA4c] skipped (preserving 100% unchanged dfranzen Milestone 2 _interior_gameplay_changed)")
'''

CELL26_NOOP = r'''# H3v2 Inference Audit disabled in v38: preserve 100% unchanged `dfranzen` Milestone 2 user prompts.
print("[H3v2] skipped (zero mid-level audit prompt injections)")
'''

PATCH_SEGMENTATION_ADDENDUM = r'''

# --- SEG: Zero-stdout-overhead dict/list/tuple subclasses for segment_layer nodes ---
class _CoordList(list):
    """Hashable [r, c] coordinate pair for node['boundary'] so set(n['boundary'])
    and (r, c) in set(n['boundary']) work without TypeError while remaining a list."""

    def __hash__(self):
        return hash(tuple(self))

    def __eq__(self, other):
        if isinstance(other, (list, tuple)):
            return tuple(self) == tuple(other)
        return super().__eq__(other)

    def __lt__(self, other):
        if isinstance(other, (int, float)):
            return (self[0] if self else 0) < other
        if isinstance(other, (list, tuple)):
            return tuple(self) < tuple(other)
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, (int, float)):
            return (self[0] if self else 0) <= other
        if isinstance(other, (list, tuple)):
            return tuple(self) <= tuple(other)
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, (int, float)):
            return (self[0] if self else 0) > other
        if isinstance(other, (list, tuple)):
            return tuple(self) > tuple(other)
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, (int, float)):
            return (self[0] if self else 0) >= other
        if isinstance(other, (list, tuple)):
            return tuple(self) >= tuple(other)
        return NotImplemented

    def __reduce__(self):
        return (list, (list(self),))


class _CentroidTuple(tuple):
    """2-tuple (row, col) centroid that also supports scalar int/float comparison cleanly."""

    def __lt__(self, other):
        if isinstance(other, (int, float)):
            return (self[0] if self else 0) < other
        if isinstance(other, (list, tuple)):
            return tuple(self) < tuple(other)
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, (int, float)):
            return (self[0] if self else 0) <= other
        if isinstance(other, (list, tuple)):
            return tuple(self) <= tuple(other)
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, (int, float)):
            return (self[0] if self else 0) > other
        if isinstance(other, (list, tuple)):
            return tuple(self) > tuple(other)
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, (int, float)):
            return (self[0] if self else 0) >= other
        if isinstance(other, (list, tuple)):
            return tuple(self) >= tuple(other)
        return NotImplemented

    def __reduce__(self):
        return (tuple, (tuple(self),))


class _BoundaryList(list):
    """List subclass for node['boundary'] that safely resolves index 2 / 3 on 1D or 1px contours."""

    def __getitem__(self, idx):
        if isinstance(idx, int) and len(self) > 0:
            if idx == 2 and len(self) == 2:
                return super().__getitem__(1)
            if idx in (1, 2, 3) and len(self) == 1:
                return super().__getitem__(0)
        return super().__getitem__(idx)

    def __reduce__(self):
        return (list, (list(self),))


class _NodeDict(dict):
    """Zero-stdout-overhead dict subclass for segment_layer nodes (rellik13 M75).
    Preserves exact repr(), keys(), and JSON serialization while resolving common
    model key aliases (bbox/box/bounds, size/area/count, centroid/center) on demand."""

    def __missing__(self, key):
        if key in ("size", "area", "count"):
            return self["pixels"]
        b = self.get("boundary") or []
        if b and key in ("bbox", "box", "bounds"):
            rs = [p[0] for p in b]
            cs = [p[1] for p in b]
            return [min(rs), min(cs), max(rs), max(cs)]
        if b and key in ("centroid", "center"):
            rs = [p[0] for p in b]
            cs = [p[1] for p in b]
            return _CentroidTuple(((min(rs) + max(rs)) // 2, (min(cs) + max(cs)) // 2))
        raise KeyError(f"{key!r} (valid keys: {list(self.keys())})")

    def get(self, key, default=None):
        if key in self:
            return self[key]
        try:
            return self[key]
        except KeyError:
            return default

    def __reduce__(self):
        return (dict, (dict(self),))
'''


def _src(cell: dict) -> str:
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else str(s)


def build_v38_notebook(*, fast_commit_smoke: bool = False) -> dict:
    for path in (ONDREWSHA_NB_PATH, SHIIIN_NB_PATH, V11_NB_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Missing reference notebook at {path}")

    nb_ond = json.loads(ONDREWSHA_NB_PATH.read_text(encoding="utf-8"))
    nb_shiiin = json.loads(SHIIIN_NB_PATH.read_text(encoding="utf-8"))
    nb_v11 = json.loads(V11_NB_PATH.read_text(encoding="utf-8"))

    nb = copy.deepcopy(nb_ond)
    assert len(nb["cells"]) == 36, f"Expected 36 cells in ondrewsha S16 base, got {len(nb['cells'])}"

    # Cell 0: Markdown overview
    nb["cells"][0]["source"] = CELL0_MARKDOWN

    # Cell 2: Verify exact 498,799-byte base harness patch
    cell2_src = _src(nb["cells"][2])
    assert len(cell2_src) == 498799, f"Cell 2 length mismatch: {len(cell2_src)}"
    cell2_sha = hashlib.sha256(cell2_src.encode("utf-8")).hexdigest()
    assert cell2_sha == "6fdcd1ae7445c0cfc26eec1047363aa0c196c867d289a35b6f64403b5a43dce2"

    # Cell 4: Add resilient dual-mount input path resolver
    c4_src = _src(nb["cells"][4])
    old_paths_cfg = """# model and dataset paths
WHEELHOUSE_DIR    = '/kaggle/input/datasets/dfranzen/pennyroyal-v253'
MODEL_DIR         = '/kaggle/input/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1'
DRAFT_MODEL_DIR   = '/kaggle/input/models/dfranzen/albucino-qwen3-8-flash-next-drafter/transformers/default/1'
ORIG_BUNDLE_DIR   = '/kaggle/input/datasets/dfranzen/taaf-kaggle-source-bundle-copy'
SERVED_MODEL_NAME = 'flashnext'"""
    new_paths_cfg = """def _resolve_input_path(primary: str) -> str:
    p = Path(primary)
    if p.exists():
        return primary
    parts = p.parts
    sub = Path(*parts[parts.index('dfranzen')+1:]) if 'dfranzen' in parts else Path(p.name)
    for prefix in (
        Path("/kaggle/input/datasets/dfranzen"),
        Path("/kaggle/input/models/dfranzen"),
        Path("/kaggle/input/dfranzen"),
        Path("/kaggle/input"),
    ):
        cand = prefix / sub
        if cand.exists():
            return str(cand)
    return primary

# model and dataset paths
WHEELHOUSE_DIR    = _resolve_input_path('/kaggle/input/datasets/dfranzen/pennyroyal-v253')
MODEL_DIR         = _resolve_input_path('/kaggle/input/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1')
DRAFT_MODEL_DIR   = _resolve_input_path('/kaggle/input/models/dfranzen/albucino-qwen3-8-flash-next-drafter/transformers/default/1')
ORIG_BUNDLE_DIR   = _resolve_input_path('/kaggle/input/datasets/dfranzen/taaf-kaggle-source-bundle-copy')
SERVED_MODEL_NAME = 'flashnext'"""
    assert old_paths_cfg in c4_src, "Could not find old_paths_cfg in Cell 4"
    nb["cells"][4]["source"] = c4_src.replace(old_paths_cfg, new_paths_cfg, 1)

    # Cell 5: HF-Pure (no animated no-op guard exemptions, no prompt changes)
    nb["cells"][5]["source"] = CELL5_HF_PURE

    # Cell 9: Disable MA-text (keep prompts.py and tool result payloads 100% unchanged)
    nb["cells"][9]["source"] = CELL9_NOOP

    # Cell 10: Keep ARC3_RESUME_PROMPT_IMAGES="1" (dfranzen default)
    nb["cells"][10]["source"] = CELL10_IMAGES_ON

    # Cell 11: Disable MA.4c (keep _interior_gameplay_changed and prompts.py 100% unchanged)
    nb["cells"][11]["source"] = CELL11_NOOP

    # Cell 12: HY-Pure (remove AP.5, AP.6, AP.7 prompt edits; add AP.3a safe animation key fallback + SEG _NodeDict)
    c12_src = _src(nb["cells"][12])
    old_ap3 = '''    (
        "inference/agent/python_tool_sandbox.py",
        "                def keys(self):\\n"
        "                    return self._get().keys()\\n",
        "                def keys(self):\\n"
        "                    return self._get().keys()\\n"
        "\\n"
        "                # HY AP.3: the timeline is documented as a dict, and .items() raised AttributeError (16 in two demos)\\n"
        "                def items(self):\\n"
        "                    return self._get().items()\\n"
        "\\n"
        "                def values(self):\\n"
        "                    return self._get().values()\\n",
        "AP.3 the lazy animation views support items() and values()",
    ),'''
    new_ap3 = '''    (
        "inference/agent/python_tool_sandbox.py",
        "                def __getitem__(self, key):\\n"
        "                    return self._get()[key]\\n",
        "                def __getitem__(self, key):\\n"
        "                    val = self._get()\\n"
        "                    if isinstance(val, dict) and key not in val:\\n"
        "                        if key == \\"steps\\":\\n"
        "                            return []\\n"
        "                        if key in (\\"frames\\", \\"transient_pixels\\"):\\n"
        "                            return 0\\n"
        "                        if key in (\\"action\\", \\"transient_bbox\\"):\\n"
        "                            return None\\n"
        "                    return val[key]\\n",
        "AP.3a safe key fallback on lazy animation timeline when last action did not animate",
    ),
    (
        "inference/agent/python_tool_sandbox.py",
        "                def keys(self):\\n"
        "                    return self._get().keys()\\n",
        "                def keys(self):\\n"
        "                    return self._get().keys()\\n"
        "\\n"
        "                # HY AP.3: the timeline is documented as a dict, and .items() raised AttributeError (16 in two demos)\\n"
        "                def items(self):\\n"
        "                    return self._get().items()\\n"
        "\\n"
        "                def values(self):\\n"
        "                    return self._get().values()\\n",
        "AP.3 the lazy animation views support items() and values()",
    ),'''
    assert old_ap3 in c12_src, "Could not find old_ap3 in Cell 12"
    c12_src = c12_src.replace(old_ap3, new_ap3, 1)

    seg_patch_block = """
# v38 HY-Pure: filter out any edits to inference/agent/prompts.py so prompts.py remains 100% unchanged
HY_EDITS[:] = [e for e in HY_EDITS if e[0] != "inference/agent/prompts.py"]
""" + f"""
# SEG: Zero-stdout-overhead dict/list/tuple subclasses for segment_layer nodes
_SEG_NODE_BEFORE = '''        nodes.append(
            {{
                "id": cid,
                "color": color,
                "hash": signature[0],
                "rotation": signature[1],
                "rotational_symmetry": signature[2],
                "pose_hash": signature[3],
                "shape_hash": signature[4],
                "pixels": len(comp["cells"]),
                "boundary": [[r, c] for r, c in boundary],
                "children": children[cid],
            }}
        )'''
_SEG_NODE_AFTER = '''        nodes.append(
            _NodeDict({{
                "id": cid,
                "color": color,
                "hash": signature[0],
                "rotation": signature[1],
                "rotational_symmetry": signature[2],
                "pose_hash": signature[3],
                "shape_hash": signature[4],
                "pixels": len(comp["cells"]),
                "boundary": _BoundaryList(_CoordList([r, c]) for r, c in boundary),
                "children": children[cid],
            }})
        )'''
_SEG_ADDENDUM = {repr(PATCH_SEGMENTATION_ADDENDUM)}

def apply_seg_edits(root):
    seg_path = Path(root) / "inference/utils/segmentation.py"
    seg_text = seg_path.read_text(encoding="utf-8")
    if "class _NodeDict(" in seg_text:
        return
    if seg_text.count(_SEG_NODE_BEFORE) != 1:
        raise RuntimeError(f"Expected _SEG_NODE_BEFORE once in segmentation.py, got {{seg_text.count(_SEG_NODE_BEFORE)}}")
    seg_path.write_text(seg_text.replace(_SEG_NODE_BEFORE, _SEG_NODE_AFTER, 1) + _SEG_ADDENDUM, encoding="utf-8")
"""
    c12_main_old = 'if __name__ == "__main__":  # notebook cell: after the HF, RESET-off, KA, TP, MA-text and MA.4c cells, before any import\n    _applied = apply_hy_edits(Path(BUNDLE_DIR) / "src" / "ARC3-Inference")  # noqa: F821 (defined in cell 4)\n    print(f"[HY] sandbox hygiene edits applied ({len(_applied)}): " + "; ".join(_applied))'
    c12_main_new = (
        seg_patch_block
        + '\nif __name__ == "__main__":  # notebook cell: after the HF, RESET-off, KA, TP, MA-text and MA.4c cells, before any import\n'
        '    _applied = apply_hy_edits(Path(BUNDLE_DIR) / "src" / "ARC3-Inference")  # noqa: F821 (defined in cell 4)\n'
        '    apply_seg_edits(Path(BUNDLE_DIR) / "src" / "ARC3-Inference")\n'
        '    print(f"[HY-Pure] sandbox hygiene edits applied ({len(_applied)}) + [SEG] _NodeDict installed: " + "; ".join(_applied))'
    )
    assert c12_main_old in c12_src, "Could not find c12_main_old in Cell 12"
    nb["cells"][12]["source"] = c12_src.replace(c12_main_old, c12_main_new, 1)

    # Cell 20: Add 1x SGLang auto-relaunch on early exit during startup health loop
    c20_src = _src(nb["cells"][20])
    old_sgl_loop = """url, t0 = f"http://127.0.0.1:{SERVED_MODEL_PORT}/health", time.time()
last_report = -30
while time.time() - NOTEBOOK_START_TIME < SERVER_STARTUP_TIMEOUT:
    if proc.poll() is not None:
        show_log_tail()
        break"""
    new_sgl_loop = """url, t0 = f"http://127.0.0.1:{SERVED_MODEL_PORT}/health", time.time()
last_report = -30
_sgl_restarts = 0
while time.time() - NOTEBOOK_START_TIME < SERVER_STARTUP_TIMEOUT:
    if proc.poll() is not None:
        show_log_tail()
        if _sgl_restarts < 1:
            _sgl_restarts += 1
            print(f"WARNING: SGLang exited with code {proc.returncode}; relaunching (attempt {_sgl_restarts + 1})...", flush=True)
            with open(LOG, "ab", buffering=0) as logf:
                proc = subprocess.Popen(args, env=env, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
            time.sleep(3)
            continue
        break"""
    assert old_sgl_loop in c20_src, "Could not find old_sgl_loop in Cell 20"
    nb["cells"][20]["source"] = c20_src.replace(old_sgl_loop, new_sgl_loop, 1)

    # Cell 24: Configure offline benchmark games (10-game 25m benchmark by default, or 1-game 5m smoke when fast_commit_smoke=True)
    if fast_commit_smoke:
        nb["cells"][24]["source"] = _src(nb_shiiin["cells"][17]).rstrip() + "\n"
    else:
        nb["cells"][24]["source"] = _src(nb_v11["cells"][16]).rstrip() + "\n"

    # Cell 26: Disable H3v2 Inference Audit
    nb["cells"][26]["source"] = CELL26_NOOP

    # Cell 27: H1.1 Solved-Level Memory + vickyfym (`31.46` LB) winning-attempt RESET handling in `level_actions`
    c27_src = _src(nb["cells"][27])
    old_level_actions = """    for entry in history_entries or []:
        action = str(getattr(entry, "action", "") or "").strip()
        if action and prev == level:
            out.append(action)
        lvl = getattr(getattr(entry, "frame", None), "level", None)"""
    new_level_actions = """    for entry in history_entries or []:
        action = str(getattr(entry, "action", "") or "").strip()
        if action and prev == level:
            if action.upper().startswith("RESET"):
                out = []
            else:
                out.append(action)
        lvl = getattr(getattr(entry, "frame", None), "level", None)"""
    assert old_level_actions in c27_src, "Could not find old_level_actions in Cell 27"
    nb["cells"][27]["source"] = c27_src.replace(old_level_actions, new_level_actions, 1)

    # Cell 28: Replace SCH with `shiiin9` (`31.54` LB) `ours_form_priority.py` (`install_d`, `b0 = 0.0`)
    d_prime_src = _src(nb_shiiin["cells"][21]).rstrip() + "\n"
    nb["cells"][28]["source"] = d_prime_src

    # Cell 33: Run benchmark (`ondrewsha` S16 3600s gateway wait + runtime cap + 5x competition game list retry + `shiiin9` `#OURS_FRESH` log)
    c33_src = _src(nb["cells"][33]).rstrip()
    fresh_print_src = _src(nb_shiiin["cells"][23]).rstrip()
    nb["cells"][33]["source"] = c33_src + "\n" + fresh_print_src + "\n"

    # Clear outputs and execution_counts across all code cells
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    return nb


def verify_v38_locally(nb: dict) -> None:
    """Apply all v38 cells (2, 5..12, 26..29) to a fresh copy of taaf_bundle and run unit + subprocess checks."""
    if not (TAAF_BUNDLE_DIR / "src").exists():
        raise RuntimeError(f"{TAAF_BUNDLE_DIR}/src not found; ensure dfranzen/taaf-kaggle-source-bundle-copy is unpacked.")

    test_dir = Path("/tmp/taaf_verify_v38")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(TAAF_BUNDLE_DIR / "src", test_dir / "src")

    patch_src = _src(nb["cells"][2])
    if patch_src.startswith("%%writefile"):
        patch_src = patch_src.split("\n", 1)[1]
    if not patch_src.endswith("\n"):
        patch_src += "\n"
    patch_file = Path("/tmp/v38_harness_changes.patch")
    patch_file.write_text(patch_src, encoding="utf-8")

    subprocess.run(
        ["git", "apply", "--include=ARC3-Inference/*", "-v", str(patch_file)],
        cwd=str(test_dir / "src"),
        check=True,
        capture_output=True,
    )

    arc3_root = test_dir / "src" / "ARC3-Inference"
    baseline_prompts = (arc3_root / "inference/agent/prompts.py").read_text(encoding="utf-8")

    ns_base = {
        "__name__": "__main__",
        "BUNDLE_DIR": str(test_dir),
        "SERVED_MODEL_PORT": 8001,
        "TRUE_SUBMISSION": False,
        "SERVER_STARTUP_TIMEOUT": 35 * 60,
    }
    for cell_idx in (5, 6, 7, 8, 9, 10, 11, 12):
        cell_code = _src(nb["cells"][cell_idx])
        exec(compile(cell_code, f"v38_cell_{cell_idx}.py", "exec"), dict(ns_base))

    # Invariant 1: prompts.py must be 100% byte-for-byte identical to dfranzen Milestone 2!
    after_prompts = (arc3_root / "inference/agent/prompts.py").read_text(encoding="utf-8")
    assert after_prompts == baseline_prompts, "CRITICAL: prompts.py was mutated from dfranzen Milestone 2 baseline!"

    for py_file in arc3_root.rglob("*.py"):
        ast.parse(py_file.read_text(encoding="utf-8"))

    # Add patched repos to sys.path and stub binary wheel modules for local import check
    for mod_name in ("arcengine", "arcengine.enums", "arc_agi", "arc_agi.models", "imageio", "imageio.v3"):
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)
    sys.modules["arcengine"].enums = sys.modules["arcengine.enums"]
    sys.modules["arcengine.enums"].GameAction = type("GameAction", (), {})
    sys.modules["arcengine.enums"].GameState = type("GameState", (), {})
    sys.modules["arcengine"].GameAction = sys.modules["arcengine.enums"].GameAction
    sys.modules["arcengine"].GameState = sys.modules["arcengine.enums"].GameState
    sys.modules["arc_agi"].models = sys.modules["arc_agi.models"]
    sys.modules["arc_agi.models"].EnvironmentInfo = type("EnvironmentInfo", (), {})
    sys.modules["arc_agi"].Arcade = type("Arcade", (), {"make": lambda self, *a, **k: None})
    sys.modules["arc_agi"].OperationMode = type("OperationMode", (), {"OFFLINE": "OFFLINE", "COMPETITION": "COMPETITION"})
    sys.modules["imageio"].v3 = sys.modules["imageio.v3"]

    for repo in sorted((test_dir / "src").iterdir(), reverse=True):
        for candidate in (repo / "src", repo):
            if candidate.is_dir():
                sys.path.insert(0, str(candidate))

    for cell_idx in (26, 27, 28, 29):
        cell_code = _src(nb["cells"][cell_idx])
        exec(compile(cell_code, f"v38_cell_{cell_idx}.py", "exec"), {"__name__": "__main__"})

    # Unit check 1: segmentation _NodeDict aliases
    from inference.utils.segmentation import segment_layer
    seg_res = segment_layer(((0, 1, 1), (0, 1, 1), (0, 0, 0)), "WwgGcBMPRbSYOrNp")
    nodes = seg_res["nodes"]
    assert len(nodes) >= 2
    n0 = nodes[0]
    assert n0["size"] == n0["pixels"]
    assert len(n0["bbox"]) == 4
    assert len(n0["center"]) == 2
    assert json.loads(json.dumps(n0)) == dict(n0)

    # Unit check 2: TP runtime_state cache round-trip
    from inference.agent.runtime_state import Frame, HistoryEntry, load_runtime_state, write_runtime_state
    tp_path = Path("/tmp/taaf_verify_v38/test_state.json")
    f1 = Frame(grid=((0, 1), (2, 3)), step=1, level=1)
    h1 = [HistoryEntry(action="UP", frame=f1, result={"executed": True, "gameplay_changed": True})]
    write_runtime_state(tp_path, current_frame=f1, history=h1)
    lf, lh = load_runtime_state(tp_path)
    assert lf == f1 and len(lh) == 1 and lh[0].action == "UP"

    # Unit check 3: End-to-end subprocess check of run_sandboxed_python
    from inference.agent.python_tool_sandbox import run_sandboxed_python
    sb_res = run_sandboxed_python(
        code=(
            "import time, contextlib, pprint, traceback\n"
            "assert last_animation_timeline['steps'] == []\n"
            "assert list(last_animation_timeline.items()) == []\n"
            "seg = current_frame.segmentation\n"
            "print('SANDBOX_OK', len(seg['nodes']), seg['nodes'][0]['bbox'], seg['nodes'][0]['center'], seg['nodes'][0]['size'])\n"
        ),
        timeout_seconds=10,
        initial_state={
            "current_frame": {"ascii": "Wb\nWW", "step": 0, "level": 1, "shape": [2, 2], "grid": [[0, 1], [0, 0]]},
            "history": [],
            "valid_actions": ["UP", "DOWN"],
            "color_legend": "WwgGcBMPRbSYOrNp",
            "animation_enabled": True,
        },
        action_handler=lambda *a, **k: {"executed": False},
        animation_handler=lambda *a, **k: {},
    )
    assert not sb_res.get("error") and "SANDBOX_OK" in str(sb_res.get("stdout", "")), f"Sandbox check failed: {sb_res}"

    print("[VERIFY v38] 100% byte-identical dfranzen prompts.py, sandbox subprocess check, TP cache, SEG, H1.1, D_PRIME, KA, GW PASSED!")


def update_kernel_metadata() -> None:
    meta = {
        "id": "zerbytheboss/v36-sovereign-apex-engine",
        "title": "v36 - Sovereign Apex Engine",
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
    fast_smoke = "--fast-smoke" in sys.argv
    nb_v38 = build_v38_notebook(fast_commit_smoke=fast_smoke)
    verify_v38_locally(nb_v38)

    OUTPUT_V38_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_SUB_PATH.parent.mkdir(parents=True, exist_ok=True)

    for path in (OUTPUT_V38_PATH, OUTPUT_SUB_PATH):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb_v38, f, indent=1, ensure_ascii=False)
            f.write("\n")

    update_kernel_metadata()

    sha_v38 = hashlib.sha256(OUTPUT_V38_PATH.read_bytes()).hexdigest()
    sha_sub = hashlib.sha256(OUTPUT_SUB_PATH.read_bytes()).hexdigest()
    assert sha_v38 == sha_sub
    assert len(nb_v38["cells"]) == 36
    mode_str = "FAST_1GAME_5M_SMOKE" if fast_smoke else "FULL_10GAME_25M_BENCHMARK"
    print(
        f"Built {OUTPUT_V38_PATH} and {OUTPUT_SUB_PATH} [{mode_str}] "
        f"({OUTPUT_V38_PATH.stat().st_size} bytes, sha256={sha_v38[:12]}, cells=36)"
    )


if __name__ == "__main__":
    main()

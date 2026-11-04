#!/usr/bin/env python3
"""
scripts/build_v40_notebook.py
=============================
Deterministic Builder & Local Verifier for ARC Prize 2026 — Version 40 Sovereign Orthogonal Synthesis.

Architectural Synthesis (v39_v10 44.40% Mean / 44.64% Median / 45 Solved Levels + v39 Transcript Fixes):
1. Pure `dfranzen` Milestone 2 Base (`v11` / `shiiin9` / `ondrewsha` Cell 2 patch, 498,799 bytes):
   - Zero invasive sandbox state leaks, standard 30s snippet timeout, intact stale-state and batch guards.
2. Refined `HF` (Cell 5) + `RESET=off` (Cell 6):
   - Keeps `HF.1a`: Animated actions with no lasting interior change are reported as `ANIMATED` in the action
     echo and turn trace so single-action probes know an animation occurred and can inspect `last_animation_frames`
     and `last_animation_timeline`.
   - Removes `HF.1b` (solver batch guard exemption for `animated`) and `HF.1c` (`_stale_state` exemption for
     `animated`): when an action bounces back to its starting state (`gameplay_changed=False`, e.g., `r11l:4085`
     `MOUSE(16,56)`), multi-action batches and multi-action loops stop immediately instead of executing subsequent
     moves on the un-moved board.
   - Keeps `HF.3` (`UNDO` / `ACTION7` exempted from arming `_stale_state` up to 3 no-ops and never refused by
     stale-state guard), `HF.4` (timeout reports exact executed game actions and preserves retained functions),
     and strips the whole-run clock from payload; `EXPOSE_RESET="off"`.
3. `MA-text` (Cell 9) + `MA.4c` (Cell 11):
   - Truthful `StaleStateActionError` refusal text (`MA.1b`, `sb26 = 100.00`), removes misleading `score` field
     (`MA.2`), one action per `UNDO` (`MA.3`), case-sensitive color legend (`MA.7`), and `_edge_change_is_bar_like`
     2-cell strip test in the 4-cell edge band (`MA.4c`).
4. `HY + SEG + RF-Norm v2` (Cell 12):
   - Extended `SAFE_MODULES` including `pickle`, `struct`, `array`, `queue`, `types`, `abc`, `base64`, `hashlib`,
     `zlib`, `textwrap`, `typing`, `dataclasses`, `enum`, `decimal`, `numbers`, `contextlib`, `pprint`, `traceback`
     (fixes `[r11l:7382] ImportError: Module 'pickle' is not allowed in the sandbox`).
   - Safe `builtins` proxy in `_safe_import` (`import builtins; builtins.mp = mp` in `[lp85:5312]`) + automatic
     `_json_safe` fallback on `json.dumps` when dict keys are tuples (`[tr87:2300]`).
   - `SEG`: Zero-stdout-overhead `_NodeDict`, `_BoundaryList`, `_CoordList`, `_CentroidTuple` on `segmentation.py`
     (0 segmentation `KeyError`s across all 10 games in `v39_v10`).
   - `RF-Norm v2`: AST function retention normalization in `inference/utils/retained_functions.py` with:
     * `__rf_consts__` persistence across turns: inlined module-level literal constants (`ORD2`, `D2`, `OPP2` in
       `lp85`; `SH` in `re86:7233`; `KEYP` in `tr87:6987`) are both inlined into the helper function AND restored
       into `runtime_globals` + `_extract_literal_consts` on subsequent turns so new or edited helpers can use them.
     * `_STDLIB_AUTO_IMPORTS` fallback in `collect_functions`: helpers that use `deque`, `defaultdict`, `Counter`,
       `heappush`, `heappop`, `product`, `permutations`, `combinations`, `deepcopy`, etc. without an explicit import
       in the same snippet (`bfs4` in `tu93:7138 -> 7240`) automatically have the explicit import prepended.
5. Board Images Preserved on Resumptions (`K3` in Cell 10 + `H3v2` disabled in Cell 26) + `vickyfym` / `H1.1`
   Cache-Safe Eviction-Only Solved-Level Memory (Cell 27) + `shiiin9` `D'` Scheduler (Cell 28).
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
OUTPUT_V40_PATH = REPO_ROOT / "notebooks" / "submission_v40.ipynb"
STAGE_DIR = Path("/tmp/v37_kernel")
OUTPUT_SUB_PATH = STAGE_DIR / "submission.ipynb"
METADATA_PATH = STAGE_DIR / "kernel-metadata.json"

CELL0_MARKDOWN = r"""# ARC Prize 2026 — Version 40 Sovereign Orthogonal Synthesis

## Building on `v39_v10` (`44.40%` Mean, `44.64%` Median, `45 / 72` Levels Solved) + `v39` Transcript Fixes

Version 40 builds directly on our highest-scoring 10-game benchmark architecture (`v39_v10`: `44.40%` mean, `44.64%` median, `45/72` levels solved, `5` `StaleStateActionError`s, `0` `AttributeError`s) and resolves the exact residual failure modes observed in the `v39_v10` transcripts:

1. **Refined `HF` Animated Bounce-Back Guard (`r11l:4085`, Cell 5)**:
   - Keeps `ANIMATED` reporting in the action echo and turn trace so single-action probes know an animation occurred and can inspect `last_animation_frames` / `last_animation_timeline`.
   - Restores the batch no-op stop and `_stale_state` guard when an animated action produces no lasting gameplay change (`gameplay_changed=False`), preventing invalid bounce-back moves in `r11l` from executing subsequent planned swaps on an un-moved board.
2. **`RF-Norm v2` Cross-Turn Constant Persistence + `_STDLIB_AUTO_IMPORTS` (`re86:7233`, `tr87:6987`, `tu93:7138`, Cell 12)**:
   - Tagging inlined module-level literal constants with `__rf_consts__` restores them into `runtime_globals` and `_extract_literal_consts` on subsequent turns so new helpers (`route` using `SH` in `re86`) and edited helpers (`build3` using `KEYP` in `tr87`) never raise `NameError`.
   - `_STDLIB_AUTO_IMPORTS` automatically prepends explicit stdlib imports (`from collections import deque`, etc.) when a helper (`bfs4` in `tu93:7138`) uses an unimported stdlib symbol.
3. **Zero-`ImportError` & Tuple-Key `json.dumps` Hygiene (`r11l:7382`, `lp85:5343`, `tr87:2300`, Cell 12)**:
   - Extends `SAFE_MODULES` with `pickle`, `struct`, `array`, `queue`, `types`, `abc`, `base64`, `hashlib`, `zlib`, `textwrap`.
   - Provides a safe `builtins` proxy in `_safe_import` (`import builtins; builtins.mp = mp` in `lp85:5312`) and automatic `_json_safe` fallback when `json.dumps` is called on dicts with tuple keys (`tr87:2300`).
4. **Full Board Images on Resumptions (`K3` Cell 10 + `H3v2` Disabled Cell 26) + `vickyfym` / `H1.1` Cache-Safe Eviction-Only Solved-Level Memory (Cell 27) + `shiiin9` `D'` Continuous Slot Priority Scheduler (Cell 28)**.
"""

CELL10_IMAGES_ON = r'''# Keep ARC3_RESUME_PROMPT_IMAGES="1" (dfranzen Milestone 2 default, verified in v38_v9 and v39_v10).
# Nearly 50% of turns in multi-step CSP games are turn_token_budget resumptions; keeping the board
# image attached preserves full visual attention across multi-turn reasoning.
import os

K3_ENV = {"ARC3_RESUME_PROMPT_IMAGES": "1"}
if __name__ == "__main__":
    os.environ.update(K3_ENV)
    print(f"[K3] env: {K3_ENV} (board images preserved on resumption messages)")
'''

CELL26_NOOP = r'''# H3v2 Inference Audit disabled in v40: verified in v38_v9 and v39_v10 that keeping per-turn user prompts
# free of mid-level audit injections improves multi-turn visual and symbolic reasoning.
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

PATCH_RETAINED_FUNCTIONS_HELPERS = r'''
import copy as _rf_copy

_STDLIB_AUTO_IMPORTS = {
    "abc": "import abc",
    "array": "import array",
    "base64": "import base64",
    "bisect": "import bisect",
    "collections": "import collections",
    "contextlib": "import contextlib",
    "copy": "import copy",
    "dataclasses": "import dataclasses",
    "decimal": "import decimal",
    "difflib": "import difflib",
    "enum": "import enum",
    "fractions": "import fractions",
    "functools": "import functools",
    "hashlib": "import hashlib",
    "heapq": "import heapq",
    "itertools": "import itertools",
    "json": "import json",
    "math": "import math",
    "numbers": "import numbers",
    "operator": "import operator",
    "pickle": "import pickle",
    "pprint": "import pprint",
    "queue": "import queue",
    "random": "import random",
    "re": "import re",
    "statistics": "import statistics",
    "string": "import string",
    "struct": "import struct",
    "textwrap": "import textwrap",
    "time": "import time",
    "traceback": "import traceback",
    "types": "import types",
    "typing": "import typing",
    "zlib": "import zlib",
    "deque": "from collections import deque",
    "defaultdict": "from collections import defaultdict",
    "Counter": "from collections import Counter",
    "namedtuple": "from collections import namedtuple",
    "OrderedDict": "from collections import OrderedDict",
    "heappush": "from heapq import heappush",
    "heappop": "from heapq import heappop",
    "heapify": "from heapq import heapify",
    "bisect_left": "from bisect import bisect_left",
    "bisect_right": "from bisect import bisect_right",
    "insort": "from bisect import insort",
    "reduce": "from functools import reduce",
    "lru_cache": "from functools import lru_cache",
    "cache": "from functools import cache",
    "partial": "from functools import partial",
    "product": "from itertools import product",
    "permutations": "from itertools import permutations",
    "combinations": "from itertools import combinations",
    "combinations_with_replacement": "from itertools import combinations_with_replacement",
    "chain": "from itertools import chain",
    "islice": "from itertools import islice",
    "groupby": "from itertools import groupby",
    "accumulate": "from itertools import accumulate",
    "count": "from itertools import count",
    "cycle": "from itertools import cycle",
    "repeat": "from itertools import repeat",
    "zip_longest": "from itertools import zip_longest",
    "deepcopy": "from copy import deepcopy",
    "sqrt": "from math import sqrt",
    "ceil": "from math import ceil",
    "floor": "from math import floor",
    "gcd": "from math import gcd",
    "lcm": "from math import lcm",
    "inf": "from math import inf",
    "nan": "from math import nan",
    "atan2": "from math import atan2",
    "hypot": "from math import hypot",
    "Fraction": "from fractions import Fraction",
    "Decimal": "from decimal import Decimal",
}


def _extract_inlined_rf_consts(sources):
    """Extract module-level literal constants previously tagged via __rf_consts__ inside retained functions."""
    consts = {}
    for src in sources or ():
        try:
            tree = ast.parse(src)
            if not tree.body or not isinstance(tree.body[-1], ast.FunctionDef):
                continue
            body = tree.body[-1].body
            idx = 0
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(getattr(body[0], "value", None), ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                idx = 1
            if idx >= len(body):
                continue
            first = body[idx]
            if (
                isinstance(first, ast.Assign)
                and len(first.targets) == 1
                and isinstance(first.targets[0], ast.Name)
                and first.targets[0].id == "__rf_consts__"
            ):
                names = ast.literal_eval(first.value)
                if isinstance(names, tuple):
                    name_set = set(names)
                    for stmt in body[idx + 1 : idx + 1 + len(names)]:
                        if (
                            isinstance(stmt, ast.Assign)
                            and len(stmt.targets) == 1
                            and isinstance(stmt.targets[0], ast.Name)
                            and stmt.targets[0].id in name_set
                        ):
                            ast.literal_eval(stmt.value)
                            consts[stmt.targets[0].id] = stmt.value
        except Exception:
            pass
    return consts


def _extract_literal_consts(tree, previous=None):
    """Extract top-level literal constant assignments (<= 2048 chars) from previous retained functions and snippet AST."""
    consts = _extract_inlined_rf_consts(previous.values() if isinstance(previous, dict) else previous)
    for stmt in getattr(tree, "body", ()):
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            tgt = stmt.targets[0]
            if isinstance(tgt, ast.Name):
                try:
                    val = ast.literal_eval(stmt.value)
                    if len(repr(val)) <= 2048:
                        consts[tgt.id] = stmt.value
                except Exception:
                    pass
            elif (
                isinstance(tgt, (ast.Tuple, ast.List))
                and isinstance(stmt.value, (ast.Tuple, ast.List))
                and len(tgt.elts) == len(stmt.value.elts)
            ):
                for t_el, v_el in zip(tgt.elts, stmt.value.elts):
                    if isinstance(t_el, ast.Name):
                        try:
                            val = ast.literal_eval(v_el)
                            if len(repr(val)) <= 2048:
                                consts[t_el.id] = v_el
                        except Exception:
                            pass
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
            try:
                val = ast.literal_eval(stmt.value)
                if len(repr(val)) <= 2048:
                    consts[stmt.target.id] = stmt.value
            except Exception:
                pass
    return consts


def _normalize_func_for_retention(node, raw_seg, consts, skip_names):
    """If raw_seg already passes _function_record and reads no top-level literal constants,
    return raw_seg untouched. Otherwise strip parameter/return annotations, strip top-level
    `global` declarations inside the function, substitute literal constant default args, and
    inline referenced top-level literal constants at the start of the function body."""
    try:
        _, _, reads = _function_record(raw_seg)
        needed = [n for n in sorted(reads) if n in consts and n not in skip_names]
        if not needed:
            return raw_seg
    except Exception:
        pass

    try:
        fn = _rf_copy.deepcopy(node)
        fn.decorator_list = []
        if hasattr(fn, "type_params"):
            fn.type_params = []
        fn.returns = None
        all_args = [*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs]
        all_args += [a for a in (fn.args.vararg, fn.args.kwarg) if a is not None]
        for a in all_args:
            a.annotation = None
        arg_names = {a.arg for a in all_args}

        fn.args.defaults = [
            _rf_copy.deepcopy(consts[d.id]) if isinstance(d, ast.Name) and d.id in consts else d
            for d in fn.args.defaults
        ]
        fn.args.kw_defaults = [
            _rf_copy.deepcopy(consts[d.id]) if isinstance(d, ast.Name) and d.id in consts else d
            for d in fn.args.kw_defaults
        ]

        global_names = []

        class _StripGlobal(ast.NodeTransformer):
            def visit_FunctionDef(self, n):
                return n

            def visit_AsyncFunctionDef(self, n):
                return n

            def visit_ClassDef(self, n):
                return n

            def visit_Global(self, n):
                for gname in n.names:
                    if gname not in global_names:
                        global_names.append(gname)
                return ast.Pass()

        fn.body = [_StripGlobal().visit(s) for s in fn.body]

        loaded_names = set(global_names)
        for sub in ast.walk(fn):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                loaded_names.add(sub.id)

        inlined = []
        inlined_const_names = []
        for cname in sorted(loaded_names):
            if cname not in arg_names and cname not in skip_names and cname in consts:
                inlined_const_names.append(cname)
                inlined.append(
                    ast.Assign(
                        targets=[ast.Name(id=cname, ctx=ast.Store())],
                        value=_rf_copy.deepcopy(consts[cname]),
                    )
                )
            elif cname in global_names and cname not in arg_names and cname not in skip_names:
                inlined.append(
                    ast.Assign(
                        targets=[ast.Name(id=cname, ctx=ast.Store())],
                        value=ast.Constant(value=None),
                    )
                )

        if inlined_const_names:
            tag_stmt = ast.Assign(
                targets=[ast.Name(id="__rf_consts__", ctx=ast.Store())],
                value=ast.Tuple(
                    elts=[ast.Constant(value=n) for n in inlined_const_names],
                    ctx=ast.Load(),
                ),
            )
            inlined = [tag_stmt, *inlined]

        if inlined:
            if (
                fn.body
                and isinstance(fn.body[0], ast.Expr)
                and isinstance(getattr(fn.body[0], "value", None), ast.Constant)
                and isinstance(fn.body[0].value.value, str)
            ):
                fn.body = [fn.body[0], *inlined, *fn.body[1:]]
            else:
                fn.body = [*inlined, *fn.body]

        ast.fix_missing_locations(fn)
        norm_seg = ast.unparse(fn)
        _function_record(norm_seg)
        return norm_seg
    except Exception:
        return raw_seg
'''


def _src(cell: dict) -> str:
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else str(s)


def build_v40_notebook(*, fast_commit_smoke: bool = False) -> dict:
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

    # Cell 5: Refine HF_EDITS — keep ANIMATED echo/trace (HF.1a), but remove HF.1b (solver batch guard
    # exemption for animated no-ops) and HF.1c (_stale_state exemption for animated no-ops) so bounce-back
    # animations (like r11l:4085 MOUSE(16,56)) stop batches/loops before executing subsequent moves on a stale board.
    c5_src = _src(nb["cells"][5]).replace("\r\n", "\n")
    old_hf1b = """    (
        "inference/framework/solver.py",
        '                and not payload.get("gameplay_changed", True)\\n            ):\\n',
        '                and not payload.get("gameplay_changed", True)\\n'
        '                and not int(((payload.get("animation") or {}).get("transient_pixels")) or 0)\\n'
        "            ):\\n",
        "the batch no-op guard does not stop on an animated action",
    ),\n"""
    old_hf1c = """    (
        # HF.2 (layer audit 21.10): an ANIMATED action still armed the snippet's stale-state guard, so the next
        # action() of the same snippet was refused as following "a no-op" (S6 demos: 61 refusals, 22% of all)
        "inference/agent/python_tool_sandbox.py",
        '            if gameplay_changed is False and action_result.get("executed"):\\n',
        '            if (gameplay_changed is False and action_result.get("executed")\\n'
        '                    and not int(((action_result.get("animation") or {}).get("transient_pixels")) or 0)):\\n',
        "an animated action does not arm the stale-state guard",
    ),\n"""
    assert old_hf1b in c5_src, "Could not find old_hf1b in Cell 5"
    assert old_hf1c in c5_src, "Could not find old_hf1c in Cell 5"
    c5_src = c5_src.replace(old_hf1b, "", 1).replace(old_hf1c, "", 1)
    nb["cells"][5]["source"] = c5_src

    # Cells 6, 7, 8, 9, 11: Keep RESET=off, KA, TP, MA-text, and MA.4c active!
    # Cell 10: Keep ARC3_RESUME_PROMPT_IMAGES="1" (dfranzen default, verified in v38_v9 and v39_v10)
    nb["cells"][10]["source"] = CELL10_IMAGES_ON

    # Cell 12: HY + AP.3a safe animation key fallback + extended SAFE_MODULES + builtins proxy + json.dumps fallback + SEG + RF-Norm v2
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

    old_ap4 = '''        '        "contextlib",\\n        "pprint",\\n        "traceback",\\n    }\\n','''
    new_ap4 = (
        '''        '        "abc",\\n        "array",\\n        "base64",\\n        "contextlib",\\n'''
        '''        "dataclasses",\\n        "decimal",\\n        "enum",\\n        "hashlib",\\n'''
        '''        "numbers",\\n        "pickle",\\n        "pprint",\\n        "queue",\\n'''
        '''        "struct",\\n        "textwrap",\\n        "traceback",\\n        "types",\\n'''
        '''        "typing",\\n        "zlib",\\n    }\\n','''
    )
    assert old_ap4 in c12_src, "Could not find old_ap4 in Cell 12"
    c12_src = c12_src.replace(old_ap4, new_ap4, 1)

    seg_and_rf_patch_block = f"""
# Sandbox builtins proxy (lp85:5312 `import builtins; builtins.mp = mp`) & json.dumps tuple-key fallback (tr87:2300)
_SB_IMPORT_BEFORE = '''    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = str(name or "").split(".", 1)[0]
        if root not in SAFE_MODULES:
            raise ImportError(f"Module '{{name}}' is not allowed in the sandbox.")
        return builtins.__import__(name, globals, locals, fromlist, level)'''

_SB_IMPORT_AFTER = '''    _ORIG_JSON_DUMPS = json.dumps

    def _safe_json_dumps(obj, *args, **kwargs):
        try:
            return _ORIG_JSON_DUMPS(obj, *args, **kwargs)
        except TypeError:
            return _ORIG_JSON_DUMPS(_json_safe(obj), *args, **kwargs)

    json.dumps = _safe_json_dumps
    _ACTIVE_RUNTIME_GLOBALS = None

    class _BuiltinsProxy:
        def __getattr__(self, key):
            rg = _ACTIVE_RUNTIME_GLOBALS
            if isinstance(rg, dict):
                b = rg.get("__builtins__")
                if isinstance(b, dict) and key in b:
                    return b[key]
                if key in rg:
                    return rg[key]
            raise AttributeError(key)

        def __setattr__(self, key, value):
            rg = _ACTIVE_RUNTIME_GLOBALS
            if isinstance(rg, dict):
                b = rg.get("__builtins__")
                if isinstance(b, dict):
                    b[key] = value
                rg[key] = value

    _BUILTINS_PROXY = _BuiltinsProxy()

    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = str(name or "").split(".", 1)[0]
        if root == "builtins":
            return _BUILTINS_PROXY
        if root not in SAFE_MODULES:
            raise ImportError(f"Module '{{name}}' is not allowed in the sandbox.")
        return builtins.__import__(name, globals, locals, fromlist, level)'''

_SB_RG_BEFORE = '''        runtime_globals["__builtins__"]["__import__"] = _safe_import'''
_SB_RG_AFTER = '''        global _ACTIVE_RUNTIME_GLOBALS
        _ACTIVE_RUNTIME_GLOBALS = runtime_globals
        runtime_globals["__builtins__"]["__import__"] = _safe_import'''

def apply_sb_import_edits(root):
    sb_path = Path(root) / "inference/agent/python_tool_sandbox.py"
    sb_text = sb_path.read_text(encoding="utf-8")
    if "class _BuiltinsProxy:" in sb_text:
        return
    if sb_text.count(_SB_IMPORT_BEFORE) != 1 or sb_text.count(_SB_RG_BEFORE) != 1:
        raise RuntimeError("Could not locate exact _safe_import target lines in python_tool_sandbox.py")
    sb_text = sb_text.replace(_SB_IMPORT_BEFORE, _SB_IMPORT_AFTER, 1).replace(_SB_RG_BEFORE, _SB_RG_AFTER, 1)
    sb_path.write_text(sb_text, encoding="utf-8")

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

# RF-Norm v2: AST function retention normalization + cross-turn __rf_consts__ + _STDLIB_AUTO_IMPORTS
_RF_HELPERS = {repr(PATCH_RETAINED_FUNCTIONS_HELPERS)}
_RF_RESTORE_BEFORE = '''    for name in kept.keys() - survivors.keys():
        if name not in available:
            namespace.pop(name, None)
    return survivors, rejected'''
_RF_RESTORE_AFTER = '''    for name in kept.keys() - survivors.keys():
        if name not in available:
            namespace.pop(name, None)
    for _cname, _cnode in _extract_inlined_rf_consts(survivors.values()).items():
        if _cname not in namespace and _cname not in available:
            try:
                namespace[_cname] = ast.literal_eval(_cnode)
            except Exception:
                pass
    return survivors, rejected'''
_RF_BEFORE = '''    imports, declared_imports = (_matching_imports(import_nodes, namespace)
                                if retain_imports or repair_hints else ({{}}, {{}}))
    for name in dict.fromkeys([*previous, *fresh]):'''
_RF_AFTER = '''    imports, declared_imports = (_matching_imports(import_nodes, namespace)
                                if retain_imports or repair_hints else ({{}}, {{}}))
    _literal_consts = _extract_literal_consts(tree, previous)
    _skip_const_names = set(available) | set(imports.keys()) | set(previous.keys()) | set(fresh.keys())
    for name in dict.fromkeys([*previous, *fresh]):'''
_RF_CAND_BEFORE = '''        candidates[name] = ast.get_source_segment(code_text, node)'''
_RF_CAND_AFTER = '''        _raw_seg = ast.get_source_segment(code_text, node)
        candidates[name] = _normalize_func_for_retention(node, _raw_seg, _literal_consts, _skip_const_names)'''
_RF_IMPORTS_BEFORE = '''            _, _, reads = _function_record(definition)
            statements = [imports[n] for n in sorted(reads & imports.keys()) if n not in available]
            candidates[name] = '\\\\n'.join([*statements, definition])'''
_RF_IMPORTS_AFTER = '''            _, _, reads = _function_record(definition)
            _eff_imports = dict(imports)
            for _rn in sorted(reads):
                if (
                    _rn not in _eff_imports
                    and _rn not in available
                    and _rn not in candidates
                    and _rn not in fresh
                    and _rn not in previous
                    and _rn not in namespace
                    and _rn in _STDLIB_AUTO_IMPORTS
                ):
                    _eff_imports[_rn] = _STDLIB_AUTO_IMPORTS[_rn]
            statements = [_eff_imports[n] for n in sorted(reads & _eff_imports.keys()) if n not in available]
            candidates[name] = '\\\\n'.join([*statements, definition])'''

def apply_rf_norm_edits(root):
    rf_path = Path(root) / "inference/utils/retained_functions.py"
    rf_text = rf_path.read_text(encoding="utf-8")
    if "def _normalize_func_for_retention(" in rf_text:
        return
    for target_snippet in (_RF_RESTORE_BEFORE, _RF_BEFORE, _RF_CAND_BEFORE, _RF_IMPORTS_BEFORE):
        if rf_text.count(target_snippet) != 1:
            raise RuntimeError(f"Could not locate exact target snippet in retained_functions.py: {{target_snippet[:60]!r}}")
    rf_text = rf_text.replace("import types\\n", "import types\\n" + _RF_HELPERS + "\\n", 1)
    rf_text = rf_text.replace(_RF_RESTORE_BEFORE, _RF_RESTORE_AFTER, 1)
    rf_text = rf_text.replace(_RF_BEFORE, _RF_AFTER, 1)
    rf_text = rf_text.replace(_RF_CAND_BEFORE, _RF_CAND_AFTER, 1)
    rf_text = rf_text.replace(_RF_IMPORTS_BEFORE, _RF_IMPORTS_AFTER, 1)
    rf_path.write_text(rf_text, encoding="utf-8")
"""
    c12_main_old = 'if __name__ == "__main__":  # notebook cell: after the HF, RESET-off, KA, TP, MA-text and MA.4c cells, before any import\n    _applied = apply_hy_edits(Path(BUNDLE_DIR) / "src" / "ARC3-Inference")  # noqa: F821 (defined in cell 4)\n    print(f"[HY] sandbox hygiene edits applied ({len(_applied)}): " + "; ".join(_applied))'
    c12_main_new = (
        seg_and_rf_patch_block
        + '\nif __name__ == "__main__":  # notebook cell: after the HF, RESET-off, KA, TP, MA-text and MA.4c cells, before any import\n'
        '    _applied = apply_hy_edits(Path(BUNDLE_DIR) / "src" / "ARC3-Inference")  # noqa: F821 (defined in cell 4)\n'
        '    apply_sb_import_edits(Path(BUNDLE_DIR) / "src" / "ARC3-Inference")\n'
        '    apply_seg_edits(Path(BUNDLE_DIR) / "src" / "ARC3-Inference")\n'
        '    apply_rf_norm_edits(Path(BUNDLE_DIR) / "src" / "ARC3-Inference")\n'
        '    print(f"[HY] sandbox hygiene edits applied ({len(_applied)}) + [SEG] _NodeDict + [RF-Norm v2] installed: " + "; ".join(_applied))'
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

    # Cell 27: H1.1 Solved-Level Memory + vickyfym (`33.94` LB / `lalalia9` `35.79` LB) winning-attempt RESET handling in `level_actions`
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


def verify_v40_locally(nb: dict) -> None:
    """Apply all v40 cells (2, 5..12, 26..29) to a fresh copy of taaf_bundle and run unit + subprocess checks."""
    if not (TAAF_BUNDLE_DIR / "src").exists():
        raise RuntimeError(f"{TAAF_BUNDLE_DIR}/src not found; ensure dfranzen/taaf-kaggle-source-bundle-copy is unpacked.")

    test_dir = Path("/tmp/taaf_verify_v40")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(TAAF_BUNDLE_DIR / "src", test_dir / "src")

    patch_src = _src(nb["cells"][2])
    if patch_src.startswith("%%writefile"):
        patch_src = patch_src.split("\n", 1)[1]
    if not patch_src.endswith("\n"):
        patch_src += "\n"
    patch_file = Path("/tmp/v40_harness_changes.patch")
    patch_file.write_text(patch_src, encoding="utf-8")

    subprocess.run(
        ["git", "apply", "--include=ARC3-Inference/*", "-v", str(patch_file)],
        cwd=str(test_dir / "src"),
        check=True,
        capture_output=True,
    )

    arc3_root = test_dir / "src" / "ARC3-Inference"

    ns_base = {
        "__name__": "__main__",
        "BUNDLE_DIR": str(test_dir),
        "SERVED_MODEL_PORT": 8001,
        "TRUE_SUBMISSION": False,
        "SERVER_STARTUP_TIMEOUT": 35 * 60,
    }
    for cell_idx in (5, 6, 7, 8, 9, 10, 11, 12):
        cell_code = _src(nb["cells"][cell_idx])
        exec(compile(cell_code, f"v40_cell_{cell_idx}.py", "exec"), dict(ns_base))

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
        exec(compile(cell_code, f"v40_cell_{cell_idx}.py", "exec"), {"__name__": "__main__"})

    import os
    assert os.environ.get("ARC3_RESUME_PROMPT_IMAGES") == "1", "ARC3_RESUME_PROMPT_IMAGES must be '1'"

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

    # Unit check 2: MA.4c _edge_change_is_bar_like
    from inference.framework.solver import _edge_change_is_bar_like
    g0 = [[0] * 64 for _ in range(64)]
    g_bar = [list(r) for r in g0]
    for c in range(10, 20):
        g_bar[1][c] = 5
    assert _edge_change_is_bar_like(g0, g_bar) is True
    g_obj = [list(r) for r in g0]
    for r in range(1, 4):
        for c in range(10, 13):
            g_obj[r][c] = 8
    assert _edge_change_is_bar_like(g0, g_obj) is False

    # Unit check 3: End-to-end subprocess check of run_sandboxed_python + RF-Norm v2 across 2 calls!
    from inference.agent.python_tool_sandbox import run_sandboxed_python
    init_state = {
        "current_frame": {"ascii": "Wb\nWW", "step": 0, "level": 1, "shape": [2, 2], "grid": [[0, 1], [0, 0]]},
        "history": [],
        "valid_actions": ["UP", "DOWN"],
        "color_legend": "WwgGcBMPRbSYOrNp",
        "animation_enabled": True,
    }
    sb_res1 = run_sandboxed_python(
        code=(
            "import time, contextlib, pprint, traceback, typing, dataclasses, enum, decimal, numbers, pickle\n"
            "import builtins, json\n"
            "ORD2 = ['SW','S','SE','E','NE','N','NW','W']\n"
            "D2 = {'N':(-3,0),'S':(3,0),'E':(0,3),'W':(0,-3),'NE':(-3,3),'NW':(-3,-3),'SE':(3,3),'SW':(3,-3)}\n"
            "OPP2 = {'N':'S','S':'N','E':'W','W':'E','NE':'SW','SW':'NE','NW':'SE','SE':'NW'}\n"
            "KEYP = [(4, [(4, 6)], [(4, 16)])]\n"
            "def perms6(center: tuple, longarm: str = 'N') -> tuple:\n"
            "    cr, cc = center\n"
            "    return (cr + D2[longarm][0], cc + D2[longarm][1], OPP2[longarm], len(ORD2))\n"
            "def build3(fr):\n"
            "    return [r for (r, ys, bs) in KEYP]\n"
            "def bfs4_unimported(x):\n"
            "    q = deque([x])\n"
            "    return q.popleft()\n"
            "def mp(x): return x + 1\n"
            "builtins.mp = mp\n"
            "SLOTIDX, MOVES = None, None\n"
            "def setup() -> tuple:\n"
            "    global SLOTIDX, MOVES\n"
            "    SLOTIDX = (1, 2)\n"
            "    MOVES = {'A': 1}\n"
            "    return SLOTIDX, MOVES\n"
            "print('CALL1_OK', perms6((15, 14), 'SE'), build3(None), mp(4), setup(), json.dumps({(0, 1): 'ok'}))\n"
        ),
        timeout_seconds=10,
        initial_state=init_state,
        action_handler=lambda *a, **k: {"executed": False},
        animation_handler=lambda *a, **k: {},
        kept_functions=[],
        retain_imports=True,
        repair_hints=True,
    )
    assert not sb_res1.get("error") and "CALL1_OK" in str(sb_res1.get("stdout", "")), f"Sandbox call 1 failed: {sb_res1}"
    kept_funcs = sb_res1.get("keepable_functions") or []
    kept_names = {f["name"] for f in kept_funcs}
    for req_fn in ("perms6", "build3", "bfs4_unimported", "mp", "setup"):
        assert req_fn in kept_names, f"Expected {req_fn} in keepable_functions, got {kept_names} (rejected={sb_res1.get('retention_rejected')})"

    # Call 2: Redefine build3(fr) WITHOUT redefining KEYP (tr87:6987), define route() referencing D2 (re86:7233),
    # and call bfs4_unimported(99) which used `deque` without importing it in Call 1 (tu93:7138 -> 7240)!
    sb_res2 = run_sandboxed_python(
        code=(
            "assert last_animation_timeline['steps'] == []\n"
            "assert list(last_animation_timeline.items()) == []\n"
            "def build3(fr):\n"
            "    return [r + 1 for (r, ys, bs) in KEYP]\n"
            "def route(d):\n"
            "    return D2[d]\n"
            "seg = current_frame.segmentation\n"
            "print('CALL2_OK', perms6((45, 29), 'N'), build3(None), route('N'), bfs4_unimported(99), setup(), seg['nodes'][0]['bbox'])\n"
        ),
        timeout_seconds=10,
        initial_state=init_state,
        action_handler=lambda *a, **k: {"executed": False},
        animation_handler=lambda *a, **k: {},
        kept_functions=[f["source"] for f in kept_funcs],
        retain_imports=True,
        repair_hints=True,
    )
    assert not sb_res2.get("error") and "CALL2_OK (42, 29, 'S', 8) [5] (-3, 0) 99" in str(sb_res2.get("stdout", "")), f"Sandbox call 2 failed: {sb_res2}"
    kept_funcs2 = {f["name"] for f in (sb_res2.get("keepable_functions") or [])}
    assert "build3" in kept_funcs2 and "route" in kept_funcs2, f"Expected build3 and route in Call 2 kept_funcs, got {kept_funcs2}"
    print("[VERIFY] All v40 local patch, syntax, segmentation, builtins/json hygiene, and 2-turn RF-Norm v2 checks PASSED!")


def main() -> None:
    fast_smoke = "--fast-smoke" in sys.argv
    nb = build_v40_notebook(fast_commit_smoke=fast_smoke)
    verify_v40_locally(nb)

    OUTPUT_V40_PATH.parent.mkdir(parents=True, exist_ok=True)
    STAGE_DIR.mkdir(parents=True, exist_ok=True)

    nb_json = json.dumps(nb, indent=1, ensure_ascii=False) + "\n"
    OUTPUT_V40_PATH.write_text(nb_json, encoding="utf-8")
    OUTPUT_SUB_PATH.write_text(nb_json, encoding="utf-8")

    meta = {
        "id": "zerbytheboss/v36-sovereign-apex-engine",
        "title": "v36-sovereign-apex-engine",
        "code_file": "submission.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "false",
        "machine_shape": "NvidiaRtxPro6000",
        "dataset_sources": [
            "dfranzen/pennyroyal-v253",
            "dfranzen/taaf-kaggle-source-bundle-copy",
        ],
        "competition_sources": [
            "arc-prize-2026-arc-agi-3",
        ],
        "kernel_sources": [],
        "model_sources": [
            "dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1",
            "dfranzen/albucino-qwen3-8-flash-next-drafter/transformers/default/1",
        ],
    }
    METADATA_PATH.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    sha = hashlib.sha256(nb_json.encode("utf-8")).hexdigest()[:12]
    mode_label = "FAST-SMOKE (1 game / 5m)" if fast_smoke else "FULL-BENCHMARK (10 games / 25m)"
    print(f"[BUILD] Wrote {OUTPUT_V40_PATH} and {OUTPUT_SUB_PATH} ({len(nb_json):,} bytes, sha256={sha}, mode={mode_label})")


if __name__ == "__main__":
    main()

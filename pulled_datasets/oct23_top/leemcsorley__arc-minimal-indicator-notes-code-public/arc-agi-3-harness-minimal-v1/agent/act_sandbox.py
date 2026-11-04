"""Interactive `run` sandbox: model code runs in a restricted subprocess and may call
`act([...])` to execute real environment actions, which the parent performs. v2.11: the same worker also runs the
model's code for the harness (loading the code file, the model check, the path search, single predictions), so the
model's code never runs in the harness process.

Protocol (over the worker's stdin/stdout, pickle frames with a 6-byte tag + 8-byte length):
  worker -> parent   ACTREQ {"actions": [...], "repeat": bool}
  parent -> worker   ACTRSP {"ok", "refused", "steps", "new_boards", "stopped_early",
                             "stop_reason", "terminal", "message"}
  worker -> parent   RESULT {"output", "error", "act_calls", "actions_executed"}
  worker -> parent   REFREQ {"method", "args", "kwargs"} (live reference helpers, run only)
  parent -> worker   REFRSP {"result"} or {"error"}
  worker -> parent   ANIREQ {"index"} (minimal run only; one archived animation)
  parent -> worker   ANIRSP {"frames": [(bytes, shape), ...]} or {"error"}

The parent enforces: a compute-time limit that excludes time spent executing actions,
a memory limit, a per-snippet action cap, and the guards (batch no-op stop, stale-state
block, terminal latch) inside `act_callback`.
"""
from __future__ import annotations

import pickle
import queue
import struct
import subprocess
import sys
import threading
import time
from typing import Any, Callable

import numpy as np

RUN_MEM_LIMIT_MB = 512


def _norm_rects(rects: Any) -> list[tuple[int, int, int, int]]:
    if rects is None:
        return []
    rects = list(rects)
    if not rects:
        return []
    if len(rects) == 4 and all(isinstance(x, (int, np.integer)) for x in rects):
        return [tuple(int(x) for x in rects)]
    return [tuple(int(x) for x in r) for r in rects]
# modules a run snippet / the code file may import (v2.4: the same safe stdlib set as the milestone harness)
ALLOWED_RUN_MODULES = {"numpy", "collections", "itertools", "functools", "math", "heapq", "bisect", "re", "copy",
                       "json", "operator", "statistics", "fractions", "string", "random", "time"}   # v2.8: + time
try:
    from .config import SANDBOX_RECURSION_LIMIT as _RECURSION_LIMIT
except Exception:   # pragma: no cover
    _RECURSION_LIMIT = 10000
RUN_TIME_LIMIT_SEC = 30.0
ACT_WAIT_LIMIT_SEC = 600.0  # safety: a single act() round-trip may never block longer than this

_TAG_LEN = 6
_HDR = struct.Struct(">Q")


def _frame(tag: bytes, payload: Any) -> bytes:
    data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    return tag + _HDR.pack(len(data)) + data


def _read_exact(stream: Any, n: int) -> bytes | None:
    buf = b""
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def _read_frame(stream: Any) -> tuple[bytes, Any] | None:
    head = _read_exact(stream, _TAG_LEN + _HDR.size)
    if head is None:
        return None
    tag = head[:_TAG_LEN]
    (length,) = _HDR.unpack(head[_TAG_LEN:])
    data = _read_exact(stream, length)
    if data is None:
        return None
    return tag, pickle.loads(data)


WORKER_SOURCE = r'''
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import sys, io, resource, pickle, struct, types, builtins as _b
import numpy as np

MEM_BYTES = int(sys.argv[1]) * 1024 * 1024
try:
    resource.setrlimit(resource.RLIMIT_AS, (MEM_BYTES, MEM_BYTES))
except Exception:
    pass
try:   # v2.11: lower CPU priority than the LLM server and the game loop
    os.nice(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
except Exception:
    pass

_TAG_LEN = 6
_HDR = struct.Struct(">Q")
_in = sys.stdin.buffer
_out = sys.stdout.buffer

def _read_exact(n):
    buf = b""
    while len(buf) < n:
        chunk = _in.read(n - len(buf))
        if not chunk:
            raise EOFError("parent closed the pipe")
        buf += chunk
    return buf

def _send(tag, payload):
    data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    _out.write(tag + _HDR.pack(len(data)) + data)
    _out.flush()

def _recv():
    head = _read_exact(_TAG_LEN + _HDR.size)
    (length,) = _HDR.unpack(head[_TAG_LEN:])
    return head[:_TAG_LEN], pickle.loads(_read_exact(length))

tag, data = _recv()
assert tag == b"INIT__", tag
JOB = str(data.get("job") or "run")
IN_MODEL = JOB in ("check", "path", "predict")
_ALLOWED_MODULES = set(data.get("allowed_modules") or ["numpy", "collections", "itertools", "functools", "math"])
import traceback as _tbm
import re as _re_w
_LINES = {"<run>": str(data.get("code") or "").split("\n"), "<codefile>": str(data["source"] or "").split("\n")}

# --- harness modules (trusted code, full builtins), registered as real modules so their objects pickle --------------
def _load_mod(name, src):
    m = types.ModuleType(name)
    m.__dict__["__builtins__"] = _b
    sys.modules[name] = m
    exec(compile(src, "<harness " + name + ">", "exec"), m.__dict__)
    return m

_srcs = data["module_sources"]
_OBJ = _load_mod("arc_objects", _srcs["objects"])
_FD = _load_mod("arc_frame_diff", _srcs["frame_diff"])
_GL = _load_mod("arc_gridlib", _srcs["gridlib"])
_MJ = _load_mod("arc_modeljobs", _srcs["modeljobs"])
_CHECKS = _load_mod("arc_reference_checks", _srcs["reference_checks"]) if "reference_checks" in _srcs else None
_MINIMAL = bool(data.get("minimal"))
_MM = _load_mod("arc_minimal_memory", _srcs["minimal_memory"]) if _MINIMAL else None
if _MINIMAL:
    _MM.bind(data["palette"])
    _memory_before = pickle.loads(data.get("memory_blob") or pickle.dumps({"notes": {}}))
color_names = {int(k): str(v) for k, v in (data.get("color_names") or {}).items()}
_GL.bind(_OBJ, _FD, color_names)

def _where(exc):
    """Locate the innermost user-code frame and explain relevant name/scope errors."""
    loc = ""
    for fr in _tbm.extract_tb(exc.__traceback__):
        if fr.filename in _LINES:
            lines = _LINES[fr.filename]
            txt = lines[fr.lineno - 1].strip() if fr.lineno and 0 < fr.lineno <= len(lines) else ""
            loc = (f" [line {fr.lineno} of your {'run code' if fr.filename == '<run>' else 'code file'}"
                   + (f": {txt[:160]}" if txt else "") + "]")
    if isinstance(exc, NameError):
        nm = getattr(exc, "name", None)
        if not nm:
            m = _re_w.search(r"name '([^']+)'", str(exc))
            nm = m.group(1) if m else None
        if _MINIMAL and JOB == "run" and loc:
            if isinstance(exc, UnboundLocalError):
                loc += (" [Python scope] This function reads a local variable before assigning it. "
                        "Assign it on every path before use, or pass it as an argument.")
            else:
                loc += (" [Python scope] Plain variables and import aliases do NOT persist between run() calls, "
                        "even within the same turn. "
                        + (f"Define {nm!r}" if nm else "Define the missing name")
                        + " in this call (check spelling), or retrieve its value from store/notes if previously saved there. "
                        "Save reusable values in store or notes. Retained helpers keep definitions, not earlier run variables; "
                        "pass needed values as arguments or read store/notes.")
        elif not _MINIMAL and nm and nm in _GL.BARE_HINTS:
            loc += f" ({nm} is {_GL.BARE_HINTS[nm]} now)"
    return loc

# --- restricted namespace for the model's code ----------------------------------------
_SAFE_NAMES = [
    "abs", "min", "max", "sum", "round", "len", "range", "enumerate",
    "zip", "map", "filter", "sorted", "reversed", "all", "any",
    "int", "float", "bool", "str", "list", "tuple", "dict", "set",
    "frozenset", "print", "isinstance", "divmod", "pow", "slice",
    "next", "iter", "repr", "type", "bin", "hex", "oct", "ord", "chr", "format", "callable", "hash",
    "hasattr", "id", "issubclass", "object",
    "True", "False", "None", "Exception", "ValueError", "TypeError",
    "KeyError", "IndexError", "RuntimeError", "StopIteration",
    "ZeroDivisionError", "AssertionError", "AttributeError", "NameError", "NotImplementedError",
    "ArithmeticError", "LookupError", "OverflowError", "RecursionError",
]
SAFE_BUILTINS = {n: getattr(_b, n) for n in _SAFE_NAMES if hasattr(_b, n)}

def _guarded_import(name, *a, **kw):
    if name.split(".")[0] not in _ALLOWED_MODULES:
        raise ImportError(f"import of {name!r} is not allowed")
    return _b.__import__(name, *a, **kw)
SAFE_BUILTINS["__import__"] = _guarded_import

class ActStopped(BaseException):
    """Raised inside the snippet when further actions are refused; cannot be caught by
    `except Exception` so a model loop cannot swallow it."""

store = {}
if data.get("store_blob"):
    try:
        store = pickle.loads(data["store_blob"])
    except Exception:
        store = {}

ns = {"__builtins__": SAFE_BUILTINS, "np": np}
for _m in sorted(_ALLOWED_MODULES):
    if _m in ("numpy", "np"):
        continue
    try:
        ns[_m] = _b.__import__(_m)
    except Exception:
        pass
try:
    sys.setrecursionlimit(max(1000, int(data.get("recursion_limit") or 1000)))
except Exception:
    pass
buf = io.StringIO()
def _print(*args, **kwargs):
    kwargs.pop("file", None)
    kwargs.pop("flush", None)
    print(*args, file=buf, **kwargs)
def _no_print(*args, **kwargs):
    return None
ns["print"] = _no_print if IN_MODEL else _print
if data.get("full_array_print"):
    np.set_printoptions(threshold=4096, linewidth=1000000)

class _CpuLimit(BaseException):
    pass

def _on_cpu_limit(signum, frame):
    raise _CpuLimit()

try:   # stop just before the parent's limit so the results so far are returned
    import signal as _signal
    _signal.signal(_signal.SIGVTALRM, _on_cpu_limit)
    _signal.setitimer(_signal.ITIMER_VIRTUAL, max(1.0, float(data.get("time_limit") or 30.0) - 2.0))
except Exception:
    pass

states = np.frombuffer(data["states_bytes"], dtype=np.uint8).reshape(data["states_shape"]).copy()
state_list = [states[i] for i in range(len(states))]
actions_history = list(data["actions_history"])
indicator_rect = [tuple(r) for r in (data.get("indicator_rect") or [])]   # list of [r0,r1,c0,c1] rects (usually 0 or 1)
anim_frames = [np.frombuffer(b, dtype=np.uint8).reshape(shape) for b, shape in data.get("anim_frames") or []]
history = list(data.get("history") or [])
past_levels = list(data.get("past_levels") or [])
current_attempt = int(data.get("current_attempt") or 1)
act_enabled = bool(data.get("act_enabled"))
max_actions = int(data.get("max_actions") or 12)
scene_list = list(data.get("scenes") or [])          # v2.6.5: tracker scene (object table) per board of states, or None
scene_edits = list(data.get("scene_edits") or [])    # per transition: {"lines": [...], "edits": [...]}
scene_on = bool(data.get("scene_on"))
object_refs_data = dict(data.get("object_refs") or {})
_references_live = False

def _reference_request(method, args, kwargs):
    if JOB != "run" or not _references_live:
        raise ValueError("live object references are available only in run, not during code loading/model prediction")
    if _latched:
        raise ActStopped(_latched)
    if not data.get("references_enabled"):
        raise ValueError("object references are unavailable here")
    _send(b"REFREQ", {"method": method, "args": args, "kwargs": kwargs})
    tag, rsp = _recv()
    assert tag == b"REFRSP", tag
    if rsp.get("error"):
        raise ValueError(rsp["error"])
    if "object_refs" in rsp:
        object_refs_data.clear()
        object_refs_data.update(rsp["object_refs"])
    return rsp.get("result")

# v2.11: the lib namespace (generic grid tools), the same in run and inside next_state
def _make_lib():
    return _GL.make_lib({"get_states": lambda: state_list, "get_history": lambda: history,
                    "get_actions": lambda: actions_history, "rects": indicator_rect, "view": data.get("view"),
                    "names": color_names, "anim_frames": anim_frames, "action_labels": data.get("action_labels") or [],
                    "printer": ns["print"], "in_model": IN_MODEL,
                    "get_object_refs": lambda: object_refs_data, "reference_request": _reference_request})
lib = _make_lib()
ns["lib"] = lib
if _MINIMAL:
    ns.pop("lib")
    def _minimal_values(value):
        if isinstance(value, dict):
            result = {k: _minimal_values(v) for k, v in value.items()
                      if k not in ("id", "ref", "from_id", "to_id")}
            if isinstance(value, _GL.Obj):
                rect = result.get("rect")
                result["center"] = (((rect[0] + rect[1] - 1) / 2,
                                     (rect[2] + rect[3] - 1) / 2) if rect is not None else None)
                return _GL.MinimalComponent(result)
            return result
        if isinstance(value, (list, tuple)):
            return [_minimal_values(v) for v in value]
        if isinstance(value, str):
            return value.replace("lib.anim_diff()", "anim_diff()").replace(
                "the remaining-actions indicator", "your marked resource/counter region")
        return value
    def extract_components(board=None, only=None, exclude=None):
        return _minimal_values(lib.extract_components(board, only, exclude))
    def find(board, o, min_score=1.0, rotate=False, flip=False, match_colors=True):
        return _minimal_values(lib.find(board, o, min_score, rotate, flip, match_colors))
    def frame_diff(before=None, after=None):
        return _minimal_values(lib.frame_diff(before, after))
    def anim_diff(cap=40):
        return _minimal_values(lib.anim_diff(cap))
    def board_key(board):
        return lib.board_key(board)
    def click_candidates(board=None):
        return _minimal_values(lib.click_candidates(board))
    def _indicator_request(method, value=None):
        if JOB != "run" or not _references_live:
            raise ValueError("indicator helpers are available inside run, not while loading saved helpers")
        if _latched:
            raise ActStopped(_latched)
        _send(b"INDREQ", {"method": method, "value": value})
        tag, rsp = _recv()
        if tag != b"INDRSP":
            raise RuntimeError("indicator pipe failed")
        if rsp.get("error"):
            raise ValueError(rsp["error"])
        return rsp["result"]
    def mark_remaining_actions_indicator(rects):
        global lib
        result = _indicator_request("mark", rects)
        indicator_rect[:] = [tuple(r) for r in result]
        lib = _make_lib()
        for transition in history:
            before, after = transition.get("before"), transition.get("after")
            if before is not None and after is not None and before.shape == after.shape:
                changed = before != after
                for r0, r1, c0, c1 in indicator_rect:
                    changed[r0:r1, c0:c1] = False
                transition["changed_outside_indicator"] = int(np.count_nonzero(changed))
        return tuple(indicator_rect)
    def indicator():
        return _indicator_request("facts")
    def get_animation(i=-1):
        if JOB != "run":
            raise RuntimeError("get_animation is available only inside run")
        if isinstance(i, bool) or not isinstance(i, (int, np.integer)):
            raise TypeError("get_animation expects an integer index into history")
        if not -len(history) <= i < len(history):
            raise IndexError("get_animation index is outside this level's history")
        index = history[i].get("animation_index")
        if index is None:
            raise LookupError("animation unavailable: this action has no confirmed recorded transition")
        _send(b"ANIREQ", {"index": index})
        tag, rsp = _recv()
        if tag != b"ANIRSP":
            raise RuntimeError("animation archive pipe failed")
        if rsp.get("error"):
            raise LookupError(rsp["error"])
        return [np.frombuffer(raw, dtype=np.uint8).reshape(shape).copy() for raw, shape in rsp["frames"]]
    def render(value, *, rect=None, mask=None, title="", size=640):
        image = _MM.render(value, rect=rect, mask=mask, title=title, size=size)
        image["label"] += f"; rendered in attempt {current_attempt}, after {len(actions_history)} action(s)"
        return image
    def show(value):
        if not _MM.is_image(value):
            return lib.show(value)
        _MM.validate(value)
        _send(b"SHOWIM", value)
        tag, rsp = _recv()
        if tag != b"SHOWOK":
            raise RuntimeError("image output pipe failed")
        if rsp.get("error"):
            raise ValueError(rsp["error"])
        _print("[Image shown: " + rsp["label"] + "]")
    ns.update({"notes": _MM.copy.deepcopy(_memory_before["notes"]),
               "COLOR_NAMES": dict(color_names),
               "render": render, "extract_components": extract_components,
               "find": find, "frame_diff": frame_diff, "anim_diff": anim_diff,
               "mark_remaining_actions_indicator": mark_remaining_actions_indicator,
               "indicator": indicator, "board_key": board_key, "click_candidates": click_candidates,
               "get_animation": get_animation,
               "show": show, "describe": _GL.describe})

_load_error = None
source = data["source"]
_reserved_names = set(data.get("reserved_names") or ())
_missing_binding = object()
_protected_bindings = {n: ns.get(n, _missing_binding) for n in _reserved_names}
for _warning in data.get("source_warnings") or []:
    _print("[Saved helpers] Not loaded: " + _warning)
try:
    if source:
        exec(compile(source, "<codefile>", "exec"), ns)
except _CpuLimit:
    _load_error = "loading it did not finish within the time limit (top-level code should only define functions and constants)"
except MemoryError:
    _load_error = "MEMORY_LIMIT"
except BaseException as exc:
    _load_error = repr(exc) + _where(exc)
finally:
    # Defence for older saved assignments/import aliases as well as helper defs.
    # This does not prevent ordinary local shadowing inside the current snippet.
    _overwritten = []
    for _name, _original in _protected_bindings.items():
        if ns.get(_name, _missing_binding) is not _original:
            _overwritten.append(_name)
            if _original is _missing_binding:
                ns.pop(_name, None)
            else:
                ns[_name] = _original
    if _overwritten:
        _print("[Saved helpers] Restored reserved names: " + ", ".join(sorted(_overwritten))
               + ". Rename those saved bindings; the provided harness/Python names remain active.")
if _load_error is not None:
    try:
        _signal.setitimer(_signal.ITIMER_VIRTUAL, 0)
    except Exception:
        pass
    _send(b"RESULT", {"output": buf.getvalue(), "error": ("MEMORY_LIMIT" if _load_error == "MEMORY_LIMIT" else
                                                      "your code file failed to load: " + _load_error),
                      "store": None, "store_note": None, "act_calls": 0, "actions_executed": 0, "load_failed": True})
    sys.exit(0)

if JOB == "load":
    try:
        _signal.setitimer(_signal.ITIMER_VIRTUAL, 0)
    except Exception:
        pass
    _send(b"RESULT", {"output": buf.getvalue(), "error": None, "store": None, "store_note": None, "act_calls": 0,
                      "actions_executed": 0, "job_result": {"ok": True}})
    sys.exit(0)

_act_calls = 0
_stops_raised = 0
_actions_executed = 0
_latched = None  # message once no more actions may be executed in this snippet

_snip = None   # the snippet's own namespace (a copy of ns): snippet assignments never overwrite code-file names

def _refresh():
    vals = {"states": np.stack(state_list, axis=0) if state_list else np.zeros((0, 64, 64), dtype=np.uint8),
            "current_board": state_list[-1] if state_list else None,
            "actions_history": actions_history}
    ns.update(vals)
    if _snip is not None:
        _snip.update(vals)

def changes(n=1):
    """The layered tracker's explanation of the last n actions of this attempt (oldest first): a list of dicts
    {action, lines (every line, uncapped), edits (the tracker's raw edits: type, id, from, to, ...)}."""
    out = []
    m = len(scene_edits)
    for i in range(max(0, m - int(n)), m):
        e = scene_edits[i] or {}
        out.append({"action": actions_history[i] if i < len(actions_history) else None,
                    "lines": list(e.get("lines") or []), "edits": list(e.get("edits") or [])})
    return out

def act(actions, repeat=False, force=None, expect=None):
    """Execute environment actions now. `actions` is a list like ["UP", "RIGHT", "CLICK 10 20 3"]
    (a single string is accepted). repeat=True executes an action even if it is a known no-op / known fatal
    action in this exact board state. Returns a dict with per-step results; see the prompt."""
    if force is not None:
        repeat = bool(force) or bool(repeat)   # legacy name
    global _act_calls, _actions_executed, _latched, _stops_raised
    if not act_enabled:
        raise RuntimeError("act() is not available here" + (" (inside next_state)" if IN_MODEL else ""))
    if isinstance(actions, str):
        actions = [actions]
    actions = [str(a).strip() for a in actions if str(a).strip()]
    if _latched:
        _stops_raised += 1
        raise ActStopped(_latched)
    if not actions:
        return {"executed": [], "stopped_early": False, "stop_reason": None, "terminal": False}
    remaining = max_actions - _actions_executed
    if remaining <= 0:
        _latched = f"action cap reached: at most {max_actions} actions may be executed per run call"
        _stops_raised += 1
        raise ActStopped(_latched)
    truncated = len(actions) > remaining
    actions = actions[:remaining]
    _act_calls += 1
    request = {"actions": actions, "repeat": bool(repeat)}
    if expect is not None:
        if _CHECKS is None:
            raise ValueError("expectations are unavailable")
        request["expect"] = _CHECKS.expectations(expect)
    _send(b"ACTREQ", request)
    tag, rsp = _recv()
    assert tag == b"ACTRSP", tag
    if "object_refs" in rsp:
        object_refs_data.clear()
        object_refs_data.update(rsp["object_refs"] or {})
    if rsp.get("refused"):
        _latched = rsp["refused"]
        _stops_raised += 1
        raise ActStopped(rsp["refused"])
    _n_before = len(state_list)
    for board_bytes, shape in rsp.get("new_boards") or []:
        state_list.append(np.frombuffer(board_bytes, dtype=np.uint8).reshape(shape).copy())
    if scene_on:
        while len(scene_list) < _n_before:
            scene_list.append(None)
        for tb in rsp.get("new_scenes") or []:
            scene_list.append(tb)
        for ed in rsp.get("new_edits") or []:
            scene_edits.append(ed)
    for _k, step in enumerate(rsp.get("steps") or []):
        actions_history.append(step["action"])
        _bi = _n_before - 1 + _k
        _before = state_list[_bi] if 0 <= _bi < len(state_list) else None
        _after = state_list[_bi + 1] if 0 <= _bi + 1 < len(state_list) else None
        history.append({"attempt": current_attempt, "action": step["action"], "outcome": step.get("outcome"),
                        "changed_cells": step.get("changed_cells"),
                        "changed_outside_indicator": step.get("changed_outside_indicator"),
                        "before": _before, "after": _after})
        if _MINIMAL:
            history[-1]["animation_index"] = step.get("animation_index")
    _actions_executed += len(rsp.get("steps") or [])
    anim_frames[:] = [np.frombuffer(b, dtype=np.uint8).reshape(shape) for b, shape in rsp.get("anim_frames") or []]
    _refresh()
    if rsp.get("abort"):
        # a rejected / invalid action ends the run call (state may differ from what the code planned for)
        _latched = rsp["abort"]
        _r = {"executed": [s["action"] for s in rsp.get("steps") or []], "steps": rsp.get("steps") or [],
              "stopped_early": True, "stop_reason": rsp["abort"], "terminal": False, "outcome": None}
        ns["last_act_result"] = _r
        if _snip is not None:
            _snip["last_act_result"] = _r
        _stops_raised += 1
        raise ActStopped(rsp["abort"])
    result = {
        "executed": [s["action"] for s in rsp.get("steps") or []],
        "steps": rsp.get("steps") or [],
        "stopped_early": bool(rsp.get("stopped_early")) or truncated,
        "stop_reason": rsp.get("stop_reason") or ("action cap" if truncated else None),
        "terminal": bool(rsp.get("terminal")),
        "outcome": rsp.get("outcome"),
        "message": rsp.get("message"),
    }
    if rsp.get("latch"):
        _latched = rsp["latch"]
    ns["last_act_result"] = result
    if _snip is not None:
        _snip["last_act_result"] = result
    if result["terminal"] and result["outcome"] == "reset":
        # RESET is applied by the parent after this worker returns. Finish now,
        # without unwinding model frames: even bare except/finally blocks must
        # not read the old board or save conclusions from it after acceptance.
        try:
            _finish_run(None, reset_stopped=True)
        finally:
            os._exit(0)
    return result

_references_live = JOB == "run"
_refresh()
ns.update({
    "print": ns["print"], "act": act, "last_act_result": {}, "anim_frames": anim_frames,
    "indicator_rects": indicator_rect, "history": history, "past_levels": past_levels, "CODE": source, "store": store,
})
_proposals = []
if _MINIMAL:
    # The Python signature itself rejects expect=; marking is a separate parent-owned API.
    _legacy_act = act
    def act(actions, repeat=False, force=None):
        return _legacy_act(actions, repeat=repeat, force=force)
    ns["act"] = act
    ns.pop("indicator_rects", None)
    def propose_actions(candidates, probe_omission_reason=""):
        # Snapshot at call time. Only the parent controller may validate/execute it,
        # after successful completion of this worker and only during recovery.
        if len(_proposals) >= 2:
            raise ValueError("submit exactly one proposal call per response")
        _raw = _b.__import__("json").dumps({"candidates": candidates, "probe_omission_reason": probe_omission_reason})
        if len(_raw) > 8192:
            raise ValueError("proposal exceeds 8192 characters")
        _proposals.append(_raw)
    ns["propose_actions"] = propose_actions
if scene_on:
    ns["changes"] = changes

if IN_MODEL:
    _jd = data.get("job_data") or {}
    _mask = None
    if indicator_rect and state_list:
        _mask = np.zeros(state_list[-1].shape, dtype=bool)
        for (_r0, _r1, _c0, _c1) in indicator_rect:
            _mask[_r0:_r1, _c0:_c1] = True
    _res, _err = None, None

    def _arm(seconds):
        try:
            _signal.setitimer(_signal.ITIMER_VIRTUAL, max(0.5, float(seconds)))
        except Exception:
            pass
    try:
        if JOB == "check":
            _res = _MJ.check(ns, history, list(_jd.get("recs") or []), _mask, time_limit=float(_jd.get("budget") or 60.0),
                             limit_exc=(_CpuLimit,), where=_where, arm=_arm, per_call=float(_jd.get("per_call") or 5.0))
        elif JOB == "path":
            _start = np.frombuffer(_jd["start"], dtype=np.uint8).reshape(_jd["shape"]).copy()
            _res = _MJ.path(ns, _start, list(_jd["specs"]), _mask, max_states=int(_jd["max_states"]),
                            max_depth=int(_jd["max_depth"]), time_limit=float(_jd["budget"]),
                            click_cells=_GL.click_cells, limit_exc=(_CpuLimit,))
        elif JOB == "predict":
            _pairs = [dict(p, before=np.frombuffer(p["before"], dtype=np.uint8).reshape(p["shape"]).copy())
                      for p in _jd.get("pairs") or []]
            _res = _MJ.predict(ns, _pairs, limit_exc=(_CpuLimit,), where=_where, arm=_arm)
    except _CpuLimit:
        _err = "did not finish within the time limit"
    except MemoryError:
        _err = "MEMORY_LIMIT"
    except BaseException as exc:
        _err = repr(exc)[:400] + _where(exc)
    try:
        _signal.setitimer(_signal.ITIMER_VIRTUAL, 0)
    except Exception:
        pass
    try:
        _send(b"RESULT", {"output": "", "error": _err, "store": None, "store_note": None, "act_calls": 0,
                          "actions_executed": 0, "job_result": _res})
    except MemoryError:
        _send(b"RESULT", {"output": "", "error": "MEMORY_LIMIT", "store": None, "store_note": None, "act_calls": 0,
                          "actions_executed": 0, "job_result": None})
    sys.exit(0)

def _finish_run(error, reset_stopped=False):
    """Flush completed work both on normal exit and at an accepted RESET."""
    global store
    try:
        _signal.setitimer(_signal.ITIMER_VIRTUAL, 0)
    except Exception:
        pass
    _new_store = _snip.get("store")
    if isinstance(_new_store, dict) and _new_store is not store:
        store = _new_store
    _store_out, _store_note = None, None
    if not _timed_out:
        _max_bytes = int(data.get("store_max_mb") or 64) * 1024 * 1024
        _bad = []
        try:
            pickle.dumps(store, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            for _k in list(store.keys()):
                try:
                    pickle.dumps(store[_k], protocol=pickle.HIGHEST_PROTOCOL)
                except Exception as _exc:
                    _bad.append(f"{_k!r} ({type(_exc).__name__})")
                    del store[_k]
        try:
            _blob = pickle.dumps(store, protocol=pickle.HIGHEST_PROTOCOL)
            if len(_blob) > _max_bytes:
                _store_note = (f"store changes from this call NOT saved (the previous store is kept): {len(_blob) // (1024 * 1024)} MB "
                               f"exceeds the {_max_bytes // (1024 * 1024)} MB limit")
            else:
                _store_out = store
                if _bad:
                    _store_note = ("store keys that cannot be saved (functions, lambdas, open objects) were dropped: "
                                   + ", ".join(_bad[:10]) + "; everything else was saved")
        except Exception as _exc:
            _store_note = f"store changes from this call NOT saved (the previous store is kept): {_exc!r}"
    _memory_out, _memory_messages = None, []
    if _MINIMAL:
        _memory_out = {}
        for _kind in ("notes",):
            _saved, _msgs = _MM.commit(_memory_before[_kind], _snip.get(_kind), _kind)
            _memory_out[_kind] = _saved
            _memory_messages.extend(_msgs)
    # Do not retain defs that occur after RESET, including later redefinitions
    # of an already defined name. Normal runs keep their existing contract.
    _defined = ({name: fn.__code__.co_firstlineno for name, fn in _snip.items()
                 if isinstance(fn, types.FunctionType) and fn.__code__.co_filename == "<run>"}
                if reset_stopped else None)
    _send(b"RESULT", {"output": buf.getvalue(), "error": error, "store": _store_out, "store_note": _store_note,
                      "memory": _memory_out, "memory_messages": _memory_messages,
                      "proposals": _proposals, "reset_stopped": reset_stopped, "defined_functions": _defined,
                      "act_calls": _act_calls, "actions_executed": _actions_executed})

error = None
_timed_out = False
_snip = dict(ns)   # snippet-local namespace: shares the objects (store, lists) but not the name bindings
try:
    exec(compile(data["code"], "<run>", "exec"), _snip)
except ActStopped as exc:
    error = f"STOPPED: {exc}"
except _CpuLimit:
    _timed_out = True
    error = (f"Execution timed out (about {float(data.get('time_limit') or 30.0):.0f} s compute limit, time spent inside act() "
             "excluded); the output printed before the limit is shown, store changes from this call were NOT saved")
except MemoryError:
    error = "MEMORY_LIMIT"
except Exception as exc:
    error = repr(exc) + _where(exc)
if error is None and _stops_raised and _latched:
    error = f"STOPPED (the stop was caught by an except in your code; later act() calls did nothing): {_latched}"
_finish_run(error)
'''


def _job_slots() -> int:
    try:
        from .config import WORKER_JOB_SLOTS
        return max(1, int(WORKER_JOB_SLOTS))
    except Exception:   # pragma: no cover
        import os
        return max(2, (os.cpu_count() or 4) - 2)


_JOB_SEM = threading.BoundedSemaphore(_job_slots())
try:
    from .config import WORKER_NICE as _NICE
except Exception:   # pragma: no cover
    _NICE = 10


def run_interactive(
    *,
    source: str,
    code: str,
    states: np.ndarray,
    actions_history: list[str],
    module_sources: dict[str, str],
    act_callback: Callable[[list[str], bool], dict[str, Any]] | None,
    indicator_rect: Any = None,          # list of [r0,r1,c0,c1] rects (or one rect, or None)
    store: dict[str, Any] | None = None,
    color_names: dict[int, str] | None = None,
    anim_frames: list[np.ndarray] | None = None,
    history: list[dict[str, Any]] | None = None,
    past_levels: list[dict[str, Any]] | None = None,
    current_attempt: int = 1,
    max_actions: int = 12,
    time_limit: float = RUN_TIME_LIMIT_SEC,
    mem_limit_mb: int = RUN_MEM_LIMIT_MB,
    scenes: list[Any] | None = None,
    scene_edits: list[Any] | None = None,
    scene_on: bool = False,
    full_array_print: bool = False,
    store_max_mb: int = 64,
    job: str = "run",
    job_data: dict[str, Any] | None = None,
    view: list[int] | None = None,
    action_labels: list[str] | None = None,
    object_refs: dict[str, Any] | None = None,
    reference_callback: Callable | None = None,
    minimal: bool = False,
    memory: dict | None = None,
    palette: dict | None = None,
    image_callback: Callable | None = None,
    animation_callback: Callable | None = None,
    indicator_callback: Callable | None = None,
) -> tuple[str, str | None, dict[str, Any]]:
    """Run `code` (with the model's code file `source` pre-executed) in a fresh worker process, or a model job
    (job = "load" / "check" / "path" / "predict", v2.11) that only runs the code file and next_state there.

    Returns (output, error, info) where info has act_calls, actions_executed, compute_seconds, store, job_result.
    """
    reserved_names, source_warnings = (), []
    if minimal:
        from .sandbox import MINIMAL_RESERVED_NAMES, strip_reserved_helpers
        reserved_names = sorted(MINIMAL_RESERVED_NAMES)
        source, source_warnings = strip_reserved_helpers(source or "", MINIMAL_RESERVED_NAMES)
    try:
        store_blob = pickle.dumps(dict(store or {}), protocol=pickle.HIGHEST_PROTOCOL) if store else b""
    except Exception as exc:   # never run with an empty store: the result would replace the real one
        return "", f"Runtime error: the store could not be passed to the sandbox ({exc!r})", {
            "act_calls": 0, "actions_executed": 0, "compute_seconds": 0.0, "store": None, "store_note": None, "job_result": None}
    init = {
        "minimal": minimal,
        "memory_blob": pickle.dumps(memory or {"notes": {}}, protocol=4),
        "palette": palette,
        "job": job,
        "job_data": job_data or {},
        "module_sources": module_sources,
        "source": source or "",
        "reserved_names": reserved_names,
        "source_warnings": source_warnings,
        "code": code,
        "states_bytes": np.ascontiguousarray(states, dtype=np.uint8).tobytes(),
        "states_shape": tuple(states.shape),
        "actions_history": list(actions_history),
        "indicator_rect": [list(r) for r in _norm_rects(indicator_rect)],
        "store_blob": store_blob,
        "color_names": {int(k): str(v) for k, v in (color_names or {}).items()},
        "anim_frames": [(np.ascontiguousarray(f, dtype=np.uint8).tobytes(), tuple(f.shape)) for f in (anim_frames or [])],
        "history": list(history or []),
        "past_levels": list(past_levels or []),
        "current_attempt": int(current_attempt),
        "act_enabled": act_callback is not None,
        "allowed_modules": sorted(ALLOWED_RUN_MODULES),
        "recursion_limit": int(_RECURSION_LIMIT),
        "max_actions": int(max_actions),
        "scenes": list(scenes or []),
        "scene_edits": list(scene_edits or []),
        "scene_on": bool(scene_on),
        "full_array_print": bool(full_array_print),
        "store_max_mb": int(store_max_mb),
        "time_limit": float(time_limit),
        "view": list(view) if view else None,
        "action_labels": list(action_labels or []),
        "object_refs": dict(object_refs or {}),
        "references_enabled": reference_callback is not None,
    }
    slot = job in ("run", "check", "path")
    if slot:
        _JOB_SEM.acquire()
    try:
        return _run_worker(init, act_callback, time_limit, mem_limit_mb, reference_callback, image_callback,
                           animation_callback, indicator_callback)
    finally:
        if slot:
            _JOB_SEM.release()


def _run_worker(init: dict[str, Any], act_callback: Any, time_limit: float, mem_limit_mb: int, reference_callback=None,
                image_callback=None, animation_callback=None, indicator_callback=None
                ) -> tuple[str, str | None, dict[str, Any]]:
    proc = subprocess.Popen(
        [sys.executable, "-c", WORKER_SOURCE, str(mem_limit_mb + 300), str(int(_NICE))],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        info: dict[str, Any] = {"act_calls": 0, "actions_executed": 0, "compute_seconds": 0.0}
        frames: "queue.Queue[tuple[bytes, Any] | None]" = queue.Queue()

        def reader() -> None:
            try:
                while True:
                    fr = _read_frame(proc.stdout)
                    frames.put(fr)
                    if fr is None or fr[0] == b"RESULT":
                        return
            except Exception:
                frames.put(None)

        t = threading.Thread(target=reader, daemon=True)
        t.start()

        def kill() -> None:
            try:
                proc.kill()
            except Exception:
                pass

        try:
            proc.stdin.write(_frame(b"INIT__", init))
            proc.stdin.flush()
        except Exception as exc:
            kill()
            return "", f"Runtime error: could not start the sandbox ({exc!r})", info

        # the worker stops itself 2 s before time_limit (CPU time); the parent allows some extra wall time for a niced
        # process on a busy machine before it kills it
        wall_limit = time_limit * 1.5 + 10.0
        compute_used = 0.0
        last_mark = time.perf_counter()
        while True:
            remaining = wall_limit - compute_used
            if init.get("job", "run") == "run":
                timeout_msg = (f"Execution timed out ({time_limit:.0f}s compute limit, time spent inside act() excluded). "
                               "Keep run() snippets short; for large searches use `path` or reduce the search space.")
                if init.get("minimal"):
                    timeout_msg = (f"Execution timed out ({time_limit:.0f}s compute limit, excluding act time). "
                                   "Split computation across runs using store. Unsaved memory writes may be lost; "
                                   "executed actions are reported by the parent harness.")
            else:
                timeout_msg = f"the {init.get('job')} job did not finish within {wall_limit:.0f} s and was stopped"
            if remaining <= 0:
                kill()
                return "", timeout_msg, info
            try:
                fr = frames.get(timeout=remaining)
            except queue.Empty:
                kill()
                return "", timeout_msg, info
            now = time.perf_counter()
            compute_used += now - last_mark
            last_mark = now
            if fr is None:
                try:
                    proc.wait(timeout=5)
                except Exception:
                    kill()
                stderr = (proc.stderr.read() or b"").decode(errors="replace").strip()
                if "MemoryError" in stderr or "Cannot allocate memory" in stderr:
                    return "", f"Memory limit exceeded ({mem_limit_mb}MB).", info
                return "", f"Runtime error (sandbox process ended unexpectedly): {stderr[-800:]}", info
            tag, payload = fr
            if tag == b"INDREQ":
                info["indicator_requests"] = info.get("indicator_requests", 0) + 1
                try:
                    if not init.get("minimal") or init.get("job") != "run" or indicator_callback is None:
                        raise ValueError("indicator helpers are unavailable in this job")
                    rsp = {"result": indicator_callback(payload["method"], payload.get("value"))}
                except Exception as exc:
                    rsp = {"error": str(exc)[:600]}
                try:
                    proc.stdin.write(_frame(b"INDRSP", rsp))
                    proc.stdin.flush()
                except Exception as exc:
                    kill()
                    return "", f"Runtime error: indicator pipe broke ({exc!r})", info
                continue
            if tag == b"ANIREQ":
                info["animation_requests"] = info.get("animation_requests", 0) + 1
                try:
                    if not init.get("minimal") or init.get("job") != "run" or animation_callback is None:
                        raise ValueError("animation archive is unavailable in this job")
                    rsp = {"frames": animation_callback(payload["index"])}
                except Exception as exc:
                    rsp = {"error": str(exc)[:600]}
                try:
                    proc.stdin.write(_frame(b"ANIRSP", rsp))
                    proc.stdin.flush()
                except Exception as exc:
                    kill()
                    return "", f"Runtime error: animation archive pipe broke ({exc!r})", info
                # Retrieval counts toward the existing compute budget.
                continue
            if tag == b"SHOWIM":
                try:
                    if not init.get("minimal") or init.get("job") != "run" or image_callback is None:
                        raise ValueError("image output is unavailable in this job")
                    rsp = {"label": image_callback(payload)}
                except Exception as exc:
                    rsp = {"error": str(exc)[:600]}
                try:
                    proc.stdin.write(_frame(b"SHOWOK", rsp))
                    proc.stdin.flush()
                except Exception as exc:
                    kill()
                    return "", f"Runtime error: image output pipe broke ({exc!r})", info
                continue
            if tag == b"REFREQ":
                try:
                    if init.get("job") != "run" or reference_callback is None:
                        raise ValueError("object references are unavailable in this job")
                    result = reference_callback(payload["method"], payload.get("args", []), payload.get("kwargs", {}))
                    rsp = {"result": result}
                except Exception as exc:
                    rsp = {"error": str(exc)[:600]}
                try:
                    proc.stdin.write(_frame(b"REFRSP", rsp))
                    proc.stdin.flush()
                except Exception as exc:
                    kill()
                    return "", f"Runtime error: reference pipe broke ({exc!r})", info
                # Reference computations count toward the sandbox's compute budget.
                continue
            if tag == b"ACTREQ":
                t0 = time.perf_counter()
                try:
                    args = (list(payload.get("actions") or []), bool(payload.get("repeat")))
                    rsp = (act_callback(*args, payload["expect"]) if "expect" in payload else act_callback(*args))
                except Exception as exc:  # harness-side failure must not hang the worker
                    rsp = {"refused": f"harness error while executing actions: {exc!r}", "steps": [], "new_boards": []}
                info["act_calls"] += 1
                info["actions_executed"] += len(rsp.get("steps") or [])
                info.setdefault("action_results", []).append(rsp)
                try:
                    proc.stdin.write(_frame(b"ACTRSP", rsp))
                    proc.stdin.flush()
                except Exception as exc:
                    kill()
                    return "", f"Runtime error: sandbox pipe broke during act() ({exc!r})", info
                # time inside the harness does not count against the snippet
                last_mark = time.perf_counter()
                if last_mark - t0 > ACT_WAIT_LIMIT_SEC:
                    kill()
                    return "", "Runtime error: act() took too long on the harness side", info
                continue
            if tag == b"RESULT":
                try:
                    proc.stdin.close()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=10)
                except Exception:
                    kill()
                # Parent counts survive timeout, broken pipe, and missing/faulty worker results.
                info["compute_seconds"] = compute_used
                info["store"] = payload.get("store")
                info["store_note"] = payload.get("store_note")
                info["job_result"] = payload.get("job_result")
                info["load_failed"] = bool(payload.get("load_failed"))
                info["memory"] = payload.get("memory")
                info["memory_messages"] = payload.get("memory_messages") or []
                info["proposals"] = payload.get("proposals") or []
                info["reset_stopped"] = bool(payload.get("reset_stopped"))
                info["defined_functions"] = payload.get("defined_functions")
                output = payload.get("output", "")
                error = payload.get("error")
                if error == "MEMORY_LIMIT":
                    return output, f"Memory limit exceeded ({mem_limit_mb}MB).", info
                if error:
                    return output, f"Runtime error: {error}", info
                return output, None, info
            kill()
            return "", f"Runtime error: unexpected sandbox frame {tag!r}", info
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=5)
        if 't' in locals():
            t.join(timeout=2)
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            if stream is not None:
                stream.close()

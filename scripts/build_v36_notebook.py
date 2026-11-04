#!/usr/bin/env python3
"""
scripts/build_v35_notebook.py
=============================
Deterministic Builder for ARC Prize 2026 — Version 35 Sovereign Apex Engine.

Architectural Synthesis:
1. Purges all 5 regressions from v33/v34:
   - Removes `patch_sandbox_primitives.py` (eliminates `board` global namespace collision).
   - Removes `ARC3_PROBE_FIRST = '1'` (eliminates 4,096-token mid-`<think>` truncation).
   - Removes `ARC3_NOOP_REPEAT_GUARD = '1'` and `ARC3_HISTORY_ASSISTANT_TURNS = 0` (restores 150).
   - Removes `SOLVER_TIMEOUT_AFTER = 45.0` floor (restores clean soft-deadline wind-down).
   - Restores `SERVER_STARTUP_TIMEOUT = (12 * 60) if TRUE_SUBMISSION else (35 * 60)` in Cell 4.
2. Graceful In-Child `SIGALRM` `TimeoutError` (`python_tool_sandbox.py`):
   - Fires `ITIMER_REAL` at `timeout_seconds - 0.5` inside `_SANDBOX_BOOTSTRAP` before parent `SIGKILL`,
     preserving 100% of `stdout.getvalue()`, reporting the exact hanging line number in `Traceback`,
     and retaining both previous and newly defined helper functions via `_collect_retention_payload()`.
3. Conditional `RESETC` (`EXPOSE_RESET=on` only in games without `UNDO` / `ACTION7`):
   - Hides `RESET` and strips `RESET_INFO_ADDENDUM` in games with `UNDO` (preserving 100% v24/v32
     behavior and undo search), while keeping `RESET` in games without `UNDO` so the agent can reset
     unrecoverable states in 1 action instead of burning its entire action budget.
4. Proven v32 Zero-Prompt Foundations:
   - `shiiin9` continuous D' priority scheduler with unscaled `B = (10, 14, 16)` for `remaining >= 1`
     and `b0 = 8.0 * C` for `remaining == 0, level > 1`.
   - 3-layer transient animation guard (`transient_pixels > 0` reported as animated effect).
   - `"time"` in `SAFE_MODULES` + zero-global `_NodeDict` & `_BoundaryList` segmentation aliases.
   - Pre-checked single-thread cold-boot warmup shield (`patch_startup_shield.py`).
5. Unlocked 10-Game, 25-Minute Kaggle GPU Benchmark (`nb_v11["cells"][16]`):
   - Runs the 10-game multi-stream benchmark (`ar25`, `ft09`, `lp85`, `r11l`, `re86`, `sb26`, `sc25`,
     `tr87`, `tu93`, `vc33`) across all 10 SGLang streams for 25 minutes on the 48GB `Qwen3.8-Flash-Next`
     model during `not TRUE_SUBMISSION`, and all 110 games for 532 minutes in `TRUE_SUBMISSION`.
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
OUTPUT_V35_PATH = REPO_ROOT / "notebooks" / "submission_v36.ipynb"
STAGE_DIR = Path("/tmp/v36_kernel")
OUTPUT_SUB_PATH = STAGE_DIR / "submission.ipynb"
METADATA_PATH = STAGE_DIR / "kernel-metadata.json"

CELL0_MARKDOWN = r"""# ARC Prize 2026 — Version 36 Sovereign Apex Engine

## Ground-Truth Synthesis: `v35 v4` (`47.69` GPU Base) + Cross-Turn Per-Game `/tmp` Scratch Persistence + 45s Bounded-Search Headroom + Clean `SystemExit` + `FrameView` Iterability

Version 36 builds directly on our verified `v35 v4` (`47.69` 10-game GPU mean, `0` crashes, `vc33` Level 4 breakthrough `35.71`, `55.48` 4-run union) and resolves the 5 remaining bottlenecks mined from the 40-game `v35` transcript corpus:

1. **Cross-Turn Per-Game `/tmp` Scratch Persistence (`ar25` Level 6 Fix)**:
   - Scopes `open('/tmp/...')` inside `_SANDBOX_BOOTSTRAP` to a deterministic per-game scratch directory (`/tmp/arc3_scratch_<seed_board_hash>`) keyed by `history[0].frame.ascii`, allowing multi-turn solvers (`ar25`, `lp85`) to save parsed board graphs in Turn $T$ and load them in Turn $T+1$ with zero cross-game leakage.
2. **45s Sandbox CPU Headroom for Bounded A\*/BFS Searches (`r11l`, `re86` Fix)**:
   - Grants `max(timeout_seconds, 45.0)` seconds (`44.5s` `SIGALRM`, `RLIMIT_CPU=46`) to the sandbox child while keeping the prompt's `30s` guidance so bounded 30–44s A\*/BFS searches complete on the first turn under 10-stream CPU load.
3. **Clean `SystemExit` / `exit()` Completion (`tr87` Fix)**:
   - Binds `SystemExit` and `exit()` to a sandbox-local `_SandboxExit(BaseException)` caught at the end of `exec(...)`, turning early exits from nested plan loops into normal `"type": "final"` completions.
4. **`FrameView` Iterability & Indexing (`lp85` Fix)** & **`last_action_timeline` Alias (`sb26` Fix)**:
   - Adds `__len__`, `__iter__`, and `__getitem__` to `FrameView` and aliases `last_action_timeline` / `last_action_frames` without touching `_available_names`.
"""

PATCH_STARTUP_SHIELD = r'''"""Install pre-checked periodic warmup RESET keepalive during slow SGLang cold boots (Topics 750173 & 750885)."""
import ast
from pathlib import Path
import sys

BEFORE = """        if self.solver._claim_warmup_slot():
            # Straight to the engine - _execute_auto_reset takes the same path
            # and needs no analyzer, so this works with the endpoint still
            # loading. One action off this game's budget buys the whole
            # submission a deadline it would otherwise miss.
            # warning, not info: _configure_logging only runs under run.py, so
            # the notebook has no handler and falls back to logging's lastResort,
            # which emits WARNING and above. An info line would vanish exactly
            # where this message matters most.
            log.warning(
                "%s: warmup RESET issued before the analyzer is reachable",
                run.game_id,
            )
            self._execute_auto_reset()
"""

AFTER = """        _is_warmup_slot = bool(self.solver._claim_warmup_slot())
        if _is_warmup_slot:
            _warmup_base_url = os.environ.get("LOCAL_ANALYZER_BASE_URL", "http://127.0.0.1:8001/v1")
            _warmup_health_url = (
                _warmup_base_url[:-3] + "/health"
                if _warmup_base_url.endswith("/v1")
                else _warmup_base_url.rstrip("/") + "/health"
            )
            import urllib.request as _warmup_urllib
            if not getattr(self.solver, "_analyzer_ready", False):
                try:
                    with _warmup_urllib.urlopen(_warmup_health_url, timeout=3) as _wresp:
                        if getattr(_wresp, "status", 200) == 200:
                            setattr(self.solver, "_analyzer_ready", True)
                except Exception:
                    pass
            if not getattr(self.solver, "_analyzer_ready", False):
                log.warning(
                    "%s: warmup RESET issued before the analyzer is reachable",
                    run.game_id,
                )
                self._execute_auto_reset()
                if int(os.environ.get("ARC3_WARMUP_ACTION_GAMES", "0") or 0) > 0:
                    _warmup_last_ts = time.monotonic()
                    _warmup_grace_s = float(os.environ.get("ARC3_HTTP_RETRY_INITIAL_SECONDS", "2400") or 2400.0)
                    _warmup_deadline = _warmup_last_ts + _warmup_grace_s
                    _warmup_interval_s = float(os.environ.get("ARC3_WARMUP_KEEPALIVE_SECONDS", "480") or 480.0)
                    while time.monotonic() < _warmup_deadline and not self.stop_event.is_set():
                        if getattr(self.solver, "_analyzer_ready", False):
                            break
                        try:
                            with _warmup_urllib.urlopen(_warmup_health_url, timeout=3) as _wresp:
                                if getattr(_wresp, "status", 200) == 200:
                                    setattr(self.solver, "_analyzer_ready", True)
                                    break
                        except Exception:
                            pass
                        if (time.monotonic() - _warmup_last_ts) >= _warmup_interval_s:
                            log.warning(
                                "%s: periodic warmup RESET issued while analyzer is still loading",
                                run.game_id,
                            )
                            self._execute_auto_reset()
                            _warmup_last_ts = time.monotonic()
                        time.sleep(min(3.0, max(0.05, _warmup_interval_s / 20.0)))
        elif not getattr(self.solver, "_analyzer_ready", False) and int(os.environ.get("ARC3_WARMUP_ACTION_GAMES", "0") or 0) > 0:
            _wait_deadline = time.monotonic() + float(os.environ.get("ARC3_HTTP_RETRY_INITIAL_SECONDS", "2400") or 2400.0)
            while not getattr(self.solver, "_analyzer_ready", False) and time.monotonic() < _wait_deadline and not self.stop_event.is_set():
                time.sleep(1.0)
        if (getattr(self.solver, "max_runtime_s_per_game", 0) or 0) <= 3600:
            self.started_at = time.monotonic()
"""


def apply(root):
    path = Path(root) / "inference/framework/solver.py"
    text = path.read_text(encoding="utf-8")
    if "periodic warmup RESET issued while analyzer is still loading" in text:
        return
    if text.count(BEFORE) != 1:
        raise RuntimeError(f"Expected warmup block once in solver.py, got {text.count(BEFORE)}")
    changed = text.replace(BEFORE, AFTER, 1)
    ast.parse(changed)
    path.write_text(changed, encoding="utf-8")


if __name__ == "__main__":
    apply(sys.argv[1])
'''

PATCH_TRANSIENT_AND_TIMEOUT = r'''"""Install 3-layer transient-animation guard, sandbox allowlist & view polish, IPC-paused SIGALRM TimeoutError, StaleStateActionError disarm, and 2D border gameplay detector."""
import ast
from pathlib import Path
import sys

ANIM_BEFORE = """    empty = {
        "error": (
            "No animation is available: the last executed action returned a single frame."
        )
    }
"""

ANIM_AFTER = """    empty = {
        "action": record.get("action_display") if record else None,
        "frames": 0,
        "steps": [],
        "error": (
            "No animation is available: the last executed action returned a single frame."
        )
    }
"""

SANDBOX_BEFORE = """            gameplay_changed = action_result.get("gameplay_changed")
            if gameplay_changed is True:
                # a real change means the snippet's beliefs are current again
                del _stale_state[:]
            if gameplay_changed is False and action_result.get("executed"):
                _executed = action_result.get("executed_actions")
                _name = (
                    _executed[-1]
                    if isinstance(_executed, list) and _executed
                    else action_result.get("action_display")
                    or action_result.get("action_name")
                    or "?"
                )
                # Arm rather than raise. The action executed and changed
                # nothing, which is often exactly what the snippet went looking
                # for - it printed the board before, and wants to print it
                # after. Killing it here throws away the second half of the
                # measurement AND the batch stop's own trace, which is computed
                # and then discarded. So let the snippet finish observing, and
                # stop it only if it tries to ACT on the stale belief.
                if not _stale_state:
                    _stale_state.append(str(_name))
            return action_result"""

SANDBOX_AFTER = """            gameplay_changed = action_result.get("gameplay_changed")
            animated = bool((action_result.get("animation") or {}).get("transient_pixels", 0))
            if gameplay_changed is True or animated:
                # Visible or intermediate motion keeps the snippet's beliefs live.
                del _stale_state[:]
                del _last_noop_actions[:]
                _INSPECTED_SINCE_NOOP[0] = False
            if gameplay_changed is False and not animated and action_result.get("executed"):
                _INSPECTED_SINCE_NOOP[0] = False
                _last_noop_actions[:] = normalized_actions
                _executed = action_result.get("executed_actions")
                _name = (
                    _executed[-1]
                    if isinstance(_executed, list) and _executed
                    else action_result.get("action_display")
                    or action_result.get("action_name")
                    or "?"
                )
                if not _stale_state:
                    _stale_state.append(str(_name))
            return _ActionResultDict(action_result)"""

SAFE_MODULES_BEFORE = """        "statistics",
        "string",
    }"""

SAFE_MODULES_AFTER = """        "inspect",
        "linecache",
        "pickle",
        "statistics",
        "string",
        "time",
        "traceback",
        "typing",
    }"""

SAFE_BUILTINS_BEFORE = """        "TypeError",
        "type",
        "ValueError",
        "RuntimeError",
        "zip",
    }"""

SAFE_BUILTINS_AFTER = """        "TypeError",
        "type",
        "ValueError",
        "RuntimeError",
        "delattr",
        "globals",
        "locals",
        "setattr",
        "vars",
        "zip",
    }"""

HISTORY_ENTRY_BEFORE = """    class FrameView:
        def __init__(self, *, ascii, step, level, shape, grid):
            self.ascii = ascii
            self.step = step
            self.level = level
            self.shape = tuple(shape)
            self._grid = grid
            self._segmentation = None

        @property
        def segmentation(self):
            if self._segmentation is None:
                self._segmentation = segment_layer(self._grid, COLOR_CHARS)
            return self._segmentation

        def __str__(self):
            rows, cols = self.shape
            return f"AsciiFrameView(level={self.level}, step={self.step}, shape={rows}x{cols})"

        __repr__ = __str__


    class HistoryEntryView:
        def __init__(self, *, action, frame, result=None):
            self.action = action
            self.frame = frame
            self.result = dict(result) if isinstance(result, dict) else {}

        def __str__(self):
            return f"AsciiHistoryEntryView(action={self.action!r}, frame={self.frame})"

        __repr__ = __str__


    class TransitionView:
        def __init__(self, *, action, before_frame, after_frame, result):
            self.action = action
            self.before_frame = before_frame
            self.after_frame = after_frame
            self.frame = after_frame
            self.result = dict(result) if isinstance(result, dict) else {}

        def __str__(self):"""

HISTORY_ENTRY_AFTER = """    _INSPECTED_SINCE_NOOP = [False]

    class _AnimSummaryDict(dict):
        def __missing__(self, key):
            if key == "steps":
                return []
            if key == "transient_bbox":
                return None
            if key in ("frames", "transient_pixels"):
                return 0
            raise KeyError(key)

        def get(self, key, default=None):
            if super().__contains__(key):
                return super().__getitem__(key)
            if default is None:
                if key == "steps":
                    return []
                if key == "transient_bbox":
                    return None
                if key in ("frames", "transient_pixels"):
                    return 0
            return default

        def __reduce__(self):
            return (dict, (dict(self),))

    class _ActionResultDict(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if super().__contains__("animation") and isinstance(super().__getitem__("animation"), dict):
                super().__setitem__("animation", _AnimSummaryDict(super().__getitem__("animation")))

        def __missing__(self, key):
            if key == "animation":
                return _AnimSummaryDict({"frames": 0, "steps": [], "transient_pixels": 0, "transient_bbox": None})
            raise KeyError(key)

        def __getitem__(self, key):
            _INSPECTED_SINCE_NOOP[0] = True
            return super().__getitem__(key)

        def get(self, key, default=None):
            _INSPECTED_SINCE_NOOP[0] = True
            if key == "animation" and not super().__contains__(key) and default is None:
                return _AnimSummaryDict({"frames": 0, "steps": [], "transient_pixels": 0, "transient_bbox": None})
            return super().get(key, default)

        def __contains__(self, key):
            _INSPECTED_SINCE_NOOP[0] = True
            return super().__contains__(key)

        def __reduce__(self):
            return (dict, (dict(self),))

    class _AsciiStr(str):
        def split(self, *args, **kwargs):
            _INSPECTED_SINCE_NOOP[0] = True
            return super().split(*args, **kwargs)

        def splitlines(self, *args, **kwargs):
            _INSPECTED_SINCE_NOOP[0] = True
            return super().splitlines(*args, **kwargs)

        def __getitem__(self, idx):
            _INSPECTED_SINCE_NOOP[0] = True
            if isinstance(idx, int):
                lines = super().splitlines()
                if len(lines) > 1 and -len(lines) <= idx < len(lines):
                    return lines[idx]
            if isinstance(idx, tuple) and len(idx) == 2:
                r, c = idx
                return super().splitlines()[r][c]
            return super().__getitem__(idx)

        def __reduce__(self):
            return (str, (str(self),))


    class FrameView:
        def __init__(self, *, ascii, step, level, shape, grid):
            self._ascii = _AsciiStr(ascii)
            self.step = step
            self.level = level
            self.shape = tuple(shape)
            self._grid = grid
            self._segmentation = None

        @property
        def ascii(self):
            _INSPECTED_SINCE_NOOP[0] = True
            return self._ascii

        @property
        def segmentation(self):
            _INSPECTED_SINCE_NOOP[0] = True
            if self._segmentation is None:
                self._segmentation = segment_layer(self._grid, COLOR_CHARS)
            return self._segmentation

        def __len__(self):
            _INSPECTED_SINCE_NOOP[0] = True
            return self.shape[0]

        def __iter__(self):
            _INSPECTED_SINCE_NOOP[0] = True
            return iter(self._ascii.splitlines())

        def __getitem__(self, idx):
            _INSPECTED_SINCE_NOOP[0] = True
            if isinstance(idx, tuple) and len(idx) == 2:
                r, c = idx
                return self._ascii.splitlines()[r][c]
            return self._ascii.splitlines()[idx]

        def __str__(self):
            rows, cols = self.shape
            return f"AsciiFrameView(level={self.level}, step={self.step}, shape={rows}x{cols})"

        __repr__ = __str__


    class HistoryEntryView:
        def __init__(self, *, action, frame, result=None):
            self.action = action
            self.frame = frame
            self.result = _ActionResultDict(result) if isinstance(result, dict) else _ActionResultDict()

        def __getattr__(self, name):
            if name in ("step", "level", "ascii", "segmentation", "shape") and self.frame is not None:
                return getattr(self.frame, name)
            raise AttributeError(f"'{type(self).__name__}' object has no attribute {name!r}")

        def __str__(self):
            return f"AsciiHistoryEntryView(action={self.action!r}, frame={self.frame})"

        __repr__ = __str__


    class TransitionView:
        def __init__(self, *, action, before_frame, after_frame, result):
            self.action = action
            self.before_frame = before_frame
            self.after_frame = after_frame
            self.frame = after_frame
            self.result = _ActionResultDict(result) if isinstance(result, dict) else _ActionResultDict()

        def __getattr__(self, name):
            if name == "before":
                return self.before_frame
            if name == "after":
                return self.after_frame
            if name in ("step", "level", "ascii", "segmentation", "shape") and self.after_frame is not None:
                return getattr(self.after_frame, name)
            raise AttributeError(f"'{type(self).__name__}' object has no attribute {name!r}")

        def __str__(self):"""

JSON_SAFE_BEFORE = """    def _json_safe(value):
        if value is None or isinstance(value, (str, int, float, bool)):"""

JSON_SAFE_AFTER = """    def _json_safe(value):
        if hasattr(value, "_get") and callable(getattr(value, "_get", None)):
            value = value._get()
        if value is None or isinstance(value, (str, int, float, bool)):"""

SANDBOX_OPEN_BEFORE = """        runtime_globals["__builtins__"]["__import__"] = _safe_import"""

SANDBOX_OPEN_AFTER = """        _st0 = initial.get("state") or {}
        _h0 = _st0.get("history") or []
        _seed_ascii = str(
            _st0.get("seed_id")
            or ((_h0[0].get("frame") if _h0 and isinstance(_h0[0], dict) else None)
                or _st0.get("current_frame") or {}).get("ascii", "")
        )
        _seed_id = 0
        for _ch in _seed_ascii:
            _seed_id = ((_seed_id * 131) + ord(_ch)) & 0xFFFFFFFFFFFFFFFF
        _game_scratch_dir = os.path.join("/tmp", f"arc3_scratch_{_seed_id:016x}")
        try:
            os.makedirs(_game_scratch_dir, exist_ok=True)
        except Exception:
            _game_scratch_dir = sandbox_cwd or "/tmp"

        def _sandbox_open(file, mode="r", *args, **kwargs):
            fname = os.path.basename(str(file)) or "sandbox_tmp"
            safe_path = os.path.join(_game_scratch_dir, fname)
            return builtins.open(safe_path, mode, *args, **kwargs)

        class _SandboxExit(BaseException):
            pass

        def _sandbox_exit(code=None):
            raise _SandboxExit(code)

        import types as _sb_types_init
        _builtins_mod = _sb_types_init.ModuleType("builtins")
        _builtins_mod.__dict__.update(runtime_globals["__builtins__"])

        def _safe_import_v36(name, globals=None, locals=None, fromlist=(), level=0):
            root = str(name or "").split(".", 1)[0]
            if root == "builtins":
                return _builtins_mod
            if root == "sys":
                import types as _sb_types
                _sm = _sb_types.ModuleType("sys")
                _sm.exit = _sandbox_exit
                _sm.stdout = stdout
                _sm.stderr = stdout
                _sm.maxsize = sys.maxsize
                _sm.version = sys.version
                _sm.version_info = sys.version_info
                _sm.getrecursionlimit = sys.getrecursionlimit
                _sm.setrecursionlimit = sys.setrecursionlimit
                _sm.int_info = sys.int_info
                _sm.float_info = sys.float_info
                return _sm
            return _safe_import(name, globals, locals, fromlist, level)

        runtime_globals["__builtins__"]["__import__"] = _safe_import_v36
        runtime_globals["__builtins__"]["open"] = _sandbox_open
        runtime_globals["__builtins__"]["SystemExit"] = _SandboxExit
        runtime_globals["__builtins__"]["exit"] = _sandbox_exit
        _builtins_mod.__dict__.update(runtime_globals["__builtins__"])"""

STALE_STATE_RAISE_BEFORE = """        def action(actions):
            if _terminal_state:
                raise TerminalStateActionError(_TERMINAL_DETAIL[_terminal_state[0]])
            normalized_actions = _normalize_actions(actions)
            _send({
                "type": "action",
                "actions": normalized_actions,
                # the solver decides precedence: a known death or a known no-op
                # is a sharper message than "your state may be stale", and it
                # carries a concrete override
                "stale_after": _stale_state[0] if _stale_state else None,
            })
            reply = _recv()
            if reply.get("type") == "action_error":
                raise RuntimeError(str(reply.get("error", "action failed")))
            if reply.get("type") != "action_result":
                raise RuntimeError("Invalid action response from sandbox host.")
            action_result = reply.get("action_result") or {}
            action_results.append(action_result)
            # Cached animations describe the previous action. Invalidate the
            # existing views so aliases also refresh, even if a guard raises.
            for view in _animation_views:
                view.invalidate()
            _refresh_state(reply.get("state") or {})
            if action_result.get("stop_reason") == "reset_not_first":
                raise RuntimeError(str(action_result.get("stop_detail")))
            if action_result.get("stop_reason") == "stale_state":
                raise StaleStateActionError(
                    str(action_result.get("stop_detail") or "stale state")
                )"""

STALE_STATE_RAISE_AFTER = """        _last_noop_actions: list = []

        def action(actions):
            if _terminal_state:
                raise TerminalStateActionError(_TERMINAL_DETAIL[_terminal_state[0]])
            normalized_actions = _normalize_actions(actions)
            _both_mouse = (
                bool(_last_noop_actions)
                and bool(normalized_actions)
                and all(isinstance(x, dict) and x.get("action") == "MOUSE" for x in _last_noop_actions)
                and all(isinstance(x, dict) and x.get("action") == "MOUSE" for x in normalized_actions)
            )
            if _stale_state and normalized_actions != _last_noop_actions and (_INSPECTED_SINCE_NOOP[0] or _both_mouse):
                del _stale_state[:]
            _send({
                "type": "action",
                "actions": normalized_actions,
                # the solver decides precedence: a known death or a known no-op
                # is a sharper message than "your state may be stale", and it
                # carries a concrete override
                "stale_after": _stale_state[0] if _stale_state else None,
            })
            reply = _recv()
            if reply.get("type") == "action_error":
                raise RuntimeError(str(reply.get("error", "action failed")))
            if reply.get("type") != "action_result":
                raise RuntimeError("Invalid action response from sandbox host.")
            action_result = reply.get("action_result") or {}
            action_results.append(action_result)
            # Cached animations describe the previous action. Invalidate the
            # existing views so aliases also refresh, even if a guard raises.
            for view in _animation_views:
                view.invalidate()
            _refresh_state(reply.get("state") or {})
            if action_result.get("stop_reason") == "reset_not_first":
                raise RuntimeError(str(action_result.get("stop_detail")))
            if action_result.get("stop_reason") == "stale_state":
                del _stale_state[:]
                raise StaleStateActionError(
                    str(action_result.get("stop_detail") or "stale state")
                )"""

LAZY_ANIM_BEFORE = """            class _LazyAnimation:
                def __init__(self, kind, empty):
                    self._kind = kind
                    self._empty = empty
                    self._value = None

                def _get(self):
                    if self._value is None:
                        self._value = _animation_fetch(self._kind) or self._empty
                    return self._value

                def invalidate(self):
                    self._value = None

                def __len__(self):
                    return len(self._get())

                def __iter__(self):
                    return iter(self._get())

                def __getitem__(self, key):
                    return self._get()[key]

                def __bool__(self):
                    return bool(self._get())

                def __contains__(self, key):
                    return key in self._get()

                def keys(self):
                    return self._get().keys()

                def get(self, key, default=None):
                    return self._get().get(key, default)

                def __repr__(self):
                    return repr(self._get())

            runtime_globals["last_animation_frames"] = _LazyAnimation("frames", [])
            runtime_globals["last_animation_timeline"] = _LazyAnimation("timeline", {})"""

LAZY_ANIM_AFTER = """            class _LazyFrames(list):
                def __init__(self, empty):
                    super().__init__()
                    self._empty = empty
                    self._loaded = False

                def _get(self):
                    if not self._loaded:
                        val = _animation_fetch("frames") or self._empty
                        super().clear()
                        if isinstance(val, (list, tuple)):
                            super().extend(val)
                        self._loaded = True
                    return self

                def invalidate(self):
                    self._loaded = False
                    super().clear()

                def __len__(self):
                    self._get()
                    return super().__len__()

                def __iter__(self):
                    self._get()
                    return super().__iter__()

                def __getitem__(self, key):
                    self._get()
                    return super().__getitem__(key)

                def __bool__(self):
                    self._get()
                    return bool(super().__len__())

                def __contains__(self, key):
                    self._get()
                    return super().__contains__(key)

                def __repr__(self):
                    self._get()
                    return super().__repr__()

            class _LazyTimeline(dict):
                def __init__(self, empty):
                    super().__init__()
                    self._empty = empty
                    self._loaded = False

                def _get(self):
                    if not self._loaded:
                        val = _animation_fetch("timeline") or self._empty
                        super().clear()
                        if isinstance(val, dict):
                            super().update(val)
                        self._loaded = True
                    return self

                def invalidate(self):
                    self._loaded = False
                    super().clear()

                def __len__(self):
                    self._get()
                    return super().__len__()

                def __iter__(self):
                    self._get()
                    return super().__iter__()

                def __getitem__(self, key):
                    self._get()
                    if isinstance(key, slice):
                        return list(super().get("steps", []))[key]
                    return super().__getitem__(key)

                def __bool__(self):
                    self._get()
                    return bool(super().__len__())

                def __contains__(self, key):
                    self._get()
                    return super().__contains__(key)

                def keys(self):
                    self._get()
                    return super().keys()

                def items(self):
                    self._get()
                    return super().items()

                def values(self):
                    self._get()
                    return super().values()

                def get(self, key, default=None):
                    self._get()
                    return super().get(key, default)

                def __getattr__(self, name):
                    if name.startswith("_"):
                        raise AttributeError(name)
                    self._get()
                    if super().__contains__(name):
                        return super().__getitem__(name)
                    raise AttributeError(f"'{type(self).__name__}' object has no attribute {name!r}")

                def __repr__(self):
                    self._get()
                    return super().__repr__()

            runtime_globals["last_animation_frames"] = _LazyFrames([])
            runtime_globals["last_animation_timeline"] = _LazyTimeline({})
            runtime_globals["last_action_frames"] = runtime_globals["last_animation_frames"]
            runtime_globals["last_action_timeline"] = runtime_globals["last_animation_timeline"]"""

SET_LIMITS_BEFORE = """    def _set_limits(timeout_seconds):
        if resource is None:
            return
        cpu_limit = max(1, int(timeout_seconds)) + 1
        for limit, value in (
            (getattr(resource, "RLIMIT_CPU", None), cpu_limit),
            (getattr(resource, "RLIMIT_FSIZE", None), 1_000_000),
            (getattr(resource, "RLIMIT_NOFILE", None), 32),
        ):"""

SET_LIMITS_AFTER = """    def _set_limits(timeout_seconds):
        if resource is None:
            return
        _eff_timeout = 45 if int(timeout_seconds) == 30 else max(1, int(timeout_seconds))
        cpu_limit = _eff_timeout + 5
        for limit, value in (
            (getattr(resource, "RLIMIT_CPU", None), cpu_limit),
            (getattr(resource, "RLIMIT_FSIZE", None), 64_000_000),
            (getattr(resource, "RLIMIT_NOFILE", None), 64),
        ):"""

REFRESH_LAST_RES_BEFORE = """            runtime_globals["death_ledger"] = state_payload.get("death_ledger") or {"attempts": []}
            runtime_globals["last_action_call_result"] = action_result"""

REFRESH_LAST_RES_AFTER = """            runtime_globals["death_ledger"] = state_payload.get("death_ledger") or {"attempts": []}
            runtime_globals["last_action_call_result"] = _ActionResultDict(action_result)"""

IPC_CHILD_BEFORE = """    def _send(payload):
        HOST_STDOUT.write(json.dumps(payload, ensure_ascii=False) + "\\n")
        HOST_STDOUT.flush()


    def _recv():
        line = sys.stdin.readline()
        if not line:
            raise EOFError("sandbox input closed")
        return json.loads(line)"""

IPC_CHILD_AFTER = """    import time as _sb_time

    _orig_json_dumps = json.dumps
    _orig_json_dump = json.dump

    def _preload_lazy(obj):
        if hasattr(obj, "_get") and callable(getattr(obj, "_get", None)):
            obj._get()
        elif isinstance(obj, dict):
            for _v in obj.values():
                if hasattr(_v, "_get") and callable(getattr(_v, "_get", None)):
                    _v._get()
        elif isinstance(obj, (list, tuple)):
            for _v in obj:
                if hasattr(_v, "_get") and callable(getattr(_v, "_get", None)):
                    _v._get()
        return obj

    def _safe_json_dumps(obj, *args, **kwargs):
        return _orig_json_dumps(_preload_lazy(obj), *args, **kwargs)

    def _safe_json_dump(obj, fp, *args, **kwargs):
        return _orig_json_dump(_preload_lazy(obj), fp, *args, **kwargs)

    json.dumps = _safe_json_dumps
    json.dump = _safe_json_dump

    _ITIMER_STATE = {
        "sig": None,
        "active": False,
        "paused_at": None,
        "deadline": 0.0,
        "ipc_budget": 239.5,
    }


    def _send(payload):
        if (
            _ITIMER_STATE["active"]
            and _ITIMER_STATE["sig"] is not None
            and isinstance(payload, dict)
            and payload.get("type") in ("action", "animation")
        ):
            _ITIMER_STATE["paused_at"] = _sb_time.monotonic()
            try:
                _ITIMER_STATE["sig"].setitimer(_ITIMER_STATE["sig"].ITIMER_REAL, 0.0)
            except Exception:
                pass
        HOST_STDOUT.write(_orig_json_dumps(payload, ensure_ascii=False) + "\\n")
        HOST_STDOUT.flush()


    def _recv():
        try:
            line = sys.stdin.readline()
            if not line:
                raise EOFError("sandbox input closed")
            return json.loads(line)
        finally:
            if (
                _ITIMER_STATE["active"]
                and _ITIMER_STATE["sig"] is not None
                and _ITIMER_STATE["paused_at"] is not None
            ):
                _elapsed = max(0.0, _sb_time.monotonic() - _ITIMER_STATE["paused_at"])
                _ITIMER_STATE["paused_at"] = None
                _credit = min(_ITIMER_STATE["ipc_budget"], _elapsed)
                _ITIMER_STATE["ipc_budget"] -= _credit
                _ITIMER_STATE["deadline"] += _credit
                _rem = max(0.05, _ITIMER_STATE["deadline"] - _sb_time.monotonic())
                try:
                    _ITIMER_STATE["sig"].setitimer(_ITIMER_STATE["sig"].ITIMER_REAL, _rem)
                except Exception:
                    pass"""

EXEC_BLOCK_BEFORE = """        _available_names = set(runtime_globals) | set(runtime_globals["__builtins__"])
        if _persistence_enabled:
            _before_restore = dict(runtime_globals)
            try:
                _kept_pool, _retention_rejected = restore_functions(
                    initial.get("kept_functions") or [], runtime_globals, _available_names,
                    **_retention_options)
            except Exception:
                # Optional cache maintenance must never prevent this snippet running.
                runtime_globals.clear()
                runtime_globals.update(_before_restore)
                _kept_pool = {}
                _retention_rejected = [{"name": "all", "reason": "restoration failed; redefine needed functions"}]

        def _collect_retention_payload():
            if not _persistence_enabled:
                return {}
            try:
                kept, rejected = collect_functions(
                    _code_text, _kept_pool, runtime_globals, _available_names,
                    **_retention_options)
                return {
                    "keepable_functions": [{"name": n, "source": src} for n, src in kept.items()],
                    "retention_rejected": _retention_rejected + rejected,
                }
            except Exception:
                # Bookkeeping failures may drop helpers, but must never replace
                # the snippet's original error/output or replay its actions.
                return {"keepable_functions": [], "retention_rejected": [
                    {"name": "all", "reason": "retention failed; redefine needed functions"}]}

        try:
            _code_text = str(initial.get("code", ""))
            _tree = ast.parse(_code_text, "<python_tool>")
            _last_expr = None
            if _tree.body and isinstance(_tree.body[-1], ast.Expr):
                _last_expr = ast.Expression(_tree.body[-1].value)
                ast.copy_location(_last_expr, _tree.body[-1])
                _tree.body = _tree.body[:-1]
            with contextlib.redirect_stdout(stdout):
                if _tree.body:
                    exec(compile(_tree, "<python_tool>", "exec"), runtime_globals, runtime_globals)
                if _last_expr is not None:
                    _value = eval(compile(_last_expr, "<python_tool>", "eval"), runtime_globals, runtime_globals)
                    if _value is not None:
                        print(repr(_value))
                        if runtime_globals.get("result") is None:
                            runtime_globals["result"] = _value
            _send(
                {
                    "type": "final",
                    **_collect_retention_payload(),
                    "stdout": stdout.getvalue(),
                    "result": _json_safe(runtime_globals.get("result")),
                    "action_results": _json_safe(action_results),
                }
            )
        except (
            KnownNoOpActionError,
            RepeatedActionInStateError,
            KnownDeathActionError,
            TerminalStateActionError,
            Exception,
        ) as exc:"""

EXEC_BLOCK_AFTER = """        import inspect as _sb_inspect
        import linecache as _sb_linecache
        import marshal as _sb_marshal
        import pickle as _sb_pickle
        import types as _sb_types

        _base_available_names = set(runtime_globals) | set(runtime_globals["__builtins__"])
        _code_text = str(initial.get("code", ""))
        _sb_linecache.cache["<python_tool>"] = (
            len(_code_text),
            None,
            [line + "\\n" for line in _code_text.splitlines()],
            "<python_tool>",
        )
        _code_func_defs = set()
        _code_func_sources = {}
        _code_imports = set()
        for _src_item in [*(initial.get("kept_functions") or []), _code_text]:
            try:
                _s_str = str(_src_item)
                _p_tree = ast.parse(_s_str)
                for _nd in _p_tree.body:
                    if isinstance(_nd, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        _code_func_defs.add(_nd.name)
                        try:
                            _code_func_sources[_nd.name] = ast.get_source_segment(_s_str, _nd) or ast.unparse(_nd)
                        except Exception:
                            pass
                    elif isinstance(_nd, ast.Import):
                        for _al in _nd.names:
                            _code_imports.add(_al.asname or _al.name.split(".")[0])
                    elif isinstance(_nd, ast.ImportFrom):
                        for _al in _nd.names:
                            _code_imports.add(_al.asname or _al.name)
            except Exception:
                pass

        def _has_frame_view(obj, depth=0):
            if depth > 4:
                return False
            if type(obj).__name__ in ("FrameView", "HistoryEntryView", "TransitionView"):
                return True
            if isinstance(obj, dict):
                return any(_has_frame_view(k, depth + 1) or _has_frame_view(v, depth + 1) for k, v in list(obj.items())[:60])
            if isinstance(obj, (list, tuple, set, frozenset)):
                return any(_has_frame_view(x, depth + 1) for x in list(obj)[:60])
            return False

        def _dump_val(v, memo=None, depth=0):
            if depth > 8:
                raise TypeError("max depth")
            if memo is None:
                memo = {}
            for _sym_k in ("last_animation_frames", "last_animation_timeline", "current_frame", "previous_frame", "last_transition"):
                _sym_v = runtime_globals.get(_sym_k)
                if _sym_v is not None and v is _sym_v:
                    return ("SYM_REF", _sym_k)
            _cf_obj = runtime_globals.get("current_frame")
            if _cf_obj is not None:
                _cf_asc = str(getattr(_cf_obj, "_ascii", getattr(_cf_obj, "ascii", "")))
                if isinstance(v, str) and len(v) > 16 and v == _cf_asc:
                    return ("CUR_ASCII",)
                if isinstance(v, list) and len(v) > 2 and v == _cf_asc.splitlines():
                    return ("CUR_ASCII_LINES",)
            if id(v) in memo:
                return ("REF", memo[id(v)])
            if isinstance(v, _sb_types.FunctionType):
                fid = len(memo)
                memo[id(v)] = fid
                cells = [_dump_val(c.cell_contents, memo, depth + 1) for c in (v.__closure__ or ())]
                _fn_src = getattr(v, "_arc3_source", None) or _code_func_sources.get(v.__name__) or ""
                return ("FN", fid, _sb_marshal.dumps(v.__code__), v.__name__, _dump_val(v.__defaults__, memo, depth + 1), cells, _fn_src)
            try:
                return ("PKL", _sb_pickle.dumps(v))
            except Exception:
                pass
            if isinstance(v, list):
                return ("LST", [_dump_val(x, memo, depth + 1) for x in v])
            if isinstance(v, tuple):
                return ("TUP", [_dump_val(x, memo, depth + 1) for x in v])
            if isinstance(v, dict):
                return ("DCT", [(_dump_val(k, memo, depth + 1), _dump_val(val, memo, depth + 1)) for k, val in v.items()])
            if isinstance(v, set):
                return ("SET", [_dump_val(x, memo, depth + 1) for x in v])
            raise TypeError("unserializable")

        def _load_val(tag_tuple, env_globals, memo=None):
            if memo is None:
                memo = {}
            kind = tag_tuple[0]
            if kind == "SYM_REF":
                return env_globals.get(tag_tuple[1])
            if kind == "CUR_ASCII":
                return env_globals["current_frame"].ascii
            if kind == "CUR_ASCII_LINES":
                return env_globals["current_frame"].ascii.splitlines()
            if kind == "PKL":
                return _sb_pickle.loads(tag_tuple[1])
            if kind == "REF":
                return memo[tag_tuple[1]]
            if kind == "LST":
                return [_load_val(x, env_globals, memo) for x in tag_tuple[1]]
            if kind == "TUP":
                return tuple(_load_val(x, env_globals, memo) for x in tag_tuple[1])
            if kind == "DCT":
                return {_load_val(k, env_globals, memo): _load_val(val, env_globals, memo) for k, val in tag_tuple[1]}
            if kind == "SET":
                return {_load_val(x, env_globals, memo) for x in tag_tuple[1]}
            if kind == "FN":
                fid, code_bytes, name, defs_tag, cells_tags = tag_tuple[1:6]
                _fn_src = tag_tuple[6] if len(tag_tuple) > 6 else ""
                code = _sb_marshal.loads(code_bytes)
                cells = tuple(_sb_types.CellType(None) for _ in cells_tags)
                fn = _sb_types.FunctionType(code, env_globals, name, _load_val(defs_tag, env_globals, memo), cells)
                if _fn_src:
                    try:
                        fn._arc3_source = _fn_src
                    except Exception:
                        pass
                memo[fid] = fn
                for cell_obj, ctag in zip(cells, cells_tags):
                    cell_obj.cell_contents = _load_val(ctag, env_globals, memo)
                return fn
            return None

        _orig_builtins_keys = set(_builtins_mod.__dict__.keys())
        _available_names = _base_available_names
        _kept_pool = {}
        if _persistence_enabled:
            _before_restore = dict(runtime_globals)
            try:
                _kept_pool, _retention_rejected = restore_functions(
                    initial.get("kept_functions") or [], runtime_globals, _available_names,
                    **_retention_options)
                for _fn_k, _fn_s in _kept_pool.items():
                    _fn_obj = runtime_globals.get(_fn_k)
                    if isinstance(_fn_obj, _sb_types.FunctionType):
                        try:
                            _fn_obj._arc3_source = _fn_s
                        except Exception:
                            pass
            except Exception:
                # Optional cache maintenance must never prevent this snippet running.
                runtime_globals.clear()
                runtime_globals.update(_before_restore)
                _kept_pool = {}
                _retention_rejected = [{"name": "all", "reason": "restoration failed; redefine needed functions"}]

        _orig_getsource = _sb_inspect.getsource
        _orig_getsourcelines = _sb_inspect.getsourcelines

        def _arc3_getsource(obj):
            _src = (
                getattr(obj, "_arc3_source", None)
                or _kept_pool.get(getattr(obj, "__name__", ""))
                or _code_func_sources.get(getattr(obj, "__name__", ""))
            )
            if _src:
                return _src if _src.endswith("\\n") else _src + "\\n"
            return _orig_getsource(obj)

        def _arc3_getsourcelines(obj):
            _src = (
                getattr(obj, "_arc3_source", None)
                or _kept_pool.get(getattr(obj, "__name__", ""))
                or _code_func_sources.get(getattr(obj, "__name__", ""))
            )
            if _src:
                return ([line + "\\n" for line in _src.splitlines()], 1)
            return _orig_getsourcelines(obj)

        _sb_inspect.getsource = _arc3_getsource
        _sb_inspect.getsourcelines = _arc3_getsourcelines

        _vars_game_path = os.path.join(_game_scratch_dir, "_vars_game.pkl")
        _loaded_var_names = set()
        try:
            if os.path.isfile(_vars_game_path):
                with builtins.open(_vars_game_path, "rb") as _vf:
                    _raw_vars = _sb_pickle.load(_vf)
                if isinstance(_raw_vars, dict):
                    for _vk, _vtag in _raw_vars.items():
                        if isinstance(_vk, str) and _vk.startswith("__builtins__."):
                            _battr = _vk[13:]
                            if _battr and _battr not in _orig_builtins_keys:
                                try:
                                    _bval = _load_val(_vtag, runtime_globals)
                                    setattr(_builtins_mod, _battr, _bval)
                                except Exception:
                                    pass
                            continue
                        if (
                            isinstance(_vk, str)
                            and _vk not in runtime_globals
                            and _vk not in _base_available_names
                            and _vk not in _code_func_defs
                            and _vk not in _code_imports
                            and _vk not in SAFE_MODULES
                            and not _vk.startswith("_")
                        ):
                            try:
                                runtime_globals[_vk] = _load_val(_vtag, runtime_globals)
                                _loaded_var_names.add(_vk)
                            except Exception:
                                pass
        except Exception:
            pass

        _injected_stdlib = {}
        try:
            import collections as _sb_collections
            import itertools as _sb_itertools
            import heapq as _sb_heapq
            import math as _sb_math
            import re as _sb_re
            import copy as _sb_copy
            import functools as _sb_functools
            import bisect as _sb_bisect
            import random as _sb_random
            for _mname, _mobj in (
                ("inspect", _sb_inspect),
                ("linecache", _sb_linecache),
                ("pickle", _sb_pickle),
                ("traceback", traceback),
                ("collections", _sb_collections),
                ("itertools", _sb_itertools),
                ("heapq", _sb_heapq),
                ("math", _sb_math),
                ("json", json),
                ("re", _sb_re),
                ("copy", _sb_copy),
                ("functools", _sb_functools),
                ("bisect", _sb_bisect),
                ("random", _sb_random),
                ("time", _sb_time),
                ("deque", _sb_collections.deque),
                ("defaultdict", _sb_collections.defaultdict),
                ("Counter", _sb_collections.Counter),
            ):
                if _mname not in runtime_globals and _mname not in _code_func_defs:
                    runtime_globals[_mname] = _mobj
                    _injected_stdlib[_mname] = _mobj
        except Exception:
            pass

        def _clean_injected_stdlib():
            for _mname, _mobj in list(_injected_stdlib.items()):
                if _mname not in _code_imports and runtime_globals.get(_mname) is _mobj:
                    runtime_globals.pop(_mname, None)

        def _save_vars_post_collect(kept_dict):
            _saved_keys = set()
            try:
                _to_save = {}
                if os.path.isfile(_vars_game_path):
                    try:
                        with builtins.open(_vars_game_path, "rb") as _vf:
                            _prev_saved = _sb_pickle.load(_vf)
                        if isinstance(_prev_saved, dict):
                            _to_save.update(_prev_saved)
                    except Exception:
                        pass
                for _bk, _bv in list(_builtins_mod.__dict__.items()):
                    if _bk not in _orig_builtins_keys and not _bk.startswith("__"):
                        try:
                            _bdumped = _dump_val(_bv)
                            _bsz = len(_bdumped[1]) if _bdumped[0] == "PKL" else len(_sb_pickle.dumps(_bdumped))
                            if _bsz <= 16_000_000:
                                _to_save[f"__builtins__.{_bk}"] = _bdumped
                        except Exception:
                            pass
                for _k, _v in list(runtime_globals.items()):
                    if (
                        not isinstance(_k, str)
                        or _k.startswith("_")
                        or _k in _base_available_names
                        or _k in SAFE_MODULES
                        or _k in _injected_stdlib
                        or (len(_k) == 1 and _k in "ijkrcxyuvdmnpq" and isinstance(_v, (int, float, tuple)))
                        or isinstance(_v, _sb_types.ModuleType)
                    ):
                        continue
                    if isinstance(_v, _sb_types.FunctionType) and not getattr(_v, "__closure__", None) and (
                        _k in kept_dict
                    ):
                        _to_save.pop(_k, None)
                        continue
                    _is_sym_ref = any(
                        _v is runtime_globals.get(_sk)
                        for _sk in ("last_animation_frames", "last_animation_timeline", "current_frame", "previous_frame", "last_transition")
                    )
                    if not _is_sym_ref and _has_frame_view(_v):
                        continue
                    try:
                        _dumped = _dump_val(_v)
                        _dsz = len(_dumped[1]) if _dumped[0] == "PKL" else len(_sb_pickle.dumps(_dumped))
                        _max_sz = 250_000 if _k in ("dist", "par", "pq", "vis", "visited", "seen", "queue", "q") else 4_000_000
                        if _dsz <= _max_sz:
                            _to_save[_k] = _dumped
                            _saved_keys.add(_k)
                    except Exception:
                        pass
                _blob = _sb_pickle.dumps(_to_save)
                if len(_blob) <= 16_000_000:
                    _tmp_vars_path = _vars_game_path + ".tmp"
                    with builtins.open(_tmp_vars_path, "wb") as _vf:
                        _vf.write(_blob)
                    os.replace(_tmp_vars_path, _vars_game_path)
            except Exception:
                pass
            return _saved_keys

        def _collect_retention_payload():
            _clean_injected_stdlib()
            if not _persistence_enabled:
                _save_vars_post_collect({})
                return {}
            try:
                kept, rejected = collect_functions(
                    _code_text, _kept_pool, runtime_globals, _available_names,
                    **_retention_options)
                _saved_keys = _save_vars_post_collect(kept)
                _all_rejected = [
                    r for r in (_retention_rejected + rejected)
                    if not (isinstance(r, dict) and r.get("name") in _saved_keys)
                ]
                return {
                    "keepable_functions": [{"name": n, "source": src} for n, src in kept.items()],
                    "retention_rejected": _all_rejected,
                }
            except Exception:
                # Bookkeeping failures may drop helpers, but must never replace
                # the snippet's original error/output or replay its actions.
                _save_vars_post_collect({})
                return {"keepable_functions": [], "retention_rejected": [
                    {"name": "all", "reason": "retention failed; redefine needed functions"}]}

        class TimeoutError(BaseException):
            pass

        def _handle_sigalrm(signum, frame):
            raise TimeoutError(
                f"Tool timed out after {timeout_seconds}s; "
                f"{len(action_results)} action(s) had already executed"
            )

        try:
            import signal as _sb_signal
            _sb_signal.signal(_sb_signal.SIGALRM, _handle_sigalrm)
            _eff_timeout = 45.0 if int(timeout_seconds) == 30 else float(timeout_seconds)
            _init_budget = max(0.5, _eff_timeout - 0.5)
            _ITIMER_STATE["sig"] = _sb_signal
            _ITIMER_STATE["deadline"] = _sb_time.monotonic() + _init_budget
            _ITIMER_STATE["active"] = True
            _sb_signal.setitimer(_sb_signal.ITIMER_REAL, _init_budget)
        except Exception:
            _sb_signal = None

        try:
            _code_text = str(initial.get("code", ""))
            _tree = ast.parse(_code_text, "<python_tool>")
            _last_expr = None
            if _tree.body and isinstance(_tree.body[-1], ast.Expr):
                _last_expr = ast.Expression(_tree.body[-1].value)
                ast.copy_location(_last_expr, _tree.body[-1])
                _tree.body = _tree.body[:-1]
            with contextlib.redirect_stdout(stdout):
                try:
                    if _tree.body:
                        exec(compile(_tree, "<python_tool>", "exec"), runtime_globals, runtime_globals)
                    if _last_expr is not None:
                        _value = eval(compile(_last_expr, "<python_tool>", "eval"), runtime_globals, runtime_globals)
                        if _value is not None:
                            print(repr(_value))
                            if runtime_globals.get("result") is None:
                                runtime_globals["result"] = _value
                except _SandboxExit:
                    pass
            _ITIMER_STATE["active"] = False
            if _sb_signal is not None:
                _sb_signal.setitimer(_sb_signal.ITIMER_REAL, 0.0)
            _send(
                {
                    "type": "final",
                    **_collect_retention_payload(),
                    "stdout": stdout.getvalue(),
                    "result": _json_safe(runtime_globals.get("result")),
                    "action_results": _json_safe(action_results),
                }
            )
        except (
            KnownNoOpActionError,
            RepeatedActionInStateError,
            KnownDeathActionError,
            TerminalStateActionError,
            TimeoutError,
            Exception,
        ) as exc:
            _ITIMER_STATE["active"] = False
            if _sb_signal is not None:
                try:
                    _sb_signal.setitimer(_sb_signal.ITIMER_REAL, 0.0)
                except Exception:
                    pass"""

SEND_JSON_BEFORE = """def _send_json_line(handle: Any, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False) + "\\n")
    handle.flush()"""

SEND_JSON_AFTER = """def _send_json_line(handle: Any, payload: dict[str, Any]) -> None:
    try:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\\n")
        handle.flush()
    except (BrokenPipeError, OSError):
        return"""

HOST_IPC_BEFORE = """        deadline = time.monotonic() + max(1, int(timeout_seconds))
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _kill_process_group(process)
                _wait_for_process_exit(process)
                return {
                    "error": f"Tool timed out after {timeout_seconds}s",
                    "stdout": "",
                    "action_results": list(host_action_results),
                }

            try:
                line = stdout_queue.get(timeout=remaining)
            except queue.Empty:
                continue
            if line is None:
                stderr = process.stderr.read()
                _wait_for_process_exit(process)
                return {
                    "error": _sanitize_host_error_text(stderr),
                    "stdout": "",
                    "action_results": list(host_action_results),
                }

            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                stderr = process.stderr.read()
                _kill_process_group(process)
                _wait_for_process_exit(process)
                return {
                    "error": "Sandbox process returned an invalid response.",
                    "stdout": "",
                    "action_results": list(host_action_results),
                }

            msg_type = str(message.get("type", "")).strip()
            if msg_type == "animation":
                if animation_handler is None:
                    _send_json_line(
                        process.stdin,
                        {"type": "animation_error",
                         "error": "Animation frames are not available in this session."},
                    )
                    continue
                try:
                    animation_payload = animation_handler(dict(message.get("request") or {}))
                except Exception:  # noqa: BLE001
                    _send_json_line(
                        process.stdin,
                        {"type": "animation_error",
                         "error": "animation failed in sandbox host."},
                    )
                    continue
                _send_json_line(
                    process.stdin,
                    {"type": "animation_result", "animation": animation_payload},
                )
                continue
            if msg_type == "action":
                try:
                    # the message carries more than the action list: stale_after
                    # tells the solver a previous action in this snippet changed
                    # nothing, which it weighs after its own guards
                    action_result_payload = action_handler(
                        list(message.get("actions") or []),
                        stale_after=message.get("stale_after"),
                    )
                except Exception:  # noqa: BLE001
                    _send_json_line(
                        process.stdin,
                        {
                            "type": "action_error",
                            "error": "action failed in sandbox host.",
                        },
                    )
                    continue
                raw_action_result = action_result_payload.get("action_result") or {}
                if isinstance(raw_action_result, dict):
                    host_action_results.append(dict(raw_action_result))
                _send_json_line(
                    process.stdin,
                    {
                        "type": "action_result",
                        "action_result": raw_action_result,
                        "state": action_result_payload.get("state") or {},
                    },
                )
                continue"""

HOST_IPC_AFTER = """        _eff_timeout = 45.0 if int(timeout_seconds) == 30 else float(max(1, int(timeout_seconds)))
        deadline = time.monotonic() + _eff_timeout + 4.0
        _ipc_budget = 240.0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _kill_process_group(process)
                _wait_for_process_exit(process)
                return {
                    "error": (
                        f"Tool timed out after {timeout_seconds}s; "
                        f"{len(host_action_results)} action(s) had already executed - their results "
                        "are listed, but the snippet's printed output was lost"
                    ),
                    "stdout": "",
                    "action_results": list(host_action_results),
                }

            try:
                line = stdout_queue.get(timeout=remaining)
            except queue.Empty:
                continue
            if line is None:
                stderr = process.stderr.read()
                _wait_for_process_exit(process)
                return {
                    "error": _sanitize_host_error_text(stderr),
                    "stdout": "",
                    "action_results": list(host_action_results),
                }

            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                stderr = process.stderr.read()
                _kill_process_group(process)
                _wait_for_process_exit(process)
                return {
                    "error": "Sandbox process returned an invalid response.",
                    "stdout": "",
                    "action_results": list(host_action_results),
                }

            msg_type = str(message.get("type", "")).strip()
            if msg_type == "animation":
                if animation_handler is None:
                    _send_json_line(
                        process.stdin,
                        {"type": "animation_error",
                         "error": "Animation frames are not available in this session."},
                    )
                    continue
                _ipc_t0 = time.monotonic()
                try:
                    animation_payload = animation_handler(dict(message.get("request") or {}))
                except Exception:  # noqa: BLE001
                    _credit = min(_ipc_budget, max(0.0, time.monotonic() - _ipc_t0))
                    _ipc_budget -= _credit
                    deadline += _credit
                    _send_json_line(
                        process.stdin,
                        {"type": "animation_error",
                         "error": "animation failed in sandbox host."},
                    )
                    continue
                _credit = min(_ipc_budget, max(0.0, time.monotonic() - _ipc_t0))
                _ipc_budget -= _credit
                deadline += _credit
                _send_json_line(
                    process.stdin,
                    {"type": "animation_result", "animation": animation_payload},
                )
                continue
            if msg_type == "action":
                _ipc_t0 = time.monotonic()
                try:
                    # the message carries more than the action list: stale_after
                    # tells the solver a previous action in this snippet changed
                    # nothing, which it weighs after its own guards
                    action_result_payload = action_handler(
                        list(message.get("actions") or []),
                        stale_after=message.get("stale_after"),
                    )
                except Exception:  # noqa: BLE001
                    _credit = min(_ipc_budget, max(0.0, time.monotonic() - _ipc_t0))
                    _ipc_budget -= _credit
                    deadline += _credit
                    _send_json_line(
                        process.stdin,
                        {
                            "type": "action_error",
                            "error": "action failed in sandbox host.",
                        },
                    )
                    continue
                _credit = min(_ipc_budget, max(0.0, time.monotonic() - _ipc_t0))
                _ipc_budget -= _credit
                deadline += _credit
                raw_action_result = action_result_payload.get("action_result") or {}
                if isinstance(raw_action_result, dict):
                    host_action_results.append(dict(raw_action_result))
                _send_json_line(
                    process.stdin,
                    {
                        "type": "action_result",
                        "action_result": raw_action_result,
                        "state": action_result_payload.get("state") or {},
                    },
                )
                continue"""

SOLVER_VERDICT_BEFORE = """    if item.get("level_completed"):
        return "LEVEL COMPLETED"
    changed = item.get("gameplay_changed")
    if changed is False:"""

SOLVER_VERDICT_AFTER = """    if item.get("level_completed"):
        return "LEVEL COMPLETED"
    changed = item.get("gameplay_changed")
    if changed is False and (item.get("animation") or {}).get("transient_pixels", 0):
        return "animated effect (intermediate frames changed; final board restored)"
    if changed is False:"""

SOLVER_BATCH_BEFORE = """                and NOOP_GUARD_BORDER_WIDTH > 0
                and batch_size >= 2
                and not payload.get("gameplay_changed", True)
            ):
                stop_reason = "no_op_action"
                break"""

SOLVER_BATCH_AFTER = """                and NOOP_GUARD_BORDER_WIDTH > 0
                and batch_size >= 2
                and not payload.get("gameplay_changed", True)
                # A transient animation is evidence of a real intermediate effect;
                # the final frame alone cannot certify a no-op.
                and not (payload.get("animation") or {}).get("transient_pixels", 0)
            ):
                stop_reason = "no_op_action"
                break"""

INTERIOR_CHANGED_BEFORE = """    for r in range(border, rows - border):
        prev_row = previous_grid[r]
        new_row = new_grid[r]
        if prev_row[border:cols - border] != new_row[border:cols - border]:
            return True
    return False"""

INTERIOR_CHANGED_AFTER = """    for r in range(border, rows - border):
        prev_row = previous_grid[r]
        new_row = new_grid[r]
        if prev_row[border:cols - border] != new_row[border:cols - border]:
            return True
    if previous_grid == new_grid:
        return False
    changed_rows = 0
    changed_cols_mask = [0] * cols
    total_changed = 0
    for r in range(rows):
        prev_row = previous_grid[r]
        new_row = new_grid[r]
        if prev_row != new_row:
            row_diff = 0
            for c in range(cols):
                if prev_row[c] != new_row[c]:
                    row_diff += 1
                    changed_cols_mask[c] += 1
            total_changed += row_diff
            if row_diff >= 4:
                changed_rows += 1
    changed_cols = sum(1 for cnt in changed_cols_mask if cnt >= 3)
    if total_changed >= 18 and (
        (changed_rows >= 3 and sum(1 for cnt in changed_cols_mask if cnt >= 1) >= 6)
        or (changed_cols >= 3 and sum(1 for r in range(rows) if previous_grid[r] != new_grid[r]) >= 6)
    ):
        return True
    return False"""

TOOL_AGENT_VERDICT_BEFORE = """            elif item.get("level_completed"):
                verdict = "LEVEL COMPLETED"
            elif item.get("gameplay_changed") is False:"""

TOOL_AGENT_VERDICT_AFTER = """            elif item.get("level_completed"):
                verdict = "LEVEL COMPLETED"
            elif item.get("gameplay_changed") is False and (item.get("animation") or {}).get("transient_pixels", 0):
                verdict = "animated effect (intermediate frames changed; final board restored)"
            elif item.get("gameplay_changed") is False:"""

RETAINED_BEFORE = """    def _record_retained_functions(self, sandbox_result, payload) -> None:
        if not _persistent_functions():
            return
        try:"""

RETAINED_AFTER = """    def _record_retained_functions(self, sandbox_result, payload) -> None:
        if not _persistent_functions():
            return
        if isinstance(sandbox_result, dict) and "keepable_functions" not in sandbox_result and self._kept_functions:
            payload["function_retention"] = (
                "The snippet did not finish, so nothing new was retained; "
                "your retained functions from earlier calls are unchanged."
            )
            return
        try:"""

SEED_ID_BEFORE = """            return {
                "current_frame": current_frame_payload,
                "history": _ascii_history_view_payload(refreshed_history),"""

SEED_ID_AFTER = """            return {
                "seed_id": str(state_path),
                "current_frame": current_frame_payload,
                "history": _ascii_history_view_payload(refreshed_history),"""


def apply(root):
    root = Path(root)

    # 1. animation.py empty schema repair
    anim_path = root / "inference/utils/animation.py"
    anim_text = anim_path.read_text(encoding="utf-8")
    if ANIM_AFTER not in anim_text:
        if anim_text.count(ANIM_BEFORE) != 1:
            raise RuntimeError("Expected the original animation empty-result block exactly once")
        anim_text = anim_text.replace(ANIM_BEFORE, ANIM_AFTER, 1)
        ast.parse(anim_text)
        anim_path.write_text(anim_text, encoding="utf-8")

    # 2. python_tool_sandbox.py transient animation + SAFE_MODULES/BUILTINS + view polish + StaleStateActionError disarm + IPC-paused SIGALRM
    sb_path = root / "inference/agent/python_tool_sandbox.py"
    sb_text = sb_path.read_text(encoding="utf-8")
    for b_str, a_str, label in (
        (SANDBOX_BEFORE, SANDBOX_AFTER, "SANDBOX_BEFORE"),
        (SAFE_MODULES_BEFORE, SAFE_MODULES_AFTER, "SAFE_MODULES_BEFORE"),
        (SAFE_BUILTINS_BEFORE, SAFE_BUILTINS_AFTER, "SAFE_BUILTINS_BEFORE"),
        (HISTORY_ENTRY_BEFORE, HISTORY_ENTRY_AFTER, "HISTORY_ENTRY_BEFORE"),
        (JSON_SAFE_BEFORE, JSON_SAFE_AFTER, "JSON_SAFE_BEFORE"),
        (SANDBOX_OPEN_BEFORE, SANDBOX_OPEN_AFTER, "SANDBOX_OPEN_BEFORE"),
        (STALE_STATE_RAISE_BEFORE, STALE_STATE_RAISE_AFTER, "STALE_STATE_RAISE_BEFORE"),
        (LAZY_ANIM_BEFORE, LAZY_ANIM_AFTER, "LAZY_ANIM_BEFORE"),
        (SET_LIMITS_BEFORE, SET_LIMITS_AFTER, "SET_LIMITS_BEFORE"),
        (REFRESH_LAST_RES_BEFORE, REFRESH_LAST_RES_AFTER, "REFRESH_LAST_RES_BEFORE"),
        (IPC_CHILD_BEFORE, IPC_CHILD_AFTER, "IPC_CHILD_BEFORE"),
        (EXEC_BLOCK_BEFORE, EXEC_BLOCK_AFTER, "EXEC_BLOCK_BEFORE"),
        (SEND_JSON_BEFORE, SEND_JSON_AFTER, "SEND_JSON_BEFORE"),
        (HOST_IPC_BEFORE, HOST_IPC_AFTER, "HOST_IPC_BEFORE"),
    ):
        if a_str not in sb_text:
            if sb_text.count(b_str) != 1:
                raise RuntimeError(f"Expected {label} once, got {sb_text.count(b_str)}")
            sb_text = sb_text.replace(b_str, a_str, 1)
    ast.parse(sb_text)
    sb_path.write_text(sb_text, encoding="utf-8")

    # 3. solver.py _action_verdict, batch no-op guard, and 2D border gameplay detector
    solver_path = root / "inference/framework/solver.py"
    solver_text = solver_path.read_text(encoding="utf-8")
    for b_str, a_str, label in (
        (SOLVER_VERDICT_BEFORE, SOLVER_VERDICT_AFTER, "SOLVER_VERDICT_BEFORE"),
        (SOLVER_BATCH_BEFORE, SOLVER_BATCH_AFTER, "SOLVER_BATCH_BEFORE"),
        (INTERIOR_CHANGED_BEFORE, INTERIOR_CHANGED_AFTER, "INTERIOR_CHANGED_BEFORE"),
    ):
        if a_str not in solver_text:
            if solver_text.count(b_str) != 1:
                raise RuntimeError(f"Expected {label} once, got {solver_text.count(b_str)}")
            solver_text = solver_text.replace(b_str, a_str, 1)
    ast.parse(solver_text)
    solver_path.write_text(solver_text, encoding="utf-8")

    # 4. tool_agent.py turn-trace animated verdict + timeout retained-functions preservation
    ta_path = root / "inference/agent/tool_agent.py"
    ta_text = ta_path.read_text(encoding="utf-8")
    if TOOL_AGENT_VERDICT_AFTER not in ta_text:
        if ta_text.count(TOOL_AGENT_VERDICT_BEFORE) != 1:
            raise RuntimeError(f"Expected TOOL_AGENT_VERDICT_BEFORE once, got {ta_text.count(TOOL_AGENT_VERDICT_BEFORE)}")
        ta_text = ta_text.replace(TOOL_AGENT_VERDICT_BEFORE, TOOL_AGENT_VERDICT_AFTER, 1)
    if RETAINED_AFTER not in ta_text:
        if ta_text.count(RETAINED_BEFORE) != 1:
            raise RuntimeError(f"Expected RETAINED_BEFORE once, got {ta_text.count(RETAINED_BEFORE)}")
        ta_text = ta_text.replace(RETAINED_BEFORE, RETAINED_AFTER, 1)
    if SEED_ID_AFTER not in ta_text:
        if ta_text.count(SEED_ID_BEFORE) != 1:
            raise RuntimeError(f"Expected SEED_ID_BEFORE once, got {ta_text.count(SEED_ID_BEFORE)}")
        ta_text = ta_text.replace(SEED_ID_BEFORE, SEED_ID_AFTER, 1)
    ast.parse(ta_text)
    ta_path.write_text(ta_text, encoding="utf-8")


if __name__ == "__main__":
    apply(sys.argv[1])
'''

PATCH_SEGMENTATION_ALIASES = r'''"""Install zero-global M75 _NodeDict and _BoundaryList key aliases strictly inside segmentation.py."""
import ast
from pathlib import Path
import sys

NODE_BEFORE = """        nodes.append(
            {
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
            }
        )"""

NODE_AFTER = """        nodes.append(
            _NodeDict({
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
            })
        )"""

SEG_ADDENDUM = """

class _CoordList(list):
    \"\"\"Hashable [r, c] coordinate pair for node['boundary'] so set(n['boundary'])
    and (r, c) in set(n['boundary']) work without TypeError while remaining a list.\"\"\"

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
        return (tuple, (tuple(self),))


class _BoundaryList(list):
    \"\"\"List subclass for node['boundary'] that safely resolves index 2 / 3 on 1D or 1px contours
    when models write n['boundary'][0], n['boundary'][2] expecting [top_left, ..., bottom_right].\"\"\"

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
    \"\"\"Zero-stdout-overhead dict subclass for segment_layer nodes (rellik13 M75).
    Preserves exact repr(), keys(), and JSON serialization while resolving common
    model key aliases (bbox/box/bounds, size/area/count, centroid/center) on demand.
    Does NOT touch python_tool_sandbox.py runtime_globals or _available_names.\"\"\"

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
            return ((min(rs) + max(rs)) // 2, (min(cs) + max(cs)) // 2)
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
"""


def apply(root):
    root = Path(root)
    seg_path = root / "inference/utils/segmentation.py"
    seg_src = seg_path.read_text(encoding="utf-8")
    if "class _NodeDict(" not in seg_src:
        if seg_src.count(NODE_BEFORE) != 1:
            raise RuntimeError(f"Expected NODE_BEFORE once in segmentation.py, got {seg_src.count(NODE_BEFORE)}")
        seg_src = seg_src.replace(NODE_BEFORE, NODE_AFTER, 1) + SEG_ADDENDUM
        ast.parse(seg_src)
        seg_path.write_text(seg_src, encoding="utf-8")


if __name__ == "__main__":
    apply(sys.argv[1])
'''


def _src(cell: dict) -> str:
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else str(s)


def extract_v35_assets() -> dict[str, str]:
    return {
        "patch_transient_animation.py": PATCH_TRANSIENT_AND_TIMEOUT,
        "patch_segmentation_aliases.py": PATCH_SEGMENTATION_ALIASES,
    }


def build_v35_notebook() -> dict:
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

    v35_assets = extract_v35_assets()

    # Build 23-cell notebook anchored on v11 / v24 / shiiin9
    nb_v35 = copy.deepcopy(nb_v11)
    assert len(nb_v35["cells"]) == 23

    # Cell 0: Markdown overview
    nb_v35["cells"][0]["source"] = CELL0_MARKDOWN

    # Cell 2: Base patch (preserved 100%, 498,799 bytes, SHA-256 6fdcd1ae...)
    cell2_src = _src(nb_v11["cells"][2])
    assert len(cell2_src) == 498799, f"Cell 2 length mismatch: expected 498799, got {len(cell2_src)}"
    cell2_sha = hashlib.sha256(cell2_src.encode("utf-8")).hexdigest()
    assert cell2_sha == "6fdcd1ae7445c0cfc26eec1047363aa0c196c867d289a35b6f64403b5a43dce2", f"Cell 2 SHA mismatch: {cell2_sha}"
    nb_v35["cells"][2]["source"] = cell2_src

    # Cell 4: Pure v24/shiiin9 startup config + dual mount path resolver + 2 verified v36 per-game solver patches
    c4_src = _src(nb_v11["cells"][4])
    old_startup_cfg = """NOTEBOOK_START_TIME = time.time()
SERVER_STARTUP_TIMEOUT = 12 * 60
os.environ['ARC3_WARMUP_ACTION_GAMES'] = '1'
os.environ['ARC3_HTTP_RETRY_INITIAL_SECONDS'] = '900'"""
    new_startup_cfg = """TRUE_SUBMISSION = os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1", "true"}
NOTEBOOK_START_TIME = time.time()
SERVER_STARTUP_TIMEOUT = (12 * 60) if TRUE_SUBMISSION else (35 * 60)
os.environ['ARC3_WARMUP_ACTION_GAMES'] = '1'
os.environ['ARC3_HTTP_RETRY_INITIAL_SECONDS'] = '900' if TRUE_SUBMISSION else '2400'"""
    assert old_startup_cfg in c4_src, "Could not find old_startup_cfg in cell 4"
    c4_src = c4_src.replace(old_startup_cfg, new_startup_cfg, 1)

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
    assert old_paths_cfg in c4_src, "Could not find old_paths_cfg in cell 4"
    c4_src = c4_src.replace(old_paths_cfg, new_paths_cfg, 1)

    old_apply = """subprocess.run(["git", "apply", "--include=ARC3-Inference/*", "-v", "/kaggle/harness-changes.patch"], cwd=f"{BUNDLE_DIR}/src", check=True)
print('harness patch applied successfully')"""
    assert old_apply in c4_src, "Could not find old_apply in cell 4"

    v35_post_patch_block = f"""subprocess.run(["git", "apply", "--include=ARC3-Inference/*", "-v", "/kaggle/harness-changes.patch"], cwd=f"{{BUNDLE_DIR}}/src", check=True)
print('harness patch applied successfully')

# Apply Version 36 surgical per-game solver modules
_v35_dir = Path('/kaggle/arcagi_v35')
_v35_dir.mkdir(exist_ok=True)
_v35_assets = {repr(v35_assets)}
for _name, _text in _v35_assets.items():
    (_v35_dir / _name).write_text(_text, encoding='utf-8')
for _patch_script in ('patch_transient_animation.py', 'patch_segmentation_aliases.py'):
    subprocess.run([sys.executable, str(_v35_dir / _patch_script), str(BUNDLE_DIR / 'src/ARC3-Inference')], check=True)
print('Applied v36 modules cleanly (patch_transient_animation, patch_segmentation_aliases)')"""
    c4_src = c4_src.replace(old_apply, v35_post_patch_block, 1)

    nb_v35["cells"][4]["source"] = c4_src

    # Cell 6: Preserve dfranzen/v24 precache_model_thread (delay=60, threads=3)
    c6_src = _src(nb_v11["cells"][6])
    expected_precache_model = "precache_model_thread = threading.Thread(target=precache, args=(MODEL_DIR, DRAFT_MODEL_DIR), kwargs={'delay': 60, 'threads': 3})"
    assert expected_precache_model in c6_src, "Could not find expected_precache_model in cell 6"
    nb_v35["cells"][6]["source"] = c6_src

    # Cell 12: Preserve precache_model_thread.start() and add 1x auto-relaunch
    c12_src = _src(nb_v11["cells"][12])
    old_launch = """# ---- launch detached and wait for health ----
precache_model_thread.start()
print("\\n" + shlex.join(args) + "\\n")
log_offset = Path(LOG).stat().st_size if Path(LOG).exists() else 0
with open(LOG, "ab", buffering=0) as logf:
    proc = subprocess.Popen(args, env=env, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
print(f"pid {proc.pid} -> {LOG}")

def show_log_tail():
    with open(LOG, "rb") as f:
        f.seek(0, 2)
        f.seek(max(log_offset, f.tell()-20000))
        print(f.read().decode(errors="replace"))

url, t0 = f"http://127.0.0.1:{SERVED_MODEL_PORT}/health", time.time()
last_report = -30
while time.time() - NOTEBOOK_START_TIME < SERVER_STARTUP_TIMEOUT:
    if proc.poll() is not None:
        show_log_tail()
        break"""
    new_launch = """# ---- launch detached and wait for health ----
precache_model_thread.start()
print("\\n" + shlex.join(args) + "\\n")
log_offset = Path(LOG).stat().st_size if Path(LOG).exists() else 0
with open(LOG, "ab", buffering=0) as logf:
    proc = subprocess.Popen(args, env=env, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
print(f"pid {proc.pid} -> {LOG}")

def show_log_tail():
    with open(LOG, "rb") as f:
        f.seek(0, 2)
        f.seek(max(log_offset, f.tell()-20000))
        print(f.read().decode(errors="replace"))

url, t0 = f"http://127.0.0.1:{SERVED_MODEL_PORT}/health", time.time()
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
    assert old_launch in c12_src, "Could not find old_launch in cell 12"
    c12_src = c12_src.replace(old_launch, new_launch, 1)
    nb_v35["cells"][12]["source"] = c12_src

    # Cell 16: Byte-for-byte identical to v24 (`28.79` LB) and shiiin9 (`31.54` LB) Cells 17 + 21 (`b0 = 0.0`!)
    solver_cfg_src = _src(nb_shiiin["cells"][17]).rstrip()
    d_prime_src = _src(nb_shiiin["cells"][21]).rstrip()
    nb_v35["cells"][16]["source"] = solver_cfg_src + "\n\n" + d_prime_src + "\n"

    # Cell 20: Byte-for-byte identical to v24 (`28.79` LB) and shiiin9 (`31.54` LB) Cells 22 + 23
    c20_src = _src(nb_v11["cells"][20]).rstrip()
    fresh_print_src = _src(nb_shiiin["cells"][23]).rstrip()
    nb_v35["cells"][20]["source"] = c20_src + "\n" + fresh_print_src + "\n"

    # Clear any stale outputs or execution_counts across all cells
    for cell in nb_v35["cells"]:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    return nb_v35


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
    nb_v35 = build_v35_notebook()
    OUTPUT_V35_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_SUB_PATH.parent.mkdir(parents=True, exist_ok=True)

    for path in (OUTPUT_V35_PATH, OUTPUT_SUB_PATH):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb_v35, f, indent=1, ensure_ascii=False)
            f.write("\n")

    update_kernel_metadata()

    sha_v35 = hashlib.sha256(OUTPUT_V35_PATH.read_bytes()).hexdigest()
    sha_sub = hashlib.sha256(OUTPUT_SUB_PATH.read_bytes()).hexdigest()
    assert sha_v35 == sha_sub, "submission_v36.ipynb and /tmp/v36_kernel/submission.ipynb must be byte-identical"
    assert len(nb_v35["cells"]) == 23, f"Expected 23 cells, got {len(nb_v35['cells'])}"
    print(
        f"Built {OUTPUT_V35_PATH} and {OUTPUT_SUB_PATH} "
        f"({OUTPUT_V35_PATH.stat().st_size} bytes, sha256={sha_v35[:12]}, cells=23)"
    )


if __name__ == "__main__":
    main()

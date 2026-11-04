"""Harness v2 runtime: everything agent.py / llm.py need beyond the core game loop.

  - per-game context (thread-local): objects cache, last action frames, per-turn log
  - object-level feedback for every executed action (counts, moves, bounded CSV rects)
  - the interactive `run` sandbox with act(), and the guard bookkeeping for it
  - unified transition-model resolution (cells / objects / symbolic) and masked validation
  - death bookkeeping (indicator depleted?) and extra images (diff image, death image)
  - level-start object inventory text
  - replay/event logging payloads
"""
from __future__ import annotations

import copy
import hashlib
import re
import struct
import threading
import time
import zlib
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import act_sandbox as _act
from . import frame_diff as _fd
from . import gridlib as _gl
from . import modeljobs as _mj
from . import objects as _obj
from . import scene_feedback as _sfb
from . import reference_checks
from .animation_archive import AnimationArchive
from .minimal_coverage import ActionCoverage
from .object_refs import ObjectReferences
from .exploration_feedback import ExplorationFeedback, shape_changes, rect_text
from .board import compute_changed_rects, format_rect_region, norm_rects, rects_mask, to_uint8_board, board_to_png_data_url
from .config import (
    MINIMAL_HARNESS,
    EVIDENCE_NOTES,
    ACTION_LABEL_ALIASES,
    ANIM_AUTO_DIFF_LINES,
    ANIM_MAX_FRAMES,
    BFS_TIME_LIMIT_SECONDS,
    BOARD_COLOR_MAP_HEX,
    BOARD_COLOR_NAME_BY_ID,
    MINIMAL_COLOR_NAMES,
    DEATH_IMAGE,
    WIN_FRAME_IMAGE,
    WIN_TRANSITION_FRACTION,
    DIFF_IMAGE,
    ENABLE_REPLAYS,
    LAYERED_TRACKER,
    LAYERED_TRACKER_GAME_BUDGET,
    LAYERED_TRACKER_CHARS,
    LAYERED_TRACKER_LINES,
    LAYERED_TRACKER_PREWARM_TIMEOUT,
    LAYERED_TRACKER_TIMEOUT,
    MAX_ACT_ACTIONS,
    MAX_ACTION_DEPTH,
    MAX_DISTINCT_STATES,
    MODEL_NAME,
    OBJ_CHANGES_CAP,
    PROMPT_VARIANT,
    RECT_CSV_MAX_CELLS,
    board_small_image_scale_for_model,
    FULL_ARRAY_PRINT,
    STORE_MAX_MB,
    MODEL_CODE_IN_WORKER,
    MODEL_CHECK_IMAGE,
    MODEL_CHECK_TIME_LIMIT,
    CODE_LOAD_TIME_LIMIT,
    PATH_MEM_LIMIT_MB,
    STEP_REPORT_COMPRESS,
    STEP_REPORT_FULL_CHARS,
)
from .log import board_to_hex_rows
from .sandbox import _postprocess_numpy_to_csv, is_safe, keep_functions_from_run, safety_error
from .validation import CandidateAction

DIFF_UNCHANGED_RGB = (16, 24, 48)  # off-palette dark navy

_PKG_DIR = Path(__file__).resolve().parent


def color_names() -> dict[int, str]:
    names = MINIMAL_COLOR_NAMES if MINIMAL_HARNESS else BOARD_COLOR_NAME_BY_ID
    return {int(k): str(v) for k, v in names.items()}


_gl.bind(_obj, _fd, color_names())   # v2.11: the lib module (also used for the harness's own change lines)


# ============================================================================ context

class GameCtx:
    def __init__(self, game_id: str, board: np.ndarray) -> None:
        self.memory = {"notes": {}}
        self.display_pending_images = []
        self.shown_images = []
        self.memory_events = []
        self.action_ledger = []
        self.action_coverage = ActionCoverage()         # level-scoped, survives resets/worker loss
        self.game_id = game_id
        self.obj_cache: dict[tuple[bytes, tuple[int, ...] | None], list[dict[str, Any]]] = {}
        self.obj_cache_order: list[tuple[bytes, tuple[int, ...] | None]] = []
        self.last_frames: list[np.ndarray] = []          # animation frames of the last executed action
        self.animation_archive = AnimationArchive()     # parent only; current level, across resets
        self.animation_observed = False                 # includes frames omitted by the storage cap
        self.attempt_steps = 0                          # successful env steps; refusals do not count
        self.manual_resets = 0                          # scoped to this level
        self.available_actions: list[str] = []
        self.exploration = ExplorationFeedback()
        self.last_step: dict[str, Any] | None = None     # result of the last executed action
        self.steps: list[dict[str, Any]] = []            # all executed actions this turn
        self.guards: list[dict[str, Any]] = []           # guard events this turn
        self.run_calls: list[dict[str, Any]] = []        # run() calls this turn
        self.turn_actions_executed = 0                   # actions executed via act() this turn
        self.turn_latch: str | None = None               # once set, no more act() this turn
        self.snippet_stale: str | None = None            # stale-state guard within one run() call
        self.new_color_latch: str | None = None          # v2.9: a new colour appeared in this run() call -> further act() refused
        self.store_version: int = 0                      # v2.10: bumped whenever the store changes (validation cache key)
        self.after_env_error: bool = False               # v2.10: the next recorded transition follows an environment error
        self.env_step_failed: bool = False               # v2.9: the last env.step() raised or returned no observation
        self.win_board: np.ndarray | None = None         # v2.9: the winning moment of the last completed level
        self.win_level: int = 0
        self.win_pending_image: bool = False
        self.pending_terminal: tuple[str, int, int, list[str]] | None = None
        self.death_board: np.ndarray | None = None
        self.death_text: str = ""
        self.death_pending_image: bool = False
        self.last_request_board: np.ndarray | None = None
        self.prev_level_start_board: np.ndarray | None = None
        self.level_start_board: np.ndarray = to_uint8_board(board).copy()
        self.extra_images: list[tuple[str, str, int, str | None]] = []
        self.model_mode: str = "none"
        self.validation: dict[str, Any] | None = None
        self.prediction_mismatch: dict[str, Any] | None = None
        self.search_space: tuple[Any, ...] | None = None
        self.known_noop: set[tuple[bytes, str]] = set()      # (board key without indicator, action text) seen to change nothing
        self.known_death: set[tuple[bytes, str]] = set()     # (board key, action text) that caused a game over
        self.last_refused: tuple[bytes, str] | None = None   # the pair refused last, for the "repeat" escape
        self.tracker: Any = None                             # v2.6: scene_worker.TrackerClient of this game (None = off)
        self.tracker_scene: dict[str, Any] | None = None     # the tracker's scene after the latest board
        self.tracker_off: str | None = None                  # why the tracker stopped for this game
        self.tracker_stats_final: dict[str, Any] | None = None
        self.tracker_skipping = False
        # v2.6.5: the tracker's scene (object table) per board of the current attempt, and the edits per transition
        self.scenes: list[Any] = []                          # aligned with the game loop's all_boards (None = unknown)
        self.scene_edits: list[Any] = []                     # scene_edits[i] explains all_boards[i] -> all_boards[i+1]
        self.scene_by_hash: dict[bytes, Any] = {}            # board hash -> latest scene table for that board
        self.scene_by_hash_order: list[bytes] = []
        # v2.11
        self.view_exclude: list[int] | None = None           # set_object_view: colours treated as gaps in the reports
        self.mc_sig: Any = None                              # what the last shown model check report said
        self.mc_first: Any = None                            # the first difference the last image showed
        self.mc_image: tuple[Any, ...] | None = None         # (before, predicted, actual, mismatch) for the next request
        self.ns_src: str | None = None                       # next_state's source when it last changed
        self.ns_level: int | None = None                     # the level on which it last changed
        self.mc_shown_level: int | None = None               # the level of the last shown model check report
        self.references = ObjectReferences(game_id, board, color_names())


_tls = threading.local()


def start_game(game_id: str, board: np.ndarray) -> GameCtx:
    c = GameCtx(game_id, board)
    _tls.ctx = c
    return c


def ctx() -> GameCtx:
    c = getattr(_tls, "ctx", None)
    if c is None:
        c = start_game("unknown", np.zeros((64, 64), dtype=np.uint8))
    return c


def begin_turn() -> None:
    c = ctx()
    if MINIMAL_HARNESS:
        c.action_ledger = []
        c.shown_images = []
    c.exploration.begin_turn()
    c.references.begin_turn()
    c.steps = []
    c.guards = []
    c.run_calls = []
    c.turn_actions_executed = 0
    c.turn_latch = None
    c.snippet_stale = None
    c.new_color_latch = None
    c.pending_terminal = None
    c.prediction_mismatch = None


# ============================================================================ v2.6 layered tracker

def reference_sync(board, level, attempt, step):
    if MINIMAL_HARNESS:
        return {}
    refs = ctx().references
    refs.configure(level, attempt, step)
    return refs.refresh(board)


def reference_text(full=False):
    if MINIMAL_HARNESS:
        return ""
    return ctx().references.text(full=full)


def reference_expect(raw):
    return reference_checks.expectations(raw)


def reference_request(method, args, kwargs):
    return ctx().references.request(method, args, kwargs)


def resolve_named_action(text, board):
    if MINIMAL_HARNESS:
        return text, None
    parts = text.split()
    if len(parts) in (2, 4) and normalize_label(parts[0]) == "CLICK" and (parts[1].startswith("@") or re.fullmatch(r"[CG]\d+", parts[1])):
        try:
            color = None
            if len(parts) == 4:
                if parts[2].lower() != "color":
                    raise ValueError("reference click syntax: CLICK C11 [color 3] or CLICK @name")
                color = int(parts[3])
                if not 0 <= color <= 15:
                    raise ValueError("click colour must be 0..15")
            refs = ctx().references
            refs.refresh(board)
            r, c, value = refs.resolve_click(parts[1], color)
        except ValueError as exc:
            return text, str(exc)
        return f"CLICK {r} {c} {value}", None
    return text, None

TRACKER_HEADER = "  Layered tracker:"
TRACKER_HEADER_REREAD = "  Layered tracker (re-read the whole board after a large change):"


def tracker_start(board: np.ndarray, level: int) -> None:
    """Start this game's tracker worker and read the first board (no lines for it)."""
    c = ctx()
    if not LAYERED_TRACKER:
        c.tracker_off = "disabled (LAYERED_TRACKER = False)"
        return
    try:
        from . import scene_worker as _sw
        if not _sw.wait_prewarm(LAYERED_TRACKER_PREWARM_TIMEOUT):
            c.tracker_off = "kernels not available: " + str(_sw.prewarm_info().get("error"))[:300]
            return
        cl = _sw.TrackerClient(c.game_id)
        if not cl.alive:
            c.tracker_off = f"worker did not start: {cl.dead_reason}"
            return
        r = cl.result(cl.feed(to_uint8_board(board), int(level)), LAYERED_TRACKER_TIMEOUT + 60)
        if r is None or not r.get("ok"):
            c.tracker_off = f"first board failed: {cl.dead_reason or (r or {}).get('error')}"
            cl.close()
            return
        c.tracker = cl
        c.tracker_scene = r.get("rec")
        scene_reset(board, (r.get("rec") or {}).get("table"))
    except Exception as exc:
        c.tracker = None
        c.tracker_off = f"start failed: {exc!r}"


def _tracker_stop(reason: str) -> None:
    c = ctx()
    if c.tracker is not None:
        try:
            c.tracker.close()
        except Exception:
            pass
        c.tracker_stats_final = c.tracker.summary()
    c.tracker = None
    c.tracker_off = reason


def tracker_feed(board: np.ndarray, level: int) -> dict[str, Any] | None:
    """Explain one new board. Returns {"prev": scene before, "rec": record, "ms": ...} or {"off": reason} (once)
    or None when the tracker is off."""
    c = ctx()
    cl = c.tracker
    if cl is None:
        return None
    prev = c.tracker_scene
    try:
        r = cl.result(cl.feed(to_uint8_board(board), int(level)), LAYERED_TRACKER_TIMEOUT)
    except Exception as exc:
        r = None
        cl.dead_reason = cl.dead_reason or repr(exc)
    if r is None:
        reason = cl.dead_reason or "no answer"
        _tracker_stop(reason)
        return {"off": reason}
    if not r.get("ok"):
        c.tracker_scene = None
        return {"error": r.get("error"), "prev": prev}
    if r.get("skipped"):
        c.tracker_scene = None
        return {"skipped": r.get("skipped"), "prev": prev, "ms": r.get("ms")}
    c.tracker_scene = r.get("rec")
    out = {"prev": prev, "rec": r.get("rec"), "ms": r.get("ms"), "restarted": bool(r.get("restarted")),
           "error": r.get("error")}
    out["table"] = (r.get("rec") or {}).get("table")
    if cl.stats.get("waits_ms", 0.0) / 1000.0 > LAYERED_TRACKER_GAME_BUDGET:
        _tracker_stop(f"used its time budget for this game ({LAYERED_TRACKER_GAME_BUDGET:.0f} s)")
        out["stopped"] = c.tracker_off
    return out


def tracker_lines(res: dict[str, Any] | None, rect: Any, shape: tuple[int, int], before: Any = None,
                  after: Any = None) -> list[str]:
    """Feedback lines (with header) for one tracked board inside a level."""
    if not res:
        return []
    if res.get("off"):
        return [f"  Layered tracker: stopped for the rest of this game ({res['off']})."]
    if res.get("error") and not res.get("rec"):
        return ["  Layered tracker: failed on this board; it restarts from the next one (new #ids from there on)."]
    c = ctx()
    if res.get("skipped"):
        first = not c.tracker_skipping
        c.tracker_skipping = True
        if first:
            return [f"  Layered tracker: skipped this board ({res['skipped']}: too complex for it); it starts again "
                    "with new #ids on the next board it can handle."]
        return []
    c.tracker_skipping = False
    rec = res.get("rec") or {}
    if res.get("restarted"):
        if res.get("error"):
            return ["  Layered tracker: failed on this board and read it afresh (new #ids from here on)."]
        return ["  Layered tracker: starts again on this board (new #ids from here on)."]
    try:
        scored = _sfb.step_lines(res.get("prev"), rec, norm_rects(rect), color_names(), shape, before=before, after=after)
        lines, omitted = _sfb.order_and_cap(scored, LAYERED_TRACKER_LINES, LAYERED_TRACKER_CHARS)
    except Exception as exc:
        return [f"  (layered tracker lines unavailable: {exc!r})"]
    header = TRACKER_HEADER_REREAD if rec.get("fresh") == "reread" else TRACKER_HEADER
    if not lines:
        out = [f"{header} no object change found (the changed cells are inside the marked indicator, or nothing changed)"] \
            if rec.get("wrong0", 0) else []
    elif len(lines) == 1 and not omitted:
        out = [f"{header} {lines[0]}"]
    else:
        out = [header] + [f"    {ln}" for ln in lines]
        if omitted:
            out.append(f"    ... and {omitted} more (changes() in run lists all of them)")
    if res.get("stopped"):
        out.append(f"  Layered tracker: stopped for the rest of this game ({res['stopped']}).")
    return out


def tracker_level_text(res: dict[str, Any] | None) -> str:
    """One paragraph for a level change (the new level's first board as the tracker read it)."""
    if not res or not res.get("rec"):
        return ""
    rec = res["rec"]
    if rec.get("fresh") != "level":
        return ""
    try:
        return "  Layered tracker: " + " ".join(_sfb.level_lines(res.get("prev"), rec, color_names()))
    except Exception:
        return ""


def tracker_restart_text(board: np.ndarray, level: int, rect: Any) -> str:
    """Tracker lines for the board after a game-over reset or a RESET (compared with the board before it)."""
    res = tracker_feed(board, level)
    if not res:
        return ""
    scene_reset(board, res.get("table"))
    lines = tracker_lines(res, rect, tuple(to_uint8_board(board).shape))
    if not lines:
        return ""
    if lines[0].startswith("  Layered tracker"):
        lines[0] = "Layered tracker, the restarted board compared with the board before the restart" + lines[0][len("  Layered tracker"):]
    return "\n".join(ln[2:] if ln.startswith("  ") else ln for ln in lines)


def tracker_summary() -> dict[str, Any]:
    c = ctx()
    out: dict[str, Any] = {"enabled": bool(LAYERED_TRACKER)}
    st = getattr(c, "tracker_stats_final", None)
    if c.tracker is not None:
        st = c.tracker.summary()
    if st:
        out.update(st)
    out["off_reason"] = c.tracker_off
    try:
        from . import scene_worker as _sw
        out["prewarm"] = _sw.prewarm_info()
    except Exception:
        pass
    return out


def end_game() -> dict[str, Any]:
    """Stop this game's tracker worker; returns its statistics."""
    c = ctx()
    summ = tracker_summary()
    if c.tracker is not None:
        try:
            c.tracker.close()
        except Exception:
            pass
        c.tracker_stats_final = summ
        c.tracker = None
    return summ


# ---------------------------------------------------------------- v2.6.5 scene bookkeeping

def _scene_store(board: np.ndarray, table: list | None) -> None:
    c = ctx()
    if table is None:
        return
    key = hashlib.sha1(to_uint8_board(board).tobytes()).digest()
    if key not in c.scene_by_hash:
        c.scene_by_hash_order.append(key)
        if len(c.scene_by_hash_order) > 600:
            old = c.scene_by_hash_order.pop(0)
            c.scene_by_hash.pop(old, None)
    c.scene_by_hash[key] = table


def scene_reset(board: np.ndarray, table: list | None) -> None:
    """A new attempt / level: the scene list restarts with this board."""
    c = ctx()
    c.scenes = [table]
    c.scene_edits = []
    _scene_store(board, table)


def scene_append(board: np.ndarray, table: list | None, edits: dict | None) -> None:
    c = ctx()
    c.scenes.append(table)
    c.scene_edits.append(edits)
    _scene_store(board, table)


def scene_for_board(board: np.ndarray) -> list | None:
    return ctx().scene_by_hash.get(hashlib.sha1(to_uint8_board(board).tobytes()).digest())


def scenes_for_boards(boards: list[np.ndarray]) -> list:
    """Scene tables aligned with the given boards (the attempt's states), by position when consistent, else by hash."""
    c = ctx()
    if len(c.scenes) == len(boards):
        return list(c.scenes)
    return [scene_for_board(b) for b in boards]


def scene_objects(table: list | None, rect: Any, include_background: bool = True) -> list[dict[str, Any]] | None:
    """The model-facing object list of a scene table (adds colour names and the indicator flag)."""
    if table is None:
        return None
    names = color_names()
    rects = norm_rects(rect)
    out = []
    for d in table:
        if not include_background and d.get("is_background"):
            continue
        o = dict(d)
        o["color_name"] = names.get(int(o["color"]), str(o["color"]))
        o["cells"] = [tuple(x) for x in o.get("cells") or ()]
        o["guessed_cells"] = [tuple(x) for x in o.get("guessed_cells") or ()]
        r = o.get("rect")
        o["is_remaining_actions_indicator"] = bool(r and rects and any(
            r[0] >= q[0] and r[1] <= q[1] and r[2] >= q[2] and r[3] <= q[3] for q in rects))
        out.append(o)
    return out


def scene_inventory_text(table: list | None, rect: Any, board: np.ndarray) -> str | None:
    """Level-start inventory from the scene (None when no scene is available)."""
    objs = scene_objects(table, rect)
    if objs is None:
        return None
    try:
        return _sfb.inventory_text(objs, color_names(), tuple(to_uint8_board(board).shape))
    except Exception as exc:
        return f"(scene inventory unavailable: {exc!r})"


_CELLS_ARR_CACHE: dict[int, tuple[Any, np.ndarray]] = {}


def _cells_array(cells: Any) -> np.ndarray:
    """(n, 2) int array of a cells list; cached by the list object (lists of large objects are shared across copies)."""
    key = id(cells)
    hit = _CELLS_ARR_CACHE.get(key)
    if hit is not None and hit[0] is cells:
        return hit[1]
    arr = np.asarray(list(cells), dtype=np.int64).reshape(-1, 2) if len(cells) else np.zeros((0, 2), np.int64)
    if len(_CELLS_ARR_CACHE) > 4096:
        _CELLS_ARR_CACHE.clear()
    _CELLS_ARR_CACHE[key] = (cells, arr)
    return arr


def scene_render(objs: list[dict[str, Any]], shape: tuple[int, int]) -> np.ndarray:
    """Paint scene objects back to front: full shapes (observed + guessed cells), parts in their colours."""
    h, w = int(shape[0]), int(shape[1])
    bg = [o for o in objs if o.get("is_background")]
    fill = int(bg[0]["color"]) if bg else (int(max(objs, key=lambda o: o.get("size", 0))["color"]) if objs else 0)
    board = np.full((h, w), fill, dtype=np.uint8)
    ordered = sorted(objs, key=lambda o: -int(o.get("layer", 0)))     # back (large layer) first
    for o in ordered:
        parts = o.get("parts") or [{"color": o.get("color"), "cells": o.get("cells") or (), "guessed": o.get("guessed_cells") or ()}]
        for p in parts:
            for cells in (p.get("cells") or (), p.get("guessed") or ()):
                if not len(cells):
                    continue
                arr = _cells_array(cells)
                ok = (arr[:, 0] >= 0) & (arr[:, 0] < h) & (arr[:, 1] >= 0) & (arr[:, 1] < w)
                board[arr[ok, 0], arr[ok, 1]] = np.uint8(int(p["color"]) & 255)
    return board


SCENE_COPY_SHARE_ABOVE = 200     # cell lists of objects larger than this are shared between copies (never mutated in place)


def scene_copy(objs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for o in objs:
        d = dict(o)
        big = len(o.get("cells") or ()) > SCENE_COPY_SHARE_ABOVE
        d["cells"] = o.get("cells") if big else list(o.get("cells") or ())
        d["guessed_cells"] = o.get("guessed_cells") if big else list(o.get("guessed_cells") or ())
        d["rect"] = list(o.get("rect") or ())
        d["parts"] = [{"color": p["color"], "cells": (p.get("cells") if big else list(p.get("cells") or ())),
                       "guessed": (p.get("guessed") if big else list(p.get("guessed") or ()))}
                      for p in (o.get("parts") or [])]
        out.append(d)
    return out


# ============================================================================ objects

def objects_for(board: np.ndarray, rect: tuple[int, int, int, int] | None) -> list[dict[str, Any]]:
    c = ctx()
    arr = to_uint8_board(board)
    key = (hashlib.sha1(arr.tobytes()).digest(), norm_rects(rect))
    hit = c.obj_cache.get(key)
    if hit is not None:
        return hit
    objs = _obj.segment(arr, indicator_rect=rect, color_names=color_names())
    c.obj_cache[key] = objs
    c.obj_cache_order.append(key)
    while len(c.obj_cache_order) > 12:
        old = c.obj_cache_order.pop(0)
        c.obj_cache.pop(old, None)
    return objs


def level_start_objects_text(board: np.ndarray, rect: tuple[int, int, int, int] | None) -> str:
    if MINIMAL_HARNESS:
        return ""
    c = ctx()
    if LAYERED_TRACKER and c.tracker is not None:
        txt = scene_inventory_text(scene_for_board(board), rect, board)
        if txt:
            return txt + "\n" + c.references.text(full=True)
    if c.view_exclude:   # v2.11: the object view set with set_object_view
        objs = _gl.diff_objects(to_uint8_board(board), rect, c.view_exclude, None)
        prev = (_gl.diff_objects(to_uint8_board(c.prev_level_start_board), rect, c.view_exclude, None)
                if c.prev_level_start_board is not None else None)
    else:
        objs = objects_for(board, rect)
        prev = objects_for(c.prev_level_start_board, rect) if c.prev_level_start_board is not None else None
    text = _obj.inventory_text(objs)
    if c.view_exclude:
        text += (f"\n(object view: colours {', '.join(map(str, c.view_exclude))} are treated as gaps, so touching other colours "
                 "form one component; set_object_view(exclude=[]) goes back to single-colour components)")
    if prev is not None:   # v2.10: same indicator rect for both (it was always "gone")
        delta = _obj.inventory_delta_text(prev, objs)
        if delta:
            text += "\n" + delta
    _rr = norm_rects(rect)
    if _rr:   # v2.10
        text += ("\nThe remaining-actions indicator area you marked is kept for this level: "
                 + ", ".join(f"[{r[0]}:{r[1]}, {r[2]}:{r[3]}]" for r in _rr) + " (re-mark it if it moved).")
    return text + "\n" + c.references.text(full=True)


def level_advanced(prev_level_start_board: np.ndarray, new_board: np.ndarray) -> None:
    c = ctx()
    c.action_coverage = ActionCoverage()
    c.animation_archive.clear()
    c.prev_level_start_board = to_uint8_board(prev_level_start_board).copy()
    c.level_start_board = to_uint8_board(new_board).copy()
    c.last_request_board = None  # a new level: no diff image on the next request
    c.attempt_steps = 0
    c.manual_resets = 0
    c.known_noop.clear()
    c.known_death.clear()
    c.last_refused = None
    c.exploration = ExplorationFeedback()


def level_restarted(board: np.ndarray, *, manual: bool = False) -> None:
    c = ctx()
    c.level_start_board = to_uint8_board(board).copy()
    c.last_request_board = None
    c.attempt_steps = 0
    c.manual_resets += int(manual)
    c.known_noop.clear()  # a reset can restore hidden state, even on the same visible board
    c.last_refused = None
    c.exploration.restart()


def restart_feedback(before: np.ndarray | None, after: np.ndarray, rect: Any, *, manual: bool) -> str:
    """Measured reset effects only; never feed this comparison to action-result history."""
    if MINIMAL_HARNESS:
        return ""
    lines = []
    if before is not None and before.shape == after.shape:
        try:
            diff, changes = _gl.change_report(before, after, rect, exclude=ctx().view_exclude, cap=2)
            shapes = [f"color {e['color']}: {e['from_size']} -> {e['to_size']} cells "
                      f"within {rect_text(region)}" for e, region in shape_changes(diff, norm_rects(rect))]
            detail = shapes[:2] if shapes else changes[:2]
            lines.append("Reset effect (caused by RESET, not an ordinary action; component pairing tentative): "
                         + ("; ".join(detail) if detail else
                            f"{diff['changed_outside_indicator']} cells changed outside the marked indicator") + ".")
        except Exception:
            pass  # optional feedback must not interrupt a restart
    if manual and ctx().manual_resets:
        lines.append("Before another manual RESET on this level, state in the same response what you expect to be "
                     "different on the next attempt and which observation supports restarting. "
                     "An unexplained shape change alone does not establish that the attempt cannot succeed.")
    return "\n".join(lines)


# ============================================================================ feedback

def bounded_board_diff(before: np.ndarray, after: np.ndarray) -> str:
    """Changed-region feedback for one action: prints the CSV of the changed rectangles only
    when their total area is small; otherwise lists the rectangles with cell counts."""
    b, a = to_uint8_board(before), to_uint8_board(after)
    rects = compute_changed_rects(b, a)
    if not rects:
        return "(no changes)"
    total_changed = int(np.count_nonzero(b != a))
    total_area = sum((r1 - r0) * (c1 - c0) for r0, r1, c0, c1 in rects)
    parts = [f"{total_changed} cells changed in {len(rects)} region(s):"]
    if total_area <= RECT_CSV_MAX_CELLS:
        for r0, r1, c0, c1 in rects:
            parts.append(f"  Region [{r0}:{r1}, {c0}:{c1}]:")
            for line in format_rect_region(a, r0, r1, c0, c1).split("\n"):
                parts.append(f"    {line}")
    else:
        for r0, r1, c0, c1 in rects[:12]:
            n = int(np.count_nonzero(b[r0:r1, c0:c1] != a[r0:r1, c0:c1]))
            parts.append(f"  Region [{r0}:{r1}, {c0}:{c1}]: {n} cells changed")
        if len(rects) > 12:
            parts.append(f"  ... and {len(rects) - 12} more regions")
        parts.append("  (region contents not printed: too large; see the change report below, the board image, or run())")
    return "\n".join(parts)


def execute_action_wrapped(base: Callable[..., tuple[str, str, int]], env: Any, action: CandidateAction,
                           all_boards: list[np.ndarray], actions_taken: list[str], history: list[Any],
                           current_level: int, seen_colors: set[int],
                           rect: tuple[int, int, int, int] | None, canonical_idx: dict[bytes, int],
                           *, quiet: bool = False) -> tuple[str, str, int]:
    """Wrap agent._execute_action: append object-level feedback and record the step."""
    c = ctx()
    before = to_uint8_board(all_boards[-1]).copy()
    if MINIMAL_HARNESS:
        uncertain = c.after_env_error
        first = c.attempt_steps == 0
        history_before = len(history)
        c.env_step_failed = False
        c.last_frames = []
        c.animation_observed = False
        text, outcome, level_after = base(env, action, all_boards, actions_taken, history, current_level,
                                         seen_colors, rect, canonical_idx, quiet=quiet)
        failed = c.env_step_failed
        c.env_step_failed = False
        after = to_uint8_board(all_boards[-1])
        changed = int(np.count_nonzero(before != after)) if before.shape == after.shape else None
        outside = (int(np.count_nonzero((before != after) & ~rects_mask(rect, before.shape)))
                   if before.shape == after.shape else None)
        transient = any(f.shape == before.shape and np.any(f != before) for f in c.last_frames)
        step = {"action": action.as_text(), "outcome": "env_error" if failed else outcome,
                "first_after_restart": first, "unreliable": uncertain or failed,
                "animation_observed": c.animation_observed,
                "level_before": current_level, "level_after": level_after,
                "changed_cells": changed, "changed_outside_indicator": outside,
                "indicator_rects": norm_rects(rect),
                "anim_frames": len(c.last_frames), "anim_transient_cells": int(transient),
                "hist_index": len(history)-1 if history else None}
        step["animation_index"] = history_before if len(history) > history_before else None
        step["short"] = f"{step['outcome']}; changed_cells={changed}; animation_frames={len(c.last_frames)}"
        if norm_rects(rect):
            step["short"] += f"; outside_marked_region={outside}"
        _add_step_replay_fields(step, after, actions_taken)
        c.steps.append(step)
        c.last_step = step
        c.attempt_steps += not failed
        return (f"Action {action.as_text()}: {step['short']}"
                + (f"; {text}" if failed else "")
                + (f"; level {current_level} -> {level_after}" if level_after != current_level else "")), outcome, level_after
    ip = c.references
    ip.indicator_rects = norm_rects(rect)
    ip_before = ip.before_action(before, current_level, len(actions_taken))
    ip.last_checks, ip.last_report = [], {}
    c.env_step_failed = False
    c.last_frames = []
    c.animation_observed = False
    first_after_restart = c.attempt_steps == 0
    uncertain_transition = c.after_env_error
    text, outcome, level_after = base(env, action, all_boards, actions_taken, history, current_level,
                                      seen_colors, rect, canonical_idx, quiet=quiet)
    if c.env_step_failed:
        ip.expected = []
        # v2.9: env.step() raised or returned nothing: no transition, no no-op / death record, the batch stops (caller)
        c.env_step_failed = False
        step = {"action": action.as_text(), "outcome": "env_error", "level_before": current_level, "level_after": level_after,
                "changed_cells": None, "changed_outside_indicator": None, "diff": None, "changed_rects": None,
                "anim_frames": 0, "anim_transient_cells": None}
        c.steps.append(step)
        c.last_step = step
        return text, outcome, level_after
    c.attempt_steps += 1
    after = to_uint8_board(all_boards[-1])
    ip_event = ip.after_action(ip_before, before, after, action.as_text(), level_after, len(actions_taken),
                               click=(action.row, action.col) if action.row >= 0 else None, reliable=not uncertain_transition,
                               snapshot=ENABLE_REPLAYS)
    if action.requested:
        ip.metrics["reference_clicks"] += 1
        ip_event["requested"] = action.requested
        text += f"\n  Resolved {action.requested} -> {action.as_text()}"
    ip_text = ip.action_text()
    if ip_text:
        text += "\n" + ip_text
    if outcome == "win" or level_after > current_level:
        # v2.4: a level change replaces the board; no object/animation diff across levels
        step = {"action": action.as_text(), "outcome": outcome, "level_before": current_level, "level_after": level_after,
                "changed_cells": None, "changed_outside_indicator": None, "diff": None, "changed_rects": None,
                "anim_frames": len(c.last_frames), "anim_transient_cells": None,
                "first_after_restart": first_after_restart,
                "animation_observed": c.animation_observed,
                "hist_index": None}   # v2.11c: the level's history is cleared after a level change
        step["object_refs"] = ip_event
        if outcome != "win" and c.tracker is not None:
            try:
                _tres = tracker_feed(after, level_after)
                scene_reset(after, (_tres or {}).get("table"))
                _tl = tracker_level_text(_tres)
                if _tl:
                    text = text + "\n" + _tl
                step["tracker_ms"] = (_tres or {}).get("ms")
                if _tl:
                    step["tracker_lines"] = [_tl]
            except Exception as exc:
                step["tracker_error"] = repr(exc)
        _add_step_replay_fields(step, after, actions_taken)
        c.steps.append(step)
        c.last_step = step
        return text, outcome, level_after
    extra_lines: list[str] = []
    diff: dict[str, Any] | None = None
    obj_lines: list[str] = []
    try:
        if before.shape == after.shape:
            # v2.11: the same code as lib.frame_diff() (and the object view, when set)
            diff, obj_lines = _gl.change_report(before, after, rect, exclude=c.view_exclude, cap=OBJ_CHANGES_CAP)
            if norm_rects(rect):
                extra_lines.append(
                    f"  Changed cells: {diff['changed_cells']} total, {diff['changed_outside_indicator']} outside the remaining-actions indicator")
            else:
                extra_lines.append(f"  Changed cells: {diff['changed_cells']}")
            if c.tracker is None:      # v2.6.5: with the tracker alive, its lines replace this block
                if obj_lines:
                    extra_lines.append("  Component changes:")
                    extra_lines.extend(f"    {ln}" for ln in obj_lines)
                elif diff["changed_outside_indicator"] > 0:
                    extra_lines.append("  Component changes: none matched (cells changed inside existing components or background)")
    except Exception as exc:  # feedback must never break the game loop
        extra_lines.append(f"  (component analysis unavailable: {exc!r})")
    trans = None
    try:
        trans = _animation_lines(before, after, rect, diff, extra_lines)
    except Exception as exc:
        extra_lines.append(f"  (animation analysis unavailable: {exc!r})")
    _t_lines: list[str] = []
    _t_ms = None
    if c.tracker is not None:          # v2.6: the layered tracker's view of this action
        try:
            _tres = tracker_feed(after, level_after)
            _t_ms = (_tres or {}).get("ms")
            _t_lines = tracker_lines(_tres, rect, tuple(after.shape), before=before, after=after)
            _all_lines = []
            try:
                if _tres and _tres.get("rec"):
                    _all_lines = [ln for _p, ln in sorted(_sfb.step_lines(_tres.get("prev"), _tres["rec"], norm_rects(rect), color_names(),
                                                                          tuple(after.shape), before=before, after=after), key=lambda t: -t[0])]
            except Exception:
                _all_lines = [ln.strip() for ln in _t_lines]
            scene_append(after, (_tres or {}).get("table"), {"lines": _all_lines,
                                                              "edits": ((_tres or {}).get("rec") or {}).get("edits")})
            if not _t_lines and (_tres is None or _tres.get("off")):
                if obj_lines:
                    extra_lines.append("  Component changes:")
                    extra_lines.extend(f"    {ln}" for ln in obj_lines)
            extra_lines.extend(_t_lines)
        except Exception as exc:
            extra_lines.append(f"  (layered tracker unavailable: {exc!r})")
    text = compress_action_text(text, has_object_lines=bool(obj_lines))
    if EVIDENCE_NOTES and history:   # v2.12b: every step report carries its history index (for citing it in notes)
        _hd = f"Action {action.as_text()}:"
        if text.startswith(_hd):
            text = f"Action {action.as_text()} (history[{len(history) - 1}]):" + text[len(_hd):]
    step = {
        "action": action.as_text(), "outcome": outcome, "level_before": current_level, "level_after": level_after,
        "first_after_restart": first_after_restart,
        "animation_observed": c.animation_observed,
        "unreliable": uncertain_transition,
        "changed_cells": int(np.count_nonzero(before != after)) if before.shape == after.shape else None,
        "changed_outside_indicator": (diff["changed_outside_indicator"] if diff else None),
        "diff": {k: v for k, v in diff.items() if k != "changed_rects"} if diff else None,
        "changed_rects": diff["changed_rects"] if diff else None,
        "anim_frames": len(c.last_frames),
        "anim_transient_cells": (trans or {}).get("transient"),
        "anim_bbox": (trans or {}).get("bbox"),                 # v2.11c: for the animation note of the one-line form
        "hist_index": (len(history) - 1) if history else None,  # v2.11c: this step is history[hist_index] in run
        "short": (_gl.short_summary(diff) if diff else None),   # v2.11: one-line form for long step reports
    }
    if outcome == "continue" and not uncertain_transition and before.shape == after.shape:
        try:
            rects = norm_rects(rect)
            feedback = c.exploration.observe(
                action.as_text(), diff, state_key(before, rect), state_key(after, rect), c.available_actions,
                scope=(current_level, tuple(after.shape), tuple(tuple(r) for r in rects), tuple(c.view_exclude or ())),
                indicator_rects=rects, history_index=step["hist_index"],
            )
            if feedback:
                step["exploration_feedback"] = feedback
                extra_lines.extend("  " + line for line in feedback)
        except Exception as exc:
            step["exploration_feedback_error"] = str(exc)[:200]  # do not stop an otherwise valid action
    if _t_ms is not None or _t_lines:
        step["tracker_ms"] = _t_ms
        step["tracker_lines"] = [ln.strip() for ln in _t_lines]
    step["object_refs"] = ip_event
    _add_step_replay_fields(step, after, actions_taken)
    c.steps.append(step)
    c.last_step = step
    return (text + ("\n" + "\n".join(extra_lines) if extra_lines else "")), outcome, level_after


def _add_step_replay_fields(step: dict[str, Any], after: np.ndarray, actions_taken: list[str]) -> None:
    """v2.5-log: the board after every executed action (and its number in the game) for the replay viewer."""
    if not ENABLE_REPLAYS:
        return
    try:
        step["n"] = len(actions_taken)        # 1-based number of this action among all actions of the game
        step["board"] = board_to_hex_rows(after)
    except Exception:
        pass


def _animation_lines(before: np.ndarray, after: np.ndarray, rect: tuple[int, int, int, int] | None,
                     diff: dict[str, Any] | None, extra_lines: list[str]) -> dict[str, Any] | None:
    """Feedback for an animated action: how many intermediate frames, which cells changed during the
    animation and changed back (invisible in before/after), and - when the board ended unchanged but
    the animation did not (rejected move, failed check) - the first animation steps."""
    frames = ctx().last_frames
    if not frames or before.shape != after.shape:
        return None
    t = _fd.transient_summary(before, frames, after, indicator_rect=rect)
    if t is None:
        return None
    if not t["transient"]:
        extra_lines.append(f"  Animation: {len(frames)} intermediate frame(s); nothing changed and changed back.")
        return t
    names = color_names()
    cols = ", ".join(f"{a} {names.get(a, a)}->{b} {names.get(b, b)} x{n}" for a, b, n in t["transitions"])
    extra_lines.append(
        f"  Animation: {len(frames)} intermediate frame(s); {t['transient']} cell(s) changed during it and changed back "
        f"before the end (not visible in the before/after boards) around {_obj.rect_text(t['bbox'])} ({cols}). "
        "lib.anim_diff() in run lists the animation step by step.")
    lasting = diff["changed_outside_indicator"] if diff else int(np.count_nonzero(before != after))
    if lasting == 0:
        extra_lines.append(("  Apart from the remaining-actions indicator, the board" if norm_rects(rect) else "  The board")
                           + " ended in the same state as before the action, but an animation played during the action. "
                           "Animation steps:")
        _view = ctx().view_exclude
        steps = _fd.anim_steps(before, frames, after, lambda b: _gl.diff_objects(b, rect, _view, None),
                               indicator_rect=rect, cap=ANIM_AUTO_DIFF_LINES, max_boards=80)
        extra_lines.extend(f"    {ln}" for ln in steps)
    return t


_INDICATOR_BLOCK_RE = re.compile(
    r"  Also, (\d+) cell\(s\) changed in the area you currently have marked as the remaining-actions indicator ((?:\[[^\]]*\](?:, )?)+)"
    r"(?::\n((?:    \[\d+,\d+\]: \d+ -> \d+\n?)+)|\.)"
)
_COLOR_BLOCK_RE = re.compile(r"\n  Color count changes \(value,hex,name,-away,\+into\):(?:\n    Color [^\n]*)+")


def compress_action_text(text: str, *, has_object_lines: bool) -> str:
    """Shorten the per-action boilerplate: one line for the indicator tick, and drop the
    colour-count table when object changes already describe the change."""
    def _ind(m: "re.Match[str]") -> str:
        n, rect, cells = m.group(1), m.group(2), m.group(3)
        detail = ""
        if cells:
            items = [ln.strip() for ln in cells.strip().splitlines()]
            detail = " (" + "; ".join(items[:3]) + (f"; +{len(items) - 3} more" if len(items) > 3 else "") + ")"
        tail = "\n" if m.group(0).endswith("\n") else ""
        return f"  Remaining-actions indicator {rect}: {n} cell(s) changed{detail}{tail}"
    text = _INDICATOR_BLOCK_RE.sub(_ind, text)
    if has_object_lines:
        text = _COLOR_BLOCK_RE.sub("", text)
    return text


def record_frames(before: np.ndarray, frames: list[Any], *, history_index: int | None = None) -> None:
    """Called from _execute_action with the raw engine frames of the action."""
    c = ctx()
    if MINIMAL_HARNESS and history_index is not None:
        # Preserve all intermediate frames, even beyond anim_frames' per-action
        # cap. The archive bounds memory and explicitly refuses oversized records.
        c.animation_archive.record(history_index, frames[:-1])
    c.animation_observed = len(frames) > 1
    kept: list[np.ndarray] = []
    try:
        for f in frames[:-1][:ANIM_MAX_FRAMES]:
            arr = to_uint8_board(np.asarray(f))
            if arr.shape == to_uint8_board(before).shape:
                kept.append(arr)
    except Exception:
        kept = []
    c.last_frames = kept


def last_step_is_noop() -> bool:
    """Final-board equality for pausing a batch, NOT evidence for blocking retries."""
    s = ctx().last_step
    if not s:
        return False
    v = s.get("changed_outside_indicator")
    if v is None:
        v = s.get("changed_cells")
    return v == 0


def unchanged_board_text(action_text: str) -> str:
    """Explain a batch pause without treating final-board equality as no reaction."""
    step = ctx().last_step or {}
    text = (f"{action_text} ended on the same board outside your marked resource/counter region"
            if MINIMAL_HARNESS and step.get("indicator_rects") else
            f"{action_text} ended on the same board" if MINIMAL_HARNESS else
            f"{action_text} ended on the same board outside the remaining-actions indicator")
    if step.get("animation_observed") or step.get("anim_frames"):
        text += "; intermediate frames were returned, so inspect the reaction"
    return text


def log_guard(kind: str, **data: Any) -> None:
    ctx().guards.append({"kind": kind, **data})


def state_key(board: np.ndarray, rect: Any) -> bytes:
    """Board identity with selected cells blanked; death guards explicitly pass no mask."""
    arr = to_uint8_board(board).copy()
    arr[rects_mask(rect, arr.shape)] = np.uint8(255)
    return hashlib.sha1(arr.tobytes()).digest()


def known_guard(board: np.ndarray, rect: Any, action_text: str, *, repeat: bool) -> str | None:
    """Return a refusal reason if `action_text` is a known no-op or a known fatal action in this exact board
    state (indicator ignored), unless the caller asked to repeat it deliberately. Records the refusal."""
    c = ctx()
    key = (state_key(board, rect), action_text)
    dkey = (state_key(board, None), action_text)   # v2.10: deaths compare the full board, remaining-actions indicator included
    if dkey in c.known_death:
        if repeat:
            return None
        c.last_refused = key
        return (f"not executed (no action used): last time, {action_text} from this exact board (remaining-actions "
                "indicator included) ended in a game over. To do it anyway, call act([...], repeat=True).")
    if c.attempt_steps > 0 and key in c.known_noop:
        if repeat:
            return None
        c.last_refused = key
        return ((f"known no-op: {action_text} previously produced no visible change "
                  + ("outside your marked resource/counter region " if norm_rects(rect) else "")
                 if MINIMAL_HARNESS else
                 f"known no-op: {action_text} previously produced no visible change outside the remaining-actions indicator ")
                + "and no animation from this board (not executed, no action used). Hidden state may differ; "
                "to test it again deliberately, call act([...], repeat=True).")
    return None


def record_known(board_before: np.ndarray, rect: Any, action_text: str, *, noop: bool, death: bool) -> None:
    c = ctx()
    key = (state_key(board_before, rect), action_text)
    step = c.last_step or {}
    if step.get("action") == action_text and step.get("outcome") == "continue":
        # A first press can be swallowed by a start/hint gate. Any returned
        # animation also prevents a lasting no-op inference, even if it ends on
        # identical pixels (or its frames were truncated/unreadable).
        quiet_noop = (noop and not step.get("first_after_restart", True)
                      and not step.get("unreliable") and not step.get("animation_observed") and not step.get("anim_frames"))
        if quiet_noop and normalize_label((action_text.split() or [""])[0]) not in ("UNDO", "RESET"):
            c.known_noop.add(key)
        else:
            c.known_noop.discard(key)  # retract stale evidence after a deliberate retry reacts
    if death:
        c.known_death.add((state_key(board_before, None), action_text))


# ============================================================================ run / act

_MODULE_SOURCES: dict[str, str] = {}


def _module_sources() -> dict[str, str]:
    if not _MODULE_SOURCES:
        for name in ("objects", "frame_diff", "gridlib", "modeljobs", "reference_checks", "minimal_memory"):
            _MODULE_SOURCES[name] = (_PKG_DIR / f"{name}.py").read_text()
    return _MODULE_SOURCES


def normalize_label(raw: str) -> str:
    label = str(raw).strip().upper()
    return ACTION_LABEL_ALIASES.get(label, label)


def click_rejection_text(board: np.ndarray, r: int, c: int, claimed: int) -> str:
    """Why a CLICK with a wrong value was refused, with the nearest cells that do have that value."""
    names = color_names()
    arr = to_uint8_board(board)
    actual = int(arr[r, c])

    def _nm(v: int) -> str:
        return names.get(int(v), "not a colour value") if 0 <= int(v) <= 15 else "not a colour value"
    msg = (f"Rejected by the harness before reaching the game (no action used): board[{r},{c}] is {actual} ({_nm(actual)}), "
           f"not {claimed} ({_nm(claimed)}).")
    cells = np.argwhere(arr == claimed) if 0 <= claimed <= 255 else np.zeros((0, 2), dtype=int)
    if len(cells):
        dist = np.abs(cells[:, 0] - r) + np.abs(cells[:, 1] - c)
        order = np.argsort(dist, kind="stable")[:3]
        near = ", ".join(f"[{int(cells[i, 0])},{int(cells[i, 1])}]" for i in order)
        msg += f" Nearest cells with value {claimed}: {near}."
    else:
        msg += f" No cell currently has value {claimed}."
    return msg + " Resend with the exact value of the cell you click."


def parse_action_line(text: str, action_specs: list[Any], current_board: np.ndarray) -> tuple[CandidateAction | None, str | None]:
    requested = text
    text, error = resolve_named_action(text, current_board)
    if error:
        return None, error
    parts = text.split()
    if not parts:
        return None, "empty action"
    label = normalize_label(parts[0])
    spec = next((s for s in action_specs if s.label == label), None)
    if spec is None:
        return None, f"Unknown action '{parts[0]}'. Available: {[s.label for s in action_specs]}"
    r = c_ = -1
    value = -1
    if not spec.requires_xy and len(parts) > 1:   # v2.10: "RIGHT 3" used to run one RIGHT silently
        return None, (f"'{text}': {label} takes no arguments (to repeat an action, list it several times, "
                      f"e.g. [\"{label}\", \"{label}\", \"{label}\"])")
    if spec.requires_xy and len(parts) < 4:   # v2.10: the colour value is required in act() too
        return None, f"'{text}': {label} needs row col value, where value is the current colour of that cell"
    if spec.requires_xy and len(parts) > 4:
        return None, f"'{text}': {label} takes exactly row col value"
    if spec.requires_xy:
        if len(parts) < 3:
            return None, f"{label} requires: {label} row col [value]"
        try:
            r, c_ = int(parts[1]), int(parts[2])
        except ValueError:
            return None, f"Invalid coordinates for {label}."
        h, w = current_board.shape[:2]
        if not (0 <= r < h and 0 <= c_ < w):
            return None, f"Action rejected: ({r},{c_}) is outside the board"
        actual = int(current_board[r, c_])
        if len(parts) >= 4:
            try:
                value = int(parts[3])
            except ValueError:
                return None, f"Invalid value '{parts[3]}' for {label}."
            if value != actual:
                return None, click_rejection_text(current_board, r, c_, value)
        else:
            value = actual
    return CandidateAction(spec.action_id, spec.label, row=r, col=c_, value=value,
                           requested=requested if requested != text else None), None


def run_code(code_file: Any, code: str, states: np.ndarray, actions_history: list[str],
             rect: tuple[int, int, int, int] | None, act_callback: Callable[[list[str], bool], dict[str, Any]] | None,
             history_summary: list[dict[str, Any]], past_levels: list[dict[str, Any]] | None = None,
             current_attempt: int = 1, store: dict[str, Any] | None = None,
             action_labels: list[str] | None = None, indicator_callback: Callable | None = None
             ) -> tuple[str, str | None, dict[str, Any]]:
    c = ctx()
    safe, hits = is_safe(code)
    if not safe:
        return "", safety_error(hits, 3), {"act_calls": 0, "actions_executed": 0}
    c.snippet_stale = None
    c.references.indicator_rects = norm_rects(rect)
    c.new_color_latch = None
    _scene_on = bool(LAYERED_TRACKER and c.tracker is not None)
    _boards = [states[i] for i in range(len(states))] if _scene_on else []
    try:
        return _run_code_worker(code_file, code, states, actions_history, rect, act_callback,
                                history_summary, past_levels, current_attempt, store, action_labels, c, _scene_on, _boards,
                                indicator_callback)
    except Exception as exc:
        if not MINIMAL_HARNESS:
            raise
        return "", f"Sandbox failed: {exc!r}. Executed actions are reported separately.", {}


def _run_code_worker(code_file, code, states, actions_history, rect, act_callback, history_summary,
                     past_levels, current_attempt, store, action_labels, c, _scene_on, _boards, indicator_callback=None):
    out, err, info = _act.run_interactive(
        source=code_file.source or "", code=code, states=states, actions_history=actions_history,
        module_sources=_module_sources(), act_callback=act_callback, indicator_rect=rect,
        color_names=color_names(), anim_frames=c.last_frames, history=history_summary,
        past_levels=past_levels, current_attempt=current_attempt, max_actions=MAX_ACT_ACTIONS, store=store,
        scenes=(scenes_for_boards(_boards) if _scene_on else None),
        scene_edits=(list(c.scene_edits) if _scene_on else None), scene_on=_scene_on,
        full_array_print=FULL_ARRAY_PRINT, store_max_mb=STORE_MAX_MB,
        view=c.view_exclude, action_labels=action_labels,
        object_refs=(None if MINIMAL_HARNESS else c.references.payload()),
        reference_callback=(None if MINIMAL_HARNESS else reference_request),
        minimal=MINIMAL_HARNESS, memory=c.memory, palette=BOARD_COLOR_MAP_HEX,
        image_callback=(queue_shown_image if MINIMAL_HARNESS else None),
        animation_callback=(c.animation_archive.get if MINIMAL_HARNESS else None),
        indicator_callback=indicator_callback,
    )
    processed, had_csv = _postprocess_numpy_to_csv(out)
    if had_csv:
        processed = "(2D numpy array outputs below were auto-converted to CSV)\n" + processed
    c.run_calls.append({"code": code, "error": err, "act_calls": info.get("act_calls", 0),
                        "actions_executed": info.get("actions_executed", 0),
                        "animation_requests": info.get("animation_requests", 0),
                        "indicator_requests": info.get("indicator_requests", 0),
                        "compute_seconds": round(float(info.get("compute_seconds", 0.0)), 3)})
    return processed, err, info


def keep_run_functions(code_file: Any, code: str, *, defined_functions=None) -> tuple[list[str], str | None]:
    """Auto-keep the top-level functions a run snippet defined (see sandbox.keep_functions_from_run)."""
    try:
        from .sandbox import MINIMAL_RESERVED_NAMES
        return keep_functions_from_run(code_file, code,
                                       reserved_names=MINIMAL_RESERVED_NAMES if MINIMAL_HARNESS else (),
                                       defined_functions=defined_functions)
    except Exception as exc:  # never break the turn
        return [], f"could not keep functions: {exc!r}"


def act_response(steps: list[dict[str, Any]], boards: list[np.ndarray], *, stopped_early: bool, stop_reason: str | None,
                 terminal: bool, outcome: str | None, latch: str | None, message: str | None,
                 abort: str | None = None) -> dict[str, Any]:
    c = ctx()
    return {
        "abort": abort,
        "object_refs": c.references.payload(),
        "steps": steps,
        "new_scenes": [scene_for_board(b) for b in boards] if c.tracker is not None else [],
        "new_edits": list(c.scene_edits[-len(boards):]) if (c.tracker is not None and boards) else [],
        "new_boards": [(np.ascontiguousarray(b, dtype=np.uint8).tobytes(), tuple(b.shape)) for b in boards],
        "anim_frames": [(np.ascontiguousarray(f, dtype=np.uint8).tobytes(), tuple(f.shape)) for f in ctx().last_frames],
        "stopped_early": stopped_early, "stop_reason": stop_reason, "terminal": terminal,
        "outcome": outcome, "latch": latch, "message": message,
    }


def act_refusal(reason: str) -> dict[str, Any]:
    log_guard("act_refused", reason=reason)
    return {"refused": reason, "steps": [], "new_boards": []}


# ============================================================================ model (v2.11: model code runs in the worker)

def _hist_index_recs(history_recs: list[Any]) -> list[int]:
    """Indices of the transitions the model check compares (UNDO and the step after an environment error are left out)."""
    return [i for i, r in enumerate(history_recs)
            if getattr(r.action, "label", "") != "UNDO" and not getattr(r, "unreliable", False)]


def _inproc_namespace(source: str, states: np.ndarray, actions_history: list[str], rect: Any, store: dict[str, Any] | None,
                      history_summary: list[dict[str, Any]], past_levels: list[dict[str, Any]] | None, view: Any,
                      action_labels: list[str] | None) -> dict[str, Any]:
    """MODEL_CODE_IN_WORKER off: the model's code file loaded in this process (v2.10 behaviour, no time limits)."""
    from .sandbox import SAFE_BUILTINS, preimported_namespace
    st = [states[i] for i in range(len(states))]
    hist = list(history_summary or [])
    ns: dict[str, Any] = {"__builtins__": SAFE_BUILTINS, **preimported_namespace(), "print": lambda *a, **k: None}
    ns["lib"] = _gl.make_lib({"get_states": lambda: st, "get_history": lambda: hist, "get_actions": lambda: list(actions_history),
                              "rects": norm_rects(rect), "view": view, "names": color_names(), "anim_frames": [],
                              "action_labels": list(action_labels or []), "printer": lambda *a, **k: None, "in_model": True})
    exec(compile(source or "", "<codefile>", "exec"), ns)

    def _no_act(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("act() is not available inside next_state")
    ns.update({"states": states, "current_board": st[-1] if st else None, "actions_history": list(actions_history),
               "history": hist, "past_levels": list(past_levels or []), "CODE": source or "", "store": copy.deepcopy(store or {}),
               "act": _no_act, "last_act_result": {}, "anim_frames": [], "indicator_rects": [tuple(r) for r in norm_rects(rect)]})
    return ns


def model_job(job: str, *, source: str, states: np.ndarray, actions_history: list[str], rect: Any,
              store: dict[str, Any] | None, history_summary: list[dict[str, Any]] | None,
              past_levels: list[dict[str, Any]] | None, job_data: dict[str, Any], time_limit: float,
              mem_mb: int = 512, action_labels: list[str] | None = None) -> tuple[Any, str | None]:
    """Run one model job ("check" / "path" / "predict") on the code file. Returns (result, error)."""
    c = ctx()
    if MODEL_CODE_IN_WORKER:
        _out, err, info = _act.run_interactive(
            source=source or "", code="", states=states, actions_history=actions_history, module_sources=_module_sources(),
            act_callback=None, indicator_rect=rect, store=store, color_names=color_names(), history=history_summary,
            past_levels=past_levels, job=job, job_data=job_data, time_limit=time_limit, mem_limit_mb=mem_mb,
            view=c.view_exclude, action_labels=action_labels, full_array_print=FULL_ARRAY_PRINT)
        if err and err.startswith("Runtime error: "):
            err = err[len("Runtime error: "):]
        return info.get("job_result"), err
    try:
        ns = _inproc_namespace(source, states, actions_history, rect, store, list(history_summary or []), past_levels,
                               c.view_exclude, action_labels)
    except Exception as exc:
        return None, f"your code file failed to load: {exc!r}"
    st = [states[i] for i in range(len(states))]
    mask = rects_mask(rect, st[-1].shape) if (norm_rects(rect) and st) else None
    try:
        if job == "check":
            return _mj.check(ns, list(history_summary or []), list(job_data.get("recs") or []), mask,
                             time_limit=float(job_data.get("budget") or 60.0)), None
        if job == "path":
            start = np.frombuffer(job_data["start"], dtype=np.uint8).reshape(job_data["shape"]).copy()
            return _mj.path(ns, start, list(job_data["specs"]), mask, max_states=int(job_data["max_states"]),
                            max_depth=int(job_data["max_depth"]), time_limit=float(job_data["budget"]),
                            click_cells=_gl.click_cells), None
        if job == "predict":
            pairs = [dict(p, before=np.frombuffer(p["before"], dtype=np.uint8).reshape(p["shape"]).copy())
                     for p in job_data.get("pairs") or []]
            return _mj.predict(ns, pairs), None
    except Exception as exc:
        return None, repr(exc)
    return None, f"unknown job {job!r}"


def code_loader(source: str) -> str | None:
    """VirtualCodeFile loader: load the code file in a fresh worker (top-level code runs there). None = loaded."""
    if MINIMAL_HARNESS:
        from .sandbox import MINIMAL_RESERVED_NAMES, strip_reserved_helpers
        source, _ = strip_reserved_helpers(source or "", MINIMAL_RESERVED_NAMES)
    if not MODEL_CODE_IN_WORKER:
        from .sandbox import SAFE_BUILTINS, preimported_namespace
        namespace: dict[str, Any] = {"__builtins__": SAFE_BUILTINS, **preimported_namespace(), "print": lambda *a, **k: None}
        namespace["lib"] = _gl.make_lib({"get_states": lambda: [], "get_history": lambda: [], "names": color_names(),
                                         "printer": lambda *a, **k: None})
        try:
            exec(source, namespace)
        except Exception as exc:
            return repr(exc)
        return None
    _out, err, info = _act.run_interactive(
        source=source or "", code="", states=np.zeros((1, 64, 64), dtype=np.uint8), actions_history=[],
        module_sources=_module_sources(), act_callback=None, color_names=color_names(), job="load",
        time_limit=CODE_LOAD_TIME_LIMIT, minimal=MINIMAL_HARNESS,
        palette=BOARD_COLOR_MAP_HEX if MINIMAL_HARNESS else None)
    if not err:
        return None
    for pre in ("Runtime error: ", "your code file failed to load: "):
        if err.startswith(pre):
            err = err[len(pre):]
    return err


def _defs_text(source: str, name: str) -> str | None:
    from .sandbox import top_level_functions
    for n, f in reversed(top_level_functions(source or "")):
        if n == name:
            return f
    return None


def _ns_deps_text(source: str) -> str | None:
    """v2.11c: the source of next_state plus every top-level definition it uses (directly or through other top-level
    functions); the last definition of a name wins, as when the file is loaded. A change outside this (e.g. a helper
    kept from run that next_state never calls) does not count as a change to the model."""
    import ast
    try:
        tree = ast.parse(source or "")
    except Exception:
        return source or None
    defs: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defs[node.name] = node
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for tg in targets:
                for nm in ast.walk(tg):
                    if isinstance(nm, ast.Name):
                        defs.setdefault(nm.id, [])
                        if isinstance(defs[nm.id], list):
                            defs[nm.id].append(node)
                        else:
                            defs[nm.id] = [node]
    if "next_state" not in defs:
        return None
    seen: set[str] = set()
    todo = ["next_state"]
    while todo:
        nm = todo.pop()
        if nm in seen or nm not in defs:
            continue
        seen.add(nm)
        nodes = defs[nm] if isinstance(defs[nm], list) else [defs[nm]]
        for nd in nodes:
            for sub in ast.walk(nd):
                if isinstance(sub, ast.Name) and sub.id not in seen and sub.id in defs:
                    todo.append(sub.id)
    parts = []
    for nm in sorted(seen):
        nodes = defs[nm] if isinstance(defs[nm], list) else [defs[nm]]
        for nd in nodes:
            parts.append(ast.get_source_segment(source, nd) or ast.dump(nd))
    return "\n".join(parts)


def _lines_text(lines: list[str], empty: str) -> str:
    return "; ".join(lines) if lines else empty


def _color_pairs(pred: np.ndarray, actual: np.ndarray, mism: np.ndarray) -> str:
    names = color_names()
    from collections import Counter
    cnt = Counter(zip(pred[mism].tolist(), actual[mism].tolist()))
    parts = [f"{names.get(int(a), a)} -> {names.get(int(b), b)} x{n}" for (a, b), n in cnt.most_common(4)]
    if len(cnt) > 4:
        parts.append(f"{len(cnt) - 4} more colour pairs")
    return ", ".join(parts)


def model_check(*, code_file: Any, history_recs: list[Any], hist_summary: list[dict[str, Any]], states: np.ndarray,
                actions_history: list[str], rect: Any, store: dict[str, Any], past_levels: list[dict[str, Any]],
                level: int, report: bool, store_version: int, action_labels: list[str],
                force: bool = False) -> dict[str, Any]:
    """v2.11 model check: replay this level's transitions through next_state (in the worker) and report facts.
    Returns {has_model, ok, lines}. The report is shown when it differs from the last one shown, when next_state (or a
    function it uses) changed (report=True and v2.11c: the change reaches next_state), or when force=True."""
    c = ctx()
    out: dict[str, Any] = {"has_model": False, "ok": False, "lines": []}
    src = code_file.source or ""
    if not code_file.has("next_state"):
        c.validation = None
        c.ns_src, c.ns_level, c.mc_sig = None, None, None
        if report and (code_file.has("board_to_symbolic") or code_file.has("symbolic_to_board")):
            out["lines"].append("[Model] board_to_symbolic/symbolic_to_board found but no next_state(state, action, row, col); "
                                "path stays unavailable until next_state is defined.")
        return out
    out["has_model"] = True
    ns_text = _ns_deps_text(src)   # v2.11c: next_state plus what it uses (was: next_state's own source)
    if ns_text != c.ns_src:
        c.ns_src, c.ns_level = ns_text, level
    elif report and not force:
        report = False   # v2.11c: the code file changed, but not next_state or anything it uses: no forced report
    idx = _hist_index_recs(history_recs)
    if not idx:
        c.validation = {"passed": 0, "total": 0}
        out["ok"] = True
        if report:
            written = f" (written on level {c.ns_level})" if (c.ns_level is not None and c.ns_level < level) else ""
            out["lines"].append(f"[Model check] next_state{written} defined; no transitions on this level to check yet "
                                "(UNDO steps are not checked).")
        c.mc_sig = None
        return out
    key = (hash(src), repr(norm_rects(rect)), int(store_version))
    # newest first: when the time budget runs out, the oldest transitions wait for the next check
    pending = [i for i in reversed(idx) if getattr(history_recs[i], "_mc", (None,))[0] != key]
    job_error = None
    stopped = None
    if pending:
        recs = [{"i": i, "label": history_recs[i].action.label, "row": history_recs[i].action.row,
                 "col": history_recs[i].action.col, "outcome": history_recs[i].outcome} for i in pending]
        res, err = model_job("check", source=src, states=states, actions_history=actions_history, rect=rect, store=store,
                             history_summary=hist_summary, past_levels=past_levels,
                             job_data={"recs": recs, "budget": MODEL_CHECK_TIME_LIMIT, "per_call": 5.0},
                             time_limit=MODEL_CHECK_TIME_LIMIT + 10, action_labels=action_labels)
        if res is None or err:
            job_error = err or "the model check returned nothing"
        elif res.get("mode") == "none":
            job_error = ("next_state in your code file is not a function: define it as "
                         "def next_state(board, action, row=-1, col=-1)")
        else:
            stopped = res.get("stopped")
            for i, r in zip(pending, res.get("results") or []):
                try:
                    history_recs[i]._mc = (key, r)
                except Exception:
                    pass
    if job_error:
        c.validation = {"passed": 0, "total": len(idx), "error": job_error}
        sig = ("job_error", job_error)
        if report or sig != c.mc_sig:
            c.mc_sig = sig
            out["lines"].append(f"[Model check] next_state could not be checked: {job_error}")
        return out
    results = []
    for i in idx:
        _k, _r = getattr(history_recs[i], "_mc", (None, None))
        if _k == key and _r is not None:
            results.append((i, _r))
    unchecked = len(idx) - len(results)
    counts = {"match": 0, "differ": 0, "unknown": 0, "error": 0}
    for _i, r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    out["ok"] = counts["differ"] == 0 and counts["error"] == 0 and unchecked == 0
    c.validation = {"passed": counts["match"], "total": len(idx), "differ": counts["differ"], "unknown": counts["unknown"],
                    "error": counts["error"], "unchecked": unchecked}
    # v2.11c: what the report would say (not which code file produced it: a helper next_state does not use changes nothing)
    sig = (hash(ns_text), key[1], tuple((i, r["status"], r.get("n"), r.get("text"),
                                         hash(r["pred"]) if isinstance(r.get("pred"), (bytes, bytearray)) else None)
                                        for i, r in results), unchecked)
    # v2.11 (after the first test run): a next_state written on an earlier level and not changed on this one is reported
    # once per level; while next_state is being changed on this level, every changed result is reported
    active = c.ns_level == level
    if not (report or (sig != c.mc_sig and (active or c.mc_shown_level != level))):
        return out
    c.mc_sig = sig
    c.mc_shown_level = level
    lines = out["lines"]
    written = f" (written on level {c.ns_level})" if (c.ns_level is not None and c.ns_level < level) else ""
    head = f"[Model check] next_state{written} on the {len(idx)} transition(s) of level {level}: {counts['match']} match"
    if counts["differ"]:
        head += f", {counts['differ']} differ"
    if counts["unknown"]:
        head += f", {counts['unknown']} don't know (returned None)"
    if counts["error"]:
        head += f", {counts['error']} raised an error"
    if unchecked:
        head += f", {unchecked} not checked ({stopped or 'time limit'})"
    head += " (remaining-actions indicator ignored"
    if any(getattr(r.action, "label", "") == "UNDO" for r in history_recs):
        head += "; UNDO steps are not checked"
    head += ")."
    if not active and not report:
        head += " Shown once on this level while next_state is unchanged; shown again after each change to next_state."
    lines.append(head)
    diffs = [(i, r) for i, r in results if r["status"] == "differ"]
    errs = [(i, r) for i, r in results if r["status"] == "error"]
    if diffs:
        i, r = diffs[0]
        rec = history_recs[i]
        before, actual = to_uint8_board(rec.before), to_uint8_board(rec.after)
        lines.append(f"First difference: history[{i}] ({rec.action.as_text()})")
        if r.get("kind") == "cells" and r.get("pred") is not None:
            pred = np.frombuffer(r["pred"], dtype=np.uint8).reshape(actual.shape)
            try:   # v2.12: compare the full lists, cut only for display (a capped "... and N more" line is not a change)
                _d1, l_pred = _gl.change_report(before, pred, rect, exclude=c.view_exclude, cap=None)
                _d2, l_act = _gl.change_report(before, actual, rect, exclude=c.view_exclude, cap=None)
            except Exception as exc:
                l_pred, l_act = [f"(not available: {exc!r})"], []
            only_p = [x for x in l_pred if x not in l_act]
            only_a = [x for x in l_act if x not in l_pred]
            both = [x for x in l_pred if x in l_act]

            def _cut(xs: list[str], k: int = 6) -> str:
                return _lines_text(xs[:k], "nothing") + (f"; and {len(xs) - k} more (lib.frame_diff(...) has all)" if len(xs) > k else "")
            if only_p or only_a:
                lines.append("  predicted, not observed: " + _cut(only_p))
                lines.append("  observed, not predicted: " + _cut(only_a))
            else:
                lines.append("  the object change lines are the same for both; only cells differ")
            if both:
                lines.append(f"  in both: {_lines_text(both[:3], '')}" + (f"; and {len(both) - 3} more" if len(both) > 3 else ""))
            mism = (pred != actual) & ~rects_mask(rect, actual.shape)
            bb = r.get("bbox") or [0, 0, 0, 0]
            lines.append(f"  differing cells: {r.get('n')}, within [{bb[0]}:{bb[1]}, {bb[2]}:{bb[3]}] (predicted -> actual: "
                         f"{_color_pairs(pred, actual, mism)})")
            first_key = (i, hash(r["pred"]))
            if MODEL_CHECK_IMAGE and first_key != c.mc_first:
                c.mc_first = first_key
                c.mc_image = (before.copy(), pred.copy(), actual.copy(), mism)
        else:
            lines.append(f"  {r.get('text')}")
        more = []
        for j, rj in diffs[1:5]:
            rj_rec = history_recs[j]
            if rj.get("kind") == "cells":
                bb = rj.get("bbox") or [0, 0, 0, 0]
                more.append(f"history[{j}] {rj_rec.action.as_text()} ({rj.get('n')} cells within [{bb[0]}:{bb[1]}, {bb[2]}:{bb[3]}])")
            else:
                more.append(f"history[{j}] {rj_rec.action.as_text()} ({rj.get('text')})")
        if more:
            lines.append("Other differences: " + "; ".join(more) + (f"; and {len(diffs) - 5} more" if len(diffs) > 5 else ""))
    if errs:
        shown = [f"history[{i}] {history_recs[i].action.as_text()}: {r.get('text')}" for i, r in errs[:2]]
        lines.append("Errors: " + "; ".join(shown) + (f"; and {len(errs) - 2} more" if len(errs) > 2 else ""))
    if out["ok"]:
        lines.append("This covers only the transitions you have observed: whatever next_state assumes about situations never "
                     "observed (collisions, blocking, win conditions) is still an untested guess.")
    return out


def path_search(*, code_file: Any, start: np.ndarray, specs: list[Any], rect: Any, states: np.ndarray,
                actions_history: list[str], store: dict[str, Any], hist_summary: list[dict[str, Any]],
                past_levels: list[dict[str, Any]], time_limit: float, action_labels: list[str]) -> dict[str, Any]:
    """v2.11: the BFS over next_state runs in the worker; returns the path as CandidateActions plus the predicted board
    (or None) and outcome of every step."""
    st = to_uint8_board(start)
    job_data = {"start": st.tobytes(), "shape": tuple(st.shape),
                "specs": [(int(s.action_id), str(s.label), bool(s.requires_xy)) for s in specs],
                "max_states": int(MAX_DISTINCT_STATES), "max_depth": int(MAX_ACTION_DEPTH), "budget": float(time_limit)}
    res, err = model_job("path", source=code_file.source or "", states=states, actions_history=actions_history, rect=rect,
                         store=store, history_summary=hist_summary, past_levels=past_levels, job_data=job_data,
                         time_limit=float(time_limit) + 15.0, mem_mb=PATH_MEM_LIMIT_MB, action_labels=action_labels)
    if res is None or err:
        return {"found": False, "error": err or "the search returned nothing", "path": [], "pred": [], "explored": 0}
    out = dict(res)
    out["path"] = [CandidateAction(int(a), str(l), row=int(r), col=int(cc), value=int(v)) for a, l, r, cc, v in res.get("path") or []]
    out["pred"] = [(np.frombuffer(b, dtype=np.uint8).reshape(st.shape) if b is not None else None, o)
                   for b, o in res.get("pred") or []]
    out.setdefault("explored", 0)
    out.setdefault("error", None)
    return out


def win_prediction(*, code_file: Any, before: np.ndarray, action: Any, states: np.ndarray, actions_history: list[str],
                   rect: Any, store: dict[str, Any], past_levels: list[dict[str, Any]], action_labels: list[str]) -> str | None:
    """v2.10/v2.11: what the model's next_state predicted for the step that actually completed the level."""
    if not code_file.has("next_state") or action is None:
        return None
    b = to_uint8_board(before)
    res, err = model_job("predict", source=code_file.source or "", states=states, actions_history=actions_history,
                         rect=rect, store=store, history_summary=[], past_levels=past_levels,
                         job_data={"pairs": [{"before": b.tobytes(), "shape": tuple(b.shape), "label": action.label,
                                              "row": action.row, "col": action.col}]},
                         time_limit=MODEL_CHECK_TIME_LIMIT, action_labels=action_labels)
    if err or not res or not res.get("results"):
        return f"[Model] For the winning step ({action.as_text()}) your next_state could not be run: {err or 'no result'}."
    r = res["results"][0]
    if r.get("error"):
        return f"[Model] For the winning step ({action.as_text()}) your next_state failed: {r['error']}."
    if r.get("unknown"):
        return f"[Model] For the winning step ({action.as_text()}) your next_state returned None (don't know)."
    out = r.get("outcome")
    if out == "level_complete":
        return f"[Model] For the winning step ({action.as_text()}) your next_state predicted 'level_complete': your win condition matched."
    return (f"[Model] For the winning step ({action.as_text()}) your next_state predicted {out!r}, not 'level_complete': "
            "your win condition did not recognise this win.")


# ============================================================================ v2.11 step reports

def _anim_note(s: dict[str, Any]) -> str:
    """v2.11c: the animation of a step in the one-line form (frames, and the cells that changed and changed back)."""
    n = int(s.get("anim_frames") or 0)
    if n <= 0:
        return ""
    tr = s.get("anim_transient_cells") or 0
    bb = s.get("anim_bbox")
    if tr and bb:
        return f"animation: {n} frame(s), {tr} cell(s) changed and changed back within {_obj.rect_text(bb)}"
    return f"animation: {n} frame(s)"


def step_entry(text: str, action_text: str) -> dict[str, Any]:
    """One executed step for a run/path report (the full text and the one-line form)."""
    s = ctx().last_step or {}
    short = s.get("short")
    oc = s.get("outcome")
    if oc == "env_error":
        short = "the environment returned an error"
    elif s.get("level_after") is not None and s.get("level_before") is not None and s["level_after"] > s["level_before"]:
        short = "level completed"
    elif oc == "win":
        short = "game won"
    else:
        if oc == "game_over":
            short = (short + "; " if short else "") + "GAME OVER"
        note = _anim_note(s)
        if note:   # v2.11c: the one-line form never drops an animation (before the indicator part, which ends the line)
            short = short or ""
            k = short.rfind("; indicator: ")
            short = (f"{short[:k]}; {note}{short[k:]}" if k >= 0 else (f"{short}; {note}" if short else note))
    # v2.11c: a step keeps its full text when the one-line form cannot describe it: the level / attempt ended, the
    # environment failed, cells changed that no object change explains, or the board ended unchanged (apart from the
    # indicator) while an animation changed cells and changed them back (its frames are not available later)
    anim_only = bool((s.get("anim_transient_cells") or 0) > 0 and s.get("changed_outside_indicator") == 0)
    ip = s.get("object_refs") or {}
    if ip:
        named = list((ip.get("report") or {}).get("lines", []))[:3]
        named += [f"@{q['entity']} {q['test']}: {q['status']}" for q in ip.get("expectations", [])]
        if named:
            short = (short + "; " if short else "") + "; ".join(named)
    if s.get("exploration_feedback"):
        short = (short + "; " if short else "") + " ".join(s["exploration_feedback"])
    return {"text": text, "action": action_text, "short": short or "(no summary)", "hist_index": s.get("hist_index"),
            "full_needed": bool(oc in ("env_error", "win", "game_over") or "no object change matched" in (short or "")
                                or (s.get("level_after") or 0) > (s.get("level_before") or 0) or anim_only
                                or (any(
                                    q["status"] != "satisfied" for q in ip.get("expectations", []))))}


_IND_PART_RE = re.compile(r"; indicator: \d+ cell\(s\)$")


def format_step_reports(entries: list[Any], *, prefix: str = "") -> tuple[str, bool]:
    """Full texts when they fit in STEP_REPORT_FULL_CHARS (or when there is only one step), otherwise one line per step
    with identical consecutive steps merged (steps the one-line form cannot describe keep their full text, see
    step_entry). Lines are labelled with the step's index in history. v2.12: steps that differ only in whether the
    indicator changed are merged too ("indicator changed on k of n"). Returns (text, compressed)."""
    full = "\n".join((f"{prefix}{e['text']}" if isinstance(e, dict) else str(e)) for e in entries)
    n_steps = sum(1 for e in entries if isinstance(e, dict))
    if not STEP_REPORT_COMPRESS or n_steps < 2 or len(full) <= STEP_REPORT_FULL_CHARS:
        return full, False
    out: list[str] = []
    groups: list[dict[str, Any]] = []   # {a, b, action, base, shorts, full_entry}

    def _lab(g: dict[str, Any]) -> str:
        a, b = g["a"], g["b"]
        if a is None:
            return "[-]"
        return f"[{a}]" if a == b else f"[{a}-{b}]"

    def flush() -> None:
        for g in groups:
            shorts = g["shorts"]
            cnt = len(shorts)
            if cnt == 1:
                body = shorts[0]
            elif len(set(shorts)) == 1:
                body = shorts[0] + " (each)"
            else:
                k = sum(1 for x in shorts if _IND_PART_RE.search(x))
                body = (g["base"] or "final board unchanged outside the remaining-actions indicator") + \
                    f" (each); indicator changed on {k} of {cnt}"
            out.append(f"  {_lab(g):<9} {g['action']}{(' x' + str(cnt)) if cnt > 1 else ''}: {body}")
            fe = g["full_entry"]
            if fe is not None:
                tx = fe["text"]
                out.append("    " + (tx if len(tx) <= 4000 else tx[:4000] + " ...").replace("\n", "\n    "))
        groups.clear()
    for e in entries:
        if not isinstance(e, dict):
            flush()
            out.append(str(e))
            continue
        hi = e.get("hist_index")
        base = _IND_PART_RE.sub("", e["short"])
        g = groups[-1] if groups else None
        if g is not None and g["action"] == e["action"] and g["base"] == base and g["full_entry"] is None \
                and not e["full_needed"] and hi is not None and g["b"] is not None and hi == g["b"] + 1:
            g["b"] = hi
            g["shorts"].append(e["short"])
            continue
        groups.append({"a": hi, "b": hi, "action": e["action"], "base": base, "shorts": [e["short"]],
                       "full_entry": e if e["full_needed"] else None})
    flush()
    return ("  (long report: one line per step, identical consecutive steps merged; [i] is the step's index in history, "
            "lib.frame_diff(history[i]['before'], history[i]['after']) shows its board change in full; steps the line "
            "cannot describe are shown in full)\n" + "\n".join(out)), True


# ============================================================================ death / images

def record_death(death_board: np.ndarray, level_start_board: np.ndarray,
                 rect: tuple[int, int, int, int] | None) -> None:
    """Keep the board at the game over (attached as an image after the reset). v2.4: no guessed cause."""
    c = ctx()
    c.death_board = to_uint8_board(death_board).copy()
    c.death_pending_image = DEATH_IMAGE
    c.death_text = ""


def win_result_frame(before: np.ndarray, frames: list[Any]) -> np.ndarray | None:
    """v2.9: the board at the winning moment = the last frame of the winning action's own animation. `frames` are the
    intermediate frames (the new level's board excluded); the scan stops before a single frame step that changes more than
    WIN_TRANSITION_FRACTION of the board (a level-transition wipe). Falls back to the first frame that differs."""
    b = to_uint8_board(before)
    prev, chosen, first_diff = b, None, None
    limit = WIN_TRANSITION_FRACTION * b.size
    for f in frames or []:
        try:
            fa = to_uint8_board(np.asarray(f))
        except Exception:
            continue
        if fa.shape != b.shape:
            continue
        if first_diff is None and np.any(fa != b):
            first_diff = fa
        if np.count_nonzero(fa != prev) > limit:
            break
        chosen, prev = fa, fa
    if chosen is None or not np.any(chosen != b):
        chosen = first_diff
    return None if chosen is None else chosen.copy()


def record_win_frame(board: np.ndarray, level: int) -> None:
    """v2.9: keep the winning moment of `level`; attached as an image to the next request when WIN_FRAME_IMAGE is on."""
    c = ctx()
    c.win_board = to_uint8_board(board).copy()
    c.win_level = int(level)
    c.win_pending_image = bool(WIN_FRAME_IMAGE)


def death_text() -> str:
    return ctx().death_text


def reset_message(current_level: int) -> str:
    """The [GAME OVER / LEVEL RESET] notice. Facts only: the harness does not know why the game ended."""
    if MINIMAL_HARNESS:
        return (f"[GAME OVER / LEVEL RESET] Level {current_level} automatically restarted. "
                "Completed levels and Python memory/helpers are kept. The current image shows the restarted board. "
                "states/actions_history now cover the new attempt; history retains this level's previous attempts.\n\n")
    return (
        f"[GAME OVER / LEVEL RESET] Game over. Level {current_level} has restarted from its initial state; "
        "previously completed levels are kept, and your code file, conversation and knowledge are kept.\n"
        "The restarted board (attached image) is the new current state; the board as it was at the game over is also attached. "
        "The game is deterministic: repeating the same sequence leads to the same result.\n"
        "states now covers the new attempt; history and the model check cover every attempt on this level; "
        "the remaining-actions indicator selection is kept.\n\n"
    )


def _rgb_png_data_url(rgb: np.ndarray, scale: int) -> str:
    import base64
    arr = np.asarray(rgb, dtype=np.uint8)
    if scale > 1:
        arr = np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)
    h, w = arr.shape[:2]
    raw = bytearray()
    for row in arr:
        raw.append(0)
        raw.extend(row.tobytes())

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)

    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(bytes(raw), 6)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _palette() -> np.ndarray:
    pal = np.zeros((256, 3), dtype=np.uint8)
    for k, hx in BOARD_COLOR_MAP_HEX.items():
        s = hx.strip().lstrip("#")
        if len(s) >= 6 and 0 <= int(k) < 256:
            pal[int(k)] = (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
    return pal


def diff_image_data_url(before: np.ndarray, after: np.ndarray, scale: int) -> str:
    b, a = to_uint8_board(before), to_uint8_board(after)
    rgb = np.empty((*a.shape, 3), dtype=np.uint8)
    rgb[...] = DIFF_UNCHANGED_RGB
    changed = b != a
    rgb[changed] = _palette()[a[changed]]
    return _rgb_png_data_url(rgb, scale)


def queue_shown_image(value) -> str:
    """Accept each show(image) immediately, so later worker failure cannot lose it."""
    from .minimal_memory import is_image, validate, image_url, MAX_IMAGES
    from PIL import Image
    import io
    if not is_image(value):
        raise ValueError("show(image) requires an image made by render()")
    validate(value)
    if len(value["png"]) < 24:
        raise ValueError("invalid PNG image")
    w, h = struct.unpack(">II", value["png"][16:24])
    if not (1 <= w <= 640 and 1 <= h <= 640):
        raise ValueError("show image dimensions must be at most 640x640")
    with Image.open(io.BytesIO(value["png"])) as im:
        im.verify()
    c = ctx()
    if len(c.shown_images) >= MAX_IMAGES:
        raise ValueError(f"show image limit: {MAX_IMAGES} per response; save images in store and show later")
    label = f"show[{len(c.shown_images) + 1}]: {str(value['label'])[:500]}"
    url = image_url(value)
    c.shown_images.append({"label": label, "data_url": url})
    c.display_pending_images.append((label, url, 1, None))
    return label


def prepare_request_images(current_board: np.ndarray, level_start_pending: bool) -> None:
    """Decide which extra images accompany the next request (read by llm.py)."""
    c = ctx()
    if MINIMAL_HARNESS:
        c.extra_images = list(c.display_pending_images)
        c.display_pending_images.clear()
        return
    c.extra_images = []
    small = board_small_image_scale_for_model(MODEL_NAME)
    cur = to_uint8_board(current_board)
    try:
        if c.death_pending_image and c.death_board is not None:
            c.extra_images.append(("Board at the game over (before the automatic reset)",
                                   board_to_png_data_url(c.death_board, scale=small), small, None))
            c.death_pending_image = False
        if c.win_pending_image and c.win_board is not None:   # v2.9
            c.extra_images.append((f"Winning moment of level {c.win_level} (last frame of the winning action, before the level switched)",
                                   board_to_png_data_url(c.win_board, scale=small), small, None))
            c.win_pending_image = False
        if c.mc_image is not None:   # v2.11: the model check's first difference
            _b, _p, _a, _m = c.mc_image
            c.extra_images.append(("Model check, first difference: the board before the action | what your next_state predicted | "
                                   "what actually happened (differing cells outlined in cyan)",
                                   model_check_image_url(_b, _p, _a, _m, small), small, None))
            c.mc_image = None
        if (DIFF_IMAGE and not level_start_pending and c.last_request_board is not None
                and c.last_request_board.shape == cur.shape and np.any(c.last_request_board != cur)):
            c.extra_images.append(("Changes since your previous turn (changed cells in their new colour, unchanged cells dimmed)",
                                   diff_image_data_url(c.last_request_board, cur, small), small, None))
    except Exception:
        c.extra_images = []
    c.last_request_board = cur.copy()


def model_check_image_url(before: np.ndarray, pred: np.ndarray, actual: np.ndarray, mism: np.ndarray, scale: int) -> str:
    """v2.11: three full boards side by side (before | predicted | actual), the differing cells outlined in cyan."""
    pal = _palette()
    b, p, a = to_uint8_board(before), to_uint8_board(pred), to_uint8_board(actual)
    h, w = a.shape
    s = max(2, int(scale))

    def panel(x: np.ndarray, outline: bool) -> np.ndarray:
        rgb = np.repeat(np.repeat(pal[x], s, axis=0), s, axis=1)
        if outline:
            for r, c_ in np.argwhere(mism).tolist():
                y0, x0 = r * s, c_ * s
                rgb[y0, x0:x0 + s] = (0, 255, 255)
                rgb[y0 + s - 1, x0:x0 + s] = (0, 255, 255)
                rgb[y0:y0 + s, x0] = (0, 255, 255)
                rgb[y0:y0 + s, x0 + s - 1] = (0, 255, 255)
        return rgb
    gap = np.empty((h * s, s, 3), dtype=np.uint8)
    gap[...] = DIFF_UNCHANGED_RGB
    img = np.concatenate([panel(b, False), gap, panel(p, True), gap, panel(a, True)], axis=1)
    return _rgb_png_data_url(img, 1)


def take_extra_images() -> list[tuple[str, str, int, str | None]]:
    c = ctx()
    imgs = c.extra_images
    c.extra_images = []
    return imgs


# ============================================================================ logging

def turn_log_kwargs(current_board: np.ndarray, rect: tuple[int, int, int, int] | None) -> dict[str, Any]:
    c = ctx()
    if not ENABLE_REPLAYS:
        return {}  # scored rerun: replays are not written, keep memory flat
    try:
        objs = _obj.compact_for_log(objects_for(current_board, rect))
    except Exception:
        objs = None
    payload = {
        "objects": objs,
        "steps": c.steps,
        "guards": c.guards,
        "run_calls": c.run_calls,
        "model_mode": c.model_mode,
        "validation": c.validation,
        "prediction_mismatch": c.prediction_mismatch,
        "death_text": c.death_text or None,
        "prompt_variant": PROMPT_VARIANT,
    }
    if MINIMAL_HARNESS:
        from .minimal_memory import present, image_url, NOTES_CHARS
        text, images, _ = present([("notes", c.memory["notes"])], max_chars=NOTES_CHARS)
        payload["minimal_memory"] = {
            "notes": text, "notes_images": [{"label": p, "data_url": image_url(v)} for p, v in images],
            "run_updates": list(c.memory_events), "action_ledger": list(c.action_ledger),
            "shown_images": list(c.shown_images)}
    else:
        payload["object_refs"] = c.references.log()
    return {"v2": payload}

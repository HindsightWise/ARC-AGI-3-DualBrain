from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

import numpy as np
from arcengine import FrameDataRaw, GameAction

from .board import board_from_key, board_key, to_uint8_board
from .config import (
    ACTION_LABEL_BY_ID,
    ACTION_SYMBOL_BY_LABEL,
    RESET_LABEL,
    BFS_PROGRESS_EVERY,
    MAX_ACTION_DEPTH,
    MAX_DISTINCT_STATES,
)


@dataclass(slots=True, frozen=True)
class CandidateAction:
    action_id: int
    label: str
    row: int = -1
    col: int = -1
    value: int = -1
    requested: str | None = None  # named action before resolution; never sent to the environment

    def as_text(self) -> str:
        if self.row == -1:
            return self.label
        if self.value != -1:
            return f"{self.label} {self.row} {self.col} {self.value}"
        return f"{self.label} {self.row} {self.col}"

    def as_symbol(self) -> str:
        sym = ACTION_SYMBOL_BY_LABEL.get(self.label, self.label)
        if self.row == -1:
            return sym
        return f"{sym}({self.row},{self.col})"


@dataclass(slots=True)
class SearchResult:
    found: bool
    path: list[CandidateAction]
    explored_states: int
    hit_state_limit: bool
    error: str | None = None
    notes: str | None = None          # v2.10: actions skipped because next_state failed
    stop_reason: str | None = None    # v2.10: exhausted / states / depth / time


@dataclass(slots=True, frozen=True)
class ActionSpec:
    action_id: int
    label: str
    requires_xy: bool


def action_specs_from_env(env: Any, *, include_reset: bool = True) -> list[ActionSpec]:
    reset_val = None
    reset_attr = getattr(GameAction, "RESET", None)
    if reset_attr is not None:
        try:
            reset_val = int(reset_attr.value)
        except Exception:
            pass

    specs: dict[int, ActionSpec] = {}
    for action in env.action_space:
        try:
            aid = int(action.value)
        except Exception:
            continue
        if reset_val is not None and aid == reset_val:
            continue
        if not (1 <= aid <= 7):
            continue
        requires_xy = aid == 6
        label = ACTION_LABEL_BY_ID.get(aid, f"ACTION{aid}")
        specs[aid] = ActionSpec(action_id=aid, label=label, requires_xy=requires_xy)
    out = [specs[k] for k in sorted(specs)]
    if include_reset:
        # Harness-level restart of the current level (executed with env.reset(), never env.step()).
        out.append(ActionSpec(action_id=0, label=RESET_LABEL, requires_xy=False))
    return out


def search_specs(action_specs: list[ActionSpec]) -> list[ActionSpec]:
    """The actions a BFS / model search may use (RESET excluded)."""
    return [s for s in action_specs if s.label != RESET_LABEL]


def region_click_cells(board: np.ndarray, mask: np.ndarray | None = None) -> list[tuple[int, int]]:
    """v2.10: one representative cell per 4-connected single-colour region (the region's cell nearest its centre),
    regions entirely inside `mask` (the remaining-actions indicator) skipped."""
    arr = to_uint8_board(board)
    h, w = arr.shape[:2]
    flat = arr.ravel().tolist()
    seen = bytearray(h * w)
    out: list[tuple[int, int]] = []
    mflat = mask.ravel().tolist() if mask is not None else None
    for start in range(h * w):
        if seen[start]:
            continue
        col = flat[start]
        stack = [start]
        seen[start] = 1
        cells = []
        while stack:
            i = stack.pop()
            cells.append(i)
            r, c = divmod(i, w)
            for j in ((i - w) if r > 0 else -1, (i + w) if r < h - 1 else -1, (i - 1) if c > 0 else -1, (i + 1) if c < w - 1 else -1):
                if j >= 0 and not seen[j] and flat[j] == col:
                    seen[j] = 1
                    stack.append(j)
        if mflat is not None and all(mflat[i] for i in cells):
            continue
        rr = sum(i // w for i in cells) / len(cells)
        cc = sum(i % w for i in cells) / len(cells)
        best = min(cells, key=lambda i: ((i // w) - rr) ** 2 + ((i % w) - cc) ** 2)
        out.append(divmod(best, w))
    return out


def candidate_actions_for_board(
    board: np.ndarray, action_specs: list[ActionSpec], max_clicks: int = 256,
    click_targets_fn: Callable[..., Any] | None = None, mask: np.ndarray | None = None,
) -> list[CandidateAction]:
    """v2.10: CLICK candidates = CLICK_TARGETS(board) when the model defines it, else one cell per region
    (was: the first 256 non-zero cells). UNDO is never a search candidate (its effect depends on history)."""
    actions: list[CandidateAction] = []
    arr = to_uint8_board(board)
    h, w = arr.shape[:2]
    for spec in action_specs:
        if spec.label in (RESET_LABEL, "UNDO"):
            continue
        if spec.requires_xy:
            cells = None
            if click_targets_fn is not None:
                try:
                    cells = [(int(r), int(c)) for r, c in click_targets_fn(arr.copy())]
                except Exception:
                    cells = None
            if cells is None:
                cells = region_click_cells(arr, mask)
            seen: set[tuple[int, int]] = set()
            for r, c in cells:
                if 0 <= r < h and 0 <= c < w and (r, c) not in seen:
                    seen.add((r, c))
                    actions.append(CandidateAction(spec.action_id, spec.label, row=r, col=c, value=int(arr[r, c])))
        else:
            actions.append(CandidateAction(spec.action_id, spec.label))
    return actions


def bfs_find_path(
    next_state_fn: Callable[..., Any],
    start_board: np.ndarray,
    action_specs: list[ActionSpec],
    max_states: int = MAX_DISTINCT_STATES,
    max_depth: int = MAX_ACTION_DEPTH,
    *,
    mask: np.ndarray | None = None,
    time_limit: float = 60.0,
    click_targets_fn: Callable[..., Any] | None = None,
) -> SearchResult:
    """v2.10: visited states compare boards with the indicator masked; a time limit checked at every expansion;
    an action whose next_state raises (or returns a malformed board) is skipped and reported, not fatal."""
    start_arr = to_uint8_board(start_board)
    board_shape = start_arr.shape[:2]

    def _key(a: np.ndarray) -> bytes:
        if mask is None:
            return a.tobytes()
        b = a.copy()
        b[mask] = 255
        return b.tobytes()

    start_key = _key(start_arr)
    queue: deque[bytes] = deque([start_key])
    boards: dict[bytes, np.ndarray] = {start_key: start_arr}
    depth: dict[bytes, int] = {start_key: 0}
    parent: dict[bytes, bytes | None] = {start_key: None}
    action_to: dict[bytes, CandidateAction] = {}
    errors: dict[str, list[Any]] = {}
    depth_cut = False
    t0 = perf_counter()

    def _notes() -> str | None:
        if not errors:
            return None
        return "; ".join(f"{lbl} skipped {n:,} time(s) because next_state failed ({first})" for lbl, (n, first) in errors.items())

    while queue:
        if perf_counter() - t0 > time_limit:
            return SearchResult(found=False, path=[], explored_states=len(boards), hit_state_limit=True,
                                error=None, notes=_notes(), stop_reason="time")
        cur_key = queue.popleft()
        cur_depth = depth[cur_key]
        cur_board = boards[cur_key]
        if cur_depth >= max_depth:
            depth_cut = True
            continue
        for action in candidate_actions_for_board(cur_board, action_specs, click_targets_fn=click_targets_fn, mask=mask):
            try:
                result = next_state_fn(cur_board.copy(), action.label, action.row, action.col)
                if not isinstance(result, tuple) or len(result) != 2:
                    raise ValueError("next_state must return (board, outcome)")
                next_board, outcome = result
                if not isinstance(outcome, str):
                    raise ValueError(f"the outcome must be a string, got {type(outcome).__name__}")
                if outcome == "level_complete":
                    path = _reconstruct_path(cur_key, parent, action_to) + [action]
                    return SearchResult(found=True, path=path, explored_states=len(boards), hit_state_limit=False,
                                        notes=_notes())
                if outcome == "game_over" or next_board is None:
                    continue
                nb = to_uint8_board(np.asarray(next_board))
                if nb.shape[:2] != board_shape:
                    raise ValueError(f"returned a board of shape {nb.shape}, expected {board_shape}")
            except Exception as exc:
                e = errors.setdefault(action.label, [0, repr(exc)[:160]])
                e[0] += 1
                continue
            nk = _key(nb)
            if nk in boards:
                continue
            boards[nk] = nb
            parent[nk] = cur_key
            action_to[nk] = action
            depth[nk] = cur_depth + 1
            queue.append(nk)
            if len(boards) >= max_states:
                return SearchResult(found=False, path=[], explored_states=len(boards), hit_state_limit=True,
                                    notes=_notes(), stop_reason="states")
    return SearchResult(found=False, path=[], explored_states=len(boards), hit_state_limit=False,
                        notes=_notes(), stop_reason=("depth" if depth_cut else "exhausted"))


bfs_find_path._started = 0.0


def _reconstruct_path(
    state_key: bytes,
    parent: dict[bytes, bytes | None],
    action_to: dict[bytes, CandidateAction],
) -> list[CandidateAction]:
    actions: list[CandidateAction] = []
    cursor = state_key
    while cursor in action_to:
        actions.append(action_to[cursor])
        p = parent.get(cursor)
        if p is None:
            break
        cursor = p
    actions.reverse()
    return actions


def candidate_to_env_call(action: CandidateAction) -> tuple[GameAction, dict[str, int] | None]:
    action_enum = getattr(GameAction, f"ACTION{action.action_id}", None)
    if action_enum is None:
        raise RuntimeError(f"No GameAction enum for ACTION{action.action_id}")
    if action.row == -1:
        return action_enum, {}
    # env expects {"x": col, "y": row} (graphics convention)
    return action_enum, {"x": action.col, "y": action.row}


def get_current_level(env: Any, obs: Any = None) -> int:
    """Get 1-based current level. Tries observation first, falls back to env._game."""
    if obs is not None:
        levels_completed = getattr(obs, "levels_completed", None)
        if isinstance(levels_completed, int):
            return levels_completed + 1
    game = getattr(env, "_game", None)
    if game is not None:
        idx = getattr(game, "level_index", None)
        if isinstance(idx, int):
            return idx + 1
    return 1


def observation_board(obs: FrameDataRaw) -> np.ndarray:
    if obs is None or not obs.frame:
        raise RuntimeError("Observation missing frame data")
    return to_uint8_board(np.asarray(obs.frame[-1]))


@dataclass
class TransitionRecord:
    before: np.ndarray
    action: CandidateAction
    after: np.ndarray
    outcome: str
    animation_frames: int = 0
    animation_bbox: str = ""
    unreliable: bool = False   # v2.10: the step right after an environment error (it may span two actions)


def _to_plain_symbolic(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _to_plain_symbolic(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain_symbolic(v) for v in value]
    if isinstance(value, set):
        converted = [_to_plain_symbolic(v) for v in value]
        return sorted(converted, key=repr)
    return value

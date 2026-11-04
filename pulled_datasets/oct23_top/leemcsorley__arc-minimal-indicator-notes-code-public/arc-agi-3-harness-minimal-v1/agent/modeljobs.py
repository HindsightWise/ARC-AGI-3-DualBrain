"""Model jobs (v2.11): every call of the model's next_state made by the harness goes through these functions - the model
check, the path search and single predictions. They run inside the game's sandbox worker process (or in-process when
MODEL_CODE_IN_WORKER is off). Standalone module: numpy + stdlib only.

next_state(board, action, row=-1, col=-1) returns (next_board_or_None, outcome) or None ("don't know").
"""
from __future__ import annotations

import copy
import time
from collections import deque
from typing import Any, Callable

import numpy as np


def _u8(x: Any) -> np.ndarray:
    a = np.asarray(x)
    if a.ndim != 2:
        raise ValueError(f"a board must be 2-D, got shape {a.shape}")
    if a.dtype != np.uint8:
        if a.size and (a.min() < 0 or a.max() > 255):
            raise ValueError("board values must be 0-255")
        a = a.astype(np.uint8)
    return a


def resolve(ns: dict) -> tuple[Callable[..., Any] | None, str, Any]:
    """(fn(board, label, row, col) -> raw result, mode, symbolic search space or None)."""
    nxt = ns.get("next_state")
    if not callable(nxt):
        return None, "none", None
    b2s, s2b = ns.get("board_to_symbolic"), ns.get("symbolic_to_board")
    if callable(b2s) and callable(s2b):
        def to_state(board: Any) -> Any:
            return b2s(_u8(board).copy())

        def to_board(state: Any) -> np.ndarray:
            return _u8(np.asarray(s2b(copy.deepcopy(state))))

        def fn(board: Any, action: str, row: int = -1, col: int = -1) -> Any:
            res = nxt(copy.deepcopy(to_state(board)), action, row, col)
            if res is None:
                return None
            if not isinstance(res, tuple) or len(res) != 2:
                raise TypeError("next_state(state, action, row, col) must return (state, outcome), or None for \"don't know\"")
            st, out = res
            if st is None or out != "continue":
                return None, out
            return to_board(st), out
        return fn, "symbolic", (to_state, nxt, to_board, copy.deepcopy)
    return nxt, "cells", None


def classify(res: Any, obs_outcome: str, after: np.ndarray, mask: np.ndarray | None) -> dict:
    """Compare one next_state result with the observed transition."""
    if res is None:
        return {"status": "unknown"}
    if not isinstance(res, tuple) or len(res) != 2:
        return {"status": "error", "text": "next_state must return (board, outcome), or None for \"don't know\""}
    pred, outcome = res
    if pred is None and outcome is None:   # v2.12: (None, None) is no prediction, like None and (None, "continue")
        return {"status": "unknown"}
    if not isinstance(outcome, str):
        return {"status": "error", "text": f"the outcome (second value) must be a string such as 'continue', got {type(outcome).__name__}"}
    if obs_outcome in ("level_complete", "game_over", "win"):
        if outcome == obs_outcome or (obs_outcome == "win" and outcome == "level_complete"):
            return {"status": "match"}
        return {"status": "differ", "kind": "outcome", "text": f"observed {obs_outcome}, predicted {outcome!r}"}
    if outcome != "continue":
        return {"status": "differ", "kind": "outcome", "text": f"observed 'continue', predicted {outcome!r}"}
    if pred is None:   # v2.11: (None, "continue") is no prediction: treated like None ("don't know")
        return {"status": "unknown"}
    try:
        p = _u8(pred)
    except Exception as exc:
        return {"status": "error", "text": f"the predicted board could not be read ({exc})"}
    if p.shape != after.shape:
        return {"status": "error", "text": f"predicted a board of shape {p.shape}, the observed one is {after.shape}"}
    mism = p != after
    if mask is not None:
        mism &= ~mask
    n = int(np.count_nonzero(mism))
    if n == 0:
        return {"status": "match"}
    rr, cc = np.nonzero(mism)
    return {"status": "differ", "kind": "cells", "n": n,
            "bbox": [int(rr.min()), int(rr.max()) + 1, int(cc.min()), int(cc.max()) + 1], "pred": p.tobytes()}


def _call(fn: Callable[..., Any], board: np.ndarray, label: str, row: int, col: int) -> Any:
    return fn(board.copy(), label, int(row), int(col))


def check(ns: dict, history: list[dict], recs: list[dict], mask: np.ndarray | None, *, time_limit: float,
          limit_exc: tuple = (), where: Callable[[BaseException], str] = lambda e: "",
          arm: Callable[[float], None] | None = None, per_call: float = 5.0) -> dict:
    """recs: [{i (index into history), label, row, col, outcome}]. Returns {"mode", "results": [per rec], "stopped"}.
    arm(seconds) re-arms the worker's CPU timer before every call, so one next_state call that does not finish within
    per_call seconds is recorded as an error for that transition and the check goes on with the next one."""
    fn, mode, _ = resolve(ns)
    if fn is None:
        return {"mode": "none", "results": []}
    t0 = time.perf_counter()
    out: list[dict] = []
    stopped = None
    for k, rec in enumerate(recs):
        if time.perf_counter() - t0 > time_limit:
            stopped = f"the model check stopped after {k} of {len(recs)} transitions (time limit {time_limit:.0f} s)"
            break
        h = history[int(rec["i"])]
        before, after = _u8(h["before"]), _u8(h["after"])
        try:
            if arm is not None:
                arm(per_call)
            res = _call(fn, before, rec["label"], rec.get("row", -1), rec.get("col", -1))
        except limit_exc:
            if arm is None:
                stopped = (f"next_state did not finish within the time limit on history[{rec['i']}] "
                           f"({rec['label']}); the model check stopped there")
                break
            out.append({"status": "error", "text": f"next_state did not finish within {per_call:.0f} s of compute on this transition"})
            continue
        except RecursionError as exc:
            out.append({"status": "error", "text": f"next_state raised RecursionError{where(exc)}"})
            continue
        except Exception as exc:
            out.append({"status": "error", "text": f"next_state raised {exc!r}"[:400] + where(exc)})
            continue
        try:
            out.append(classify(res, rec["outcome"], after, mask))
        except Exception as exc:
            out.append({"status": "error", "text": f"the prediction could not be compared ({exc!r})"})
    return {"mode": mode, "results": out, "stopped": stopped}


def predict(ns: dict, pairs: list[dict], *, limit_exc: tuple = (), where: Callable[[BaseException], str] = lambda e: "",
            arm: Callable[[float], None] | None = None, per_call: float = 10.0) -> dict:
    """pairs: [{before (array), label, row, col}] -> {"mode", "results": [{"outcome", "board" bytes or None, "error"}]}."""
    fn, mode, _ = resolve(ns)
    if fn is None:
        return {"mode": "none", "results": []}
    out = []
    for p in pairs:
        try:
            if arm is not None:
                arm(per_call)
            res = _call(fn, _u8(p["before"]), p["label"], p.get("row", -1), p.get("col", -1))
        except limit_exc:
            out.append({"error": "did not finish within the time limit"})
            break
        except Exception as exc:
            out.append({"error": f"next_state raised {exc!r}"[:300] + where(exc)})
            continue
        if res is None:
            out.append({"outcome": None, "unknown": True})
            continue
        if not isinstance(res, tuple) or len(res) != 2:
            out.append({"error": "next_state must return (board, outcome), or None"})
            continue
        b, o = res
        if b is None and (o is None or o == "continue"):   # v2.12: no prediction ("don't know")
            out.append({"outcome": None, "unknown": True})
            continue
        try:
            out.append({"outcome": o if isinstance(o, str) else None, "board": (_u8(b).tobytes() if b is not None else None)})
        except Exception as exc:
            out.append({"outcome": o if isinstance(o, str) else None, "board": None, "error": repr(exc)})
    return {"mode": mode, "results": out}


def _candidates(board: np.ndarray, specs: list, click_fn: Callable[..., Any] | None, click_cells: Callable[..., Any],
                mask: np.ndarray | None) -> list[tuple]:
    """(action_id, label, row, col, value) for every search candidate; UNDO and RESET are never candidates."""
    out = []
    h, w = board.shape
    for aid, label, xy in specs:
        if label in ("RESET", "UNDO"):
            continue
        if xy:
            cells = None
            if click_fn is not None:
                try:
                    cells = [(int(r), int(c)) for r, c in click_fn(board.copy())]
                except Exception:
                    cells = None
            if cells is None:
                cells = click_cells(board, mask)
            seen = set()
            for r, c in cells:
                if 0 <= r < h and 0 <= c < w and (r, c) not in seen:
                    seen.add((r, c))
                    out.append((aid, label, r, c, int(board[r, c])))
        else:
            out.append((aid, label, -1, -1, -1))
    return out


def path(ns: dict, start: np.ndarray, specs: list, mask: np.ndarray | None, *, max_states: int, max_depth: int,
         time_limit: float, click_cells: Callable[..., Any], limit_exc: tuple = ()) -> dict:
    """BFS from start to a predicted level_complete. Returns {found, path [(aid, label, row, col, value)], pred
    [(board bytes or None, outcome)], explored, stop_reason, notes, error, mode}."""
    fn, mode, space = resolve(ns)
    if fn is None:
        return {"found": False, "error": "no next_state in the code file", "mode": "none"}
    click_fn = ns.get("CLICK_TARGETS") if callable(ns.get("CLICK_TARGETS")) else None
    start = _u8(start)
    shape = start.shape

    def key(a: np.ndarray) -> bytes:
        if mask is None:
            return a.tobytes()
        b = a.copy()
        b[mask] = 255
        return b.tobytes()
    errors: dict[str, list] = {}
    unknown: dict[str, int] = {}

    def notes() -> str | None:
        # v2.12: counted per call (a CLICK is tried once per click target in every state), so "times", not "states"
        parts = [f"{lbl} skipped {n:,} time(s) because next_state failed ({first})" for lbl, (n, first) in errors.items()]
        parts += [f"{lbl} skipped {n:,} time(s) because next_state returned no prediction (don't know)" for lbl, n in unknown.items()]
        return "; ".join(parts) or None
    if space is not None:
        to_state, step_fn, to_board, copy_state = space
        try:
            start_state = to_state(start)
        except Exception as exc:
            return {"found": False, "error": f"could not build the start state: {exc!r}", "mode": mode}
    else:
        start_state = None
    k0 = key(start)
    q: deque = deque([k0])
    boards = {k0: start}
    states = {k0: start_state}
    depth = {k0: 0}
    parent: dict = {k0: None}
    via: dict = {}
    depth_cut = False
    t0 = time.perf_counter()

    def finish(last_key: bytes, action: tuple) -> dict:
        seq = [action]
        keys = []
        cur = last_key
        while cur in via:
            seq.append(via[cur])
            keys.append(cur)
            cur = parent[cur]
        seq.reverse()
        keys.reverse()
        pred = [(boards[k].tobytes(), "continue") for k in keys] + [(None, "level_complete")]
        return {"found": True, "path": seq, "pred": pred, "explored": len(boards), "stop_reason": None,
                "notes": notes(), "error": None, "mode": mode}
    try:
        while q:
            if time.perf_counter() - t0 > time_limit:
                return {"found": False, "explored": len(boards), "stop_reason": "time", "notes": notes(), "mode": mode}
            ck = q.popleft()
            d = depth[ck]
            if d >= max_depth:
                depth_cut = True
                continue
            cb = boards[ck]
            for cand in _candidates(cb, specs, click_fn, click_cells, mask):
                aid, label, r, c, v = cand
                try:
                    if space is not None:
                        res = step_fn(copy_state(states[ck]), label, r, c)
                    else:
                        res = fn(cb.copy(), label, r, c)
                    if res is None:
                        unknown[label] = unknown.get(label, 0) + 1
                        continue
                    if not isinstance(res, tuple) or len(res) != 2:
                        raise ValueError("next_state must return (board, outcome) or None")
                    nb_raw, outcome = res
                    if nb_raw is None and (outcome is None or outcome == "continue"):   # v2.12: "don't know"
                        unknown[label] = unknown.get(label, 0) + 1
                        continue
                    if not isinstance(outcome, str):
                        raise ValueError(f"the outcome must be a string, got {type(outcome).__name__}")
                    if outcome == "level_complete":
                        return finish(ck, cand)
                    if outcome == "game_over" or nb_raw is None:
                        continue
                    nstate = nb_raw if space is not None else None
                    nb = to_board(nb_raw) if space is not None else _u8(nb_raw)
                    if nb.shape != shape:
                        raise ValueError(f"returned a board of shape {nb.shape}, expected {shape}")
                except limit_exc:
                    raise
                except Exception as exc:
                    e = errors.setdefault(label, [0, repr(exc)[:160]])
                    e[0] += 1
                    continue
                nk = key(nb)
                if nk in boards:
                    continue
                boards[nk] = nb
                states[nk] = nstate
                depth[nk] = d + 1
                parent[nk] = ck
                via[nk] = cand
                q.append(nk)
                if len(boards) >= max_states:
                    return {"found": False, "explored": len(boards), "stop_reason": "states", "notes": notes(), "mode": mode}
    except limit_exc:
        return {"found": False, "explored": len(boards), "stop_reason": "time", "notes": notes(), "mode": mode}
    return {"found": False, "explored": len(boards), "stop_reason": ("depth" if depth_cut else "exhausted"),
            "notes": notes(), "mode": mode}

"""Layered tracker worker: one subprocess per game runs scene_tracker.Tracker on every board of the game.

Protocol (one JSON object per line):
  parent -> worker  {"op": "step", "seq": n, "level": L, "h": 64, "w": 64, "grid": "<hex digit per cell>"}
                    {"op": "quit"}
  worker -> parent  {"seq": n, "ok": true, "ms": wall ms, "cpu_ms": cpu ms, "rec": {...}, "restarted": bool, "error": str}

The worker never stops on a tracker error: it restarts the tracker from that board (ids continue upwards).
Kernels are compiled by numba once and cached on disk (NUMBA_CACHE_DIR); prewarm() fills the cache in a separate
process before the games start, so every worker only loads them.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Any

_PKG_DIR = Path(__file__).resolve().parent
_PKG_NAME = __package__ or "agent"
NUMBA_CACHE_DIR = os.environ.get("ARC_SCENE_NUMBA_CACHE") or os.path.join(tempfile.gettempdir(), "arc_scene_numba_cache")

TURN_NAMES = {1: "turned 90 degrees clockwise", 2: "turned 180 degrees", 3: "turned 90 degrees anticlockwise",
              4: "mirrored left-right", 5: "mirrored across the anti-diagonal", 6: "mirrored top-bottom",
              7: "mirrored across the diagonal"}


# ============================================================================ worker side

def _tpl_info(T) -> tuple:
    """(seen colour counts [[c, n]], seen-cell box relative to the template (y0, y1, x0, x1) or None, n, nSeen)."""
    import numpy as np
    sel = T.seen != 0
    if np.any(sel):
        cols, cnt = np.unique(T.col[sel], return_counts=True)
        ys, xs = T.ys[sel], T.xs[sel]
        box = (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max()))
    else:
        cols, cnt = np.unique(T.col, return_counts=True)
        box = None
    return [[int(c), int(n)] for c, n in zip(cols.tolist(), cnt.tolist())], box, int(T.n), int(T.nSeen)


_INFO_CACHE: dict[int, tuple] = {}


def _info(T) -> tuple:
    key = id(T)
    v = _INFO_CACHE.get(key)
    if v is None or v[0] is not T:
        v = (T, _tpl_info(T))
        if len(_INFO_CACHE) > 20000:
            _INFO_CACHE.clear()
        _INFO_CACHE[key] = v
    return v[1]


_SHAPE_HASH_CACHE: dict[int, tuple] = {}


def _shape_hash(t) -> int:
    """rotation- and mirror-free hash of the shape alone (colours ignored)."""
    from . import scene_kernels as K
    from . import scene_tracker as ST
    import numpy as np
    v = _SHAPE_HASH_CACHE.get(id(t))
    if v is not None and v[0] is t:
        return v[1]
    best = None
    for k in ST.distinct_orients(t):
        T = ST.oriented(t, k)
        h = int(K.tpl_hash(T.th, T.tw, T.idx, np.zeros(T.n, np.int64)))
        if best is None or h < best:
            best = h
    if len(_SHAPE_HASH_CACHE) > 20000:
        _SHAPE_HASH_CACHE.clear()
    _SHAPE_HASH_CACHE[id(t)] = (t, best)
    return best


def _snapshot(tr) -> dict:
    """per object: (template, orientation, observed cell grid indices) before a step"""
    from . import scene_tracker as ST
    snap = {}
    for o in tr.objs:
        T = ST.T_of(o)
        y, x, ins, g = tr.cells_g(o, T)
        sel = ins & (T.seen != 0)
        snap[o.id] = (o.t, o.k, g[sel])
    return snap


def summarise(tr, rec: dict, snap: dict | None = None, like: dict | None = None, seen_desc: dict | None = None,
              full: bool = True) -> dict:
    """What the harness needs from one frame: the edits and the scene after it (full object table when `full`)."""
    import numpy as np
    from . import scene_tracker as ST
    view = tr.view
    h, w = tr.h, tr.w
    HW = h * w
    grid = tr.grid
    objs = {}
    table = []
    vis_of = {r[0]: r[6] for r in rec["objs"]}
    zs = sorted(o.z for o in tr.objs)
    zrank = {z: i for i, z in enumerate(zs)}
    cover_of: dict[int, list] = {}
    by_id = {}
    for o in tr.objs:
        T = ST.T_of(o)
        cols, box, n, n_seen = _info(T)
        vis = int(vis_of.get(o.id, 0))
        b = [o.r + box[0], o.r + box[1] + 1, o.c + box[2], o.c + box[3] + 1] if box else None
        objs[o.id] = {"z": o.z, "vis": vis, "n": n, "seen": n_seen, "cols": cols, "box": b, "k": o.k,
                      "shape": ST.tpl_hash(T)}
        if not full:
            continue
        y, x, ins, g = tr.cells_g(o, T)
        seen = T.seen != 0
        obs = ins & seen
        gue = ins & ~seen
        own_at = view.own[g]
        visible = obs & (own_at == o.id)
        parts = []
        for c in sorted(set(T.col[ins].tolist())):
            m = ins & (T.col == c)
            parts.append({"color": int(c), "cells": np.stack((y[m & seen], x[m & seen]), 1).tolist(),
                          "guessed": np.stack((y[m & ~seen], x[m & ~seen]), 1).tolist()})
        covered_by = [int(i) for i in np.unique(own_at[obs & (own_at != o.id) & (own_at >= 0)]).tolist()]
        for cid in covered_by:
            cover_of.setdefault(cid, []).append(o.id)
        # adjacent: objects shown next to this object's visible cells
        adj: set = set()
        if np.any(visible):
            vy, vx = y[visible], x[visible]
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                yy, xx = vy + dy, vx + dx
                ok = (yy >= 0) & (xx >= 0) & (yy < h) & (xx < w)
                nb = view.own[yy[ok] * w + xx[ok]]
                adj |= {int(i) for i in np.unique(nb[(nb >= 0) & (nb != o.id)]).tolist()}
        oy, ox = y[obs], x[obs]
        fy, fx = y[ins], x[ins]
        rect = [int(oy.min()), int(oy.max()) + 1, int(ox.min()), int(ox.max()) + 1] if oy.shape[0] else None
        full_rect = [int(fy.min()), int(fy.max()) + 1, int(fx.min()), int(fx.max()) + 1] if fy.shape[0] else None
        dom = max(cols, key=lambda cn: cn[1])[0] if cols else int(T.col[0])
        d = {"id": int(o.id), "color": int(dom), "colors": cols, "size": int(obs.sum()), "visible": int(visible.sum()),
             "cells": np.stack((oy, ox), 1).tolist(), "parts": parts,
             "guessed_cells": np.stack((y[gue], x[gue]), 1).tolist(), "rect": rect, "full_rect": full_rect,
             "full_size": int(ins.sum()), "layer": int(zrank[o.z]), "covered_by": covered_by, "adjacent": sorted(adj),
             "hash": int(ST.tpl_hash(T)), "shape_hash": int(_shape_hash(o.t)), "same_kind_as": (like or {}).get(o.id)}
        if 0 < d["size"] <= 100 and rect:
            d["crop"] = "\n".join(",".join(str(int(v)) for v in row) for row in grid[rect[0]:rect[1], rect[2]:rect[3]])
        table.append(d)
        by_id[o.id] = d
    if full:
        for d in table:
            d["covers"] = sorted(cover_of.get(d["id"], []))
            d["is_background"] = False
        back = [d for d in table if d["full_size"] >= 0.2 * HW]
        if back:
            bg = max(back, key=lambda d: (d["layer"], d["full_size"]))
            bg["is_background"] = True
        if seen_desc is not None:
            for d in table:
                seen_desc[d["id"]] = {"colors": d["colors"], "size": d["size"], "rect": d["rect"]}
    # objects with nothing visible: which objects show where their known cells are
    covered = {}
    for o in tr.objs:
        if objs[o.id]["vis"]:
            continue
        T = ST.T_of(o)
        y, x, ins, g = tr.cells_g(o, T)
        sel = ins & (T.seen != 0)
        own = view.own[g[sel]]
        ids = [int(i) for i in np.unique(own[own >= 0]).tolist() if int(i) != o.id]
        covered[o.id] = ids[:6]
    edits = []
    for e in rec["edits"]:
        d = {}
        for k, v in e.items():
            if k == "map" and isinstance(v, dict):
                d[k] = {str(int(a)): int(b) for a, b in v.items()}
            elif isinstance(v, (np.integer,)):
                d[k] = int(v)
            elif isinstance(v, (np.floating,)):
                d[k] = float(v)
            elif isinstance(v, list):
                d[k] = [int(x) if isinstance(x, (np.integer, int)) else x for x in v]
            else:
                d[k] = v
        t = e.get("type")
        if t == "rotate" and e.get("from") and e.get("to"):
            o = tr.index.get(e["id"])
            if o is not None:
                Tf = ST.oriented(o.t, e["from"][2])
                ht = ST.tpl_hash(ST.oriented(o.t, e["to"][2]))
                for kr in range(1, 8):
                    if ST.tpl_hash(ST.oriented(Tf, kr)) == ht:
                        d["turn"] = TURN_NAMES[kr]
                        break
                # exact rotation: the object's shape after the whole step equals its old shape turned
                bt = (snap or {}).get(e["id"])
                if bt is not None:
                    Tnow = ST.T_of(o)
                    Tthen = ST.oriented(bt[0], e["to"][2])
                    d["exact"] = bool(Tnow.n == Tthen.n and np.array_equal(Tnow.idx, Tthen.idx) and np.array_equal(Tnow.col, Tthen.col)
                                      and Tnow.th == Tthen.th and Tnow.tw == Tthen.tw)
                    d["cells_before"], d["cells_after"] = int(bt[0].n), int(o.t.n)
        if t == "disappear" and snap and e.get("id") in snap:
            g_old = snap[e["id"]][2]
            if g_old.shape[0]:
                own = view.own[g_old]
                u, c = np.unique(own, return_counts=True)
                d["now_shows"] = [int(u[i]) for i in np.argsort(-c)[:3]]
        if t == "appear" and e.get("how") == "kind" and like is not None and e.get("kindOf") is not None:
            like[e["id"]] = int(e["kindOf"])
            if seen_desc is not None and int(e["kindOf"]) in seen_desc:
                d["like_desc"] = seen_desc[int(e["kindOf"])]
        edits.append(d)
    out = {"fresh": rec.get("fresh"), "exact": bool(rec.get("exact")), "level": rec.get("level"),
           "wrong0": int(rec.get("wrong0") or 0), "cost": rec.get("cost"), "edits": edits,
           "objs": {str(k): v for k, v in objs.items()}, "covered": {str(k): v for k, v in covered.items()},
           "table": table}
    if rec.get("alt"):
        out["alt"] = rec["alt"]
    return out


class _SoftTimeout(Exception):
    pass


def _on_alarm(signum, frame):   # noqa: ARG001
    raise _SoftTimeout()


def main() -> None:
    os.environ.setdefault("NUMBA_CACHE_DIR", NUMBA_CACHE_DIR)
    import signal
    import numpy as np
    from . import scene_kernels as K
    from . import scene_tracker as ST
    soft_limit = float(os.environ.get("ARC_SCENE_SOFT_LIMIT", "15"))     # s per board, then this board is skipped
    max_pieces = int(os.environ.get("ARC_SCENE_MAX_PIECES", "1200"))     # boards with more separate pieces are skipped
    fault_at = int(os.environ.get("ARC_SCENE_FAULT_AT", "0") or 0)        # test hooks
    hang_at = int(os.environ.get("ARC_SCENE_HANG_AT", "0") or 0)
    try:
        signal.signal(signal.SIGALRM, _on_alarm)
        has_alarm = True
    except Exception:
        has_alarm = False
    tracker = None
    next_first = 1
    too_complex_at = None          # piece count of a board that took too long
    like: dict = {}                # id -> id of the earlier object it has the same shape and colours as
    seen_desc: dict = {}           # id -> short description, for the "like #n" lines
    out = sys.stdout
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception:
            continue
        if msg.get("op") == "quit":
            break
        if msg.get("op") != "step":
            continue
        t0 = time.perf_counter()
        c0 = time.process_time()
        seq = msg.get("seq")
        resp: dict[str, Any] = {"seq": seq, "ok": False, "restarted": False}
        try:
            h, w = int(msg["h"]), int(msg["w"])
            flat = np.frombuffer(bytes.fromhex("".join("0" + ch for ch in msg["grid"])), dtype=np.uint8)
            grid = flat.reshape(h, w).astype(np.int64)
            lv = msg.get("level")
            _cid, pieces = K.components(np.ascontiguousarray(grid).ravel(), h, w)
            limit = max_pieces if too_complex_at is None else min(max_pieces, max(50, int(too_complex_at * 0.7)))
            if pieces > limit:
                if tracker is not None:
                    next_first = max(next_first, tracker.next_id)
                tracker = None
                resp.update(ok=True, skipped=f"{int(pieces)} separate pieces on the board", rec=None)
            else:
                if has_alarm and soft_limit > 0:
                    signal.setitimer(signal.ITIMER_REAL, soft_limit)
                try:
                    if tracker is not None and (tracker.h != h or tracker.w != w):
                        next_first = tracker.next_id
                        tracker = None
                    fresh_tracker = tracker is None
                    snap = _snapshot(tracker) if tracker is not None else {}
                    try:
                        if tracker is None:
                            tracker = ST.Tracker(h, w, first_id=next_first)
                        if fault_at and seq == fault_at:
                            raise RuntimeError("injected fault (test)")
                        if hang_at and seq == hang_at:
                            time.sleep(10 ** 4)
                        rec = tracker.step(grid, lv, True)
                    except _SoftTimeout:
                        raise
                    except Exception as exc:
                        resp["error"] = f"{type(exc).__name__}: {exc}"
                        resp["trace"] = traceback.format_exc()[-1500:]
                        next_first = max(next_first, tracker.next_id if tracker is not None else next_first)
                        tracker = ST.Tracker(h, w, first_id=next_first)
                        fresh_tracker = True
                        snap = {}
                        rec = tracker.step(grid, lv, True)
                    resp["rec"] = summarise(tracker, rec, snap, like, seen_desc, full=bool(msg.get("full", True)))
                    if len(seen_desc) > 5000:
                        for k in list(seen_desc)[:2500]:
                            del seen_desc[k]
                    resp["restarted"] = bool(fresh_tracker and seq != 1)
                    resp["ok"] = True
                except _SoftTimeout:
                    too_complex_at = int(pieces)
                    if tracker is not None:
                        next_first = max(next_first, tracker.next_id)
                    tracker = None
                    resp.update(ok=True, skipped=f"over {soft_limit:.0f} s on this board ({int(pieces)} pieces)", rec=None)
                finally:
                    if has_alarm:
                        signal.setitimer(signal.ITIMER_REAL, 0)
        except Exception as exc:   # even the restart failed: drop the tracker, try again from the next board
            resp["error"] = f"{type(exc).__name__}: {exc}"
            resp["trace"] = traceback.format_exc()[-1500:]
            if tracker is not None:
                next_first = max(next_first, tracker.next_id)
            tracker = None
        resp["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        resp["cpu_ms"] = round((time.process_time() - c0) * 1000, 1)
        out.write(json.dumps(resp, separators=(",", ":")) + "\n")
        out.flush()


def _warm() -> None:
    """Compile (or load) every kernel by tracking a few synthetic boards (moves, a level change, a big change)."""
    os.environ.setdefault("NUMBA_CACHE_DIR", NUMBA_CACHE_DIR)
    import numpy as np
    from . import scene_kernels as K
    from . import scene_tracker as ST
    if not K.HAVE_NUMBA and os.environ.get("ARC_SCENE_ALLOW_NOJIT") != "1":
        print("numba is not available: the layered tracker would be far too slow in plain Python", flush=True)
        sys.exit(3)
    rng = np.random.default_rng(0)
    g = np.zeros((64, 64), np.int64)
    g[10:20, 10:30] = 3
    g[12:15, 12:15] = 9
    g[40:44, 40:41] = 12
    g[50:60, 5:15] = 8
    g[52:58, 7:13] = 0
    frames = [g.copy()]
    for k in range(1, 6):
        f = frames[-1].copy()
        f[12:15, 12 + k - 1:15 + k - 1] = 3
        f[12:15, 12 + k:15 + k] = 9
        f[40:44, 40:41] = 0
        f[40 + k:44 + k, 40:41] = 12
        frames.append(f)
    f = frames[-1].copy()
    f[30:34, 30:34] = 14
    frames.append(f)
    big = rng.integers(0, 4, (64, 64))
    frames.append(np.repeat(np.repeat(big[:16, :16], 4, 0), 4, 1))
    tr = ST.Tracker(64, 64)
    for i, f in enumerate(frames):
        tr.step(f, 1 if i < 7 else 2, True)
    shifted = np.roll(frames[-1], 3, axis=1)
    tr.step(shifted, 2, True)
    tr.step(frames[0], 2, True)


# ============================================================================ parent side

_prewarm_lock = threading.Lock()
_prewarm_state: dict[str, Any] = {"proc": None, "done": False, "ok": False, "seconds": None, "error": None, "t0": None}


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    env["NUMBA_CACHE_DIR"] = NUMBA_CACHE_DIR
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        env[k] = "1"
    env["PYTHONPATH"] = str(_PKG_DIR.parent) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def _launcher(entry: str) -> list[str]:
    code = (f"import sys; sys.path.insert(0, {str(_PKG_DIR.parent)!r}); "
            f"import importlib; importlib.import_module({(_PKG_NAME + '.scene_worker')!r}).{entry}()")
    return [sys.executable, "-u", "-c", code]


def prewarm_async() -> None:
    """Start compiling / caching the kernels in a background process (once per Python process)."""
    with _prewarm_lock:
        if _prewarm_state["proc"] is not None or _prewarm_state["done"]:
            return
        os.makedirs(NUMBA_CACHE_DIR, exist_ok=True)
        _prewarm_state["t0"] = time.time()
        try:
            _prewarm_state["proc"] = subprocess.Popen(_launcher("_warm"), stdin=subprocess.DEVNULL,
                                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=_child_env())
        except Exception as exc:
            _prewarm_state.update(done=True, ok=False, error=repr(exc))


def wait_prewarm(timeout: float = 600.0) -> bool:
    """Block until the kernels are cached (True) or the warm-up failed / timed out (False)."""
    prewarm_async()
    with _prewarm_lock:
        if _prewarm_state["done"]:
            return bool(_prewarm_state["ok"])
        p = _prewarm_state["proc"]
        try:
            outp, _ = p.communicate(timeout=timeout)
            ok = p.returncode == 0
            err = None if ok else (outp or b"").decode("utf-8", "replace")[-2000:]
        except subprocess.TimeoutExpired:
            p.kill()
            ok, err = False, f"warm-up took over {timeout:.0f}s"
        except Exception as exc:
            ok, err = False, repr(exc)
        _prewarm_state.update(done=True, ok=ok, error=err, seconds=round(time.time() - (_prewarm_state["t0"] or time.time()), 1))
        print(f"[layered tracker] kernels {'ready' if ok else 'NOT ready'} after {_prewarm_state['seconds']}s"
              + ("" if ok else f": {err}"), flush=True)
        return ok


def prewarm_info() -> dict[str, Any]:
    return {k: v for k, v in _prewarm_state.items() if k in ("done", "ok", "seconds", "error")}


class TrackerClient:
    """The game's tracker worker: feed(board, level) sends a board; result(seq) waits for its record."""

    def __init__(self, label: str = "") -> None:
        self.label = label
        self.proc: subprocess.Popen | None = None
        self.q: queue.Queue = queue.Queue()
        self.seq = 0
        self.alive = False
        self.dead_reason: str | None = None
        self.last: dict[int, dict] = {}            # seq -> response not yet collected
        self.scene: dict[str, Any] | None = None   # objects after the latest collected board
        self.stats = {"frames": 0, "wall_ms": 0.0, "cpu_ms": 0.0, "max_ms": 0.0, "waits_ms": 0.0, "restarts": 0,
                      "errors": 0, "timeouts": 0, "ms_list": []}
        try:
            self.proc = subprocess.Popen(_launcher("main"), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.DEVNULL, env=_child_env(), bufsize=1 << 16)
            self.alive = True
            threading.Thread(target=self._reader, daemon=True).start()
        except Exception as exc:
            self.alive = False
            self.dead_reason = f"could not start: {exc!r}"

    def _reader(self) -> None:
        p = self.proc
        try:
            for raw in p.stdout:
                try:
                    self.q.put(json.loads(raw))
                except Exception:
                    continue
        except Exception:
            pass
        self.q.put(None)

    def feed(self, board, level: int) -> int | None:
        """Send one board (non-blocking). Returns its sequence number, or None when the worker is gone."""
        if not self.alive or self.proc is None:
            return None
        import numpy as np
        arr = np.asarray(board, dtype=np.uint8)
        self.seq += 1
        msg = {"op": "step", "seq": self.seq, "level": int(level), "h": int(arr.shape[0]), "w": int(arr.shape[1]),
               "grid": "".join("0123456789abcdef"[int(v) & 15] for v in arr.ravel().tolist())}
        try:
            self.proc.stdin.write((json.dumps(msg, separators=(",", ":")) + "\n").encode())
            self.proc.stdin.flush()
        except Exception as exc:
            self._die(f"pipe closed: {exc!r}")
            return None
        return self.seq

    def result(self, seq: int | None, timeout: float) -> dict | None:
        """Wait for the response to `seq` (earlier responses are consumed in order). None on timeout / failure."""
        if seq is None:
            return None
        t0 = time.time()
        while seq not in self.last:
            if not self.alive:
                return None
            left = timeout - (time.time() - t0)
            if left <= 0:
                self.stats["timeouts"] += 1
                self._die(f"no answer within {timeout:.0f}s")
                return None
            try:
                r = self.q.get(timeout=left)
            except queue.Empty:
                continue
            if r is None:
                self._die("worker exited")
                return None
            self._account(r)
            self.last[int(r.get("seq") or 0)] = r
        self.stats["waits_ms"] += (time.time() - t0) * 1000
        r = self.last.pop(seq)
        for k in [k for k in self.last if k < seq]:
            self.last.pop(k, None)
        return r

    def _account(self, r: dict) -> None:
        st = self.stats
        st["frames"] += 1
        ms = float(r.get("ms") or 0)
        st["wall_ms"] += ms
        st["cpu_ms"] += float(r.get("cpu_ms") or 0)
        st["max_ms"] = max(st["max_ms"], ms)
        st["ms_list"].append(ms)
        if r.get("restarted"):
            st["restarts"] += 1
        if r.get("error"):
            st["errors"] += 1
            st["last_error"] = str(r.get("error"))[:300]
        if r.get("skipped"):
            st["skipped"] = st.get("skipped", 0) + 1
        if r.get("ok") and r.get("rec") is not None:
            self.scene = r["rec"]

    def _die(self, reason: str) -> None:
        if self.alive:
            self.alive = False
            self.dead_reason = reason
            try:
                if self.proc is not None:
                    self.proc.kill()
            except Exception:
                pass

    def close(self) -> None:
        if self.proc is None:
            return
        try:
            if self.alive:
                self.proc.stdin.write(b'{"op":"quit"}\n')
                self.proc.stdin.flush()
                self.proc.wait(timeout=5)
        except Exception:
            pass
        try:
            if self.proc.poll() is None:
                self.proc.kill()
        except Exception:
            pass
        self.alive = False

    def summary(self) -> dict[str, Any]:
        st = dict(self.stats)
        ms = sorted(st.pop("ms_list"))
        n = len(ms)
        st["p50_ms"] = ms[n // 2] if n else 0.0
        st["p90_ms"] = ms[min(n - 1, int(0.9 * n))] if n else 0.0
        st["mean_ms"] = round(st["wall_ms"] / n, 1) if n else 0.0
        st["wall_s"] = round(st.pop("wall_ms") / 1000, 2)
        st["cpu_s"] = round(st.pop("cpu_ms") / 1000, 2)
        st["waits_s"] = round(st.pop("waits_ms") / 1000, 2)
        st["dead_reason"] = self.dead_reason
        return st


if __name__ == "__main__":
    main()

"""lib (v2.11): generic grid tools for the agent's own code, in run and inside next_state.

Standalone (numpy + stdlib only): the harness imports this file as agent.gridlib; the sandbox worker executes the same
source as the module "arc_gridlib". Every function returns new values; nothing here touches the game.

The harness also uses change_report() / short_summary() / diff_objects() from here, so the change lines in its feedback,
lib.frame_diff() and the model check all come from the same code.
"""
from __future__ import annotations

import difflib
import hashlib
import sys
from collections import Counter, deque
from typing import Any

import numpy as np

_MODNAME = "arc_gridlib"
_OBJ: Any = None          # objects module (segment, shape_signature): bound by bind()
_FD: Any = None           # frame_diff module (compute, render_text, anim_steps)
_NAMES: dict[int, str] = {i: str(i) for i in range(16)}
_RO = ("an Obj is read-only: lib.moved / lib.placed / lib.rotated / lib.recolored ... return changed copies "
       "(o.copy() gives a plain editable dict)")


def bind(objects_module: Any, frame_diff_module: Any, color_names: dict[int, str] | None = None) -> None:
    global _OBJ, _FD, _NAMES
    _OBJ, _FD = objects_module, frame_diff_module
    if color_names:
        _NAMES = {int(k): str(v) for k, v in color_names.items()}


# =========================================================================================== the Obj type

def _make_obj(data: dict) -> "Obj":
    return Obj(data)


class Obj(dict):
    """A read-only dict describing an object: cells, colors, color, color_name, size, rect, hash, shape_hash (+ more from
    lib.extract_components()). Fields are also attributes (o.cells). o == p: same look in the same place."""
    __slots__ = ()

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"Obj has no field {name!r} (fields: {', '.join(map(str, self.keys()))})") from None

    def __setattr__(self, name: str, value: Any) -> None:
        raise TypeError(_RO)

    def __delattr__(self, name: str) -> None:
        raise TypeError(_RO)

    def __setitem__(self, key: Any, value: Any) -> None:
        raise TypeError(_RO)

    def __delitem__(self, key: Any) -> None:
        raise TypeError(_RO)

    def _ro(self, *a: Any, **k: Any) -> Any:
        raise TypeError(_RO)

    clear = pop = popitem = setdefault = update = __ior__ = _ro

    def copy(self) -> dict:
        return dict(self)

    def __reduce__(self) -> Any:
        return (_make_obj, (dict(self),))

    def __eq__(self, other: Any) -> Any:
        if isinstance(other, dict) and "cells" in other:
            try:
                return _look_place(self) == _look_place(other)
            except Exception:
                return False
        return NotImplemented

    def __ne__(self, other: Any) -> Any:
        r = self.__eq__(other)
        return r if r is NotImplemented else not r

    def __hash__(self) -> int:
        return hash(_look_place(self))

    def __repr__(self) -> str:
        head = f"Obj#{self['id']}" if self.get("id") is not None else "Obj"
        r = self.get("rect")
        if not r:
            return f"{head} (no cells)"
        s = f"{head} {self.get('color_name', '')} {r[1] - r[0]}x{r[3] - r[2]} at [{r[0]}:{r[1]}, {r[2]}:{r[3]}], {self.get('size', 0)} cells"
        if self.get("score") is not None:
            s += f", score {float(self['score']):.2f}"
        return s

    __str__ = __repr__


_make_obj.__module__ = _MODNAME
Obj.__module__ = _MODNAME


def _make_minimal_component(data: dict) -> "MinimalComponent":
    return MinimalComponent(data)


class MinimalComponent(dict):
    """An ordinary component dictionary with compact display and lossless IPC.

    Unlike Obj, this retains normal dictionary equality and mutability. Explicit
    field access, dict(component) and JSON serialization keep all measured data.
    """
    __slots__ = ()

    def __repr__(self) -> str:
        counts = Counter(self.get("colors") or ())
        if len(counts) > 1:
            ids = "/".join(str(c) for c in sorted(counts, key=lambda c: (-counts[c], c)))
            colour = f"colours {ids} {self.get('color_name', '')}"
        else:
            colour = f"colour {self.get('color')} {self.get('color_name', '')}"
        rect = self.get("rect")
        text = (f"{colour}, size {self.get('size', 0)}, "
                f"rect {tuple(rect) if rect is not None else None}, center {self.get('center')}")
        if self.get("score") is not None:
            text += f", score {float(self['score']):.2f}"
        return text

    __str__ = __repr__

    def copy(self) -> "MinimalComponent":
        return MinimalComponent(self)

    def __reduce__(self) -> Any:
        return (_make_minimal_component, (dict(self),))


_make_minimal_component.__module__ = _MODNAME
MinimalComponent.__module__ = _MODNAME

if _MODNAME not in sys.modules and __name__ != _MODNAME:
    sys.modules[_MODNAME] = sys.modules[__name__]


def _look_place(o: dict) -> tuple:
    if isinstance(o, Obj):
        return tuple(zip(o["cells"], o["colors"]))
    cells, colors = _cells_colors(o)
    return tuple(sorted(zip(cells, colors)))


def _name_for(colors: Any) -> str:
    cnt = Counter(int(v) for v in colors)
    return "+".join(_NAMES.get(c, str(c)) for c, _ in sorted(cnt.items(), key=lambda t: (-t[1], t[0])))


def _main_color(colors: Any) -> int | None:
    cnt = Counter(int(v) for v in colors)
    if not cnt:
        return None
    return sorted(cnt.items(), key=lambda t: (-t[1], t[0]))[0][0]


def _look_hash(cells: Any, colors: Any) -> str:
    if not cells:
        return ""
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    key = tuple(sorted((r - r0, c - c0, int(v)) for (r, c), v in zip(cells, colors)))
    return hashlib.sha1(repr(key).encode()).hexdigest()[:16]


def _shape_hash(cells: Any) -> str:
    if not cells:
        return ""
    return _OBJ.shape_signature(cells, 0)["shape_hash"]


def _build(cells: Any, colors: Any, extra: dict | None = None, *, hash_: str | None = None, shape_hash: str | None = None,
           signature: bool = False, presorted: bool = False) -> Obj:
    if presorted:
        cl = tuple(cells)
        co = tuple(int(v) for v in colors)
    else:
        pairs: dict[tuple[int, int], int] = {}
        for (r, c), v in zip(cells, colors):
            pairs[(int(r), int(c))] = int(v)
        items = sorted(pairs.items())
        cl = tuple(p for p, _ in items)
        co = tuple(v for _, v in items)
    d: dict[str, Any] = {}
    if extra and "id" in extra:
        d["id"] = extra["id"]
    d["cells"] = cl
    d["colors"] = co
    d["color"] = _main_color(co)
    d["color_name"] = _name_for(co)
    d["size"] = len(cl)
    if cl:
        rs = [r for r, _ in cl]
        cs = [c for _, c in cl]
        d["rect"] = (min(rs), max(rs) + 1, min(cs), max(cs) + 1)
        d["hash"] = hash_ or _look_hash(cl, co)
        if signature:
            sig = _OBJ.shape_signature(cl, 0)
            d["shape_hash"] = sig["shape_hash"]
            d["rotation"] = sig["rotation"]
            d["symmetry"] = sig["symmetry"]
        else:
            d["shape_hash"] = shape_hash or _shape_hash(cl)
    else:
        d["rect"] = None
        d["hash"] = ""
        d["shape_hash"] = ""
    if extra:
        for k, v in extra.items():
            if k not in d:
                d[k] = v
    return Obj(d)


def _cells_colors(x: Any) -> tuple[list, list]:
    """(cells, colours) of an Obj, a dict with cells (+ colors or color), or a list of (r, c) cells (colours None)."""
    if isinstance(x, dict):
        if "cells" not in x:
            raise TypeError("expected an object (an Obj or a dict with 'cells')")
        cells = [(int(r), int(c)) for r, c in x["cells"]]
        if "colors" in x and x["colors"] is not None and len(x["colors"]) == len(cells):
            colors = [int(v) for v in x["colors"]]
        elif x.get("color") is not None:
            colors = [int(x["color"])] * len(cells)
        else:
            colors = None
        return cells, colors
    if isinstance(x, (list, tuple, set, frozenset)):
        cells = [(int(r), int(c)) for r, c in x]
        return cells, None
    raise TypeError(f"expected an object or a list of (row, col) cells, got {type(x).__name__}")


def _as_obj(x: Any) -> Obj:
    if isinstance(x, Obj):
        return x
    cells, colors = _cells_colors(x)
    if colors is None:
        raise TypeError("this needs an object with colours (an Obj, or a dict with 'cells' and 'colors' or 'color')")
    extra = {k: v for k, v in x.items() if k in ("id",)} if isinstance(x, dict) else None
    return _build(cells, colors, extra)


def _cells(x: Any) -> list[tuple[int, int]]:
    if isinstance(x, Obj):
        return list(x["cells"])
    return _cells_colors(x)[0]


_ARR: dict[int, tuple] = {}


def _rc(o: Obj) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cl = o["cells"]
    hit = _ARR.get(id(cl))
    if hit is not None and hit[0] is cl:
        return hit[1], hit[2], hit[3]
    a = np.asarray(cl, dtype=np.int64).reshape(-1, 2)
    v = np.asarray(o["colors"], dtype=np.int64).reshape(-1)
    if len(_ARR) > 20000:
        _ARR.clear()
    _ARR[id(cl)] = (cl, a[:, 0], a[:, 1], v)
    return a[:, 0], a[:, 1], v


def _board_arr(board: Any) -> np.ndarray:
    a = np.asarray(board)
    if a.ndim != 2:
        raise ValueError(f"a board must be a 2-D array, got shape {a.shape}")
    return a


def _colors_list(c: Any) -> list[int]:
    if isinstance(c, (int, np.integer)):
        return [int(c)]
    return [int(v) for v in c]


def _objs_list(objs: Any) -> list:
    return [objs] if isinstance(objs, dict) else list(objs)


# =========================================================================================== changed copies

def _keep(o: Obj, *fields: str) -> dict:
    return {k: o[k] for k in ("id",) + fields if k in o}


def moved(o: Any, dr: int, dc: int) -> Obj:
    """moved(o, dr, dc) -> a copy of o moved by dr rows and dc columns.
    Example: lib.moved(p, 0, 1) is p one cell to the right (nothing is drawn; see lib.draw / lib.move_on_board)."""
    o = _as_obj(o)
    dr, dc = int(dr), int(dc)
    if not o["cells"]:
        return o
    r = o["rect"]
    d = _keep(o, "rotation", "symmetry")
    d.update(cells=tuple((a + dr, b + dc) for a, b in o["cells"]), colors=o["colors"], color=o["color"],
             color_name=o["color_name"], size=o["size"], rect=(r[0] + dr, r[1] + dr, r[2] + dc, r[3] + dc),
             hash=o["hash"], shape_hash=o["shape_hash"])
    out = {"id": d.pop("id")} if "id" in d else {}
    out.update(d)
    return Obj(out)


def placed(o: Any, r: int, c: int) -> Obj:
    """placed(o, r, c) -> a copy of o with the top-left corner of its rect at (r, c)."""
    o = _as_obj(o)
    if not o["cells"]:
        return o
    return moved(o, int(r) - o["rect"][0], int(c) - o["rect"][2])


def rotated(o: Any, quarter_turns: int = 1) -> Obj:
    """rotated(o, quarter_turns=1) -> a copy turned clockwise by 90 degrees per quarter turn; the top-left corner of its
    rect stays where it was. The change report calls this "rotated 90 deg clockwise"."""
    o = _as_obj(o)
    if not o["cells"]:
        return o
    r0, c0 = o["rect"][0], o["rect"][2]
    pts = [(r - r0, c - c0) for r, c in o["cells"]]
    for _ in range(int(quarter_turns) % 4):
        h = max(y for y, _ in pts) + 1
        pts = [(x, h - 1 - y) for y, x in pts]
    return _build([(y + r0, x + c0) for y, x in pts], o["colors"], _keep(o), shape_hash=o["shape_hash"])


def mirrored(o: Any, axis: str = "left-right") -> Obj:
    """mirrored(o, axis="left-right" | "up-down") -> a copy flipped inside its rect ("left-right" swaps left and right,
    "up-down" swaps top and bottom). The change report calls this "mirrored left-right" / "mirrored up-down"."""
    o = _as_obj(o)
    if not o["cells"]:
        return o
    r0, r1, c0, c1 = o["rect"]
    ax = str(axis).lower().replace("_", "-").replace(" ", "-")
    if ax in ("up-down", "updown", "ud", "vertical", "top-bottom"):
        cells = [(r0 + r1 - 1 - r, c) for r, c in o["cells"]]
    elif ax in ("left-right", "leftright", "lr", "horizontal"):
        cells = [(r, c0 + c1 - 1 - c) for r, c in o["cells"]]
    else:
        raise ValueError('axis must be "left-right" or "up-down"')
    return _build(cells, o["colors"], _keep(o), shape_hash=o["shape_hash"])


def _scale_factor(k: Any) -> tuple[int, int]:
    """(up, down): exactly one is > 1 unless k == 1."""
    k = float(k)
    if k <= 0:
        raise ValueError("k must be positive")
    if k >= 1:
        if abs(k - round(k)) > 1e-9:
            raise ValueError("k must be a whole number (2, 3, ...) or 1/n (0.5, 1/3, ...)")
        return int(round(k)), 1
    n = int(round(1.0 / k))
    if abs(n * k - 1.0) > 1e-6:
        raise ValueError("k below 1 must be 1/n (0.5, 1/3, 0.25, ...)")
    return 1, n


def scaled(x: Any, k: Any) -> Any:
    """scaled(x, k) -> x resized by k. x is an Obj (its top-left corner stays) or a 2-D array (a board region).
    k is a whole number (each cell becomes a k x k block) or 1/n (each n x n block becomes one cell; only when every block
    is one colour - otherwise an error names the first mixed block). Example: lib.scaled(region, 1/3)."""
    up, down = _scale_factor(k)
    if isinstance(x, np.ndarray) or (isinstance(x, list) and x and isinstance(x[0], (list, np.ndarray))):
        a = np.asarray(x)
        if a.ndim != 2:
            raise ValueError("scaled() takes an object or a 2-D array")
        if up > 1:
            return np.repeat(np.repeat(a, up, axis=0), up, axis=1)
        if down == 1:
            return a.copy()
        h, w = a.shape
        if h % down or w % down:
            raise ValueError(f"the region is {h}x{w}, not a multiple of {down} in both directions")
        blocks = a.reshape(h // down, down, w // down, down)
        uniform = (blocks == blocks[:, :1, :, :1]).all(axis=(1, 3))
        if not uniform.all():
            i, j = map(int, np.argwhere(~uniform)[0])
            vals = sorted(set(int(v) for v in blocks[i, :, j, :].ravel()))
            raise ValueError(f"not exact: the {down}x{down} block at rows {i * down}:{i * down + down}, cols "
                             f"{j * down}:{j * down + down} has colours {vals}")
        return blocks[:, 0, :, 0].copy()
    o = _as_obj(x)
    if not o["cells"]:
        return o
    r0, c0 = o["rect"][0], o["rect"][2]
    if up > 1:
        cells, colors = [], []
        for (r, c), v in zip(o["cells"], o["colors"]):
            for i in range(up):
                for j in range(up):
                    cells.append((r0 + (r - r0) * up + i, c0 + (c - c0) * up + j))
                    colors.append(v)
        return _build(cells, colors, _keep(o))
    if down == 1:
        return o
    blocks: dict[tuple[int, int], list[int]] = {}
    for (r, c), v in zip(o["cells"], o["colors"]):
        blocks.setdefault(((r - r0) // down, (c - c0) // down), []).append(v)
    cells, colors = [], []
    for (bi, bj), vals in sorted(blocks.items()):
        if len(vals) != down * down or len(set(vals)) != 1:
            raise ValueError(f"not exact: the {down}x{down} block at rows {r0 + bi * down}:{r0 + bi * down + down}, cols "
                             f"{c0 + bj * down}:{c0 + bj * down + down} is "
                             + ("only partly part of the object" if len(vals) != down * down else f"mixed {sorted(set(vals))}"))
        cells.append((r0 + bi, c0 + bj))
        colors.append(vals[0])
    return _build(cells, colors, _keep(o))


def recolored(o: Any, mapping: dict) -> Obj:
    """recolored(o, {old: new, ...}) -> a copy with those colours swapped. Example: lib.recolored(p, {12: 8})."""
    o = _as_obj(o)
    m = {int(k): int(v) for k, v in dict(mapping).items()}
    return _build(o["cells"], [m.get(v, v) for v in o["colors"]], _keep(o), shape_hash=o["shape_hash"], presorted=True)


def merged(a: Any, b: Any) -> Obj:
    """merged(a, b) -> one object made of both (where both have a cell, b's colour wins)."""
    a, b = _as_obj(a), _as_obj(b)
    pairs = dict(zip(a["cells"], a["colors"]))
    pairs.update(zip(b["cells"], b["colors"]))
    return _build(list(pairs.keys()), list(pairs.values()))


# =========================================================================================== board in, board out

def draw(board: Any, objs: Any) -> np.ndarray:
    """draw(board, objs) -> a new board with one object or a list of objects painted on it (later ones on top; cells off
    the board are ignored). The board passed in is unchanged."""
    b = np.array(board, dtype=np.uint8, copy=True)
    h, w = b.shape[:2]
    for o in _objs_list(objs):
        o = _as_obj(o)
        rr, cc, vv = _rc(o)
        ok = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
        b[rr[ok], cc[ok]] = vv[ok]
    return b


def erase(board: Any, objs: Any, fill: int) -> np.ndarray:
    """erase(board, objs, fill) -> a new board with the cells of the object(s) set to colour fill (usually the floor)."""
    b = np.array(board, dtype=np.uint8, copy=True)
    h, w = b.shape[:2]
    for o in _objs_list(objs):
        cells = _cells(o)
        if not cells:
            continue
        a = np.asarray(cells, dtype=np.int64).reshape(-1, 2)
        ok = (a[:, 0] >= 0) & (a[:, 0] < h) & (a[:, 1] >= 0) & (a[:, 1] < w)
        b[a[ok, 0], a[ok, 1]] = int(fill)
    return b


def move_on_board(board: Any, o: Any, dr: int, dc: int, fill: int) -> np.ndarray:
    """move_on_board(board, o, dr, dc, fill) -> a new board with o moved by (dr, dc): its old cells become fill, then it
    is drawn at the new place. The usual way to move something inside next_state."""
    o = _as_obj(o)
    return draw(erase(board, o, fill), moved(o, dr, dc))


def hits(board: Any, o: Any, colors: Any) -> bool:
    """hits(board, o, colors) -> True when any cell of o lies on a board cell of these colours; cells off the board count
    as hits. The wall check: lib.hits(board, lib.moved(p, 0, 1), [5])."""
    b = _board_arr(board)
    cells = _cells(o)
    if not cells:
        return False
    a = np.asarray(cells, dtype=np.int64).reshape(-1, 2)
    h, w = b.shape
    if ((a[:, 0] < 0) | (a[:, 0] >= h) | (a[:, 1] < 0) | (a[:, 1] >= w)).any():
        return True
    return bool(np.isin(b[a[:, 0], a[:, 1]], _colors_list(colors)).any())


def object_mask(o: Any, shape: Any = (64, 64)) -> np.ndarray:
    """object_mask(o) -> a board-sized True/False array, True on o's cells (combine with & | ~, index boards with it)."""
    m = np.zeros(tuple(shape)[:2], dtype=bool)
    for r, c in _cells(o):
        if 0 <= r < m.shape[0] and 0 <= c < m.shape[1]:
            m[r, c] = True
    return m


# =========================================================================================== relations

def overlap(a: Any, b: Any) -> bool:
    """overlap(a, b) -> True when they share at least one cell."""
    return bool(set(_cells(a)) & set(_cells(b)))


def touching(a: Any, b: Any, diagonal: bool = False) -> bool:
    """touching(a, b, diagonal=False) -> True when a cell of a is next to a cell of b and they do not overlap."""
    sa, sb = set(_cells(a)), set(_cells(b))
    if not sa or not sb or sa & sb:
        return False
    nb = ((-1, 0), (1, 0), (0, -1), (0, 1)) + (((-1, -1), (-1, 1), (1, -1), (1, 1)) if diagonal else ())
    small, big = (sa, sb) if len(sa) <= len(sb) else (sb, sa)
    return any((r + dr, c + dc) in big for r, c in small for dr, dc in nb)


def holes(o: Any) -> list[tuple[int, int]]:
    """holes(o) -> the cells o encloses (not part of o, not reachable from outside its rect without crossing o)."""
    cells = set(_cells(o))
    if not cells:
        return []
    r0 = min(r for r, _ in cells) - 1
    r1 = max(r for r, _ in cells) + 2
    c0 = min(c for _, c in cells) - 1
    c1 = max(c for _, c in cells) + 2
    seen = {(r0, c0)}
    q = deque([(r0, c0)])
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            n = (r + dr, c + dc)
            if r0 <= n[0] < r1 and c0 <= n[1] < c1 and n not in seen and n not in cells:
                seen.add(n)
                q.append(n)
    return [(r, c) for r in range(r0, r1) for c in range(c0, c1) if (r, c) not in seen and (r, c) not in cells]


def contains(a: Any, b: Any) -> bool:
    """contains(a, b) -> True when every cell of b lies in a's holes (b is enclosed by a)."""
    hb = _cells(b)
    if not hb:
        return False
    hs = set(holes(a))
    return all(c in hs for c in hb)


def around(board: Any, o: Any, diagonal: bool = False) -> dict:
    """around(board, o, diagonal=False) -> {(r, c): colour} of the board cells directly next to o (any shape): what it
    touches."""
    b = _board_arr(board)
    cells = set(_cells(o))
    h, w = b.shape
    nb = ((-1, 0), (1, 0), (0, -1), (0, 1)) + (((-1, -1), (-1, 1), (1, -1), (1, 1)) if diagonal else ())
    out = {}
    for r, c in cells:
        for dr, dc in nb:
            n = (r + dr, c + dc)
            if n not in cells and 0 <= n[0] < h and 0 <= n[1] < w:
                out[n] = int(b[n])
    return dict(sorted(out.items()))


def gap(a: Any, b: Any) -> int:
    """gap(a, b) -> number of empty cells between their nearest cells (rows or columns, whichever is larger):
    0 = touching (also diagonally), -1 = overlapping."""
    ca, cb = _cells(a), _cells(b)
    if not ca or not cb:
        raise ValueError("gap() needs two non-empty objects")
    x = np.asarray(ca, dtype=np.int64).reshape(-1, 1, 2)
    y = np.asarray(cb, dtype=np.int64).reshape(1, -1, 2)
    d = np.abs(x - y).max(axis=2).min()
    return int(d) - 1


def offset(a: Any, b: Any) -> tuple[int, int]:
    """offset(a, b) -> (dr, dc) from a's top-left corner to b's top-left corner; lib.moved(a, *lib.offset(a, b)) puts a's
    corner on b's."""
    ca, cb = _cells(a), _cells(b)
    if not ca or not cb:
        raise ValueError("offset() needs two non-empty objects")
    return (min(r for r, _ in cb) - min(r for r, _ in ca), min(c for _, c in cb) - min(c for _, c in ca))


# =========================================================================================== comparing

def _norm_look(o: Any) -> tuple:
    cells, colors = _cells_colors(o)
    if not cells:
        return ()
    if colors is None:
        colors = [0] * len(cells)
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0, v) for (r, c), v in zip(cells, colors)))


def _variants(pts: tuple, rotate: bool, flip: bool) -> list[tuple]:
    """D4 variants (normalized, deduplicated) of ((r, c, v), ...)."""
    def norm(p):
        r0 = min(t[0] for t in p)
        c0 = min(t[1] for t in p)
        return tuple(sorted((t[0] - r0, t[1] - c0, t[2]) for t in p))

    def rot(p):
        h = max(t[0] for t in p) + 1
        return norm([(t[1], h - 1 - t[0], t[2]) for t in p])

    def flp(p):
        w = max(t[1] for t in p) + 1
        return norm([(t[0], w - 1 - t[1], t[2]) for t in p])
    base = [pts]
    if rotate:
        for _ in range(3):
            base.append(rot(base[-1]))
    out = list(base)
    if flip:
        out += [flp(p) for p in base]
    uniq: list[tuple] = []
    for p in out:
        if p not in uniq:
            uniq.append(p)
    return uniq


def same_look(a: Any, b: Any, rotate: bool = False, flip: bool = False) -> bool:
    """same_look(a, b, rotate=False, flip=False) -> True when a and b have the same shape and colours (anywhere on the
    board); rotate / flip = True also accept turned or mirrored copies."""
    la, lb = _norm_look(a), _norm_look(b)
    if len(la) != len(lb):
        return False
    if not la:
        return True
    return lb in _variants(la, rotate, flip)


def same_shape(a: Any, b: Any, rotate: bool = True, flip: bool = True) -> bool:
    """same_shape(a, b, rotate=True, flip=True) -> True when a and b have the same outline (colours ignored)."""
    la = tuple((r, c, 0) for r, c, _ in _norm_look(a))
    lb = tuple((r, c, 0) for r, c, _ in _norm_look(b))
    if len(la) != len(lb):
        return False
    if not la:
        return True
    return lb in _variants(la, rotate, flip)


# =========================================================================================== segmentation

def _flood_components(b: np.ndarray, allowed: np.ndarray | None, rects=()) -> list[tuple[list, list]]:
    """4-connected components in reading order of their first cell. allowed None: equal colour; else: connected through
    allowed cells regardless of colour (cells not allowed belong to no component)."""
    h, w = b.shape
    flat = b.ravel().tolist()
    zones = np.zeros(b.shape, dtype=np.int16)
    for zone, (r0, r1, c0, c1) in enumerate(rects, 1):
        zones[r0:r1, c0:c1] = zone
    zones = zones.ravel().tolist()
    ok = allowed.ravel().tolist() if allowed is not None else None
    seen = bytearray(h * w)
    out = []
    for start in range(h * w):
        if seen[start] or (ok is not None and not ok[start]):
            continue
        col = flat[start]
        stack = [start]
        seen[start] = 1
        idx = []
        while stack:
            i = stack.pop()
            idx.append(i)
            r, c = divmod(i, w)
            for j in ((i - w) if r > 0 else -1, (i + w) if r < h - 1 else -1, (i - 1) if c > 0 else -1, (i + 1) if c < w - 1 else -1):
                if j < 0 or seen[j]:
                    continue
                if zones[j] == zones[start] and (ok[j] if ok is not None else flat[j] == col):
                    seen[j] = 1
                    stack.append(j)
        idx.sort()
        out.append(([divmod(i, w) for i in idx], [flat[i] for i in idx]))
    return out


def _filter_mask(b: np.ndarray, only: Any, exclude: Any) -> np.ndarray | None:
    if only is not None and exclude is not None:
        raise ValueError("give only=[...] or exclude=[...], not both")
    if only is not None:
        return np.isin(b, _colors_list(only))
    if exclude is not None:
        return ~np.isin(b, _colors_list(exclude))
    return None


def _in_rects(rect: Any, rects: Any) -> bool:
    return bool(rects) and any(rect[0] >= q[0] and rect[1] <= q[1] and rect[2] >= q[2] and rect[3] <= q[3] for q in rects)


def diff_objects(board: Any, rects: Any = None, exclude: Any = None, touch: np.ndarray | None = None) -> list[dict]:
    """Objects for frame_diff.compute (harness + lib.frame_diff): single-colour regions (largest = background), or with
    exclude the multi-colour components of the other colours. Signatures only for objects next to changed cells (touch)."""
    b = np.asarray(board).astype(np.int64, copy=False)
    rects = list(_OBJ.norm_rects(rects)) if rects is not None else []
    allowed = (~np.isin(b, _colors_list(exclude))) if exclude else None
    if allowed is not None:   # the indicator is reported on its own: it never joins the objects around it
        for r0, r1, c0, c1 in rects:
            allowed[r0:r1, c0:c1] = False
    comps = _flood_components(b, allowed, rects)
    bg = None
    if allowed is None and comps:
        bg = max(range(len(comps)), key=lambda i: (len(comps[i][0]), -i))
    h, w = b.shape
    out = []
    for i, (cells, vals) in enumerate(comps):
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        rect = [min(rs), max(rs) + 1, min(cs), max(cs) + 1]
        o: dict[str, Any] = {"id": i, "cells": cells, "colors": vals, "size": len(cells), "rect": rect,
                             "color": _main_color(vals), "is_background": i == bg,
                             "is_remaining_actions_indicator": _in_rects(rect, rects)}
        o["color_name"] = _name_for(vals)
        need = touch is None or bool(touch[max(0, rect[0] - 1):min(h, rect[1] + 1), max(0, rect[2] - 1):min(w, rect[3] + 1)].any())
        if need and i != bg:
            o["hash"] = _look_hash(cells, vals)
            o["shape_hash"] = _OBJ.shape_signature(cells, 0)["shape_hash"]
        else:
            o["hash"] = f"untouched-{i}"
            o["shape_hash"] = f"untouched-{i}"
        out.append(o)
    return out


def change_report(before: Any, after: Any, rects: Any = None, exclude: Any = None, cap: int | None = None
                  ) -> tuple[dict, list[str]]:
    """(diff, lines): the object-level change between two boards; the lines are what the feedback shows."""
    b = np.asarray(before).astype(np.uint8)
    a = np.asarray(after).astype(np.uint8)
    if b.shape != a.shape:
        d = _FD.compute(b, a, [], [], indicator_rect=rects)
        return d, _FD.render_text(d, cap=10 ** 6)
    touch = b != a
    ob = diff_objects(b, rects, exclude, touch)
    oa = diff_objects(a, rects, exclude, touch)
    d = _FD.compute(b, a, ob, oa, indicator_rect=rects)
    return d, _FD.render_text(d, cap=(10 ** 6 if cap is None else int(cap)))


def _shape_word(e: dict, rect_key: str = "rect") -> str:
    rect = e.get(rect_key) or e.get("to") or e.get("from") or [0, 0, 0, 0]
    hh, ww = rect[1] - rect[0], rect[3] - rect[2]
    if e.get("size") == hh * ww:
        return f"{hh}x{ww}" if (hh, ww) != (1, 1) else "1-cell"
    return f"{e.get('size')}-cell"


def _ename(e: dict, key: str = "color", name_key: str = "color_name") -> str:
    name = e.get(name_key)
    return str(name) if name else f"color {e.get(key)}"


def short_summary(diff: dict) -> str:
    """One line for a step, positions left out of moves (so repeated identical steps can be merged)."""
    if diff.get("shape_changed"):
        return "board shape changed"
    parts: list[str] = []
    mv: dict[tuple, int] = {}
    for e in diff.get("moved", []):
        k = (_ename(e), _shape_word(e, "to"), tuple(e["delta"]))
        mv[k] = mv.get(k, 0) + 1
    for (nm, sh, dl), n in mv.items():
        parts.append(f"{(str(n) + ' x ') if n > 1 else ''}{nm} {sh} moved ({dl[0]:+d},{dl[1]:+d})")
    for e in diff.get("rotated", []):
        parts.append(f"{_ename(e)} {_shape_word(e)} {e.get('transform') or 'turned or flipped'} at "
                     f"[{e['rect'][0]}:{e['rect'][1]}, {e['rect'][2]}:{e['rect'][3]}]")
    for e in diff.get("recolored", []):
        r = e["rect"]
        parts.append(f"{_shape_word(e)} at [{r[0]}:{r[1]}, {r[2]}:{r[3]}] recolored "
                     f"{e.get('from_color_name') or e['from_color']} -> {e.get('to_color_name') or e['to_color']}")
    for key, word in (("appeared", "appeared at"), ("disappeared", "disappeared from")):
        for e in diff.get(key, []):
            r = e["rect"]
            parts.append(f"{_ename(e)} {_shape_word(e)} {word} [{r[0]}:{r[1]}, {r[2]}:{r[3]}]")
    # v2.12: reshapes last (often a floor or background region changing shape as things move over it)
    for e in diff.get("reshaped", []):
        r = e["to_rect"]
        parts.append(f"{_ename(e)} reshaped {e['from_size']} -> {e['to_size']} cells at [{r[0]}:{r[1]}, {r[2]}:{r[3]}]")
    if len(parts) > 6:
        parts = parts[:6] + [f"{len(parts) - 6} more changes"]
    outside = diff.get("changed_outside_indicator", 0)
    inside = (diff.get("changed_cells", 0) or 0) - (outside or 0)
    if not parts:
        parts.append(f"{outside} cells changed, no object change matched" if outside else
                     "final board unchanged outside the remaining-actions indicator")
    if inside > 0:
        parts.append(f"indicator: {inside} cell(s)")
    return "; ".join(parts)


# =========================================================================================== the bound library

def _components_objs(b: np.ndarray, only: Any, exclude: Any, rects: Any) -> list[Obj]:
    allowed = _filter_mask(b, only, exclude)
    out = []
    for i, (cells, vals) in enumerate(_flood_components(b, allowed, rects)):
        o = _build(cells, vals, {"id": i}, signature=True, presorted=True)
        out.append(Obj(dict(o, is_remaining_actions_indicator=_in_rects(o["rect"], rects))))
    return out


_SEG_CACHE: dict[tuple, list] = {}
_SEG_ORDER: list[tuple] = []


def _segment_objs(b: np.ndarray, rects: Any) -> list[Obj]:
    key = (b.tobytes(), b.shape, tuple(tuple(r) for r in rects))
    hit = _SEG_CACHE.get(key)
    if hit is not None:
        return hit
    segs = _OBJ.segment(b, indicator_rect=list(rects) or None, color_names=_NAMES)
    out = []
    for s in segs:
        n = len(s["cells"])
        cells = tuple(tuple(x) for x in s["cells"])
        cols = (int(s["color"]),) * n
        d = {"id": s["id"], "cells": cells, "colors": cols, "color": int(s["color"]), "color_name": s["color_name"],
             "size": n, "rect": tuple(s["rect"]), "hash": _look_hash(cells, cols), "shape_hash": s["shape_hash"],
             "rotation": s["rotation"], "symmetry": s["symmetry"], "children": s["children"], "adjacent": s["adjacent"],
             "is_background": s["is_background"], "is_remaining_actions_indicator": s["is_remaining_actions_indicator"],
             "crop": s.get("crop")}
        out.append(Obj(d))
    _SEG_CACHE[key] = out
    _SEG_ORDER.append(key)
    while len(_SEG_ORDER) > 16:
        _SEG_CACHE.pop(_SEG_ORDER.pop(0), None)
    return out


def _find(b: np.ndarray, o: Any, min_score: float, rotate: bool, flip: bool, match_colors: bool) -> list[Obj]:
    o = _as_obj(o)
    if not o["cells"]:
        return []
    H, W = b.shape
    n = o["size"]
    base = _norm_look(o)
    found: dict[tuple, Obj] = {}
    need = max(1, int(np.ceil(float(min_score) * n - 1e-9))) if match_colors else n
    for var in _variants(base, rotate, flip):
        var_vals = [int(t[2]) for t in var]
        var_hash = _look_hash([(t[0], t[1]) for t in var], var_vals)
        h = max(t[0] for t in var) + 1
        w = max(t[1] for t in var) + 1
        if h > H or w > W:
            continue
        ph, pw = H - h + 1, W - w + 1
        if match_colors:
            score = np.zeros((ph, pw), dtype=np.int32)
            for r, c, v in var:
                score += (b[r:r + ph, c:c + pw] == v)
            okm = score >= need
        else:
            groups: dict[int, list] = {}
            for r, c, v in var:
                groups.setdefault(v, []).append((r, c))
            okm = np.ones((ph, pw), dtype=bool)
            refs = []
            for pts in groups.values():
                r0, c0 = pts[0]
                ref = b[r0:r0 + ph, c0:c0 + pw]
                for r, c in pts[1:]:
                    okm &= (b[r:r + ph, c:c + pw] == ref)
                refs.append(ref)
            for i in range(len(refs)):
                for j in range(i + 1, len(refs)):
                    okm &= refs[i] != refs[j]
            score = np.full((ph, pw), n, dtype=np.int32)
        for pr, pc in np.argwhere(okm)[:2000].tolist():
            cells = [(pr + r, pc + c) for r, c, _ in var]
            vals = [int(b[r, c]) for r, c in cells]
            key = tuple(sorted(zip(cells, vals)))
            if key in found:
                continue
            ob = _build(cells, vals, {"score": round(float(score[pr, pc]) / n, 4)},
                        hash_=(var_hash if vals == var_vals else None), shape_hash=o["shape_hash"])
            found[key] = ob
        if len(found) >= 1000:
            break
    res = sorted(found.values(), key=lambda x: (x["rect"][0], x["rect"][2], -x["score"]))
    return res[:1000]


def _object_at(b: np.ndarray, r: int, c: int, only: Any, exclude: Any) -> Obj | None:
    h, w = b.shape
    r, c = int(r), int(c)
    if not (0 <= r < h and 0 <= c < w):
        return None
    allowed = _filter_mask(b, only, exclude)
    if allowed is not None and not allowed[r, c]:
        return None
    col = int(b[r, c])
    seen = {(r, c)}
    q = [(r, c)]
    while q:
        y, x = q.pop()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            n = (y + dy, x + dx)
            if n in seen or not (0 <= n[0] < h and 0 <= n[1] < w):
                continue
            if (allowed[n] if allowed is not None else int(b[n]) == col):
                seen.add(n)
                q.append(n)
    cells = sorted(seen)
    return _build(cells, [int(b[p]) for p in cells], presorted=True)


def _walk(b: np.ndarray, o: Any, step: int, only: Any, exclude: Any, goal: Any = None) -> tuple:
    allowed = _filter_mask(b, only, exclude)
    if allowed is None:
        raise ValueError("give only=[colours it may stand on] or exclude=[colours that block it]")
    allowed = allowed.copy()
    cells = _cells(o)
    if not cells:
        raise ValueError("the object has no cells")
    h, w = b.shape
    for r, c in cells:
        if 0 <= r < h and 0 <= c < w:
            allowed[r, c] = True
    goal_set: set = set()
    if goal is not None:
        if isinstance(goal, tuple) and len(goal) == 2 and all(isinstance(v, (int, np.integer)) for v in goal):
            goal_set = {(int(goal[0]), int(goal[1]))}
        else:
            goal_set = set(_cells(goal))
        for r, c in goal_set:
            if 0 <= r < h and 0 <= c < w:
                allowed[r, c] = True
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    rel = np.asarray([(r - r0, c - c0) for r, c in cells], dtype=np.int64)
    rel_list = [tuple(x) for x in rel.tolist()]
    step = max(1, int(step))

    def fits(R: int, C: int) -> bool:
        rr = rel[:, 0] + R
        cc = rel[:, 1] + C
        if rr.min() < 0 or cc.min() < 0 or rr.max() >= h or cc.max() >= w:
            return False
        return bool(allowed[rr, cc].all())
    start = (r0, c0)
    dist = {start: 0}
    prev: dict = {}
    q = deque([start])
    found = None
    dirs = (("UP", -1, 0), ("DOWN", 1, 0), ("LEFT", 0, -1), ("RIGHT", 0, 1))
    while q:
        R, C = q.popleft()
        if goal_set and any((R + y, C + x) in goal_set for y, x in rel_list):
            found = (R, C)
            break
        for name, dy, dx in dirs:
            ok = True
            for k in range(1, step + 1):
                if not fits(R + dy * k, C + dx * k):
                    ok = False
                    break
            if not ok:
                continue
            nxt = (R + dy * step, C + dx * step)
            if nxt in dist:
                continue
            dist[nxt] = dist[(R, C)] + 1
            prev[nxt] = ((R, C), name)
            q.append(nxt)
    return dist, prev, found


def _show_text(x: Any) -> str:
    if isinstance(x, dict) and "cells" in x:
        o = x if isinstance(x, Obj) else _as_obj(x)
        if not o["cells"]:
            return repr(o)
        r0, r1, c0, c1 = o["rect"]
        m = dict(zip(o["cells"], o["colors"]))
        rows = [",".join(str(m[(r, c)]) if (r, c) in m else "." for c in range(c0, c1)) for r in range(r0, r1)]
        return repr(o) + "\n" + "\n".join(rows)
    if isinstance(x, np.ndarray) and x.ndim == 2:
        if x.dtype == bool:
            return "\n".join(",".join("#" if v else "." for v in row) for row in x.tolist())
        return "\n".join(",".join(str(int(v)) for v in row) for row in x.tolist())
    if isinstance(x, (list, tuple)) and x and all(isinstance(e, dict) for e in x):
        return "\n\n".join(_show_text(e) for e in x)
    if isinstance(x, list) and x and isinstance(x[0], list):
        return _show_text(np.asarray(x))
    return repr(x)


def describe(o: Any) -> str:
    """describe(o) -> one line, e.g. "Orange+Blue 2x2 block at [3:5, 2:4]"."""
    o = _as_obj(o)
    if not o["cells"]:
        return "empty object"
    r0, r1, c0, c1 = o["rect"]
    hh, ww = r1 - r0, c1 - c0
    shape = (f"{hh}x{ww} block" if (hh, ww) != (1, 1) else "1 cell") if o["size"] == hh * ww else f"{o['size']}-cell shape in {hh}x{ww}"
    return f"{o['color_name']} {shape} at [{r0}:{r1}, {c0}:{c1}]"


def bands(board: Any) -> dict:
    """bands(board) -> {"rows": [(row, colour), ...], "cols": [(col, colour), ...], "inside": (r0, r1, c0, c1)}: the full
    rows and columns of one colour (borders, separators, HUD strips) and the area inside the outer ones."""
    b = _board_arr(board)
    H, W = b.shape
    rows = [(r, int(b[r, 0])) for r in range(H) if (b[r] == b[r, 0]).all()]
    cols = [(c, int(b[0, c])) for c in range(W) if (b[:, c] == b[0, c]).all()]
    rs = {r for r, _ in rows}
    cs = {c for c, _ in cols}
    r0, r1, c0, c1 = 0, H, 0, W
    while r0 < r1 and r0 in rs:
        r0 += 1
    while r1 > r0 and (r1 - 1) in rs:
        r1 -= 1
    while c0 < c1 and c0 in cs:
        c0 += 1
    while c1 > c0 and (c1 - 1) in cs:
        c1 -= 1
    return {"rows": rows, "cols": cols, "inside": (r0, r1, c0, c1)}


def group(objs: Any, by: str = "look") -> list[list]:
    """group(objs, by="look" | "shape" | "color" | "size") -> lists of objects that are alike, largest group first."""
    keyf = {"look": lambda o: o["hash"], "shape": lambda o: o["shape_hash"], "color": lambda o: o["color"],
            "size": lambda o: o["size"]}.get(str(by))
    if keyf is None:
        raise ValueError('by must be "look", "shape", "color" or "size"')
    groups: dict[Any, list] = {}
    for o in objs:
        groups.setdefault(keyf(_as_obj(o)), []).append(o)
    return sorted(groups.values(), key=lambda g: -len(g))


def _ray(b: np.ndarray, r: int, c: int, dr: int, dc: int, only: Any, exclude: Any) -> tuple:
    allowed = _filter_mask(b, only, exclude)
    h, w = b.shape
    if dr == 0 and dc == 0:
        raise ValueError("dr and dc cannot both be 0")
    n = 0
    y, x = int(r), int(c)
    while True:
        y += int(dr)
        x += int(dc)
        if not (0 <= y < h and 0 <= x < w):
            return n, None
        if allowed is not None and not allowed[y, x]:
            return n, (y, x, int(b[y, x]))
        n += 1


def _nearest(b: np.ndarray, r: int, c: int, only: Any, exclude: Any) -> tuple | None:
    m = _filter_mask(b, only, exclude)
    if m is None:
        raise ValueError("give only=[colours to look for] or exclude=[colours to skip]")
    pts = np.argwhere(m)
    if not len(pts):
        return None
    d = np.abs(pts[:, 0] - int(r)) + np.abs(pts[:, 1] - int(c))
    i = int(np.lexsort((pts[:, 1], pts[:, 0], d))[0])
    return int(pts[i, 0]), int(pts[i, 1]), int(d[i])


def click_cells(b: np.ndarray, mask: np.ndarray | None = None) -> list[tuple[int, int]]:
    """One cell per single-colour region (the region's cell nearest its centre), regions inside mask left out."""
    out = []
    grid = b.astype(np.int64, copy=True)
    if mask is not None:
        grid[mask] = -1  # never bridge two game objects through a same-colour counter
    for cells, _ in _flood_components(grid, None):
        if mask is not None and all(mask[r, c] for r, c in cells):
            continue
        rr = sum(r for r, _ in cells) / len(cells)
        cc = sum(c for _, c in cells) / len(cells)
        out.append(min(cells, key=lambda p: ((p[0] - rr) ** 2 + (p[1] - cc) ** 2, p)))
    return sorted(out)


def _rects_mask(rects: Any, shape: tuple) -> np.ndarray:
    m = np.zeros(tuple(shape[:2]), dtype=bool)
    for r0, r1, c0, c1 in rects or ():
        m[r0:r1, c0:c1] = True
    return m


def _runs(seq: list[int]) -> list[tuple[str, int]]:
    out: list[list] = []
    for v in seq:
        if out and out[-1][0] == v:
            out[-1][1] += 1
        else:
            out.append([v, 1])
    return [(_NAMES.get(v, str(v)), n) for v, n in out]


def _strip(board: np.ndarray, rect: tuple) -> list[int]:
    r0, r1, c0, c1 = rect
    sub = np.asarray(board)[r0:r1, c0:c1]
    if sub.size == 0:
        return []
    if (c1 - c0) >= (r1 - r0):
        return [int(Counter(sub[:, j].tolist()).most_common(1)[0][0]) for j in range(sub.shape[1])]
    return [int(Counter(sub[i, :].tolist()).most_common(1)[0][0]) for i in range(sub.shape[0])]


def _indicator_part(states: list, rect: tuple, *, names=None, strict=False) -> dict:
    names = _NAMES if names is None else names
    first, now = _strip(states[0], rect), _strip(states[-1], rect)
    changed_after = []
    deltas = []
    r0, r1, c0, c1 = rect
    for i in range(1, len(states)):
        a, b = np.asarray(states[i - 1])[r0:r1, c0:c1], np.asarray(states[i])[r0:r1, c0:c1]
        if a.shape == b.shape and not np.array_equal(a, b):
            changed_after.append(i)
            ca, cb = Counter(_strip(states[i - 1], rect)), Counter(_strip(states[i], rect))
            deltas.append({k: cb.get(k, 0) - ca.get(k, 0) for k in set(ca) | set(cb) if cb.get(k, 0) != ca.get(k, 0)})
    out: dict[str, Any] = {"rect": tuple(rect), "runs": _runs(now), "runs_at_start": _runs(first),
                           "actions": len(states) - 1, "changed_after": changed_after}
    if strict:
        # Actual pixels, not the majority-colour projection used for bar trends.
        out["counts"] = dict(Counter(map(int, np.asarray(states[-1])[r0:r1, c0:c1].ravel())))
        out["counts_at_start"] = dict(Counter(map(int, np.asarray(states[0])[r0:r1, c0:c1].ravel())))
        out["runs"] = [(names.get(v, str(v)), n) for v, n in _runs_ids(now)]
        out["runs_at_start"] = [(names.get(v, str(v)), n) for v, n in _runs_ids(first)]
    est = "not enough changes to see a pattern"
    if len(deltas) >= (3 if strict else 2):
        shrinking = None
        for k in deltas[0]:
            if deltas[0][k] < 0 and all(d.get(k) == deltas[0][k] for d in deltas):
                shrinking = k
        gaps = [b - a for a, b in zip([0] + changed_after[:-1], changed_after)][1:]
        if shrinking is None:
            est = "no colour shrinks by the same amount at every change: no estimate"
        elif gaps and len(set(gaps)) == 1:
            per = -deltas[0][shrinking]
            left = Counter(now).get(shrinking, 0)
            k = gaps[0]
            since = (len(states) - 1) - changed_after[-1]
            more = max(0, (left // per) * k - since) if per else 0
            est = (f"{names.get(shrinking, shrinking)} -{per} per change, "
                   + ("1 change per action" if k == 1 else f"1 change every {k} actions")
                   + f" -> about {more} more actions")
            if strict:
                est = "if the observed bar trend continues: " + est
                if since >= k:
                    est = "the latest action no longer follows the observed bar trend: no estimate"
        else:
            est = "changes are not evenly spaced: no estimate"
    out["if_this_continues"] = est
    return out


def _runs_ids(seq):
    out = []
    for value in seq:
        if out and out[-1][0] == value:
            out[-1][1] += 1
        else:
            out.append([value, 1])
    return out


def indicator_facts(states, rects, *, names=None, strict=False, uncertain=False):
    if not rects or not len(states):
        return None
    parts = [_indicator_part(states, r, names=names, strict=strict) for r in rects]
    if uncertain:
        for part in parts:
            part["if_this_continues"] = "no action estimate: an observation/action count is uncertain"
    if len(parts) == 1:
        result = dict(parts[0])
        result["rects"] = [result.pop("rect")]
        return result
    return {"rects": list(rects), "parts": parts}


LIB_SUMMARY = """lib: generic grid tools, called as lib.name(...) in run and inside next_state. They only compute and return new values (act() is the only thing that touches the game). They are generic tools, not rules of this game: a game may move, grow, rotate, recolour or block things in its own way, and a library choice (what counts as one object, how scaling works, what reachable() treats as passable) may differ from the game. Use them to express what you observed and check against real transitions.
Geometric values use Obj: a read-only dict (o["cells"] or o.cells) with cells (sorted (row, col) tuples), colors (one per cell), color (the most common), color_name, size, rect (r0, r1, c0, c1; half-open slice bounds), hash (shape + colours, position-free) and shape_hash (shape only, any rotation or mirror). Obj is a geometry container, not a semantic label. o == p: same look in the same place. o.copy() is a plain editable dict; plain dicts with cells and colors (or color) are accepted everywhere.
Colour filters everywhere: only=[...] = just these colours, exclude=[...] = every colour except these (never both).
 Extracting components (measurements, not interpreted game objects)
  extract_components(board=None, only=None, exclude=None) -> [Obj]  default: 4-connected equal-colour regions, including background/indicator. With only/exclude (never both), edge-connected allowed cells join EVEN across colours. only=[] selects nothing; exclude=[] joins all cells. IDs are per extraction, not tracked identities. All results include id and is_remaining_actions_indicator; only unfiltered results also include adjacent, children, is_background (largest, not necessarily floor), crop. Use only=[colour] for geometry; a multi-colour filter is a grouping hypothesis to check. lib.help("extract_components") has details.
  objects(...) -> compatibility alias for extract_components; unchanged behaviour
  object_at(board, r, c, only=None, exclude=None) -> the Obj containing cell (r, c), or None
  find(board, o, min_score=1.0, rotate=False, flip=False, match_colors=True) -> [Obj]  every place o's look appears, in reading order; .score = share of matching cells; match_colors=False: the same pattern in any colours. Fast enough inside next_state
 Comparing
  same_look(a, b, rotate=False, flip=False), same_shape(a, b, rotate=True, flip=True) -> bool  (anywhere on the board; same_shape ignores colours)
  board_key(board) -> short str for a whole board, the marked indicator ignored (for sets, dicts, loop checks; on arrays a == b compares cell by cell: use np.array_equal or board_key)
 Changed copies of an object (nothing is drawn)
  moved(o, dr, dc), placed(o, r, c)  by an offset / with the top-left corner at (r, c)
  rotated(o, quarter_turns=1), mirrored(o, axis="left-right" or "up-down")  clockwise / flipped, top-left corner kept (the change report uses the same words)
  scaled(x, k)  an Obj or a 2-D array; k a whole number, or 1/n when every n x n block is one colour (otherwise an error)
  recolored(o, {old: new}), merged(a, b)  (merged: b wins where both have a cell)
 Editing a board (each returns a new board; the board passed in is unchanged)
  draw(board, objs), erase(board, objs, fill), move_on_board(board, o, dr, dc, fill)  (move_on_board: erase o, draw it moved)
  hits(board, o, colors) -> bool  does o lie on any cell of these colours (cells off the board count as hits): the wall check
  object_mask(o) -> board-sized True/False array of o's cells
 Relations
  overlap(a, b), touching(a, b, diagonal=False), contains(a, b) -> bool  (contains: b lies in a's holes)
  holes(o) -> the cells o encloses;  around(board, o, diagonal=False) -> {(r, c): colour} of the cells directly next to o
  gap(a, b) -> empty cells between their nearest cells (0 touching, -1 overlapping);  offset(a, b) -> (dr, dc) between top-left corners
 Looking
  show(x)  prints a board, Obj, mask or list in the harness's comma-separated format ("." = not part of the object)
  describe(o) -> one line;  bands(board) -> full one-colour rows/cols and the area inside them;  group(objs, by="look"|"shape"|"color"|"size")
  help(name=None)  prints this list, or one function's full description with an example
 Navigating the board as it is now (doors that open, pushed blocks etc. are not modelled)
  ray(board, r, c, dr, dc, only=None, exclude=None) -> (cells walked, (r, c, colour) of the cell that stopped it, or None at the edge)
  nearest(board, r, c, only=None, exclude=None) -> (r, c, distance) of the closest matching cell, or None
  reachable(board, o, step=1, only=None, exclude=None) -> top-left positions o can reach moving step cells at a time (every cell passed over must be allowed)
  walk_distances(board, o, step=1, only=None, exclude=None) -> {position: moves};  route(board, o, goal, step=1, only=None, exclude=None) -> ["UP", ...] or None (assumes UP/DOWN/LEFT/RIGHT move o by step)
 The real history of this level (run)
  frame_diff(before=None, after=None, exclude=<object view>) -> dict: changed_cells, changed_outside_indicator, moved, rotated, recolored, reshaped, appeared, disappeared, text (the feedback's change lines, uncapped); default: the last action (then also anim_frames, anim_transient_cells and a text line when an animation played)
  anim_diff(cap=40) -> the last action's animation step by step
  heat() -> number of changes per cell;  when(r, c) -> [(step, action)];  seen_before(board) -> indices of states equal to this board
  track(o) -> [(step, action, (r, c) or None)]: where o is in every state of this attempt
  action_effects(action=None) -> per action (CLICK per clicked colour): tries, changes, typical effects
  indicator() -> facts about the marked remaining-actions indicator (colour runs now and at the start, after which actions it changed), or None
 Exploring
  click_candidates(board=None) -> one (r, c) per single-colour region (indicator left out): the list path uses
  untried(board=None) -> actions and click candidates never tried on this level (with a board: never tried from that board)"""


def make_lib(ctx: dict) -> "_Lib":
    """The `lib` namespace for one sandbox / model-check context. ctx: get_states() -> list of boards, get_history() ->
    list of transition dicts, get_actions() -> actions of the attempt, rects, view (exclude list or None), names,
    anim_frames (list), action_labels, printer, in_model (bool: next_state is running)."""
    names = ctx.get("names")
    if names:
        global _NAMES
        _NAMES = {int(k): str(v) for k, v in names.items()}
    rects = [tuple(int(v) for v in r) for r in (ctx.get("rects") or [])]
    printer = ctx.get("printer") or (lambda *a, **k: None)
    in_model = bool(ctx.get("in_model"))

    def _states() -> list:
        return list(ctx["get_states"]())

    def _cur() -> np.ndarray:
        st = _states()
        if not st:
            raise ValueError("no board available")
        return np.asarray(st[-1])

    def _b(board: Any, fname: str) -> np.ndarray:
        if board is None:
            if in_model:
                raise ValueError(f"inside next_state pass the board explicitly: lib.{fname}(board, ...) "
                                 "(the default is the real current board, not the one next_state is given)")
            return _cur()
        return _board_arr(board)

    def extract_components(board=None, only=None, exclude=None):
        b = _b(board, "extract_components")
        if only is None and exclude is None:
            result = list(_segment_objs(np.asarray(b), rects))
            if board is None and not in_model and ctx.get("get_object_refs"):
                snapshot = ctx["get_object_refs"]()
                refs = {tuple(map(tuple, e["cells"])): e["id"] for e in snapshot.get("components", [])
                        if e["id"].startswith("C")}
                result = [Obj({**o.copy(), "ref": refs.get(tuple(map(tuple, o.cells)))}) for o in result]
            return result
        return _components_objs(b.astype(np.int64, copy=False), only, exclude, rects)

    def object_at(board, r, c, only=None, exclude=None):
        return _object_at(_b(board, "object_at"), r, c, only, exclude)

    def find(board, o, min_score=1.0, rotate=False, flip=False, match_colors=True):
        return _find(_b(board, "find"), o, min_score, rotate, flip, match_colors)

    def board_key(board):
        a = np.array(_board_arr(board), dtype=np.uint8, copy=True)
        a[_rects_mask(rects, a.shape)] = 255
        return hashlib.sha1(a.tobytes()).hexdigest()[:16]

    def show(x):
        printer(_show_text(x))

    def help(name=None):   # noqa: A001 - lib.help
        if name is None:
            printer(LIB_SUMMARY)
            return
        f = getattr(lib, str(name), None)
        if f is None or not callable(f):
            printer(f"lib has no function {name!r}")
            return
        printer((f.__doc__ or str(name)).strip())

    def ray(board, r, c, dr, dc, only=None, exclude=None):
        return _ray(_b(board, "ray"), r, c, dr, dc, only, exclude)

    def nearest(board, r, c, only=None, exclude=None):
        return _nearest(_b(board, "nearest"), r, c, only, exclude)

    def reachable(board, o, step=1, only=None, exclude=None):
        dist, _, _ = _walk(_b(board, "reachable"), o, step, only, exclude)
        return sorted(dist)

    def walk_distances(board, o, step=1, only=None, exclude=None):
        dist, _, _ = _walk(_b(board, "walk_distances"), o, step, only, exclude)
        return dict(sorted(dist.items()))

    def route(board, o, goal, step=1, only=None, exclude=None):
        dist, prev, found = _walk(_b(board, "route"), o, step, only, exclude, goal=goal)
        if found is None:
            return None
        acts = []
        cur = found
        while cur in prev:
            cur, a = prev[cur]
            acts.append(a)
        return acts[::-1]

    def _hist() -> list:
        return list(ctx["get_history"]())

    def frame_diff(before=None, after=None, exclude="view"):
        st = _states()
        last_action = before is None and after is None and len(st) >= 2
        if before is None and after is None:
            if len(st) >= 2:
                before, after = st[-2], st[-1]
            else:
                before = after = st[-1]
        elif before is None:
            before = st[-2] if len(st) >= 2 else after
        elif after is None:
            after = st[-1]
        ex = ctx.get("view") if (isinstance(exclude, str) and exclude == "view") else exclude
        d, lines = change_report(before, after, rects, ex or None, cap=None)
        d = dict(d)
        lines = list(lines)
        if last_action:   # v2.11c: the default (the last action) also says whether an animation played during it
            frames = list(ctx.get("anim_frames") or [])
            d["anim_frames"], d["anim_transient_cells"] = len(frames), 0
            if frames:
                t = _FD.transient_summary(before, frames, after, indicator_rect=rects or None) or {}
                tr = int(t.get("transient") or 0)
                d["anim_transient_cells"] = tr
                head = ""
                if not d.get("changed_cells"):
                    head = "board unchanged; "
                elif not d.get("changed_outside_indicator"):
                    head = ("board unchanged apart from the remaining-actions indicator; " if rects else "board unchanged; ")
                lines.append(f"{head}an animation played during the last action: {len(frames)} frame(s)"
                             + (f", {tr} cell(s) changed and changed back within {_OBJ.rect_text(t['bbox'])}"
                                if tr and t.get("bbox") else "")
                             + " (lib.anim_diff() lists it frame by frame)")
        d["text"] = lines
        return d

    def anim_diff(cap=40):
        frames = list(ctx.get("anim_frames") or [])
        if not frames:
            return ["the last executed action did not animate"]
        st = _states()
        before, after = (st[-2], st[-1]) if len(st) >= 2 else (frames[0], st[-1])
        view = ctx.get("view") or None
        return _FD.anim_steps(before, frames, after, lambda x: diff_objects(np.asarray(x), rects, view, None),
                              indicator_rect=rects or None, cap=int(cap))

    def heat():
        st = _states()
        shape = np.asarray(st[-1]).shape if st else (64, 64)
        out = np.zeros(shape, dtype=np.int64)
        for h in _hist():
            b, a = h.get("before"), h.get("after")
            if b is not None and a is not None and np.asarray(b).shape == shape and np.asarray(a).shape == shape:
                out += (np.asarray(b) != np.asarray(a))
        return out

    def when(r, c):
        out = []
        for i, h in enumerate(_hist()):
            b, a = h.get("before"), h.get("after")
            if b is not None and a is not None and np.asarray(b).shape == np.asarray(a).shape and \
                    int(np.asarray(b)[r, c]) != int(np.asarray(a)[r, c]):
                out.append((i, h.get("action")))
        return out

    def seen_before(board):
        k = board_key(board)
        return [i for i, s in enumerate(_states()) if board_key(s) == k]

    def track(o):
        st = _states()
        acts = list(ctx.get("get_actions", lambda: [])())
        o = _as_obj(o)
        prev = o["rect"][0], o["rect"][2]
        out = []
        for i, s in enumerate(st):
            ms = _find(_board_arr(s), o, 1.0, False, False, True)
            pos = None
            if ms:
                best = min(ms, key=lambda m: (abs(m["rect"][0] - prev[0]) + abs(m["rect"][2] - prev[1]), m["rect"][0], m["rect"][2]))
                pos = (best["rect"][0], best["rect"][2])
                prev = pos
            out.append((i, acts[i - 1] if 0 < i <= len(acts) else None, pos))
        return out

    def _label(h: dict) -> str:
        a = str(h.get("action") or "")
        parts = a.split()
        if parts and parts[0].upper() == "CLICK" and len(parts) >= 3:
            try:
                v = int(np.asarray(h["before"])[int(parts[1]), int(parts[2])])
                return f"CLICK on {_NAMES.get(v, v)}"
            except Exception:
                return "CLICK"
        return a

    def action_effects(action=None):
        groups: dict[str, list] = {}
        for h in _hist():
            groups.setdefault(_label(h), []).append(h)
        out = {}
        for key, hs in groups.items():
            if action is not None and str(action).upper() not in (key.upper(), key.split()[0].upper()):
                continue
            changed = [h for h in hs if (h.get("changed_outside_indicator") if h.get("changed_outside_indicator") is not None
                                          else h.get("changed_cells")) or 0]
            effects: Counter = Counter()
            for h in hs[-8:]:
                b, a = h.get("before"), h.get("after")
                if b is None or a is None:
                    continue
                try:
                    d, _ = change_report(b, a, rects, ctx.get("view") or None, cap=0)
                    effects[short_summary(d)] += 1
                except Exception:
                    pass
            out[key] = {"tries": len(hs), "changed": len(changed), "no_change": len(hs) - len(changed),
                        "game_over": sum(1 for h in hs if h.get("outcome") == "game_over"),
                        "level_complete": sum(1 for h in hs if h.get("outcome") in ("level_complete", "win")),
                        "effects_of_last_tries": effects.most_common(4)}
        return out

    def indicator():
        return indicator_facts(_states(), rects)

    def click_candidates(board=None):
        b = _b(board, "click_candidates")
        return click_cells(np.asarray(b), _rects_mask(rects, np.asarray(b).shape) if rects else None)

    def untried(board=None):
        b = _cur() if board is None else _board_arr(board)
        hs = _hist()
        if board is not None:
            k = board_key(b)
            hs = [h for h in hs if h.get("before") is not None and board_key(h["before"]) == k]
        labels = [str(x) for x in (ctx.get("action_labels") or [])]
        tried = {str(h.get("action") or "").split()[0].upper() for h in hs if h.get("action")}
        acts = [a for a in labels if a not in ("RESET", "CLICK") and a.upper() not in tried]
        clicks: list = []
        n_cand = 0
        if "CLICK" in labels:
            cand = click_cells(np.asarray(b), _rects_mask(rects, np.asarray(b).shape) if rects else None)
            n_cand = len(cand)
            clicked = set()
            for h in hs:
                p = str(h.get("action") or "").split()
                if p and p[0].upper() == "CLICK" and len(p) >= 3:
                    try:
                        clicked.add((int(p[1]), int(p[2])))
                    except ValueError:
                        pass
            for r, c in cand:
                reg = _object_at(np.asarray(b), r, c, None, None)
                if reg is None or not any(p in clicked for p in reg["cells"]):
                    clicks.append((r, c))
        out = {"actions": acts, "clicks": clicks, "click_candidates": n_cand}
        return out

    def _reference(method, *args, **kwargs):
        if ctx.get("in_model"):
            raise ValueError("live object references are for run only; use the supplied board inside next_state")
        callback = ctx.get("reference_request")
        if callback is None:
            raise ValueError("object references are unavailable here")
        return callback(method, args, kwargs)

    def components(colors=None, size=None, detail=False):
        """Compact clickable C IDs, colour counts and rectangles; optional exact filters. detail=True adds cells/values."""
        return _reference("components", colors=colors, size=size, detail=detail)

    def resolve(ref):
        """Follow an old C/G ID or saved name to its unique current selection; raises on ambiguity/stale scope."""
        return _reference("resolve", ref)

    def remember(name, refs, reason=""):
        """Save one ID/name or a list of parts as a reusable object reference. Available to act() immediately."""
        return _reference("remember", name, refs, reason=reason)

    def forget(name):
        """Remove a saved name."""
        return _reference("forget", name)

    def references():
        """Saved names, their current resolution and the agent's grouping reasons."""
        return _reference("references")

    def relate(source, relation, target, evidence="", follow=False):
        """Record an agent assertion. Only identity links with follow=True and evidence can redirect old clicks."""
        return _reference("relate", source, relation, target, evidence=evidence, follow=follow)

    def relations(ref, relation=None):
        """Get graph edges and current geometric relations; every edge records provenance and scope."""
        return _reference("relations", ref, relation=relation)

    fns = {
        "extract_components": extract_components, "objects": extract_components,  # compatibility alias; same semantics
        "object_at": object_at, "find": find, "same_look": same_look, "same_shape": same_shape,
        "board_key": board_key, "moved": moved, "placed": placed, "rotated": rotated, "mirrored": mirrored,
        "scaled": scaled, "recolored": recolored, "merged": merged, "draw": draw, "erase": erase,
        "move_on_board": move_on_board, "hits": hits, "object_mask": object_mask, "overlap": overlap,
        "touching": touching, "contains": contains, "holes": holes, "around": around, "gap": gap, "offset": offset,
        "show": show, "describe": describe, "bands": bands, "group": group, "help": help, "ray": ray,
        "nearest": nearest, "reachable": reachable, "walk_distances": walk_distances, "route": route,
        "frame_diff": frame_diff, "anim_diff": anim_diff, "heat": heat, "when": when, "seen_before": seen_before,
        "track": track, "action_effects": action_effects, "indicator": indicator,
        "click_candidates": click_candidates, "untried": untried,
        "components": components, "resolve": resolve, "remember": remember, "forget": forget, "references": references,
        "relate": relate, "relations": relations,
    }
    _DOCS = {
        "extract_components": "extract_components(board=None, only=None, exclude=None) -> [Obj]\n  Pure segmentation, not semantic object recognition or tracking. board defaults to current_board in run; pass board explicitly inside next_state. Uses 4-connectivity (shared edges, never diagonals). With no filters: one component per connected equal-colour region, including background and the marked indicator. With only=[...] or exclude=[...] (never both): all edge-connected allowed cells join, EVEN across different colours. only=[] returns nothing; exclude=[] allows every cell, usually joining the whole board.\n  Every result has cells, colors, color (modal), color_name, size, rect (r0,r1,c0,c1; half-open), hash, shape_hash, id, rotation, symmetry, is_remaining_actions_indicator. Only unfiltered extraction additionally supplies adjacent, children, is_background and crop. IDs are local to this extraction, not persistent identities; hashes describe appearance, not roles. is_background labels the largest component, not a learned floor. Indicator components are marked, not removed.\n  Use unfiltered extraction to inspect colour regions; only=[3] to measure one colour's geometry; only=[3,4] or exclude=[0,5] to test a connected multi-colour grouping. Filters can wrongly merge touching entities or split a real entity: verify important geometry with cells, lib.holes, lib.touching or board slices, and behavioural grouping with observed transitions. lib.objects is a compatibility alias with identical behaviour.",
        "object_at": "object_at(board, r, c, only=None, exclude=None) -> Obj or None\n  The component containing cell (r, c): by default its 4-connected single-colour region; with filters, all edge-connected allowed cells, possibly several colours. This does not establish a game object.\n  Example: p = lib.object_at(board, 3, 2, exclude=[0, 5])",
        "find": "find(board, o, min_score=1.0, rotate=False, flip=False, match_colors=True) -> [Obj]\n  Every place where o's look appears, top to bottom then left to right; each result has the board's colours at its cells and .score (share of matching cells). min_score < 1 allows partial matches; rotate/flip also accept turned/mirrored copies; match_colors=False finds the same pattern in any colours (exact only).\n  Example: p = lib.find(board, store['player'])[0]",
        "frame_diff": "frame_diff(before=None, after=None, exclude=<object view>) -> dict\n  The component-level change between two boards (default: the last action): changed_cells, changed_outside_indicator, changed_rects, moved, rotated, recolored, reshaped, appeared, disappeared, and text = the same lines the feedback shows, uncapped. With no boards given (the last action) it also has anim_frames and anim_transient_cells, and text gets a line when an animation played (cells can change and change back during it, invisible in the two boards; lib.anim_diff() lists it). exclude defaults to the object view set with set_object_view; pass exclude=None for single-colour components.\n  The component pairing is a best guess: identical twins can be paired wrongly, and a thing that moves and changes shape at once shows as disappeared + appeared. changed_cells is exact.",
        "route": "route(board, o, goal, step=1, only=None, exclude=None) -> list of actions or None\n  Shortest UP/DOWN/LEFT/RIGHT sequence that makes o cover a cell of goal (an object, a list of cells or one (r, c)) on the board as it is now. Assumes each action moves o by step cells in that direction and that o may stand only on allowed colours (its own and the goal's cells are always allowed). Check that this is how your game moves before relying on it.\n  Example: lib.route(board, p, key, step=5, only=[0])",
    }
    for k, f in fns.items():
        doc = _DOCS.get(k)
        if doc:
            f.__doc__ = doc
        elif not f.__doc__:
            line = next((ln.strip() for ln in LIB_SUMMARY.split("\n") if ln.strip().startswith(k + "(")), k)
            f.__doc__ = line
    lib = _Lib(fns)
    return lib


class _Lib:
    """The lib namespace: lib.find(...), lib.moved(...), ... (lib.help() lists them)."""

    def __init__(self, fns: dict) -> None:
        object.__setattr__(self, "_fns", dict(fns))

    def __getattr__(self, name: str) -> Any:
        fns = object.__getattribute__(self, "_fns")
        if name in fns:
            return fns[name]
        close = difflib.get_close_matches(name, list(fns), n=3)
        hint = OLD_NAMES.get(name)
        msg = f"lib has no function {name!r}"
        if hint:
            msg += f" (use {hint})"
        elif close:
            msg += f"; similar: {', '.join(close)}"
        raise AttributeError(msg + ". lib.help() lists them all.")

    def __setattr__(self, name: str, value: Any) -> None:
        raise TypeError("lib is read-only")

    def __dir__(self) -> list:
        return sorted(object.__getattribute__(self, "_fns"))

    def __repr__(self) -> str:
        return "<lib: " + ", ".join(sorted(object.__getattribute__(self, "_fns"))) + ">"


# names of the v2.10 helpers (and draft names) -> their lib replacement, for error hints
OLD_NAMES = {
    "shifted_object": "lib.moved", "rotated_object": "lib.rotated", "mirrored_object": "lib.mirrored",
    "draw_objects": "lib.draw", "erase_objects": "lib.erase", "move_object": "lib.move_on_board",
    "extract": "lib.object_at", "extract_all": "lib.extract_components", "search": "lib.find", "move": "lib.moved",
    "place": "lib.placed", "rotate": "lib.rotated", "flip_lr": 'lib.mirrored(o, "left-right")',
    "flip_ud": 'lib.mirrored(o, "up-down")', "clear": "lib.erase", "diff": "lib.frame_diff", "key": "lib.board_key",
    "same": "o == p or lib.same_look", "counter": "lib.indicator", "click_targets": "lib.click_candidates",
    "direction": "lib.offset", "distance": "lib.gap", "distances": "lib.walk_distances", "effects": "lib.action_effects",
    "seen": "lib.seen_before", "obj": "lib.extract_components", "lib_help": "lib.help",
}
# bare names a run snippet / code file might still use -> the lib function to call instead
BARE_HINTS = {
    "objects": "lib.extract_components", "frame_diff": "lib.frame_diff", "anim_diff": "lib.anim_diff",
    "shifted_object": "lib.moved", "rotated_object": "lib.rotated", "mirrored_object": "lib.mirrored",
    "draw_objects": "lib.draw", "erase_objects": "lib.erase", "move_object": "lib.move_on_board",
    "obj": "lib.extract_components", "indicator_rect": "indicator_rects",
}
for _n in ("extract_components", "objects", "object_at", "find", "same_look", "same_shape", "board_key", "moved", "placed", "rotated", "mirrored",
           "scaled", "recolored", "merged", "draw", "erase", "move_on_board", "hits", "object_mask", "overlap", "touching",
           "contains", "holes", "around", "gap", "offset", "show", "describe", "bands", "group", "ray", "nearest",
           "reachable", "walk_distances", "route", "heat", "when", "seen_before", "track", "action_effects", "indicator",
           "click_candidates", "untried"):
    BARE_HINTS.setdefault(_n, "lib." + _n)

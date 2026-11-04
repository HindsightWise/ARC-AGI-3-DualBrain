"""Object segmentation of a 2D colour grid (stdlib + numpy only).

The same source is used by the harness process and by the `run` sandbox worker, so it
must not import anything from the agent package.

An *object* is a 4-connected component of equal colour.  Each object is a plain dict:

  id        index in reading order of the top-most-left-most cell (0-based)
  color     integer colour value (0-15)
  color_name  colour name when a name map is supplied, else str(color)
  size      number of cells
  cells     list of (row, col) tuples (always present)
  rect      [row_start, row_end, col_start, col_end]  (half-open, NumPy slice semantics,
            i.e. board[row_start:row_end, col_start:col_end] is the bounding box)
  crop      the bounding box cut from the board as CSV text (only when size <= CROP_MAX_CELLS),
            otherwise None
  outline   clockwise list of [row, col] corner cells of the outer contour
  hash      16-hex signature of colour + exact shape, translation-invariant
  shape_hash  16-hex signature of the shape only, invariant to translation, 90-degree
            rotations and mirroring (colour-free)
  rotation  degrees (0/90/180/270) to rotate the canonical orientation clockwise to obtain
            this object; None when the shape is 4-fold rotationally symmetric
  symmetry  {"rotational": 1|2|4, "mirror_h": bool, "mirror_v": bool}
  children  ids of objects fully enclosed by this object
  adjacent  ids of objects sharing an edge with this object
  is_background  True for the largest object (by cell count)
  is_remaining_actions_indicator  True when every cell lies inside the marked indicator rect
"""
from __future__ import annotations

import hashlib
from typing import Any, Iterable

import numpy as np



def norm_rects(rects):
    """0..n indicator rectangles [r0,r1,c0,c1) as a tuple of 4-tuples; accepts None, one rect or a list of rects.
    (Duplicated in board.py: this module is also exec'd standalone inside the run sandbox.)"""
    if rects is None:
        return ()
    rects = list(rects)
    if not rects:
        return ()
    if len(rects) == 4 and all(isinstance(x, (int, np.integer)) for x in rects):
        return (tuple(int(x) for x in rects),)
    return tuple(tuple(int(x) for x in r) for r in rects)


def rects_mask(rects, shape):
    m = np.zeros(tuple(shape[:2]), dtype=bool)
    for r0, r1, c0, c1 in norm_rects(rects):
        m[r0:r1, c0:c1] = True
    return m

CROP_MAX_CELLS = 100

_ORTH = ((-1, 0), (1, 0), (0, -1), (0, 1))
# clockwise Moore-neighbour offsets, starting at NW
_CW = ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))
_CW_INDEX = {off: i for i, off in enumerate(_CW)}


def _as_grid(board: Any) -> np.ndarray:
    arr = np.asarray(board)
    if arr.ndim != 2:
        raise ValueError(f"board must be 2-D, got shape {arr.shape}")
    return arr.astype(np.int64, copy=False)


# ----------------------------------------------------------------------------- shapes

def _normalize(cells: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    cells = list(cells)
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _rot90(cells: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    # clockwise rotation: (r, c) -> (c, -r)
    return _normalize((c, -r) for r, c in cells)


def _mirror_h(cells: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    # left-right flip
    return _normalize((r, -c) for r, c in cells)


def _mirror_v(cells: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    # top-bottom flip
    return _normalize((-r, c) for r, c in cells)


def _digest(payload: Any) -> str:
    return hashlib.sha1(repr(payload).encode()).hexdigest()[:16]


def shape_signature(cells: Iterable[tuple[int, int]], color: int) -> dict[str, Any]:
    """Colour+shape hash, D4-canonical shape hash, rotation and symmetry flags."""
    norm = _normalize(cells)
    rotations = [norm]
    for _ in range(3):
        rotations.append(_rot90(rotations[-1]))
    rot_keys = [repr(v) for v in rotations]
    distinct_rot = len(set(rot_keys))
    rotational = {1: 4, 2: 2, 4: 1}.get(distinct_rot, 1)
    # reference orientation: lexicographically smallest rotation (colour-free)
    k = min(range(4), key=lambda i: rot_keys[i])
    rotation: int | None = ((4 - k) % 4) * 90
    if rotational == 4:
        rotation = None
    elif rotational == 2:
        rotation = rotation % 180
    mirrored = [_mirror_h(v) for v in rotations]
    all_keys = rot_keys + [repr(v) for v in mirrored]
    shape_hash = _digest(min(all_keys))
    return {
        "hash": _digest((int(color), norm)),
        "shape_hash": shape_hash,
        "rotation": rotation,
        "symmetry": {
            "rotational": rotational,
            "mirror_h": _mirror_h(norm) == norm,
            "mirror_v": _mirror_v(norm) == norm,
        },
    }


# ----------------------------------------------------------------------------- contour

def _trace_outline(cells: set[tuple[int, int]], start: tuple[int, int]) -> list[tuple[int, int]]:
    """Moore-neighbour trace of the outer perimeter, clockwise, reduced to corner cells."""
    if len(cells) == 1:
        return [start]
    contour = [start]
    b = start
    prev = (start[0], start[1] - 1)
    second = None
    for _ in range(8 * len(cells) + 16):
        idx = _CW_INDEX[(prev[0] - b[0], prev[1] - b[1])]
        nxt = None
        new_prev = prev
        for k in range(1, 9):
            off = _CW[(idx + k) % 8]
            cand = (b[0] + off[0], b[1] + off[1])
            if cand in cells:
                nxt = cand
                back = _CW[(idx + k - 1) % 8]
                new_prev = (b[0] + back[0], b[1] + back[1])
                break
        if nxt is None:
            break
        if second is None:
            second = nxt
        elif b == start and nxt == second:
            break
        contour.append(nxt)
        prev, b = new_prev, nxt
    if len(contour) > 1 and contour[-1] == contour[0]:
        contour.pop()
    if len(contour) <= 2:
        return contour
    m = len(contour)
    corners = []
    for i in range(m):
        p, cur, n = contour[i - 1], contour[i], contour[(i + 1) % m]
        if (cur[0] - p[0], cur[1] - p[1]) != (n[0] - cur[0], n[1] - cur[1]):
            corners.append(cur)
    return corners or contour[:1]


# ----------------------------------------------------------------------------- segmentation

def _components(grid: np.ndarray, zones=None) -> tuple[np.ndarray, list[dict[str, Any]]]:
    h, w = grid.shape
    cid_grid = [[-1] * w for _ in range(h)]
    comps: list[dict[str, Any]] = []
    # Different zone labels separate marked bars even when their colours touch game objects.
    g = grid.tolist() if zones is None else (grid.astype(np.int64) * (int(zones.max()) + 1) + zones).tolist()
    for sr in range(h):
        row = g[sr]
        for sc in range(w):
            if cid_grid[sr][sc] != -1:
                continue
            value = row[sc]
            cid = len(comps)
            cells: list[tuple[int, int]] = []
            stack = [(sr, sc)]
            cid_grid[sr][sc] = cid
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                if r > 0 and cid_grid[r - 1][c] == -1 and g[r - 1][c] == value:
                    cid_grid[r - 1][c] = cid; stack.append((r - 1, c))
                if r + 1 < h and cid_grid[r + 1][c] == -1 and g[r + 1][c] == value:
                    cid_grid[r + 1][c] = cid; stack.append((r + 1, c))
                if c > 0 and cid_grid[r][c - 1] == -1 and g[r][c - 1] == value:
                    cid_grid[r][c - 1] = cid; stack.append((r, c - 1))
                if c + 1 < w and cid_grid[r][c + 1] == -1 and g[r][c + 1] == value:
                    cid_grid[r][c + 1] = cid; stack.append((r, c + 1))
            comps.append({"value": int(grid[sr, sc]), "cells": cells, "start": (sr, sc)})
    return np.asarray(cid_grid, dtype=np.int32), comps


def _adjacency(comp_id: np.ndarray, n: int) -> list[set[int]]:
    adj: list[set[int]] = [set() for _ in range(n)]
    a, b = comp_id[:-1, :], comp_id[1:, :]
    mask = a != b
    for x, y in zip(a[mask].tolist(), b[mask].tolist()):
        adj[x].add(y); adj[y].add(x)
    a, b = comp_id[:, :-1], comp_id[:, 1:]
    mask = a != b
    for x, y in zip(a[mask].tolist(), b[mask].tolist()):
        adj[x].add(y); adj[y].add(x)
    return adj


def _containment(comp_id: np.ndarray, comps: list[dict[str, Any]], adj: list[set[int]]) -> list[list[int]]:
    """children[b] = ids directly enclosed by b (innermost encloser wins)."""
    h, w = comp_id.shape
    n = len(comps)
    touches_border = [False] * n
    for cid in set(comp_id[0, :].tolist()) | set(comp_id[-1, :].tolist()) | set(comp_id[:, 0].tolist()) | set(comp_id[:, -1].tolist()):
        touches_border[cid] = True
    enclosers: list[set[int]] = [set() for _ in range(n)]
    # Only components that do not touch the border can be enclosed; only components with
    # at least one such neighbour (transitively) can enclose.  Flood the complement of b
    # from the border; everything unreached is enclosed by b.
    candidates = [b for b in range(n) if any(not touches_border[a] for a in adj[b])]
    cid_list = comp_id.tolist()
    inner = [a for a in range(n) if not touches_border[a]]
    for b in candidates:
        reached = [[v == b for v in row] for row in cid_list]  # b's own cells are walls
        stack: list[tuple[int, int]] = []
        for r in range(h):
            for c in (0, w - 1):
                if not reached[r][c]:
                    reached[r][c] = True; stack.append((r, c))
        for c in range(w):
            for r in (0, h - 1):
                if not reached[r][c]:
                    reached[r][c] = True; stack.append((r, c))
        while stack:
            r, c = stack.pop()
            if r > 0 and not reached[r - 1][c]:
                reached[r - 1][c] = True; stack.append((r - 1, c))
            if r + 1 < h and not reached[r + 1][c]:
                reached[r + 1][c] = True; stack.append((r + 1, c))
            if c > 0 and not reached[r][c - 1]:
                reached[r][c - 1] = True; stack.append((r, c - 1))
            if c + 1 < w and not reached[r][c + 1]:
                reached[r][c + 1] = True; stack.append((r, c + 1))
        for a in inner:
            if a == b:
                continue
            ar, ac = comps[a]["start"]
            if not reached[ar][ac]:
                enclosers[a].add(b)
    children: list[list[int]] = [[] for _ in range(n)]
    for a in range(n):
        if enclosers[a]:
            parent = max(enclosers[a], key=lambda e: (len(enclosers[e]), -e))
            children[parent].append(a)
    for lst in children:
        lst.sort()
    return children


def crop_csv(board: Any, rect: list[int]) -> str:
    grid = _as_grid(board)
    r0, r1, c0, c1 = rect
    sub = grid[r0:r1, c0:c1]
    return "\n".join(",".join(str(int(v)) for v in row) for row in sub.tolist())


def segment(
    board: Any,
    indicator_rect: Any = None,   # one rect [r0,r1,c0,c1] or a list of rects
    color_names: dict[int, str] | None = None,
    include_background: bool = True,
) -> list[dict[str, Any]]:
    """Segment a board into objects (see module docstring)."""
    grid = _as_grid(board)
    h, w = grid.shape
    inds = norm_rects(indicator_rect)
    zones = np.zeros(grid.shape, dtype=np.int16) if inds else None
    for zone, (r0, r1, c0, c1) in enumerate(inds, 1):
        zones[r0:r1, c0:c1] = zone
    comp_id, comps = _components(grid, zones)
    n = len(comps)
    adj = _adjacency(comp_id, n)
    children = _containment(comp_id, comps, adj)
    # background = the largest component (ties: lowest id); frames/walls touching the border
    # are usually smaller than the floor they enclose
    background_id = max(range(n), key=lambda i: (len(comps[i]["cells"]), -i)) if n else None
    inds = norm_rects(indicator_rect)

    objs: list[dict[str, Any]] = []
    for cid, comp in enumerate(comps):
        cells = comp["cells"]
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        rect = [min(rs), max(rs) + 1, min(cs), max(cs) + 1]
        sig = shape_signature(cells, comp["value"])
        in_ind = any(rect[0] >= ind[0] and rect[1] <= ind[1] and rect[2] >= ind[2] and rect[3] <= ind[3] for ind in inds)
        obj: dict[str, Any] = {
            "id": cid,
            "color": comp["value"],
            "color_name": (color_names or {}).get(comp["value"], str(comp["value"])),
            "size": len(cells),
            "cells": sorted(cells),  # list of (row, col) tuples
            "rect": rect,
            "crop": crop_csv(grid, rect) if len(cells) <= CROP_MAX_CELLS else None,
            "outline": [[r, c] for r, c in _trace_outline(set(cells), comp["start"])],
            "hash": sig["hash"],
            "shape_hash": sig["shape_hash"],
            "rotation": sig["rotation"],
            "symmetry": sig["symmetry"],
            "children": children[cid],
            "adjacent": sorted(adj[cid]),
            "is_background": cid == background_id,
            "is_remaining_actions_indicator": in_ind,
        }
        objs.append(obj)
    if not include_background and background_id is not None:
        objs = [o for o in objs if not o["is_background"]]
    return objs


# ----------------------------------------------------------------------------- rendering

def render(objs: list[dict[str, Any]], shape: tuple[int, int], background_color: int | None = None) -> np.ndarray:
    """Paint objects onto a blank board (background first, then every other object in id order).

    Only `cells` and `color` are read, so a model-written next_state may move an object by
    shifting its `cells`.  Missing/invalid cells are ignored.
    """
    h, w = int(shape[0]), int(shape[1])
    if background_color is None:
        bg = [o for o in objs if o.get("is_background")]
        if bg:
            background_color = int(bg[0]["color"])
        else:
            background_color = int(max(objs, key=lambda o: o.get("size", 0))["color"]) if objs else 0
    board = np.full((h, w), int(background_color), dtype=np.uint8)
    # the background object is the fill colour itself; paint the rest in id order
    ordered = sorted((o for o in objs if not o.get("is_background")), key=lambda o: int(o.get("id", 0)))
    for o in ordered:
        color = int(o["color"])
        cells = o.get("cells") or ()
        if not cells:
            continue
        arr = np.asarray(cells, dtype=np.int64).reshape(-1, 2)
        ok = (arr[:, 0] >= 0) & (arr[:, 0] < h) & (arr[:, 1] >= 0) & (arr[:, 1] < w)
        board[arr[ok, 0], arr[ok, 1]] = color
    return board


def copy_objects(objs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cheap copy for search: new dicts, new cells lists (cells themselves are immutable tuples)."""
    out = []
    for o in objs:
        d = dict(o)
        d["cells"] = list(o.get("cells") or ())
        d["rect"] = list(o.get("rect") or ())
        out.append(d)
    return out


# ----------------------------------------------------------------------------- text helpers

def rect_text(rect: list[int] | tuple[int, ...]) -> str:
    r0, r1, c0, c1 = rect
    return f"[{r0}:{r1}, {c0}:{c1}]"


def shape_text(o: dict[str, Any]) -> str:
    r0, r1, c0, c1 = o["rect"]
    hh, ww = r1 - r0, c1 - c0
    if o["size"] == hh * ww:
        return f"{hh}x{ww} block" if hh != 1 or ww != 1 else "1 cell"
    return f"{o['size']}-cell shape in {hh}x{ww}"


def obj_text(o: dict[str, Any], with_id: bool = True) -> str:
    """One-line description, e.g. 'obj 3: color 2 (red), 2x2 block at [1:3, 1:3]'."""
    head = f"obj {o['id']}: " if with_id else ""
    tags = []
    if o.get("is_background"):
        tags.append("background")
    if o.get("is_remaining_actions_indicator"):
        tags.append("remaining-actions indicator")
    if o.get("children"):
        tags.append(f"encloses {len(o['children'])}")
    tag_text = f" ({', '.join(tags)})" if tags else ""
    return (
        f"{head}color {o['color']} ({o['color_name']}), {shape_text(o)} at {rect_text(o['rect'])}, "
        f"shape {o['shape_hash'][:6]}{tag_text}"
    )


def _multi(o: dict[str, Any]) -> bool:
    return "+" in str(o.get("color_name") or "")


def _kind_key(o: dict[str, Any]) -> tuple:
    """v2.11: single-colour objects: colour + shape (any rotation) + size; multi-colour (object view): the exact look."""
    if _multi(o):
        return (o["color"], o["shape_hash"], o["size"], o.get("hash"))
    return (o["color"], o["shape_hash"], o["size"])


def _col_text(o: dict[str, Any]) -> str:
    return str(o["color_name"]) if _multi(o) else f"color {o['color']} ({o['color_name']})"


def inventory(objs: list[dict[str, Any]], exclude_background: bool = True, exclude_indicator: bool = True) -> list[dict[str, Any]]:
    """Group objects by (color, shape_hash, size): what kinds of things are on the board."""
    groups: dict[tuple[int, str, int], dict[str, Any]] = {}
    for o in objs:
        if exclude_background and o.get("is_background"):
            continue
        if exclude_indicator and o.get("is_remaining_actions_indicator"):
            continue
        key = _kind_key(o)
        g = groups.get(key)
        if g is None:
            g = groups[key] = {"color": o["color"], "color_name": o["color_name"], "shape_hash": o["shape_hash"],
                               "size": o["size"], "count": 0, "example": o, "ids": []}
        g["count"] += 1
        g["ids"].append(o["id"])
    return sorted(groups.values(), key=lambda g: (-g["count"] * g["size"], g["color"]))


def inventory_text(objs: list[dict[str, Any]], max_groups: int = 200) -> str:  # v2.3: every kind (200 = safety cap only)
    groups = inventory(objs)
    by_id = {o["id"]: o for o in objs}
    lines = []
    for g in groups[:max_groups]:
        ex = g["example"]
        if g["count"] == 1:
            where = rect_text(ex["rect"])
        else:  # v2.4: every instance's position (up to 20 per kind)
            rects = [rect_text(by_id[i]["rect"]) for i in g["ids"][:20] if i in by_id]
            where = "at " + ", ".join(rects) + (f" (+{g['count'] - 20} more; lib.extract_components() lists all)" if g["count"] > 20 else "")
        lines.append(f"  {g['count']} x {_col_text(ex)} {shape_text(ex)} [shape {g['shape_hash'][:6]}] {where}")
    if len(groups) > max_groups:
        lines.append(f"  ... and {len(groups) - max_groups} more kinds (lib.extract_components() lists everything)")
    bg = [o for o in objs if o.get("is_background")]
    head = f"Components: {sum(1 for o in objs if not o.get('is_background'))} (plus background color {bg[0]['color']} ({bg[0]['color_name']}), {bg[0]['size']} cells)" if bg else f"Components: {len(objs)}"
    return head + "\n" + "\n".join(lines)


def inventory_delta_text(prev_objs: list[dict[str, Any]] | None, objs: list[dict[str, Any]]) -> str:
    """What kinds of objects are new / gone compared with another board (e.g. previous level start)."""
    if prev_objs is None:
        return ""
    prev = {_kind_key(g["example"]): g for g in inventory(prev_objs)}
    cur = {_kind_key(g["example"]): g for g in inventory(objs)}
    new = [cur[k] for k in cur if k not in prev]
    gone = [prev[k] for k in prev if k not in cur]
    lines = []
    if new:
        lines.append("New kinds of objects not present at the previous level start:")
        for g in new[:12]:
            lines.append(f"  {g['count']} x {_col_text(g['example'])} {shape_text(g['example'])} e.g. {rect_text(g['example']['rect'])}")
    if gone:
        lines.append("Kinds of objects from the previous level start that are gone:")
        for g in gone[:12]:
            lines.append(f"  {g['count']} x {_col_text(g['example'])} {shape_text(g['example'])}")
    if not lines:
        lines.append("Same kinds of objects as at the previous level start (counts/positions may differ).")
    return "\n".join(lines)


def compact_for_log(objs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replay/log representation: everything except crop/outline (derivable), cells kept."""
    out = []
    for o in objs:
        out.append({
            "id": o["id"], "color": o["color"], "size": o["size"], "rect": o["rect"],
            "hash": o["hash"], "shape_hash": o["shape_hash"], "rotation": o["rotation"],
            "children": o["children"], "is_background": o["is_background"],
            "is_remaining_actions_indicator": o["is_remaining_actions_indicator"],
            "cells": o["cells"] if o["size"] <= 400 else None,
        })
    return out


# ---------------------------------------------------------------------------- v2.10: drawing helpers
# Objects are plain dicts; only "color" and "cells" matter for drawing (a new object is just
# {"color": c, "cells": [(row, col), ...]}). These functions never modify their arguments.
_STALE_FIELDS = ("outline", "crop", "hash", "shape_hash", "rotation", "symmetry", "adjacent", "children")


def _with_cells(obj: dict[str, Any], cells: Iterable[tuple[int, int]]) -> dict[str, Any]:
    out = dict(obj)
    cl = [(int(r), int(c)) for r, c in cells]
    out["cells"] = cl
    out["size"] = len(cl)
    if cl:
        rs = [r for r, _ in cl]
        cs = [c for _, c in cl]
        out["rect"] = [min(rs), max(rs) + 1, min(cs), max(cs) + 1]
    else:
        out["rect"] = None
    for k in _STALE_FIELDS:
        out.pop(k, None)
    return out


def shifted_object(obj: dict[str, Any], dr: int, dc: int) -> dict[str, Any]:
    """A copy of obj moved by dr rows and dc columns."""
    return _with_cells(obj, [(r + int(dr), c + int(dc)) for r, c in obj["cells"]])


def rotated_object(obj: dict[str, Any], quarter_turns: int = 1) -> dict[str, Any]:
    """A copy of obj rotated clockwise by 90 degrees x quarter_turns; the top-left corner of its bounding box stays put."""
    cells = [(int(r), int(c)) for r, c in obj["cells"]]
    if not cells:
        return _with_cells(obj, [])
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    pts = [(r - r0, c - c0) for r, c in cells]
    for _ in range(int(quarter_turns) % 4):
        h = max(y for y, _ in pts) + 1
        pts = [(x, h - 1 - y) for y, x in pts]
    return _with_cells(obj, [(y + r0, x + c0) for y, x in pts])


def mirrored_object(obj: dict[str, Any], axis: str = "left-right") -> dict[str, Any]:
    """A copy of obj flipped inside its bounding box: axis "left-right" swaps left and right, "up-down" swaps top and bottom."""
    cells = [(int(r), int(c)) for r, c in obj["cells"]]
    if not cells:
        return _with_cells(obj, [])
    r0, r1 = min(r for r, _ in cells), max(r for r, _ in cells)
    c0, c1 = min(c for _, c in cells), max(c for _, c in cells)
    if str(axis).lower().replace("_", "-") in ("up-down", "updown", "ud", "vertical", "top-bottom"):
        return _with_cells(obj, [(r0 + r1 - r, c) for r, c in cells])
    return _with_cells(obj, [(r, c0 + c1 - c) for r, c in cells])


def _as_list(objs: Any) -> list[dict[str, Any]]:
    return [objs] if isinstance(objs, dict) else list(objs)


def draw_objects(board: Any, objs: Any) -> np.ndarray:
    """A copy of board with the object(s) drawn on it in order (later ones on top); cells outside the board are ignored."""
    b = np.array(board, dtype=np.uint8, copy=True)
    h, w = b.shape[:2]
    for o in _as_list(objs):
        col = int(o["color"])
        for r, c in o["cells"]:
            if 0 <= r < h and 0 <= c < w:
                b[r, c] = col
    return b


def erase_objects(board: Any, objs: Any, fill: int) -> np.ndarray:
    """A copy of board with the cells of the object(s) set to colour `fill`."""
    b = np.array(board, dtype=np.uint8, copy=True)
    h, w = b.shape[:2]
    for o in _as_list(objs):
        for r, c in o["cells"]:
            if 0 <= r < h and 0 <= c < w:
                b[r, c] = int(fill)
    return b


def move_object(board: Any, obj: dict[str, Any], dr: int, dc: int, fill: int) -> np.ndarray:
    """A copy of board with obj moved by (dr, dc): its old cells become colour `fill`, then it is drawn at the new place."""
    return draw_objects(erase_objects(board, obj, fill), shifted_object(obj, dr, dc))


DRAW_HELPERS = ("shifted_object", "rotated_object", "mirrored_object", "draw_objects", "erase_objects", "move_object")

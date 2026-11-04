"""Object-level difference between two boards (stdlib + numpy).

compute(before, after, objs_before, objs_after, indicator_rect) returns a dict:

  changed_cells            number of cells that differ (whole board)
  changed_outside_indicator  number of differing cells outside the marked indicator rect
  changed_rects            list of [r0, r1, c0, c1] bounding rects of changed regions
                           (indicator rect excluded)
  moved        [{color, size, shape_hash, from: rect, to: rect, delta: [dr, dc]}]
  rotated      [{color, size, shape_hash, rect, rotated_by}]
  recolored    [{from_color, to_color, size, rect}]
  reshaped     [{color, from_size, to_size, from_rect, to_rect}]   same colour, overlapping, new shape
  appeared     [{color, size, shape_hash, rect}]
  disappeared  [{color, size, shape_hash, rect}]
  unchanged    number of objects that are identical in both boards

Objects inside the indicator rect and the background object are ignored for matching.
Matching is a heuristic: identical hashes are paired by minimum total displacement; the
remaining same-colour pieces are paired by overlap.  Equal hashes do not prove identity.
"""
from __future__ import annotations

from typing import Any

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


def _rect_overlap(a: list[int], b: list[int]) -> int:
    r0, r1 = max(a[0], b[0]), min(a[1], b[1])
    c0, c1 = max(a[2], b[2]), min(a[3], b[3])
    return max(0, r1 - r0) * max(0, c1 - c0)


def _rect_area(a: list[int]) -> int:
    return max(0, a[1] - a[0]) * max(0, a[3] - a[2])


def _center(a: list[int]) -> tuple[float, float]:
    return ((a[0] + a[1]) / 2.0, (a[2] + a[3]) / 2.0)


def _dist(a: list[int], b: list[int]) -> float:
    (ra, ca), (rb, cb) = _center(a), _center(b)
    return abs(ra - rb) + abs(ca - cb)


def _assign_min_cost(cost: list[list[float]]) -> list[int]:
    """Minimum-cost one-to-one assignment, rows -> columns (n <= m). Hungarian algorithm."""
    n = len(cost)
    m = len(cost[0]) if n else 0
    if n == 0:
        return []
    INF = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    result = [-1] * n
    for j in range(1, m + 1):
        if p[j]:
            result[p[j] - 1] = j - 1
    return result


def changed_rects(mask: np.ndarray, merge_gap: int = 1) -> list[list[int]]:
    """Bounding rects of changed regions; nearby regions (within merge_gap) are merged."""
    coords = np.argwhere(mask)
    if coords.shape[0] == 0:
        return []
    # union-find over changed cells using a dilated connectivity would be heavy; use the
    # simple iterative rectangle expansion from the harness: start with one rect per cell
    # cluster found by scanning rows.
    rects: list[list[int]] = []
    cells = [tuple(int(x) for x in rc) for rc in coords.tolist()]
    for r, c in cells:
        placed = False
        for rect in rects:
            if rect[0] - merge_gap <= r < rect[1] + merge_gap and rect[2] - merge_gap <= c < rect[3] + merge_gap:
                rect[0] = min(rect[0], r); rect[1] = max(rect[1], r + 1)
                rect[2] = min(rect[2], c); rect[3] = max(rect[3], c + 1)
                placed = True
                break
        if not placed:
            rects.append([r, r + 1, c, c + 1])
    # merge overlapping/adjacent rects until stable
    changed = True
    while changed:
        changed = False
        out: list[list[int]] = []
        for rect in rects:
            merged = False
            for o in out:
                if (rect[0] - merge_gap <= o[1] and o[0] - merge_gap <= rect[1]
                        and rect[2] - merge_gap <= o[3] and o[2] - merge_gap <= rect[3]):
                    o[0] = min(o[0], rect[0]); o[1] = max(o[1], rect[1])
                    o[2] = min(o[2], rect[2]); o[3] = max(o[3], rect[3])
                    merged = True
                    changed = True
                    break
            if not merged:
                out.append(rect)
        rects = out
    rects.sort()
    return rects


def _norm_cells(cells):
    rs = [int(r) for r, _ in cells]
    cs = [int(c) for _, c in cells]
    r0, c0 = min(rs), min(cs)
    return frozenset((int(r) - r0, int(c) - c0) for r, c in cells)


def _exact_transform(cells_b, cells_a, colors_b=None, colors_a=None):
    """v2.10: which turn / flip maps the old shape onto the new one (None when none does exactly). v2.11: with colours,
    the cells must keep their colours under the transform."""
    try:
        if colors_b is not None and colors_a is not None:
            rb = min(int(r) for r, _ in cells_b); cb = min(int(c) for _, c in cells_b)
            ra = min(int(r) for r, _ in cells_a); ca = min(int(c) for _, c in cells_a)
            b = frozenset((int(r) - rb, int(c) - cb, int(v)) for (r, c), v in zip(cells_b, colors_b))
            a = frozenset((int(r) - ra, int(c) - ca, int(v)) for (r, c), v in zip(cells_a, colors_a))
        else:
            b = frozenset((y, x, 0) for y, x in _norm_cells(cells_b))
            a = frozenset((y, x, 0) for y, x in _norm_cells(cells_a))
    except Exception:
        return None
    def rot(sh):
        h = max(y for y, _, _ in sh) + 1
        return frozenset((x, h - 1 - y, v) for y, x, v in sh)
    def norm(sh):
        r0 = min(y for y, _, _ in sh); c0 = min(x for _, x, _ in sh)
        return frozenset((y - r0, x - c0, v) for y, x, v in sh)
    cur = b
    for k in (1, 2, 3):
        cur = norm(rot(cur))
        if cur == a:
            return f"rotated {90 * k} deg clockwise"
    w = max(x for _, x, _ in b)
    h = max(y for y, _, _ in b)
    if norm(frozenset((y, w - x, v) for y, x, v in b)) == a:
        return "mirrored left-right"
    if norm(frozenset((h - y, x, v) for y, x, v in b)) == a:
        return "mirrored up-down"
    if norm(frozenset((x, y, v) for y, x, v in b)) == a or norm(frozenset((w - x, h - y, v) for y, x, v in b)) == a:
        return "mirrored across a diagonal"
    return None


def compute(
    before: Any,
    after: Any,
    objs_before: list[dict[str, Any]],
    objs_after: list[dict[str, Any]],
    indicator_rect: tuple[int, int, int, int] | list[int] | None = None,
    max_group_match: int = 400,   # v2.10: was 40
) -> dict[str, Any]:
    b = np.asarray(before)
    a = np.asarray(after)
    result: dict[str, Any] = {
        "changed_cells": 0, "changed_outside_indicator": 0, "changed_rects": [],
        "moved": [], "rotated": [], "recolored": [], "reshaped": [],
        "appeared": [], "disappeared": [], "unchanged": 0,
    }
    if b.shape != a.shape:
        result["shape_changed"] = [list(b.shape), list(a.shape)]
        return result
    mask = b != a
    result["changed_cells"] = int(mask.sum())
    outside = mask.copy()
    outside &= ~rects_mask(indicator_rect, outside.shape)
    result["changed_outside_indicator"] = int(outside.sum())
    result["changed_rects"] = changed_rects(outside)
    if result["changed_cells"] == 0:
        result["unchanged"] = len(objs_before)
        return result

    def relevant(o: dict[str, Any]) -> bool:
        return not o.get("is_background") and not o.get("is_remaining_actions_indicator")

    def touched(o: dict[str, Any]) -> bool:
        r0, r1, c0, c1 = o["rect"]
        r0, c0 = max(0, r0 - 1), max(0, c0 - 1)
        r1, c1 = min(mask.shape[0], r1 + 1), min(mask.shape[1], c1 + 1)
        return bool(mask[r0:r1, c0:c1].any())

    gone = [o for o in objs_before if relevant(o) and touched(o)]
    new = [o for o in objs_after if relevant(o) and touched(o)]
    result["unchanged"] = len([o for o in objs_before if relevant(o)]) - len(gone)

    def entry(o: dict[str, Any]) -> dict[str, Any]:
        return {"color": o["color"], "color_name": o.get("color_name", str(o["color"])), "size": o["size"],
                "shape_hash": o["shape_hash"], "rect": list(o["rect"])}

    # ---- exact same object at the same place (touched only by a neighbour) -> unchanged
    key_before = {(o["hash"], tuple(o["rect"])): o for o in gone}
    still: set[int] = set()
    for o in new:
        k = (o["hash"], tuple(o["rect"]))
        if k in key_before:
            still.add(id(o)); still.add(id(key_before[k]))
    gone = [o for o in gone if id(o) not in still]
    new = [o for o in new if id(o) not in still]
    result["unchanged"] += len(still) // 2

    # ---- moved: same hash (colour + exact shape) at a different place, min total displacement
    used_gone: set[int] = set()
    used_new: set[int] = set()
    by_hash_before: dict[str, list[dict[str, Any]]] = {}
    by_hash_after: dict[str, list[dict[str, Any]]] = {}
    for o in gone:
        by_hash_before.setdefault(o["hash"], []).append(o)
    for o in new:
        by_hash_after.setdefault(o["hash"], []).append(o)
    for h, group_b in by_hash_before.items():
        group_a = by_hash_after.get(h)
        if not group_a:
            continue
        if len(group_b) > max_group_match or len(group_a) > max_group_match:
            continue  # too many identical pieces: report as appeared/disappeared counts
        swap = len(group_b) > len(group_a)
        rows, cols = (group_a, group_b) if swap else (group_b, group_a)
        cost = [[_dist(x["rect"], y["rect"]) for y in cols] for x in rows]
        assign = _assign_min_cost(cost)
        for i, j in enumerate(assign):
            if j < 0:
                continue
            ob, oa = (cols[j], rows[i]) if swap else (rows[i], cols[j])
            used_gone.add(id(ob)); used_new.add(id(oa))
            e = entry(ob)
            e["from"] = list(ob["rect"]); e["to"] = list(oa["rect"]); e.pop("rect", None)
            e["delta"] = [oa["rect"][0] - ob["rect"][0], oa["rect"][2] - ob["rect"][2]]
            result["moved"].append(e)
    gone = [o for o in gone if id(o) not in used_gone]
    new = [o for o in new if id(o) not in used_new]

    # ---- rotated: same colour, same D4 shape, same size, overlapping rects, different exact hash
    for ob in list(gone):
        best = None
        for oa in new:
            if oa["color"] == ob["color"] and oa["shape_hash"] == ob["shape_hash"] and oa["size"] == ob["size"] and oa["hash"] != ob["hash"]:
                if ob.get("colors") is not None and oa.get("colors") is not None:
                    # v2.11: a turn / flip must carry each cell's colour along; otherwise it is a recolouring or a reshape
                    if _exact_transform(ob["cells"], oa["cells"], ob["colors"], oa["colors"]) is None:
                        continue
                ov = _rect_overlap(ob["rect"], oa["rect"])
                if ov > 0 and (best is None or ov > best[0]):
                    best = (ov, oa)
        if best is not None:
            oa = best[1]
            e = entry(ob)
            e["rect"] = list(oa["rect"])
            e["rotated_by"] = None
            e["transform"] = _exact_transform(ob["cells"], oa["cells"], ob.get("colors"), oa.get("colors"))   # v2.10: the exact turn or flip
            result["rotated"].append(e)
            gone.remove(ob); new.remove(oa)

    # ---- recolored: identical cell set, different colour
    cells_after = {}
    for oa in new:
        cells_after.setdefault(tuple(map(tuple, oa["cells"])), []).append(oa)
    for ob in list(gone):
        key = tuple(map(tuple, ob["cells"]))
        cands = cells_after.get(key)
        if cands:
            oa = cands.pop(0)
            result["recolored"].append({"from_color": ob["color"], "to_color": oa["color"],
                                        "from_color_name": ob.get("color_name"), "to_color_name": oa.get("color_name"),
                                        "size": ob["size"], "rect": list(ob["rect"])})
            gone.remove(ob); new.remove(oa)

    # ---- reshaped: same colour, rects overlap >= 50% of the smaller, greedy by overlap
    pairs = []
    for ob in gone:
        for oa in new:
            if oa["color"] != ob["color"]:
                continue
            ov = _rect_overlap(ob["rect"], oa["rect"])
            smaller = min(_rect_area(ob["rect"]), _rect_area(oa["rect"])) or 1
            if ov / smaller >= 0.5:
                pairs.append((ov / smaller, -abs(ob["size"] - oa["size"]), id(ob), id(oa), ob, oa))
    pairs.sort(key=lambda t: (t[0], t[1]), reverse=True)
    taken_b: set[int] = set(); taken_a: set[int] = set()
    for _, _, ib, ia, ob, oa in pairs:
        if ib in taken_b or ia in taken_a:
            continue
        taken_b.add(ib); taken_a.add(ia)
        result["reshaped"].append({"color": ob["color"], "color_name": ob.get("color_name"),
                                   "from_size": ob["size"], "to_size": oa["size"],
                                   "from_rect": list(ob["rect"]), "to_rect": list(oa["rect"]),
                                   "from_shape_hash": ob["shape_hash"], "to_shape_hash": oa["shape_hash"]})
    gone = [o for o in gone if id(o) not in taken_b]
    new = [o for o in new if id(o) not in taken_a]

    result["disappeared"] = [entry(o) for o in gone]
    result["appeared"] = [entry(o) for o in new]
    return result


# ----------------------------------------------------------------------------- text

def _rt(rect: list[int]) -> str:
    return f"[{rect[0]}:{rect[1]}, {rect[2]}:{rect[3]}]"


def _shape(e: dict[str, Any], rect_key: str = "rect") -> str:
    rect = e.get(rect_key) or e.get("to") or e.get("from") or [0, 0, 0, 0]
    hh, ww = rect[1] - rect[0], rect[3] - rect[2]
    if e["size"] == hh * ww:
        return f"{hh}x{ww}" if (hh, ww) != (1, 1) else "1-cell"
    return f"{e['size']}-cell"


def _col(e: dict[str, Any], key: str = "color", name_key: str = "color_name") -> str:
    name = e.get(name_key)
    if name and "+" in str(name):   # v2.11: a multi-colour object (object view) is named by its colours
        return str(name)
    return f"color {e[key]}" + (f" ({name})" if name and str(name) != str(e[key]) else "")


def render_text(diff: dict[str, Any], cap: int = 10) -> list[str]:
    """Compact lines describing the object-level change; moves are grouped by
    (colour, shape, delta) so a scene shift is one line."""
    lines: list[str] = []
    if diff.get("shape_changed"):
        return [f"board shape changed {diff['shape_changed'][0]} -> {diff['shape_changed'][1]}"]
    entries: list[str] = []
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for e in diff["moved"]:
        groups.setdefault((e["color"], e["shape_hash"], tuple(e["delta"])), []).append(e)
    for (color, _, delta), es in groups.items():
        e0 = es[0]
        d = f"({delta[0]:+d} rows, {delta[1]:+d} cols)"
        if len(es) == 1:
            entries.append(f"moved: {_col(e0)} {_shape(e0, 'to')} {_rt(e0['from'])} -> {_rt(e0['to'])} {d}")
        else:
            entries.append(f"moved: {len(es)} x {_col(e0)} {_shape(e0, 'to')} by {d}, e.g. {_rt(es[0]['from'])} -> {_rt(es[0]['to'])}")
    for e in diff["rotated"]:
        _tf = e.get("transform") or "turned or flipped (same shape, new orientation)"
        entries.append(f"{_tf}: {_col(e)} {_shape(e)} at {_rt(e['rect'])}")
    for e in diff["recolored"]:
        if "+" in str(e.get("from_color_name") or "") or "+" in str(e.get("to_color_name") or ""):   # v2.11
            entries.append(f"recolored: {_shape(e)} at {_rt(e['rect'])} {e.get('from_color_name')} -> {e.get('to_color_name')}")
        else:
            entries.append(f"recolored: {_shape(e)} at {_rt(e['rect'])} color {e['from_color']} -> {e['to_color']}")
    for e in diff["reshaped"]:
        entries.append(f"reshaped: {_col(e)} {e['from_size']} cells {_rt(e['from_rect'])} -> {e['to_size']} cells {_rt(e['to_rect'])}")
    agroups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for e in diff["appeared"]:
        agroups.setdefault((e["color"], e["shape_hash"], e["size"]), []).append(e)
    for (color, _, size), es in agroups.items():
        e0 = es[0]
        if len(es) == 1:
            entries.append(f"appeared: {_col(e0)} {_shape(e0)} at {_rt(e0['rect'])}")
        else:
            entries.append(f"appeared: {len(es)} x {_col(e0)} {_shape(e0)}, e.g. {_rt(e0['rect'])}")
    dgroups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for e in diff["disappeared"]:
        dgroups.setdefault((e["color"], e["shape_hash"], e["size"]), []).append(e)
    for (color, _, size), es in dgroups.items():
        e0 = es[0]
        if len(es) == 1:
            entries.append(f"disappeared: {_col(e0)} {_shape(e0)} from {_rt(e0['rect'])}")
        else:
            entries.append(f"disappeared: {len(es)} x {_col(e0)} {_shape(e0)}, e.g. {_rt(e0['rect'])}")
    lines.extend(entries[:cap])
    if len(entries) > cap:
        lines.append(f"... and {len(entries) - cap} more component changes (lib.frame_diff() in run has all of them)")
    return lines


# ----------------------------------------------------------------------------- animations

def transient_summary(before: Any, frames: list[Any], after: Any,
                      indicator_rect: tuple[int, int, int, int] | list[int] | None = None) -> dict[str, Any] | None:
    """Cells that changed during an animation and are back to their starting value in the final
    board (invisible when only `before` and `after` are compared). None when there are no frames."""
    if not frames:
        return None
    b = np.asarray(before)
    a = np.asarray(after)
    if a.shape != b.shape:
        return None
    touched = np.zeros(b.shape, dtype=bool)
    first = np.zeros(b.shape, dtype=np.int64)
    for f in frames:
        f = np.asarray(f)
        if f.shape != b.shape:
            continue
        diff = f != b
        new = diff & ~touched
        first[new] = f[new]
        touched |= diff
    transient = touched & (a == b)
    transient &= ~rects_mask(indicator_rect, transient.shape)
    n = int(transient.sum())
    out: dict[str, Any] = {"frames": len(frames), "transient": n, "bbox": None, "transitions": []}
    if n:
        coords = np.argwhere(transient)
        (r0, c0), (r1, c1) = coords.min(axis=0), coords.max(axis=0)
        out["bbox"] = [int(r0), int(r1) + 1, int(c0), int(c1) + 1]
        pairs: dict[tuple[int, int], int] = {}
        for x, y in zip(b[transient].tolist(), first[transient].tolist()):
            pairs[(int(x), int(y))] = pairs.get((int(x), int(y)), 0) + 1
        out["transitions"] = sorted(((k[0], k[1], v) for k, v in pairs.items()), key=lambda t: -t[2])[:4]
    return out


def anim_steps(before: Any, frames: list[Any], after: Any, segment: Any,
               indicator_rect: tuple[int, int, int, int] | list[int] | None = None,
               cap: int = 40, max_boards: int = 160) -> list[str]:
    """Object-level changes step by step through an animation: before -> frame 1 -> ... -> final.
    `segment(board)` returns the objects of a board. Steps without changes (outside the indicator)
    are skipped, identical consecutive steps are merged and A/B/A/B blinking is collapsed."""
    boards = [np.asarray(before)] + [np.asarray(f) for f in frames] + [np.asarray(after)]
    truncated = len(boards) > max_boards
    if truncated:
        boards = boards[: max_boards - 1] + [boards[-1]]
    last = len(boards) - 1
    cache: dict[bytes, Any] = {}

    def seg(x: np.ndarray) -> Any:
        k = x.tobytes()
        if k not in cache:
            cache[k] = segment(x)
        return cache[k]

    raw: list[tuple[int, str]] = []
    for i in range(1, len(boards)):
        p, q = boards[i - 1], boards[i]
        if p.shape != q.shape:
            raw.append((i, f"board shape changed {list(p.shape)} -> {list(q.shape)}"))
            continue
        if np.array_equal(p, q):
            continue
        d = compute(p, q, seg(p), seg(q), indicator_rect=indicator_rect)
        if d["changed_outside_indicator"] == 0:
            continue
        lines = render_text(d, cap=4)
        if lines:
            text = "; ".join(lines)
        else:
            rects = ", ".join(_rt(r) for r in d["changed_rects"][:3])
            text = f"{d['changed_outside_indicator']} cells changed in {rects}"
        raw.append((i, text))

    def label(i: int, j: int) -> str:
        name = lambda k: "final" if k == last else f"frame {k}"
        return f"{name(i)}" if i == j else f"{name(i)}..{name(j)}"

    groups: list[list[Any]] = []  # [first_step, last_step, text, count]
    for i, t in raw:
        if groups and groups[-1][2] == t:
            groups[-1][1] = i
            groups[-1][3] += 1
        else:
            groups.append([i, i, t, 1])
    out: list[str] = []
    j = 0
    while j < len(groups):
        if (j + 3 < len(groups) and groups[j][2] == groups[j + 2][2] and groups[j + 1][2] == groups[j + 3][2]
                and groups[j][2] != groups[j + 1][2]):
            k = j + 3
            while k + 1 < len(groups) and groups[k + 1][2] == groups[k - 1][2]:
                k += 1
            out.append(f"{label(groups[j][0], groups[k][1])}: alternating {k - j + 1} times between "
                       f"[{groups[j][2]}] and [{groups[j + 1][2]}]")
            j = k + 1
            continue
        g = groups[j]
        rep = f" (x{g[3]})" if g[3] > 1 else ""
        out.append(f"{label(g[0], g[1])}{rep}: {g[2]}")
        j += 1
    if not out:
        out.append("no changes outside the indicator between consecutive frames")
    if len(out) > cap:
        out = out[:cap] + [f"... and {len(out) - cap} more steps (lib.anim_diff(cap=...) shows more)"]
    if truncated:
        out.append(f"(only the first {max_boards - 1} frames were analysed)")
    return out

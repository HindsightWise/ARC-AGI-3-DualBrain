"""Text for the model from the layered tracker (scene_worker.summarise output).

Everything is worded as what the tracker observed or decided, never as game semantics: objects are named by their
tracker id (#n), colours, cell counts and boxes of their observed cells. Guessed cells (the tracker's completion of a
shape behind other objects) are never stated as facts; the inventory shows how many cells of an object are visible.

v2.6.5: these lines are the only object-change block; they are ordered by importance and capped by characters,
small-object dust is collapsed, rotations are reported only when exact, recolour pairs that look like a jump and
identical objects that may have swapped are flagged as ambiguous, and a shift of most of the board is reported once.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def _rt(b) -> str:
    return f"[{b[0]}:{b[1]}, {b[2]}:{b[3]}]" if b else "[?]"


def _inside(b, rects) -> bool:
    if not b or not rects:
        return False
    for r in rects:
        if b[0] >= r[0] and b[1] <= r[1] and b[2] >= r[2] and b[3] <= r[3]:
            return True
    return False


class Fmt:
    def __init__(self, prev: dict | None, cur: dict, rects, names: dict[int, str]):
        self.prev = (prev or {}).get("objs", {}) if prev else {}
        self.cur = cur.get("objs", {})
        self.covered = cur.get("covered", {})
        self.rects = [list(r) for r in (rects or [])]
        self.names = names

    def info(self, oid, when="cur"):
        k = str(oid)
        if when == "cur":
            return self.cur.get(k) or self.prev.get(k)
        return self.prev.get(k) or self.cur.get(k)

    def col(self, c) -> str:
        return f"{c} {self.names.get(int(c), str(c))}"

    def desc(self, info, *, vis=True) -> str:
        if not info:
            return "?"
        cols = info.get("cols") or []
        if len(cols) == 1:
            s = f"{self.col(cols[0][0])}, {cols[0][1]} cell{'s' if cols[0][1] != 1 else ''}"
        else:
            s = " + ".join(f"{self.col(c)} {n}" for c, n in cols[:4]) + (" + ..." if len(cols) > 4 else "") + " cells"
        if vis and info.get("vis") is not None and info.get("vis") < info.get("seen", 0):
            s += f", {info['vis']} visible"
        return s

    def size(self, oid) -> int:
        a = self.info(oid)
        return int(sum(n for _c, n in (a.get("cols") or []))) if a else 0

    def ind(self, oid) -> bool:
        """an object entirely inside the remaining-actions indicator (reported separately by the harness)"""
        a = self.prev.get(str(oid))
        b = self.cur.get(str(oid))
        boxes = [x.get("box") for x in (a, b) if x]
        return bool(boxes) and all(_inside(bx, self.rects) for bx in boxes)


def _off(b, shape) -> str:
    if not b or not shape:
        return ""
    h, w = shape
    if b[1] <= 0 or b[0] >= h or b[3] <= 0 or b[2] >= w:
        return " (outside the board now)"
    return ""


# ---------------------------------------------------------------- whole-board shift

def board_shift(before, after, rects, candidates) -> dict | None:
    """Does one shift (dr, dc) explain most of the changed cells? candidates = offsets to test (from the edits)."""
    if before is None or after is None:
        return None
    b = np.asarray(before)
    a = np.asarray(after)
    if b.shape != a.shape or b.ndim != 2:
        return None
    h, w = a.shape
    changed = b != a
    for r in rects or []:
        changed[max(0, r[0]):max(0, r[1]), max(0, r[2]):max(0, r[3])] = False
    n = int(changed.sum())
    if n < 64:
        return None
    best = None
    for dr, dc in candidates:
        if (dr == 0 and dc == 0) or abs(dr) >= h or abs(dc) >= w:
            continue
        # after[y, x] == before[y - dr, x - dc] for the changed cells that have a source
        ys, xs = np.nonzero(changed)
        sy, sx = ys - dr, xs - dc
        ok = (sy >= 0) & (sx >= 0) & (sy < h) & (sx < w)
        expl = int(np.count_nonzero(a[ys[ok], xs[ok]] == b[sy[ok], sx[ok]]))
        if best is None or expl > best[0] or (expl == best[0] and abs(dr) + abs(dc) < abs(best[1]) + abs(best[2])):
            best = (expl, dr, dc)
    if best is None or best[0] < 0.7 * n:
        return None
    expl, dr, dc = best
    # cells that entered at the leading edge
    edge = np.zeros_like(changed)
    if dr > 0:
        edge[:dr, :] = True
    elif dr < 0:
        edge[h + dr:, :] = True
    if dc > 0:
        edge[:, :dc] = True
    elif dc < 0:
        edge[:, w + dc:] = True
    where = []
    if dr > 0:
        where.append("top")
    if dr < 0:
        where.append("bottom")
    if dc > 0:
        where.append("left")
    if dc < 0:
        where.append("right")
    return {"dr": int(dr), "dc": int(dc), "explained": expl, "changed": n, "edge_cells": int((changed & edge).sum()),
            "edge": " and ".join(where)}


# ---------------------------------------------------------------- per-action lines

def step_lines(prev: dict | None, cur: dict, rects=None, names: dict[int, str] | None = None,
               shape=(64, 64), before=None, after=None) -> list[tuple[int, str]]:
    """(importance, line) for one action within a level; the caller orders and caps them."""
    f = Fmt(prev, cur, rects, names or {})
    edits = cur.get("edits") or []
    out: list[tuple[int, str]] = []
    touched: set[str] = set()
    moves: dict[tuple[int, int], list] = {}
    reshapes: dict[tuple, int] = {}
    recolors: list[dict] = []
    dust: dict[str, int] = {}
    for e in edits:
        if e.get("type") == "reshape" and e.get("why") != "guess was wrong":
            k = (e.get("id"), "colour" if e.get("why") in ("feature", "recolour") else e.get("why"))
            reshapes[k] = reshapes.get(k, 0) + int(e.get("cells") or 0)
    main = {e.get("id") for e in edits if e.get("type") in ("appear", "recolor", "split", "merge", "rotate")}
    main |= {e.get("other") for e in edits if e.get("type") == "split"}
    done_reshape: set = set()
    seen_moves: set = set()
    small_total = sum(1 for e in edits if e.get("type") in ("appear", "disappear", "move", "recolor", "forget")
                      and e.get("id") is not None and f.size(e["id"]) <= 2)
    collapse_dust = small_total >= 4

    def is_dust(oid) -> bool:
        return collapse_dust and f.size(oid) <= 2

    for e in edits:
        t = e.get("type")
        oid = e.get("id")
        if oid is not None:
            touched.add(str(oid))
        if t in ("confirm", "scroll", "continue"):
            continue
        if t == "reshape" and e.get("why") == "guess was wrong":
            continue
        if oid is not None and f.ind(oid) and t not in ("merge", "split"):
            continue
        if t in ("appear", "disappear", "move", "recolor", "forget") and is_dust(oid):
            key = {"appear": "appeared", "disappear": "vanished", "move": "moved", "recolor": "changed colour",
                   "forget": "were dropped from tracking"}[t]
            dust[key] = dust.get(key, 0) + 1
            continue
        if t == "move":
            fr, to = e.get("from"), e.get("to")
            if not fr or not to or (fr[0] == to[0] and fr[1] == to[1]) or (oid, tuple(fr), tuple(to)) in seen_moves:
                continue
            seen_moves.add((oid, tuple(fr), tuple(to)))
            moves.setdefault((to[0] - fr[0], to[1] - fr[1]), []).append(e)
        elif t == "rotate":
            a, b = f.info(oid, "prev"), f.info(oid, "cur")
            if e.get("exact", True):
                out.append((88, f"#{oid} ({f.desc(a, vis=False)}) {e.get('turn', 'changed orientation').replace('turned', 'rotated')}: "
                                f"{_rt(a.get('box') if a else None)} -> {_rt(b.get('box') if b else None)}"))
            else:
                out.append((82, f"#{oid} ({f.desc(a, vis=False)}) changed shape ({e.get('cells_before', '?')} -> "
                                f"{e.get('cells_after', '?')} cells), now at {_rt(b.get('box') if b else None)}"))
        elif t == "recolor":
            recolors.append(e)
        elif t == "appear":
            b = f.info(oid, "cur")
            if not b or not b.get("vis"):
                continue          # created and covered within the same board: nothing of it is visible
            how = e.get("how")
            if how == "library":
                out.append((86, f"#{oid} visible again with its earlier shape: {f.desc(b)} at {_rt(b.get('box') if b else None)}"))
            else:
                extra = ""
                if how == "kind" and e.get("kindOf") is not None:
                    ld = e.get("like_desc")
                    extra = f" (same shape and colours as #{e['kindOf']}" + (f", last seen at {_rt(ld.get('rect'))}" if ld and ld.get("rect") else "") + ")"
                out.append((85 + min(10, int(math.log2(1 + f.size(oid)))), f"#{oid} appeared: {f.desc(b)} at {_rt(b.get('box') if b else None)}{extra}"))
        elif t == "disappear":
            a = f.info(oid, "prev")
            shows = [i for i in (e.get("now_shows") or []) if i >= 0 and i != oid]
            tail = (" (those cells now show " + ", ".join(f"#{i}" for i in shows[:3]) + ")") if shows else ""
            if a and a.get("vis", 1) == 0:
                out.append((70, f"#{oid} ({f.desc(a, vis=False)}, last known at {_rt(a.get('box'))} while covered) dropped: the tracker "
                                "no longer finds it there (it may have gone at any time while covered)"))
            else:
                out.append((85 + min(10, int(math.log2(1 + f.size(oid)))),
                            f"#{oid} ({f.desc(a, vis=False)}) is gone from {_rt(a.get('box') if a else None)}{tail}"))
        elif t == "split":
            other = e.get("other")
            a, b = f.info(oid, "cur"), f.info(other, "cur")
            d = e.get("to") or [0, 0]
            out.append((84, f"#{oid} split in two: part #{other} ({f.desc(b)}) moved ({d[0]:+d} rows, {d[1]:+d} cols) to "
                            f"{_rt(b.get('box') if b else None)}; #{oid} kept the rest ({f.desc(a)}) at {_rt(a.get('box') if a else None)}"))
            touched.add(str(other))
        elif t == "merge":
            parts = e.get("parts") or []
            b = f.info(oid, "cur")
            pl = " + ".join(f"#{p}" for p in parts) if len(parts) <= 6 else \
                f"{len(parts)} objects (" + ", ".join(f"#{p}" for p in parts[:6]) + ", ...)"
            out.append((83, f"#{oid} = {pl} (moved identically twice, never apart; now tracked as one object): "
                            f"{f.desc(b)} at {_rt(b.get('box') if b else None)}"))
        elif t == "layer":
            out.append((40, f"#{oid} is now drawn in front of #{e.get('other')} where they overlap"))
        elif t == "reshape":
            if oid in main:
                continue
            b = f.info(oid, "cur")
            why = "colour" if e.get("why") in ("feature", "recolour") else e.get("why")
            if (oid, why) in done_reshape:
                continue
            done_reshape.add((oid, why))
            n = reshapes.get((oid, why), e.get("cells"))
            sz = f.size(oid)
            pr = 30 + min(20, int(5 * math.log2(1 + (n or 1))))
            if why == "grow":
                out.append((pr, f"#{oid} grew by {n} cell(s): now {f.desc(b)} at {_rt(b.get('box') if b else None)}"))
            elif why == "shrink":
                out.append((pr, f"#{oid} lost {n} cell(s): now {f.desc(b)} at {_rt(b.get('box') if b else None)}"))
            else:
                out.append((pr + 10, f"#{oid}: {n} of its cells changed colour (now {f.desc(b)})"))
        elif t == "forget":
            a = f.info(oid, "prev")
            out.append((20, f"#{oid} ({f.desc(a, vis=False)}) not visible for 8 boards in a row; no longer tracked"))

    # recolours: a pair that swaps colours between two same-shape objects may be a jump
    used_rc: set = set()
    for i, e in enumerate(recolors):
        if i in used_rc:
            continue
        oid = e["id"]
        mp = e.get("map") or {}
        a = f.info(oid, "prev")
        partner = None
        if len(mp) == 1:
            (c_from, c_to), = ((int(k), int(v)) for k, v in mp.items())
            for j, e2 in enumerate(recolors):
                if j == i or j in used_rc:
                    continue
                mp2 = e2.get("map") or {}
                if len(mp2) != 1:
                    continue
                (d_from, d_to), = ((int(k), int(v)) for k, v in mp2.items())
                a2 = f.info(e2["id"], "prev")
                if d_from == c_to and d_to == c_from and a and a2 and a.get("shape") == a2.get("shape"):
                    partner = (j, e2, a2)
                    break
        if partner is not None:
            j, e2, a2 = partner
            used_rc.add(i); used_rc.add(j)
            out.append((89, f"#{oid} at {_rt(a.get('box'))} took colour {f.col(c_to)} and #{e2['id']} at {_rt(a2.get('box'))} took colour "
                            f"{f.col(c_from)} (same shape): the {f.names.get(c_from, c_from)} thing jumped from {_rt(a.get('box'))} to "
                            f"{_rt(a2.get('box'))}, or both recoloured (not distinguishable)"))
            touched.add(str(e2["id"]))
            continue
        ch = ", ".join(f"{f.col(int(k))} -> {f.col(v)}" for k, v in mp.items())
        b = f.info(oid, "cur")
        out.append((75 + min(10, int(math.log2(1 + f.size(oid)))),
                    f"#{oid} recoloured {ch} (now {f.desc(b)} at {_rt(b.get('box') if b else None)})"))

    # moves, grouped by offset; identical objects that may have swapped
    shift_cands = {k for k in moves}
    sc = next((e for e in edits if e.get("type") == "scroll"), None)
    if sc:
        shift_cands.add((int(sc.get("dr", 0)), int(sc.get("dc", 0))))
    shift = board_shift(before, after, f.rects, shift_cands) if (before is not None and after is not None) else None
    swapped: set = set()
    all_moves = [e for es in moves.values() for e in es]
    for i, e in enumerate(all_moves):
        for e2 in all_moves[i + 1:]:
            a, a2 = f.info(e["id"], "prev"), f.info(e2["id"], "prev")
            if not a or not a2 or a.get("shape") != a2.get("shape"):
                continue
            if list(e["to"][:2]) == list(e2["from"][:2]) and list(e2["to"][:2]) == list(e["from"][:2]):
                swapped.add(e["id"]); swapped.add(e2["id"])
                out.append((60, f"#{e['id']} and #{e2['id']} (identical: {f.desc(a, vis=False)}) at {_rt(a.get('box'))} and "
                                f"{_rt(a2.get('box'))} exchanged positions, or neither moved (not distinguishable)"))
    for (dr, dc), es in moves.items():
        es = [e for e in es if e["id"] not in swapped]
        if not es:
            continue
        if shift and (dr, dc) == (shift["dr"], shift["dc"]):
            continue          # reported by the shift line
        items = []
        biggest = 0
        for e in es:
            oid = e["id"]
            a = f.info(oid, "prev")
            biggest = max(biggest, f.size(oid))
            b_box = None
            if a and a.get("box"):
                b_box = [a["box"][0] + dr, a["box"][1] + dr, a["box"][2] + dc, a["box"][3] + dc]
            front = " (now in front)" if e.get("front") else ""
            items.append(f"#{oid} ({f.desc(a, vis=False)}) {_rt(a.get('box') if a else None)} -> {_rt(b_box)}{front}{_off(b_box, shape)}")
        pr = 90 + min(10, int(math.log2(1 + biggest)))
        if len(items) <= 3:
            out.append((pr, f"moved ({dr:+d} rows, {dc:+d} cols): " + "; ".join(items)))
        else:
            ids = ", ".join(f"#{e['id']}" for e in es[:12]) + (", ..." if len(es) > 12 else "")
            out.append((pr, f"{len(es)} objects moved together ({dr:+d} rows, {dc:+d} cols): {ids}; e.g. " + "; ".join(items[:2])))
    if shift:
        same = [e["id"] for e in moves.get((shift["dr"], shift["dc"]), [])]
        stayed = [int(k) for k, v in f.cur.items() if k in f.prev and int(k) not in same and k not in touched
                  and (v.get("vis") or 0) > 0 and f.size(int(k)) > 2]
        line = (f"most of the board shifted by ({shift['dr']:+d} rows, {shift['dc']:+d} cols): {shift['explained']} of "
                f"{shift['changed']} changed cells are explained by that shift")
        if shift["edge_cells"]:
            line += f"; {shift['edge_cells']} new cells entered at the {shift['edge']} edge"
        if same:
            line += "; shifted with it: " + ", ".join(f"#{i}" for i in same[:10]) + (", ..." if len(same) > 10 else "")
        if stayed:
            line += "; did not shift: " + ", ".join(f"#{i}" for i in stayed[:8]) + (", ..." if len(stayed) > 8 else "")
        out.append((99, line))

    # objects that stayed in the scene but stopped / started showing
    hidden: dict[tuple, list] = {}
    shown: list[str] = []
    small_hidden = small_shown = 0
    for k, b in f.cur.items():
        if k in touched:
            continue
        a = f.prev.get(k)
        if not a or f.ind(int(k)):
            continue
        if a.get("vis", 0) > 0 and b.get("vis", 0) == 0:
            if f.size(int(k)) <= 2:
                small_hidden += 1
                continue
            hidden.setdefault(tuple(f.covered.get(k) or []), []).append(f"#{k} ({f.desc(a, vis=False)} at {_rt(a.get('box'))})")
        elif a.get("vis", 0) == 0 and b.get("vis", 0) > 0:
            if f.size(int(k)) <= 2:
                small_shown += 1
                continue
            shown.append(f"#{k} ({f.desc(b)} at {_rt(b.get('box'))})")
    if small_hidden:
        dust["became covered"] = dust.get("became covered", 0) + small_hidden
    if small_shown:
        dust["became visible again"] = dust.get("became visible again", 0) + small_shown
    for cov, items in hidden.items():
        by = (" by " + ", ".join(f"#{c}" for c in cov)) if cov else ""
        if len(items) == 1:
            out.append((65, f"{items[0]} is no longer visible: its known cells are covered{by} (kept as a hidden object)"))
        else:
            out.append((65, "no longer visible, their known cells covered" + by + " (kept as hidden objects): " + "; ".join(items[:6])
                        + ("; ..." if len(items) > 6 else "")))
    if shown:
        out.append((66, "visible again: " + "; ".join(shown[:6]) + ("; ..." if len(shown) > 6 else "")))
    if dust:
        out.append((10, "small objects (1-2 cells): " + ", ".join(f"{n} {k}" for k, n in dust.items())))
    return out


def order_and_cap(scored: list[tuple[int, str]], max_lines: int, max_chars: int) -> tuple[list[str], int]:
    """Most important first; stop at the line or character budget. Returns (lines, omitted count)."""
    ordered = [ln for _p, ln in sorted(scored, key=lambda t: -t[0])]
    out: list[str] = []
    used = 0
    for ln in ordered:
        if len(out) >= max_lines or (out and used + len(ln) > max_chars):
            break
        out.append(ln)
        used += len(ln) + 1
    return out, len(ordered) - len(out)


# ---------------------------------------------------------------- level start

def level_lines(prev: dict | None, cur: dict, names: dict[int, str] | None = None, cap: int = 10) -> list[str]:
    """Summary of a new level's first board as the tracker read it (the inventory lists the objects themselves)."""
    f = Fmt(prev, cur, None, names or {})
    edits = cur.get("edits") or []
    kept = [e["id"] for e in edits if e.get("type") == "continue"]
    kind = [(e["id"], e.get("kindOf"), e.get("like_desc")) for e in edits if e.get("type") == "appear" and e.get("how") == "kind"]
    revived = [e["id"] for e in edits if e.get("type") == "appear" and e.get("how") == "library"]
    new = [e["id"] for e in edits if e.get("type") == "appear" and e.get("how") in ("peel", "cell")]

    def ids(lst):
        return ", ".join(f"#{i}" for i in lst[:cap]) + (f", ... ({len(lst)} in all)" if len(lst) > cap else "")
    out = [f"read the new level's board as {len(f.cur)} objects (ids continue from the previous level; the inventory below lists them)."]
    if kept:
        out.append(f"Same shape at the same place as on the previous level: {ids(kept)}.")
    if revived:
        out.append(f"Same shape as an earlier object of this level: {ids(revived)}.")
    if kind:
        groups: dict[Any, list] = {}
        descs: dict[Any, Any] = {}
        for i, k, ld in kind:
            groups.setdefault(k, []).append(i)
            if ld:
                descs[k] = ld
        parts = []
        for k, ids_ in list(groups.items())[:8]:
            lst = ", ".join(f"#{i}" for i in ids_[:6]) + (f", ... ({len(ids_)} in all)" if len(ids_) > 6 else "")
            ld = descs.get(k)
            dtxt = ""
            if ld:
                cols = ld.get("colors") or []
                dtxt = ": " + (" + ".join(f"{f.col(c)} {n}" for c, n in cols[:3]) + " cells")
            parts.append(f"{lst} (like #{k}{dtxt})")
        out.append("Same shape and colours as an object seen on an earlier level: " + "; ".join(parts)
                   + (f"; ... ({len(groups) - 8} more kinds)" if len(groups) > 8 else "") + ".")
    if new:
        out.append(f"Shapes not seen before: {ids(new)}.")
    return out


def _shape_text(o: dict) -> str:
    r = o.get("rect")
    if not r:
        return f"{o.get('size', 0)} cells"
    hh, ww = r[1] - r[0], r[3] - r[2]
    full = o.get("full_size", o.get("size", 0))
    size = o.get("size", 0)
    if size == hh * ww:
        base = f"{hh}x{ww} block"
    elif hh == 1 or ww == 1:
        base = f"{size} cells in a {hh}x{ww} line" if size != hh * ww else f"{hh}x{ww} line"
    else:
        base = f"{size} cells in {hh}x{ww}"
    if full > size:
        fr = o.get("full_rect") or r
        base += f" (completed behind other objects to {full} cells, {fr[1] - fr[0]}x{fr[3] - fr[2]})"
    return base


def inventory_text(objs: list[dict], names: dict[int, str], shape, max_groups: int = 80) -> str:
    """Level-start inventory from the scene: objects grouped by identical shape and colours, ids included."""
    def col(c):
        return f"{c} {names.get(int(c), str(c))}"

    def colours(o):
        cols = o.get("colors") or [[o.get("color"), o.get("size")]]
        if len(cols) == 1:
            return f"color {col(cols[0][0])}"
        return "colours " + " + ".join(f"{col(c)} {n}" for c, n in cols[:4]) + (" + ..." if len(cols) > 4 else "")

    bg = [o for o in objs if o.get("is_background")]
    rest = [o for o in objs if not o.get("is_background")]
    groups: dict[Any, list] = {}
    for o in rest:
        groups.setdefault(o.get("hash"), []).append(o)
    ordered = sorted(groups.values(), key=lambda g: (-max(x.get("size", 0) for x in g), -len(g)))
    lines = []
    head = f"Objects (layered tracker; ids persist across actions, objects() in run has every field): {len(rest)}"
    if bg:
        b = bg[0]
        head += (f" (plus background #{b['id']} {colours(b)}, {_shape_text(b)}" +
                 (f", {b.get('visible', b.get('size'))} visible" if b.get("visible", b.get("size")) != b.get("size") else "") + ")")
    for g in ordered[:max_groups]:
        ex = g[0]
        tag = f"[shape {ex.get('hash', 0) & 0xFFFFFF:06x}]"
        if len(g) == 1:
            o = ex
            extra = []
            if o.get("visible", o.get("size")) < o.get("size", 0):
                extra.append(f"{o.get('visible')} of {o.get('size')} cells visible" + (f" (behind #{', #'.join(map(str, o['covered_by']))})" if o.get("covered_by") else ""))
            if o.get("same_kind_as") is not None:
                extra.append(f"same shape and colours as #{o['same_kind_as']} earlier")
            lines.append(f"  #{o['id']} {colours(o)} {_shape_text(o)} {tag} {_rt(o.get('rect'))}" + (" (" + "; ".join(extra) + ")" if extra else ""))
        else:
            where = ", ".join(f"#{o['id']} {_rt(o.get('rect'))}" for o in g[:20]) + (f" (+{len(g) - 20} more)" if len(g) > 20 else "")
            hidden = [o for o in g if o.get("visible", o.get("size")) < o.get("size", 0)]
            lines.append(f"  {len(g)} x {colours(ex)} {_shape_text(ex)} {tag}: {where}"
                         + (f" ({len(hidden)} of them partly behind other objects)" if hidden else ""))
    if len(ordered) > max_groups:
        lines.append(f"  ... and {len(ordered) - max_groups} more kinds (objects() lists everything)")
    return head + "\n" + "\n".join(lines)


def block(lines: list[str], header: str, cap: int, max_chars: int = 1500) -> list[str]:
    if not lines:
        return []
    out = [header]
    used = 0
    shown = 0
    for ln in lines:
        if shown >= cap or (shown and used + len(ln) > max_chars):
            break
        out.append(f"    {ln}")
        used += len(ln)
        shown += 1
    if shown < len(lines):
        out.append(f"    ... and {len(lines) - shown} more (changes() in run lists all of them)")
    return out

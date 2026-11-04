"""Scene tracker: a Python port of the teammate's PEEL decomposition + TRACK object tracker
(arc_object_tracker.html, first pass only - causal: every frame is explained from the past frames alone).

PEEL reads a frame as layered objects: single-colour pieces, each described by its cheapest shape (filled polygon,
outline or bag of lines) and allowed to extend behind the objects removed before it (occlusion), removed front to back.
TRACK explains every new frame as cheap edits of the previous scene (moves, turns, recolours, disappearances, splits,
merges, layer swaps, growth/shrink, new objects), priced in the same parameter currency, so object identities persist.

Faithful port of the JS (same decisions, same tie-breaks, same ids); kernels in scene_kernels.py.
Used by the harness through a worker process (scene_worker.py).
"""
from __future__ import annotations

import math
from functools import cmp_to_key, reduce
from typing import Any

import numpy as np

from . import scene_kernels as K

INF = math.inf

# ====================================================================== PEEL

_EMPTY_U8 = np.zeros(0, np.uint8)


class Fit:
    __slots__ = ("kind", "cost", "r0", "c0", "bh", "bw", "mask", "size")

    def __init__(self, kind, cost, r0, c0, bh, bw, mask, size):
        self.kind = kind
        self.cost = cost
        self.r0 = r0
        self.c0 = c0
        self.bh = bh
        self.bw = bw
        self.mask = mask
        self.size = size


def fit_cells(O: np.ndarray, U: np.ndarray | None, w: int) -> Fit:
    if U is None:
        kind, cost, r0, c0, bh, bw, mask, cells = K.fit_cells(O, _EMPTY_U8, False, w)
    else:
        kind, cost, r0, c0, bh, bw, mask, cells = K.fit_cells(O, U, True, w)
    return Fit(K.KIND_NAMES[kind], int(cost), int(r0), int(c0), int(bh), int(bw), mask, int(cells))


def _bbox(O: np.ndarray, w: int) -> list[int]:
    r = O // w
    c = O - r * w
    return [int(r[0]), int(r[-1]), int(c.min()), int(c.max())] if O.shape[0] else [INF, -1, INF, -1]


def _box_overlap(a, b) -> bool:
    return a[0] <= b[1] and b[0] <= a[1] and a[2] <= b[3] and b[2] <= a[3]


def _box_gap(a, b) -> int:
    return max(0, b[0] - a[1] - 1, a[0] - b[1] - 1) + max(0, b[2] - a[3] - 1, a[2] - b[3] - 1)


def _cells_in_box(O: np.ndarray, box, w: int) -> bool:
    r = O // w
    c = O - r * w
    return bool(np.any((r >= box[0]) & (r <= box[1]) & (c >= box[2]) & (c <= box[3])))


class _PObj:
    __slots__ = ("wid", "color", "O", "n", "first", "bbox", "fit", "alive", "stale", "removed_round", "merged_into")


class Piece:
    """A peel object as TRACK uses it."""
    __slots__ = ("id", "color", "size", "cells_r", "cells_c", "rect", "layer", "rep_kind", "rep_cost", "full")

    def __init__(self, id, color, size, cells_r, cells_c, rect, layer, rep_kind, rep_cost, full):
        self.id = id
        self.color = color
        self.size = size
        self.cells_r = cells_r
        self.cells_c = cells_c
        self.rect = rect
        self.layer = layer
        self.rep_kind = rep_kind
        self.rep_cost = rep_cost
        self.full = full      # Fit (r0, c0, bh, bw, mask)


class PeelResult:
    __slots__ = ("objects", "cid", "h", "w", "colors", "sizes")


def peel_decompose(grid: np.ndarray) -> PeelResult:
    """PEEL.decompose(grid, { order: 'lowest' })."""
    h, w = int(grid.shape[0]), int(grid.shape[1])
    HW = h * w
    flat = np.ascontiguousarray(grid, dtype=np.int64).ravel()
    U = np.zeros(HW, np.uint8)
    owner = np.full(HW, -1, np.int64)
    objs: list[_PObj] = []

    def make(O, color):
        o = _PObj()
        o.wid = len(objs)
        o.color = color
        o.O = O
        o.n = int(O.shape[0])
        o.first = int(O[0])
        o.bbox = _bbox(O, w)
        o.fit = None
        o.alive = True
        o.stale = False
        o.removed_round = -1
        o.merged_into = -1
        owner[O] = o.wid
        objs.append(o)
        return o

    LAZY_AREA = 1024

    def score_of(o):
        return ((2 if o.n == 1 else 3) + 1) if o.stale else o.fit.cost

    def refresh(o):
        if o.stale:
            o.stale = False
            o.fit = fit_cells(o.O, U, w)

    cid0, ncomp = K.components(flat, h, w)
    order = np.argsort(cid0, kind="stable")
    counts = np.bincount(cid0, minlength=ncomp)
    starts = np.concatenate(([0], np.cumsum(counts)))
    for k in range(ncomp):
        O = order[starts[k]:starts[k + 1]].astype(np.int64)
        make(O, int(flat[O[0]]))
    for o in objs:
        o.fit = fit_cells(o.O, U, w)
    alive = [o for o in objs if o.alive]

    pair_cache: dict[tuple[int, int], tuple] = {}

    def area(b):
        return (b[1] - b[0] + 1) * (b[3] - b[2] + 1)

    def eval_pair(a, b):
        refresh(objs[a])
        refresh(objs[b])
        key = (a, b)
        e = pair_cache.get(key)
        if e is None:
            O = np.sort(np.concatenate((objs[a].O, objs[b].O)))
            e = (O, fit_cells(O, U, w), _bbox(O, w))
            pair_cache[key] = e
        return objs[a].fit.cost + objs[b].fit.cost - e[1].cost

    def try_merges(pairs):
        lst = [(a, b) if a < b else (b, a) for a, b in pairs]
        while True:
            best = None
            best_s = 0
            for a, b in lst:
                if not objs[a].alive or not objs[b].alive:
                    continue
                s = eval_pair(a, b)
                if s > best_s:
                    best_s = s
                    best = (a, b)
            if best is None:
                return
            a, b = best
            A, B = objs[a], objs[b]
            e = pair_cache[(a, b)]
            A.alive = B.alive = False
            m = make(e[0], A.color)
            A.merged_into = B.merged_into = m.wid
            m.fit = e[1]
            partners: dict[int, None] = {}
            for x, y in lst:
                if x == a or x == b:
                    partners[y] = None
                if y == a or y == b:
                    partners[x] = None
            lst = [(x, y) for x, y in lst if objs[x].alive and objs[y].alive]
            for p in partners:
                if objs[p].alive:
                    lst.append((p, m.wid))

    NEAREST = 6

    def nearest_pairs(group, pairs, seen_p):
        def pick(o, cands):
            near = sorted((_box_gap(o.bbox, q.bbox), q.wid) for q in cands if q is not o)[:NEAREST]
            for _g, q in near:
                a, b = min(o.wid, q), max(o.wid, q)
                if (a, b) not in seen_p:
                    seen_p.add((a, b))
                    pairs.append((a, b))
        if len(group) <= 4 * NEAREST:
            for o in group:
                pick(o, group)
            return
        B = 8
        buckets: dict[int, list] = {}
        max_ring = math.ceil(max(h, w) / B)

        def at(o):
            return ((o.bbox[0] + o.bbox[1]) >> 1) // B, ((o.bbox[2] + o.bbox[3]) >> 1) // B
        for o in group:
            br, bc = at(o)
            buckets.setdefault(br * 1000 + bc, []).append(o)
        for o in group:
            br, bc = at(o)
            cands = []
            ring = 0
            while ring <= max_ring and len(cands) <= 2 * NEAREST:
                for r in range(br - ring, br + ring + 1):
                    for c in range(bc - ring, bc + ring + 1):
                        if max(abs(r - br), abs(c - bc)) != ring:
                            continue
                        lst = buckets.get(r * 1000 + c)
                        if lst:
                            cands.extend(lst)
                ring += 1
            pick(o, cands)

    def merge_pairs_around(seed):
        touch = K.flood_touch(seed, U, owner, h, w, 0, 0, 0, 0, False)
        by_color: dict[int, list] = {}
        for wid in np.nonzero(touch)[0].tolist():
            o = objs[wid]
            by_color.setdefault(o.color, []).append(o)
        pairs: list = []
        seen_p: set = set()
        for group in by_color.values():
            if len(group) > 1:
                nearest_pairs(group, pairs, seen_p)
        return pairs

    rnd = 0
    while alive:
        while True:
            X = min(alive, key=lambda o: (score_of(o), o.n, o.first))
            if not X.stale:
                break
            refresh(X)
        X.alive = False
        X.removed_round = rnd
        U[X.O] = 1
        owner[X.O] = -1
        alive = [o for o in objs if o.alive]
        xb = X.bbox
        for y in alive:
            yb = y.bbox
            if not (xb[0] <= yb[1] and yb[0] <= xb[1] and xb[2] <= yb[3] and yb[2] <= xb[3]):
                continue
            if not _cells_in_box(X.O, yb, w):
                continue
            if area(yb) > LAZY_AREA:
                y.stale = True
                continue
            y.fit = fit_cells(y.O, U, w)
        if pair_cache:
            for k in [k for k, e in pair_cache.items() if _box_overlap(e[2], xb) and _cells_in_box(X.O, e[2], w)]:
                del pair_cache[k]
        pairs = merge_pairs_around(X.O)
        if pairs:
            try_merges(pairs)
            alive = [o for o in objs if o.alive]
        rnd += 1

    removed = sorted((o for o in objs if o.removed_round >= 0), key=lambda o: o.first)
    cid = np.zeros(HW, np.int64)
    pieces = []
    for pid, o in enumerate(removed):
        cid[o.O] = pid
        b = o.bbox
        f = o.fit
        pieces.append(Piece(pid, o.color, o.n, o.O // w, o.O % w, [b[0], b[1] + 1, b[2], b[3] + 1], o.removed_round,
                            f.kind, f.cost, f))
    res = PeelResult()
    res.objects = pieces
    res.cid = cid
    res.h = h
    res.w = w
    res.colors = np.array([p.color for p in pieces], np.int64)
    res.sizes = np.array([p.size for p in pieces], np.int64)
    return res


# ====================================================================== TRACK

COST = {"move": 2, "rotate": 1, "recolor": 1, "layer": 1, "split": 1, "disappear": 1, "merge": 1, "place": 3}
DEFAULTS = {"range": 16, "keep": 3, "mergeAfter": 2, "levelFrac": 0.15, "freshFrac": 0.4, "rereadFrac": 0.15,
            "hiddenMax": 0.25, "scroll": 32, "scrollMin": 64, "scrollFrac": 0.5, "featureMax": 9,
            "rotateMax": 400, "moveMax": 1500, "sample": 64, "sampleMax": 256, "forgetAfter": 8,
            "libMin": 3, "libMax": 400}
N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
FOLLOW_BIAS = 0.5
BATCH_FROM = 200
TYPE_RANK = {"move": 0, "revive": 1, "kind": 2}


def pair_key(a: int, b: int) -> str:
    return f"{a},{b}" if a < b else f"{b},{a}"


class Tpl:
    """An object's shape in its own box: cells idx = y * tw + x (sorted), colour and seen flag per cell."""
    __slots__ = ("th", "tw", "n", "nSeen", "idx", "col", "seen", "at", "ys", "xs", "orc", "cost", "parts", "base",
                 "hash", "distinct", "sample", "cbox", "colset")

    def __init__(self):
        self.orc = {}
        self.cost = -1
        self.parts = None
        self.base = None
        self.hash = None
        self.distinct = None
        self.sample = None
        self.cbox = None
        self.colset = None


def make_tpl(th: int, tw: int, Y, X, C, S) -> Tpl:
    Y = np.asarray(Y, np.int64)
    X = np.asarray(X, np.int64)
    C = np.asarray(C, np.int64)
    S = np.asarray(S, np.int64)
    order = np.lexsort((X, Y))
    t = Tpl()
    t.th = int(th)
    t.tw = int(tw)
    t.ys = Y[order]
    t.xs = X[order]
    t.col = C[order]
    t.seen = (S[order] != 0).astype(np.int64)
    t.idx = t.ys * t.tw + t.xs
    t.n = int(Y.shape[0])
    t.nSeen = int(t.seen.sum())
    at = np.full(max(0, t.th * t.tw), -1, np.int64)
    if t.n:
        at[t.idx] = np.arange(t.n, dtype=np.int64)
    t.at = at
    return t


def tpl_from_cells(Y, X, C, S):
    Y = np.asarray(Y, np.int64)
    X = np.asarray(X, np.int64)
    if Y.shape[0] == 0:
        raise RuntimeError("empty template")   # the JS would fail here too (typed array of length Infinity)
    y0, x0 = int(Y.min()), int(X.min())
    y1, x1 = int(Y.max()), int(X.max())
    return make_tpl(y1 - y0 + 1, x1 - x0 + 1, Y - y0, X - x0, C, S), y0, x0


def oriented(t: Tpl, k: int) -> Tpl:
    if not k:
        return t
    o = t.orc.get(k)
    if o is not None:
        return o
    y, x, H, W = t.ys, t.xs, t.th, t.tw
    if k == 1:
        ny, nx = x, H - 1 - y
    elif k == 2:
        ny, nx = H - 1 - y, W - 1 - x
    elif k == 3:
        ny, nx = W - 1 - x, y
    elif k == 4:
        ny, nx = y, W - 1 - x
    elif k == 5:
        ny, nx = W - 1 - x, H - 1 - y
    elif k == 6:
        ny, nx = H - 1 - y, x
    else:
        ny, nx = x, y
    odd = (k & 1) == 1
    o = make_tpl(t.tw if odd else t.th, t.th if odd else t.tw, ny, nx, t.col, t.seen)
    o.base = t
    t.orc[k] = o
    return o


def tpl_hash(T: Tpl) -> int:
    if T.hash is None:
        T.hash = int(K.tpl_hash(T.th, T.tw, T.idx, T.col))
    return T.hash


def distinct_orients(t: Tpl) -> list[int]:
    if t.distinct is None:
        seen = set()
        out = []
        for k in range(8):
            hh = tpl_hash(oriented(t, k))
            if hh not in seen:
                seen.add(hh)
                out.append(k)
        t.distinct = out
    return t.distinct


def tpl_cost(T: Tpl) -> int:
    t = T.base if T.base is not None else T
    if t.cost < 0 or not t.parts:
        s = 0
        parts = []
        for c in sorted(set(t.col.tolist())):
            cells = t.idx[t.col == c]
            f = fit_cells(cells, None, t.tw)
            s += f.cost
            parts.append((c, f.kind, f.cost, int(cells.shape[0])))
        t.cost = s
        t.parts = parts
    return t.cost


def sample_of(T: Tpl, opts) -> tuple:
    if T.sample is not None:
        return T.sample
    if T.n <= opts["sample"]:
        pick = np.arange(T.n, dtype=np.int64)
    else:
        sel = np.zeros(T.n, bool)
        for dy, dx in N4:
            yy = T.ys + dy
            xx = T.xs + dx
            inside = (yy >= 0) & (xx >= 0) & (yy < T.th) & (xx < T.tw)
            j = np.full(T.n, -1, np.int64)
            j[inside] = T.at[yy[inside] * T.tw + xx[inside]]
            diff = np.ones(T.n, bool)
            ok = j >= 0
            diff[ok] = T.col[j[ok]] != T.col[ok]
            sel |= (j < 0) | diff
        pick = np.nonzero(sel)[0].astype(np.int64)
    if pick.shape[0] > opts["sampleMax"]:
        step = pick.shape[0] / opts["sampleMax"]
        pick = pick[np.floor(np.arange(opts["sampleMax"]) * step).astype(np.int64)]
    T.sample = (T.ys[pick].copy(), T.xs[pick].copy(), T.col[pick].copy())
    return T.sample


def colour_boxes(T: Tpl) -> list[tuple[int, tuple]]:
    """seen cells of each colour: [(colour, (n, y0, y1, x0, x1))] in order of first appearance."""
    if T.cbox is None:
        sel = T.seen != 0
        cols = T.col[sel]
        ys = T.ys[sel]
        xs = T.xs[sel]
        out = []
        if cols.shape[0]:
            u, first = np.unique(cols, return_index=True)
            for c in u[np.argsort(first)].tolist():
                m = cols == c
                yy = ys[m]
                xx = xs[m]
                out.append((c, (int(m.sum()), int(yy.min()), int(yy.max()), int(xx.min()), int(xx.max()))))
        T.cbox = out
    return T.cbox


class SObj:
    """Scene object: template t (base shape), position r, c of the oriented template, orientation k, layer z."""
    __slots__ = ("id", "t", "r", "c", "k", "z", "_box")

    def __init__(self, id, t, r, c, k, z):
        self.id = id
        self.t = t
        self.r = r
        self.c = c
        self.k = k
        self.z = z
        self._box = None

    def copy(self, **kw):
        n = SObj(self.id, self.t, self.r, self.c, self.k, self.z)
        for key, v in kw.items():
            setattr(n, key, v)
        return n


def T_of(o: SObj) -> Tpl:
    return oriented(o.t, o.k)


def box_of(o: SObj):
    b = o._box
    if b is None:
        T = T_of(o)
        b = o._box = (o.r, o.r + T.th - 1, o.c, o.c + T.tw - 1)
    return b


def union_box(a, b):
    return (min(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), max(a[3], b[3]))


class LibEntry:
    __slots__ = ("t", "id", "key", "levels", "level")

    def __init__(self, t, id, key, levels, level):
        self.t = t
        self.id = id
        self.key = key
        self.levels = levels
        self.level = level

    def copy(self):
        return LibEntry(self.t, self.id, self.key, set(self.levels), self.level)


class Ctr:
    __slots__ = ("t", "n")

    def __init__(self, t, n):
        self.t = t
        self.n = n


class Track:
    __slots__ = ("id", "spans", "how", "parents", "children", "n", "end", "kind", "levels", "t", "cur", "stable")

    def copy(self):
        tr = Track()
        tr.id = self.id
        tr.spans = [list(s) for s in self.spans]
        tr.how = self.how
        tr.parents = list(self.parents)
        tr.children = list(self.children)
        tr.n = dict(self.n)
        tr.end = self.end
        tr.kind = self.kind
        tr.levels = set(self.levels)
        tr.t = self.t
        tr.cur = Ctr(self.cur.t, self.cur.n) if self.cur is not None else None
        tr.stable = Ctr(self.stable.t, self.stable.n) if self.stable is not None else None
        return tr


class Prop:
    __slots__ = ("type", "id", "r", "c", "k", "cost", "map", "follow", "seq", "obj", "obj_set", "box", "ub", "stale",
                 "ubStale", "trial", "gain", "front")

    def __init__(self, type, id, cost, r=None, c=None, k=None, map=None, follow=False, seq=0):
        self.type = type
        self.id = id
        self.cost = cost
        self.r = r
        self.c = c
        self.k = k
        self.map = map
        self.follow = follow
        self.seq = seq
        self.obj = None
        self.obj_set = False
        self.box = None
        self.ub = 0
        self.stale = True
        self.ubStale = False
        self.trial = None
        self.gain = 0
        self.front = False


class Cand:
    __slots__ = ("type", "src", "k", "r", "c", "ids", "cells", "whole", "n")


class Area:
    __slots__ = ("P", "r0", "c0", "ph", "pw")


class View:
    __slots__ = ("col", "own", "zt", "uid", "ucol")


class Tracker:
    """TRACK.runPass, first pass, one frame at a time: step(grid, level, step_last) -> record of the frame."""

    def __init__(self, h: int = 64, w: int = 64, opts: dict | None = None, first_id: int = 1):
        self.opts = dict(DEFAULTS)
        if opts:
            self.opts.update(opts)
        self.h, self.w, self.HW = h, w, h * w
        self.objs: list[SObj] = []
        self.next_id = first_id
        self.F = None
        self.grid = None
        self.view = None
        self.fi = 0
        self.index: dict[int, SObj] = {}
        self.tracks: dict[int, Track] = {}
        self.lib: dict[tuple, LibEntry] = {}
        self.co_move: dict[str, int] = {}
        self.blocked: set[str] = set()
        self.hidden_for: dict[int, int] = {}
        self.level = 0
        self.lib_version = 0
        self.lib_cache = None
        self.unknown_below = 0
        self.pending = None
        self._ver = 0
        self._pack = None
        self._pack_ver = -1

    # ------------------------------------------------------------ scene bookkeeping
    def add_obj(self, o):
        self.objs.append(o)
        self.index[o.id] = o
        self._ver += 1

    def drop_obj(self, id):
        self.objs = [o for o in self.objs if o.id != id]
        self.index.pop(id, None)
        self.hidden_for.pop(id, None)
        self._ver += 1

    def swap_obj(self, o):
        self.objs = [o if x.id == o.id else x for x in self.objs]
        self.index[o.id] = o
        self._ver += 1

    def set_objs(self, objs):
        self.objs = objs
        self.index = {o.id: o for o in objs}
        self._ver += 1

    def normalise_z(self):
        self.objs.sort(key=lambda o: (o.z, o.id))
        for i, o in enumerate(self.objs):
            o.z = i
        self._ver += 1

    def pack(self):
        """Concatenated arrays of the scene (in list order) for the kernels."""
        if self._pack is not None and self._pack_ver == self._ver:
            return self._pack
        objs = self.objs
        n = len(objs)
        s_id = np.empty(n, np.int64)
        s_r = np.empty(n, np.int64)
        s_c = np.empty(n, np.int64)
        s_z = np.empty(n, np.float64)
        s_th = np.empty(n, np.int64)
        s_tw = np.empty(n, np.int64)
        s_atoff = np.zeros(n + 1, np.int64)
        s_coff = np.zeros(n + 1, np.int64)
        ats, yss, xss, cols = [], [], [], []
        maxid = 0
        for j, o in enumerate(objs):
            T = T_of(o)
            s_id[j] = o.id
            s_r[j] = o.r
            s_c[j] = o.c
            s_z[j] = o.z
            s_th[j] = T.th
            s_tw[j] = T.tw
            s_atoff[j + 1] = s_atoff[j] + T.at.shape[0]
            s_coff[j + 1] = s_coff[j] + T.n
            ats.append(T.at)
            yss.append(T.ys)
            xss.append(T.xs)
            cols.append(T.col)
            if o.id > maxid:
                maxid = o.id
        z_by_id = np.full(maxid + 2, np.nan)
        if n:
            z_by_id[s_id] = s_z
        e = np.zeros(0, np.int64)
        p = (s_id, s_r, s_c, s_z, s_th, s_tw, s_atoff[:-1].copy(), np.concatenate(ats) if ats else e,
             s_coff[:-1].copy(), np.concatenate(cols) if cols else e, s_coff, np.concatenate(yss) if yss else e,
             np.concatenate(xss) if xss else e, z_by_id)
        self._pack = p
        self._pack_ver = self._ver
        return p

    def redraw(self):
        self._ver += 1   # z values may have been changed in place
        p = self.pack()
        order = np.lexsort((p[0], p[3]))[::-1].copy()     # back to front: z, then id, descending
        v = View()
        v.col, v.own, v.zt, v.uid, v.ucol = K.render_scene(order, p[1], p[2], p[0], p[3], p[10], p[11], p[12], p[9],
                                                            self.h, self.w)
        self.view = v

    def wrong_count(self) -> int:
        return int(np.count_nonzero(self.view.col != self.F))

    def in_grid(self, y, x) -> bool:
        return 0 <= y < self.h and 0 <= x < self.w

    def clip(self, b):
        r = (max(0, b[0]), min(self.h - 1, b[1]), max(0, b[2]), min(self.w - 1, b[3]))
        return r if r[0] <= r[1] and r[2] <= r[3] else None

    def covers(self, o, g) -> bool:
        T = T_of(o)
        y = g // self.w - o.r
        x = g % self.w - o.c
        return 0 <= y < T.th and 0 <= x < T.tw and T.at[y * T.tw + x] >= 0

    def cells_g(self, o, T=None):
        """grid positions of o's cells: (y, x, inside-grid mask, g (valid where inside))."""
        T = T or T_of(o)
        y = o.r + T.ys
        x = o.c + T.xs
        ins = (y >= 0) & (x >= 0) & (y < self.h) & (x < self.w)
        g = np.where(ins, y * self.w + x, 0)
        return y, x, ins, g

    def render_list(self, lst, rect):
        """renderRect: (col, own, RW) of the objects in lst inside rect."""
        r0, r1, c0, c1 = rect
        hits = [o for o in lst if _box_overlap(box_of(o), rect)]
        hits.sort(key=lambda o: (o.z, o.id), reverse=True)
        n = len(hits)
        if n:
            Ts = [T_of(o) for o in hits]
            off = np.zeros(n + 1, np.int64)
            np.cumsum([T.n for T in Ts], out=off[1:])
            col, own = K.render_rect(np.array([o.r for o in hits], np.int64), np.array([o.c for o in hits], np.int64),
                                     np.array([o.id for o in hits], np.int64), off,
                                     np.concatenate([T.ys for T in Ts]), np.concatenate([T.xs for T in Ts]),
                                     np.concatenate([T.col for T in Ts]), r0, r1, c0, c1)
        else:
            RW = c1 - c0 + 1
            col = np.full((r1 - r0 + 1) * RW, -1, np.int64)
            own = np.full((r1 - r0 + 1) * RW, -1, np.int64)
        return col, own, c1 - c0 + 1

    def trial(self, subst: dict, extra: list, box):
        rect = self.clip(box)
        if rect is None:
            return {"fixed": 0, "broken": 0, "rect": None}
        lst = []
        for o in self.objs:
            if o.id in subst:
                s = subst[o.id]
                if s is not None:
                    lst.append(s)
            else:
                lst.append(o)
        lst.extend(extra)
        col, own, RW = self.render_list(lst, rect)
        r0, r1, c0, c1 = rect
        w = self.w
        vcol = self.view.col.reshape(self.h, w)[r0:r1 + 1, c0:c1 + 1].ravel()
        Fs = self.F.reshape(self.h, w)[r0:r1 + 1, c0:c1 + 1].ravel()
        was = vcol != Fs
        is_ = col != Fs
        return {"fixed": int(np.count_nonzero(was & ~is_)), "broken": int(np.count_nonzero(~was & is_)), "rect": rect,
                "col": col, "own": own, "RW": RW}

    def commit(self, tr):
        if tr["rect"] is None:
            return
        r0, r1, c0, c1 = tr["rect"]
        h, w = self.h, self.w
        own = tr["own"]
        zt = np.full(own.shape[0], INF)
        for oid in np.unique(own[own >= 0]).tolist():
            zt[own == oid] = self.index[oid].z
        v = self.view
        v.col.reshape(h, w)[r0:r1 + 1, c0:c1 + 1] = tr["col"].reshape(r1 - r0 + 1, c1 - c0 + 1)
        v.own.reshape(h, w)[r0:r1 + 1, c0:c1 + 1] = own.reshape(r1 - r0 + 1, c1 - c0 + 1)
        v.zt.reshape(h, w)[r0:r1 + 1, c0:c1 + 1] = zt.reshape(r1 - r0 + 1, c1 - c0 + 1)
        v.uid.reshape(h, w)[r0:r1 + 1, c0:c1 + 1] = -2

    # ------------------------------------------------------------ library and tracks
    def add_lib(self, t, id, levels, latest):
        if t.nSeen < self.opts["libMin"]:
            return
        key = (tpl_hash(t), id)
        old = self.lib.get(key)
        e = old if old is not None else LibEntry(t, id, key, set(), latest)
        for l in levels:
            e.levels.add(l)
        e.level = latest
        self.lib_version += 1
        self.lib.pop(key, None)
        self.lib[key] = e
        if len(self.lib) > self.opts["libMax"]:
            del self.lib[next(iter(self.lib))]

    def born(self, o, how, parents=None, kind=None):
        tr = self.tracks.get(o.id)
        if tr is None:
            tr = Track()
            tr.id = o.id
            tr.spans = []
            tr.how = how
            tr.parents = list(parents or [])
            tr.children = []
            tr.n = {}
            tr.end = None
            tr.kind = kind
            tr.levels = set()
            tr.t = None
            tr.cur = None
            tr.stable = None
            self.tracks[o.id] = tr
        tr.spans.append([self.fi, self.fi])
        tr.end = None
        tr.t = o.t
        tr.levels.add(self.level)

    def died(self, o, why):
        tr = self.tracks.get(o.id)
        if tr is not None:
            tr.end = why
            tr.t = o.t
        lv = (list(tr.levels) + [self.level]) if tr is not None else [self.level]
        self.add_lib(o.t, o.id, lv, self.level)
        if tr is not None and tr.stable is not None and tr.stable.t is not o.t and tr.stable.n > 1:
            self.add_lib(tr.stable.t, o.id, list(tr.levels) + [self.level], self.level)

    def count(self, id, kind):
        tr = self.tracks.get(id)
        if tr is not None:
            tr.n[kind] = tr.n.get(kind, 0) + 1

    def from_peel(self, po: Piece, A: Area):
        f = po.full
        m = f.mask.reshape(f.bh, f.bw)
        ys, xs = np.nonzero(m)
        cid = A.P.cid.reshape(A.ph, A.pw)
        seen = (cid[f.r0 + ys, f.c0 + xs] == po.id).astype(np.int64)
        t = make_tpl(f.bh, f.bw, ys, xs, np.full(ys.shape[0], po.color, np.int64), seen)
        t.cost = po.rep_cost
        t.parts = [(po.color, po.rep_kind, po.rep_cost, t.n)]
        o = SObj(self.next_id, t, f.r0 + A.r0, f.c0 + A.c0, 0, 0)
        self.next_id += 1
        return o

    # ------------------------------------------------------------ reading an area from its own pixels
    def read_area(self, r0, r1, c0, c1) -> Area:
        A = Area()
        A.P = peel_decompose(self.grid[r0:r1 + 1, c0:c1 + 1])
        A.r0, A.c0, A.ph, A.pw = r0, c0, r1 - r0 + 1, c1 - c0 + 1
        return A

    def match_pieces(self, T: Tpl, r, c, A: Area):
        sel = T.seen != 0
        if not T.nSeen:
            return None
        ys = r + T.ys[sel] - A.r0
        xs = c + T.xs[sel] - A.c0
        if np.any((ys < 0) | (xs < 0) | (ys >= A.ph) | (xs >= A.pw)):
            return None
        j = ys * A.pw + xs
        ids = A.P.cid[j]
        good = A.P.colors[ids] == T.col[sel]
        if int(np.count_nonzero(~good)) > math.floor(T.nSeen * self.opts["hiddenMax"]):
            return None
        cells = j[good]
        if not cells.shape[0]:
            return None
        gid = ids[good]
        u, first = np.unique(gid, return_index=True)
        uids = u[np.argsort(first)]
        n = int(A.P.sizes[uids].sum())
        return uids.tolist(), cells, n == cells.shape[0]

    def whole_match(self, T, r, c, A):
        m = self.match_pieces(T, r, c, A)
        return m[0] if (m is not None and m[2]) else None

    def same_full(self, o, A):
        T = T_of(o)
        if T.n == 0 or np.any(T.col[1:] != T.col[0]):
            return None
        c0col = int(T.col[0])
        ys = o.r + T.ys - A.r0
        xs = o.c + T.xs - A.c0
        ins = (ys >= 0) & (xs >= 0) & (ys < A.ph) & (xs < A.pw)
        jj = np.where(ins, ys * A.pw + xs, 0)
        ok = ins & (A.P.colors[A.P.cid[jj]] == c0col)
        hit = np.nonzero(ok)[0]
        if not hit.shape[0]:
            return None
        pid = int(A.P.cid[jj[hit[0]]])
        f = A.P.objects[pid].full
        if int(f.mask.sum()) != T.n:
            return None
        yy = ys - f.r0
        xx = xs - f.c0
        if np.any((yy < 0) | (xx < 0) | (yy >= f.bh) | (xx >= f.bw)):
            return None
        if not np.all(f.mask[yy * f.bw + xx]):
            return None
        return [pid]

    def shape_index(self, entries, one_each):
        """[(src, t)] -> {colour: [(src, k, T, box)]}"""
        idx: dict[int, list] = {}
        shapes = set()
        for src, t in entries:
            if one_each:
                key = tpl_hash(t)
                if key in shapes:
                    continue
                shapes.add(key)
            for k in (distinct_orients(t) if t.n <= self.opts["rotateMax"] else [0]):
                T = oriented(t, k)
                for c, b in colour_boxes(T):
                    idx.setdefault(c, []).append((src, k, T, b))
        return idx

    def source_lists(self, with_scene):
        lc = self.lib_cache
        if lc is None or lc[0] != self.lib_version or lc[1] != self.level:
            mine, other = [], []
            for e in self.lib.values():
                (mine if self.level in e.levels else other).append((e, e.t))
            other.reverse()
            lc = self.lib_cache = (self.lib_version, self.level, self.shape_index(mine, False), self.shape_index(other, True))
        lists = [("revive", lc[2], True), ("kind", lc[3], False)]
        if with_scene:
            lists.insert(0, ("move", self.shape_index([(o, o.t) for o in self.objs], False), True))
        return lists

    def choose_matches(self, A: Area, need, lists, in_place):
        cands = []
        tried = set()
        P = A.P
        NP = len(P.objects)
        hm = self.opts["hiddenMax"]
        for pid in range(NP):
            if not need[pid]:
                continue
            po = P.objects[pid]
            top = A.r0 + po.rect[0]
            bot = A.r0 + po.rect[1] - 1
            lft = A.c0 + po.rect[2]
            rgt = A.c0 + po.rect[3] - 1
            for ltype, idx, shared in lists:
                for (src, k, T, b) in idx.get(po.color, ()):
                    bn, y0, y1, x0, x1 = b
                    if bn * (1 - hm) > po.size or (not shared and bn < po.size):
                        continue
                    if ltype == "move" and T.nSeen < 4:
                        continue
                    for r, c in ((top - y0, lft - x0), (bot - y1, rgt - x1), (top - y0, rgt - x1), (bot - y1, lft - x0)):
                        tk = (ltype, src.id, k, r, c)
                        if tk in tried:
                            continue
                        tried.add(tk)
                        m = self.match_pieces(T, r, c, A)
                        if ltype == "move":
                            copy = in_place is not None and src.id in in_place
                        else:
                            copy = ltype == "revive" and src.id in self.index
                        if m is None or (not m[2] and (not shared or copy)) or not all(need[i] for i in m[0]):
                            continue
                        cd = Cand()
                        cd.type = "kind" if copy else ltype
                        cd.src = src
                        cd.k = k
                        cd.r = r
                        cd.c = c
                        cd.ids = m[0]
                        cd.cells = m[1]
                        cd.whole = m[2]
                        cd.n = int(m[1].shape[0])
                        cands.append(cd)
        cands.sort(key=lambda p: (-p.n, -int(p.whole), TYPE_RANK[p.type], p.src.id, p.r, p.c, p.k))
        taken = np.zeros(A.pw * A.ph, np.uint8)
        used_src = set()
        out = []
        for cd in cands:
            if np.any(taken[cd.cells]):
                continue
            sid = ("o" if cd.type == "move" else "l", cd.src.id)
            if sid in used_src:
                if not cd.whole:
                    continue
                cd.type = "kind"
            if cd.type != "kind":
                used_src.add(sid)
            taken[cd.cells] = 1
            out.append(cd)
        used = np.zeros(NP, np.uint8)
        while True:
            cover = np.zeros(NP, np.int64)
            for cd in out:
                np.add.at(cover, P.cid[cd.cells], 1)
            part = set(np.nonzero((cover > 0) & (cover < P.sizes))[0].tolist())
            if not part:
                used = (cover > 0).astype(np.uint8)
                break
            out = [cd for cd in out if not any(i in part for i in cd.ids)]
        return out, used

    def place(self, cd, ctype, step, rec, z=None):
        src = cd.src
        same = ctype == "revive" and src.id not in self.index
        if same:
            oid = src.id
        else:
            oid = self.next_id
            self.next_id += 1
        o = SObj(oid, src.t, cd.r, cd.c, cd.k, 0)
        if z is None:
            self.insert_z(o)
            T = T_of(o)
            y, x, ins, g = self.cells_g(o, T)
            v = self.view
            F = self.F
            useful = bool(np.any(ins & (T.col == F[g]) & (v.col[g] != F[g]) & ((v.own[g] < 0) | (o.z < v.zt[g]))))
            if not useful:
                if not same:
                    self.next_id -= 1
                return None
        else:
            o.z = z
        self.add_obj(o)
        gone = -1
        if same:
            for i, e in enumerate(rec["edits"]):
                if e["type"] == "disappear" and e["id"] == src.id and e.get("from"):
                    gone = i
                    break
        if gone >= 0:
            frm = rec["edits"][gone]["from"]
            tr = self.tracks[src.id]
            turned = cd.k != frm[2]
            del rec["edits"][gone]
            tr.end = None
            tr.t = o.t
            tr.n["disappear"] = tr.n.get("disappear", 0) - 1
            if cd.r == frm[0] and cd.c == frm[1] and not turned:
                return o
            self.count(src.id, "rotate" if turned else "move")
            rec["edits"].append({"step": step, "type": "rotate" if turned else "move", "far": True, "id": o.id,
                                 "from": frm, "to": [cd.r, cd.c, cd.k], "cost": COST["move"] + (COST["rotate"] if turned else 0)})
            return o
        kind = None
        if not same:
            kind = {"id": src.id, "level": src.level if isinstance(src, LibEntry) else self.level}
        self.born(o, "library" if same else "kind", None, kind)
        e = {"step": step, "type": "appear", "how": "library" if same else "kind", "id": o.id,
             "cost": COST["place"] + (COST["rotate"] if cd.k else 0)}
        if kind is not None:
            e["kindOf"] = kind["id"]
            e["kindLevel"] = kind["level"]
        rec["edits"].append(e)
        return o

    # ------------------------------------------------------------ 11 reading the frame afresh
    def read_afresh(self, rec, step, new_level=None):
        A = self.read_area(0, self.h - 1, 0, self.w - 1)
        P = A.P
        NP = len(P.objects)
        taken = np.zeros(NP, np.uint8)
        stay: dict[int, None] = {}
        for o in sorted(self.objs, key=lambda o: (-T_of(o).n, o.id)):
            ids = self.whole_match(T_of(o), o.r, o.c, A) or self.same_full(o, A)
            if not ids or any(taken[i] for i in ids):
                continue
            for i in ids:
                taken[i] = 1
            stay[o.id] = None
        others = [o for o in self.objs if o.id not in stay]
        self.set_objs([o for o in self.objs if o.id in stay])
        lists = self.source_lists(False)
        if new_level is not None:
            for o in others:
                self.died(o, "level ended")
            for key in list(self.co_move.keys()):
                a, b = (int(x) for x in key.split(","))
                if a not in stay or b not in stay:
                    del self.co_move[key]
            for id in list(self.hidden_for.keys()):
                if id not in stay:
                    del self.hidden_for[id]
            self.level = new_level
            for o in self.objs:
                rec["edits"].append({"step": step, "type": "continue", "id": o.id, "cost": 0})
        else:
            lists.insert(0, ("move", self.shape_index([(o, o.t) for o in others], False), True))
        need = (1 - taken).astype(np.uint8)
        out, used = self.choose_matches(A, need, self.source_lists(False) if new_level is not None else lists, None)
        fresh = []
        for cd in out:
            if cd.type != "move":
                fresh.append(cd)
                continue
            o = cd.src
            turned = cd.k != o.k
            self.add_obj(o.copy(r=cd.r, c=cd.c, k=cd.k))
            if cd.r == o.r and cd.c == o.c and not turned:
                continue
            self.count(o.id, "rotate" if turned else "move")
            rec["edits"].append({"step": step, "type": "rotate" if turned else "move", "far": True, "id": o.id,
                                 "from": [o.r, o.c, o.k], "to": [cd.r, cd.c, cd.k],
                                 "cost": COST["move"] + (COST["rotate"] if turned else 0)})
        if new_level is None:
            for o in others:
                if o.id in self.index:
                    continue
                self.hidden_for.pop(o.id, None)
                self.died(o, "disappeared")
                self.count(o.id, "disappear")
                rec["edits"].append({"step": step, "type": "disappear", "id": o.id, "from": [o.r, o.c, o.k],
                                     "cost": COST["disappear"]})
        rest = [i for i in range(NP) if need[i] and not used[i]]
        self.slot_in(A, fresh, rest, step, rec)
        self.normalise_z()
        self.redraw()
        self.fix_layers(rec)
        self.learn(rec)
        self.repair(rec)

    def level_start(self, rec, new_level, first):
        self.read_afresh(rec, 0 if first else 11, new_level)

    def piece_cells(self, A, pid):
        po = A.P.objects[pid]
        return ((A.r0 + po.cells_r) * self.w + A.c0 + po.cells_c).tolist()

    def slot_in(self, A, chosen, pieces, step, rec):
        P = A.P
        direct = not self.objs
        items = [(cd, None, min(P.objects[i].layer for i in cd.ids)) for cd in chosen] + \
                [(None, i, P.objects[i].layer) for i in pieces]
        items.sort(key=lambda it: -it[2])
        self.redraw()
        F = self.F
        for cd, piece, layer in items:
            if cd is not None:
                cells = ((A.r0 + cd.cells // A.pw) * self.w + A.c0 + cd.cells % A.pw).tolist()
                if not direct and all(self.view.col[g] == F[g] for g in cells):
                    continue
                host = self.feature_of(cells) if not direct else None
                if host is not None and self.recolour_in(host, {g: int(F[g]) for g in cells}, step, "feature", rec,
                                                         COST["place"] + 1):
                    self.redraw()
                    continue
                self.place(cd, cd.type, step, rec, layer if direct else None)
            else:
                po = P.objects[piece]
                cells = self.piece_cells(A, piece)
                if not direct and all(self.view.col[g] == F[g] for g in cells):
                    continue
                host = self.feature_of(cells) if not direct else None
                if host is not None and self.recolour_in(host, {g: po.color for g in cells}, step, "feature", rec,
                                                         COST["place"] + 1):
                    self.redraw()
                    continue
                o = self.from_peel(po, A)
                if direct:
                    o.z = layer
                else:
                    self.insert_z(o)
                self.add_obj(o)
                self.born(o, "peel")
                rec["edits"].append({"step": step, "type": "appear", "how": "peel", "id": o.id, "cost": tpl_cost(o.t)})
            if not direct:
                self.redraw()

    # ------------------------------------------------------------ 4 common fate: the view scrolls
    def scroll(self, rec):
        S = self.opts["scroll"]
        v = self.view
        F = self.F
        h, w = self.h, self.w
        wrong = np.nonzero(v.col != F)[0]
        step = max(1, wrong.shape[0] // 1024)
        sample = wrong[::step].copy()
        odr, odc, on = K.scroll_counts(sample, v.col, F, h, w, S)
        if not odr.shape[0]:
            return False
        cands = sorted(zip(odr.tolist(), odc.tolist(), on.tolist()), key=lambda t: (-t[2], abs(t[0]) + abs(t[1])))
        keep = self.objs
        wrong0 = int(wrong.shape[0])
        best = None
        gidx = np.arange(self.HW)
        gy = gidx // w
        gx = gidx % w
        for dr, dc, n in cands[:3]:
            v = self.view          # (the view of the previous candidate's shift, as in the JS)
            own = v.own
            has = own >= 0
            maxid = max([o.id for o in keep] + [int(own.max()), 0])
            now_w = np.where(v.col == F, 1, -1)
            now = np.bincount(own[has], weights=now_w[has], minlength=maxid + 1)
            y = gy + dr
            x = gx + dc
            ok = has & (y >= 0) & (x >= 0) & (y < h) & (x < w)
            sh = np.where(ok, y * w + x, 0)
            then_w = np.where(v.col == F[sh], 1, -1)
            then = np.bincount(own[ok], weights=then_w[ok], minlength=maxid + 1)
            movers = {o.id: None for o in keep if then[o.id] > now[o.id]}
            if len(movers) < 3:
                continue
            objs = [o.copy(r=o.r + dr, c=o.c + dc) if o.id in movers else o for o in keep]
            self.set_objs(objs)
            self.redraw()
            gain = wrong0 - self.wrong_count() - COST["move"]
            if gain + COST["move"] >= self.opts["scrollFrac"] * wrong0 and (best is None or gain > best[3]):
                best = ((dr, dc), movers, objs, gain)
        self.set_objs(best[2] if best is not None else keep)
        self.redraw()
        if best is None:
            return False
        (dr, dc), movers, _objs, gain = best
        rec["edits"].append({"step": 4, "type": "scroll", "dr": dr, "dc": dc, "n": len(movers), "cost": COST["move"],
                             "fixed": gain + COST["move"]})
        for o in keep:
            if o.id in movers:
                self.count(o.id, "move")
                rec["edits"].append({"step": 4, "type": "move", "id": o.id, "scroll": True, "cost": 0,
                                     "from": [o.r, o.c, o.k], "to": [o.r + dr, o.c + dc, o.k]})
        return True

    # ------------------------------------------------------------ 3-5 suspects, proposals, greedy edits
    def suspects(self):
        h, w = self.h, self.w
        wrong = (self.view.col != self.F).reshape(h, w)
        near = wrong.copy()
        near[1:, :] |= wrong[:-1, :]
        near[:-1, :] |= wrong[1:, :]
        n2 = near.copy()
        n2[:, 1:] |= near[:, :-1]
        n2[:, :-1] |= near[:, 1:]
        nearf = n2.ravel()
        out = []
        for o in self.objs:
            y, x, ins, g = self.cells_g(o)
            if np.any(ins & nearf[g]):
                out.append(o)
        out.sort(key=lambda o: o.id)
        return out

    def search_poses(self, o):
        To = T_of(o)
        cy2 = 2 * o.r + To.th
        cx2 = 2 * o.c + To.tw
        hash_o = tpl_hash(To)
        ks = [o.k]
        if o.t.n <= self.opts["rotateMax"]:
            for k in distinct_orients(o.t):
                if k != o.k and tpl_hash(oriented(o.t, k)) != hash_o:
                    ks.append(k)
        rbs, cbs, sdy, sdx, scol, soff = [], [], [], [], [], [0]
        for k in ks:
            T = oriented(o.t, k)
            dy, dx, col = sample_of(T, self.opts)
            rbs.append(math.floor((cy2 - T.th) / 2 + 0.5))
            cbs.append(math.floor((cx2 - T.tw) / 2 + 0.5))
            sdy.append(dy)
            sdx.append(dx)
            scol.append(col)
            soff.append(soff[-1] + dy.shape[0])
        v = self.view
        bs, bd, br, bc, bk = K.search_poses(np.array(ks, np.int64), np.array(rbs, np.int64), np.array(cbs, np.int64),
                                            np.array(soff, np.int64), np.concatenate(sdy), np.concatenate(sdx),
                                            np.concatenate(scol), o.k, o.r, o.c, o.id, float(o.z), self.opts["range"],
                                            self.opts["keep"], v.own, v.zt, v.col, self.F, self.h, self.w)
        return [(int(bs[i]), int(bd[i]), int(br[i]), int(bc[i]), int(bk[i])) for i in range(bs.shape[0])]

    def recolour_map(self, o):
        T = T_of(o)
        y, x, ins, g = self.cells_g(o, T)
        sel = ins & (self.view.own[g] == o.id)
        cols = T.col[sel]
        if not cols.shape[0]:
            return None
        Fg = self.F[g[sel]]
        u, first, cnt = np.unique(cols, return_index=True, return_counts=True)
        frm = [(int(u[i]), int(cnt[i])) for i in np.argsort(first)]
        bad = cols != Fg
        keys = cols[bad] * 16 + Fg[bad]
        tally = []
        if keys.shape[0]:
            ku, kf, kc = np.unique(keys, return_index=True, return_counts=True)
            tally = [(int(ku[i]), int(kc[i])) for i in np.argsort(kf)]
        mp = {}
        n = 0
        for c, total in frm:
            to = -1
            best = 0
            for key, vv in tally:
                if (key >> 4) == c and vv > best:
                    best = vv
                    to = key & 15
            if to >= 0 and best * 2 >= total:
                mp[c] = to
                n += 1
        return (mp, n) if n else None

    def recolour_tpl(self, t, mp):
        col = t.col.copy()
        for c, to in mp.items():
            col[t.col == c] = to
        return make_tpl(t.th, t.tw, t.ys, t.xs, col, t.seen)

    def proposals(self, o):
        out = []
        To = T_of(o)
        for score, dist, r, c, k in (self.search_poses(o) if o.t.n <= self.opts["moveMax"] else []):
            Tn = oriented(o.t, k)
            centre_moved = 2 * r + Tn.th != 2 * o.r + To.th or 2 * c + Tn.tw != 2 * o.c + To.tw
            if k == o.k:
                out.append(Prop("move", o.id, COST["move"], r, c, k))
            else:
                out.append(Prop("rotate", o.id, COST["rotate"] + (COST["move"] if centre_moved else 0), r, c, k))
        out.append(Prop("disappear", o.id, COST["disappear"]))
        rc = self.recolour_map(o)
        if rc is not None:
            out.append(Prop("recolor", o.id, COST["recolor"] * rc[1], map=rc[0]))
        return out

    # ------------------------------------------------------------ bounds, trial swaps, greedy
    def _pk(self):
        p = self.pack()
        return p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9]

    def bound(self, p: Prop):
        o = self.index[p.id]
        if not p.obj_set:
            if p.type == "disappear":
                p.obj = None
            elif p.type == "recolor":
                p.obj = o.copy(t=self.recolour_tpl(o.t, p.map))
            else:
                p.obj = o.copy(r=p.r, c=p.c, k=p.k)
            p.obj_set = True
        p.box = union_box(box_of(o), box_of(p.obj)) if p.obj is not None else box_of(o)
        T = T_of(o)
        v = self.view
        is_rec = p.type == "recolor"
        rmap = np.full(16, -1, np.int64)
        if is_rec:
            for c, to in p.map.items():
                rmap[c] = to
        if p.obj is not None and not is_rec:
            Tn = T_of(p.obj)
            nys, nxs, ncol, nr, nc = Tn.ys, Tn.xs, Tn.col, p.obj.r, p.obj.c
            has_n = True
        else:
            nys = nxs = ncol = np.zeros(0, np.int64)
            nr = nc = 0
            has_n = False
        n = K.bound_count(T.ys, T.xs, T.col, o.r, o.c, o.id, float(o.z), is_rec, rmap, has_n, nys, nxs, ncol, nr, nc,
                          v.col, v.own, v.uid, v.ucol, self.F, self.h, self.w, self.pack()[13], *self._pk())
        p.ub = n - p.cost
        p.stale = True

    def trial_swap(self, o, n):
        To = T_of(o)
        if n is not None:
            Tn = T_of(n)
            args = (True, Tn.ys, Tn.xs, Tn.col, n.r, n.c, n.id, float(n.z))
        else:
            e = np.zeros(0, np.int64)
            args = (False, e, e, e, 0, 0, 0, 0.0)
        v = self.view
        fixed, broken, cg, ccol, cown, cz, cells, err = K.trial_swap(
            To.ys, To.xs, To.col, o.r, o.c, o.id, float(o.z), *args, v.col, v.own, v.zt, v.uid, v.ucol, self.F,
            self.h, self.w, self.pack()[13], *self._pk())
        if err:
            raise RuntimeError("nextBelow: object below no longer in the scene")
        return {"fixed": int(fixed), "broken": int(broken), "cg": cg, "ccol": ccol, "cown": cown, "cz": cz,
                "cells": cells}

    def commit_swap(self, tr):
        v = self.view
        v.col[tr["cg"]] = tr["ccol"]
        v.own[tr["cg"]] = tr["cown"]
        v.zt[tr["cg"]] = tr["cz"]
        v.uid[tr["cells"]] = -2
        self.unknown_below += int(tr["cells"].shape[0])
        if self.unknown_below > 512:
            self.redraw()
            self.unknown_below = 0

    def evaluate(self, p: Prop):
        o = self.index[p.id]
        p.trial = self.trial_swap(o, p.obj)
        p.gain = p.trial["fixed"] - p.trial["broken"] - p.cost + (FOLLOW_BIAS if p.follow else 0)
        p.stale = False
        if p.obj is not None and p.type != "recolor" and not p.front:
            Tn = T_of(p.obj)
            y, x, ins, g = self.cells_g(p.obj, Tn)
            v = self.view
            F = self.F
            covered = bool(np.any(ins & (v.col[g] != F[g]) & (Tn.col == F[g]) & (v.own[g] != o.id) & (v.zt[g] < o.z)))
            if covered:
                front = p.obj.copy(z=min([0] + [q.z for q in self.objs]) - 0.5)
                tr = self.trial_swap(o, front)
                gain = tr["fixed"] - tr["broken"] - p.cost - COST["layer"] + (FOLLOW_BIAS if p.follow else 0)
                if gain > p.gain:
                    p.obj = front
                    p.trial = tr
                    p.gain = gain
                    p.cost += COST["layer"]
                    p.front = True

    def followers(self, o, dr, dc, all_, done, offered):
        T = T_of(o)
        box = (o.r - 1, o.r + T.th, o.c - 1, o.c + T.tw)
        h, w = self.h, self.w
        near = np.zeros(self.HW, bool)
        ys = o.r + T.ys
        xs = o.c + T.xs
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yy = ys + dy
                xx = xs + dx
                ok = (yy >= 0) & (xx >= 0) & (yy < h) & (xx < w)
                near[yy[ok] * w + xx[ok]] = True
        for b in self.objs:
            if b.id == o.id or b.id in done or not _box_overlap(box_of(b), box):
                continue
            key = (b.id, dr, dc)
            if key in offered:
                continue
            y, x, ins, g = self.cells_g(b)
            if not np.any(ins & near[g]):
                continue
            offered.add(key)
            p = Prop("move", b.id, COST["move"], b.r + dr, b.c + dc, b.k, follow=True, seq=len(all_))
            self.bound(p)
            p.ub += FOLLOW_BIAS
            all_.append(p)

    def prefix_of(self, cells):
        if not len(cells):
            return None
        h, w = self.h, self.w
        cnt = np.bincount(np.asarray(cells, np.int64), minlength=self.HW).reshape(h, w)
        s = np.zeros((h + 1, w + 1), np.int64)
        s[1:, 1:] = cnt.cumsum(0).cumsum(1)
        return s

    def box_count(self, s, b):
        if s is None:
            return 0
        r0, r1, c0, c1 = max(0, b[0]), min(self.h - 1, b[1]), max(0, b[2]), min(self.w - 1, b[3])
        if r0 > r1 or c0 > c1:
            return 0
        return int(s[r1 + 1, c1 + 1] - s[r0, c1 + 1] - s[r1 + 1, c0] + s[r0, c0])

    @staticmethod
    def acceptable(p):
        return p.gain > 0 or (p.gain == 0 and p.trial["fixed"] > 0 and p.trial["broken"] == 0)

    @staticmethod
    def better(p, q):
        if q is None or p.gain > q.gain:
            return True
        if p.gain != q.gain:
            return False
        if bool(p.follow) != bool(q.follow):
            return bool(p.follow)
        if p.trial["fixed"] != q.trial["fixed"]:
            return p.trial["fixed"] > q.trial["fixed"]
        if p.id != q.id:
            return p.id < q.id
        return p.seq < q.seq

    def greedy(self, props, rec):
        all_ = [p for lst in props.values() for p in lst]
        done = set()
        offered = set()
        for i, p in enumerate(all_):
            p.seq = i
            self.bound(p)
        better = self.better
        acceptable = self.acceptable
        while True:
            best = None
            for p in all_:
                if p.id not in done and not p.stale and acceptable(p) and better(p, best):
                    best = p
            opn = [p for p in all_ if p.id not in done and p.stale and p.ub >= 0 and (best is None or p.ub >= best.gain)]
            opn.sort(key=lambda p: (-p.ub, p.seq))
            for p in opn:
                if best is not None and p.ub < best.gain:
                    break
                if p.ubStale:
                    self.bound(p)
                    if p.follow:
                        p.ub += FOLLOW_BIAS
                    p.ubStale = False
                    if p.ub < 0 or (best is not None and p.ub < best.gain):
                        continue
                self.evaluate(p)
                if acceptable(p) and better(p, best):
                    best = p
            if best is None:
                return
            batch = [best]
            ids = {best.id}
            if len(all_) > BATCH_FROM:
                cand = [q for q in all_ if q is not best and not q.stale and q.id not in done and acceptable(q)]
                cand.sort(key=cmp_to_key(lambda a, b: -1 if better(a, b) else 1))
                for p in cand:
                    if p.id in ids:
                        continue
                    pb = (p.box[0] - 1, p.box[1] + 1, p.box[2] - 1, p.box[3] + 1)
                    if any(_box_overlap(q.box, pb) for q in batch):
                        continue
                    batch.append(p)
                    ids.add(p.id)
            shown, below, moved = [], [], []
            for p in batch:
                o = self.index[p.id]
                if p.obj is not None:
                    self.swap_obj(p.obj)
                else:
                    self.drop_obj(p.id)
                    self.died(o, "disappeared")
                self.commit_swap(p.trial)
                done.add(p.id)
                self.count(p.id, p.type)
                e = {"step": 5, "type": p.type, "id": p.id, "cost": p.cost, "fixed": p.trial["fixed"],
                     "broken": p.trial["broken"], "from": [o.r, o.c, o.k],
                     "to": [p.obj.r, p.obj.c, p.obj.k] if p.obj is not None else None}
                if p.map is not None:
                    e["map"] = dict(p.map)
                if p.follow:
                    e["follow"] = True
                if p.front:
                    e["front"] = True
                rec["edits"].append(e)
                shown.extend(p.trial["cg"].tolist())
                below.extend(p.trial["cells"].tolist())
                if p.type == "move":
                    moved.append((o, p.obj.r - o.r, p.obj.c - o.c))
            sh = self.prefix_of(shown)
            bl = self.prefix_of(below)
            all_ = [p for p in all_ if p.id not in done]
            for p in all_:
                grown = 2 * self.box_count(sh, p.box) + self.box_count(bl, p.box)
                if grown:
                    p.stale = True
                    p.ub += grown
                    p.ubStale = True
            for o, dr, dc in moved:
                self.followers(o, dr, dc, all_, done, offered)

    # ------------------------------------------------------------ 6 splits
    def split_trial(self, o, b, dr, dc, rec, both_stay):
        T = oriented(b.t, b.k)
        h, w = self.h, self.w
        ok, lab = K.split_labels(T.ys, T.xs, T.col, T.seen, T.at, T.th, T.tw, b.r, b.c, dr, dc, self.F, h, w, both_stay)
        if not ok:
            return None
        m = lab == 2
        S, sy0, sx0 = tpl_from_cells(T.ys[~m], T.xs[~m], T.col[~m], T.seen[~m])
        M, my0, mx0 = tpl_from_cells(T.ys[m], T.xs[m], T.col[m], T.seen[m])
        oS = SObj(o.id, S, b.r + sy0, b.c + sx0, 0, o.z)
        oM = SObj(0, M, b.r + dr + my0, b.c + dc + mx0, 0, o.z - 0.25)
        box = union_box(union_box(box_of(o), box_of(oS)), box_of(oM))
        tr = self.trial({o.id: oS}, [oM], box)
        moved = None
        for e in rec["edits"]:
            if e["step"] == 5 and e["type"] == "move" and e["id"] == o.id:
                moved = e
                break
        cost = COST["split"] + max(0, tpl_cost(S) + tpl_cost(M) - tpl_cost(b.t)) + COST["move"]
        # what one object reshaped into the same picture would cost: cells keyed by (y * w + x), the later one wins
        Y = np.concatenate((b.r + T.ys[~m], b.r + dr + T.ys[m]))
        X = np.concatenate((b.c + T.xs[~m], b.c + dc + T.xs[m]))
        C = np.concatenate((T.col[~m], T.col[m]))
        Sn = np.concatenate((T.seen[~m], T.seen[m]))
        keys = Y * w + X
        _u, last_rev = np.unique(keys[::-1], return_index=True)
        last = keys.shape[0] - 1 - last_rev
        R, _a, _b = tpl_from_cells(Y[last], X[last], C[last], Sn[last])
        reshape = 2 + max(0, tpl_cost(R) - tpl_cost(b.t))
        return {"o": o, "oS": oS, "oM": oM, "box": box, "cost": cost, "moved": moved, "dr": dr, "dc": dc,
                "reshape": reshape, "gain": tr["fixed"] - tr["broken"] - cost + (moved["cost"] if moved else 0)}

    def splits(self, sus, props, before, rec):
        moves = []
        for e in rec["edits"]:
            if e["step"] == 5 and e["type"] == "move" and e["id"] in before:
                bb = before[e["id"]]
                U = oriented(bb[0], bb[3])
                moves.append((e["id"], e["to"][0] - e["from"][0], e["to"][1] - e["from"][1],
                              (bb[1] - 1, bb[1] + U.th, bb[2] - 1, bb[2] + U.tw)))
        for s in sus:
            o = self.index.get(s.id)
            b = before.get(s.id)
            if o is None or b is None or o.t is not b[0] or o.k != b[3]:
                continue
            bt, br_, bc_, bk, bz = b
            offs: dict[tuple, None] = {}
            fate = set()
            for p in props.get(s.id, ()):
                if p.type == "move":
                    offs[(p.r - br_, p.c - bc_)] = None
            if o.r != br_ or o.c != bc_:
                offs[(o.r - br_, o.c - bc_)] = None
            U = oriented(bt, bk)
            bbox = (br_, br_ + U.th - 1, bc_, bc_ + U.tw - 1)
            for mid, mdr, mdc, mbox in moves:
                if mid != s.id and _box_overlap(mbox, bbox):
                    offs[(mdr, mdc)] = None
                    fate.add((mdr, mdc))
            best = None
            bo = SObj(o.id, bt, br_, bc_, bk, bz)
            for key in offs:
                for both_stay in (False, True):
                    cd = self.split_trial(o, bo, key[0], key[1], rec, both_stay)
                    if cd is not None and cd["gain"] > 0 and (key in fate or cd["cost"] <= cd["reshape"] + COST["split"]) \
                            and (best is None or cd["gain"] > best["gain"]):
                        best = cd
            if best is None:
                continue
            best["oM"].id = self.next_id
            self.next_id += 1
            self.swap_obj(best["oS"])
            self.add_obj(best["oM"])
            tr = self.trial({}, [], best["box"])
            self.commit(tr)
            if best["moved"] is not None:
                for i, e in enumerate(rec["edits"]):
                    if e is best["moved"]:
                        del rec["edits"][i]
                        break
            self.born(best["oM"], "split", [o.id])
            self.tracks[o.id].children.append(best["oM"].id)
            self.blocked.add(pair_key(o.id, best["oM"].id))
            self.count(o.id, "split")
            rec["edits"].append({"step": 6, "type": "split", "id": o.id, "other": best["oM"].id, "cost": best["cost"],
                                 "fixed": tr["fixed"], "broken": tr["broken"], "to": [best["dr"], best["dc"]]})

    # ------------------------------------------------------------ 7 layer order from overlaps
    def fix_layers(self, rec):
        p = self.pack()
        fa, fb = K.layer_votes(p[0], p[1], p[2], p[3], p[10], p[11], p[12], p[9], self.F, self.h, self.w)
        if not fa.shape[0]:
            return
        a = np.minimum(fa, fb)
        b = np.maximum(fa, fb)
        which = (fa >= fb).astype(np.int64)        # 0: the front object is the smaller id
        code = a * (1 << 32) + b
        votes: dict[int, list] = {}
        for cd, wv in zip(code.tolist(), which.tolist()):
            v = votes.get(cd)
            if v is None:
                v = votes[cd] = [0, 0]
            v[wv] += 1
        cons = []
        for cd, v in votes.items():
            ia, ib = cd >> 32, cd & 0xFFFFFFFF
            net = v[0] - v[1]
            if net > 0 and self.index[ia].z > self.index[ib].z:
                cons.append((ia, ib, net))
            elif net < 0 and self.index[ib].z > self.index[ia].z:
                cons.append((ib, ia, -net))
        if not cons:
            return
        cons.sort(key=lambda t: (-t[2], t[0], t[1]))
        wrong = self.wrong_count()
        for ia, ib, _n in cons:
            A = self.index[ia]
            B = self.index[ib]
            if A.z < B.z:
                continue
            za, zb = A.z, B.z
            A.z = zb - 0.5
            self.redraw()
            forward = self.wrong_count()
            A.z = za
            B.z = za + 0.5
            self.redraw()
            back = self.wrong_count()
            B.z = zb
            if min(forward, back) < wrong:
                if forward <= back:
                    A.z = zb - 0.5
                else:
                    B.z = za + 0.5
                wrong = min(forward, back)
                self.normalise_z()
                self.redraw()
                self.count(ia, "layer")
                rec["edits"].append({"step": 7, "type": "layer", "id": ia, "other": ib, "cost": COST["layer"], "fixed": 0})
            else:
                self.redraw()

    # ------------------------------------------------------------ 8 guessed cells that became visible
    def below_of(self, gs, tops_id, tops_z):
        """nextBelow for cells gs: (id or -1 / -2, colour)."""
        v = self.view
        p = self.pack()
        return K.below_cells(np.asarray(gs, np.int64), np.asarray(tops_id, np.int64), np.asarray(tops_z, np.float64),
                             v.own, v.uid, v.ucol, p[13], self.w, *self._pk())

    def learn(self, rec):
        changed = False
        F = self.F
        for o in list(self.objs):
            T = T_of(o)
            if T.nSeen == T.n:
                continue
            y, x, ins, g = self.cells_g(o, T)
            cand = (T.seen == 0) & ins & (self.view.own[g] == o.id)
            if not np.any(cand):
                continue
            agree = cand & (T.col == F[g])
            dis = np.nonzero(cand & ~agree)[0]
            drop = np.zeros(T.n, bool)
            if dis.shape[0]:
                bid, bcol = self.below_of(g[dis], np.full(dis.shape[0], o.id), np.full(dis.shape[0], float(o.z)))
                bad_sel = (bid != -1) & (bcol == F[g[dis]])
                drop[dis[bad_sel]] = True
            conf = int(np.count_nonzero(agree))
            bad = int(np.count_nonzero(drop))
            if not conf and not bad:
                continue
            changed = True
            keep = ~drop
            if not np.any(keep):
                self.drop_obj(o.id)
                self.died(o, "disappeared")
                continue
            seen = np.where(agree, 1, T.seen)
            N, y0, x0 = tpl_from_cells(T.ys[keep], T.xs[keep], T.col[keep], seen[keep])
            if not bad:
                base = o.t.base if o.t.base is not None else o.t
                N.cost = tpl_cost(base)
                N.parts = base.parts
            self.swap_obj(o.copy(t=N, r=o.r + y0, c=o.c + x0, k=0))
            if conf:
                rec["edits"].append({"step": 8, "type": "confirm", "id": o.id, "cells": conf, "cost": 0})
            if bad:
                self.count(o.id, "reshape")
                rec["edits"].append({"step": 8, "type": "reshape", "why": "guess was wrong", "id": o.id, "cells": bad,
                                     "cost": max(1, tpl_cost(N) - tpl_cost(o.t))})
        if changed:
            self.redraw()

    # ------------------------------------------------------------ 9 cells still wrong
    def shrink(self, rec):
        v = self.view
        F = self.F
        gs = np.nonzero((v.col != F) & (v.own >= 0))[0]
        if not gs.shape[0]:
            return
        tops = v.own[gs]
        tz = np.array([self.index[int(t)].z for t in tops.tolist()], np.float64)
        bid, bcol = self.below_of(gs, tops, tz)
        hit = (bid != -1) & (bcol == F[gs])
        cut: dict[int, dict] = {}
        for g, t in zip(gs[hit].tolist(), tops[hit].tolist()):
            cut.setdefault(t, {})[g] = None
        for id, cells in cut.items():
            o = self.index[id]
            T = T_of(o)
            y, x, ins, g = self.cells_g(o, T)
            cs = np.fromiter(cells.keys(), np.int64, len(cells))
            rm = ins & np.isin(g, cs)
            keep = ~rm
            if not np.any(keep):
                self.drop_obj(id)
                self.died(o, "disappeared")
                self.count(id, "disappear")
                rec["edits"].append({"step": 9, "type": "disappear", "id": id, "from": [o.r, o.c, o.k], "cost": COST["disappear"]})
                continue
            N, y0, x0 = tpl_from_cells(y[keep], x[keep], T.col[keep], T.seen[keep])
            self.swap_obj(o.copy(t=N, r=y0, c=x0, k=0))
            self.count(id, "reshape")
            rec["edits"].append({"step": 9, "type": "reshape", "why": "shrink", "id": id, "cells": len(cells),
                                 "cost": max(1, tpl_cost(N) - tpl_cost(o.t))})
        if cut:
            self.redraw()

    def grow(self, rec):
        first: dict[int, Tpl] = {}
        added: dict[int, int] = {}
        h, w = self.h, self.w
        for _pass in range(64):
            v = self.view
            F = self.F
            col = v.col
            own = v.own
            zt = v.zt
            gain: dict[int, list] = {}
            for g in np.nonzero(col != F)[0].tolist():
                y, x = divmod(g, w)
                fg = F[g]
                tally: dict[int, int] = {}
                for dy, dx in N4:
                    yy, xx = y + dy, x + dx
                    if yy < 0 or xx < 0 or yy >= h or xx >= w:
                        continue
                    q = yy * w + xx
                    oid = int(own[q])
                    if oid >= 0 and col[q] == fg:
                        tally[oid] = tally.get(oid, 0) + 1
                best = None
                bn = 0
                for oid, n in tally.items():
                    o = self.index[oid]
                    if o.z >= zt[g] or self.covers(o, g):
                        continue
                    if n > bn or (n == bn and (o.t.n > best.t.n or (o.t.n == best.t.n and (o.z, o.id) < (best.z, best.id)))):
                        bn = n
                        best = o
                if best is not None:
                    gain.setdefault(best.id, []).append((y, x, int(fg), 1))
            if not gain:
                break
            for oid, cells in gain.items():
                o = self.index[oid]
                T = T_of(o)
                if oid not in first:
                    first[oid] = o.t
                added[oid] = added.get(oid, 0) + len(cells)
                Y = np.concatenate((o.r + T.ys, np.array([c[0] for c in cells], np.int64)))
                X = np.concatenate((o.c + T.xs, np.array([c[1] for c in cells], np.int64)))
                C = np.concatenate((T.col, np.array([c[2] for c in cells], np.int64)))
                S = np.concatenate((T.seen, np.ones(len(cells), np.int64)))
                N, y0, x0 = tpl_from_cells(Y, X, C, S)
                self.swap_obj(o.copy(t=N, r=y0, c=x0, k=0))
            self.redraw()
        for oid, n in added.items():
            self.count(oid, "reshape")
            rec["edits"].append({"step": 9, "type": "reshape", "why": "grow", "id": oid, "cells": n,
                                 "cost": max(1, tpl_cost(self.index[oid].t) - tpl_cost(first[oid]))})

    def insert_z(self, o):
        v = self.view
        F = self.F
        for attempt in (0, 1):
            T = T_of(o)
            y, x, ins, g = self.cells_g(o, T)
            sel = ins & (v.own[g] >= 0)
            right = T.col == F[g]
            shows = v.col[g] == F[g]
            a = sel & right & ~shows
            b = sel & ~right & shows
            zMax = float(v.zt[g[a]].min()) if np.any(a) else INF
            zMin = float(v.zt[g[b]].max()) if np.any(b) else -INF
            if zMin < zMax or attempt:
                if zMax == INF:
                    o.z = 0 if zMin == -INF else zMin + 0.5
                elif zMin == -INF or zMin >= zMax:
                    o.z = zMax - 0.5
                else:
                    o.z = (zMin + zMax) / 2
                o._box = None
                return
            drop = sel & ~right & shows & (v.zt[g] >= zMax)
            keep = ~drop
            N, y0, x0 = tpl_from_cells(T.ys[keep], T.xs[keep], T.col[keep], T.seen[keep])
            o.t = N
            o.r += y0
            o.c += x0
            o.k = 0
            o._box = None

    def rematch_local(self, rec):
        h, w, HW = self.h, self.w, self.HW
        v = self.view
        F = self.F
        wrongm = (v.col != F)
        mark = np.zeros(HW, bool)
        groups = []
        wl = wrongm.tolist()
        for g0 in np.nonzero(wrongm)[0].tolist():
            if mark[g0]:
                continue
            grp = [g0]
            mark[g0] = True
            k = 0
            while k < len(grp):
                y, x = divmod(grp[k], w)
                k += 1
                for dy in (-2, -1, 0, 1, 2):
                    yy = y + dy
                    if yy < 0 or yy >= h:
                        continue
                    for dx in (-2, -1, 0, 1, 2):
                        xx = x + dx
                        if xx < 0 or xx >= w:
                            continue
                        q = yy * w + xx
                        if not mark[q] and wl[q]:
                            mark[q] = True
                            grp.append(q)
            groups.append(grp)
        moved_far: set[int] = set()
        for grp in groups:
            v = self.view
            garr = np.array(grp, np.int64)
            still = garr[v.col[garr] != F[garr]]
            if not still.shape[0]:
                continue
            ys = still // w
            xs = still % w
            r0, r1, c0, c1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
            A = self.read_area(max(0, r0 - 2), min(h - 1, r1 + 2), max(0, c0 - 2), min(w - 1, c1 + 2))
            P = A.P
            NP = len(P.objects)
            need = np.zeros(NP, np.uint8)
            need[P.cid[(ys - A.r0) * A.pw + xs - A.c0]] = 1
            own = v.own
            has = own >= 0
            mx = max(int(own.max()), 0) + 1
            vis = np.bincount(own[has], minlength=mx)
            badc = np.bincount(own[has & (v.col != F)], minlength=mx)
            in_place = set(np.nonzero((vis > 0) & (badc * 2 < vis))[0].tolist())
            in_place |= moved_far
            for pid in range(NP):
                if not need[pid]:
                    continue
                po = P.objects[pid]
                cells = self.piece_cells(A, pid)
                host = self.feature_of(cells)
                if host is not None and self.recolour_in(host, {g: po.color for g in cells}, 9, "feature", rec, COST["place"] + 1):
                    need[pid] = 0
                    self.redraw()
            if not need.any():
                continue
            out, used = self.choose_matches(A, need, self.source_lists(True), in_place)
            fresh = []
            for cd in out:
                if cd.type != "move":
                    fresh.append(cd)
                    continue
                o = self.index[cd.src.id]
                turned = cd.k != o.k
                n = o.copy(r=cd.r, c=cd.c, k=cd.k)
                self.swap_obj(n)
                self.count(o.id, "rotate" if turned else "move")
                self.redraw()
                moved_far.add(o.id)
                rec["edits"].append({"step": 9, "type": "rotate" if turned else "move", "far": True, "id": o.id,
                                     "from": [o.r, o.c, o.k], "to": [cd.r, cd.c, cd.k],
                                     "cost": COST["move"] + (COST["rotate"] if turned else 0)})
            for cd in fresh:
                self.place(cd, cd.type, 9, rec)
            self.redraw()
            for pid in range(NP):
                if not need[pid] or used[pid]:
                    continue
                po = P.objects[pid]
                cells = self.piece_cells(A, pid)
                host = self.feature_of(cells)
                if host is not None and self.recolour_in(host, {g: po.color for g in cells}, 9, "feature", rec, COST["place"] + 1):
                    self.redraw()
                    continue
                o = self.from_peel(po, A)
                self.insert_z(o)
                self.add_obj(o)
                self.born(o, "peel")
                rec["edits"].append({"step": 9, "type": "appear", "how": "peel", "id": o.id, "cost": tpl_cost(o.t)})
            self.redraw()

    def feature_of(self, cells):
        if len(cells) > self.opts["featureMax"]:
            return None
        inside = set(cells)
        h, w = self.h, self.w
        own = self.view.own
        fid = -1
        for g in cells:
            y, x = divmod(g, w)
            for dy, dx in N4:
                yy, xx = y + dy, x + dx
                if yy < 0 or xx < 0 or yy >= h or xx >= w:
                    return None
                q = yy * w + xx
                if q in inside:
                    continue
                ow = int(own[q])
                if ow < 0 or (fid >= 0 and ow != fid):
                    return None
                fid = ow
        if fid < 0:
            return None
        o = self.index[fid]
        return None if o.z >= max([-INF] + [q.z for q in self.objs]) else o

    def recolour_in(self, o, cells: dict, step, why, rec, limit) -> bool:
        T = T_of(o)
        w = self.w
        y, x, ins, g = self.cells_g(o, T)
        cols = T.col.copy()
        seen = T.seen.copy()
        left = dict(cells)
        gl = g.tolist()
        insl = ins.tolist()
        for i in range(T.n):
            if insl[i]:
                c = cells.get(gl[i])
                if c is not None:
                    cols[i] = c
                    seen[i] = 1
                    left.pop(gl[i], None)
        Y = [y]
        X = [x]
        C = [cols]
        S = [seen]
        if left:
            Y.append(np.array([gg // w for gg in left], np.int64))
            X.append(np.array([gg % w for gg in left], np.int64))
            C.append(np.array(list(left.values()), np.int64))
            S.append(np.ones(len(left), np.int64))
        N, y0, x0 = tpl_from_cells(np.concatenate(Y), np.concatenate(X), np.concatenate(C), np.concatenate(S))
        cost = max(1, tpl_cost(N) - tpl_cost(o.t))
        if cost > limit:
            return False
        self.swap_obj(o.copy(t=N, r=y0, c=x0, k=0))
        self.count(o.id, "reshape")
        rec["edits"].append({"step": step, "type": "reshape", "why": why, "id": o.id, "cells": len(cells), "cost": cost})
        return True

    def groups_of(self, cells):
        st = set(cells)
        seen = set()
        n = 0
        h, w = self.h, self.w
        for g in cells:
            if g in seen:
                continue
            n += 1
            q = [g]
            seen.add(g)
            k = 0
            while k < len(q):
                y, x = divmod(q[k], w)
                k += 1
                for dy, dx in N4:
                    yy, xx = y + dy, x + dx
                    p = yy * w + xx
                    if 0 <= yy < h and 0 <= xx < w and p in st and p not in seen:
                        seen.add(p)
                        q.append(p)
        return n

    def recolour_cells(self, rec):
        v = self.view
        F = self.F
        cut: dict[int, dict] = {}
        for g in np.nonzero((v.col != F) & (v.own >= 0))[0].tolist():
            o = self.index[int(v.own[g])]
            T = T_of(o)
            if T.colset is None:
                T.colset = set(T.col.tolist())
            fg = int(F[g])
            if fg not in T.colset:
                continue
            cut.setdefault(o.id, {})[g] = fg
        done = 0
        for oid, cells in cut.items():
            if self.recolour_in(self.index[oid], cells, 9, "recolour", rec, COST["place"] * self.groups_of(list(cells.keys()))):
                done += 1
        if done:
            self.redraw()

    def repair(self, rec):
        rnd = 0
        while rnd < 3 and self.wrong_count():
            self.shrink(rec)
            if self.wrong_count():
                self.grow(rec)
            if self.wrong_count():
                self.recolour_cells(rec)
            if self.wrong_count():
                self.rematch_local(rec)
            rnd += 1
        if not self.wrong_count():
            return
        front = min([0] + [o.z for o in self.objs]) - 1
        v = self.view
        F = self.F
        for g in np.nonzero(v.col != F)[0].tolist():
            t = make_tpl(1, 1, [0], [0], [int(F[g])], [1])
            o = SObj(self.next_id, t, g // self.w, g % self.w, 0, front)
            self.next_id += 1
            self.add_obj(o)
            self.born(o, "cell")
            rec["edits"].append({"step": 9, "type": "appear", "how": "cell", "id": o.id, "cost": tpl_cost(t)})
        self.redraw()

    # ------------------------------------------------------------ 10 merging objects that move together
    def comotion(self, before, rec):
        tf: dict[int, str] = {}
        for e in rec["edits"]:
            if e["step"] != 5 or e["id"] not in self.index:
                continue
            if e["type"] == "move":
                tf[e["id"]] = f"{e['to'][0] - e['from'][0]},{e['to'][1] - e['from'][1]}"
            elif e["type"] == "rotate":
                tf[e["id"]] = "turn" + str(e["id"])
        if not tf:
            return
        pairs: set[str] = set()
        for oid in tf:
            bt, br_, bc_, bk, _bz = before[oid]
            T = oriented(bt, bk)
            ys = br_ + T.ys
            xs = bc_ + T.xs
            near = np.unique(np.concatenate([(ys + dy) * 4096 + xs + dx for dy in (-1, 0, 1) for dx in (-1, 0, 1)]))
            box = (br_ - 1, br_ + T.th, bc_ - 1, bc_ + T.tw)
            for xid, ob in before.items():
                if xid == oid or xid not in self.index:
                    continue
                U = oriented(ob[0], ob[3])
                if not _box_overlap(box, (ob[1], ob[1] + U.th - 1, ob[2], ob[2] + U.tw - 1)):
                    continue
                keys = (ob[1] + U.ys) * 4096 + ob[2] + U.xs
                if np.any(np.isin(keys, near)):
                    pairs.add(pair_key(oid, xid))
        ready = []
        for key in sorted(pairs):
            if key in self.blocked:
                continue
            a, b = (int(x) for x in key.split(","))
            if a in tf and tf[a] == tf.get(b) and not tf[a].startswith("turn"):
                n = self.co_move.get(key, 0) + 1
                self.co_move[key] = n
                if n >= self.opts["mergeAfter"]:
                    ready.append((a, b))
            else:
                self.blocked.add(key)
                self.co_move.pop(key, None)
        if not ready:
            return
        up: dict[int, int] = {}

        def find(x):
            while x in up and up[x] != x:
                x = up[x]
            return x
        for a, b in ready:
            up[a] = find(a)
            up[b] = find(b)
            up[find(a)] = find(b)
        groups: dict[int, list] = {}
        for x in up:
            groups.setdefault(find(x), []).append(x)
        for ids in groups.values():
            self.merge_group(sorted(set(ids)), rec)

    def merge_group(self, ids, rec):
        parts = sorted([self.index[i] for i in ids if i in self.index], key=lambda o: (o.z, o.id))
        if len(parts) < 2:
            return
        cells: dict[int, tuple] = {}
        for o in parts:
            T = T_of(o)
            ys = (o.r + T.ys).tolist()
            xs = (o.c + T.xs).tolist()
            cl = T.col.tolist()
            sl = T.seen.tolist()
            for i in range(T.n):
                key = (ys[i] + 1024) * 4096 + xs[i] + 1024
                if key not in cells:
                    cells[key] = (ys[i], xs[i], cl[i], sl[i])
        cv = list(cells.values())
        N, y0, x0 = tpl_from_cells([c[0] for c in cv], [c[1] for c in cv], [c[2] for c in cv], [c[3] for c in cv])
        m = SObj(self.next_id, N, y0, x0, 0, parts[0].z)
        tr = self.trial({o.id: None for o in parts}, [m], reduce(union_box, [box_of(o) for o in parts]))
        if tr["fixed"] or tr["broken"]:
            return
        self.next_id += 1
        for o in parts:
            self.drop_obj(o.id)
            self.died(o, "merged")
            self.tracks[o.id].children.append(m.id)
        self.add_obj(m)
        self.commit(tr)
        self.born(m, "merge", [o.id for o in parts])
        rec["edits"].append({"step": 10, "type": "merge", "id": m.id, "parts": [o.id for o in parts], "cost": COST["merge"]})
        in_group = {o.id for o in parts}
        for key, v in list(self.co_move.items()):
            a, b = (int(x) for x in key.split(","))
            if a not in in_group and b not in in_group:
                continue
            del self.co_move[key]
            if a in in_group and b in in_group:
                continue
            k2 = pair_key(m.id, b if a in in_group else a)
            self.co_move[k2] = max(self.co_move.get(k2, 0), v)
        for key in list(self.blocked):
            a, b = (int(x) for x in key.split(","))
            if (a in in_group) != (b in in_group):
                self.blocked.add(pair_key(m.id, b if a in in_group else a))

    # ------------------------------------------------------------ one frame
    def save_state(self):
        return {"objs": [o.copy() for o in self.objs], "next_id": self.next_id, "level": self.level,
                "co_move": dict(self.co_move), "blocked": set(self.blocked), "hidden_for": dict(self.hidden_for),
                "lib": [e.copy() for e in self.lib.values()], "tracks": [tr.copy() for tr in self.tracks.values()]}

    def restore_state(self, st):
        self.set_objs([o.copy() for o in st["objs"]])
        self.next_id = st["next_id"]
        self.level = st["level"]
        self.co_move = dict(st["co_move"])
        self.blocked = set(st["blocked"])
        self.hidden_for = dict(st["hidden_for"])
        self.lib = {}
        for e in st["lib"]:
            self.lib[e.key] = e.copy()
        self.lib_version += 1
        self.tracks = {tr.id: tr.copy() for tr in st["tracks"]}
        self.redraw()

    def step_frame(self, rec):
        big = rec["wrong0"] > self.opts["rereadFrac"] * self.HW
        s0 = self.save_state() if big else None
        self.edits(rec)
        if not big or rec["fresh"]:
            return
        cost_a = sum(e["cost"] for e in rec["edits"])
        sA = self.save_state()
        edits_a = rec["edits"]
        wrong5 = rec["wrong5"]
        self.restore_state(s0)
        rec["edits"] = []
        rec["fresh"] = "reread"
        self.read_afresh(rec, 11)
        cost_b = sum(e["cost"] for e in rec["edits"])
        rec["alt"] = {"edits": cost_a, "reread": cost_b}
        if cost_a <= cost_b:
            self.restore_state(sA)
            rec["edits"] = edits_a
            rec["fresh"] = None
            rec["wrong5"] = wrong5

    def edits(self, rec):
        if rec["wrong0"] >= self.opts["scrollMin"]:
            self.scroll(rec)
        before = {o.id: (o.t, o.r, o.c, o.k, o.z) for o in self.objs}
        if rec["wrong0"]:
            sus = self.suspects()
            props = {o.id: self.proposals(o) for o in sus}
            self.greedy(props, rec)
            self.splits(sus, props, before, rec)
            rec["wrong5"] = self.wrong_count()
            self.fix_layers(rec)
            self.learn(rec)
            self.repair(rec)
        self.comotion(before, rec)

    def step(self, grid: np.ndarray, lv: int | None = None, step_last: bool = True) -> dict:
        """Explain one frame (h x w colours). lv = the game's level count for this frame (None = unknown)."""
        self.grid = np.ascontiguousarray(grid, dtype=np.int64)
        self.F = self.grid.ravel()
        rec = {"fi": self.fi, "edits": [], "cost": 0, "wrong0": 0, "wrong5": 0, "fresh": None, "exact": False, "level": 0}
        if self.fi == 0:
            rec["fresh"] = "start"
            rec["wrong0"] = self.HW
            self.pending = None
            self.level_start(rec, lv if lv is not None else 0, True)
        else:
            if lv is not None and lv != self.level:
                self.pending = lv
            self.redraw()
            rec["wrong0"] = self.wrong_count()
            if self.pending is not None and (rec["wrong0"] > self.opts["levelFrac"] * self.HW or step_last):
                nl = self.pending
                self.pending = None
                rec["fresh"] = "level"
                self.level_start(rec, nl, False)
            else:
                self.step_frame(rec)
        rec["level"] = self.level
        self.normalise_z()
        self.redraw()
        rec["exact"] = self.wrong_count() == 0
        own = self.view.own
        vis = np.bincount(own[own >= 0], minlength=max([o.id for o in self.objs] + [0]) + 1)
        for o in list(self.objs):
            n = 0 if vis[o.id] > 0 else self.hidden_for.get(o.id, 0) + 1
            self.hidden_for[o.id] = n
            if n >= self.opts["forgetAfter"]:
                self.drop_obj(o.id)
                self.died(o, "forgotten")
                rec["edits"].append({"step": 12, "type": "forget", "id": o.id, "cost": 0})
        for o in self.objs:
            tr = self.tracks[o.id]
            tr.spans[-1][1] = self.fi
            tr.levels.add(self.level)
            if tr.cur is not None and tr.cur.t is o.t:
                tr.cur.n += 1
            else:
                tr.cur = Ctr(o.t, 1)
            if tr.stable is None or tr.cur.n > tr.stable.n:
                tr.stable = tr.cur
        rec["cost"] = sum(e["cost"] for e in rec["edits"])
        rec["objs"] = [(o.id, o.t, o.r, o.c, o.k, o.z, int(vis[o.id]) if o.id < vis.shape[0] else 0) for o in self.objs]
        self.fi += 1
        return rec

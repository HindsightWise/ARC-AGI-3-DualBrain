"""Numeric kernels of the scene tracker (scene_tracker.py).

Every function here is plain Python over numpy arrays and ints, written so numba can compile it (nopython mode).
With numba available (it ships with vllm) they are compiled on first use and cached on disk (NUMBA_CACHE_DIR);
without numba they run as ordinary (slow) Python, with identical results.

Port of the PEEL fitting code (fitCells and its parts) and the hot cell loops of TRACK from the teammate's
arc_object_tracker.html. Comments name the JS function each kernel mirrors.
"""
from __future__ import annotations

import os

import numpy as np

HAVE_NUMBA = False
if os.environ.get("SCENE_TRACKER_NO_JIT", "") != "1":
    try:
        import numba as _nb   # noqa: F401
        HAVE_NUMBA = True
    except Exception:          # pragma: no cover - numba missing: plain Python
        HAVE_NUMBA = False


def _jit(fn):
    if HAVE_NUMBA:
        import numba
        return numba.njit(cache=True, nogil=True)(fn)
    return fn


# PEEL COST = { point: 2, hvLine: 3, diagLine: 4, rectCorner: 1, diagCorner: 2, thickness: 1, primitive: 1 }
# kind codes: 0 rect, 1 polygon, 2 outline, 3 point, 4 line, 5 lines;  KIND_RANK = rect/polygon 0, outline 1, lines 2
KIND_NAMES = ("rect", "polygon", "outline", "point", "line", "lines")


@_jit
def _kind_rank(k):
    if k <= 1:
        return 0
    if k == 2:
        return 1
    return 2


@_jit
def _line_set(W, H, d):
    """forEachLine: (starts, lengths, step) of every line of a W x H window in LINE_DIRS[d], in JS order."""
    if d == 0:
        dr, dc = 0, 1
    elif d == 1:
        dr, dc = 1, 0
    elif d == 2:
        dr, dc = 1, 1
    else:
        dr, dc = 1, -1
    step = dr * W + dc
    if dr == 0:
        S = np.empty(H, np.int64)
        L = np.empty(H, np.int64)
        for y in range(H):
            S[y] = y * W
            L[y] = W
        return S, L, step
    if dc == 0:
        S = np.empty(W, np.int64)
        L = np.empty(W, np.int64)
        for x in range(W):
            S[x] = x
            L[x] = H
        return S, L, step
    S = np.empty(W + H - 1, np.int64)
    L = np.empty(W + H - 1, np.int64)
    n = 0
    if dc == 1:
        for x in range(W):
            S[n] = x
            L[n] = min(H, W - x)
            n += 1
        for y in range(1, H):
            S[n] = y * W
            L[n] = min(H - y, W)
            n += 1
    else:
        for x in range(W):
            S[n] = x
            L[n] = min(H, x + 1)
            n += 1
        for y in range(1, H):
            S[n] = y * W + W - 1
            L[n] = min(H - y, W)
            n += 1
    return S, L, step


@_jit
def gap_fill(base, unk, W, H, nDirs):
    """gapFill: close gaps between two shape cells on a line (rows/columns, also diagonals) when the gap is all unknown."""
    m = base.copy()
    changed = True
    while changed:
        changed = False
        for d in range(nDirs):
            S, L, step = _line_set(W, H, d)
            for li in range(S.shape[0]):
                s = S[li]
                ln = L[li]
                prev = -1
                for t in range(ln):
                    if not m[s + t * step]:
                        continue
                    if prev >= 0 and t - prev > 1:
                        ok = True
                        for u in range(prev + 1, t):
                            if not unk[s + u * step]:
                                ok = False
                                break
                        if ok:
                            for u in range(prev + 1, t):
                                m[s + u * step] = 1
                            changed = True
                    prev = t
    return m


@_jit
def hole_fill(mask, unk, W, H):
    """holeFill: (mask with all-unknown holes filled, mask with every hole filled, number of holes)."""
    N = W * H
    seen = np.zeros(N, np.uint8)
    stack = np.empty(N, np.int64)
    comp = np.empty(N, np.int64)
    sp = 0
    holes = 0
    outU = mask.copy()
    outA = mask.copy()
    for x in range(W):
        for i in (x, N - W + x):
            if not mask[i] and not seen[i]:
                seen[i] = 1
                stack[sp] = i
                sp += 1
    for y in range(1, H - 1):
        for i in (y * W, y * W + W - 1):
            if not mask[i] and not seen[i]:
                seen[i] = 1
                stack[sp] = i
                sp += 1
    while sp:
        sp -= 1
        i = stack[sp]
        x = i % W
        if x > 0 and not mask[i - 1] and not seen[i - 1]:
            seen[i - 1] = 1
            stack[sp] = i - 1
            sp += 1
        if x < W - 1 and not mask[i + 1] and not seen[i + 1]:
            seen[i + 1] = 1
            stack[sp] = i + 1
            sp += 1
        if i >= W and not mask[i - W] and not seen[i - W]:
            seen[i - W] = 1
            stack[sp] = i - W
            sp += 1
        if i < N - W and not mask[i + W] and not seen[i + W]:
            seen[i + W] = 1
            stack[sp] = i + W
            sp += 1
    for i0 in range(N):
        if mask[i0] or seen[i0]:
            continue
        holes += 1
        n = 0
        ok = True
        seen[i0] = 1
        stack[sp] = i0
        sp += 1
        while sp:
            sp -= 1
            j = stack[sp]
            x = j % W
            comp[n] = j
            n += 1
            if not unk[j]:
                ok = False
            if x > 0 and not mask[j - 1] and not seen[j - 1]:
                seen[j - 1] = 1
                stack[sp] = j - 1
                sp += 1
            if x < W - 1 and not mask[j + 1] and not seen[j + 1]:
                seen[j + 1] = 1
                stack[sp] = j + 1
                sp += 1
            if j >= W and not mask[j - W] and not seen[j - W]:
                seen[j - W] = 1
                stack[sp] = j - W
                sp += 1
            if j < N - W and not mask[j + W] and not seen[j + W]:
                seen[j + W] = 1
                stack[sp] = j + W
                sp += 1
        for k in range(n):
            outA[comp[k]] = 1
        if not ok:
            continue
        for k in range(n):
            outU[comp[k]] = 1
    return outU, outA, holes


@_jit
def _group_runs(rd, rl, n, s):
    """groupRuns: number of edges, and whether any is a 45-degree staircase."""
    i = 0
    ng = 0
    anyd = False
    while i < n:
        if rl[(s + i) % n] == 1:
            j = i
            hd = -1
            vd = -1
            while j < n:
                q = (s + j) % n
                if rl[q] != 1:
                    break
                if rd[q] % 2 == 0:
                    if hd < 0:
                        hd = rd[q]
                    elif hd != rd[q]:
                        break
                else:
                    if vd < 0:
                        vd = rd[q]
                    elif vd != rd[q]:
                        break
                j += 1
            if j - i >= 2:
                ng += 1
                anyd = True
                i = j
                continue
        ng += 1
        i += 1
    return ng, anyd


@_jit
def _ring_cost(rd, rl, n):
    """ringCost: (cost, uses 45-degree edges)."""
    best = n * 1 + 1
    diag = False
    unit_pair = False
    for i in range(n):
        if rl[i] == 1 and rl[(i + 1) % n] == 1:
            unit_pair = True
            break
    if not unit_pair:
        return best, diag
    non_unit = -1
    for i in range(n):
        if rl[i] > 1:
            non_unit = i
            break
    starts = np.empty(2, np.int64)
    ns = 0
    if non_unit >= 0:
        starts[0] = (non_unit + 1) % n
        ns = 1
    else:
        c = -1
        for i in range(n):
            if rd[i] != rd[(i - 2 + n) % n]:
                c = i
                break
        starts[0] = (c - 1 + n) % n
        starts[1] = c
        ns = 2
    for si in range(ns):
        s = starts[si]
        ng, anyd = _group_runs(rd, rl, n, s)
        if not anyd:
            continue
        cost = ng * 2 + 1
        if cost < best:
            best = cost
            diag = True
    return best, diag


@_jit
def polygon_cost(mask, W, H):
    """polygonOf + polyKind: (total ring cost, kind 0 = rect / 1 = polygon). traceRings with interior on the right."""
    LW = W + 1
    NP = LW * (H + 1)
    out = np.zeros(NP, np.uint8)
    for y in range(1, H - 1):
        for x in range(1, W - 1):
            i = y * W + x
            if not mask[i]:
                continue
            if not mask[i - W]:
                out[y * LW + x] |= 1
            if not mask[i + 1]:
                out[y * LW + x + 1] |= 2
            if not mask[i + W]:
                out[(y + 1) * LW + x + 1] |= 4
            if not mask[i - 1]:
                out[(y + 1) * LW + x] |= 8
    used = np.zeros(NP, np.uint8)
    dirs = np.empty(4 * NP + 4, np.int64)
    rd = np.empty(4 * NP + 4, np.int64)
    rl = np.empty(4 * NP + 4, np.int64)
    DXa = (1, 0, -1, 0)
    DYa = (0, 1, 0, -1)
    total = 0
    nrings = 0
    first_diag = False
    first_runs = 0
    for p in range(NP):
        for d in range(4):
            if not ((out[p] >> d) & 1) or ((used[p] >> d) & 1):
                continue
            sx = p % LW
            sy = p // LW
            nd = 0
            x = sx
            y = sy
            dr_ = d
            guard = 4 * LW * (H + 1)
            while True:
                used[y * LW + x] |= (1 << dr_)
                dirs[nd] = dr_
                nd += 1
                x += DXa[dr_]
                y += DYa[dr_]
                o = out[y * LW + x]
                rt = (dr_ + 1) & 3
                lt = (dr_ + 3) & 3
                if (o >> rt) & 1:
                    dr_ = rt
                elif (o >> dr_) & 1:
                    pass
                else:
                    dr_ = lt
                guard -= 1
                if (x == sx and y == sy and dr_ == d) or guard <= 0:
                    break
            # ringFromDirs: runs starting at a corner
            k = 0
            while k < nd and dirs[k] == dirs[(k + nd - 1) % nd]:
                k += 1
            nr = 0
            for i in range(nd):
                dd = dirs[(k + i) % nd]
                if nr > 0 and rd[nr - 1] == dd:
                    rl[nr - 1] += 1
                else:
                    rd[nr] = dd
                    rl[nr] = 1
                    nr += 1
            cost, diag = _ring_cost(rd, rl, nr)
            if nrings == 0:
                first_diag = diag
                first_runs = nr
            nrings += 1
            total += cost
    kind = 0 if (nrings == 1 and not first_diag and first_runs == 4) else 1
    return total, kind


@_jit
def dist_to_outside(F, W, H):
    """distToOutside: 4-neighbour distance of every mask cell to the nearest cell outside the mask."""
    N = W * H
    dist = np.zeros(N, np.int64)
    q = np.empty(N, np.int64)
    qh = 0
    qt = 0
    for i in range(N):
        if not F[i]:
            continue
        x = i % W
        y = i // W
        if x == 0 or y == 0 or x == W - 1 or y == H - 1 or not F[i - 1] or not F[i + 1] or not F[i - W] or not F[i + W]:
            dist[i] = 1
            q[qt] = i
            qt += 1
    while qh < qt:
        i = q[qh]
        qh += 1
        x = i % W
        d = dist[i] + 1
        if x > 0 and F[i - 1] and not dist[i - 1]:
            dist[i - 1] = d
            q[qt] = i - 1
            qt += 1
        if x < W - 1 and F[i + 1] and not dist[i + 1]:
            dist[i + 1] = d
            q[qt] = i + 1
            qt += 1
        if i >= W and F[i - W] and not dist[i - W]:
            dist[i - W] = d
            q[qt] = i - W
            qt += 1
        if i < N - W and F[i + W] and not dist[i + W]:
            dist[i + W] = d
            q[qt] = i + W
            qt += 1
    return dist


@_jit
def outline_cost(mask, Fall, nholes, W, H):
    """outlineOf: cost of the mask as a band of some thickness inside its filled polygon, or -1."""
    if nholes == 0 or nholes > 8:
        return -1
    dist = dist_to_outside(Fall, W, H)
    t = 0
    for i in range(Fall.shape[0]):
        if mask[i] and dist[i] > t:
            t = dist[i]
    for i in range(Fall.shape[0]):
        if Fall[i] and ((dist[i] <= t) != (mask[i] == 1)):
            return -1
    pc, _k = polygon_cost(Fall, W, H)
    return pc + 1


@_jit
def _lbetter(p, q, G, C):
    a = G[p] * C[q]
    b = G[q] * C[p]
    if a != b:
        return a > b
    if G[p] != G[q]:
        return G[p] > G[q]
    if C[p] != C[q]:
        return C[p] < C[q]
    return p < q


@_jit
def lines_cost(own, allowed, W, H, bound, nOwn, maxLen):
    """linesOf: (ok, kind 3 point / 4 line / 5 lines, cost, mask) - greedy weighted set cover by lines."""
    N = W * H
    empty = np.zeros(N, np.uint8)
    if nOwn > 1 and nOwn * 4 > bound * max(2, maxLen):
        return False, 0, 0, empty
    cap = 4 * N + 8
    S = np.empty(cap, np.int64)
    ST = np.empty(cap, np.int64)
    L = np.empty(cap, np.int64)
    D = np.empty(cap, np.int64)
    C = np.empty(cap, np.int64)
    G = np.empty(cap, np.int64)
    nc = 0
    point_seen = np.zeros(N, np.uint8)
    for d in range(4):
        LS, LL, step = _line_set(W, H, d)
        for li in range(LS.shape[0]):
            s = LS[li]
            ln = LL[li]
            t = 0
            while t < ln:
                if not allowed[s + t * step]:
                    t += 1
                    continue
                a = -1
                b = -1
                g = 0
                u = t
                while u < ln and allowed[s + u * step]:
                    if own[s + u * step]:
                        if a < 0:
                            a = u
                        b = u
                        g += 1
                    u += 1
                if a >= 0 and (a != b or not point_seen[s + a * step]):
                    if a == b:
                        point_seen[s + a * step] = 1
                    S[nc] = s + a * step
                    ST[nc] = step
                    L[nc] = b - a + 1
                    D[nc] = -1 if a == b else d
                    G[nc] = g
                    C[nc] = (2 if a == b else (3 if d < 2 else 4)) + 1
                    nc += 1
                t = u
    heap = np.empty(nc + 1, np.int64)
    hn = 0
    for i in range(nc):
        heap[hn] = i
        k = hn
        hn += 1
        while k > 0:
            p = (k - 1) >> 1
            if not _lbetter(heap[k], heap[p], G, C):
                break
            tmp = heap[k]
            heap[k] = heap[p]
            heap[p] = tmp
            k = p
    covered = np.zeros(N, np.uint8)
    chosen = np.empty(nc + 1, np.int64)
    nch = 0
    left = nOwn
    total = 0
    while left > 0 and hn:
        top = heap[0]
        hn -= 1
        heap[0] = heap[hn]
        k = 0
        while True:
            l = 2 * k + 1
            r = l + 1
            m = k
            if l < hn and _lbetter(heap[l], heap[m], G, C):
                m = l
            if r < hn and _lbetter(heap[r], heap[m], G, C):
                m = r
            if m == k:
                break
            tmp = heap[k]
            heap[k] = heap[m]
            heap[m] = tmp
            k = m
        i = top
        g = 0
        c = S[i]
        for kk in range(L[i]):
            if own[c] and not covered[c]:
                g += 1
            c += ST[i]
        if not g:
            continue
        if g != G[i]:
            G[i] = g
            heap[hn] = i
            k = hn
            hn += 1
            while k > 0:
                p = (k - 1) >> 1
                if not _lbetter(heap[k], heap[p], G, C):
                    break
                tmp = heap[k]
                heap[k] = heap[p]
                heap[p] = tmp
                k = p
            continue
        chosen[nch] = i
        nch += 1
        total += C[i]
        if total > bound:
            return False, 0, 0, empty
        c = S[i]
        for kk in range(L[i]):
            if own[c] and not covered[c]:
                covered[c] = 1
                left -= 1
            c += ST[i]
    cnt = np.zeros(N, np.int64)
    for jj in range(nch):
        i = chosen[jj]
        c = S[i]
        for kk in range(L[i]):
            if own[c]:
                cnt[c] += 1
            c += ST[i]
    keep = np.ones(nch, np.uint8)
    for jj in range(nch - 1, -1, -1):
        i = chosen[jj]
        redundant = True
        c = S[i]
        for kk in range(L[i]):
            if own[c] and cnt[c] < 2:
                redundant = False
                break
            c += ST[i]
        if not redundant:
            continue
        c = S[i]
        for kk in range(L[i]):
            if own[c]:
                cnt[c] -= 1
            c += ST[i]
        total -= C[i]
        keep[jj] = 0
    mask = np.zeros(N, np.uint8)
    nseg = 0
    last = -1
    for jj in range(nch):
        if not keep[jj]:
            continue
        nseg += 1
        last = chosen[jj]
        i = chosen[jj]
        c = S[i]
        for kk in range(L[i]):
            mask[c] = 1
            c += ST[i]
    if nseg == 1:
        kind = 3 if D[last] < 0 else 4
    else:
        kind = 5
    return True, kind, total, mask


@_jit
def _same(a, b):
    if a.shape[0] != b.shape[0]:
        return False
    for i in range(a.shape[0]):
        if a[i] != b[i]:
            return False
    return True


@_jit
def fit_cells(O, U, hasU, w):
    """fitCells: cheapest description of cells O (grid indices, width w), allowed to grow into unknown cells U
    inside O's bounding box. Returns (kind, cost, r0, c0, bh, bw, mask (bh*bw), cells)."""
    r0 = 1 << 40
    r1 = -1
    c0 = 1 << 40
    c1 = -1
    for k in range(O.shape[0]):
        r = O[k] // w
        c = O[k] - r * w
        if r < r0:
            r0 = r
        if r > r1:
            r1 = r
        if c < c0:
            c0 = c
        if c > c1:
            c1 = c
    bh = r1 - r0 + 1
    bw = c1 - c0 + 1
    W = bw + 2
    H = bh + 2
    own = np.zeros(W * H, np.uint8)
    unk = np.zeros(W * H, np.uint8)
    for k in range(O.shape[0]):
        r = O[k] // w
        c = O[k] - r * w
        own[(r - r0 + 1) * W + c - c0 + 1] = 1
    n_unk = 0
    if hasU:
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                j = (r - r0 + 1) * W + c - c0 + 1
                if U[r * w + c] and not own[j]:
                    unk[j] = 1
                    n_unk += 1
    allowed = own
    bases = [own]
    if n_unk:
        allowed = np.zeros(W * H, np.uint8)
        for i in range(W * H):
            allowed[i] = own[i] | unk[i]
        hv = gap_fill(own, unk, W, H, 2)
        hv4 = gap_fill(hv, unk, W, H, 4)
        for mi in range(3):
            m = hv if mi == 0 else (hv4 if mi == 1 else allowed)
            dup = False
            for b in bases:
                if _same(b, m):
                    dup = True
                    break
            if not dup:
                bases.append(m)
    best_kind = -1
    best_cost = 0
    best_cells = 0
    best_mask = own
    polys = [own]
    npolys = 0
    for b in bases:
        outU, outA, holes = hole_fill(b, unk, W, H)
        dup = False
        for pi in range(npolys):
            if _same(polys[pi], outU):
                dup = True
                break
        if not dup:
            if npolys < len(polys):
                polys[npolys] = outU
            else:
                polys.append(outU)
            npolys += 1
            pc, pk = polygon_cost(outU, W, H)
            cells = 0
            for i in range(outU.shape[0]):
                cells += outU[i]
            if best_kind < 0 or pc < best_cost or (pc == best_cost and (cells < best_cells or (cells == best_cells and _kind_rank(pk) < _kind_rank(best_kind)))):
                best_kind = pk
                best_cost = pc
                best_cells = cells
                best_mask = outU
        oc = outline_cost(b, outA, holes, W, H)
        if oc >= 0:
            cells = 0
            for i in range(b.shape[0]):
                cells += b[i]
            if oc < best_cost or (oc == best_cost and (cells < best_cells or (cells == best_cells and _kind_rank(2) < _kind_rank(best_kind)))):
                best_kind = 2
                best_cost = oc
                best_cells = cells
                best_mask = b
    max_len = max(bh, bw)
    nal = 2 if n_unk else 1
    for ai in range(nal):
        al = own if ai == 0 else allowed
        ok, lk, lc, lm = lines_cost(own, al, W, H, best_cost, O.shape[0], max_len)
        if ok:
            cells = 0
            for i in range(lm.shape[0]):
                cells += lm[i]
            if lc < best_cost or (lc == best_cost and (cells < best_cells or (cells == best_cells and _kind_rank(lk) < _kind_rank(best_kind)))):
                best_kind = lk
                best_cost = lc
                best_cells = cells
                best_mask = lm
    mask = np.empty(bh * bw, np.uint8)
    for y in range(bh):
        for x in range(bw):
            mask[y * bw + x] = best_mask[(y + 1) * W + x + 1]
    return best_kind, best_cost, r0, c0, bh, bw, mask, best_cells


@_jit
def components(grid_flat, h, w):
    """ARCOBJ.components: 4-connected single-colour components, numbered in reading order of their first cell.
    Returns (label per cell, number of components)."""
    N = h * w
    cid = np.full(N, -1, np.int64)
    stack = np.empty(N, np.int64)
    n = 0
    for s in range(N):
        if cid[s] != -1:
            continue
        v = grid_flat[s]
        cid[s] = n
        sp = 0
        stack[sp] = s
        sp += 1
        while sp:
            sp -= 1
            g = stack[sp]
            r = g // w
            c = g - r * w
            if r > 0 and cid[g - w] == -1 and grid_flat[g - w] == v:
                cid[g - w] = n
                stack[sp] = g - w
                sp += 1
            if r + 1 < h and cid[g + w] == -1 and grid_flat[g + w] == v:
                cid[g + w] = n
                stack[sp] = g + w
                sp += 1
            if c > 0 and cid[g - 1] == -1 and grid_flat[g - 1] == v:
                cid[g - 1] = n
                stack[sp] = g - 1
                sp += 1
            if c + 1 < w and cid[g + 1] == -1 and grid_flat[g + 1] == v:
                cid[g + 1] = n
                stack[sp] = g + 1
                sp += 1
        n += 1
    return cid, n


@_jit
def flood_touch(seed, U, owner, h, w, lim0, lim1, lim2, lim3, use_lim):
    """mergePairsAround (search part): owners of the cells 8-adjacent to the unknown region (8-connected) that
    contains the seed cells, optionally inside the box lim. Returns a 0/1 flag per owner id (length = owner.max()+1)."""
    N = h * w
    seen = np.zeros(N, np.uint8)
    stack = np.empty(N + seed.shape[0], np.int64)
    sp = 0
    for k in range(seed.shape[0]):
        seen[seed[k]] = 1
        stack[sp] = seed[k]
        sp += 1
    mx = -1
    for i in range(N):
        if owner[i] > mx:
            mx = owner[i]
    touch = np.zeros(mx + 2, np.uint8)
    while sp:
        sp -= 1
        i = stack[sp]
        r = i // w
        c = i - r * w
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                rr = r + dr
                cc = c + dc
                if (dr == 0 and dc == 0) or rr < 0 or cc < 0 or rr >= h or cc >= w:
                    continue
                if use_lim and (rr < lim0 or rr > lim1 or cc < lim2 or cc > lim3):
                    continue
                j = rr * w + cc
                if U[j]:
                    if not seen[j]:
                        seen[j] = 1
                        stack[sp] = j
                        sp += 1
                elif owner[j] >= 0:
                    touch[owner[j]] = 1
    return touch


# ------------------------------------------------------------------ TRACK kernels

@_jit
def tpl_hash(th, tw, idx, col):
    """tplHash (32-bit FNV-like, Math.imul semantics), unsigned."""
    M = 0xFFFFFFFF
    h = ((th * 31) & M) ^ tw
    for i in range(idx.shape[0]):
        h = ((h ^ idx[i]) * 16777619) & M
        h = ((h ^ (col[i] + 1)) * 16777619) & M
    return h


@_jit
def render_scene(order, s_r, s_c, s_id, s_z, off, ys, xs, cols, h, w):
    """renderScene: paint the objects in `order` (indices into the scene arrays, back to front).
    Returns col, own, zt, uid, ucol (-1 = nothing)."""
    N = h * w
    col = np.full(N, -1, np.int64)
    own = np.full(N, -1, np.int64)
    zt = np.full(N, np.inf, np.float64)
    uid = np.full(N, -1, np.int64)
    ucol = np.full(N, -1, np.int64)
    for kk in range(order.shape[0]):
        k = order[kk]
        r = s_r[k]
        c = s_c[k]
        oid = s_id[k]
        z = s_z[k]
        for i in range(off[k], off[k + 1]):
            y = r + ys[i]
            x = c + xs[i]
            if y < 0 or x < 0 or y >= h or x >= w:
                continue
            g = y * w + x
            uid[g] = own[g]
            ucol[g] = col[g]
            col[g] = cols[i]
            own[g] = oid
            zt[g] = z
    return col, own, zt, uid, ucol


@_jit
def render_rect(order_r, order_c, order_id, off, ys, xs, cols, r0, r1, c0, c1):
    """renderRect: paint the listed objects (given back to front) inside the rectangle. Returns col, own."""
    RW = c1 - c0 + 1
    n = (r1 - r0 + 1) * RW
    col = np.full(n, -1, np.int64)
    own = np.full(n, -1, np.int64)
    for k in range(order_r.shape[0]):
        r = order_r[k]
        c = order_c[k]
        for i in range(off[k], off[k + 1]):
            y = r + ys[i]
            x = c + xs[i]
            if y < r0 or y > r1 or x < c0 or x > c1:
                continue
            j = (y - r0) * RW + x - c0
            col[j] = cols[i]
            own[j] = order_id[k]
    return col, own


@_jit
def search_poses(ks, rbs, cbs, soff, sdy, sdx, scol, o_k, o_r, o_c, o_id, o_z, R, keep, own, zt, vcol, F, h, w):
    """searchPoses: best `keep` positions/orientations of a shape within R cells. Returns arrays (score, dist, r, c, k)."""
    bs = np.zeros(keep + 1, np.int64)
    bd = np.zeros(keep + 1, np.int64)
    br = np.zeros(keep + 1, np.int64)
    bc = np.zeros(keep + 1, np.int64)
    bk = np.zeros(keep + 1, np.int64)
    nb = 0
    for ki in range(ks.shape[0]):
        k = ks[ki]
        rb = rbs[ki]
        cb = cbs[ki]
        s0 = soff[ki]
        sn = soff[ki + 1] - s0
        for dr in range(-R, R + 1):
            for dc in range(-R, R + 1):
                r = rb + dr
                c = cb + dc
                if k == o_k and r == o_r and c == o_c:
                    continue
                floor = bs[nb - 1] if nb == keep else 1
                score = 0
                used = 0
                for s in range(sn):
                    if score + sn - s < floor:
                        used = 0
                        break
                    y = r + sdy[s0 + s]
                    x = c + sdx[s0 + s]
                    if y < 0 or x < 0 or y >= h or x >= w:
                        continue
                    g = y * w + x
                    if own[g] != o_id and zt[g] < o_z and vcol[g] == F[g]:
                        continue
                    used += 1
                    if F[g] == scol[s0 + s]:
                        score += 1
                    else:
                        score -= 1
                if used == 0 or score <= 0:
                    continue
                dist = abs(r - o_r) + abs(c - o_c) + (0 if k == o_k else 1)
                if nb == keep and (score < bs[nb - 1] or (score == bs[nb - 1] and dist >= bd[nb - 1])):
                    continue
                # push then stable sort by (-score, dist), then pop if too long
                bs[nb] = score
                bd[nb] = dist
                br[nb] = r
                bc[nb] = c
                bk[nb] = k
                nb += 1
                j = nb - 1
                while j > 0 and (bs[j - 1] < bs[j] or (bs[j - 1] == bs[j] and bd[j - 1] > bd[j])):
                    t0 = bs[j]; bs[j] = bs[j - 1]; bs[j - 1] = t0
                    t0 = bd[j]; bd[j] = bd[j - 1]; bd[j - 1] = t0
                    t0 = br[j]; br[j] = br[j - 1]; br[j - 1] = t0
                    t0 = bc[j]; bc[j] = bc[j - 1]; bc[j - 1] = t0
                    t0 = bk[j]; bk[j] = bk[j - 1]; bk[j - 1] = t0
                    j -= 1
                if nb > keep:
                    nb -= 1
    return bs[:nb].copy(), bd[:nb].copy(), br[:nb].copy(), bc[:nb].copy(), bk[:nb].copy()


@_jit
def scroll_counts(sample, vcol, F, h, w, S):
    """scroll (candidate part): for every shift (dr, dc) in [-S, S]^2 except (0, 0), the sampled wrong cells the
    shifted view would set right. Returns (dr, dc, n) of the shifts with 4n >= samples, in loop order."""
    m = sample.shape[0]
    tot = (2 * S + 1) * (2 * S + 1)
    odr = np.empty(tot, np.int64)
    odc = np.empty(tot, np.int64)
    on = np.empty(tot, np.int64)
    k = 0
    for dr in range(-S, S + 1):
        for dc in range(-S, S + 1):
            if dr == 0 and dc == 0:
                continue
            n = 0
            for j in range(m):
                g = sample[j]
                y = g // w - dr
                x = g % w - dc
                if y >= 0 and x >= 0 and y < h and x < w and vcol[y * w + x] == F[g]:
                    n += 1
            if n * 4 >= m:
                odr[k] = dr
                odc[k] = dc
                on[k] = n
                k += 1
    return odr[:k].copy(), odc[:k].copy(), on[:k].copy()


@_jit
def next_below_slow(g, top_id, top_z, w, s_id, s_r, s_c, s_z, s_th, s_tw, s_atoff, at_all, s_coloff, col_all):
    """nextBelow (scan part): the object right behind `top` at cell g, scanning the scene in its list order.
    Returns (index into the scene arrays or -1, colour)."""
    best = -1
    bcol = -1
    bz = np.inf
    y = g // w
    x = g % w
    for j in range(s_id.shape[0]):
        if s_id[j] == top_id:
            continue
        z = s_z[j]
        if z <= top_z or z >= bz:
            continue
        ly = y - s_r[j]
        lx = x - s_c[j]
        if ly < 0 or lx < 0 or ly >= s_th[j] or lx >= s_tw[j]:
            continue
        i = at_all[s_atoff[j] + ly * s_tw[j] + lx]
        if i >= 0:
            best = j
            bcol = col_all[s_coloff[j] + i]
            bz = z
    return best, bcol


@_jit
def _below(g, top_id, top_z, vown, vuid, vucol, z_by_id, w, s_id, s_r, s_c, s_z, s_th, s_tw, s_atoff, at_all,
           s_coloff, col_all):
    """nextBelow(g, top): (id, colour, z) of the object right behind top at g; id -1 = none, -2 = the fast path
    names an object that is no longer in the scene (only its colour is known)."""
    if vown[g] == top_id and vuid[g] != -2:
        u = vuid[g]
        if u >= 0:
            if u >= z_by_id.shape[0] or np.isnan(z_by_id[u]):
                return -2, vucol[g], np.inf
            return u, vucol[g], z_by_id[u]
        return -1, -1, np.inf
    j, c = next_below_slow(g, top_id, top_z, w, s_id, s_r, s_c, s_z, s_th, s_tw, s_atoff, at_all, s_coloff, col_all)
    if j >= 0:
        return s_id[j], c, s_z[j]
    return -1, -1, np.inf


@_jit
def below_cells(gs, top_ids, top_zs, vown, vuid, vucol, z_by_id, w, s_id, s_r, s_c, s_z, s_th, s_tw, s_atoff,
                at_all, s_coloff, col_all):
    """nextBelow for many cells: arrays (id, colour) of the object right behind top_ids[k] at gs[k]."""
    n = gs.shape[0]
    bid = np.full(n, -1, np.int64)
    bcol = np.full(n, -1, np.int64)
    for k in range(n):
        b, c, z = _below(gs[k], top_ids[k], top_zs[k], vown, vuid, vucol, z_by_id, w, s_id, s_r, s_c, s_z, s_th,
                         s_tw, s_atoff, at_all, s_coloff, col_all)
        bid[k] = b
        bcol[k] = c
    return bid, bcol


@_jit
def trial_swap(o_ys, o_xs, o_col, o_r, o_c, o_id, o_z, has_n, n_ys, n_xs, n_col, n_r, n_c, n_id, n_z,
               vcol, vown, vzt, vuid, vucol, F, h, w, z_by_id, s_id, s_r, s_c, s_z, s_th, s_tw, s_atoff, at_all,
               s_coloff, col_all):
    """trialSwap: the effect of replacing object o by n (or removing it, has_n False) on the view.
    Returns (fixed, broken, change cells, their colour, owner, z, all cells touched, error)."""
    HW = h * w
    mark = np.zeros(HW, np.uint8)
    old = np.empty(HW, np.int64)
    new = np.empty(HW, np.int64)
    cells = np.empty(o_ys.shape[0] + n_ys.shape[0], np.int64)
    nc = 0
    for i in range(o_ys.shape[0]):
        y = o_r + o_ys[i]
        x = o_c + o_xs[i]
        if y < 0 or x < 0 or y >= h or x >= w:
            continue
        g = y * w + x
        mark[g] = 1
        old[g] = o_col[i]
        new[g] = -1
        cells[nc] = g
        nc += 1
    if has_n:
        for i in range(n_ys.shape[0]):
            y = n_r + n_ys[i]
            x = n_c + n_xs[i]
            if y < 0 or x < 0 or y >= h or x >= w:
                continue
            g = y * w + x
            if mark[g] == 0:
                mark[g] = 1
                old[g] = -1
                cells[nc] = g
                nc += 1
            new[g] = n_col[i]
    cg = np.empty(nc, np.int64)
    ccol = np.empty(nc, np.int64)
    cown = np.empty(nc, np.int64)
    cz = np.empty(nc, np.float64)
    m = 0
    fixed = 0
    broken = 0
    err = 0
    for k in range(nc):
        g = cells[k]
        oc = old[g]
        nn = new[g]
        if oc == nn:
            continue
        col = vcol[g]
        own = vown[g]
        z = vzt[g]
        if own == o_id:
            b, bc, bz = _below(g, o_id, o_z, vown, vuid, vucol, z_by_id, w, s_id, s_r, s_c, s_z, s_th, s_tw, s_atoff,
                               at_all, s_coloff, col_all)
            if b == -2:
                err = 1
            if b == -1:
                col = -1
                own = -1
                z = np.inf
            else:
                col = bc
                own = b
                z = bz
        if nn >= 0 and (n_z < z or (n_z == z and n_id < own)):
            col = nn
            own = n_id
            z = n_z
        if col == vcol[g] and own == vown[g]:
            continue
        was = vcol[g] != F[g]
        is_ = col != F[g]
        if was and not is_:
            fixed += 1
        elif (not was) and is_:
            broken += 1
        cg[m] = g
        ccol[m] = col
        cown[m] = own
        cz[m] = z
        m += 1
    return fixed, broken, cg[:m].copy(), ccol[:m].copy(), cown[:m].copy(), cz[:m].copy(), cells[:nc].copy(), err


@_jit
def bound_count(o_ys, o_xs, o_col, o_r, o_c, o_id, o_z, is_recolor, rmap, has_n, n_ys, n_xs, n_col, n_r, n_c,
                vcol, vown, vuid, vucol, F, h, w, z_by_id, s_id, s_r, s_c, s_z, s_th, s_tw, s_atoff, at_all,
                s_coloff, col_all):
    """bound: wrong cells the proposal could set right (before subtracting its cost)."""
    n = 0
    for i in range(o_ys.shape[0]):
        y = o_r + o_ys[i]
        x = o_c + o_xs[i]
        if y < 0 or x < 0 or y >= h or x >= w:
            continue
        g = y * w + x
        if vown[g] != o_id or vcol[g] == F[g]:
            continue
        if is_recolor:
            if rmap[o_col[i]] == F[g]:
                n += 1
            continue
        b, bc, bz = _below(g, o_id, o_z, vown, vuid, vucol, z_by_id, w, s_id, s_r, s_c, s_z, s_th, s_tw, s_atoff,
                           at_all, s_coloff, col_all)
        if b != -1 and bc == F[g]:
            n += 1
    if has_n and not is_recolor:
        for i in range(n_ys.shape[0]):
            y = n_r + n_ys[i]
            x = n_c + n_xs[i]
            if y < 0 or x < 0 or y >= h or x >= w:
                continue
            g = y * w + x
            if vcol[g] != F[g] and n_col[i] == F[g]:
                n += 1
    return n


@_jit
def layer_votes(s_id, s_r, s_c, s_z, s_coff, ys, xs, cols, F, h, w):
    """fixLayers (votes): for every cell covered by several objects, the front-most object showing the right colour
    is voted in front of each covering object showing a wrong one. Returns arrays (front id, other id)."""
    HW = h * w
    cnt = np.zeros(HW + 1, np.int64)
    n = s_id.shape[0]
    for j in range(n):
        for i in range(s_coff[j], s_coff[j + 1]):
            y = s_r[j] + ys[i]
            x = s_c[j] + xs[i]
            if y < 0 or x < 0 or y >= h or x >= w:
                continue
            cnt[y * w + x + 1] += 1
    for g in range(HW):
        cnt[g + 1] += cnt[g]
    tot = cnt[HW]
    ids = np.empty(tot, np.int64)
    zs = np.empty(tot, np.float64)
    cc = np.empty(tot, np.int64)
    fill = cnt[:HW].copy()
    for j in range(n):
        for i in range(s_coff[j], s_coff[j + 1]):
            y = s_r[j] + ys[i]
            x = s_c[j] + xs[i]
            if y < 0 or x < 0 or y >= h or x >= w:
                continue
            g = y * w + x
            ids[fill[g]] = s_id[j]
            zs[fill[g]] = s_z[j]
            cc[fill[g]] = cols[i]
            fill[g] += 1
    fa = np.empty(tot, np.int64)
    fb = np.empty(tot, np.int64)
    m = 0
    for g in range(HW):
        s = cnt[g]
        e = cnt[g + 1]
        if e - s < 2:
            continue
        front = -1
        fz = np.inf
        for j in range(s, e):
            if cc[j] == F[g]:
                if zs[j] < fz:
                    fz = zs[j]
                    front = ids[j]
        if front < 0:
            continue
        for j in range(s, e):
            if cc[j] == F[g]:
                continue
            fa[m] = front
            fb[m] = ids[j]
            m += 1
    return fa[:m].copy(), fb[:m].copy()


@_jit
def split_labels(ys, xs, col, seen, at, th, tw, b_r, b_c, dr, dc, F, h, w, both_stay):
    """splitTrial (labelling): cells that match only where the object was stay (1), cells that match only at the
    offset move (2); the rest join their nearest labelled neighbour. Returns (ok, labels)."""
    n = ys.shape[0]
    lab = np.zeros(n, np.int64)
    sS = 0
    sM = 0
    for i in range(n):
        y = b_r + ys[i]
        x = b_c + xs[i]
        old = y >= 0 and x >= 0 and y < h and x < w and F[y * w + x] == col[i]
        y2 = y + dr
        x2 = x + dc
        neu = y2 >= 0 and x2 >= 0 and y2 < h and x2 < w and F[y2 * w + x2] == col[i]
        if old and ((not neu) or both_stay):
            lab[i] = 1
            sS += seen[i]
        elif neu and not old:
            lab[i] = 2
            sM += seen[i]
    if sS < 2 or sM < 2:
        return False, lab
    stay_col = np.full(th * tw, -1, np.int64)
    for i in range(n):
        if lab[i] == 1:
            stay_col[ys[i] * tw + xs[i]] = col[i]
    useful = 0
    for i in range(n):
        if lab[i] != 2:
            continue
        y = ys[i] + dr
        x = xs[i] + dc
        if y < 0 or x < 0 or y >= th or x >= tw or stay_col[y * tw + x] != col[i]:
            useful += 1
    if useful < 2:
        return False, lab
    q = np.empty(n, np.int64)
    nq = 0
    for i in range(n):
        if lab[i]:
            q[nq] = i
            nq += 1
    qi = 0
    while qi < nq:
        i = q[qi]
        qi += 1
        for d in range(4):
            if d == 0:
                yy = ys[i] - 1
                xx = xs[i]
            elif d == 1:
                yy = ys[i] + 1
                xx = xs[i]
            elif d == 2:
                yy = ys[i]
                xx = xs[i] - 1
            else:
                yy = ys[i]
                xx = xs[i] + 1
            if yy < 0 or xx < 0 or yy >= th or xx >= tw:
                continue
            j = at[yy * tw + xx]
            if j >= 0 and lab[j] == 0:
                lab[j] = lab[i]
                q[nq] = j
                nq += 1
    return True, lab

"""Measured component IDs, explicit groups and a conservative observation graph.

No model-authored code is evaluated. IDs name observations, not semantic truths.
Only unique measured continuations or explicitly followed agent assertions can
resolve an old reference. Spatial and hypothetical edges never redirect clicks.
"""
from __future__ import annotations

import copy
import hashlib
import re
from collections import Counter, defaultdict

import numpy as np

from . import objects
from . import reference_checks as checks


def board_key(board):
    a = np.asarray(board, dtype=np.uint8)
    return hashlib.sha256(str(a.shape).encode() + a.tobytes()).hexdigest()


def measured(board, cells, ref, **extra):
    cells = sorted(set(map(tuple, cells)))
    if not cells:
        raise ValueError("a reference needs occupied cells")
    rs, cs = zip(*cells)
    values = [int(board[r, c]) for r, c in cells]
    centre = (sum(rs) / len(rs), sum(cs) / len(cs))
    click = min(cells, key=lambda p: ((p[0]-centre[0])**2 + (p[1]-centre[1])**2, p))
    return {"id": ref, "cells": [list(p) for p in cells], "values": values,
            "colors": [[k, v] for k, v in sorted(Counter(values).items())], "size": len(cells),
            "rect": [min(rs), max(rs)+1, min(cs), max(cs)+1], "click_cell": list(click), **extra}


def shape(e, with_colors=True):
    r0, _, c0, _ = e["rect"]
    pts = tuple((r-r0, c-c0) for r, c in e["cells"])
    return (pts, tuple(e["values"])) if with_colors else pts


class ObjectReferences:
    def __init__(self, game, board, names=None):
        self.game, self.names = game, names or {}
        self.level, self.attempt, self.step = 1, 1, 0
        self.serial = 0
        self.current, self.nodes, self.successors, self.aliases = {}, {}, {}, {}
        self.edges, self.clicks = [], set()
        self.turn_edge_start = 0
        self.indicator_rects = []
        self.hash, self.board = None, None
        self.last_report, self.last_checks, self.expected = {}, [], []
        self.metrics = {"refreshes": 0, "reference_clicks": 0, "bindings": 0}
        self.refresh(board)

    @property
    def scope(self):
        return [self.level, self.attempt]

    def begin_turn(self):
        self.turn_edge_start = len(self.edges)
        self.last_report, self.last_checks, self.expected = {}, [], []

    def _new(self, board, cells, prefix="C", **extra):
        self.serial += 1
        e = measured(board, cells, f"{prefix}{self.serial}", **extra)
        # Historical nodes retain small descriptors, not every old board's cells.
        self.nodes[e["id"]] = {k: copy.deepcopy(v) for k, v in e.items() if k not in {"cells", "values"}}
        self.nodes[e["id"]].update(scope=self.scope, step=self.step)
        return e

    def _edge(self, source, relation, target, *, provenance="measured", follow=False, evidence=""):
        edge = {"source": source, "relation": relation, "target": target, "provenance": provenance,
                "follow": follow, "evidence": evidence[:480], "scope": self.scope, "step": self.step}
        if follow:
            self.successors[source] = target
        self.edges.append(edge)
        return edge

    def configure(self, level, attempt, step):
        if [int(level), int(attempt)] != self.scope:
            self.current, self.hash = {}, None
            self.last_report, self.last_checks, self.expected = {}, [], []
        self.level, self.attempt, self.step = int(level), int(attempt), int(step)

    def refresh(self, board, *, allow_match=True):
        board = np.asarray(board, dtype=np.uint8)
        key = board_key(board)
        if key == self.hash:
            return
        old = self.current
        # IDs need connected cells, not the expensive outlines, D4 hashes and
        # containment analysis produced by the full extraction helper.
        _, raw = objects._components(board)
        largest = max(range(len(raw)), key=lambda i: len(raw[i]["cells"])) if raw else None
        exact = {(tuple(map(tuple, e["cells"])), tuple(e["values"])): e for e in old.values()
                 if e["id"].startswith("C")}
        current, fresh = {}, []
        for index, o in enumerate(raw):
            cells = sorted(map(tuple, o["cells"]))
            values = tuple(int(board[r, c]) for r, c in cells)
            e = exact.get((tuple(cells), values)) if allow_match else None
            if e is None:
                e = self._new(board, cells, is_background=index == largest)
                fresh.append(e)
            current[e["id"]] = e
        unmatched = [e for k, e in old.items() if k not in current and k.startswith("C")]
        lines = []
        if allow_match:
            by_look, old_by_look = defaultdict(list), Counter()
            by_cells, old_by_cells = {}, {}
            for b in fresh:
                if not b.get("is_background") and not self._indicator(b):
                    by_look[shape(b)].append(b)
                    by_cells[tuple(map(tuple, b["cells"]))] = b
            for a in unmatched:
                if not a.get("is_background") and not self._indicator(a):
                    old_by_look[shape(a)] += 1
                    old_by_cells[tuple(map(tuple, a["cells"]))] = a
            for a in old_by_cells.values():
                moves = by_look.get(shape(a), [])
                recolor = by_cells.get(tuple(map(tuple, a["cells"])))
                if recolor and recolor["values"] == a["values"]:
                    recolor = None
                count = len(moves) + bool(recolor)
                if not count:
                    continue
                options = [(b["id"], "moved") for b in moves[:4]]
                if recolor:
                    options = [(recolor["id"], "recoloured")] + options[:3]
                b, relation = options[0]
                dest = current[b]
                previous_here = old_by_cells.get(tuple(map(tuple, dest["cells"])))
                reverse_count = old_by_look[shape(dest)] + bool(previous_here and previous_here["values"] != dest["values"])
                if count == 1 and reverse_count == 1:
                    self._edge(a["id"], relation, b, follow=True, evidence="unique component geometry/colour match in consecutive observations")
                    lines.append(f"{a['id']} -> {relation} -> {b}")
                else:
                    # Repeated tiles must not create a quadratic history graph.
                    self.nodes[a["id"]]["candidate_count"] = int(count)
                    for target, _ in options:
                        self._edge(a["id"], "possibly_same", target)
                    lines.append(f"{a['id']}: ambiguous continuation ({count} candidates); old clicks stop")
        # A remembered composite is matched as an exact multicolour template.
        # Multiple occurrences, including identical twins, remain ambiguous.
        if allow_match:
            for a in old.values():
                if not a["id"].startswith("G"):
                    continue
                parts = []
                for ref in a.get("parts", []):
                    while ref in self.successors:
                        ref = self.successors[ref]
                    if ref not in current:
                        parts = []
                        break
                    parts.append(current[ref])
                if parts:
                    b = measured(board, [p for part in parts for p in part["cells"]], "pending")
                    if shape(a, False) == shape(b, False):
                        if a["cells"] == b["cells"] and a["values"] == b["values"]:
                            current[a["id"]] = a
                        else:
                            b = self._new(board, b["cells"], "G", parts=[p["id"] for p in parts])
                            current[b["id"]] = b
                            relation = "moved" if a["values"] == b["values"] else "recoloured"
                            self._edge(a["id"], relation, b["id"], follow=True, evidence="all selected parts uniquely continued with the same relative geometry")
                            lines.append(f"{a['id']} -> {relation} -> {b['id']}")
                        continue
                    lines.append(f"{a['id']}: parts changed relative geometry; inspect/rebind")
                    continue
                matches = self._template_matches(a, board)
                if len(matches) == 1:
                    cells = matches[0]
                    if cells == a["cells"]:
                        current[a["id"]] = a
                    else:
                        b = self._new(board, cells, "G")
                        current[b["id"]] = b
                        self._edge(a["id"], "moved", b["id"], follow=True, evidence="unique exact composite template")
                        lines.append(f"{a['id']} -> moved -> {b['id']}")
                else:
                    lines.append(f"{a['id']}: {'ambiguous' if matches else 'unresolved'} composite; rebind after inspection")
        self.current, self.board, self.hash = current, board.copy(), key
        self.metrics["refreshes"] += 1
        self.last_report = {"lines": lines}

    def _indicator(self, e):
        r0, r1, c0, c1 = e["rect"]
        return any(a <= r0 and r1 <= b and c <= c0 and c1 <= d for a, b, c, d in self.indicator_rects)

    @staticmethod
    def _template_matches(e, board):
        offsets = np.asarray(shape(e, False))
        values = np.asarray(e["values"])
        counts = Counter(map(int, values))
        anchor = min(range(len(values)), key=lambda i: counts[int(values[i])])
        h, w = board.shape
        height, width = e["rect"][1]-e["rect"][0], e["rect"][3]-e["rect"][2]
        out = []
        for p in np.argwhere(board == values[anchor]):
            origin = p - offsets[anchor]
            if np.any(origin < 0) or origin[0]+height > h or origin[1]+width > w:
                continue
            pts = offsets + origin
            if np.array_equal(board[pts[:, 0], pts[:, 1]], values):
                out.append(pts.tolist())
                if len(out) == 2:
                    break
        return out

    def _root(self, ref):
        if not isinstance(ref, str):
            raise ValueError("reference must be a component ID or saved name")
        name = ref.lstrip("@")
        return self.aliases.get(name, {}).get("ref", name)

    def resolve(self, ref):
        root = self._root(ref)
        node = self.nodes.get(root)
        if not node or node["scope"] != self.scope:
            raise ValueError(f"{ref}: unknown or from an earlier level/attempt; inspect lib.components() and rebind")
        seen = set()
        while root in self.successors:
            if root in seen:
                raise ValueError("cyclic object reference")
            seen.add(root)
            root = self.successors[root]
        if root not in self.current:
            candidates = [e["target"] for e in self.edges if e["source"] == root and e["relation"] == "possibly_same"]
            detail = f"; candidates {', '.join(candidates[:6])}" if candidates else ""
            raise ValueError(f"{ref}: unresolved or ambiguous{detail}; inspect and explicitly rebind, no click executed")
        return copy.deepcopy(self.current[root])

    def maybe_resolve(self, ref):
        try:
            return self.resolve(ref)
        except ValueError:
            return None

    def remember(self, name, refs, reason=""):
        if not isinstance(name, str) or not checks.NAME.fullmatch(name) or re.fullmatch(r"[CG]\d+", name):
            raise ValueError("use a short name, not a reserved C/G ID")
        if name not in self.aliases and len(self.aliases) >= 64:
            raise ValueError("at most 64 saved names; lib.forget(name) frees one")
        refs = [refs] if isinstance(refs, str) else refs
        if not isinstance(refs, (list, tuple)) or not 1 <= len(refs) <= 32:
            raise ValueError("refs must be one ID/name or 1-32 references")
        parts = [self.resolve(r) for r in refs]
        if len(parts) == 1:
            ref = parts[0]["id"]
        else:
            if sum(e["id"].startswith("G") for e in self.current.values()) >= 64:
                raise ValueError("at most 64 live composite selections")
            e = self._new(self.board, [p for part in parts for p in part["cells"]], "G", parts=[p["id"] for p in parts])
            ref = e["id"]
            self.current[ref] = e
            for part in parts:
                self._edge(part["id"], "part_of", ref, provenance="agent", evidence=str(reason))
        self.aliases[name] = {"ref": ref, "reason": str(reason)[:480], "scope": self.scope}
        self.metrics["bindings"] += 1
        return self.resolve(name)

    def forget(self, name):
        return self.aliases.pop(str(name).lstrip("@"), None) is not None

    def references(self):
        out = {}
        for name, binding in self.aliases.items():
            current = self.maybe_resolve(name)
            out[name] = {**copy.deepcopy(binding), "current": current["id"] if current else None}
        return out

    def relate(self, source, relation, target, evidence="", follow=False):
        if type(follow) is not bool:
            raise ValueError("follow must be true or false")
        source, target = self._root(source), self._root(target)
        if relation not in {"same_object", "recoloured", "same_shape", "part_of", "touching", "inside", "left_of", "possibly_same"}:
            raise ValueError("unknown relationship")
        if any(k not in self.nodes or self.nodes[k]["scope"] != self.scope for k in (source, target)):
            raise ValueError("relationships must refer to IDs in this level/attempt")
        if follow:
            if relation not in {"same_object", "recoloured"} or not str(evidence).strip():
                raise ValueError("follow=True needs an identity relationship and explicit evidence")
            while source in self.successors:
                source = self.successors[source]
            dest = self.resolve(target)["id"]
            if source in self.current or source in self.successors or dest == source:
                raise ValueError("only an unresolved old ID can be linked to a current target; rebind a name to correct it")
            target = dest
        return self._edge(source, relation, target, provenance="agent_assertion", follow=bool(follow), evidence=str(evidence))

    def relations(self, ref, relation=None):
        root = self._root(ref)
        if root not in self.nodes:
            raise ValueError("unknown reference")
        found = [copy.deepcopy(e) for e in self.edges if e["source"] == root or e["target"] == root]
        a = self.maybe_resolve(ref)
        if a:
            aset = set(map(tuple, a["cells"]))
            border = {(r+dr, c+dc) for r, c in aset for dr, dc in ((0,1),(0,-1),(1,0),(-1,0))} - aset
            for b in self.current.values():
                if a["id"] == b["id"]:
                    continue
                bset = set(map(tuple, b["cells"]))
                rels = []
                if border & bset:
                    rels.append("touching")
                if aset < bset:
                    rels.append("part_of")
                if a["rect"][3] <= b["rect"][2]:
                    rels.append("left_of")
                ar, br = a["rect"], b["rect"]
                if ar != br and br[0] <= ar[0] and ar[1] <= br[1] and br[2] <= ar[2] and ar[3] <= br[3]:
                    rels.append("inside_bounds")
                for rel in rels:
                    found.append({"source": a["id"], "relation": rel, "target": b["id"], "provenance": "measured",
                                  "follow": False, "scope": self.scope, "step": self.step})
        return [e for e in found if relation is None or e["relation"] == relation]

    def components(self, colors=None, size=None, detail=False):
        """Filter measured components by exact colour counts and/or size; no fuzzy click guessing."""
        wanted = sorted([[int(k), int(v)] for k, v in colors.items()]) if colors is not None else None
        selected = [e for e in self.current.values() if e["id"].startswith("C")
                    and (size is None or e["size"] == size) and (wanted is None or e["colors"] == wanted)]
        return copy.deepcopy(selected if detail else [{k:v for k,v in e.items() if k not in {"cells", "values"}} for e in selected])

    def request(self, method, args, kwargs):
        methods = {"components": self.components, "resolve": self.resolve, "remember": self.remember,
                   "forget": self.forget, "references": self.references, "relate": self.relate, "relations": self.relations}
        if method not in methods:
            raise ValueError("unknown reference operation")
        return copy.deepcopy(methods[method](*args, **kwargs))

    def resolve_click(self, ref, color=None):
        e = self.resolve(ref)
        cells = e["cells"] if color is None else [p for p, v in zip(e["cells"], e["values"]) if v == color]
        if not cells:
            raise ValueError(f"{ref} has no selected cells of colour {color}")
        centre = np.mean(cells, axis=0)
        r, c = min(cells, key=lambda p: ((p[0]-centre[0])**2 + (p[1]-centre[1])**2, p))
        return r, c, int(self.board[r, c])

    def before_action(self, board, level, step):
        self.configure(level, self.attempt, step)
        self.refresh(board)
        refs = set(self.aliases) | {q["entity"] for q in self.expected}
        return {"scope": self.scope, "edge_start": len(self.edges), "resolved": {r: self.maybe_resolve(r) for r in refs}}

    def after_action(self, before, board_before, board_after, action, level, step, click=None, reliable=True, snapshot=True):
        self.configure(level, self.attempt, step)
        self.refresh(board_after, allow_match=reliable)
        for edge in self.edges[before["edge_start"]:]:
            edge["action"] = action
        after = {r: self.maybe_resolve(r) for r in before["resolved"]}
        self.last_checks = checks.check(self.expected, before["resolved"], after, before["scope"] == self.scope)
        self.expected = []
        if click is not None:
            for r, e in before["resolved"].items():
                if e and list(click) in e["cells"]:
                    self.clicks.add(self._root(r))
        return {"snapshot": self.payload() if snapshot else None, "report": copy.deepcopy(self.last_report),
                "edges": copy.deepcopy(self.edges[before["edge_start"]:]),
                "expectations": copy.deepcopy(self.last_checks), "action": action}

    def stop_reason(self):
        bad = next((q for q in self.last_checks if q["status"] != "satisfied"), None)
        return f"expectation {bad['status']} for {bad['entity']}; remaining batch not executed" if bad else None

    def action_text(self):
        lines = self.last_report.get("lines", [])[:8]
        lines += [f"Expectation {q['entity']} {q['test']}: {q['status']}" for q in self.last_checks]
        return "\n".join("  " + line for line in lines)

    def text(self, full=False):
        rows = [f"[Component references: level {self.level}, attempt {self.attempt}; IDs can be clicked or saved]"]
        for name in list(self.aliases)[:16]:
            e = self.maybe_resolve(name)
            rows.append(f"@{name} -> {e['id']} rect={e['rect']}" if e else f"@{name}: unresolved; inspect/rebind")
        if len(self.aliases) > 16:
            rows.append("... more saved names; use lib.references()")
        visible = [e for e in self.current.values() if e["id"].startswith("C") and not e.get("is_background") and not self._indicator(e)
                   and (full or self.nodes[e["id"]]["step"] == self.step)]
        for e in visible[:12]:
            colors = ", ".join(f"{n} {self.names.get(c, str(c))}" for c, n in e["colors"])
            rows.append(f"{e['id']}: {colors}; rect={e['rect']}")
        if len(visible) > 12:
            rows.append(f"... {len(visible)-12} more; lib.components() lists/filters all IDs")
        if not full and len(rows) == 1:
            return ""
        return "\n".join(rows)

    def payload(self):
        return {"game": self.game, "level": self.level, "attempt": self.attempt, "step": self.step,
                "board_hash": self.hash, "shape": list(self.board.shape) if self.board is not None else [],
                "components": copy.deepcopy(list(self.current.values())),
                "aliases": copy.deepcopy(self.aliases)}

    def log(self):
        return {**self.payload(), "edges": copy.deepcopy(self.edges[self.turn_edge_start:]),
                "changes": copy.deepcopy(self.last_report), "expectations": copy.deepcopy(self.last_checks),
                "metrics": dict(self.metrics)}

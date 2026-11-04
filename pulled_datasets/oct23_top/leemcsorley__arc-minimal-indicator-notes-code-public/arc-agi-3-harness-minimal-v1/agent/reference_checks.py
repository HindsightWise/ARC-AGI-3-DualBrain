"""Small, shared schema for optional checks on object-reference actions."""
from __future__ import annotations

import re

NAME = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,63}\Z")
EXPECT_SCHEMA = {"type": "array", "maxItems": 8, "items": {
    "type": "object", "properties": {
        "entity": {"type": "string"},
        "test": {"type": "string", "enum": ["resolved", "unresolved", "unchanged", "moved", "color"]},
        "delta": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
        "value": {"type": "integer", "minimum": 0, "maximum": 15},
    }, "required": ["entity", "test"], "additionalProperties": False}}


def expectations(raw):
    if raw is None:
        return []
    if not isinstance(raw, list) or len(raw) > 8:
        raise ValueError("expect must be a list of at most 8 predicates")
    out = []
    for q in raw:
        if not isinstance(q, dict) or set(q) - {"entity", "test", "delta", "value"}:
            raise ValueError("expectation fields: entity, test, delta, value")
        name, test = q.get("entity"), q.get("test")
        if not isinstance(name, str) or not NAME.fullmatch(name.lstrip("@")):
            raise ValueError("expectation needs a component ID or saved name")
        if test not in {"resolved", "unresolved", "unchanged", "moved", "color"}:
            raise ValueError("unknown expectation test")
        p = {"entity": name.lstrip("@"), "test": test}
        if test == "moved":
            d = q.get("delta")
            if not isinstance(d, (list, tuple)) or len(d) != 2 or any(type(v) is not int for v in d):
                raise ValueError("moved needs integer delta=[row_offset,col_offset]")
            p["delta"] = list(d)
        if test == "color":
            v = q.get("value")
            if type(v) is not int or not 0 <= v <= 15:
                raise ValueError("color needs value=0..15")
            p["value"] = v
        out.append(p)
    return out


def translation(a, b):
    if not a or not b:
        return None
    dr, dc = b["rect"][0] - a["rect"][0], b["rect"][2] - a["rect"][2]
    return [dr, dc] if [[r + dr, c + dc] for r, c in a["cells"]] == b["cells"] else None


def check(checks, before, after, same_scope=True):
    results = []
    for q in checks:
        a, b, ok = before.get(q["entity"]), after.get(q["entity"]), None
        if same_scope:
            if q["test"] == "resolved":
                ok = b is not None
            elif q["test"] == "unresolved":
                ok = b is None
            elif q["test"] == "color" and b:
                ok = all(v == q["value"] for v in b["values"])
            elif a and b:
                if q["test"] == "unchanged":
                    ok = a["cells"] == b["cells"] and a["values"] == b["values"]
                elif q["test"] == "moved":
                    ok = translation(a, b) == q["delta"]
        results.append({**q, "status": "unknown" if ok is None else "satisfied" if ok else "contradicted"})
    return results

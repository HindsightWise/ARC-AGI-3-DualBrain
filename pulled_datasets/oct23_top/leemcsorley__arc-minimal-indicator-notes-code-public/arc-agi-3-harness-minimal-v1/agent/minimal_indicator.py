"""Validation and compact, explicitly conditional readouts for model-marked bars."""
from numbers import Integral


def validate_rects(value, shape):
    if not isinstance(value, (list, tuple)):
        raise ValueError("use (r0, r1, c0, c1), a list of those rectangles, or [] to clear")
    if len(value) == 4 and all(isinstance(v, Integral) for v in value):
        value = [value]
    if len(value) > 8:
        raise ValueError("at most 8 indicator rectangles")
    out = []
    h, w = shape
    for rect in value:
        if (not isinstance(rect, (list, tuple)) or len(rect) != 4
                or any(isinstance(v, bool) or not isinstance(v, Integral) for v in rect)):
            raise ValueError("each rectangle needs four integer half-open bounds: (r0, r1, c0, c1)")
        r0, r1, c0, c1 = map(int, rect)
        if not (0 <= r0 < r1 <= h and 0 <= c0 < c1 <= w):
            raise ValueError(f"rectangle {tuple(rect)} is outside board {tuple(shape)}")
        item = (r0, r1, c0, c1)
        if item in out:
            continue
        if any(max(r0, a) < min(r1, b) and max(c0, c) < min(c1, d) for a, b, c, d in out):
            raise ValueError("indicator rectangles must not overlap")
        out.append(item)
    return tuple(out)


def status_line(facts, names):
    if not facts:
        return ""
    parts = facts.get("parts") or [facts]
    lines = []
    for part in parts:
        now, start = part["counts"], part["counts_at_start"]
        values = [f"{names.get(c, c)}({c}) {now.get(c, 0)}/{start.get(c, 0)}"
                  for c in sorted(set(now) | set(start))]
        label = f"{tuple(part['rect'])}: " if len(parts) > 1 else ""
        lines.append(label + ", ".join(values) + "; " + part["if_this_continues"])
    return "[Your marked resource/counter region; cells now/attempt start] " + " | ".join(lines)

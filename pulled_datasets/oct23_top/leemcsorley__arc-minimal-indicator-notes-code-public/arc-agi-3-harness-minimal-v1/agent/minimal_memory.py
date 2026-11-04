"""Serializable notes, change tracking and deterministic images for the minimal harness.

Standalone NumPy/stdlib module: the exact same source runs in the sandbox. Image
values use ordinary dictionaries, so they survive IPC without unpickling classes.
"""
from __future__ import annotations

import base64
import copy
import json
import pickle
import struct
import zlib

import numpy as np

IMAGE_KEY = "_arc_image_v1"
MAX_IMAGES = 8
NOTES_CHARS = 20000
OUTPUT_CHARS = 16000
MEMORY_BYTES = 64 * 1024 * 1024
_PALETTE = None


def bind(palette):
    global _PALETTE
    _PALETTE = np.asarray([
        [int(str(palette[i]).lstrip("#")[j:j + 2], 16) for j in (0, 2, 4)]
        for i in range(16)
    ], dtype=np.uint8)


def is_image(value):
    return isinstance(value, dict) and value.get(IMAGE_KEY) == 1


def _png(rgb):
    h, w = rgb.shape[:2]
    def chunk(tag, payload):
        return (struct.pack(">I", len(payload)) + tag + payload
                + struct.pack(">I", zlib.crc32(tag + payload) & 0xffffffff))
    raw = b"".join(b"\0" + row.tobytes() for row in rgb)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def render(value, *, rect=None, mask=None, title="", size=640):
    """Return an immutable-by-convention image snapshot of a grid or measured objects.

    rect=(row_start,row_end,col_start,col_end), half-open. Boolean masks keep the
    selected cells; omitted cells use an off-palette background. Canvas is square,
    nearest-neighbour scaled and centred without stretching the board.
    """
    size = int(size)
    if not 64 <= size <= 640:
        raise ValueError("render size must be between 64 and 640 pixels")
    origin = (0, 0)
    if isinstance(value, dict) and "cells" in value:
        value = [value]
    if isinstance(value, (list, tuple)) and value and all(isinstance(o, dict) and "cells" in o for o in value):
        cells = [tuple(map(int, p)) for o in value for p in o["cells"]]
        if not cells:
            raise ValueError("cannot render empty objects")
        r0, c0 = np.min(cells, axis=0)
        r1, c1 = np.max(cells, axis=0) + 1
        if (r1 - r0) * (c1 - c0) > 65536:
            raise ValueError("object image bounds are too large")
        board = np.full((r1-r0, c1-c0), -1, dtype=np.int16)
        for o in value:
            colors = o.get("colors", [o.get("color", 0)] * len(o["cells"]))
            for (r, c), color in zip(o["cells"], colors):
                board[int(r)-r0, int(c)-c0] = int(color)
        origin = (int(r0), int(c0))
    else:
        board = np.asarray(value)
        if board.ndim != 2 or not board.size or board.size > 65536:
            raise ValueError("render needs a nonempty 2D grid (at most 65536 cells) or component(s)")
        if board.dtype.kind not in "biu" or np.any((board < 0) | (board > 15)):
            raise ValueError("grid values must be integer palette colours 0..15")
        board = board.astype(np.int16, copy=True)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape != board.shape:
            raise ValueError("mask must have the same shape as the uncropped grid")
        board[~m] = -1
    if rect is not None:
        r0, r1, c0, c1 = map(int, rect)
        if not (0 <= r0 < r1 <= board.shape[0] and 0 <= c0 < c1 <= board.shape[1]):
            raise ValueError("rect must be valid half-open bounds within the input")
        board = board[r0:r1, c0:c1].copy()
        origin = (origin[0] + r0, origin[1] + c0)
    if _PALETTE is None:
        raise ValueError("render palette has not been initialized")
    rgb = _PALETTE[np.maximum(board, 0)]
    rgb[board < 0] = (24, 24, 24)
    h, w = board.shape
    ratio = size / max(h, w)
    oh, ow = max(1, int(h * ratio)), max(1, int(w * ratio))
    rows = np.minimum((np.arange(oh) * h // oh), h-1)
    cols = np.minimum((np.arange(ow) * w // ow), w-1)
    canvas = np.full((size, size, 3), 24, dtype=np.uint8)
    top, left = (size-oh)//2, (size-ow)//2
    canvas[top:top+oh, left:left+ow] = rgb[rows[:, None], cols]
    label = (f"{str(title)[:200] or 'Grid snapshot'}; source rows "
             f"[{origin[0]}:{origin[0]+h}], cols [{origin[1]}:{origin[1]+w}]; "
             f"{h}x{w} cells, {size}x{size} image")
    return {IMAGE_KEY: 1, "png": _png(canvas), "label": label, "size": size,
            "origin": origin, "shape": (h, w)}


def image_url(value):
    return "data:image/png;base64," + base64.b64encode(value["png"]).decode("ascii")


def validate(value, *, _seen=None):
    """Allow only bounded serializable data, and reject cycles/opaque objects."""
    if _seen is None:
        _seen = set()
    if value is None or isinstance(value, (str, bool, int, float, np.number, np.bool_)):
        return
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biufUS":
            raise ValueError("memory arrays must contain numbers, booleans or strings")
        return
    if is_image(value):
        if (not isinstance(value.get("png"), bytes) or len(value["png"]) > 2 * 1024 * 1024
                or not value["png"].startswith(b"\x89PNG\r\n\x1a\n")):
            raise ValueError("invalid or oversized image; use render()")
        return
    if not isinstance(value, (dict, list, tuple, set, frozenset)):
        raise ValueError(f"unsupported memory value {type(value).__name__}; save data, not functions/live resources")
    if id(value) in _seen:
        raise ValueError("cyclic memory values are not supported")
    _seen.add(id(value))
    try:
        if isinstance(value, dict):
            for key, item in value.items():
                if not isinstance(key, (str, int, float, bool, tuple)):
                    raise ValueError("memory keys must be strings, numbers or tuples")
                validate(key, _seen=_seen)
                validate(item, _seen=_seen)
        else:
            for item in value:
                validate(item, _seen=_seen)
    finally:
        _seen.remove(id(value))


def delta(before, after, path="notes"):
    """Changed paths only; append-only lists return new elements, arrays changed cells."""
    if type(before) is not type(after):
        return [(path, after)]
    if is_image(after):
        return [] if before == after else [(path, after)]
    if isinstance(after, dict):
        out = []
        for key in before.keys() - after.keys():
            out.append((f"{path}[{key!r}]", {"deleted": True}))
        for key, value in after.items():
            p = f"{path}[{key!r}]"
            out.extend(delta(before[key], value, p) if key in before else [(p, value)])
        return out
    if isinstance(after, (tuple, list)):
        out = []
        for i in range(min(len(before), len(after))):
            out.extend(delta(before[i], after[i], f"{path}[{i}]"))
        if len(after) > len(before):
            out.append((f"{path}[{len(before)}:] (appended)", after[len(before):]))
        elif len(after) < len(before):
            out.append((f"{path}[{len(after)}:] (removed)", {"deleted": True}))
        return out
    if isinstance(after, np.ndarray):
        if before.shape != after.shape or before.dtype != after.dtype:
            return [(path, after)]
        changed = before != after
        if after.dtype.kind == "f":
            changed &= ~(np.isnan(before) & np.isnan(after))
        count = int(np.count_nonzero(changed))
        flat = changed.reshape(-1)
        indices = []
        for start in range(0, flat.size, 65536):
            indices.extend((np.flatnonzero(flat[start:start+65536]) + start)[:1024-len(indices)].tolist())
            if len(indices) >= 1024:
                break
        out = []
        for i in indices:
            idx = np.unravel_index(i, after.shape)
            out.append((f"{path}[{','.join(map(str, idx))}]", after[idx].item()))
        if count > len(indices):
            out.append((path + " (delta limit)", {"omitted_changed_cells": count-len(indices),
                                                  "inspect": "request smaller slices in run()"}))
        return out
    return [] if before == after else [(path, after)]


def present(entries, *, max_chars=OUTPUT_CHARS, max_images=MAX_IMAGES):
    """Render text plus ordered images, with an explicit overflow marker."""
    images = []
    def plain(value, path):
        if is_image(value):
            if len(images) >= max_images:
                return {"image_omitted": path, "reason": f"image limit {max_images}; value remains in Python memory"}
            images.append((path, value))
            return {"image": path, "label": value["label"]}
        if isinstance(value, np.ndarray):
            return {"shape": list(value.shape), "dtype": str(value.dtype), "values": value.tolist()}
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(k): plain(v, f"{path}[{k!r}]") for k, v in value.items()}
        if isinstance(value, (list, tuple, set, frozenset)):
            return [plain(v, f"{path}[{i}]") for i, v in enumerate(value)]
        return value
    lines, length, overflow = [], 0, False
    for path, value in entries:
        old_images = len(images)
        text = path + " = " + json.dumps(plain(value, path), ensure_ascii=False, allow_nan=True)
        available = max_chars - length
        if len(text) > available:
            images[old_images:] = []  # never send images whose labels are omitted
            lines.append(text[:max(0, available)] + "\n[Output limit reached; remaining values persist. Inspect smaller slices in run().]")
            overflow = True
            break
        lines.append(text)
        length += len(text) + 1
    return "\n".join(lines), images, overflow


def commit(previous, candidate, kind):
    """Commit valid top-level entries independently; report rejected writes."""
    if not isinstance(candidate, dict):
        return previous, [f"{kind} must remain a dictionary; changes were not saved"]
    saved = copy.deepcopy(previous)
    messages = []
    for key in previous.keys() - candidate.keys():
        del saved[key]
    for key, value in candidate.items():
        try:
            validate({key: value})
            trial = dict(saved)
            trial[key] = value
            if len(pickle.dumps(trial, protocol=4)) > MEMORY_BYTES:
                raise ValueError("64 MB memory limit exceeded")
            if kind == "notes":
                text, imgs, overflow = present([("notes", trial)], max_chars=NOTES_CHARS, max_images=MAX_IMAGES)
                if overflow or '"image_omitted"' in text:
                    raise ValueError(f"notes limit: {NOTES_CHARS} text characters and {MAX_IMAGES} images")
            saved[key] = value
        except (Exception, RecursionError) as exc:
            messages.append(f"{kind}[{key!r}] not saved: {exc}")
    return saved, messages

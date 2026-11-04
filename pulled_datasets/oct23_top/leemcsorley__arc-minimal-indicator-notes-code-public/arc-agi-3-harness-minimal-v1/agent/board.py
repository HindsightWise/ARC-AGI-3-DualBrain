from __future__ import annotations

import base64
import struct
from typing import Any

import numpy as np
import zlib

from .config import BOARD_COLOR_MAP_HEX, BOARD_MAX_SHAPE, BOARD_SHAPE


def _to_uint8_grid(
    board: np.ndarray,
    *,
    expected_shape: tuple[int, int] | None,
    max_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    arr = np.asarray(board, dtype=np.uint8)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D board array, got ndim={arr.ndim}")
    if expected_shape is not None and arr.shape != expected_shape:
        raise ValueError(f"Expected board shape {expected_shape}, got {arr.shape}")
    if max_shape is not None:
        if arr.shape[0] > max_shape[0] or arr.shape[1] > max_shape[1]:
            raise ValueError(f"Board shape {arr.shape} exceeds maximum {max_shape}")
    return np.ascontiguousarray(arr)


def to_uint8_board(board: np.ndarray) -> np.ndarray:
    return _to_uint8_grid(board, expected_shape=None, max_shape=BOARD_MAX_SHAPE)


def board_key(board: np.ndarray) -> bytes:
    return to_uint8_board(board).tobytes()


def board_from_key(key: bytes, shape: tuple[int, int] = BOARD_SHAPE) -> np.ndarray:
    return np.frombuffer(key, dtype=np.uint8).copy().reshape(shape)


def board_to_csv(board: np.ndarray) -> str:
    arr = to_uint8_board(board)
    return "\n".join(",".join(str(int(v)) for v in row) for row in arr)


def _hex_rgba_to_rgb(hex_rgba: str) -> tuple[int, int, int]:
    clean = hex_rgba.strip().lstrip("#")
    if len(clean) < 6:
        return (255, 255, 255)
    return (int(clean[0:2], 16), int(clean[2:4], 16), int(clean[4:6], 16))


def _png_chunk(tag: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + tag
        + payload
        + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
    )


def board_to_png_bytes(board: np.ndarray, *, scale: int = 1) -> bytes:
    arr = _to_uint8_grid(board, expected_shape=None)
    if scale < 1:
        raise ValueError(f"scale must be >= 1, got {scale}")

    if scale > 1:
        arr = np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)

    palette = np.zeros((256, 3), dtype=np.uint8)
    for color_id, color_hex in BOARD_COLOR_MAP_HEX.items():
        if 0 <= int(color_id) < 256:
            palette[int(color_id)] = np.asarray(_hex_rgba_to_rgb(color_hex), dtype=np.uint8)

    rgb = palette[arr]
    height, width = rgb.shape[:2]

    raw = bytearray()
    for row in rgb:
        raw.append(0)  # PNG filter type 0 (None)
        raw.extend(row.tobytes())

    header = b"\x89PNG\r\n\x1a\n"
    ihdr = _png_chunk(
        b"IHDR",
        struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0),  # RGB, non-interlaced
    )
    idat = _png_chunk(b"IDAT", zlib.compress(bytes(raw), level=6))
    iend = _png_chunk(b"IEND", b"")
    return header + ihdr + idat + iend


def board_to_png_data_url(board: np.ndarray, *, scale: int = 1) -> str:
    png_bytes = board_to_png_bytes(board, scale=scale)
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{b64}"


def board_with_striped_padding(
    board: np.ndarray,
    *,
    target_shape: tuple[int, int],
) -> np.ndarray:
    """Center a 64x64 board in a larger black/white diagonal striped canvas."""
    arr = to_uint8_board(board)
    target_rows, target_cols = target_shape
    rows, cols = arr.shape
    if target_rows < rows or target_cols < cols:
        raise ValueError(
            f"target_shape {target_shape} must be >= board shape {arr.shape}"
        )

    top = (target_rows - rows) // 2
    left = (target_cols - cols) // 2
    bottom = top + rows
    right = left + cols

    rr = np.arange(target_rows, dtype=np.int32)[:, None]
    cc = np.arange(target_cols, dtype=np.int32)[None, :]
    striped = np.where(((rr + cc) % 2) == 0, 0, 5).astype(np.uint8)
    striped[top:bottom, left:right] = arr
    return striped


def board_to_aligned_csv(board: np.ndarray) -> str:
    arr = to_uint8_board(board)
    return "\n".join(" ".join(f"{int(v):2d}" for v in row) for row in arr)


def compute_changed_rects(
    before: np.ndarray, after: np.ndarray
) -> list[tuple[int, int, int, int]]:
    """Compute bounding rectangles of changed regions.

    Returns list of (row_start, row_end, col_start, col_end) using half-open
    intervals suitable for NumPy slicing: board[rs:re, cs:ce].

    Algorithm: find changed cells, merge adjacent ones into rectangles by
    iteratively expanding each rectangle to include any changed cell within
    1 cell of its boundary.
    """
    b = to_uint8_board(before)
    a = to_uint8_board(after)
    diff_mask = b != a
    changed_coords = set(map(tuple, np.argwhere(diff_mask).tolist()))

    if not changed_coords:
        return []

    rects: list[tuple[int, int, int, int]] = []
    remaining = set(changed_coords)

    while remaining:
        seed = remaining.pop()
        rs, cs = seed
        re, ce = rs + 1, cs + 1

        expanded = True
        while expanded:
            expanded = False
            to_add = set()
            for coord in remaining:
                r, c = coord
                if (rs - 1) <= r < (re + 1) and (cs - 1) <= c < (ce + 1):
                    to_add.add(coord)
            if to_add:
                remaining -= to_add
                for r, c in to_add:
                    rs = min(rs, r)
                    re = max(re, r + 1)
                    cs = min(cs, c)
                    ce = max(ce, c + 1)
                expanded = True

        rects.append((rs, re, cs, ce))

    rects.sort()
    return rects


def format_rect_region(board: np.ndarray, rs: int, re: int, cs: int, ce: int) -> str:
    """Format a rectangular region as space-separated aligned values."""
    arr = to_uint8_board(board)
    region = arr[rs:re, cs:ce]
    max_val = int(region.max()) if region.size > 0 else 0
    width = len(str(max_val))
    lines: list[str] = []
    for row in region:
        lines.append(" ".join(f"{int(v):{width}d}" for v in row))
    return "\n".join(lines)


def animation_metadata_from_frames(
    before_board: np.ndarray, frames: list[Any]
) -> tuple[int, str]:
    total_frames = len(frames)
    animation_frame_count = max(0, total_frames - 1)
    if animation_frame_count == 0:
        return 0, ""

    before = to_uint8_board(before_board)
    changed_mask = np.zeros(before.shape[:2], dtype=bool)
    for frame in frames[:-1]:
        frame_arr = to_uint8_board(np.asarray(frame))
        # Handle shape mismatch (e.g. board resized between frames)
        if frame_arr.shape == before.shape:
            changed_mask |= frame_arr != before
        else:
            changed_mask[:] = True
            break

    coords = np.argwhere(changed_mask)
    if coords.shape[0] == 0:
        return animation_frame_count, "none"

    r_min, c_min = coords.min(axis=0)
    r_max, c_max = coords.max(axis=0)
    bbox = f"[{int(r_min)}:{int(r_max)+1}, {int(c_min)}:{int(c_max)+1}]"
    return animation_frame_count, bbox


# ---------------------------------------------------------------- remaining-actions indicator rectangles (v2.5)
# The indicator is a tuple of 0..n rectangles [r0, r1, c0, c1) (usually one; occasionally two bars). `None`
# and `()` both mean "not marked". Older code paths may still pass a single 4-tuple; norm_rects accepts both.

def norm_rects(rects: Any) -> tuple[tuple[int, int, int, int], ...]:
    if rects is None:
        return ()
    rects = list(rects)
    if not rects:
        return ()
    if len(rects) == 4 and all(isinstance(x, (int, np.integer)) for x in rects):
        return (tuple(int(x) for x in rects),)  # a single rect
    return tuple(tuple(int(x) for x in r) for r in rects)


def rects_mask(rects: Any, shape: tuple[int, ...]) -> np.ndarray:
    """Boolean mask of the marked indicator cells (all False when none is marked)."""
    m = np.zeros(tuple(shape[:2]), dtype=bool)
    for r0, r1, c0, c1 in norm_rects(rects):
        m[r0:r1, c0:c1] = True
    return m


def rects_text(rects: Any) -> str:
    rs = norm_rects(rects)
    if not rs:
        return "not set"
    return ", ".join(f"[{r0}:{r1}, {c0}:{c1}]" for r0, r1, c0, c1 in rs)

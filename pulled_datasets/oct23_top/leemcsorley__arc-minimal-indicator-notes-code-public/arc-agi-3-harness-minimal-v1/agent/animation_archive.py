"""Bounded, parent-owned intermediate frames, addressed by current-level history index."""
from collections import OrderedDict

import numpy as np

from .board import to_uint8_board


class AnimationArchive:
    # Raw bytes avoid compression work on the gameplay path. Include conservative
    # record/frame overhead in the budget and bound metadata for empty animations.
    MAX_BYTES = 32 * 1024 * 1024
    MAX_ENTRIES = 1024

    def __init__(self, max_bytes=None, max_entries=None):
        self.max_bytes = self.MAX_BYTES if max_bytes is None else max_bytes
        self.max_entries = self.MAX_ENTRIES if max_entries is None else max_entries
        if self.max_bytes < 256 or self.max_entries < 1:
            raise ValueError("animation archive requires at least 256 bytes and one entry")
        self.clear()

    def clear(self):
        self.records = OrderedDict()
        self.size_bytes = 0
        self.last_index = -1

    def record(self, index, frames):
        """Keep the whole intermediate sequence or an explicit unavailable marker."""
        saved, reason, size = [], None, 256
        try:
            for frame in frames:
                board = to_uint8_board(np.asarray(frame))
                size += board.nbytes + 256
                if size > self.max_bytes:
                    reason = "animation exceeded the archive memory limit; no partial sequence retained"
                    break
                saved.append((board.tobytes(), tuple(board.shape)))
        except Exception as exc:
            reason = f"animation could not be retained ({type(exc).__name__})"
        if reason:
            saved, size = [], 256
        previous = self.records.pop(index, None)
        if previous is not None:
            self.size_bytes -= previous[2]
        while self.records and (self.size_bytes + size > self.max_bytes
                                or len(self.records) >= self.max_entries):
            _, (_, _, removed_size) = self.records.popitem(last=False)
            self.size_bytes -= removed_size
        self.records[index] = (saved, reason, size)
        self.size_bytes += size
        self.last_index = max(self.last_index, index)

    def get(self, index):
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise TypeError("animation index must be an integer")
        if not 0 <= index <= self.last_index:
            raise LookupError("animation unavailable: no recorded transition at this index in the current level")
        record = self.records.get(index)
        if record is None:
            raise LookupError(f"animation for history[{index}] was evicted by the archive limit; "
                              "before/after boards remain in history")
        frames, reason, _ = record
        if reason:
            raise LookupError(f"animation for history[{index}] unavailable: {reason}")
        return list(frames)

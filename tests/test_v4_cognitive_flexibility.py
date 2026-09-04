"""Prototype and verification test for Version 4 Cognitive Flexibility modules."""
import numpy as np
from typing import Optional, Tuple, Dict, Any, List


class DynamicLatticeDetector:
    """Infers grid lattice pitch (step_size) and spatial offset (r_offset, c_offset) dynamically."""

    @staticmethod
    def infer_lattice(avatar_history: List[Tuple[int, int]], grid_shape: Tuple[int, int] = (64, 64)) -> Tuple[int, int, int, int, int]:
        """Returns (step_size, r_offset, c_offset, grid_h, grid_w)."""
        if len(avatar_history) < 2:
            return 1, 0, 0, grid_shape[0], grid_shape[1]

        diffs = []
        for i in range(1, len(avatar_history)):
            r0, c0 = avatar_history[i - 1]
            r1, c1 = avatar_history[i]
            dr = abs(r1 - r0)
            dc = abs(c1 - c0)
            if dr > 0 or dc > 0:
                diffs.append(max(dr, dc))

        if not diffs:
            return 1, 0, 0, grid_shape[0], grid_shape[1]

        step = int(round(float(np.median(diffs))))
        step = max(1, step)

        latest_r, latest_c = avatar_history[-1]
        r_offset = latest_r % step
        c_offset = latest_c % step

        grid_h = (grid_shape[0] - r_offset) // step
        grid_w = (grid_shape[1] - c_offset) // step
        return step, r_offset, c_offset, grid_h, grid_w


class DynamicHUDDetector:
    """Scans perimeter and border quadrants for static reference goal templates without hardcoded coordinates."""

    @staticmethod
    def find_goal_template(grid: np.ndarray, bg_color: int) -> Optional[Dict[str, Any]]:
        """Scans outer 14 pixels on all margins for high-contrast isolated frames."""
        H, W = grid.shape
        candidates = []

        margins = [
            (slice(0, 14), slice(0, W), "top"),
            (slice(H - 14, H), slice(0, W), "bottom"),
            (slice(0, H), slice(0, 14), "left"),
            (slice(0, H), slice(W - 14, W), "right"),
            (slice(H - 14, H), slice(0, 14), "bottom_left"),
            (slice(0, 14), slice(W - 14, W), "top_right"),
            (slice(0, 14), slice(0, 14), "top_left"),
            (slice(H - 14, H), slice(W - 14, W), "bottom_right"),
        ]

        for r_sl, c_sl, loc in margins:
            sub = grid[r_sl, c_sl]
            sub_h, sub_w = sub.shape
            for win_size in (5, 6, 7, 8):
                if sub_h < win_size or sub_w < win_size:
                    continue
                for wr in range(sub_h - win_size + 1):
                    for wc in range(sub_w - win_size + 1):
                        box = sub[wr:wr + win_size, wc:wc + win_size]
                        border_pixels = np.concatenate([box[0, :], box[-1, :], box[:, 0], box[:, -1]])
                        border_color = int(border_pixels[0])
                        if (border_pixels == border_color).all() and border_color != bg_color:
                            core = box[1:-1, 1:-1]
                            non_border_mask = (core != border_color) & (core != bg_color)
                            if non_border_mask.any():
                                fg_colors, counts = np.unique(core[non_border_mask], return_counts=True)
                                candidates.append({
                                    "location": loc,
                                    "r": r_sl.start + wr,
                                    "c": c_sl.start + wc,
                                    "size": win_size,
                                    "border_color": border_color,
                                    "fg_color": int(fg_colors[np.argmax(counts)]),
                                    "core_patch": core,
                                    "box_patch": box,
                                })

        if not candidates:
            return None

        candidates.sort(key=lambda c: (c["size"], -c["core_patch"].size))
        return candidates[0]


def test_dynamic_hud_on_synthetic():
    grid = np.zeros((64, 64), dtype=np.int8)
    grid[2:8, 56:62] = 5  # Border color 5
    grid[3:7, 57:61] = 0  # Interior bg
    grid[4:6, 58:60] = 9  # Foreground target 9

    template = DynamicHUDDetector.find_goal_template(grid, bg_color=0)
    assert template is not None, "Failed to find top-right template!"
    assert template["border_color"] == 5
    assert template["fg_color"] == 9
    print("[PASS] Synthetic top-right template correctly detected:", template["location"], template["r"], template["c"])


def test_dynamic_lattice():
    history = [(0, 4), (0, 9), (5, 9), (10, 9)]
    step, roff, coff, gh, gw = DynamicLatticeDetector.infer_lattice(history)
    assert step == 5, f"Expected step 5, got {step}"
    assert roff == 0, f"Expected roff 0, got {roff}"
    assert coff == 4, f"Expected coff 4, got {coff}"
    print(f"[PASS] Lattice inferred: step={step}, r_off={roff}, c_off={coff}, grid={gh}x{gw}")


if __name__ == "__main__":
    test_dynamic_hud_on_synthetic()
    test_dynamic_lattice()
    print("All prototype tests passed successfully!")

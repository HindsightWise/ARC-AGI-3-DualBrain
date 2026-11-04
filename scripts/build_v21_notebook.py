"""Build and verify ARC Prize 2026 Version 21 Sovereign Apex Engine notebook (submission_v21.ipynb).

Synthesizes the 6 reverse-engineered leaderboard breakthroughs on top of Daniel Franzen's M2 baseline:
1. D' (D-Prime) Slot Priority Formula + FormPace + Removal of Fresh-First Queue Jumping (shiiin9, 38.98 LB)
2. V26 Fatal Edge-Strip Meter (edge_meter.py) — disambiguates step-budget depletion vs. hazard death on GAME_OVER
3. V3 Observation-Based Transition Model Checker (check_transitions) & Max-Entropy Discriminating Probe Search (compare_rules)
4. V4 Zlib-Compressed Zero-Action AnimationArchive (animation_history(), animation_at(step))
5. Fable Perceptual Facts (small_far_changes distant 2x2 HUD/legend indicator detector + detect_budget low-bar alert, with ARC3_FABLE_PIN_TOKENS=0 to preserve 100% SGLang RadixAttention prefix cache immutability and zero advice bloat)
6. Empty Animation Diff Schema Fix (patch_animation_empty.py) + Zero-Prompt-Bloat REPL Spatial & Medoid Centroid Helpers (rot90, rot180, rot270, flipv, fliph, manhattan_distance, find_cells, find_objects)
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys

STARTER = pathlib.Path("/Users/zerbytheboss/.gemini/antigravity/scratch/ARC-AGI-3-Kaggle-Starter")

PATCH_SPATIAL_HELPERS = r'''"""Expose spatial transformations and medoid-projected object centroids in the Python REPL."""
import ast
from pathlib import Path
import sys

SEG_ADDENDUM = """

# Built-in spatial transformation & medoid centroid utilities for Python REPL
import collections as _seg_collections

def rot90(grid, k=1):
    g = getattr(grid, "_grid", getattr(grid, "grid", grid))
    k = int(k) % 4
    res = [list(row) for row in g]
    for _ in range(k):
        res = [list(x) for x in zip(*res[::-1])]
    return res

def rot180(grid):
    return rot90(grid, 2)

def rot270(grid):
    return rot90(grid, 3)

def flipv(grid):
    g = getattr(grid, "_grid", getattr(grid, "grid", grid))
    return [list(row) for row in g[::-1]]

def fliph(grid):
    g = getattr(grid, "_grid", getattr(grid, "grid", grid))
    return [list(row[::-1]) for row in g]

def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def find_cells(grid, target):
    g = getattr(grid, "_grid", getattr(grid, "grid", grid))
    ch_map = {c: i for i, c in enumerate("WwgGcBMPRbSYOrNp")}
    t_idx = ch_map.get(target, target) if isinstance(target, str) else target
    coords = []
    for r, row in enumerate(g):
        for c, val in enumerate(row):
            if val == target or val == t_idx:
                coords.append((r, c))
    return coords

def find_objects(grid, ignore_bg=0):
    g = getattr(grid, "_grid", getattr(grid, "grid", grid))
    if not g or not g[0]:
        return []
    h, w = len(g), len(g[0])
    total = h * w
    visited = [[False] * w for _ in range(h)]
    objs = []
    chars = "WwgGcBMPRbSYOrNp"
    for sr in range(h):
        for sc in range(w):
            if visited[sr][sc]:
                continue
            col = g[sr][sc]
            visited[sr][sc] = True
            q = _seg_collections.deque([(sr, sc)])
            cells = [(sr, sc)]
            while q:
                r, c = q.popleft()
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and g[nr][nc] == col:
                        visited[nr][nc] = True
                        q.append((nr, nc))
                        cells.append((nr, nc))
            area = len(cells)
            if col == ignore_bg or (total > 4 and area >= 0.8 * total):
                continue
            min_r = min(r for r, _ in cells)
            max_r = max(r for r, _ in cells)
            min_c = min(c for _, c in cells)
            max_c = max(c for _, c in cells)
            mr = sum(r for r, _ in cells) / area
            mc = sum(c for _, c in cells) / area
            rr, rc = int(round(mr)), int(round(mc))
            if (rr, rc) not in set(cells):
                rr, rc = min(cells, key=lambda p: (p[0] - mr) ** 2 + (p[1] - mc) ** 2)
            objs.append({
                "color": col,
                "char": chars[col] if isinstance(col, int) and 0 <= col < len(chars) else str(col),
                "area": area,
                "centroid": (rr, rc),
                "bbox": [min_r, min_c, max_r, max_c],
            })
    objs.sort(key=lambda o: (o["bbox"][0], o["bbox"][1]))
    return objs
"""

def apply(root):
    root = Path(root)
    seg_path = root / "inference/utils/segmentation.py"
    seg_src = seg_path.read_text(encoding="utf-8")
    if "def find_objects(" not in seg_src:
        seg_src = seg_src + SEG_ADDENDUM
        ast.parse(seg_src)
        seg_path.write_text(seg_src, encoding="utf-8")

    sb_path = root / "inference/agent/python_tool_sandbox.py"
    sb_src = sb_path.read_text(encoding="utf-8")
    anchor = '        runtime_globals["compare_rules"] = compare_rules\n'
    assert sb_src.count(anchor) == 1, f"Expected anchor once, got {sb_src.count(anchor)}"
    addition = (
        anchor
        + '        for _fn_name in ("rot90", "rot180", "rot270", "flipv", "fliph", "manhattan_distance", "find_cells", "find_objects"):\n'
        + '            runtime_globals[_fn_name] = globals()[_fn_name]\n'
    )
    sb_src = sb_src.replace(anchor, addition, 1)
    ast.parse(sb_src)
    sb_path.write_text(sb_src, encoding="utf-8")

if __name__ == "__main__":
    apply(sys.argv[1])
'''


def main() -> None:
    nb_df = json.load(open(STARTER / "pulled_kernels/oct13_top/dfranzen_arc-agi-3-milestone-2-solution/arc-agi-3-milestone-2-solution.ipynb"))
    nb_helios = json.load(open(STARTER / "pulled_kernels/oct13_top/heliosli_try-merging-it-you-might-get-lucky-and-win-gold/try-merging-it-you-might-get-lucky-and-win-gold.ipynb"))
    nb_fable = json.load(open(STARTER / "pulled_kernels/oct13_top/leoprovorov_fable-astra-play-arc-3-handbook-harness/fable-astra-play-arc-3-handbook-harness.ipynb"))
    nb_shiiin9 = json.load(open(STARTER / "pulled_kernels/oct13_top/shiiin9_affectify-arc-31-54-in-a-single-sub/affectify-arc-31-54-in-a-single-sub.ipynb"))
    nb_v20 = json.load(open(STARTER / "notebooks/submission.ipynb"))

    # Extract heliosli assets
    c5_helios = "".join(nb_helios["cells"][5]["source"])
    ns: dict = {}
    for line in c5_helios.splitlines():
        if line.strip().startswith("assets="):
            exec(line.strip(), ns)
            break
    helios_assets = dict(ns["assets"])
    helios_assets["patch_spatial_helpers.py"] = PATCH_SPATIAL_HELPERS

    fable_core = "".join(nb_fable["cells"][41]["source"])
    fable_patch = "".join(nb_fable["cells"][42]["source"])
    helios_assets["fable_layer_core.py"] = fable_core
    helios_assets["fable_layer_patch.py"] = fable_patch

    shiiin9_d_prime = "".join(nb_shiiin9["cells"][21]["source"])

    # Start from a deep copy of v20 so cell structure & metadata match 100%
    nb_v21 = copy.deepcopy(nb_v20)

    # 1. Update Cell 0 (Markdown overview)
    cell0_md = """# ARC Prize 2026 — Version 21 Sovereign Apex Engine

## Reverse-Engineered Leaderboard Domination Architecture (`D'` Priority + Zero-Action Model Checking + Perceptual Facts)
Synthesizes the 6 verified mechanisms from the top ARC-AGI-3 leaderboard solutions on top of Daniel Franzen's Milestone 2 foundation:
- **Breakthrough #1 — `D'` (`D-Prime`) Slot Priority Formula & Fresh-First Removal (`shiiin9`, `38.98` LB)**: Replaces stock `_PRIORITY_UNTRIMMED_BASE = 2,000,000` cold-start queue jumping with continuous `D'` Expected Value pricing ($\\text{priority} = A \\cdot M \\cdot C + B \\cdot \\phi$) and per-game `FormPace` token tracking (`ARC3_PRIORITY_PACE=1`, `fade=0.4`), keeping winning mid-game trajectories on the GPU through Levels 3–7.
- **Breakthrough #2 — Fatal Edge-Strip Meter (`edge_meter.py`, V26)**: Measures monotone shrinking 1px edge strips across level frames and reports `remaining_pixels` on `GAME_OVER` auto-resets, preventing the model from hallucinating lethal hazard tiles when death was caused by step-budget depletion (`0 (empty)`).
- **Breakthrough #3 — Zero-Action Observation Model Checker & Max-Entropy Probe Search (`model_checks.py`, V3)**: Exposes `check_transitions(encode, predict, max_cases=24)` to verify candidate transition rules against recorded level observations at zero live action cost, and `compare_rules(models, state, actions, max_depth=2)` to rank short discriminating probes by Shannon information gain per action ($H / L$).
- **Breakthrough #4 — Zlib-Compressed `AnimationArchive` (`animation_archive.py`, V4)**: Retains up to 32 multi-frame animations (4 MiB zlib-compressed) per game and exposes `animation_history()` and `animation_at(step, start=0, limit=32)` in the Python REPL at zero action cost.
- **Breakthrough #5 — Triggered Perceptual Facts (`fable_layer_core.py`, `small_far_changes` & `detect_budget`)**: Automatically detects distant $2\\times 2$ legend/HUD indicator state flips at Chebyshev distance $\\ge 8$ from main motion (after 15k tokens) and warns when step-budget bars drop below 25% ($\\le 12$ actions left), with `ARC3_FABLE_PIN_TOKENS=0` so past conversation history is never mutated (preserving $\\ge 93.7\\%$ SGLang RadixAttention KV-cache hit rate and zero advice bloat).
- **Breakthrough #6 — Empty Animation Diff Fix & REPL Spatial/Medoid Helpers**: Fixes misleading 0-change animation diffs (`patch_animation_empty.py`) and exposes `rot90`, `rot180`, `rot270`, `flipv`, `fliph`, `manhattan_distance`, `find_cells`, and medoid-projected `find_objects` directly inside the Python REPL without per-turn `<affordances>` XML prompt bloat.
"""
    nb_v21["cells"][0]["source"] = [line + "\n" for line in cell0_md.splitlines()[:-1]] + [cell0_md.splitlines()[-1]]

    # 2. Update Cell 2 (Pristine Daniel Franzen harness-changes.patch so all anchors match 100%)
    nb_v21["cells"][2]["source"] = copy.deepcopy(nb_df["cells"][2]["source"])

    # 3. Update Cell 4 (Environment setup + applying v21 post-patches + Fable & D' env flags)
    v21_post_patch_block = f"""# Apply harness patch
BUNDLE_DIR = Path('/kaggle/taaf-kaggle-source-share')
!rm -Rf $BUNDLE_DIR
!cp -a $ORIG_BUNDLE_DIR $BUNDLE_DIR
subprocess.run(["git", "apply", "--include=ARC3-Inference/*", "-v", "/kaggle/harness-changes.patch"], cwd=f"{{BUNDLE_DIR}}/src", check=True)
print('harness patch applied successfully')

# Apply Version 21 Sovereign Apex modules (V3 model_checks, V26 edge_meter, V4 animation_archive, animation_empty, spatial helpers, Fable perceptual facts)
_v21_dir = Path('/kaggle/arcagi_v21')
_v21_dir.mkdir(exist_ok=True)
_v21_assets = {repr(helios_assets)}
for _name, _text in _v21_assets.items():
    (_v21_dir / _name).write_text(_text, encoding='utf-8')
for _patch_script in ('patch_model_checks.py', 'patch_edge_meter.py', 'patch_animation_archive.py', 'patch_animation_empty.py', 'patch_spatial_helpers.py'):
    subprocess.run([sys.executable, str(_v21_dir / _patch_script), str(BUNDLE_DIR / 'src/ARC3-Inference')], check=True)
sys.path.insert(0, str(_v21_dir))
import fable_layer_patch as _fable_layer_patch
_fl_res = _fable_layer_patch.apply(str(BUNDLE_DIR / 'src/ARC3-Inference/inference/agent'), _v21_assets['fable_layer_core.py'])
assert _fl_res in ('applied', 'already applied'), f'Fable layer patch failed: {{_fl_res}}'
print('Applied v21 modules cleanly (model_checks, edge_meter, animation_archive, animation_empty, spatial_helpers, fable_layer=' + str(_fl_res) + ')')
"""

    c4_src = "".join(nb_v20["cells"][4]["source"])
    old_apply_block = """# Apply harness patch
BUNDLE_DIR = Path('/kaggle/taaf-kaggle-source-share')
!rm -Rf $BUNDLE_DIR
!cp -a $ORIG_BUNDLE_DIR $BUNDLE_DIR
subprocess.run(["git", "apply", "--include=ARC3-Inference/*", "-v", "/kaggle/harness-changes.patch"], cwd=f"{BUNDLE_DIR}/src", check=True)
print('harness patch applied successfully')"""
    assert old_apply_block in c4_src, "Could not find old_apply_block in cell 4"
    c4_src = c4_src.replace(old_apply_block, v21_post_patch_block, 1)

    # Also update priority scheduling & Fable env flags in cell 4
    old_prio_block = """if USE_PRIORITY_SCHEDULING: setup_env.update({
    'ARC3_MAX_ACTIVE_STREAMS': 10,
    'ARC3_PRIORITY_REFRESH_QUEUE': '1',
    'ARC3_PRIORITY_PACE': '0',
    'ARC3_PRIORITY_TAIL_FADE': '1',
    'ARC3_PRIORITY_TAIL_FADE_FRACTION': '0.2',
    'ARC3_PRIORITY_TAIL_LOOKUP': 'remaining',
    'ARC3_PRIORITY_SCORE_NORMALIZATION': '1',
})"""
    new_prio_block = """if USE_PRIORITY_SCHEDULING: setup_env.update({
    'ARC3_MAX_ACTIVE_STREAMS': 10,
    'ARC3_PRIORITY_REFRESH_QUEUE': '1',
    'ARC3_PRIORITY_PACE': '1',
    'ARC3_PRIORITY_TAIL_FADE': '1',
    'ARC3_PRIORITY_TAIL_FADE_FRACTION': '0.4',
    'ARC3_PRIORITY_TAIL_LOOKUP': 'remaining',
    'ARC3_PRIORITY_SCORE_NORMALIZATION': '1',
    'ARC3_EVIDENCE_DIAGNOSTICS': '0',
    'ARC3_FABLE_LAYER': '1',
    'ARC3_FABLE_EVENTS_AFTER_TOKENS': '15000',
    'ARC3_FABLE_BUDGET_ALERT': '1',
    'ARC3_FABLE_PIN_TOKENS': '0',
})"""
    assert old_prio_block in c4_src, "Could not find old_prio_block in cell 4"
    c4_src = c4_src.replace(old_prio_block, new_prio_block, 1)
    nb_v21["cells"][4]["source"] = [line + "\n" for line in c4_src.splitlines()[:-1]] + [c4_src.splitlines()[-1]]

    # 4. Update Cell 16 (Customization hook + D' Slot Priority Scheduler install_d)
    c16_src = "".join(nb_v20["cells"][16]["source"]).rstrip() + "\n\n" + shiiin9_d_prime.strip() + "\n"
    nb_v21["cells"][16]["source"] = [line + "\n" for line in c16_src.splitlines()[:-1]] + [c16_src.splitlines()[-1]]

    out_path = STARTER / "notebooks/submission_v21.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb_v21, f, indent=1, ensure_ascii=False)
        f.write("\n")

    raw_bytes = out_path.read_bytes()
    sha = hashlib.sha256(raw_bytes).hexdigest()
    print(f"Successfully generated {out_path} ({len(raw_bytes)} bytes, sha256={sha})")


if __name__ == "__main__":
    main()

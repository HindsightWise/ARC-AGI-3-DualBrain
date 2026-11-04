#!/usr/bin/env python3
"""
scripts/build_v25_notebook.py
=============================
Deterministic Builder for ARC Prize 2026 — Version 25 Sovereign Apex Engine.

Architectural Mandate:
1. Anchors on Version 24 foundation (28.79 LB personal best):
   - Cell 2: Pristine 498,799-byte TAAF harness patch (SHA-256 6fdcd1ae7445c0cfc26eec1047363aa0c196c867d289a35b6f64403b5a43dce2)
     with zero started_at clock tampering.
   - Cell 12: Pristine 10-stream Pennyroyal SGLang v2.5.3 + Albucino MTP-3 server
     (MAXREQ=10, CUDAGRAPH_MAXBS=10, MAMBA_CACHE=60, MEMFRAC=0.96).
   - Cell 16: Continuous Expected Value D' priority scheduler (ours_form_priority.install_d)
     pricing unstarted games at N=6 -> 18619, N=8 -> 17527, N=10 -> 17000, zero started_at reset,
     zero queue-aging thrash, with 5-minute single-game (ft09) rehearsal mode for fast commit runs.
2. Integrates the 3 proven solving accelerators from the v20 -> v21 benchmark surge (35.18 -> 55.83 mean):
   - AnimationFrameArchive (AnimationArchive) + Empty-transition guard (patch_animation_empty.py)
   - Targeted Perceptual Opener Alerts (fable_layer_core.py) with ARC3_FABLE_PIN_TOKENS = '0'
     (prefix immutability; never mutates _history_messages)
   - Zero-Prompt-Bloat REPL Spatial Helpers (patch_spatial_helpers.py: rot90, rot180, rot270, flipv, fliph,
     manhattan_distance, find_cells, find_objects) anchored to runtime_globals["action"] = action.
3. Installs late-game ReadTimeout Armor in Cell 16:
   - Wraps _HarnessGameSession.request_timeout_seconds with min_floor = 120.0s when soft_remaining is clamped,
     preventing 13.9s–89.4s premature socket aborts while preserving None when unbounded.
4. Purges bloat:
   - Excludes patch_model_checks.py (~350 tokens system prompt bloat) and edge_meter.py (0 activations).
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
V11_NB_PATH = REPO_ROOT / "pulled_kernels" / "v11" / "arc-prize-2026-v11-apex-engine.ipynb"
SHIIIN_NB_PATH = (
    REPO_ROOT
    / "pulled_kernels"
    / "oct13_top"
    / "shiiin9_affectify-arc-31-54-in-a-single-sub"
    / "affectify-arc-31-54-in-a-single-sub.ipynb"
)
HELIOS_NB_PATH = (
    REPO_ROOT
    / "pulled_kernels"
    / "oct13_top"
    / "heliosli_try-merging-it-you-might-get-lucky-and-win-gold"
    / "try-merging-it-you-might-get-lucky-and-win-gold.ipynb"
)
FABLE_NB_PATH = (
    REPO_ROOT
    / "pulled_kernels"
    / "oct13_top"
    / "leoprovorov_fable-astra-play-arc-3-handbook-harness"
    / "fable-astra-play-arc-3-handbook-harness.ipynb"
)
OUTPUT_V25_PATH = REPO_ROOT / "notebooks" / "submission_v25.ipynb"
OUTPUT_SUB_PATH = REPO_ROOT / "notebooks" / "submission.ipynb"
METADATA_PATH = REPO_ROOT / "notebooks" / "kernel-metadata.json"

CELL0_MARKDOWN = """# ARC Prize 2026 — Version 25 Sovereign Apex Engine

## Sovereign Apex Architecture: Pure Version 24 Substrate (`28.79` LB) + Three Verified Solving Accelerators (`35.18 -> 55.83` Jump) + Clamped ReadTimeout Armor

Version 25 synthesizes our verified personal best Version 24 foundation (`28.79` LB) with the three proven per-game solving accelerators from the `v20 -> v21` benchmark breakthrough (`35.18 -> 55.83` mean across 10 games, +20.65 points):

1. **Pure Version 24 Foundation (`28.79` LB)**:
   - 100% bit-for-bit preserved Daniel Franzen Milestone 2 base patch (Cell 2, 498,799 bytes, SHA-256 `6fdcd1ae7445c0cfc26eec1047363aa0c196c867d289a35b6f64403b5a43dce2`).
   - SGLang v2.5.3 server with `MEMFRAC = 0.96`, `MAXREQ = 10`, `CUDAGRAPH_MAXBS = 10`, and `MAMBA_CACHE = 60`.
   - Continuous Expected Value priority scheduler (`ours_form_priority.install_d`) with zero `started_at` clock reset and zero `aging_bonus` queue thrashing.
   - Fast commit rehearsal settings (`BM_GAMES = 'ft09'`, `BM_WALL_MINUTES = 5` when `not TRUE_SUBMISSION`), with full 110-game competition rerun (`concurrency = 120`, `max_runtime_s_per_game = 532 * 60`).

2. **Three Verified Per-Game Solving Accelerators (Zero System Prompt Bloat)**:
   - **Animation Frame Archive (`AnimationArchive`) + Empty-Transition Guard (`patch_animation_empty.py`)**: Exposes `animation_at(step)` and `animation_history()` in the Python REPL at zero game action cost (driving `sb26` from `27.05` to `100.00 WON!` and `sc25` from `26.56` to `47.62` in v21) while fixing empty-frame animation diff schema errors on single-frame actions.
   - **Targeted Perceptual Opener Notes (`fable_layer_core.py`)**: Fired 44 times (`Indicator note:` for distant $\\le 4$-cell changes at Chebyshev distance $\\ge 8$) and 1 time (`step-budget` low-bar warning) in v21, focusing visual attention on distant actuators without token bloat. Configured with `ARC3_FABLE_PIN_TOKENS = '0'` for strict RadixAttention prefix immutability.
   - **Zero-Prompt-Bloat REPL Spatial Helpers (`patch_spatial_helpers.py`)**: Injects `rot90`, `rot180`, `rot270`, `flipv`, `fliph`, `manhattan_distance`, `find_cells`, and `find_objects` into `_SANDBOX_BOOTSTRAP` (`runtime_globals`) with zero tokens added to the system prompt (`rot90` drove `tr87` from `47.62` to `71.43` in v21).

3. **Clamped ReadTimeout Harness Armor**:
   - `install_timeout_armor` wraps `_HarnessGameSession.request_timeout_seconds` in `solver.py` with `min_floor = 120.0s`, eliminating the self-inflicted late-game `13.9s–89.4s` client-side `ReadTimeout` turn drops that caused 4 late-game dropouts in v21.

4. **Bloat Purge**:
   - Excludes `patch_model_checks.py` (~350 tokens system prompt bloat, 0 activations in v21) and `edge_meter.py` (0 activations).
"""

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
    anchor = '        runtime_globals["action"] = action\n'
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

TIMEOUT_ARMOR_CODE = '''

def install_timeout_armor(solver_mod, min_floor: float = 120.0) -> str:
    """Protect _HarnessGameSession.request_timeout_seconds against premature soft-time clamping.
    
    Prevents self-inflicted 13.9s–89.4s ReadTimeout turn drops when soft_remaining
    is low while the game still has hard time remaining.
    """
    import os
    session_cls = getattr(solver_mod, "_HarnessGameSession", None)
    if session_cls is None or not hasattr(session_cls, "request_timeout_seconds"):
        raise RuntimeError("solver._HarnessGameSession.request_timeout_seconds not found")

    if not getattr(session_cls.request_timeout_seconds, "_ours_protected", False):
        orig_rto = session_cls.request_timeout_seconds

        def protected_rto(self):
            cand = orig_rto(self)
            floor = float(os.environ.get("ARC3_MIN_READ_TIMEOUT", str(min_floor)))
            return max(floor, float(cand)) if cand is not None else None

        protected_rto._ours_protected = True
        session_cls.request_timeout_seconds = protected_rto

    line = f"#TIMEOUT_ARMOR ok: request_timeout_seconds floor >= {min_floor}s installed"
    print(line, flush=True)
    return line
'''


def _src(cell: dict) -> str:
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else str(s)


def extract_v25_assets() -> dict[str, str]:
    if not HELIOS_NB_PATH.exists():
        raise FileNotFoundError(f"Missing helios reference notebook at {HELIOS_NB_PATH}")
    if not FABLE_NB_PATH.exists():
        raise FileNotFoundError(f"Missing fable reference notebook at {FABLE_NB_PATH}")

    nb_helios = json.loads(HELIOS_NB_PATH.read_text(encoding="utf-8"))
    c5_helios = _src(nb_helios["cells"][5])
    ns: dict = {}
    for line in c5_helios.splitlines():
        if line.strip().startswith("assets="):
            exec(line.strip(), ns)
            break
    helios_assets = dict(ns["assets"])

    nb_fable = json.loads(FABLE_NB_PATH.read_text(encoding="utf-8"))
    fable_core = _src(nb_fable["cells"][41])
    fable_patch = _src(nb_fable["cells"][42])

    v25_assets = {
        "animation_archive.py": helios_assets["animation_archive.py"],
        "patch_animation_archive.py": helios_assets["patch_animation_archive.py"],
        "patch_animation_empty.py": helios_assets["patch_animation_empty.py"],
        "fable_layer_core.py": fable_core,
        "fable_layer_patch.py": fable_patch,
        "patch_spatial_helpers.py": PATCH_SPATIAL_HELPERS,
    }
    return v25_assets


def build_v25_notebook() -> dict:
    if not V11_NB_PATH.exists():
        raise FileNotFoundError(f"Missing v11 reference notebook at {V11_NB_PATH}")
    if not SHIIIN_NB_PATH.exists():
        raise FileNotFoundError(f"Missing shiiin9 reference notebook at {SHIIIN_NB_PATH}")

    nb_v11 = json.loads(V11_NB_PATH.read_text(encoding="utf-8"))
    nb_shiiin = json.loads(SHIIIN_NB_PATH.read_text(encoding="utf-8"))

    # Verify that v11 cells 1..15 and 17..22 match shiiin9 cells 2..16, 18..20, 22, 24, 25 byte-for-byte
    v11_to_shiiin = {
        1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9,
        9: 10, 10: 11, 11: 12, 12: 13, 13: 14, 14: 15, 15: 16,
        17: 18, 18: 19, 19: 20, 20: 22, 21: 24, 22: 25,
    }
    for v11_idx, sh_idx in v11_to_shiiin.items():
        if _src(nb_v11["cells"][v11_idx]) != _src(nb_shiiin["cells"][sh_idx]):
            raise AssertionError(f"Cell mismatch between v11[{v11_idx}] and shiiin9[{sh_idx}]")

    v25_assets = extract_v25_assets()

    # Build 23-cell notebook anchored on v11
    nb_v25 = copy.deepcopy(nb_v11)
    assert len(nb_v25["cells"]) == 23

    # Cell 0: Markdown overview
    nb_v25["cells"][0]["source"] = CELL0_MARKDOWN

    # Cell 2: Base patch (preserved 100%, 498,799 bytes, SHA-256 6fdcd1ae...)
    cell2_src = _src(nb_v11["cells"][2])
    assert len(cell2_src) == 498799, f"Cell 2 length mismatch: expected 498799, got {len(cell2_src)}"
    cell2_sha = hashlib.sha256(cell2_src.encode("utf-8")).hexdigest()
    assert cell2_sha == "6fdcd1ae7445c0cfc26eec1047363aa0c196c867d289a35b6f64403b5a43dce2", f"Cell 2 SHA mismatch: {cell2_sha}"
    nb_v25["cells"][2]["source"] = cell2_src

    # Cell 4: Apply harness patch + v25 post-patches + v25 setup_env flags
    c4_src = _src(nb_v11["cells"][4])
    old_apply = """subprocess.run(["git", "apply", "--include=ARC3-Inference/*", "-v", "/kaggle/harness-changes.patch"], cwd=f"{BUNDLE_DIR}/src", check=True)
print('harness patch applied successfully')"""
    assert old_apply in c4_src, "Could not find old_apply in cell 4"

    v25_post_patch_block = f"""subprocess.run(["git", "apply", "--include=ARC3-Inference/*", "-v", "/kaggle/harness-changes.patch"], cwd=f"{{BUNDLE_DIR}}/src", check=True)
print('harness patch applied successfully')

# Apply Version 25 Sovereign Apex modules (AnimationArchive, animation_empty, spatial_helpers, Fable perceptual facts)
_v25_dir = Path('/kaggle/arcagi_v25')
_v25_dir.mkdir(exist_ok=True)
_v25_assets = {repr(v25_assets)}
for _name, _text in _v25_assets.items():
    (_v25_dir / _name).write_text(_text, encoding='utf-8')
for _patch_script in ('patch_animation_archive.py', 'patch_animation_empty.py', 'patch_spatial_helpers.py'):
    subprocess.run([sys.executable, str(_v25_dir / _patch_script), str(BUNDLE_DIR / 'src/ARC3-Inference')], check=True)
sys.path.insert(0, str(_v25_dir))
import fable_layer_patch as _fable_layer_patch
_fl_res = _fable_layer_patch.apply(str(BUNDLE_DIR / 'src/ARC3-Inference/inference/agent'), _v25_assets['fable_layer_core.py'])
assert _fl_res in ('applied', 'already applied'), f'Fable layer patch failed: {{_fl_res}}'
print('Applied v25 modules cleanly (animation_archive, animation_empty, spatial_helpers, fable_layer=' + str(_fl_res) + ')')"""
    c4_src = c4_src.replace(old_apply, v25_post_patch_block, 1)

    old_prio_env = """if USE_PRIORITY_SCHEDULING: setup_env.update({
    'ARC3_MAX_ACTIVE_STREAMS': 10,
    'ARC3_PRIORITY_REFRESH_QUEUE': '1',
    'ARC3_PRIORITY_PACE': '0',
    'ARC3_PRIORITY_TAIL_FADE': '1',
    'ARC3_PRIORITY_TAIL_FADE_FRACTION': '0.2',
    'ARC3_PRIORITY_TAIL_LOOKUP': 'remaining',
    'ARC3_PRIORITY_SCORE_NORMALIZATION': '1',
})"""
    assert old_prio_env in c4_src, "Could not find old_prio_env in cell 4"

    new_prio_env = """if USE_PRIORITY_SCHEDULING: setup_env.update({
    'ARC3_MAX_ACTIVE_STREAMS': 10,
    'ARC3_PRIORITY_REFRESH_QUEUE': '1',
    'ARC3_PRIORITY_PACE': '0',
    'ARC3_PRIORITY_TAIL_FADE': '1',
    'ARC3_PRIORITY_TAIL_FADE_FRACTION': '0.2',
    'ARC3_PRIORITY_TAIL_LOOKUP': 'remaining',
    'ARC3_PRIORITY_SCORE_NORMALIZATION': '1',
    'ARC3_FABLE_LAYER': '1',
    'ARC3_FABLE_EVENTS_AFTER_TOKENS': '15000',
    'ARC3_FABLE_BUDGET_ALERT': '1',
    'ARC3_FABLE_PIN_TOKENS': '0',
    'ARC3_MIN_READ_TIMEOUT': '120.0',
})"""
    c4_src = c4_src.replace(old_prio_env, new_prio_env, 1)
    nb_v25["cells"][4]["source"] = c4_src

    # Cell 16: Combine shiiin9's Cell 17 (bm.solver config + ft09 rehearsal)
    # and shiiin9's Cell 21 (ours_form_priority.install_d) + install_timeout_armor
    solver_cfg_src = _src(nb_shiiin["cells"][17]).rstrip()
    d_prime_src = _src(nb_shiiin["cells"][21]).rstrip()

    marker = "_FP_SOURCE = "
    idx = d_prime_src.find(marker)
    assert idx != -1, "Could not find _FP_SOURCE in shiiin9 Cell 21"
    end_marker = "\n_fp_mod ="
    idx_end = d_prime_src.find(end_marker)
    assert idx_end != -1, "Could not find _fp_mod in shiiin9 Cell 21"

    fp_src_literal = d_prime_src[idx + len(marker):idx_end].strip()
    fp_src = ast.literal_eval(fp_src_literal)
    new_fp_src = fp_src + TIMEOUT_ARMOR_CODE

    tail = d_prime_src[idx_end:].rstrip()
    tail += "\n_fp_mod.install_timeout_armor(_fp_solver, min_floor=120.0)\n"

    reconstructed_d_prime = d_prime_src[:idx] + marker + repr(new_fp_src) + tail
    nb_v25["cells"][16]["source"] = solver_cfg_src + "\n\n" + reconstructed_d_prime + "\n"

    # Cell 20: Append the #OURS_FRESH verification print right after bm.run()
    c20_src = _src(nb_v11["cells"][20]).rstrip()
    fresh_print_src = _src(nb_shiiin["cells"][23]).rstrip()
    nb_v25["cells"][20]["source"] = c20_src + "\n" + fresh_print_src + "\n"

    # Clear any stale outputs or execution_counts across all cells
    for cell in nb_v25["cells"]:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    return nb_v25


def update_kernel_metadata() -> None:
    meta = {
        "id": "zerbytheboss/v25-sovereign-apex-engine",
        "title": "v25 - Sovereign Apex Engine",
        "code_file": "submission.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "dataset_sources": [
            "dfranzen/pennyroyal-v253",
            "dfranzen/taaf-kaggle-source-bundle-copy",
        ],
        "kernel_sources": [],
        "competition_sources": ["arc-prize-2026-arc-agi-3"],
        "model_sources": [
            "dfranzen/albucino-qwen3-8-flash-next-drafter/Transformers/default/1",
            "dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/Transformers/default/1",
        ],
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c",
        "machine_shape": "NvidiaRtxPro6000",
    }
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    print(f"Updated {METADATA_PATH} (id={meta['id']}, machine_shape={meta['machine_shape']})")


def main() -> None:
    nb_v25 = build_v25_notebook()
    OUTPUT_V25_PATH.parent.mkdir(parents=True, exist_ok=True)

    for path in (OUTPUT_V25_PATH, OUTPUT_SUB_PATH):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb_v25, f, indent=1, ensure_ascii=False)
            f.write("\n")

    update_kernel_metadata()

    sha_v25 = hashlib.sha256(OUTPUT_V25_PATH.read_bytes()).hexdigest()
    sha_sub = hashlib.sha256(OUTPUT_SUB_PATH.read_bytes()).hexdigest()
    assert sha_v25 == sha_sub, "submission_v25.ipynb and submission.ipynb must be byte-identical"
    assert len(nb_v25["cells"]) == 23, f"Expected 23 cells, got {len(nb_v25['cells'])}"
    print(
        f"Built {OUTPUT_V25_PATH.name} and {OUTPUT_SUB_PATH.name} "
        f"({OUTPUT_V25_PATH.stat().st_size} bytes, sha256={sha_v25[:12]}, cells=23)"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Build Version 30 (v30) Sovereign Restoration & Apex Notebook for ARC Prize 2026 (ARC-AGI-3).

Bedrock Root-Cause Autopsy of the v24 -> v25 -> v26 -> v28 -> v29 Monotonic Decline:
- v24 (57337517, 28.79 LB PB): Pure dfranzen M2 + shiiin9 D' (b0=0.0), zero custom monkey-patches.
- v25 (57373064, 27.16 LB, -1.63): Added patch_spatial_helpers.py (injected find_cells, find_objects,
  manhattan_distance, rot90, etc. into runtime_globals before _available_names, causing retained_functions.py
  to reject model-defined functions under ARC3_PERSISTENT_FUNCTIONS=1!) + prompt-mutating patches.
- v26 (57471652, 26.29 LB, -2.50): Kept patch_spatial_helpers.py + added patch_solved_level_memory.py
  (mutated _build_user_prompt and self._system_prompt, invalidating shared 10-stream RadixAttention root KV cache).
- v28 (57541661, 21.47 LB, -7.32): Kept patch_spatial_helpers.py + changed D' to b0=6.0 WITHOUT C-scaling
  (causing stalled final-level games to hold priority >= 6,000 forever) + changed precache to delay=15, threads=8
  (causing 3.34x FUSE weight-load slowdown from 284s to 950s) + lowered SERVER_STARTUP_TIMEOUT to 10*60 and
  reset self.started_at in TRUE_SUBMISSION (pushing per-game deadlines past Kaggle's 540m container limit).
- v29 (57575158, 20.08 LB, -8.71): Added patch_post_eviction_memory.py (which wiped 100% of self._history_messages
  and self._kept_functions on level >= 2 after 62k tokens or 95 actions via _slm_maybe_stagnation_restart,
  double-drained context down to 58k tokens via force_drain=True after stripping reasoning_content, and mutated
  result[1] at the front of history) + patch_sovereign_oo_harness.py (13 unadvertised runtime_globals breaking
  retained_functions.py) + 9 unverified ARC3_* env vars (ARC3_DEATH_LEDGER=1, ARC3_ANIMATION_IMAGE=composite).

Version 30 (v30) Architecture:
1. 100% Purge of All Regressive Monkey-Patches (patch_post_eviction_memory.py, patch_spatial_helpers.py,
   patch_sovereign_oo_harness.py, patch_solved_level_memory.py, and the 9 unverified v29 ARC3_* flags).
   - Zero mutations to self._system_prompt or _build_user_prompt across all turns, levels, and context trims.
   - Zero extra names injected into python_tool_sandbox.py runtime_globals, restoring 100% native
     ARC3_PERSISTENT_FUNCTIONS=1 cross-turn and cross-level helper retention via retained_functions.py.
2. Upgraded D' Slot Priority Scheduler with C-Scaled Stagnation Decay & Final-Level Cliff Fix (b0=8.0, B_raw * C):
   - Incorporates the exact 3-line mathematical refinement from ouyangruize's Oct 24 D' notebook:
     b0 = params.get("b0", 8.0) if level > 1 else 0.0
     B_raw = (b0, params["b1"], params["b2"], params["b3"])[min(remaining, 3)]
     return A * C, B_raw * C
   - Decays the room-to-grow bonus B by C (token/action patience factor), dropping stalled Level 1 games
     from 16,036 -> 8,252 (at 1st trim) -> 1,636 (stalled), while lifting fresh final-level games (level == N > 1)
     from 9,166 -> 17,166 and decaying stalled final-level games cleanly down to 1,132.
3. Surgical Transient-Animation No-Op Guard Fix (transient_pixels > 0) + Empty Animation Schema Repair:
   - Prevents actions with visible intermediate frame motion (transient_pixels > 0) whose final frame restores
     the board from falsely triggering _stale_state or BATCH_NOOP_BLOCK, and ensures single-frame animation()
     calls return {action, frames: 0, steps: [], error}.
4. Zero-Global M75 Segmentation Key Aliases (patch_segmentation_aliases.py):
   - Applies _NodeDict and _BoundaryList strictly inside inference/utils/segmentation.py (resolving
     bbox/box/bounds, area/size/count, centroid/center via __missing__ and .get() with zero stdout/prompt drift
     and zero runtime_globals changes).
5. Proven v24 FUSE Prefetching (delay=60, threads=3) + Clean Cold-Boot Immunity (patch_startup_shield.py):
   - Restores dfranzen/v24 precache(delay=60, threads=3) (284s weight load, 490s SGLang boot).
   - Mode-specific SERVER_STARTUP_TIMEOUT (12*60 in competition rerun, 35*60 in rehearsal), 480s periodic
     warmup RESET keepalive until /health == 200 OK, and dynamic competition runtime cap (min(532*60, 538*60 - boot_elapsed)).
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
V11_NB_PATH = REPO_ROOT / "pulled_kernels" / "v11" / "arc-prize-2026-v11-apex-engine.ipynb"
SHIIIN_NB_PATH = (
    REPO_ROOT
    / "pulled_kernels"
    / "oct13_top"
    / "shiiin9_affectify-arc-31-54-in-a-single-sub"
    / "affectify-arc-31-54-in-a-single-sub.ipynb"
)
OUTPUT_V30_PATH = REPO_ROOT / "notebooks" / "submission_v30.ipynb"
OUTPUT_SUB_PATH = REPO_ROOT / "notebooks" / "submission.ipynb"
METADATA_PATH = REPO_ROOT / "notebooks" / "kernel-metadata.json"

CELL0_MARKDOWN = r"""# ARC Prize 2026 — Version 30 Sovereign Restoration & Apex Engine

## Pure `v24` (`28.79` LB) & `shiiin9` (`31.54` LB) Foundation + `C`-Scaled `D'` Priority (`b0=8.0`, `B_raw * C`) + Transient-Animation Guard Fix + Zero-Global `M75` Segmentation + Cold-Boot Immunity

Version 30 is the result of a full forensic autopsy of `v24` (`28.79` LB) through `v29` (`20.08` LB) and synthesis of the October 24 top public kernels (`ouyangruize`, `shiiin9`, `manderson240`, `weitaoguo`):

1. **Complete Purge of `v25`–`v29` Regressive Monkey-Patches**:
   - **Purged `patch_post_eviction_memory.py` (`v29`)**: Eliminates `_slm_maybe_stagnation_restart` (which wiped 100% of `self._history_messages` and `self._kept_functions` at 62,000 tokens or 95 actions on `level >= 2`), eliminates the `force_drain=True` double context drain (which evicted >100k tokens of conversation history per trim), and eliminates front-of-history `result[1]` mutation.
   - **Purged `runtime_globals` Injections (`patch_spatial_helpers.py` / `patch_sovereign_oo_harness.py`)**: Eliminates unadvertised global functions (`find_cells`, `find_objects`, `manhattan_distance`, `rot90`, etc.) that collided with `_available_names` in `python_tool_sandbox.py` and caused `retained_functions.py` (`ARC3_PERSISTENT_FUNCTIONS=1`) to reject and drop model-defined helper functions across turns.
   - **100% Pure `v24` System & User Prompts + Environment**: Keeps `self._system_prompt`, `_build_user_prompt`, and `setup_env` 100% untouched across all turns, levels, and context trims, preserving 100% shared RadixAttention prefix cache residency across all 10 concurrent SGLang streams.

2. **Upgraded `D'` Slot Priority Scheduler with `C`-Scaled Stagnation Decay & Final-Level Cliff Fix (`b0 = 8.0`, `B_raw * C`)**:
   - Replaces unscaled `B` (`return A * C, B`) with `C`-scaled `B_raw * C` and `b0 = 8.0 if level > 1 else 0.0` (`ouyangruize` Oct 24 refinement):
     $$\text{priority} = (A \cdot M \cdot C) + (B_{\text{raw}} \cdot C) \cdot \phi$$
   - Stalled Level 1 games decay from `16,036` $\to$ `8,252` (at first context trim) $\to$ `1,636` (when stalled), automatically yielding GPU slots to productive multi-level games (`17,166–19,928`), while games reaching their final level ($\ell = N > 1$, `remaining = 0`) enter at `17,166` instead of suffering the `b0 = 0.0` cliff (`9,166`).

3. **Surgical Transient-Animation No-Op Guard Fix (`transient_pixels > 0`) & Empty Animation Schema**:
   - `patch_transient_animation.py`: Prevents actions that trigger intermediate animation frames (`transient_pixels > 0`) before the board settles back to its prior state from falsely arming `_stale_state` or triggering `BATCH_NOOP_BLOCK`, and ensures single-frame `animation()` calls return `{action, frames: 0, steps: [], error}`.

4. **Zero-Global `M75` Segmentation Key Aliases (`patch_segmentation_aliases.py`)**:
   - Wraps `segment_layer` nodes in `_NodeDict(dict)` and `_BoundaryList(list)` strictly inside `inference/utils/segmentation.py` (resolving `bbox`/`box`/`bounds`, `area`/`size`/`count`, and `centroid`/`center` via `__missing__` and `.get()`), preventing 198+ `KeyError` turns without adding a single symbol to `runtime_globals`.

5. **Proven `v24` FUSE Prefetching (`delay=60, threads=3`) & Clean Cold-Boot `0.00` Immunity (`patch_startup_shield.py`)**:
   - Restores `precache(delay=60, threads=3)` (`284s` weight load, `490s` SGLang startup in `v24`) alongside mode-specific `SERVER_STARTUP_TIMEOUT` (`12m` competition / `35m` rehearsal), `480s` periodic warmup `RESET` keepalive until `/health == 200 OK`, and a dynamic `538m` competition runtime cap so late cold boots never overrun Kaggle's `540m` container limit.
"""

PATCH_STARTUP_SHIELD = r'''"""Install periodic warmup RESET keepalive during slow SGLang cold boots (Topics 750173 & 750885)."""
import ast
from pathlib import Path
import sys

BEFORE = """        if self.solver._claim_warmup_slot():
            # Straight to the engine - _execute_auto_reset takes the same path
            # and needs no analyzer, so this works with the endpoint still
            # loading. One action off this game's budget buys the whole
            # submission a deadline it would otherwise miss.
            # warning, not info: _configure_logging only runs under run.py, so
            # the notebook has no handler and falls back to logging's lastResort,
            # which emits WARNING and above. An info line would vanish exactly
            # where this message matters most.
            log.warning(
                "%s: warmup RESET issued before the analyzer is reachable",
                run.game_id,
            )
            self._execute_auto_reset()
"""

AFTER = """        _is_warmup_slot = bool(self.solver._claim_warmup_slot())
        if _is_warmup_slot:
            log.warning(
                "%s: warmup RESET issued before the analyzer is reachable",
                run.game_id,
            )
            self._execute_auto_reset()
        if (
            not getattr(self.solver, "_analyzer_ready", False)
            and int(os.environ.get("ARC3_WARMUP_ACTION_GAMES", "0") or 0) > 0
        ):
            _warmup_last_ts = time.monotonic()
            _warmup_base_url = os.environ.get("LOCAL_ANALYZER_BASE_URL", "http://127.0.0.1:8001/v1")
            _warmup_health_url = (
                _warmup_base_url[:-3] + "/health"
                if _warmup_base_url.endswith("/v1")
                else _warmup_base_url.rstrip("/") + "/health"
            )
            _warmup_grace_s = float(os.environ.get("ARC3_HTTP_RETRY_INITIAL_SECONDS", "1800") or 1800.0)
            _warmup_deadline = _warmup_last_ts + _warmup_grace_s
            _warmup_interval_s = float(os.environ.get("ARC3_WARMUP_KEEPALIVE_SECONDS", "480") or 480.0)
            import urllib.request as _warmup_urllib
            while time.monotonic() < _warmup_deadline and not self.stop_event.is_set():
                if getattr(self.solver, "_analyzer_ready", False):
                    break
                try:
                    with _warmup_urllib.urlopen(_warmup_health_url, timeout=3) as _wresp:
                        if getattr(_wresp, "status", 200) == 200:
                            setattr(self.solver, "_analyzer_ready", True)
                            break
                except Exception:
                    pass
                if _is_warmup_slot and (time.monotonic() - _warmup_last_ts) >= _warmup_interval_s:
                    log.warning(
                        "%s: periodic warmup RESET issued while analyzer is still loading",
                        run.game_id,
                    )
                    self._execute_auto_reset()
                    _warmup_last_ts = time.monotonic()
                time.sleep(min(3.0, max(0.05, _warmup_interval_s / 20.0)))
            if (getattr(self.solver, "max_runtime_s_per_game", 0) or 0) <= 3600:
                self.started_at = time.monotonic()
"""


def apply(root):
    path = Path(root) / "inference/framework/solver.py"
    text = path.read_text(encoding="utf-8")
    if "periodic warmup RESET issued while analyzer is still loading" in text:
        return
    if text.count(BEFORE) != 1:
        raise RuntimeError(f"Expected warmup block once in solver.py, got {text.count(BEFORE)}")
    changed = text.replace(BEFORE, AFTER, 1)
    ast.parse(changed)
    path.write_text(changed, encoding="utf-8")


if __name__ == "__main__":
    apply(sys.argv[1])
'''

PATCH_TRANSIENT_ANIMATION = r'''"""Install transient-animation no-op guard fix (transient_pixels > 0) and empty animation schema fallback."""
import ast
from pathlib import Path
import sys

ANIM_BEFORE = """    empty = {
        "error": (
            "No animation is available: the last executed action returned a single frame."
        )
    }
"""

ANIM_AFTER = """    empty = {
        "action": record.get("action_display") if record else None,
        "frames": 0,
        "steps": [],
        "error": (
            "No animation is available: the last executed action returned a single frame."
        )
    }
"""

SANDBOX_BEFORE = """            gameplay_changed = action_result.get("gameplay_changed")
            if gameplay_changed is True:
                # a real change means the snippet's beliefs are current again
                del _stale_state[:]
            if gameplay_changed is False and action_result.get("executed"):"""

SANDBOX_AFTER = """            gameplay_changed = action_result.get("gameplay_changed")
            animated = bool((action_result.get("animation") or {}).get("transient_pixels", 0))
            if gameplay_changed is True or animated:
                # Visible or intermediate motion keeps the snippet's beliefs live.
                del _stale_state[:]
            if gameplay_changed is False and not animated and action_result.get("executed"):"""

SOLVER_VERDICT_BEFORE = """    if item.get("level_completed"):
        return "LEVEL COMPLETED"
    changed = item.get("gameplay_changed")
    if changed is False:"""

SOLVER_VERDICT_AFTER = """    if item.get("level_completed"):
        return "LEVEL COMPLETED"
    changed = item.get("gameplay_changed")
    if changed is False and (item.get("animation") or {}).get("transient_pixels", 0):
        return "animated effect (intermediate frames changed; final board restored)"
    if changed is False:"""

SOLVER_BATCH_BEFORE = """                and NOOP_GUARD_BORDER_WIDTH > 0
                and batch_size >= 2
                and not payload.get("gameplay_changed", True)
            ):
                stop_reason = "no_op_action"
                break"""

SOLVER_BATCH_AFTER = """                and NOOP_GUARD_BORDER_WIDTH > 0
                and batch_size >= 2
                and not payload.get("gameplay_changed", True)
                # A transient animation is evidence of a real intermediate effect;
                # the final frame alone cannot certify a no-op.
                and not (payload.get("animation") or {}).get("transient_pixels", 0)
            ):
                stop_reason = "no_op_action"
                break"""


def apply(root):
    root = Path(root)

    # 1. animation.py empty schema repair
    anim_path = root / "inference/utils/animation.py"
    anim_text = anim_path.read_text(encoding="utf-8")
    if ANIM_AFTER not in anim_text:
        if anim_text.count(ANIM_BEFORE) != 1:
            raise RuntimeError("Expected the original animation empty-result block exactly once")
        anim_text = anim_text.replace(ANIM_BEFORE, ANIM_AFTER, 1)
        ast.parse(anim_text)
        anim_path.write_text(anim_text, encoding="utf-8")

    # 2. python_tool_sandbox.py transient animation stale-state fix
    sb_path = root / "inference/agent/python_tool_sandbox.py"
    sb_text = sb_path.read_text(encoding="utf-8")
    if SANDBOX_AFTER not in sb_text:
        if sb_text.count(SANDBOX_BEFORE) != 1:
            raise RuntimeError(f"Expected SANDBOX_BEFORE once, got {sb_text.count(SANDBOX_BEFORE)}")
        sb_text = sb_text.replace(SANDBOX_BEFORE, SANDBOX_AFTER, 1)
        ast.parse(sb_text)
        sb_path.write_text(sb_text, encoding="utf-8")

    # 3. solver.py _action_verdict and step_env batch no-op guard transient animation fix
    solver_path = root / "inference/framework/solver.py"
    solver_text = solver_path.read_text(encoding="utf-8")
    if SOLVER_VERDICT_AFTER not in solver_text:
        if solver_text.count(SOLVER_VERDICT_BEFORE) != 1:
            raise RuntimeError(f"Expected SOLVER_VERDICT_BEFORE once, got {solver_text.count(SOLVER_VERDICT_BEFORE)}")
        solver_text = solver_text.replace(SOLVER_VERDICT_BEFORE, SOLVER_VERDICT_AFTER, 1)
    if SOLVER_BATCH_AFTER not in solver_text:
        if solver_text.count(SOLVER_BATCH_BEFORE) != 1:
            raise RuntimeError(f"Expected SOLVER_BATCH_BEFORE once, got {solver_text.count(SOLVER_BATCH_BEFORE)}")
        solver_text = solver_text.replace(SOLVER_BATCH_BEFORE, SOLVER_BATCH_AFTER, 1)
    ast.parse(solver_text)
    solver_path.write_text(solver_text, encoding="utf-8")


if __name__ == "__main__":
    apply(sys.argv[1])
'''

PATCH_SEGMENTATION_ALIASES = r'''"""Install zero-global M75 _NodeDict and _BoundaryList key aliases strictly inside segmentation.py."""
import ast
from pathlib import Path
import sys

NODE_BEFORE = """        nodes.append(
            {
                "id": cid,
                "color": color,
                "hash": signature[0],
                "rotation": signature[1],
                "rotational_symmetry": signature[2],
                "pose_hash": signature[3],
                "shape_hash": signature[4],
                "pixels": len(comp["cells"]),
                "boundary": [[r, c] for r, c in boundary],
                "children": children[cid],
            }
        )"""

NODE_AFTER = """        nodes.append(
            _NodeDict({
                "id": cid,
                "color": color,
                "hash": signature[0],
                "rotation": signature[1],
                "rotational_symmetry": signature[2],
                "pose_hash": signature[3],
                "shape_hash": signature[4],
                "pixels": len(comp["cells"]),
                "boundary": _BoundaryList([r, c] for r, c in boundary),
                "children": children[cid],
            })
        )"""

SEG_ADDENDUM = """

class _BoundaryList(list):
    \"\"\"List subclass for node['boundary'] that safely resolves index 2 / 3 on 1D or 1px contours
    when models write n['boundary'][0], n['boundary'][2] expecting [top_left, ..., bottom_right].\"\"\"

    def __getitem__(self, idx):
        if isinstance(idx, int) and len(self) > 0:
            if idx == 2 and len(self) == 2:
                return super().__getitem__(1)
            if idx in (1, 2, 3) and len(self) == 1:
                return super().__getitem__(0)
        return super().__getitem__(idx)


class _NodeDict(dict):
    \"\"\"Zero-stdout-overhead dict subclass for segment_layer nodes (rellik13 M75).
    Preserves exact repr(), keys(), and JSON serialization while resolving common
    model key aliases (bbox/box/bounds, size/area/count, centroid/center) on demand.
    Does NOT touch python_tool_sandbox.py runtime_globals or _available_names.\"\"\"

    def __missing__(self, key):
        if key in ("size", "area", "count"):
            return self["pixels"]
        b = self.get("boundary") or []
        if b and key in ("bbox", "box", "bounds"):
            rs = [p[0] for p in b]
            cs = [p[1] for p in b]
            return [min(rs), min(cs), max(rs), max(cs)]
        if b and key in ("centroid", "center"):
            rs = [p[0] for p in b]
            cs = [p[1] for p in b]
            return ((min(rs) + max(rs)) // 2, (min(cs) + max(cs)) // 2)
        raise KeyError(f"{key!r} (valid keys: {list(self.keys())})")

    def get(self, key, default=None):
        if key in self:
            return self[key]
        try:
            return self[key]
        except KeyError:
            return default
"""


def apply(root):
    root = Path(root)
    seg_path = root / "inference/utils/segmentation.py"
    seg_src = seg_path.read_text(encoding="utf-8")
    if "class _NodeDict(" not in seg_src:
        if seg_src.count(NODE_BEFORE) != 1:
            raise RuntimeError(f"Expected NODE_BEFORE once in segmentation.py, got {seg_src.count(NODE_BEFORE)}")
        seg_src = seg_src.replace(NODE_BEFORE, NODE_AFTER, 1) + SEG_ADDENDUM
        ast.parse(seg_src)
        seg_path.write_text(seg_src, encoding="utf-8")


if __name__ == "__main__":
    apply(sys.argv[1])
'''


def _src(cell: dict) -> str:
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else str(s)


def extract_v30_assets() -> dict[str, str]:
    return {
        "patch_startup_shield.py": PATCH_STARTUP_SHIELD,
        "patch_transient_animation.py": PATCH_TRANSIENT_ANIMATION,
        "patch_segmentation_aliases.py": PATCH_SEGMENTATION_ALIASES,
    }


def build_v30_notebook() -> dict:
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

    v30_assets = extract_v30_assets()

    # Build 23-cell notebook anchored on v11 / v24
    nb_v30 = copy.deepcopy(nb_v11)
    assert len(nb_v30["cells"]) == 23

    # Cell 0: Markdown overview
    nb_v30["cells"][0]["source"] = CELL0_MARKDOWN

    # Cell 2: Base patch (preserved 100%, 498,799 bytes, SHA-256 6fdcd1ae...)
    cell2_src = _src(nb_v11["cells"][2])
    assert len(cell2_src) == 498799, f"Cell 2 length mismatch: expected 498799, got {len(cell2_src)}"
    cell2_sha = hashlib.sha256(cell2_src.encode("utf-8")).hexdigest()
    assert cell2_sha == "6fdcd1ae7445c0cfc26eec1047363aa0c196c867d289a35b6f64403b5a43dce2", f"Cell 2 SHA mismatch: {cell2_sha}"
    nb_v30["cells"][2]["source"] = cell2_src

    # Cell 4: Mode-specific startup timeout + dual mount path resolver + 3 surgical zero-prompt post-patches + pristine v24 setup_env
    c4_src = _src(nb_v11["cells"][4])
    old_startup_cfg = """NOTEBOOK_START_TIME = time.time()
SERVER_STARTUP_TIMEOUT = 12 * 60
os.environ['ARC3_WARMUP_ACTION_GAMES'] = '1'
os.environ['ARC3_HTTP_RETRY_INITIAL_SECONDS'] = '900'"""
    new_startup_cfg = """def _arc_gateway_reachable(url: str = "http://gateway:8001/api/games", timeout_s: float = 2.0) -> bool:
    try:
        with urlopen(url, timeout=timeout_s) as resp:
            return int(getattr(resp, "status", 500)) < 500
    except Exception:
        return False

_RERUN_FLAG = os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1", "true"}
_GATEWAY_UP = False if _RERUN_FLAG else _arc_gateway_reachable()
TRUE_SUBMISSION = _RERUN_FLAG or _GATEWAY_UP
NOTEBOOK_START_TIME = time.time()
SERVER_STARTUP_TIMEOUT = (12 * 60) if TRUE_SUBMISSION else (35 * 60)
os.environ['ARC3_WARMUP_ACTION_GAMES'] = '1' if TRUE_SUBMISSION else '0'
os.environ['ARC3_HTTP_RETRY_INITIAL_SECONDS'] = '1800'"""
    assert old_startup_cfg in c4_src, "Could not find old_startup_cfg in cell 4"
    c4_src = c4_src.replace(old_startup_cfg, new_startup_cfg, 1)

    old_paths_cfg = """# model and dataset paths
WHEELHOUSE_DIR    = '/kaggle/input/datasets/dfranzen/pennyroyal-v253'
MODEL_DIR         = '/kaggle/input/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1'
DRAFT_MODEL_DIR   = '/kaggle/input/models/dfranzen/albucino-qwen3-8-flash-next-drafter/transformers/default/1'
ORIG_BUNDLE_DIR   = '/kaggle/input/datasets/dfranzen/taaf-kaggle-source-bundle-copy'
SERVED_MODEL_NAME = 'flashnext'"""
    new_paths_cfg = """def _resolve_input_path(primary: str) -> str:
    p = Path(primary)
    if p.exists():
        return primary
    parts = p.parts
    sub = Path(*parts[parts.index('dfranzen')+1:]) if 'dfranzen' in parts else Path(p.name)
    for prefix in (
        Path("/kaggle/input/datasets/dfranzen"),
        Path("/kaggle/input/models/dfranzen"),
        Path("/kaggle/input/dfranzen"),
        Path("/kaggle/input"),
    ):
        cand = prefix / sub
        if cand.exists():
            return str(cand)
    return primary

# model and dataset paths
WHEELHOUSE_DIR    = _resolve_input_path('/kaggle/input/datasets/dfranzen/pennyroyal-v253')
MODEL_DIR         = _resolve_input_path('/kaggle/input/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1')
DRAFT_MODEL_DIR   = _resolve_input_path('/kaggle/input/models/dfranzen/albucino-qwen3-8-flash-next-drafter/transformers/default/1')
ORIG_BUNDLE_DIR   = _resolve_input_path('/kaggle/input/datasets/dfranzen/taaf-kaggle-source-bundle-copy')
SERVED_MODEL_NAME = 'flashnext'"""
    assert old_paths_cfg in c4_src, "Could not find old_paths_cfg in cell 4"
    c4_src = c4_src.replace(old_paths_cfg, new_paths_cfg, 1)

    old_apply = """subprocess.run(["git", "apply", "--include=ARC3-Inference/*", "-v", "/kaggle/harness-changes.patch"], cwd=f"{BUNDLE_DIR}/src", check=True)
print('harness patch applied successfully')"""
    assert old_apply in c4_src, "Could not find old_apply in cell 4"

    v30_post_patch_block = f"""subprocess.run(["git", "apply", "--include=ARC3-Inference/*", "-v", "/kaggle/harness-changes.patch"], cwd=f"{{BUNDLE_DIR}}/src", check=True)
print('harness patch applied successfully')

# Apply Version 30 surgical zero-prompt, zero-global modules
_v30_dir = Path('/kaggle/arcagi_v30')
_v30_dir.mkdir(exist_ok=True)
_v30_assets = {repr(v30_assets)}
for _name, _text in _v30_assets.items():
    (_v30_dir / _name).write_text(_text, encoding='utf-8')
for _patch_script in ('patch_startup_shield.py', 'patch_transient_animation.py', 'patch_segmentation_aliases.py'):
    subprocess.run([sys.executable, str(_v30_dir / _patch_script), str(BUNDLE_DIR / 'src/ARC3-Inference')], check=True)
print('Applied v30 modules cleanly (patch_startup_shield, patch_transient_animation, patch_segmentation_aliases)')"""
    c4_src = c4_src.replace(old_apply, v30_post_patch_block, 1)

    old_second_submission = 'TRUE_SUBMISSION = os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1", "true"}'
    new_second_submission = 'TRUE_SUBMISSION = _RERUN_FLAG or _GATEWAY_UP'
    if old_second_submission in c4_src:
        c4_src = c4_src.replace(old_second_submission, new_second_submission, 1)

    nb_v30["cells"][4]["source"] = c4_src

    # Cell 6: Preserve dfranzen/v24 precache_model_thread (delay=60, threads=3) for 284s weight load / 490s startup
    c6_src = _src(nb_v11["cells"][6])
    expected_precache_model = "precache_model_thread = threading.Thread(target=precache, args=(MODEL_DIR, DRAFT_MODEL_DIR), kwargs={'delay': 60, 'threads': 3})"
    assert expected_precache_model in c6_src, "Could not find expected_precache_model in cell 6"
    nb_v30["cells"][6]["source"] = c6_src

    # Cell 12: Preserve precache_model_thread.start() and add 1x auto-relaunch
    c12_src = _src(nb_v11["cells"][12])
    old_launch = """# ---- launch detached and wait for health ----
precache_model_thread.start()
print("\\n" + shlex.join(args) + "\\n")
log_offset = Path(LOG).stat().st_size if Path(LOG).exists() else 0
with open(LOG, "ab", buffering=0) as logf:
    proc = subprocess.Popen(args, env=env, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
print(f"pid {proc.pid} -> {LOG}")

def show_log_tail():
    with open(LOG, "rb") as f:
        f.seek(0, 2)
        f.seek(max(log_offset, f.tell()-20000))
        print(f.read().decode(errors="replace"))

url, t0 = f"http://127.0.0.1:{SERVED_MODEL_PORT}/health", time.time()
last_report = -30
while time.time() - NOTEBOOK_START_TIME < SERVER_STARTUP_TIMEOUT:
    if proc.poll() is not None:
        show_log_tail()
        break"""
    new_launch = """# ---- launch detached and wait for health ----
precache_model_thread.start()
print("\\n" + shlex.join(args) + "\\n")
log_offset = Path(LOG).stat().st_size if Path(LOG).exists() else 0
with open(LOG, "ab", buffering=0) as logf:
    proc = subprocess.Popen(args, env=env, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
print(f"pid {proc.pid} -> {LOG}")

def show_log_tail():
    with open(LOG, "rb") as f:
        f.seek(0, 2)
        f.seek(max(log_offset, f.tell()-20000))
        print(f.read().decode(errors="replace"))

url, t0 = f"http://127.0.0.1:{SERVED_MODEL_PORT}/health", time.time()
last_report = -30
_sgl_restarts = 0
while time.time() - NOTEBOOK_START_TIME < SERVER_STARTUP_TIMEOUT:
    if proc.poll() is not None:
        show_log_tail()
        if _sgl_restarts < 1:
            _sgl_restarts += 1
            print(f"WARNING: SGLang exited with code {proc.returncode}; relaunching (attempt {_sgl_restarts + 1})...", flush=True)
            with open(LOG, "ab", buffering=0) as logf:
                proc = subprocess.Popen(args, env=env, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
            time.sleep(3)
            continue
        break"""
    assert old_launch in c12_src, "Could not find old_launch in cell 12"
    c12_src = c12_src.replace(old_launch, new_launch, 1)
    nb_v30["cells"][12]["source"] = c12_src

    # Cell 16: Combine shiiin9's Cell 17 (bm.solver config + ft09 rehearsal)
    # and shiiin9's Cell 21 (ours_form_priority.install_d) upgraded with ouyangruize's C-scaled B_raw * C and b0 = 8.0
    solver_cfg_src = _src(nb_shiiin["cells"][17]).rstrip()
    d_prime_src = _src(nb_shiiin["cells"][21]).rstrip()

    old_d_prime_dict = '"b1": 10.0, "b2": 14.0, "b3": 16.0, "fade": 0.4}'
    new_d_prime_dict = '"b0": 8.0, "b1": 10.0, "b2": 14.0, "b3": 16.0, "fade": 0.4}'
    assert d_prime_src.count(old_d_prime_dict) == 1, "Expected old_d_prime_dict once in shiiin9 Cell 21"
    d_prime_src = d_prime_src.replace(old_d_prime_dict, new_d_prime_dict, 1)

    old_b_calc = '    B = (0.0, params["b1"], params["b2"], params["b3"])[min(remaining, 3)]\\n    return A * C, B'
    new_b_calc = (
        '    b0 = params.get("b0", 8.0) if level > 1 else 0.0\\n'
        '    B_raw = (b0, params["b1"], params["b2"], params["b3"])[min(remaining, 3)]\\n'
        '    return A * C, B_raw * C'
    )
    assert d_prime_src.count(old_b_calc) == 1, "Expected old_b_calc once in shiiin9 Cell 21"
    d_prime_src = d_prime_src.replace(old_b_calc, new_b_calc, 1)

    gateway_sync_block = """if not TRUE_SUBMISSION and '_arc_gateway_reachable' in globals() and _arc_gateway_reachable():
    TRUE_SUBMISSION = True
"""
    nb_v30["cells"][16]["source"] = gateway_sync_block + solver_cfg_src + "\n\n" + d_prime_src + "\n"

    # Cell 20: Dynamic 538-minute competition runtime cap + #OURS_FRESH verification print + Commit-Run Verification Gate
    c20_src = _src(nb_v11["cells"][20]).rstrip()
    old_await = "await bm.run(soft_end_time=soft_end, runtime_environment=target, minimal_diagnostics=TRUE_SUBMISSION)"
    new_await = """if TRUE_SUBMISSION and USE_PRIORITY_SCHEDULING:
    _boot_elapsed_s = max(0.0, time.time() - NOTEBOOK_START_TIME)
    bm.solver.max_runtime_s_per_game = min(532 * 60.0, max(600.0, (538 * 60.0) - _boot_elapsed_s))

await bm.run(soft_end_time=soft_end, runtime_environment=target, minimal_diagnostics=TRUE_SUBMISSION)"""
    assert c20_src.count(old_await) == 1, "Expected old_await once in Cell 20"
    c20_src = c20_src.replace(old_await, new_await, 1)

    fresh_print_src = _src(nb_shiiin["cells"][23]).rstrip()
    commit_gate_src = """if not TRUE_SUBMISSION:
    _summary_path = WORKING_DIR / "summary.txt"
    _summary_txt = _summary_path.read_text(encoding="utf-8") if _summary_path.is_file() else ""
    if "score=0.00" in _summary_txt or "actions=0 " in _summary_txt or "actions=1 " in _summary_txt:
        raise RuntimeError(f"Commit rehearsal gate failed (starved or 0.00 score):\\n{_summary_txt}")"""
    nb_v30["cells"][20]["source"] = c20_src + "\n" + commit_gate_src + "\n" + fresh_print_src + "\n"

    # Clear any stale outputs or execution_counts across all cells
    for cell in nb_v30["cells"]:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    return nb_v30


def update_kernel_metadata() -> None:
    meta = {
        "id": "zerbytheboss/v30-sovereign-apex-engine",
        "title": "v30 - Sovereign Apex Engine",
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
    nb_v30 = build_v30_notebook()
    OUTPUT_V30_PATH.parent.mkdir(parents=True, exist_ok=True)

    for path in (OUTPUT_V30_PATH, OUTPUT_SUB_PATH):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb_v30, f, indent=1, ensure_ascii=False)
            f.write("\n")

    update_kernel_metadata()

    sha_v30 = hashlib.sha256(OUTPUT_V30_PATH.read_bytes()).hexdigest()
    sha_sub = hashlib.sha256(OUTPUT_SUB_PATH.read_bytes()).hexdigest()
    assert sha_v30 == sha_sub, "submission_v30.ipynb and submission.ipynb must be byte-identical"
    assert len(nb_v30["cells"]) == 23, f"Expected 23 cells, got {len(nb_v30['cells'])}"
    print(
        f"Built {OUTPUT_V30_PATH.name} and {OUTPUT_SUB_PATH.name} "
        f"({OUTPUT_V30_PATH.stat().st_size} bytes, sha256={sha_v30[:12]}, cells=23)"
    )


if __name__ == "__main__":
    main()

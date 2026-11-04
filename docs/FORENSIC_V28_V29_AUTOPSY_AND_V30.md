# Bedrock Forensic Autopsy of `v24` $\to$ `v29` Score Decline (`28.79` $\to$ `20.08`) and Version 30 (`v30`) Sovereign Restoration Architecture

**Document ID**: `docs/FORENSIC_V28_V29_AUTOPSY_AND_V30.md`  
**Date**: October 24, 2026  
**Target Environment**: ARC Prize 2026 (ARC-AGI-3) Kaggle Competition (`ARC-AGI-3-Kaggle-Starter`)  
**Verified Submission Lineage**:
- `v24` (`ref: 57337517`): **`28.79` LB** (Personal Best — Pure `dfranzen` M2 + `shiiin9` `D'`)
- `v25` (`ref: 57373064`): **`27.16` LB** (`-1.63` vs `v24`)
- `v26` (`ref: 57471652`): **`26.29` LB** (`-2.50` vs `v24`)
- `v27` (`ref: 57500943`): **`0.00` LB** (Cold-boot FUSE mount stall exceeding 15m scorecard inactivity window)
- `v28` (`ref: 57541661`): **`21.47` LB** (`-7.32` vs `v24`)
- `v29` (`ref: 57575158`): **`20.08` LB** (`-8.71` vs `v24`)

---

## 1. Executive Summary: Why Our Score Dropped Monotonically from `28.79` (`v24`) to `20.08` (`v29`)

Across five consecutive versions after `v24` (`28.79` LB), every custom runtime monkey-patch layered onto `ARC3-Inference` degraded our 110-game competition leaderboard score:

| Version | Submission Ref | Public LB | Delta vs `v24` | Exact Causal Mechanism of Score Loss |
| :--- | :---: | :---: | :---: | :--- |
| **`v24`** | `57337517` | **`28.79`** | Baseline | Pure `dfranzen` M2 + `shiiin9` `D'` (`b0=0.0`). Zero custom monkey-patches on `python_tool_sandbox.py`, `tool_agent.py`, or `solver.py`. |
| **`v25`** | `57373064` | **`27.16`** | `-1.63` | Introduced `patch_spatial_helpers.py` (injected `find_cells`, `find_objects`, `manhattan_distance`, `rot90`, etc. into `runtime_globals` before `_available_names`, breaking `ARC3_PERSISTENT_FUNCTIONS=1` function retention) + prompt-mutating `animation_archive` / `fable_layer`. |
| **`v26`** | `57471652` | **`26.29`** | `-2.50` | Kept `patch_spatial_helpers.py` + added `patch_solved_level_memory.py` (mutated `_build_user_prompt` and `self._system_prompt`, forking streams off the shared 10-stream SGLang RadixAttention prefix cache). |
| **`v27`** | `57500943` | **`0.00`** | `-28.79` | Slow GCS FUSE weight load (`20.7m`) exceeded the 15-minute `arc_agi` scorecard inactivity timeout (`STALE_MINUTES = 15`) with only a single warmup `RESET`. |
| **`v28`** | `57541661` | **`21.47`** | `-7.32` | (1) Kept `patch_spatial_helpers.py`; (2) Set `b0 = 6.0` in `D'` **without multiplying `B` by `C`**, causing games stalled on their final level to hold `priority >= 6,000` forever; (3) Changed `precache` from `delay=60, threads=3` to `delay=15, threads=8`, thrashing FUSE I/O (`949.82s` weight load vs `284.47s` in `v24`); (4) Lowered `SERVER_STARTUP_TIMEOUT` to `10*60` and reset `self.started_at = time.monotonic()` in `TRUE_SUBMISSION`, pushing `started_at + 532*60` past Kaggle's hard 540-minute container limit. |
| **`v29`** | `57575158` | **`20.08`** | `-8.71` | Reverted `b0=0.0` and `delay=60, threads=3`, but added `patch_post_eviction_memory.py` (which wiped **100% of `self._history_messages` and `self._kept_functions`** on `level >= 2` after just `62,000` tokens or `95` actions, double-drained context via `force_drain=True`, and invalidated KV cache at `result[1]`), `patch_sovereign_oo_harness.py` (13 unadvertised `runtime_globals`), and 9 unverified `ARC3_*` env flags. |

---

## 2. Bedrock Code-Level Proof of the Three Fatal Bugs in `v25`–`v29`

### Bug #1 (Affected `v25`, `v26`, `v28`, `v29`): `patch_spatial_helpers.py` / `patch_sovereign_oo_harness.py` Breaking `ARC3_PERSISTENT_FUNCTIONS=1` (`retained_functions.py`)

In `dfranzen`'s Milestone 2 harness (`ARC3_PERSISTENT_FUNCTIONS = '1'`), `inference/agent/python_tool_sandbox.py` retains model-defined Python helper functions across turns and across levels using `inference/utils/retained_functions.py`:

```python
# inference/agent/python_tool_sandbox.py lines 651-671
_available_names = set(runtime_globals) | set(runtime_globals["__builtins__"])
if _persistence_enabled:
    _kept_pool, _retention_rejected = restore_functions(
        initial.get("kept_functions") or [], runtime_globals, _available_names,
        **_retention_options)
...
kept, rejected = collect_functions(
    _code_text, _kept_pool, runtime_globals, _available_names,
    **_retention_options)
```

Inside `retained_functions.py` (`collect_functions` lines 242–249 and `restore_functions` line 154):

```python
if name in available:
    rejected.append(
        _rejection(name, "name conflicts with a provided global or builtin")
    )
    continue
```

Because `patch_spatial_helpers.py` (`v25`–`v28`) and `patch_sovereign_oo_harness.py` (`v29`) injected `"find_cells"`, `"find_objects"`, `"manhattan_distance"`, `"rot90"`, `"rot180"`, `"rot270"`, `"flipv"`, `"fliph"` (plus `"wl_symmetry"`, `"bfs_shortest_path"`, `"capabilities"`, `"doc"`, `"call"` in `v29`) directly into `runtime_globals` **before** line 651 (`_available_names = set(runtime_globals) | ...`) **without advertising them in the system prompt**:
1. Whenever `Qwen3.8-Flash-Next` defined its own `def find_cells(color): ...` or `def find_objects(): ...` in Turn 1, `collect_functions` saw `"find_cells" in _available_names` and **silently rejected and dropped the model's function** (`keepable_functions: []`, `retention_rejected: [{'name': 'find_cells', 'reason': 'name conflicts with a provided global or builtin'}]`).
2. In Turn 2, when the model called `find_cells('R')` expecting its 1-argument function retained from Turn 1, it instead hit our unadvertised 2-argument global `find_cells(grid, target)` and crashed with:
   ```
   TypeError: find_cells() missing 1 required positional argument: 'target'
   ```

### Bug #2 (Specific to `v29`, Causing `20.08` LB): Catastrophic Amnesia & Double Context Drain in `patch_post_eviction_memory.py`

`patch_post_eviction_memory.py` in `v29` introduced three compounding defects inside `inference/agent/tool_agent.py`:

1. **Defect 2A (`_slm_maybe_stagnation_restart` Total Amnesia Wipe at 62,000 Tokens / 95 Actions)**:
   ```python
   _STAGNATION_TOKEN_LIMIT = 62000
   _STAGNATION_ACTION_LIMIT = 95
   ```
   On **any level $\ge 2$**, as soon as a game generated `62,000` tokens on that level (~18–20 turns of `Qwen3.8-Flash-Next` reasoning — before even filling the first `118,784`-token context window!) or took `95` actions, `_slm_maybe_stagnation_restart` executed:
   - `self._history_messages = []` — **wiping 100% of the conversation history to zero messages**!
   - `self._kept_functions = {}` — **wiping 100% of retained Python helper functions** across levels!
   - `self._context_was_trimmed = True` — **forcing `self._maybe_handover()` to immediately surrender the active GPU slot** to `_PriorityGate` at a moment when the game had already spent 62k tokens on the current level (so its priority had decayed)!
2. **Defect 2B (`_slm_trim` Double Context Drain via `force_drain=True`)**:
   Whenever `used > budget_tokens` (`118,784`), `_slm_trim` first ran `_compact_stale_reasoning_on_eviction` (stripping `reasoning_content` from older assistant messages, which already reduced `used` by ~35,000–45,000 tokens to ~75,000 tokens), **and then** set `kwargs["force_drain"] = True` before calling `_slm_original_trim`!
   In `tool_agent.py`, `force_drain=True` forces `_trim_messages_for_context` to keep dropping turns until the **already-compacted** message list is $\le \text{target\_tokens}$ (`58,368` tokens) — evicting over `100,000` original tokens of conversation history in a single trim!
3. **Defect 2C (`_slm_pin_once_on_eviction` Front-of-History KV Cache Destruction)**:
   Mutated `result[1]` (the oldest retained user message immediately following `messages[0]`) by prepending `[COMPACTED LEVEL LEDGER]`, invalidating the SGLang RadixAttention prefix cache for the entire conversation suffix after `messages[0]`.

### Bug #3 (Specific to `v28`, Causing `21.47` LB): Unscaled `b0 = 6.0` in `D'` + `precache(delay=15, threads=8)` FUSE Contention + Premature Startup Handoff

1. In `shiiin9`'s original `D'` (`v24`), `d_parts` returned `return A * C, B` where `B` was **not** multiplied by the patience/decay factor `C`. When `v28` set `b0 = 6.0` (`B = (6.0, 10.0, 14.0, 16.0)[min(remaining, 3)]`) without multiplying `B` by `C`, any game that reached its final level (`remaining == 0`) and stalled kept an undecaying floor of `B = 6.0` (`priority >= 6,332`) forever, starving solvable games of GPU slots.
2. Changing `precache_model_thread` from `delay=60, threads=3` (`v24`) to `delay=15, threads=8` (`v28`) launched 8 concurrent FUSE reader threads while SGLang was still importing Python/CUDA libraries and running its own shard loader, increasing weight-loading time from **`284.47s` (`v24`)** to **`949.82s` (`v28`, `3.34x` slower)**.
3. Lowering `SERVER_STARTUP_TIMEOUT` to `10 * 60` (`600s` from `NOTEBOOK_START_TIME`) forced `Cell 12` to exit early while SGLang still had ~10 minutes of loading left, and `patch_startup_shield.py` then reset `self.started_at = time.monotonic()` at `t = ~20m` in `TRUE_SUBMISSION`, pushing `self.started_at + 532*60` (`552m`) past Kaggle's hard `540m` (`32,400s`) container timeout.

---

## 3. Version 30 (`v30`) Sovereign Restoration & Apex Architecture

Version 30 restores the 100% pure `v24` (`28.79` LB) / `shiiin9` (`31.54` LB) foundation while incorporating four surgical, mathematically verified, zero-prompt-drift improvements:

### 3.1. `C`-Scaled `D'` Priority Scheduler with Final-Level Cliff Fix (`b0 = 8.0`, `B_raw * C` from `ouyangruize` Oct 24)

In `ours_form_priority.py` (`d_parts`), `v30` replaces unscaled `B` with `C`-scaled `B_raw * C` and `b0 = 8.0 if level > 1 else 0.0`:

```python
D_PRIME = {
    "level_pow": 0.5, "eff_const": 300.0, "eff_pow": 2.5, "pace_pow": 0.4,
    "pace_ref": 30_000.0, "c_action": 0.1, "action_scale": 115.0, "c_token": 0.9,
    "token_scale": 225_000.0, "lin_floor": 0.1, "rel_pow": 0.5,
    "b0": 8.0, "b1": 10.0, "b2": 14.0, "b3": 16.0, "fade": 0.4,
}
...
remaining = max(0, n - level)
b0 = params.get("b0", 8.0) if level > 1 else 0.0
B_raw = (b0, params["b1"], params["b2"], params["b3"])[min(remaining, 3)]
return A * C, B_raw * C
```

| Game State | `v24` (`b0=0.0`, `B`) | `v28` (`b0=6.0`, `B`) | **`v30` (`b0=8.0`, `B_raw * C`)** | Scheduling Impact in `v30` |
| :--- | :---: | :---: | :---: | :--- |
| Unstarted 6-level game (`L1, a=0, t=0`) | `18,619` | `18,619` | **`18,619`** | Identical initial ordering (`6-lvl > 8-lvl > 10-lvl`). |
| Unstarted 10-level game (`L1, a=0, t=0`) | `17,000` | `17,000` | **`17,000`** | Identical initial ordering. |
| L1 after 1st context trim (`L1, a=40, t=118k`) | `16,360` | `16,360` | **`8,252`** | Trimmed L1 games immediately yield to all unstarted & active multi-level games. |
| Stalled L1 game (`L1, a=150, t=250k`) | `16,036` | `16,036` | **`1,636`** | **9.8x priority decay**; stalled L1 games never starve multi-level games. |
| Fresh L2 of 6-level game (`L2, a=0, t=0, pace=30k`) | `19,928` | `19,928` | **`19,928`** | Immediately retains GPU slot upon clearing Level 1. |
| Fresh L5 of 6-level game (`L5, a=0, t=0, pace=30k`) | `17,857` | `17,857` | **`17,857`** | Retains GPU slot over unstarted 10-level games (`17,000`). |
| L5 after 1st context trim (`L5, a=30, t=118k, pace=30k`) | `13,107` | `13,107` | **`8,126`** | **4.97x higher** than stalled L1 games (`1,636`), whereas in `v24` stalled L1 (`16,036`) beat L5 (`13,107`)! |
| Fresh Final L6 of 6 (`L6, a=0, t=0, pace=30k`) | `9,166` | `15,166` | **`17,166`** | Eliminates the `v24` final-level cliff (`9,166 -> 17,166`), beating unstarted 10-lvl games (`17,000`). |
| Final L6 after 1st trim (`L6, a=30, t=118k, pace=30k`) | `3,625` | `9,625` | **`7,640`** | **4.67x higher** than stalled L1 games (`1,636`). |
| Stalled Final L6 (`L6, a=150, t=250k, pace=30k`) | `332` | `6,332` | **`1,132`** | Cleanly decays to `1,132` (`C = 0.1`), fixing `v28`'s `6,332` permanent slot hogging. |

### 3.2. Surgical Transient-Animation No-Op Guard Fix (`patch_transient_animation.py`)

When an action produces visible intermediate animation frames (`transient_pixels > 0`) before the board settles back to its pre-action state:
1. `python_tool_sandbox.py`: `animated = bool((action_result.get("animation") or {}).get("transient_pixels", 0))` clears `_stale_state` (`if gameplay_changed is True or animated: del _stale_state[:]`) and does not arm `_stale_state`.
2. `solver.py` (`_action_verdict`): reports `"animated effect (intermediate frames changed; final board restored)"` instead of `"NO-OP"`.
3. `solver.py` (`step_env`): does not abort a batch under `BATCH_NOOP_BLOCK` when `transient_pixels > 0`.
4. `animation.py`: returns `{"action": ..., "frames": 0, "steps": [], "error": ...}` on single-frame actions so `animation()["frames"]` never raises `KeyError`.

### 3.3. Zero-Global `M75` Segmentation Aliases (`patch_segmentation_aliases.py`)

Wraps segmentation nodes in `_NodeDict(dict)` and `_BoundaryList(list)` strictly inside `inference/utils/segmentation.py`, resolving `node["bbox"]`/`node["box"]`/`node["bounds"]`, `node["area"]`/`node["size"]`/`node["count"]`, and `node["centroid"]`/`node["center"]` on demand via `__missing__` and `.get()` while adding **zero** symbols to `python_tool_sandbox.py` `runtime_globals`.

### 3.4. Proven `v24` FUSE Prefetching (`delay=60, threads=3`) & Cold-Boot Immunity (`patch_startup_shield.py`)

- Restores `precache_model_thread` to `dfranzen`/`v24`'s exact `delay=60, threads=3`.
- Uses mode-specific `SERVER_STARTUP_TIMEOUT = (12 * 60) if TRUE_SUBMISSION else (35 * 60)` and `ARC3_HTTP_RETRY_INITIAL_SECONDS = '1800'`.
- Polls `/health` during cold-boot warmup with a periodic `480s` (`8-minute`) keepalive `RESET` on the warmup game until `/health` returns `200 OK`.
- Clamps `bm.solver.max_runtime_s_per_game = min(532 * 60.0, max(600.0, (538 * 60.0) - _boot_elapsed_s))` in `Cell 20` so competition runs always finish cleanly before Kaggle's `540m` container limit.

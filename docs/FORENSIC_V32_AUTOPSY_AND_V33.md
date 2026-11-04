# Forensic Autopsy of Version 32 (`26.03` LB) & Version 33 Architectural Blueprint

## 1. Official Leaderboard & Submission Ledger (`2026-10-28 UTC`)

| Version / Kernel | Ref | Date (UTC) | Public LB Score | Key Architecture |
| :--- | :--- | :--- | :--- | :--- |
| **Top #1 (`Yi-Chia Chen @ Fields Model`)** | — | `2026-10-27` | **`75.56`** | Proprietary multi-model / custom search engine |
| **Top #2 (`Tufa Labs`)** | — | `2026-10-27` | **`68.91`** | Private Tufa internal solver |
| **Top Public Fork (`romantamrazov`)** | — | `2026-10-21` | **`35.35`** | Unmodified `dfranzen` M2 (`v11`, best stochastic draw) |
| **Top Modified Public (`ondrewsha S9`)** | — | `2026-10-26` | **`32.16`** | `dfranzen` M2 + `HF` sandbox fixes + clock strip + `H1` eviction-only solved-level memory + `SCH` |
| **Top Priority Scheduler (`shiiin9`)** | — | `2026-10-13` | **`31.54`** | `dfranzen` M2 + $D'$ priority scheduler (`priority = A*M*C + B*phi`) |
| **Top Memory-Only (`vickyfym / lalalia9`)** | — | `2026-10-27` | **`31.46`** | `dfranzen` M2 + eviction-only `SOLVED_LEVEL_MEMORY_PATCH` (94.3% KV cache hit rate) |
| **Top Token-Eviction (`laimerbotrolenk`)** | — | `2026-10-26` | **`31.21`** | `dfranzen` M2 + `ARC3_HISTORY_ASSISTANT_TURNS=0` token-only history bounds + sandbox tools |
| **Top Guard Ablation (`yongjiehishi`)** | — | `2026-10-27` | **`31.05`** | `dfranzen` M2 + $D'$ priority scheduler; Finding #4: `ARC3_NOOP_REPEAT_GUARD=1` (+16 levels on 25 public games) |
| **Our Personal Best (`v24`)** | `57337517` | `2026-10-18` | **`28.79`** | `dfranzen` M2 + `shiiin9` $D'$ priority scheduler |
| **Our `v25`** | `57373064` | `2026-10-19` | **`27.16`** | `v24` + `patch_spatial_helpers.py` (broke `retained_functions.py`) |
| **Our `v26`** | `57471652` | `2026-10-21` | **`26.29`** | `v25` + `SOLVED_LEVEL_MEMORY` + `install_timeout_armor` |
| **Our `v27`** | `57500943` | `2026-10-22` | **`0.00`** | Cold-boot timeout without startup shield keepalive |
| **Our `v28`** | `57541661` | `2026-10-23` | **`21.47`** | Unscaled `b0=6.0` + `1e7` band (stalled final levels locked above unstarted) |
| **Our `v29`** | `57575158` | `2026-10-24` | **`20.08`** | `1e6` band + rule-asking prompt bloat + C1 stagnation amnesia wipe |
| **Our `v30`** | `57609842` | `2026-10-25` | `ERROR` | Rehearsal gate raised in `TRUE_SUBMISSION` when `KAGGLE_IS_COMPETITION_RERUN` unset |
| **Our `v31`** | `57672389` | `2026-10-26` | **`25.44`** | `v24` + $C$-scaled $B$ across all levels (caused premature L2/L3 demotion) |
| **Our `v32`** | `57709347` | `2026-10-27` | **`26.03`** | `v24` + $C$-scaled $b_0$ + `HF` #1-6 + pre-checked warmup `RESET` |
| **Our `v33`** | `57732483` | `2026-10-28` | `PENDING` (Rehearsal: `28.57`, `3/6` lvls, `119.69` tok/s) | `v24` + $C$-scaled $b_0$ + `ARC3_HISTORY_ASSISTANT_TURNS=0` + Eviction-Only `SOLVED_LEVEL_MEMORY` + Animation-Safe `ARC3_NOOP_REPEAT_GUARD=1` + Clock Strip |

---

## 2. Root-Cause Forensic Diagnosis of `v32` (`26.03` LB)

Comparing `v32` (`26.03` LB) against our personal best `v24` (`28.79` LB) and the four highest-scoring modified public kernels (`ondrewsha` `32.16`, `shiiin9` `31.54`, `vickyfym` `31.46`, `laimerbotrolenk` `31.21`, `yongjiehishi` `31.05`) reveals five concrete, remediable gaps:

1. **Premature Assistant-Turn Evictions (`ARC3_HISTORY_ASSISTANT_TURNS=150` vs `0`)**:
   - In `tool_agent.py` lines 6323–6349 (`_persistent_history_messages`), `_keep_recent_history_turns` runs *after* `_trim_messages_for_context` with `max_turns=150, drain_turns=30`.
   - As Daniel Franzen documented in `tool_agent.py` lines 276–288, each analyzer turn uses ~3 assistant messages (inspect, compute, act), so 150 assistant messages is only ~50 analyzer turns (~65k–85k tokens, leaving 30k–50k tokens of the 116k context window unused).
   - When `_keep_recent_history_turns` drops 30 assistant messages at 150 messages, it calls `self._note_history_evicted()`, which **both invalidates the SGLang RadixAttention KV cache prefix and increments `trim_count`** (forcing a premature GPU slot handover in `_maybe_handover()`).
   - Setting `'ARC3_HISTORY_ASSISTANT_TURNS': 0` (`laimerbotrolenk` `31.21` LB) disables the independent assistant-message cap so history is only evicted inside `_trim_messages_for_context` when tokens actually exceed the 116k token budget.

2. **Context Amnesia After 58k-Token Drains on Multi-Level Games (Missing Eviction-Only `SOLVED_LEVEL_MEMORY`)**:
   - Under `shiiin9`'s $D'$ priority scheduler (`Cell 16`), active multi-level games (`L2`, `L3`, `L4`, ..., and final levels via $b_0 = 8.0 \cdot C$) retain their active GPU slots across context trims because their continuation bonus $B \in \{10.0, 14.0, 16.0\}$ keeps their priority above unstarted games (`17,000..18,619`).
   - However, when `_trim_messages_for_context` drains 58,000 tokens of history on Level 2 or Level 3 in `v32`, the turns from Level 1 are deleted from `messages[1:]` and the agent loses its record of how Level 1 was solved.
   - Earlier, we misattributed `v26`'s drop (`26.29` LB) to `SOLVED_LEVEL_MEMORY`, when in fact `v26` was broken by `patch_spatial_helpers.py` (which injected 4 functions into `runtime_globals` before `_available_names`, breaking `retained_functions.py`) and `install_timeout_armor`.
   - `vickyfym/arc-agi-3-lalalia-final` (**`31.46` LB**) and `ondrewsha/arc-agi-3-stack-s9` (**`32.16` LB**) proved that **eviction-only `SOLVED_LEVEL_MEMORY`**—which never modifies `self._system_prompt` until `_trim_messages_for_context` or `_force_reduce_messages` already drops history (`len(result) < len(messages)`)—preserves a **94.3% prefix cache hit rate** while eliminating post-drain mechanics amnesia on multi-level games.

3. **Disabled `ARC3_NOOP_REPEAT_GUARD` (`yongjiehishi` `31.05` LB Finding #4: `+16` Levels on 25 Public Games)**:
   - `ARC3-Inference` includes `NoopRepeatGuard` (`inference/agent/noop_repeat_guard.py`), which records `(level, interior_state_hash, action_signature)` triples on Level $\ge 2$ (`ARC3_GUARDS_FROM_LEVEL=2`) that changed nothing inside the board interior and blocks repeating the exact same no-op action from the exact same board state (with a 1-slot override if the model immediately insists).
   - Franzen shipped `ARC3_NOOP_REPEAT_GUARD` disabled by default for one documented reason (`noop_repeat_guard.py` lines 21–25): actions whose effect appears only in intermediate animation frames (`transient_pixels > 0`) had `gameplay_changed=False` and would be falsely recorded as no-ops.
   - Because our `patch_transient_animation.py` already tracks `int(((item.get("animation") or {}).get("transient_pixels")) or 0) > 0`, updating `gameplay_changed_per_action` in `solver.py` and `_record_guard_observations` in `tool_agent.py` to treat `transient_pixels > 0` as `changed = True` completely eliminates false positives on animated actions and unlocks `ARC3_NOOP_REPEAT_GUARD=1` (+16 levels on 25 public games in `yongjiehishi`'s replicated ablation).

4. **Whole-Run Clock Exposure in `_compact_action_result` (`ondrewsha` `32.16` LB `HF_EDITS` #7)**:
   - In `tool_agent.py` lines 5702–5704, `_compact_action_result` copies `run_elapsed_seconds` and `time_remaining_seconds` into the compact dict returned by `action(...)` inside the Python sandbox.
   - Failure analysis (`ondrewsha` `HF_EDITS` #7) showed that the model reads `run_elapsed_seconds` and `time_remaining_seconds` from `action(...)` output as an in-game level timer. Stripping those two keys from `compact` (while still updating `self._game_elapsed_seconds` internally from `payload.get("run_elapsed_seconds")`) eliminates level-timer hallucinations and reduces action output tokens with zero change to the system prompt.

5. **Startup Timeout & `max_runtime_s_per_game` Alignment (`Cell 4`, `Cell 20`, `patch_startup_shield.py`)**:
   - In `v32`, `SERVER_STARTUP_TIMEOUT` was set to `23 * 60` (`1380s`) even in `TRUE_SUBMISSION` mode (`_IS_TRUE_RERUN=True`), overriding Franzen's `12 * 60` (`720s`) early-release deadline in Cell 12 that releases `bm.run()` before the 15-minute gateway idle timeout if SGLang is still loading.
   - Additionally, `v32` modified `bm.solver.max_runtime_s_per_game` in Cell 20 in `TRUE_SUBMISSION` mode and skipped the initial warmup `RESET` when `/health == 200 OK`.
   - In `v33`, we restore `SERVER_STARTUP_TIMEOUT = (12 * 60) if TRUE_SUBMISSION else (35 * 60)`, fire the initial warmup `RESET` unconditionally on the single warmup slot (`ARC3_WARMUP_ACTION_GAMES=1`) while keeping the periodic `480s` keepalive loop active whenever `/health` is still loading after `720s`, and preserve Cell 16's exact `bm.solver.max_runtime_s_per_game = 532 * 60` in `TRUE_SUBMISSION`.

---

## 3. Version 33 Architectural Specification (`scripts/build_v33_notebook.py`)

1. **Cell 2**: Unmodified 498,799-byte Franzen M2 (`v11`) harness patch (`SHA-256: 6fdcd1ae7445c0cfc26eec1047363aa0c196c867d289a35b6f64403b5a43dce2`).
2. **Cell 4**:
   - Mode-specific `SERVER_STARTUP_TIMEOUT = (12 * 60) if TRUE_SUBMISSION else (35 * 60)` and `ARC3_HTTP_RETRY_INITIAL_SECONDS = '2400'`.
   - Dual-mount dataset path resolution (`/kaggle/input/datasets/dfranzen/...` with fallback to `/kaggle/input/...`).
   - Three surgical post-patches applied to `/kaggle/taaf-kaggle-source-share/src/ARC3-Inference`:
     1. `patch_startup_shield.py`: Unconditional initial warmup `RESET` on the warmup slot (protected by `try/except`) + periodic `480s` keepalive `RESET` loop if `/health` is still loading after Cell 12's `720s` release + non-warmup thread `/health` probe + `started_at` reset for short rehearsal budgets (`<= 3600s`).
     2. `patch_transient_animation.py`: Full 4-layer transient-animation guard (`solver.py` verdict + batch stop + `gameplay_changed_per_action`, `tool_agent.py` turn trace + `None`-preserving `_record_guard_observations`, `python_tool_sandbox.py` stale-state check), `import time` in `SAFE_MODULES`, timeout action-count reporting, timeout `_kept_functions` preservation, and whole-run clock stripping (`run_elapsed_seconds`, `time_remaining_seconds`) from `_compact_action_result`.
     3. `patch_segmentation_aliases.py`: Zero-global $M75$ key aliases (`bbox`/`box`/`bounds`, `size`/`area`/`count`, `centroid`/`center`) on `_NodeDict` and `_BoundaryList` in `inference/utils/segmentation.py`.
   - `setup_env`:
     - `'ARC3_HISTORY_ASSISTANT_TURNS': 0` (token-only history eviction at 116k budget with 58k hysteresis drain).
     - `'ARC3_NOOP_REPEAT_GUARD': '1'` (animation-safe known-no-op repeat guard on Level $\ge 2$).
3. **Cell 12**: 10-stream Pennyroyal SGLang v2.5.3 + Albucino MTP-3 server (`MEMFRAC=0.96, MAXREQ=10, CUDAGRAPH_MAXBS=10, MAMBA_CACHE=60`) with 1x auto-relaunch protection.
4. **Cell 16**:
   - `shiiin9` $D'$ Priority Scheduler + $C$-scaled final-level $b_0$:
     $$b_0 = (8.0 \cdot C) \text{ if } \ell > 1 \text{ else } 0.0, \quad B = (b_0, 10.0, 14.0, 16.0)[\min(N - \ell, 3)]$$
   - Cache-Safe Eviction-Only `SOLVED_LEVEL_MEMORY` (`vickyfym` `31.46` LB pre-action `prev == level` `RESET`-filtered `_slm_level_actions` + 900-char winning-turn `reasoning` capture + backtick/multiline-safe `_slm_extract_rule_from_text` + `ondrewsha` `32.16` LB `H1` token-budget check & session-reset guard + `_trim_messages_for_context` & `_force_reduce_messages` eviction hooks).
5. **Cell 20**: Standard `v11`/`v24`/`shiiin9` benchmark runner + dual-mount offline environment fallback + commit rehearsal non-zero score gate.

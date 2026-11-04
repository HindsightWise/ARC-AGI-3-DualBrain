# Forensic Audit of Version 25 (`27.16` LB) vs. Version 24 (`28.79` LB) & October 20 Leaderboard Reconnaissance for Version 26

**Document Title**: Empirical Post-Mortem of Version 25 (`28.79 -> 27.16` LB), October 19–20 Public Kernel Reconnaissance (`lalalia9` `31.46–35.79` LB, `heliosli` V6 `32.60` LB), and Version 26 Sovereign Apex Specification  
**Date**: 2026-10-21  
**Target Milestone**: Version 26 (`v26`) Sovereign Apex Engine  
**Reference Artifacts**:
- `pulled_kernels/oct20_top/lalalia9_arc3-franzen-v19/arc3-franzen-v19.ipynb`
- `pulled_kernels/oct20_top/heliosli_try-merging-it-you-might-get-lucky-and-win-gold/try-merging-it-you-might-get-lucky-and-win-gold.ipynb`
- `pulled_kernels/oct20_top/sirikilohit_arc-agi-3-duck-18-1gc-submit/arc-agi-3-duck-18-1gc-submit.ipynb`
- `notebooks/submission_v24.ipynb` (`28.79` LB)
- `notebooks/submission_v25.ipynb` (`27.16` LB)

---

## 1. Ground-Truth Kaggle Competition Submission Ledger

| Engine Version | Kaggle Ref | Public LB Score | 10-Game Demo Mean | Core Architectural Configuration | Empirical Verdict |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **Version 11 (`v11`)** | Baseline | **`26.70`** | ~35.00 | Pure Daniel Franzen Milestone 2 (`franzen_v3`) + Pennyroyal SGLang v2.5.3 + Qwen3.8-Flash-Next W4A16 + Albucino MTP-3 (`MAXREQ=10`, `MEMFRAC=0.96`). | Clean M2 foundation. |
| **Version 18 (`v18`)** | Historical | `14.04` | ~20.00 | Added `ARC3_HISTORY_IMAGE_KEEP='0'` and `MEMFRAC='0.94'`. | Prefix cache invalidation & multi-turn visual amnesia (`-12.66` LB). |
| **Version 19 (`v19`)** | Historical | `24.30` | `35.18` | Restored immutable visual history and `MEMFRAC=0.96`. | Restored `94.19%` RadixAttention KV cache hit rate (`+10.26` LB over `v18`). |
| **Version 20 (`v20`)** | Historical | `17.34` | ~22.00 | Experimental multi-agent coordination branch. | Excessive coordination overhead (`-6.96` LB). |
| **Version 21 (`v21`)** | `57194042` | `25.18` | **`55.83`** | `AnimationArchive` + `fable_layer` + `patch_spatial_helpers` + `patch_model_checks` + `edge_meter` + raw `shiiin9` `D'` + `started_at` clock tampering. | Surged on 10 public games (`55.83`), but lost points on 110 hidden games due to prompt bloat and clock/timeout clamping. |
| **Version 23 (`v23`)** | `57276511` | `20.86` | `39.11` | Custom priority aging bonus + modified scheduler parameters. | Queue thrashing across 110 games (`-4.32` LB vs `v21`). |
| **Version 24 (`v24`)** | `57337517` | **`28.79`** | `47.62` (rehearsal) | **100% pure `v11` base + `shiiin9` continuous `D'` slot priority (`ours_form_priority.install_d`)**. Zero system/user prompt mutations, zero clock tampering. | **All-Time Personal Best (`+2.09` LB over `v11`, `+3.61` LB over `v21`)**. |
| **Version 25 (`v25`)** | `57373064` | `27.16` | `55.83` (equiv) | `v24` + `AnimationArchive` (`patch_animation_archive.py`) + `fable_layer` (`fable_layer_core.py`) + `patch_animation_empty.py` + `patch_spatial_helpers.py` + `install_timeout_armor`. | Regressed **`-1.63` LB** below `v24` (`28.79 -> 27.16`) while outperforming `v11` (`+0.46`) and `v21` (`+1.98`). |

---

## 2. Root-Cause Analysis: Why Version 25 Regressed `-1.63` LB Below Version 24

Cross-referencing our `v24` (`28.79` LB) vs. `v25` (`27.16` LB) ablation against the October 19–20 top-scoring public kernels (`lalalia9/arc3-franzen-v19` at `31.46–35.79` LB and `heliosli` V6 at `32.60` LB) isolates the two exact regressions introduced in `v25`:

### 2.1 Defect 1: `AnimationArchive` (`patch_animation_archive.py`) System Prompt Mutation & Per-Step IPC Overhead
1. **System Prompt Suffix Mutation (+720 chars)**:  
   Despite being categorized as "zero system prompt bloat" in earlier notes, `patch_animation_archive.py` directly slices `_build_system_prompt` in `inference/agent/tool_agent.py`:
   ```python
   # From patch_animation_archive.py:
   return base_prompt[:-1] + (
       "- When `ARC3_ANIMATION` is enabled, the harness also keeps a per-game archive of multi-frame action animations in memory (up to 32 animated actions, capped at 4 MiB compressed). "
       "Use `animation_history()` in Python to list archived animated steps ... "
       "Use `animation_at(step, start=0, limit=32)` in Python to load the archived frames ...\n"
   ]
   ```
   This injects ~720 characters into the system prompt of all 110 games from Turn 1 and alters Qwen3.8-Flash-Next's initial tool-calling distribution toward writing archive inspection scripts on games where `last_animation_frames` already suffices.
2. **Per-Turn IPC Serialization Overhead**:  
   `patch_animation_archive.py` hooks `runtime_state_payload` and `_run_python_tool` on every step to serialize compressed archive payloads across the subprocess stdin/stdout pipe.
3. **Independent Verification from `heliosli` V6 (Oct 20, `32.60` LB)**:  
   In `pulled_kernels/oct20_top/heliosli_try-merging-it-you-might-get-lucky-and-win-gold/try-merging-it-you-might-get-lucky-and-win-gold.ipynb`, `heliosli` (the original author of `animation_archive.py` in V5) **completely removed `animation_archive.py` and `patch_animation_archive.py`**, retaining only the lightweight schema fix `patch_animation_empty_v5.py`.

### 2.2 Defect 2: `fable_layer` (`fable_layer_core.py`) Per-Turn User Prompt Interference on Hidden Games
1. **Redundancy and Conflict with Native `dfranzen` M2 Diff Hints**:  
   `dfranzen` Milestone 2 (`Cell 2` + `Cell 4`) already runs with native perceptual diffing enabled:
   - `ARC3_FRAME_DIFF_HINT = '1'`
   - `ARC3_EXPLAIN_GAMEPLAY_CHANGED = '1'`
   - `ARC3_NEW_CHANGED_PROMPTS = '1'`
   - `ARC3_DIFF_IMAGE = '1'`
   Appending `fable_layer`'s `Indicator note:` and `Step budget (read from the frames by the harness):` strings to per-turn user prompts duplicates and competes with the native frame-diff prompt structure.Furthermore, `detect_budget()` in `fable_layer_core.py` heuristically guesses whether any 1-pixel edge row/column is an action-budget bar, emitting false-positive budget warnings on hidden games whose border pixels represent walls, conveyors, or color keys.
2. **Independent Verification from `lalalia9` (`31.46–35.79` LB, ARC Prize 2026 Paper Track)**:  
   In *"Luck, Tokens and Memory: What Moves a Single-GPU LLM Agent on ARC-AGI-3"* (`pulled_kernels/oct20_top/lalalia9_arc3-franzen-v19/arc3-franzen-v19.ipynb`, Cell 0), `lalalia9` proved across repeated 110-game leaderboard evaluations that altering `dfranzen`'s per-turn user prompts looks positive or neutral on the 10 public demo games but degrades performance on the 110 hidden evaluation games. Keeping `dfranzen`'s per-turn prompts 100% unchanged and adding **only** cache-safe solved-level memory (`SOLVED_LEVEL_MEMORY_PATCH`) moved the score to **`31.46` LB** (and up to `35.79` LB on top draws).

---

## 3. Version 26 (`v26`) Sovereign Apex Architecture

Version 26 restores 100% of our `28.79` LB Version 24 substrate (zero `AnimationArchive`, zero `fable_layer`, byte-identical `v24` system and user prompts on Levels 1–2) and integrates **Cache-Safe Solved-Level Memory (`patch_solved_level_memory.py`)** alongside our three zero-prompt-overhead runtime protections:

### 3.1 Cache-Safe Solved-Level Memory (`patch_solved_level_memory.py`)
Adapted from `lalalia9/arc3-franzen-v19` Cell 17 and `sirikilohit` `M85` (`+2.8` LB verified gain), engineered with a mid-turn yield compatibility guard for `D'`:

1. **The Context Drain Amnesia Problem in `v24`**:  
   In `dfranzen` M2, each game maintains a single rolling conversation (`_history_messages`) with a `116k` input token budget (`LOCAL_ANALYZER_CONTEXT_WINDOW = (116+12)*1024`). When the conversation fills this window (typically around Level 3), `_trim_messages_for_context` drains ~59k tokens (`ARC3_CONTEXT_DRAIN_TOKENS = 58*1024`) from the head of `_history_messages`. Once this drain occurs, every turn from Levels 1 and 2 is permanently evicted—forcing the model on Levels 3–8 to re-discover control mappings, object roles, and invariants from scratch.
2. **Fact & Rule Capture on Level Completion (`_slm_before_turn`)**:  
   - At the start of `ToolAgent.analyze(self, state_path, ...)`, `_slm_before_turn` checks the current level (`curr_done = current - 1`).
   - Whenever a level clears (`curr_done > self._slm_prev_done` / `missing` levels in `1..current-1`), it extracts from `runtime_state.json` and `_history_messages`:
     - `level`: cleared level number
     - `n_actions`: total actions spent on the winning attempt of that level (resetting on `RESET`)
     - `actions` (`action_seq`): the last `30` actions (`_SLM_ACTIONS = 30`) of the winning attempt
     - `reasoning` (`assistant_tail`): the last `900` characters (`_SLM_REASONING_CHARS = 900`) of the assistant's winning reasoning
     - `asks_left = 2` (`_SLM_ASK_TURNS = 2`), `rule = None`
3. **Yield-Safe Rule Elicitation (`_slm_user_prompt`)**:  
   - On the next 2 full turns (`asks_left > 0` and `not rec.get("rule")`), `_slm_user_prompt` appends:
     ```text
     Before your tool call, write a line starting `Rule for level N:` that states the goal, what each control did and the plan that won (at most 5 lines). It is kept for the rest of the game.
     ```
   - **Critical Compatibility Guard with `D'` (`LOCAL_ANALYZER_YIELD_TOKENS = '2048'`)**:  
     In `ToolAgent.analyze`, `self._build_user_prompt(...)` is invoked at line 6588 *before* `self._resume_after_yield` is cleared at line 6638. When a turn yields mid-deliberation after hitting `2048` tokens without executing an environment action, `self._resume_after_yield` is `True` on the next `analyze()` call. `_slm_user_prompt` explicitly guards:
     ```python
     if getattr(self, "_resume_after_yield", False):
         return prompt
     ```
     This guarantees that a transient mid-turn yield/resume never decrements `asks_left` and never attaches the rule request to a resumption prompt.
   - **Rule Extraction (`_slm_find_rule` / `_slm_capture_from_text`)**:  
     Extracts `Rule for level N:` from the assistant's `content`, `reasoning_content`, or `reasoning`, cleanly stripping optional markdown bold/code formatting (`**Rule for level 1:**`, `` `Rule for level 1:` ``, or `Rule for level 1 - ...`).
4. **Strict RadixAttention Prefix-Cache Invariance (`_slm_drained_once`)**:  
   - Before the first context drain (`self._slm_drained_once == False`), `self._system_prompt` is **never modified**. Why? Because before the first drain, the turns that solved Level 1–2 and the model's `Rule for level N:` statement are already present verbatim in `self._history_messages`! Leaving `self._system_prompt` untouched preserves 100% of the SGLang RadixAttention prefix cache across every turn of Levels 1–2.
   - Only when `_note_history_evicted` / `_trim_messages_for_context` actually drops old messages from `_history_messages` (which already forces SGLang to re-prefill the trimmed conversation from the system prompt boundary) does `_slm_drained_once` become `True` and `_slm_refresh_system_prompt` pin the `=== SOLVED LEVELS (#SOLVED_LEVEL_MEMORY: ...) ===` block to the end of `self._system_prompt` and `messages[0]["content"]`.

### 3.2 Retained Zero-Prompt-Overhead Protections
1. **Continuous Expected-Value `D'` Slot Priority (`ours_form_priority.install_d`)**:  
   Preserved 100% from `v24` (`28.79` LB). Prices unstarted games by `(level=1, actions=0, tokens=0, total_levels=N)` (`N=6 -> 18619`, `N=8 -> 17527`, `N=10 -> 17000`) and applies `ARC3_PRIORITY_TAIL_FADE='1'` (`0.2` fraction) so late-game closing levels (`L6/6 -> 9860 > L5/6 -> 8451 > L1/6 -> 2619`) are never starved by fresh Level 1 games.
2. **Empty-Animation Schema Guard (`patch_animation_empty.py`)**:  
   Adds `"action"`, `"frames": 0`, and `"steps": []` to the single-frame empty animation return dictionary in `inference/utils/animation.py` with zero prompt changes.
3. **Zero-Prompt-Bloat REPL Spatial Helpers (`patch_spatial_helpers.py`)**:  
   Injects `rot90`, `rot180`, `rot270`, `flipv`, `fliph`, `manhattan_distance`, `find_cells`, and `find_objects` into `runtime_globals` in `inference/agent/python_tool_sandbox.py` with zero tokens added to the system or user prompts.
4. **Clamped ReadTimeout Floor (`install_timeout_armor(min_floor=120.0)`)**:  
   Wraps `_HarnessGameSession.request_timeout_seconds` in `inference/framework/solver.py` so that when `cand is not None`, the HTTP read timeout is floored at `max(120.0, float(cand))` (`ARC3_MIN_READ_TIMEOUT = '120.0'`), preventing premature `13.9s–89.4s` socket drops when soft remaining time narrows while returning `None` when unbounded.

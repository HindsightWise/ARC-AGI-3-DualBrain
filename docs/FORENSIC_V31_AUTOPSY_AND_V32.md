# Forensic Autopsy of Version 31 (`25.44` LB) & Architecture of Version 32 (`v32-sovereign-apex-engine`)

**Date:** 2026-10-27  
**Competition:** ARC Prize 2026 — ARC-AGI-3 (`arc-prize-2026-arc-agi-3`)  
**Target Kernel:** `zerbytheboss/v32-sovereign-apex-engine` (`NvidiaRtxPro6000`, Offline Competition Execution)

---

## 1. Complete Submission Ledger & Empirical Score Trajectory

| Version | Ref ID | Submitted (UTC) | Status | Public LB Score | Key Architectural Delta |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `v23` | `57315642` | `2026-10-17 15:04` | `COMPLETE` | `20.86` | Fair-play priority gate + Level 1 guards + `EXPOSE_RESET` + world model memory |
| **`v24`** | **`57337517`** | **`2026-10-18 03:57`** | **`COMPLETE`** | **`28.79` (PB)** | Pure Franzen M2 (`v11`) + `shiiin9` $D'$ (`B = (0, 10, 14, 16)` unscaled) |
| `v25` | `57373064` | `2026-10-19 00:01` | `COMPLETE` | `27.16` | Spatial globals injected into `runtime_globals` (broke `retained_functions.py`) |
| `v26` | `57471652` | `2026-10-21 07:29` | `COMPLETE` | `26.29` | `install_timeout_armor` overwrote deadline clamping + M85 solved-level memory |
| `v27` | `57500943` | `2026-10-22 00:00` | `COMPLETE` | `0.00` | Cold-boot timeout without startup shield keepalive |
| `v28` | `57541661` | `2026-10-23 00:01` | `COMPLETE` | `21.47` | Unscaled `b0=6.0` + `1e7` band (stalled final levels locked above unstarted) |
| `v29` | `57575158` | `2026-10-24 00:00` | `COMPLETE` | `20.08` | `1e6` band + rule-asking prompt bloat + C1 stagnation amnesia wipe |
| `v30` | `57609842` | `2026-10-25 00:00` | `ERROR` | — | Rehearsal gate raised in `TRUE_SUBMISSION` when `KAGGLE_IS_COMPETITION_RERUN` unset |
| **`v31`** | **`57672389`** | **`2026-10-26 16:28`** | **`COMPLETE`** | **`25.44`** | Restored `v24` substrate + `B_raw * C` across all levels (`-3.35` vs `v24` PB) |
| **`v32`** | **`57709347`** | **`2026-10-27 17:51`** | **`PENDING`** | *Running (`ft09=47.62`)* | Restored `shiiin9` unscaled $B$ (`rem >= 1`) + $b_0 \cdot C$ (`rem == 0`) + timeout helper retention |

---

## 2. Bedrock Root-Cause Autopsy of Why `v31` Scored `25.44` (`-3.35` vs `v24` `28.79` / `-6.10` vs `shiiin9` `31.54`)

### Root Cause 1: `B_raw * C` Across All Levels Destroyed Multi-Level Continuation (`L2`–`L5`)

In `shiiin9` (`31.54` LB) and `v24` (`28.79` LB), `ours_form_priority.py` (`install_d`) prices every game—including all 110 unstarted games (`price_fresh=True`)—with:
$$\text{priority} = (A \cdot M \cdot C + B \cdot \phi) \times 1000, \quad B = (0.0, 10.0, 14.0, 16.0)[\min(N - \ell, 3)]$$

In `shiiin9`'s evolutionary calibration (`data/datarun/refine-fine`), **$B = 16,000$ is the shared additive baseline** across both unstarted games (`level=1, a=0, t=0`, where $C=1.0$ and $A_1 \in [1.0, 2.619]$) and active multi-level games (`remaining >= 3`). Consequently, **`17,000` ($16,000 + 1,000$) is the exact dividing threshold ($A \cdot M \cdot C = 1.0$)**:
1. **Unstarted games (`L1, a=0, t=0`)**:
   - $N=6$: `18,619`
   - $N=8$: `17,527`
   - $N=10$: `17,000`
2. **Stalled Level 1 games after 1 context trim (`L1, a=35..40, t=118k`)**:
   - Under `shiiin9` / `v24` (with **unscaled** $B=16.0$):
     - $N=6$: $2.619 \times 0.377 + 16.0 \to$ **`16,988 < 17,000`**
     - $N=8$: **`16,576 < 17,000`**
     - $N=10$: **`16,360 < 17,000`**
     - Deep stall (`a=150, t=250k`): **`16,036 < 17,000`**
   - Crucially, in `shiiin9` / `v24`, **stalled Level 1 games ALREADY drop strictly below `< 17,000` at their very first context trim**, yielding their GPU slot to every unstarted game without scaling $B$ by $C$!
3. **Progressing multi-level games (`L2, L3, L4`)**:
   - Under `shiiin9` / `v24` (with **unscaled** $B$):
     - `prog_L2_N6 (a=10, t=20k)`: **`19,518`** (`> 18,619`)
     - `prog_L3_N6 (a=15, t=30k)`: **`20,112`** (`> 18,619`)
     - `trim1_L2_N6 (a=25, t=75k)`: **`18,181`** (`> 17,527`)
     - `trim1_L3_N6 (a=20, t=65k)`: **`19,345`** (`> 18,619`)
     - `prog_L2_N10 (a=10, t=20k)`: **`17,343`** (`> 17,000`)
     - `prog_L3_N10 (a=15, t=30k)`: **`17,570`** (`> 17,527`)
   - Under **`v31`** (where `B_raw * C` multiplied $16,000$ by $C < 1.0$):
     - `prog_L2_N10 (a=10, t=20k)`: collapsed from `17,343` $\to$ **`15,801 < 17,000`**
     - `prog_L3_N10 (a=15, t=30k)`: collapsed from `17,570` $\to$ **`15,374 < 17,000`**
     - `trim1_L2_N6 (a=25, t=75k)`: collapsed from `18,181` $\to$ **`13,033 < 17,000`**
     - `trim1_L3_N6 (a=20, t=65k)`: collapsed from `19,345` $\to$ **`14,510 < 17,000`**

Because `v31` multiplied `B_raw` by `C` for `remaining >= 1`, **every active game on Level 2, Level 3, or Level 4 was immediately evicted at its first context trim and parked behind all ~100 unstarted games**, destroying multi-level continuation and thrashing SGLang's RadixAttention KV cache across 100 intervening games.

#### The Exact `v32` Mathematical Fix (`b0 * C` Strictly for Final Level `remaining == 0, level > 1`)
In `v32`, `b1 = 10.0, b2 = 14.0, b3 = 16.0` (`remaining >= 1`) are restored **100% unscaled** (identical to `shiiin9` `31.54` LB and `v24` `28.79` LB), while $C$-scaling is applied **strictly to `b0` (`remaining == 0, level > 1`)**:
```python
    remaining = max(0, n - level)
    b0 = (params.get("b0", 8.0) * C) if level > 1 else 0.0
    B = (b0, params["b1"], params["b2"], params["b3"])[min(remaining, 3)]
    return A * C, B
```
This satisfies every scheduling invariant simultaneously:
- **100% `shiiin9` / `v24` equivalence for all `remaining >= 1` (`L1..L_{N-1}`)**:
  - Unstarted games: `N=6 -> 18,619`, `N=8 -> 17,527`, `N=10 -> 17,000`.
  - Progressing multi-level games (`L2, L3, L4`): `17,343..20,112 > 17,000` (retain GPU slots across context trims!).
  - Trimmed/stalled Level 1–3 games: `16,036..16,988 < 17,000` (yield GPU slots to unstarted and progressing games!).
- **Final-level cliff elimination & stall decay (`remaining == 0, level == N > 1`)**:
  - Fresh final level (`L6` of 6, `a=0, t=0, pace=30k`): **`17,166 > 17,000`** (beats unstarted 10-level games and all stalled games, fixing `v24`'s `9,166` cliff when carry-over history triggers an early trim on `L6`).
  - Final level after 1 full window (`L6` of 6, `a=30, t=118k, pace=30k`): **`7,640`** (yields slot until endgame tail fade $\phi \to 0$).
  - Stalled final level (`L6` of 6, `a=150, t=250k, pace=30k`): **`1,132`** (never monopolizes a slot).
  - Single-level games (`N=1, level=1`): `b0 = 0.0` $\to$ **`2,619`**.

---

### Root Cause 2: Sandbox Timeout Helper-Retention Wipe & Missing `"time"` in `SAFE_MODULES`

Forensic inspection of `v31`'s transcript (`kaggle_output/v31/transcripts/ft09-0d8bbf25_p0.txt`) and `ondrewsha`'s Oct 26 failure-mode audit (`arc-agi-3-stack-s9`) revealed three zero-prompt harness bugs in `ARC3-Inference`:
1. **Timeout wipes all retained functions (`tool_agent.py` `_record_retained_functions`)**:
   When a Python snippet times out after 30s (`Tool timed out after 30s`), `run_sandboxed_python` returns a dict without `"keepable_functions"`. In `tool_agent.py`, `_record_retained_functions` iterated over `sandbox_result.get("keepable_functions") or []` ($\to `[]`$) and overwrote `self._kept_functions = {}`, destroying 100% of helper functions retained from earlier turns and forcing the model to waste 5,000–15,000 tokens redefining them.
   - **`v32` Fix**: If `isinstance(sandbox_result, dict) and "keepable_functions" not in sandbox_result and self._kept_functions`, preserve `self._kept_functions` unchanged and report that earlier retained functions are still valid.
2. **`import time` blocked in `python_tool_sandbox.py` (`SAFE_MODULES`)**:
   When the model writes `import time` to bound its own BFS/DFS search inside the 30s tool timeout (`t0 = time.time()`), `_safe_import` raised `ImportError: Module 'time' is not allowed in the sandbox`, aborting the entire turn.
   - **`v32` Fix**: Add `"time"` to `SAFE_MODULES` in `python_tool_sandbox.py` (injecting zero symbols into `runtime_globals`, so `_available_names` and `retained_functions.py` remain 100% unpolluted) and report how many actions had already executed before a timeout (`f"{len(host_action_results)} action(s) had already executed..."`).
3. **Turn-trace animated verdict in `tool_agent.py` (`_summarize_executed_action_items`)**:
   In `v31`, `patch_transient_animation.py` patched `solver.py` and `python_tool_sandbox.py` for `transient_pixels > 0`, but missed `tool_agent.py`'s `_summarize_executed_action_items`, which still formatted animated actions (`gameplay_changed is False` with `transient_pixels > 0`) as `NO-OP (executed, no change inside the board area)` in the conversation turn trace.
   - **`v32` Fix**: Patch `_summarize_executed_action_items` in `tool_agent.py` to report `"animated effect (intermediate frames changed; final board restored)"` when `transient_pixels > 0`.

---

### Root Cause 3: Pre-Reset Health Check in `patch_startup_shield.py` & 40-Minute Cold-Boot Grace

In `v31`'s commit log (`kaggle_output/v31/v31-sovereign-apex-engine.log`), GCS FUSE weight precaching took `734.15s` and SGLang became `READY after 973s` (`1064.3s = 17.7m` from `NOTEBOOK_START_TIME`).
- In `v31`, `patch_startup_shield.py` unconditionally called `self._execute_auto_reset()` on the warmup game before checking if `/health` was already `200 OK`, wasting 1 action on Game 0 whenever SGLang finished loading inside Cell 12.
- **`v32` Fix**:
  1. In `patch_startup_shield.py`, check `/health == 200 OK` *before* firing the initial warmup `RESET`. If SGLang is already healthy, set `self.solver._analyzer_ready = True` immediately with zero wasted actions. If SGLang is still loading, fire the initial warmup `RESET` and poll `/health` with periodic `480s` keepalive `RESET`s until `/health == 200 OK`.
  2. Increase `ARC3_HTTP_RETRY_INITIAL_SECONDS` from `'1800'` to `'2400'` (40 minutes) so even 20+ minute cold mounts never hit an HTTP retry timeout.

---

## 4. Live Kaggle RTX Pro 6000 Commit Rehearsal & Competition Submission (`ref=57709347`)

| Metric | `v31` Rehearsal (`kaggle_output/v31/summary.txt`) | `v32` Rehearsal (`kaggle_output/v32/summary.txt`) | Delta |
| :--- | :--- | :--- | :--- |
| **Game** | `ft09-0d8bbf25` (5m 0s cap) | `ft09-0d8bbf25` (5m 0s cap) | — |
| **Score** | `28.57` | **`47.62`** | **`+19.05` (`+66.7%`)** |
| **Levels Solved** | `3.0 / 6` | **`4.0 / 6`** (`per-level=7/43,8/12,14/23,21/28,2/65,0/37`) | **`+1.0` level** |
| **Total Actions** | `58` | **`52`** | `-6` actions |
| **Generated Tokens** | `39,405` (`131.26 tok/s`) | **`40,065` (`133.44 tok/s`)** | `+2.18 tok/s` |
| **30s Tool Timeouts** | `1` (wiped retained helpers) | **`0`** | `-1` timeout |
| **Cold-Boot SGLang Ready** | `973s` (`1064.3s` total) | `1384s` (`1509.3s` total; `2400s` grace verified) | Protected |
| **Competition Submission** | `ref=57672389` (`25.44` LB) | **`ref=57709347`** (`SubmissionStatus.PENDING`) | Submitted `2026-10-27 17:51 UTC` |


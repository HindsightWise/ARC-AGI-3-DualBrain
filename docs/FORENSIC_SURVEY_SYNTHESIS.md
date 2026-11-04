# Ground-Truth Forensic Survey & Clean Baseline Synthesis

## 1. Executive Summary
Following a clean-slate re-examination of the ARC Prize 2026 (`arc-prize-2026-arc-agi-3`) competition, three independent Survey Explorers investigated:
1. Live Kaggle competition status, forum discussions, and leaderboard dynamics.
2. Historical telemetry and regression points from Personal Best Version 24 (28.79 LB) through Version 30 (ERROR).
3. Public kernel anatomy and top-performing community architectures.

The findings establish universal consensus: all custom monkeypatches and memory extensions introduced in `v25`–`v30` degraded performance by violating core execution invariants. Returning to the pristine Franzen v11 base patch combined with Shiiin9/Ouyang Ruize continuous D' slot priority scheduling and 4-layer cold-boot 0.00 immunity restores the optimal state-of-the-art foundation.

## 2. Regression Autopsy of v25–v30
| Version | LB Score | Primary Modifications | Root Cause of Degradation |
|---------|----------|----------------------|---------------------------|
| **v24** | **28.79 (PB)** | Pristine Franzen v11 base + Shiiin9 D' priority | Baseline PB. Clean execution, 300.4s exact rehearsal stop. |
| **v25** | 27.16 | `install_timeout_armor(min_floor=120s)`, spatial helpers | Timeout armor caused games near deadline to block slots; spatial helpers broke `retained_functions.py` persistence. |
| **v26** | 26.29 | `SOLVED_LEVEL_MEMORY_PATCH` asking rules | Wasted 16.5k tokens, inflated FormPace by +113%, penalized D' priority by -2550 pts, mutated system prompt. |
| **v27** | 0.00 | Removed timeout armor, unscaled D' priority | 0.00 watchdog kill due to cold-boot timeout cap at 10-12m during GCS FUSE degradation. |
| **v28** | 21.47 | Added keepalive warmup, D' unscaled $b_0=6.0$ | Stalled final-level games permanently hogged slots; 8-thread prefetching caused 3.34x FUSE loading contention. |
| **v29** | 20.08 | `patch_post_eviction_memory` amnesia wipe | Erased 100% of history and retained helpers after 62k tokens, causing catastrophic context loss. |
| **v30** | ERROR | Custom packaging / startup script changes | Container launch failure / submission error. |

## 3. The Clean-Slate State-of-the-Art Architecture (Version 31)
1. **Pristine Base Patch Bedrock**: Franzen M2 / v11 Cell 2 patch (498,799 bytes, SHA-256 `6fdcd1ae7445c0cfc26eec1047363aa0c196c867d289a35b6f64403b5a43dce2`).
2. **SGLang Engine Settings**: `MEMFRAC=0.96`, `MAXREQ=10`, `ARC3_MAX_ACTIVE_STREAMS=10`, `total_deadline=532*60`.
3. **Continuous $D'$ Slot Priority Scheduler**: Continuous $A \cdot M \cdot C + B \cdot \phi$ with Ouyang Ruize Oct 24 patience decay $C$ and fresh final-level entry ($b_0 = 8.0 \text{ if } \ell > 1 \text{ else } 0.0$). Unstarted games priced via `PrioritySnapshot(1, 0, 0, N)`.
4. **Frozen System Prefix Invariant**: `self._system_prompt` (`messages[0]`) remains 100% immutable across all turns and context trims, ensuring 100% shared `<tools>` RadixAttention prefix caching across all 10 concurrent streams.
5. **Retained Functions Immunity**: Zero names injected into `runtime_globals` before `_available_names`, ensuring `retained_functions.py` (`ARC3_PERSISTENT_FUNCTIONS=1`) operates without `TypeError`.
6. **4-Layer Cold-Boot 0.00 Immunity**:
   - `delay=60, threads=3` prefetcher (284s load vs 950s with threads=8).
   - Mode-specific startup timeout + 30m HTTP grace.
   - `patch_startup_shield.py` periodic 480s keepalive `RESET` on warmup game until `/health == 200 OK`.
   - Cell 20 Hard Commit Rehearsal Gate (5m `ft09` run).

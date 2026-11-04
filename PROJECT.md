# Project: ARC-AGI-3 Sovereign Masterpiece Solver (Version 34)

## Architecture
- **Inference Runtime Engine**: Pennyroyal SGLang v2.5.3 + Qwen3.8-Flash-Next W4A16 AutoRound + Albucino MTP-3 drafter (`MEMFRAC=0.96`, `MAXREQ=10`, `ARC3_MAX_ACTIVE_STREAMS=10`, `total_deadline=532*60`).
- **Base Patch Bedrock**: Franzen M2 / v11 Cell 2 patch (498,799 bytes, SHA-256 `6fdcd1ae7445c0cfc26eec1047363aa0c196c867d289a35b6f64403b5a43dce2`).
- **Scheduling & Priority Governance**: Continuous $D'$ Slot Priority Scheduler ($A \cdot M \cdot C + B_{\text{raw}} \cdot C \cdot \phi$) with Ouyang Ruize patience decay ($C$) and fresh final-level entry ($b_0 = 8.0 \cdot C \text{ if } \ell > 1 \text{ else } 0.0$), ensuring zero slot-hogging by stalled games.
- **Prefix Cache & KV Protection**: 100% frozen `self._system_prompt` (`messages[0]`) across all turns, levels, and context drains, preserving 100% shared RadixAttention `<tools>` prefix cache across all 10 concurrent streams.
- **Completion-First Engine (Probe-First Governor)**: Shehab Anwer's probe-first governor capping `max_tokens` at 4,096 after no-action streaks, eliminating the 2.06x stall deliberation token burn and converting wasted monologue into environment feedback.
- **Cold-Boot `0.00` Immunity (4 Layers)**:
  1. Multithreaded OS page-cache prefetcher (`delay=60, threads=3` to bypass GCS FUSE contention).
  2. Mode-specific startup timeout (`55 * 60` seconds = 3,300s) + 1,800s gateway wait.
  3. `patch_startup_shield.py` periodic 420s keepalive `RESET` on warmup game until `/health == 200 OK`, eliminating Kaggle's 15m container kill and socket floods during slow 35-65m weight loads.
  4. Cell 20 Hard Commit Rehearsal Gate: 5-minute `ft09` run.
- **Execution & Timeout Armor**: Decoupled HTTP `request_timeout_seconds` with a minimum 45s-60s completion floor in `solver.py`, preventing late-game 9s/16s socket drops on Level 4/5.
- **In-Memory Algorithmic Affine & Search Primitives**:
  - Pure Python $\text{GF}(2)$ and $\mathbb{Z}_m$ affine solver (`solve_affine_lights_out`) and bounded search primitives (`bounded_search`, `board()`) injected into the sandbox (`python_tool_sandbox.py` runtime globals) with zero prompt bloat.
  - Batches human-minimum action sequences (`action(acts)`) in 1 turn per level, scoring up to 6/6 levels (100.00 score) on `ft09`.
- **Clock Telemetry Grounding**: Retained full clock telemetry (`run_elapsed_seconds`, `time_remaining_seconds`) in `_compact_action_result` to eliminate countdown timer hallucinations.

## Feature Inventory
| # | Feature | Description | Milestone | Source | Status |
|---|---------|-------------|-----------|--------|--------|
| 1 | Franzen v11 Cell 2 Base Patch | Pristine 498,799-byte patch on Pennyroyal SGLang v2.5.3 + Qwen3.8-Flash-Next | M1 | Survey E1, E2, E3 | DONE |
| 2 | SGLang Serving & Stream Invariant | `MEMFRAC=0.96`, `MAXREQ=10`, `ARC3_MAX_ACTIVE_STREAMS=10`, `total_deadline=532*60` | M1 | Survey E1, E3 | DONE |
| 3 | Continuous D' Priority Scheduler | Shiiin9 continuous D' with Ouyang Ruize patience decay ($C$) and $b_0=8.0 \cdot C$ final bonus | M1 | Survey E2, E3 | DONE |
| 4 | Frozen System Prefix Invariant | Zero system prompt mutation; RadixAttention prefix intact across all 10 streams | M1 | Survey E1, E2, E3 | DONE |
| 5 | Cold-Boot 55-Min 0.00 Armor | 55m startup timeout, 420s keepalive reset hold, 1800s gateway wait, FUSE prefetcher | M1 | Survey E1 | DONE |
| 6 | Decoupled Request Timeout Floor | Ensure `request_timeout_seconds` in `solver.py` has a 45s floor, preventing late-game ReadTimeout drops | M1 | Survey E2, E3 | DONE |
| 7 | Full Clock Telemetry Grounding | Restore `run_elapsed_seconds` and `time_remaining_seconds` to prevent timer hallucinations | M1 | Survey E2 | DONE |
| 8 | Probe-First Completion Governor | Cap `max_tokens` at 4,096 on $\ge 3$ actionless turns to eliminate stall deliberation token burn | M1 | Survey E1 | DONE |
| 9 | Sandbox Affine & Search Primitives | Inject `solve_affine_lights_out` and `bounded_search` into sandbox runtime globals | M1 | Survey E2, E3 | DONE |
| 10 | Rehearsal Benchmark (ft09 > 47.62) | Rehearsal test targeting 6/6 levels (strictly beating 47.62 / 4.0 levels PB) | M2 | User Req R2 | DONE |
| 11 | Opaque-Box E2E Test Suite | Programmatic test harness covering build, AST parsing, patch apply, and simulation | M3 | User Req R2, R3 | DONE |
| 12 | Forensic Integrity Audit | Static, AST, and runtime verification verifying authentic logic and zero hardcoded stubs | M3 | System Instruction | DONE |
| 13 | Colab 80GB A100 Red-Team Critique | Adversarial critique via `colab-redteam` MCP with immediate verified shutdown | M3 | User Req R3 | DONE |
| 14 | Local Build & Metadata Sync | Programmatic generation of `notebooks/submission.ipynb` and metadata for `v34` | M4 | User Req R3 | DONE |
| 15 | Remote Kaggle GPU Commit | Push kernel `zerbytheboss/v34-sovereign-apex-engine` to Kaggle; monitor to `COMPLETE` | M4 | User Req R3 | DONE |
| 16 | Competition Submission / Rollover | Submit completed output to `arc-prize-2026-arc-agi-3` or arm auto-submitter | M4 | User Req R3 | DONE |

## Milestones
| # | Name | Scope | Dependencies | Status | Output |
|---|------|-------|-------------|--------|--------|
| M1 | v34 Solver & Builder Engineering | Implement `scripts/build_v34_notebook.py` synthesizing Franzen base, D' scheduler, probe-first, timeout floor, clock restoration, and sandbox affine primitives into `notebooks/submission.ipynb` | none | DONE | `scripts/build_v34_notebook.py`, `notebooks/submission.ipynb`, `notebooks/submission_v34.ipynb` |
| M2 | Local Rehearsal Benchmarking Gate | Execute local rehearsal benchmark on `ft09-0d8bbf25`, verifying stability and achieving score strictly $> 47.62$ (targeting 6/6 levels) | M1 | DONE | Rehearsal telemetry report, score 100.00% (6/6 levels, 75 actions), zero timeouts/crashes |
| M3 | Multi-Agent & Colab A100 Gate | Reviewer 1 & 2 APPROVE, Challenger 1 & 2 APPROVE, Forensic Auditor CLEAN, Colab Red-Team APPROVE with 0 idle CU | M1, M2 | DONE | Review, Challenger, Audit, and Colab Red-Team handoffs (Gate Result: PASS) |
| M4 | Kaggle Staging & Competition Deployment | Push `zerbytheboss/v34-sovereign-apex-engine`, verify `COMPLETE` status on GPU, and register submission on leaderboard | M1, M2, M3 | DONE | Verified Kaggle kernel run (COMPLETE, 15m 21s) and submission ref 57754379 |

## Interface Contracts
### `scripts/build_v34_notebook.py` ↔ `notebooks/submission.ipynb`
- `build_v34_notebook.py` must run deterministically via `.venv/bin/python3` with exit code 0.
- Outputs `notebooks/submission_v34.ipynb` and byte-identical `notebooks/submission.ipynb`.
- Valid JSON format with 23 cells matching the Kaggle execution pipeline.

### `python_tool_sandbox.py` ↔ Model REPL
- Sandbox exposes `current_frame`, `action(acts)`, `solve_affine_lights_out(current_frame)`, and `bounded_search(...)` in `runtime_globals`.
- `action(acts)` accepts a list of action dicts `[{'action': 'MOUSE', 'row': r, 'col': c}, ...]` and executes them synchronously.
- Returns compact sanitized result dict containing `run_elapsed_seconds` and `time_remaining_seconds`.

### `solver.py:request_timeout_seconds` ↔ HTTP Client
- When global game time remaining is $> 60$s, `request_timeout_seconds()` must return $\ge 45.0$s, never clamping below 45s.

## Code Layout
- `scripts/build_v34_notebook.py`: Version 34 master builder script.
- `notebooks/submission.ipynb`: Deterministic output notebook submitted to Kaggle.
- `notebooks/submission_v34.ipynb`: Version-tagged notebook backup.
- `notebooks/kernel-metadata.json`: Kaggle kernel metadata (`id: "zerbytheboss/v34-sovereign-apex-engine"`).
- `tests/test_v34_e2e_requirements.py`: Programmatic E2E test suite.
- `tests/test_v34_adversarial_stress.py`: Standalone adversarial stress tests.
- `scripts/auto_submit_v34.py`: Automated competition submitter and rollover monitor.
- `docs/FORENSIC_V34_SOVEREIGN_SYNTHESIS.md`: Comprehensive engineering documentation.

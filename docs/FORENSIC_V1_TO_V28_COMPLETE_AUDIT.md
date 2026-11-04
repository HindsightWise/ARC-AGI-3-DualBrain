# Comprehensive Forensic Audit (`v1`–`v27`), Forum Intelligence Synthesis, and Version 28 (`v28`) Architecture

## 1. Complete Chronological Ground-Truth Ledger of All 25 Kaggle Submissions (`v1`–`v27`)

Retrieved via `.venv/bin/kaggle competitions submissions -c arc-prize-2026-arc-agi-3 -v`:

| # | Ref (`_ref`) | Date (UTC) | Version | Parquet Bytes | Status | Public LB | Architectural Summary & Forensic Lesson |
|:-:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 25 | `57500943` | `2026-10-22 00:00:44` | **v27** | `3,411` | `COMPLETE` | **`0.00`** | Cold-disk I/O collision (`precache_model_thread` vs `sglang serve --weight-loader-drop-cache-after-load`) stretched SGLang weight load to `974.93s` (`1,274s` total boot > `720s` cap), starving `ft09` in commit (`0.00`) and exhausting the single-shot warmup/retry window in competition rerun (`<48m`). |
| 24 | `57471652` | `2026-10-21 07:29:20` | **v26** | `3,807` | `COMPLETE` | **`26.29`** | `v24` + `SOLVED_LEVEL_MEMORY` (`_SLM_ASK_TURNS=2`) + `animation_empty` + `spatial_helpers` + `120s` Timeout Armor. `_SLM_ASK_TURNS=2` inflated thinking tokens (`828 tok/act` vs `765` in `v24`), mutated `messages[0]` before `<tools>` on context eviction, and `120s` Timeout Armor held slots past deadline (`336.1s` on `ft09`). |
| 23 | `57373064` | `2026-10-19 00:01:46` | **v25** | `3,812` | `COMPLETE` | **`27.16`** | `v24` + `animation_archive` (+720-char system prompt bloat) + `fable_layer` (per-turn user prompt mutations) + `spatial_helpers` + `120s` Timeout Armor. |
| 22 | `57337517` | `2026-10-18 03:57:48` | **v24** | `3,836` | `COMPLETE` | **`28.79`** | **ALL-TIME PERSONAL BEST**: 100% pure `v11` (`dfranzen` M2) + `shiiin9` `D'` slot priority scheduler (`ours_form_priority.install_d`). Zero prompt mutations, zero timeout armor, `300.4s` exact stop on `ft09`. |
| 21 | `57315642` | `2026-10-17 15:04:03` | **v23** | `4,005` | `COMPLETE` | **`20.86`** | `max_output=4096` truncated `<think>` before `<tool_call>`; `EXPOSE_RESET=on` + `GUARDS_FROM_LEVEL=1` + post-gate `started_at` reset broke `_PriorityGate` clock. |
| 20 | `57194042` | `2026-10-14 20:03:49` | **v21** | `3,811` | `COMPLETE` | **`25.18`** | `v11` + `D'` + 5 community modules (`model_checks`, `animation_archive`, `edge_meter`, `spatial_helpers`, `fable_layer`). Scored `55.83` on 10 public games but `25.18` LB due to ~650 tokens of system prompt bloat (`0%` `model_checks` calls). |
| 19 | `57158130` | `2026-10-13 22:56:48` | **v20** | `3,854` | `COMPLETE` | **`17.34`** | `v19` + TWIN in-memory simulator + centroid affordances. Prompt bloat distracted exploratory tool use. |
| 18 | `57088288` | `2026-10-12 00:01:06` | **v19** | `3,823` | `COMPLETE` | **`24.30`** | Pure `v11` restoration (`MEMFRAC=0.96`, full visual history, `93.7%` prefix cache hit rate), recovering `+10.26` LB over `v18`. |
| 17 | `57055261` | `2026-10-11 00:01:07` | **v18** | `3,762` | `COMPLETE` | **`14.04`** | `ARC3_HISTORY_IMAGE_KEEP=0` mutated previous user messages on every turn, destroying SGLang RadixAttention prefix cache. |
| 16 | `57024129` | `2026-10-10 00:01:08` | **v17** | `0` | `COMPLETE` | **`None`** | Inference server disconnected unexpectedly at T+29.1m. |
| 13–15 | `56890454`–`56979143` | `2026-10-06`–`10-09` | **v16 (a–c)** | `3,717`–`3,758` | `COMPLETE` | **`21.07` / `19.40` / `18.45`** | `ARC3_RESUME_PROMPT_IMAGES=0` blinded model after `2048`-token yields; `EXPOSE_RESET=on` caused voluntary resets. |
| 12 | `56836123` | `2026-10-05 00:00:26` | **v15** | `3,838` | `COMPLETE` | **`23.29`** | 30-action hard reset guard killed levels requiring >30 actions. |
| 11 | `56808760` | `2026-10-04 00:00:06` | **v14** | `3,873` | `COMPLETE` | **`24.84`** | Public-game trajectory bank (0 overlap with hidden games) + M85 + M59 prompt hooks. |
| 09–10 | `56761048`–`56784330` | `2026-10-02`–`10-03` | **v11 (a–b)** | `3,853` / `3,826` | `COMPLETE` | **`26.70` / `25.06`** | Pure Daniel Franzen Milestone 2 (`dfranzen` v3) baseline on RTX PRO 6000 (`10` streams, Pennyroyal v2.5.3, Albucino MTP-3). |
| 06–08 | `56688161`–`56722442` | `2026-09-29`–`10-01` | **v8–v10** | `3,737`–`3,814` | `COMPLETE` | **`2.68`–`3.62`** | Milestone 1 Keith Tyser Duck on Dual T4 (`Qwen2.5-7B-AWQ`). |
| 01–05 | `55971922`–`56064943` | `2026-09-03`–`09-07` | **v1–v6** | `None` | `COMPLETE` | **`0.15`–`0.35`** | Pure symbolic/heuristic solvers without LLM. |

---

## 2. Synthesis of Kaggle Discussion Forum & Ecosystem Intelligence (Through Oct 22, 2026 UTC)

1. **Forum Topics `750173`, `750127`, and `750885` — The `0.00` Cold-Startup Inactivity Kill**:
   - **Jeroen Cottaar (Tufa Labs, Topic `750173`, Oct 17)**:
     > *"If you don't do any move for 15 minutes, the run will end with the score so far — `0.00` in your case since you didn't do anything yet. You can start a thread that issues a `RESET` move to one game every 10 minutes until startup to deal with this."*
   - **CPMP & Mirza Milan Farabi (Topic `750173`, Oct 20)** and **Francise Liang (Topic `750885`, Oct 20)**:
     Confirmed that Kaggle's network/FUSE mount for the `185.40 GB` model checkpoints intermittently experiences cold-mount read throttling where SGLang startup takes **15–22+ minutes** instead of 7–8 minutes, causing **2 out of 12 online submissions** of even unmodified `dfranzen` M2 notebooks to score **`0.00`**.
   - **Why `v27` triggered this exact failure mode (and what `v28` v1 empirically proved about `precache_model_thread`)**:
     - On Kaggle's GCS FUSE mount (`/kaggle/input/models/...`), SGLang's internal `--weight-loader-prefetch-checkpoints` only performs a `one-shard lookahead in loader order (38 shards)`. When we tested removing `precache_model_thread.start()` in `v28` v1, GCS FUSE per-file cold-open latency caused SGLang to stall `77s–222s per shard` (`>2,756s` for 28/38 shards), whereas **with** `precache_model_thread.start()` (`threads=3`, `CHUNK_SIZE=32MB` warming the Linux OS page cache across all 62 files), all 38 shards loaded in **`250s–294s`** (`v24`, `v25`, `v26`) and `958s` on the worst throttled mount (`v27`), where SGLang reached `Uvicorn running` at `t = 1,245s` (`20.7m`).
     - In `v27`'s commit run (`/tmp/v27_output/`), Cell 12 had `SERVER_STARTUP_TIMEOUT = 720s` (`12m`), so it released the benchmark 8.7 minutes before SGLang finished booting at `20.7m`, causing `ft09`'s 300s clock to expire during server startup (`score = 0.00`).
     - In `v27`'s competition rerun (`ref: 57500943`), only 1 single warmup `RESET` fired at minute 12 (`ARC3_WARMUP_ACTION_GAMES = '1'`), and with no periodic keepalive and `ARC3_HTTP_RETRY_INITIAL_SECONDS = '900'`, the run terminated early with a `3,411`-byte all-zero `submission.parquet` (`0.00` LB).

2. **Why Mutating `self._system_prompt` (`messages[0]`) or `_build_user_prompt` Hurts 110-Game LB (`lalalia9`, `Francise Liang`, `skarin`, `habanwer`, and Our `v24` vs `v25`/`v26`)**:
   - In Qwen3's `chat_template.jinja`, `messages[0]["content"]` (`self._system_prompt`) is rendered **before** the ~2,500-token `<tools>...</tools>` block inside `<|im_start|>system...<|im_end|>`.
   - Therefore, when `SOLVED_LEVEL_MEMORY_PATCH` appends `=== SOLVED LEVELS ===` to `self._system_prompt` upon context eviction, it inserts tokens *before* `<tools>...</tools>`, invalidating the `<tools>` KV cache and forking that stream off the shared 10-stream RadixAttention root node.
   - Across our entire 25-submission history, **every single run that mutated `self._system_prompt` or `_build_user_prompt` scored lower than pure `v24` (`28.79` LB)**.

3. **The Final-Level (`remaining == 0`) Priority Cliff in `shiiin9`'s `D'` Formula**:
   - In `shiiin9`'s `D'` (`priority = A * M * C + B * phi`), `remaining = max(0, n - level)` and `B = (0.0, 10.0, 14.0, 16.0)[min(remaining, 3)]`.
   - When a game clears Level $N-1$ and reaches its **final level** ($\ell = N$, so $\text{remaining} = 0$—which carries the single largest RHAE weight $\frac{N}{N(N+1)/2}$ in the entire game!), $B$ drops abruptly from `10.0` to **`0.0`**.
   - As a result, a game on Level 7 of 7 ($\ell = 7, N = 7$) drops from priority `~17,857` on Level 6 to **`~7,857`** on Level 7—less than half the priority of an unstarted 10-level game (`17,000`)! At its first context trim on the final level, it gets preempted for hours while all unstarted games cycle through.
   - Setting `b0 = 6.0` for `remaining == 0` (`B = (6.0, 10.0, 14.0, 16.0)[min(remaining, 3)]`) preserves the exact monotone step-down (`16 -> 14 -> 10 -> 6`) while keeping final-level games (`~13,857–16,800`) competitive to finish the win.

4. **Zero-Prompt-Bloat `M75` Segmentation Key Aliases (`rellik13`, Topic `745813`)**:
   - In `inference/utils/segmentation.py`, `segment_layer` (`current_frame.segment()`) returns node dicts with keys `id, color, hash, rotation, rotational_symmetry, pose_hash, shape_hash, pixels, boundary, children`.
   - Models frequently access `node["bbox"]` / `node["box"]`, `node["area"]` / `node["size"]`, or `node["centroid"]` / `node["center"]`, raising a `KeyError` and wasting a full LLM turn (~2,000 tokens; `198` wasted turns across 25 games in `rellik13`'s audit).
   - Wrapping each node in `_NodeDict(dict)` with `__missing__` resolves `bbox`/`box`/`bounds`, `area`/`size`/`count`, and `centroid`/`center` on the fly while keeping `repr(node)`, `json.dumps(node)`, and `list(node.keys())` **100% byte-identical** (zero extra `stdout` or prompt tokens).

---

## 3. Version 28 (`v28`) Architectural Specification

1. **100% Pure `v24` (`28.79` LB) System & User Prompt Invariant**:
   - Purges `patch_solved_level_memory.py`, `animation_archive.py`, `fable_layer`, and `install_timeout_armor`.
   - `self._system_prompt` and `_build_user_prompt` remain **100% untouched across all levels (Levels 1–10) and all context drains**, guaranteeing 100% shared RadixAttention root prefix residency across all 10 streams.
2. **4-Layer Cold-Boot `0.00`-Score Immunity**:
   - **Layer 1 (8-Stream OS Page-Cache Prefetching, `delay=15, threads=8`)**: Upgrades `precache_model_thread` in Cell 6 from `delay=60, threads=3` to `delay=15, threads=8` and preserves `precache_model_thread.start()` in Cell 12. Empirically verified on Kaggle `NvidiaRtxPro6000` across `v28` v1/v2/v3 on cold FUSE mounts (`~70s` wheelhouse):
     - `v28` v1 (no `precache_model_thread`): `>2,756s` (`>46m`, timed out at 28/38 shards).
     - `v28` v2 (`delay=60, threads=3`): `load_weight = 2,269.16s` (`37.8m`), `scheduler_e2e = 2,449.88s` (`40.8m`).
     - `v28` v3 (`delay=15, threads=8`): **`load_weight = 949.82s` (`15.8m`, `2.39x` faster!), `scheduler_e2e = 1,134.30s`, `READY after 1165s`**, scoring **`71.43` (`5.0/6` levels in `66` actions, `301.2s`)** on `ft09`.
   - **Layer 2 (Mode-Specific Startup Timeout & 60m HTTP Grace)**:
     - `SERVER_STARTUP_TIMEOUT = (10 * 60) if TRUE_SUBMISSION else (45 * 60)`
     - `os.environ['ARC3_WARMUP_ACTION_GAMES'] = '1' if TRUE_SUBMISSION else '0'`
     - `os.environ['ARC3_HTTP_RETRY_INITIAL_SECONDS'] = '3600'`
     - Automatic 1x `sglang serve` relaunch in Cell 12 if `proc.poll() is not None` during startup.
   - **Layer 3 (`patch_startup_shield.py` — Jeroen Cottaar's Periodic Warmup Keepalive + Post-Ready Clock Reset)**:
     - In `_HarnessGameSession._play_inner`, if SGLang is not yet marked `_analyzer_ready`, the warmup game issues its initial `warmup RESET` and polls `/health`, issuing a keepalive `RESET` every `540s` (`9 minutes`) until `/health` returns `200 OK`, and then resets `self.started_at = time.monotonic()` so zero startup time is ever charged against any game's runtime budget.
   - **Layer 4 (Hard Commit-Run Gate in Cell 20)**:
     - When `not TRUE_SUBMISSION`, Cell 20 verifies that `ft09` executed `actions > 1` and solved `>= 1` level (`score > 0`) after writing `submission.parquet`; otherwise it raises `RuntimeError`.
3. **Upgraded `D'` Priority with Final-Level (`remaining == 0`) Protection (`b0 = 6.0`)**:
   - `D_PRIME["b0"] = 6.0`, `B = (params.get("b0", 6.0), params["b1"], params["b2"], params["b3"])[min(remaining, 3)]`.
4. **Zero-Prompt-Tokens Sandbox Ergonomics**:
   - `patch_animation_empty.py`: Single-frame animation schema fallback (`action, frames: 0, steps: [], error`).
   - `patch_spatial_helpers.py`: REPL helpers (`rot90`, `rot180`, `rot270`, `flipv`, `fliph`, `manhattan_distance`, `find_cells`, `find_objects`) + `M75` `_NodeDict(dict)` dynamic key aliases (`bbox`, `box`, `bounds`, `area`, `size`, `count`, `centroid`, `center`) in `segmentation.py`.

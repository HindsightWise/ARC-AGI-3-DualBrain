# Sovereign Knowledge Index for `ARC-AGI-3-Kaggle-Starter`

- **Memory Web Package**: `/Users/zerbytheboss/.gemini/antigravity/scratch/memory_web`
- **110-Paper Desktop arXiv Corpus JSON**: `/Users/zerbytheboss/.gemini/antigravity/scratch/memory_web/data/arxiv_desktop_corpus.json`
- **Executable Synthesis Module**: `/Users/zerbytheboss/.gemini/antigravity/scratch/memory_web/memory_web/arxiv_synthesis.py`

## Sovereign OO-Harness (6 Pillars) & 110-Paper Desktop arXiv Synthesis

All agents and project workspaces share the unified `Memory Web × Harness² × OO-Harness` substrate (`/Users/zerbytheboss/.gemini/antigravity/scratch/memory_web`, `3,040+` memories across `15` lineages):

### 1. Executable Object-Oriented Harness Hierarchy (`arXiv:2610.28380v1`, `memory_web/oo_harness.py`)
- **14 Cooperative Subclasses (`s7..s41`)**: Override `manage_context`, `wrap_file_write`, `normalize_tool_error`, and `verify_before_finish`, chaining ancestor contracts via `super()`.
- **Lineage A (`quant_finance`)**: `BaseHarness -> s7:FormulaLiteralHarness -> s30:PercentCheckHarness -> s40:QuantBradleyTerryHarness`
- **Lineage B (`operating_protocol` / `systems_kernel` / `ai_agents_memory`)**: `BaseHarness -> s9:CommitGateHarness -> (s15:ResetAwareOracleHarness, s12:NearBudgetReminderHarness -> (s29:ToolErrorGuidanceHarness -> s16:ShadowInterfaceHarness, s13:BudgetAwarePOMDPHarness))`
- **Lineage C (`discrete_math_algorithms` & Cross-Corpus Synthesis)**: `s14:DiscoverySetupHarness -> s17:SovereignArxivSynthesisHarness`
- **Cross-Benchmark & Craft Lineages**: `s18:OfficeDocWriteHarness`, `s21:CompletenessFinalizeHarness`, `s41:RookAntiSlopHarness`

### 2. Budget-Augmented POMDP & Multi-Turn Reward Engine (`arXiv:2610.28409v1`, `s13:BudgetAwarePOMDPHarness`)
- Tracks $\tilde{s}_t = (s_t, b_t)$ with $b_{t+1} = b_t - \kappa(a_t)$ across `steps`, `tool_calls`, `tokens`, or `turns`, injects `<budget>` tags, batches independent tool calls under step/turn limits, and scores all 11 cost-conditioned objectives (`Dual`, `Sigmoid-Gate`, `Gate-Neg`, etc.) + 3-term Dr. GRPO advantage decomposition.

### 3. Reset-Aware Settled-State Oracle & Spec-Only Regeneration (`arXiv:2610.28595v1`, `s15:ResetAwareOracleHarness`)
- Never grade pre-reset `Cycle 0` transients (which cause a `-11.4 pp` vs `+8.9 pp` sign reversal). Always evaluate post-reset settled state (`Cycle 1+` / `Tier F`).
- Split compiler/lint syntax errors (`s29` surgical repair) from lint-clean semantic bugs: when broken code poisons context ($\Delta_{c1} = \text{Pass}(\text{SpecOnly}) - \text{Pass}(\text{Broken}+\text{Spec}) > 0$), purge the broken candidate and regenerate cleanly from the specification.

### 4. Agent S4 Evolving Shadow Interfaces (`arXiv:2610.28601v1`, `s16:ShadowInterfaceHarness`)
- Expose evolved Python toolkits $R = (\mathcal{I}(R), K(R))$ via the 3-verb progressive disclosure CLI (`capabilities [scope]`, `doc <module>.<fn>`, `call <module>.<fn>`), construct interfaces strictly from ungraded training trajectories (`dummy_verifier_score == 0.0`), merge complementary branches via 2-parent DAG `crossover`, and denoise across repeated rollouts.

### 5. 110-Paper Desktop arXiv Synthesis (`s17:SovereignArxivSynthesisHarness`, `memory_web/arxiv_synthesis.py`)
1. **Discrete Maths (41 Papers)**:
   - **2-WL Symmetry-Compressed Temporal Exploration (`arXiv:2610.30123v1`)**: Bounds connected temporal/grid exploration to $O(q n \log^3 n)$ steps where $q$ is the 2-WL color class count (`WeisfeilerLeman2WLTemporalExplorer`).
   - **Weak Submodular Matroid Greedy (`arXiv:2610.26967v2`)**: Certifies harmonic-core greedy guarantee $R_{\text{gr}}(\gamma, r)$ over rank-$r$ partition/lineage matroids (`WeakSubmodularMatroidGreedy`).
   - **Kemeny–Braess Paradox Link Filter (`arXiv:2610.24143v1`)**: Prunes cross-domain links where $\Delta K(P) > 0$ on Kemeny's constant $K(P) = \sum_{j \ge 2} \frac{1}{1-\lambda_j(P)}$ (`KemenyBraessLinkFilter`).
   - **2-Adic Tree & Kraft Weight-Space Geometry (`arXiv:2610.23963v1`, `arXiv:2610.23947v1`)**: Validates binary lineage trees via $\sum_{j \ne i} 2^{-\tau_{ij}} = \frac{1}{2}$ (`KraftWeightSpaceTreeValidator`).
   - **Tree Plurality Veto (`arXiv:2610.24456v1`)**: Tournament arbitration with optimal metric distortion $3$ (`TreePluralityVetoArbiter`).
   - **Complexity & SAT Certificates**: Komlós vector balancing in $O(n^\omega \log^5(nd))$ (`arXiv:2610.20995v1`), rapid coloring mixing up to tree uniqueness $q \ge (1+o_d(1))d$ (`arXiv:2610.25391v1`), Pfaffian recognition in `coRP` (`arXiv:2610.29817v1`), ARRIVAL in $2^{O(n^{1/3}\log n)}$ (`arXiv:2610.28241v1`), Erdős–Gyárfás `reduce–dispatch–enumerate` SAT refutations (`arXiv:2610.24484v1`), and compact geometric SAT encodings (`arXiv:2610.22241v1`).
2. **Statistics (16) & Agent Harnesses (8)**:
   - **Certified Context-vs-Weight Placement (`arXiv:2610.29419v1`)**: $V(D) = \mathbb{E}[\text{Stor}_X + \text{PriorRepair} - \text{DecLoss}]$ + leave-one-out deletion $\Delta L(c_i) > 0$ toxic-context eviction (`CertifiedContextVsWeightGate`).
   - **Dreaming Spectral Regularization (`arXiv:2610.30009v1`)**: Interleaves $T_d > 1$ transitions to restore the Markov spectral gap $1 - |\lambda_2(P)|$ (`DreamingSpectralRegularizer`).
   - **Jev Luce `log-sum-exp` Calibration (`arXiv:2610.26797v1`)**: Aggregates refined candidate groups via $\tau \log \sum_{y \in G} \exp(s_y/\tau)$ and verifies teacher-student cosine alignment (`JevLuceLogSumExpCalibrator`).
   - **HarnessAdapt-RL Pareto Gate (`arXiv:2610.28069v1`) & MAD-RSI (`arXiv:2610.28201v1`)**: Routes between harness evolution and policy updates (`HarnessAdaptParetoGate`).
3. **MathFinance (18), Microstructure (13), Risk (8) & Portfolio Management (6)**:
   - **4th-Moment Excess-Kurtosis OAS Shrinkage (`arXiv:2610.22144v1`)** & **Regime NCO (`arXiv:2610.27751v1`)** (`FourthMomentOASShrinkage`).
   - **6-Defect Intraday Backtest-vs-Live Parity Gate** (`BacktestLiveParityAuditor`: intra-bar fill realism, `NEST` `arXiv:2610.21491v1` stale-tick filter, concave power-law impact `arXiv:2610.23749v1`, Wiener–Hopf causal filters `arXiv:2610.24558v1`, `NoLBERT` `arXiv:2610.22745v1` zero-lookahead NLP, and Hygentra deterministic pre-execution control).

### 6. ARC Prize 2026 (`arc-prize-2026-arc-agi-3`) Telemetry & `v33` Sovereign Apex Engine
- **`v32` Official Leaderboard Outcome (`ref: 57709347`)**: `SubmissionStatus.COMPLETE`, **`26.03` Public LB** (up `+0.59` from `v31` at `25.44`, `+5.95` from `v29` at `20.08`, `+26.03` from `v27` at `0.00`).
- **`v33` Sovereign Apex Engine (`zerbytheboss/v33-sovereign-apex-engine`, `Version 2`, `sha256=2bd6e167ce57`, 23 cells)**:
  1. **Token-Budget-Only Context Eviction (`ARC3_HISTORY_ASSISTANT_TURNS=0`)**: Disables premature 150-assistant-turn eviction so messages are only trimmed when exceeding the 116k token budget (`58*1024` hysteresis drain).
  2. **Cache-Safe Eviction-Only `SOLVED_LEVEL_MEMORY`**: Captures per-level winning action counts, last 30 pre-action `prev == level` `RESET`-filtered actions, 900-char winning-turn `reasoning` fallback, and a backtick/multiline-safe `Rule for level N:` mechanic note (with `_resume_after_yield` guard), pinning to `self._system_prompt` strictly when `_trim_messages_for_context` or `_force_reduce_messages` actually drops messages to preserve SGLang RadixAttention prefix cache hits.
  3. **Animation-Safe `ARC3_NOOP_REPEAT_GUARD=1` (`ARC3_GUARDS_FROM_LEVEL=2`)**: Blocks repeated unchanged actions on Level $\ge 2$ while treating `transient_pixels > 0` as `changed = True` across all 4 layers (`history_events`, `consecutive_noops`, `gameplay_changed_per_action`, and `_record_guard_observations`).
  4. **Whole-Run Clock Stripping in `_compact_action_result`**: Strips `run_elapsed_seconds` and `time_remaining_seconds` from the sandbox action return dict to prevent whole-run clock distraction while preserving `self._game_elapsed_seconds`.
  5. **Exact `shiiin9` Multi-Level Continuation + $C$-Scaled Final-Level $b_0$ + Unconditional Cold-Boot Shield**: Preserves `B = (10.0, 14.0, 16.0)` for `remaining >= 1` and $b_0 = 8.0 \cdot C$ for `remaining == 0, level > 1` (`#OURS_FRESH replaced=True`), plus unconditional initial warmup `RESET` and periodic `480s` keepalive `RESET`s.


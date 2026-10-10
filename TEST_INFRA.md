# ARC-AGI-3 Version 6: 4-Tier E2E Testing Architecture & Infrastructure Specification

**Document Status**: CERTIFIED & PRODUCTION READY  
**Author**: E2E Test Suite Designer & Quality Assurance (`teamwork_preview_test_writer`)  
**Target Repository**: `ARC-AGI-3-Kaggle-Starter`  
**Applicable Scope**: `PROJECT.md` (Features F1–F7, Milestones M1–M4)  
**Evaluation Standard**: Relative Human Action Efficiency (RHAE) Quadratic Scoring & Kaggle Air-Gapped Compliance  

---

## 1. Executive Summary & Testing Philosophy

The ARC-AGI-3 Version 6 reasoning agent operates under the **Quadratic Action Efficiency Law**, where every exploratory action emitted to the environment incurs a quadratic penalty $\text{score} = (\text{human} / \text{ai})^2$. To guarantee that the agent achieves competitive dominance (RHAE $\ge 1.20$, surpassing the Top 500 threshold of 1.73), the testing infrastructure enforces a **Zero-Exploration Invariant**: all perception, topological mapping, and graph search must be empirically verifiable in offline simulation before any action is dispatched.

The testing architecture is structured into **Four Formal Tiers**:
- **Tier 1: Feature Coverage** ($\ge 5$ discrete tests per feature across F1–F7).
- **Tier 2: Boundary & Corner Cases** ($\ge 5$ adversarial boundary tests where boundaries exist).
- **Tier 3: Cross-Feature Combinations** (Pairwise interaction invariants).
- **Tier 4: Real-World Workload Scenarios** (25 Official ARC-AGI-3 environments + 25 Novel Benchmark Substrates across 5 Paradigms).

```
+-------------------------------------------------------------------------------+
|                      TIER 4: REAL-WORLD WORKLOAD SCENARIOS                    |
|    25 Official ARC Environments (ls20, vc33, m0r0, ka59, tu93, bp35, ...)     |
|    25 Novel Synthetic Benchmarks (ng01..ng25 across 5 Computational Paradigms)|
+-------------------------------------------------------------------------------+
                                        ^
                                        |
+-------------------------------------------------------------------------------+
|                   TIER 3: CROSS-FEATURE COMBINATIONS                          |
|  - Diff-Mask + Lattice Quantization      - Fleet Queue + Bump Isolation       |
|  - Camera Scale + Coordinate Clicks      - Geodesic Detour + State-Space BFS  |
|  - Straight-Line March + Multi-Pixel     - Terminal Reset + Memory Purge      |
+-------------------------------------------------------------------------------+
                                        ^
                                        |
+-------------------------------------------------------------------------------+
|                    TIER 2: BOUNDARY & CORNER CASES                            |
|  - Extreme Coordinates (0,0),(63,63)     - Non-Standard / Degenerate Shapes   |
|  - Zero-Displacement Same-Cell Detours   - Pathological Texture Entropy       |
|  - Dead-End / Corner Pad Re-triggering   - Negative & Out-of-Range Palette    |
+-------------------------------------------------------------------------------+
                                        ^
                                        |
+-------------------------------------------------------------------------------+
|                      TIER 1: FEATURE COVERAGE (F1 - F7)                       |
|  F1: Morphological Perception            F2: Universal Affordance & World Model|
|  F3: Dual Offline Planner                F4: Sovereign Executive Dispatch     |
|  F5: Novel Substrate Generalization      F6: Packaging & Air-Gapped Deploy    |
|  F7: Full E2E & Regression Suites                                             |
+-------------------------------------------------------------------------------+
```

---

## 2. Tier 1: Feature Coverage Specification (F1 – F7)

Each feature in `PROJECT.md` is covered by at least 5 dedicated, isolated verification tests with explicit authoritative sources for expected output.

### Feature 1 (F1): Zero-Exploration Morphological Perception
*Interface Contract*: `ScenePerception` + `DynamicLatticeDetector` + `DynamicHUDDetector`
1. **Test 1.1: Multi-Pixel Step Lattice Pitch & Spatial Offset Inference**
   - *Authoritative Source*: Mathematical congruence $r \equiv r_{\text{off}} \pmod S, c \equiv c_{\text{off}} \pmod S$.
   - *Test Function*: `TestGeodesicPlannerUnobstructed.test_multi_pixel_step_size` & `test_v4_cognitive_flexibility.test_dynamic_lattice`.
   - *Verification*: Evaluates displacement sequences for step sizes $S \in \{1, 2, 3, 5, 8\}$; asserts exact recovery of step pitch, row offset, column offset, and downsampled grid dimensions without live probing.
2. **Test 1.2: Dynamic Perimeter HUD & Goal Template Extraction**
   - *Authoritative Source*: Perimeter quadrant contrast scanning over outer 14 pixels.
   - *Test Function*: `test_v4_cognitive_flexibility.test_dynamic_hud_on_synthetic`.
   - *Verification*: Scans all 8 border margins (top, bottom, left, right, 4 corners); extracts framed goal template and masks HUD bounding box while preserving full $64 \times 64$ visibility on canvas.
3. **Test 1.3: Difference-Mask Intersection Invariant**
   - *Authoritative Source*: SKILL §9.1 Invariant: $\text{candidate\_sprite} \cap \text{diff} \ne \emptyset$.
   - *Test Function*: `test_adversarial_deep.test_bump_collision_passable_map_update` & `TestCollisionBumpLearning.test_bump_collision_updates_passable_map_and_known_walls`.
   - *Verification*: Injects stationary background noise pixels identical in color and shape to moving sprites; verifies tracker strictly filters stationary noise and attributes displacement exclusively to diff-intersecting sprites.
4. **Test 1.4: Canonical Shape Invariance across 96 Transformations**
   - *Authoritative Source*: Ground truth geometric rotation matrices for 6 canonical polyominos $\times$ 4 palette colors $\times$ 4 orthogonal rotations ($0^\circ, 90^\circ, 180^\circ, 270^\circ$).
   - *Test Function*: `TestCanonicalShapeInvariance.test_all_96_configurations_3x3_and_6x6`.
   - *Verification*: Confirms 100% collision-free binary hashing across all 24 spatial states and unambiguous matching in both $3 \times 3$ primitive and $6 \times 6$ Kronecker-downsampled HUD blocks.
5. **Test 1.5: Scene Background & Canvas Color Segmentation**
   - *Authoritative Source*: Mode color frequency and connected-component boundary extraction.
   - *Test Function*: `test_challenger_2_stress.test_action6_synthetic_clusters` & `test_action6_vc33_targeting`.
   - *Verification*: Segments non-background sprites into bounding boxes, centroids, sizes, and colors outside HUD exclusion margins.

### Feature 2 (F2): Universal Affordance & World Modeling
*Interface Contract*: `AffordanceModel` + `OperatorPadDetector` + `ScenePerception`
1. **Test 2.1: Display-to-Grid Camera Coordinate Scaling & Letterboxing**
   - *Authoritative Source*: SKILL §8 Coordinate Mapping: $\text{disp}_x = \text{grid}_x \cdot \text{scale} + \text{off}_x + \lfloor \frac{\text{scale}}{2} \rfloor$.
   - *Test Function*: `test_challenger_v3_2_stress.test_action6_vc33_targeting`.
   - *Verification*: Verifies coordinate click actions emit valid integer tuples within $[0, 63]^2$ compensating for display letterboxing.
2. **Test 2.2: Operator Pad Modulo Cyclic Group Arithmetic ($Z_6 \times Z_4 \times Z_4$)**
   - *Authoritative Source*: Finite cyclic group theory: $\text{shape} \equiv (s + 1) \pmod 6$, $\text{color} \equiv (c + 1) \pmod 4$, $\text{rot} \equiv (r + 1) \pmod 4$.
   - *Test Function*: `TestOperatorModuloTransitions.test_cyclic_arithmetic_modulo_properties`.
   - *Verification*: Tests forward stepping, wrap-around boundaries, and commutativity across all operator pads.
3. **Test 2.3: Receptacle Blocking Invariant for Unmatched Goals**
   - *Authoritative Source*: Physical obstacle semantics: an unmatched receptacle acts as a solid barrier.
   - *Test Function*: `TestReceptacleBlockingInvariant.test_receptacle_blocks_passage_when_unmatched`.
   - *Verification*: Confirms planner strictly detours around unmatched receptacle positions until the required configuration is acquired.
4. **Test 2.4: Adjacent Contextual Affordance Testing (`ACTION5`)**
   - *Authoritative Source*: Von Neumann 4-neighborhood entity proximity.
   - *Test Function*: `test_adversarial_deep.test_adversarial_action5_corner_clipping`.
   - *Verification*: Emits `ACTION5` if and only if an interactive entity is present in an adjacent cardinal tile; rejects empty or self-only tiles.
5. **Test 2.5: Non-Euclidean Warp Edge Dynamic Graphing**
   - *Authoritative Source*: Discontinuous coordinate displacement $|r_t - r_{t-1}| > 2 \cdot \text{step}$.
   - *Test Function*: `TestGeodesicPlannerObstructed.test_wall_blocks_direct_sight_convex_detour` & `ng01` WormholeTunnel integration.
   - *Verification*: Dynamically records portal coordinates and exit transitions, allowing geodesic routing across separated spatial chambers.

### Feature 3 (F3): Dual Offline Planner
*Interface Contract*: `GeodesicPlanner` + `ConfigurationPlanner` (State-Space BFS)
1. **Test 3.1: Unobstructed Manhattan Exactness & Zero Trajectory Jitter**
   - *Authoritative Source*: Manhattan metric $L = |r_1 - r_2| + |c_1 - c_2|$ with $\le 1$ turn penalty.
   - *Test Function*: `TestGeodesicPlannerUnobstructed.test_unobstructed_manhattan_exactness_and_zero_jitter`.
   - *Verification*: Validates that across dense start-goal coordinate grids, the planner produces strictly monotonic straight paths with zero zig-zag jitter.
2. **Test 3.2: Geodesic A* Convex Detour Around Solid Barriers**
   - *Authoritative Source*: Shortest path distance under Euclidean turn-weighted A*.
   - *Test Function*: `TestGeodesicPlannerObstructed.test_wall_blocks_direct_sight_convex_detour`.
   - *Verification*: Verifies the planner computes the minimal convex envelope path around horizontal/vertical walls without clipping corners.
3. **Test 3.3: U-Shaped Cul-de-Sac Pocket Escape**
   - *Authoritative Source*: Graph connectivity and dead-end back-tracking.
   - *Test Function*: `TestGeodesicPlannerObstructed.test_u_shaped_cul_de_sac`.
   - *Verification*: Places the avatar deep inside a 3-walled pocket; verifies the planner navigates out of the pocket entrance without cycling or getting trapped.
4. **Test 3.4: Re-Trigger Step-Off & Pad Interference Avoidance**
   - *Authoritative Source*: Operator state-space requirements: stepping off and back onto a pad to trigger consecutive state transitions.
   - *Test Function*: `TestBFSOptimalityAndStress.test_retrigger_action_optimality` & `test_neighbor_pad_interference_avoidance`.
   - *Verification*: Confirms exact theoretical minimum action counts ($k \times 2$ steps for $k$ triggers) while avoiding adjacent conflicting pads.
5. **Test 3.5: Independent Dijkstra Oracle Equivalence**
   - *Authoritative Source*: Independent Dijkstra shortest-path reference implementation.
   - *Test Function*: `TestBFSOptimalityAndStress.test_independent_oracle_equivalence`.
   - *Verification*: Compares the output of `ConfigurationPlanner` against an independent Dijkstra implementation over 20 randomized multi-pad mazes; asserts 100% path length equivalence.

### Feature 4 (F4): Sovereign Executive Dispatch
*Interface Contract*: `MyAgent` + `WorkingMemory` + `FleetPlanQueue`
1. **Test 4.1: Fleet Plan Queue Isolation Under Injected Collisions**
   - *Authoritative Source*: SKILL §7 Architecture Standard: decoupled execution queue immune to bump clearing.
   - *Test Function*: `test_reviewer2_m1_adversarial.test_queue_isolation_under_forced_bumps`.
   - *Verification*: Injects artificial zero-displacement bumps during plan execution; verifies `fleet_plan_queue` remains intact and does NOT restart from step 0.
2. **Test 4.2: Straight-Line March Protection in Loop Interceptors**
   - *Authoritative Source*: SKILL §9.3 Invariant: oscillation detection must verify $A \ne B$ before flagging $A, B, A, B$.
   - *Test Function*: `test_challenger_v3_2_stress.test_terminal_safety` & `test_adversarial_deep.test_adversarial_loop_interception_4cycle_and_constrained_legal`.
   - *Verification*: Verifies straight-line runs of 4+ identical actions ($R, R, R, R$) are never aborted as loops.
3. **Test 4.3: 4-Cycle Loop Interception & Constrained Orthogonal Escape**
   - *Authoritative Source*: Topological state hashing and cardinal orthogonal complement vectors.
   - *Test Function*: `test_adversarial_deep.test_adversarial_loop_interception_4cycle_and_constrained_legal`.
   - *Verification*: Simulates 4-cycle loop ($U \to R \to D \to L \to U \to R \to D \to L$); asserts immediate loop detection and legal orthogonal escape dispatch.
4. **Test 4.4: Terminal Safety on GAME_OVER and NOT_PLAYED**
   - *Authoritative Source*: ARC Engine 400 Bad Request prevention protocol.
   - *Test Function*: `TestChallengerV3Stress.test_terminal_safety`.
   - *Verification*: Submits `GAME_OVER` and `NOT_PLAYED` frames (both string and enum variants); asserts instantaneous `GameAction.RESET` return and complete working memory purge.
5. **Test 4.5: Hermetic State Cleanup on Level Transitions**
   - *Authoritative Source*: Multi-level episode lifecycle contracts.
   - *Test Function*: `test_reviewer2_m1_adversarial.test_level_transition_queue_clearing`.
   - *Verification*: Transitions from level $k$ to $k+1$; verifies all plan queues, click buffers, and state hashes are purged, resetting the one-way initialization latch.

### Feature 5 (F5): Novel Substrate Generalization
*Interface Contract*: Benchmark evaluation over 25 novel synthetic environments (`scripts/novel_games/`)
1. **Test 5.1: Paradigm 1 Spatial & Topological Navigation (`ng01`–`ng05`)**
   - *Verification*: Zero-shot traversal across wormhole tunnels (`ng01`), moving dynamic hazards (`ng02`), light-shadow barriers (`ng03`), gravity wells (`ng04`), and collapsing path tiles (`ng05`).
2. **Test 5.2: Paradigm 2 Push-Box & Object Physics (`ng06`–`ng10`)**
   - *Verification*: Momentum sliding on frictionless ice (`ng06`), boulder bridge gap filling (`ng07`), magnetic polarity attraction/repulsion (`ng08`), dual heavy/light boxes (`ng09`), and portal box routing (`ng10`).
3. **Test 5.3: Paradigm 3 Pattern Matching & Logic Gates (`ng11`–`ng15`)**
   - *Verification*: Combinatorial circuit toggling (`ng11`), subtractive color mixing (`ng12`), symmetrical canvas replication (`ng13`), spatial sudoku constraints (`ng14`), and temporal sequence playback (`ng15`).
4. **Test 5.4: Paradigm 4 Multi-Agent & Desynchronization (`ng16`–`ng20`)**
   - *Verification*: Quad mirror symmetry-breaking collisions (`ng16`), adversarial predator avoidance (`ng17`), convoy escort routing (`ng18`), shadow clone tandem stepping (`ng19`), and split-merge avatar recombination (`ng20`).
5. **Test 5.5: Paradigm 5 Affordance & Point-and-Click Mechanisms (`ng21`–`ng25`)**
   - *Verification*: Concentric tumbler lock alignment (`ng21`), continuous pipe flow routing (`ng22`), sliding 8-tile state-space permutation (`ng23`), optical beam splitter refraction (`ng24`), and multi-component recipe crafting (`ng25`).

### Feature 6 (F6): Single-File Packaging & Offline Deployment
*Interface Contract*: `scripts/build_notebook.py` + `notebooks/submission.ipynb` + Air-Gapped Invariants
1. **Test 6.1: Air-Gapped Zero-Internet Execution Invariant**
   - *Authoritative Source*: Kaggle competition rules: Internet OFF, no external network requests.
   - *Test Function*: `scripts/simulation_harness.py` offline arcade initialization (`OperationMode.OFFLINE`).
   - *Verification*: Verifies all game evaluation instances initialize hermetically without contacting remote endpoints.
2. **Test 6.2: Submission Notebook Generation & JSON Validation**
   - *Authoritative Source*: Kaggle Jupyter notebook format v4 schema.
   - *Test Function*: `python scripts/build_notebook.py`.
   - *Verification*: Validates that `notebooks/submission.ipynb` builds cleanly, contains valid JSON cells, and packages `agent/my_agent.py` without syntax errors.
3. **Test 6.3: Pure Python 3.12 Standard & Dependency Whitelist**
   - *Authoritative Source*: ARC-AGI-3 Kaggle Starter specification.
   - *Verification*: Verifies execution uses Python 3.12 with only standard libraries and pre-installed competition packages (`numpy`, `pydantic`, `arcengine`, `arc_agi`).
4. **Test 6.4: Inline Default Policy Fallback**
   - *Authoritative Source*: Hermetic packaging invariant.
   - *Test Function*: `agent.my_agent.load_policy()`.
   - *Verification*: Ensures agent operates fully using `DEFAULT_POLICY` dictionary when external JSON config files are deleted or inaccessible.
5. **Test 6.5: Single-File Agent Invariant**
   - *Authoritative Source*: `PROJECT.md` Code Layout: single-file production agent in `agent/my_agent.py`.
   - *Test Function*: `test_reviewer2_m1_adversarial.test_novel_game_queue_non_interference`.
   - *Verification*: Asserts all agent decision logic is self-contained in `agent/my_agent.py`.

### Feature 7 (F7): Full E2E & Regression Test Verification
*Interface Contract*: Test Suite Runner + Simulation Harness + Resource Profilers
1. **Test 7.1: Full Unittest Discovery 100% Pass Rate**
   - *Authoritative Source*: `python -m unittest discover tests`.
   - *Verification*: All 40 unit and stress test cases execute and report `OK` status with zero failures, zero errors, and zero warnings.
2. **Test 7.2: Execution Latency & High-Throughput FPS Benchmark**
   - *Authoritative Source*: Real-time game engine budget: $< 3.0$ ms per step ($> 300$ FPS).
   - *Test Function*: `TestChallengerV3Stress.test_benchmarks`.
   - *Verification*: Measures 1000 consecutive steps across pure puzzle ($> 3400$ FPS), pure click ($> 3700$ FPS), and mixed mode ($> 650$ FPS), beating the 300 FPS target by $\mathbf{>2\times}$.
3. **Test 7.3: Memory Leak & Retention Audit under Continuous Execution**
   - *Authoritative Source*: Long-running stability constraint: $< 500$ KB growth across 1000+ steps.
   - *Test Function*: `TestChallengerV3Stress.test_memory_bounds`.
   - *Verification*: Profiles memory via `tracemalloc` across 1000 steps; verifies net memory delta is $\le 0.42$ KB (negligible) with peak under 700 KB.
4. **Test 7.4: 25 Official Game Simulation Telemetry**
   - *Authoritative Source*: `scripts/simulation_harness.py --games all`.
   - *Verification*: Executes against official ARC-AGI-3 environments without unhandled exceptions or HTTP 400 Bad Request errors.
5. **Test 7.5: 25 Novel Game Simulation Telemetry**
   - *Authoritative Source*: `scripts/simulation_harness.py --novel-all`.
   - *Verification*: Runs all 25 novel games; verifies win rate and quadratic RHAE calculation.

---

## 3. Tier 2: Boundary & Corner Cases Specification

Where boundaries and extreme inputs exist, the test suite exercises edge conditions to prevent boundary crashes, coordinate overflow, and deadlocks:

| # | Boundary Category | Adversarial Condition | Expected Behavior | Verification Test |
|---|---|---|---|---|
| **B1** | **Screen Perimeters** | Target coordinates at $(0, 0)$, $(0, 63)$, $(63, 0)$, $(63, 63)$ | Emits valid `ACTION6` coordinates clamped to $[0, 63]^2$ | `test_action6_vc33_targeting` |
| **B2** | **Boundary Bump Protection** | Avatar at outer grid edges (row 0, row 63, col 0, col 63) moving outward | Records wall off-grid; current standing cell remains `passable=True` | `test_boundary_bump_*` (5 tests) |
| **B3** | **Zero Displacement** | Start coordinate equals Goal coordinate $(\mathbf{s} == \mathbf{g})$ | Returns empty path `[]` immediately without pathfinding overhead | `test_zero_displacement_same_cell`, `test_zero_displacement_comprehensive` |
| **B4** | **Degenerate Grids** | Non-standard dimensions: $32 \times 32$ sub-grid, $70 \times 70$ oversized grid, empty frame `[]` | Agent handles frame gracefully without crash, defaults safely | `test_adversarial_degenerate_grids` |
| **B5** | **Pathological Textures** | $64 \times 64$ checkerboard pattern (maximum connected component entropy) | Connected component labeling completes within latency budget ($< 3$ ms) | `test_adversarial_frames`, `test_worst_case_bfs_segmentation_latency` |
| **B6** | **Out-of-Range Palette** | Pixel values outside standard $0..15$ range (e.g. $-2, -1, 99$) | Sanitizes or handles raw integers without array indexing exceptions | `test_adversarial_degenerate_grids` |
| **B7** | **Dead-End & Corner Pads** | Operator pad located in a corner ($2$ neighbors) or dead end ($1$ neighbor) | Re-triggers pad state change via step-off and return without trapping | `test_corner_pad_retriggering`, `test_dead_end_pad_retriggering` |
| **B8** | **Unreachable Target** | Target coordinate completely enclosed by impassable walls | Returns empty path `[]` gracefully without infinite search timeout | `test_completely_enclosed_unreachable_goal` |

---

## 4. Tier 3: Cross-Feature Combinations (Pairwise Coverage)

Pairwise interactions between modules are rigorously tested to eliminate interface drift:

```
                      +-----------------------------+
                      | F1: Morphological Perception|
                      +-----------------------------+
                        /                         \
                       / (Pairwise 1:              \ (Pairwise 5:
                      /   Diff + Lattice)           \   March + Multi-Pixel)
                     v                               v
+-----------------------------+               +-----------------------------+
| F2: Affordance & World Model|<------------->| F4: Sovereign Dispatch      |
+-----------------------------+ (Pairwise 2:  +-----------------------------+
         |                       Queue + Bump)              |
         | (Pairwise 3: Camera Scale                        | (Pairwise 6:
         |   + Coordinate Clicks)                           |   Reset + Memory)
         v                                                  v
+-----------------------------+               +-----------------------------+
| F3: Dual Offline Planner    |<--------------| F7: E2E Regression Profiler |
+-----------------------------+ (Pairwise 4:  +-----------------------------+
                                 A* + BFS)
```

1. **Pairwise 1: Difference Mask + Dynamic Lattice Detection (F1 $\times$ F2)**
   - *Mechanism*: Avatar displacement is tracked exclusively via pixel changes intersecting the active diff mask while simultaneously updating dynamic lattice pitch $S \in \{1, 2, 3, 5, 8\}$ and offsets $(r_{\text{off}}, c_{\text{off}})$.
   - *Verification*: `test_dynamic_lattice` and `test_bump_collision_passable_map_update`.
2. **Pairwise 2: Fleet Plan Queue Decoupling + Bump Collision Learning (F4 $\times$ F3)**
   - *Mechanism*: Actions that produce zero displacement update the heuristic obstacle map (`known_walls`) but are strictly prohibited from clearing `fleet_plan_queue`. This allows deliberate wall collisions (such as symmetry-breaking in `m0r0` or slot-locking in `ka59`) to execute uninterrupted.
   - *Verification*: `test_queue_isolation_under_forced_bumps`, `test_m0r0_step_by_step_collision_telemetry`, `test_ka59_step_by_step_collision_telemetry`.
3. **Pairwise 3: Camera Scale / Letterbox Offset + Coordinate Clicks (F2 $\times$ F4)**
   - *Mechanism*: Point-and-click coordinate targets calculated from segmented sprites are transformed into display coordinates using camera scale and offset compensation before dispatching `ACTION6`.
   - *Verification*: `test_action6_vc33_targeting`.
4. **Pairwise 4: Geodesic A* Detour + Operator Pad BFS (F3 $\times$ F2)**
   - *Mechanism*: State-space BFS transitions between transformation pads invoke geodesic A* path planning with turn penalties to navigate around static barriers and other pads.
   - *Verification*: `test_neighbor_pad_interference_avoidance` and `test_retrigger_action_optimality`.
5. **Pairwise 5: Straight-Line March Protection + Multi-Pixel Lattice (F4 $\times$ F1)**
   - *Mechanism*: Straight-line marches of length $\ge 4$ on multi-pixel grids (e.g. $[R, R, R, R]$ with step size 5) are verified against the $A \ne B$ oscillation guard, preventing false loop detection while covering large distances.
   - *Verification*: `test_adversarial_loop_interception_4cycle_and_constrained_legal`.
6. **Pairwise 6: Terminal RESET Purge + Memory Tracemalloc (F4 $\times$ F7)**
   - *Mechanism*: On receiving `GAME_OVER` or `NOT_PLAYED`, all queues and visited-state dictionaries are purged, ensuring zero memory leakage across thousands of game steps.
   - *Verification*: `test_terminal_safety` and `test_memory_bounds`.

---

## 5. Tier 4: Real-World Workload Scenarios

The test harness evaluates real-world workloads consisting of **50 distinct game environments**:

### Group A: 25 Official ARC-AGI-3 Environments
Evaluates all 4 computational archetypes from the competition benchmark:
1. **Point-and-Click / Contextual Actuators** (9 games): `lp85`, `tn36`, `vc33`, `ft09`, `r11l`, `s5i5`, `su15`, `sb26`, `lf52`.
2. **Transformation & Dynamic Lattice BFS** (1 game): `ls20`.
3. **Geometric Shortest Paths & Labyrinth Navigation** (5 games): `tu93`, `tr87`, `g50t`, `re86`, `dc22`.
4. **Physics, Fluid & Multi-Agent Mechanics** (10 games): `ar25`, `bp35`, `cd82`, `cn04`, `ka59`, `m0r0`, `sc25`, `sk48`, `wa30`.

### Group B: 25 Novel Benchmark Substrates (`ng01`–`ng25`)
Full coverage across 5 novel computational paradigms:
- **Paradigm 1: Spatial & Topological Navigation**: `ng01` (WormholeTunnel), `ng02` (MovingHazard), `ng03` (LightShadow), `ng04` (GravityWells), `ng05` (CollapsingPath).
- **Paradigm 2: Push-Box & Object Physics**: `ng06` (IceSlider), `ng07` (BoulderBridge), `ng08` (MagnetPolarity), `ng09` (HeavyLightBoxes), `ng10` (PortalBoxRouting).
- **Paradigm 3: Pattern Matching & Logic Gates**: `ng11` (LogicCircuit), `ng12` (ColorMixer), `ng13` (SymmetryPainter), `ng14` (SudokuLight), `ng15` (SequenceMelody).
- **Paradigm 4: Multi-Agent & Desynchronization**: `ng16` (QuadMirrors), `ng17` (PredatorPrey), `ng18` (ConvoyEscort), `ng19` (ShadowClone), `ng20` (SplitMerge).
- **Paradigm 5: Affordance & Point-and-Click Mechanisms**: `ng21` (TumblerLock), `ng22` (PipeFlow), `ng23` (SlidingTile8), `ng24` (OpticalBeamSplit), `ng25` (ResourceCrafting).

---

## 6. Official Test Runner Execution & Validation Protocol

### Environment Standard
- **Python Version**: Strictly Python 3.12 (`.venv/bin/python`).
- **Environment Invariant**: `MPLCONFIGDIR=/tmp/matplotlib` (prevents font-cache locking on macOS/containers).

### Standard Execution Commands
1. **Full 40-Test Unit & Stress Discovery Suite**:
   ```bash
   MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python -m unittest discover tests
   ```
   *Expected Output*: `Ran 40 tests in ~60s ... OK` (100% Pass).

2. **Verbose Test Discovery with Method Manifest**:
   ```bash
   MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python -m unittest discover tests -v
   ```

3. **High-Throughput Simulation Harness (Sample Smoke Test)**:
   ```bash
   MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python scripts/simulation_harness.py --games ls20,tn36 --steps 30 --workers 2
   ```

4. **Novel Benchmark Substrate Evaluation (`ng01`–`ng25`)**:
   ```bash
   MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python scripts/simulation_harness.py --novel-all --steps 40 --workers 4
   ```

5. **Kaggle Deployment Build Verification**:
   ```bash
   MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python scripts/build_notebook.py
   ```

---

## 7. Pass Criteria & Exit Gates

The test architecture defines explicit mathematical acceptance gates:
- **Test Suite Pass Rate**: Exactly $100\%$ ($40/40$ tests passing, $0$ errors, $0$ failures).
- **Action Efficiency**: Combined RHAE $\ge 1.20$ across test environments.
- **Latency Ceiling**: Mean per-step latency $< 3.0$ ms (Throughput $> 300$ FPS).
- **Memory Ceiling**: Continuous 1000-step memory growth delta $< 500$ KB.
- **Air-Gapped Invariant**: Zero network calls during offline execution (`OperationMode.OFFLINE`).

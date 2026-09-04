# Project: ARC-AGI-3 Dual-Brain Autonomous Agent Version 3 (Template Matching & State Transformation)

## Architecture
The ARC-AGI-3 Version 3 agent in `agent/my_agent.py` is a high-efficiency autonomous agent engineered to solve template-matching and state-transformation puzzle games (such as `ls20`, where objects must be transformed on operator pads to match reference corner/HUD templates before entering goal receptacles) while preserving 100% backward compatibility and zero regressions for directional (`ACTION1`–`ACTION4`), contextual (`ACTION5`), and coordinate targeting (`ACTION6`, as in `vc33`) games.

The architecture comprises six coordinated macro-components in pure Python / NumPy (execution latency < 0.15 ms, throughput > 5000 FPS):

1. **Scene Perception (`ScenePerception`)**:
   - Converts `latest_frame.frame[0]` to standard 64x64 `np.int8` grid.
   - Detects dynamic background and canvas colors.
   - Segments non-background sprite components for generic games.
2. **Corner Goal Template & HUD Extraction (`TemplatePerception`)**:
   - Analyzes bottom-left HUD box at rows 55–60, cols 3–8 to extract active piece configuration $(S, C, R)$ scaled 2x.
   - Matches against canonical 3x3 binary shape primitives ($S \in \{0, 1, 2, 3, 4, 5\}$).
   - Identifies color index $C \in \{0, 1, 2, 3\}$ from palette `[12, 9, 14, 8]` and rotation $R \in \{0^\circ, 90^\circ, 180^\circ, 270^\circ\}$.
   - Decodes target goal template $(S^*, C^*, R^*)$ directly from inside goal receptacles (`rjlbuycveu`).
3. **Operator Pad & Affordance Transition Mapping (`OperatorPadDetector`)**:
   - Detects 5x5 discrete grid lattice ($x = 4 + 5 \cdot gx, y = 5 \cdot gy$).
   - Scans 5x5 tile patches to classify:
     - Rotation pads (`rhsxkxzdjz`): $R \leftarrow (R + 1) \pmod 4$ (90° clockwise).
     - Color swap pads (`soyhouuebz`): $C \leftarrow (C + 1) \pmod 4$ (cycles `[12, 9, 14, 8]`).
     - Shape morph pads (`mkjdaccuuf` / `ttfwljgohq`): $S \leftarrow (S + 1) \pmod 6$ (cycles 6 canonical shapes).
     - Goal receptacles (`rjlbuycveu`): tile coordinates and target template.
     - Static walls (`ihdgageizm`, color 4) and avatar starting position.
4. **State-Space Configuration Planner (`ConfigurationPlanner` / Unified Joint BFS)**:
   - Formulates configuration space as Abelian group $G = \mathbb{Z}_6 \times \mathbb{Z}_4 \times \mathbb{Z}_4$ ($|V| = 96$).
   - Computes minimal operator activations:
     $k_S = (S^* - S_0) \pmod 6$, $k_C = (C^* - C_0) \pmod 4$, $k_R = (R^* - R_0) \pmod 4$.
   - Executes Unified Joint BFS over state space $\mathcal{S} = (gx, gy, S, C, R)$ ($|\mathcal{S}| \le 13,824$ nodes) in $< 0.3$ ms.
   - Enforces the receptacle blocking invariant: avoids entering goal slots before $(S, C, R) == (S^*, C^*, R^*)$ to prevent collisions and stamina penalty.
   - Automatically handles pad re-triggering (step off to neighbor and back on).
5. **Mode-Gated Geodesic Execution Engine (`MyAgent` & `GeodesicPlanner`)**:
   - Mode Gate:
     - If puzzle elements (HUD + pads + goal receptacle) are detected: activates Version 3 State-Space Geodesic Planner.
     - If puzzle elements are absent: falls back 100% to Version 2 engine (`ACTION6` centroid targeting for `vc33`, `ACTION5` contextual interaction, Phase 1 micro-probing, obstacle A* geodesic navigation).
   - Popping pre-planned actions from FIFO queue maintains > 400 FPS.
6. **Integrity, Terminal Safety & Kaggle Gate (`SafetyEngine`)**:
   - Immediate `GameAction.RESET` on `GAME_OVER` / `NOT_PLAYED` (prevents HTTP 400 errors).
   - Strict Kaggle Gating: `make submit` is authorized ONLY after `make play-local GAME=ls20` completes with `levels_completed >= 1`.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | R1-CornerHUD | Bottom-left HUD extraction (rows 55–60, cols 3–8) for active piece (S, C, R) | M1 | ORIGINAL_REQUEST §R1, Survey 1, 2 |
| 2 | R1-CanonicalShapes | Canonical 3x3 binary shape primitives matching (6 shapes, 4 rotations) | M1 | ORIGINAL_REQUEST §R1, Survey 2 |
| 3 | R1-GoalTemplateExtract | Goal receptacle detection and target template (S*, C*, R*) extraction | M1 | ORIGINAL_REQUEST §R1, Survey 1, 2 |
| 4 | R2-GridLatticeMapping | 5x5 tile lattice coordinate quantization ($x = 4 + 5gx, y = 5gy$) | M2 | ORIGINAL_REQUEST §R2, Survey 1, 3 |
| 5 | R2-PadClassification | 5x5 operator pad detection (Rotation 90°, Color cycle, Shape morph) | M2 | ORIGINAL_REQUEST §R2, Survey 1, 2 |
| 6 | R2-AffordanceTransitions | Pad state transition operators and re-trigger step-off mechanics | M2 | ORIGINAL_REQUEST §R2, Survey 1, 3 |
| 7 | R3-ConfigDistance | Abelian configuration distance $(k_S, k_C, k_R)$ | M3 | ORIGINAL_REQUEST §R3, Survey 2, 3 |
| 8 | R3-UnifiedJointBFS | State-space BFS over $\mathcal{S} = (gx, gy, S, C, R)$ emitting minimal actions | M3 | ORIGINAL_REQUEST §R3, Survey 3 |
| 9 | R3-ReceptacleInvariant | Receptacle collision avoidance until $(S, C, R) == (S^*, C^*, R^*)$ | M3 | ORIGINAL_REQUEST §R3, Survey 1, 3 |
| 10 | R4-ModeGating | Mode gate preserving Version 2 (ACTION1–ACTION7, ACTION6 for vc33) | M3 | ORIGINAL_REQUEST §R4, Survey 2 |
| 11 | R5-LocalWinLs20 | `make play-local GAME=ls20` achieves `levels_completed >= 1` | M4 | ORIGINAL_REQUEST §R5, Survey 1, 2, 3 |
| 12 | R5-VerifyLocalFPS | `make verify-local` executes with zero errors and > 300 FPS | M4 | ORIGINAL_REQUEST §R5, Survey 2, 3 |
| 13 | R5-NotebookBuild | `make notebook` compiles `agent/my_agent.py` to `submission.ipynb` (t4 accelerator) | M4 | ORIGINAL_REQUEST §R5, Survey 3 |
| 14 | R5-GitPushAndSubmit | Git commit & push to `origin/main` and execute `make submit` gated on local win | M4 | ORIGINAL_REQUEST §R5, Survey 3 |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Template & HUD Perception | `TemplatePerception`: HUD active piece extraction, 6 canonical shapes, goal slot template decode | none | DONE |
| M2 | Operator Pad & Affordance Detection | `OperatorPadDetector`: 5x5 lattice mapping, rotation/color/shape pad classification, wall map | M1 | DONE |
| M3 | State-Space Configuration Planner & Integration | `ConfigurationPlanner` / Unified BFS on $\mathcal{S}$, waypoint queue, receptacle guard, mode gate in `MyAgent` | M1, M2 | DONE |
| M4 | Local-Win Benchmark & Gated Kaggle Submission | Verify `make play-local GAME=ls20` (`levels_completed >= 1`), `make verify-local` (>300 FPS), `make notebook`, git push, `make submit` | M1, M2, M3 | DONE |

*(Note: In accordance with our single-file architecture in `agent/my_agent.py`, Milestones M1-M3 are consolidated and verified as an integrated system, followed by Milestone M4 verification & gated submission).*

## Interface Contracts
### Engine ↔ Agent
- `agent.choose_action(frames: list[FrameData], latest_frame: FrameData) -> GameAction`:
  - Returns `GameAction` instance.
  - On `latest_frame.state in ("GAME_OVER", GameState.GAME_OVER, "NOT_PLAYED", GameState.NOT_PLAYED)`: returns `GameAction.RESET`.
  - Directional moves: `GameAction.ACTION1` (Up), `GameAction.ACTION2` (Down), `GameAction.ACTION3` (Left), `GameAction.ACTION4` (Right).
  - Contextual interaction: `GameAction.ACTION5`.
  - Coordinate tap targeting: `act = GameAction.ACTION6; act.set_data({"x": col, "y": row})`.
- Grid access: `np.array(latest_frame.frame[0], dtype=np.int8)`.
- Available actions: filtered by `latest_frame.available_actions`.

## Code Layout
- Exclusive Write Target: `/Users/zerbytheboss/.gemini/antigravity/scratch/ARC-AGI-3-Kaggle-Starter/agent/my_agent.py`
- Generated Deployment Artifact: `/Users/zerbytheboss/.gemini/antigravity/scratch/ARC-AGI-3-Kaggle-Starter/notebooks/submission.ipynb`
- Metadata: `/Users/zerbytheboss/.gemini/antigravity/scratch/ARC-AGI-3-Kaggle-Starter/notebooks/kernel-metadata.json`
- Coordination & State: `/Users/zerbytheboss/.gemini/antigravity/scratch/ARC-AGI-3-Kaggle-Starter/.agents/*`

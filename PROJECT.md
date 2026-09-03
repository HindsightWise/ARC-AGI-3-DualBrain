# Project: ARC-AGI-3 Dual-Brain Autonomous Agent Version 2

## Architecture
The ARC-AGI-3 Version 2 agent is a high-efficiency autonomous agent running inside `agent/my_agent.py`. It operates within a strict turn-based 64x64 grid environment under the Relative Human Action Efficiency (RHAE) metric where actions are penalized quadratically $(human / ai)^2$, while internal compute costs zero score.

The architecture comprises five macro-components implemented in pure Python / NumPy (no scipy/cv2/torch):
1. **Scene Perception (`ScenePerception`)**:
   - Frame grid extraction from `latest_frame.frame[0]` (shape: 64x64, int8).
   - Dynamic background color detection using histogram mode.
   - Connected-component segmentation (8-way or 4-way pure NumPy BFS) to detect discrete entity clusters and compute exact bounding boxes and centroids $(x, y) \in [0, 63]^2$.
   - Saliency ranking prioritizing non-background, distinct interactive sprites.
2. **Entity Classification & Avatar Isolation (`AvatarTracker`)**:
   - Two-Phase micro-probing to isolate avatar position and movement delta correlation.
   - Occupancy grid tracking (`passable_map`): detects collision bumps when an action yields identical avatar position, marking target cells as static walls.
3. **Cognitive Meta-Planner & Trajectory Engine (`GeodesicPlanner`)**:
   - **Requirement R1**: Direct straight-line Manhattan geodesic trajectories (|dr| + |dc| with zero jitter) via pre-planned action sequences cached in a FIFO queue.
   - Minimal convex detours via A* with turn-penalty metric when obstacles block direct lines of sight.
   - Multi-step commitment queue: popping from queue provides $O(1)$ action latency ($< 0.5$ ms per step), guaranteeing $> 500$ FPS.
4. **Complete Action Space Specializer (`ActionSpecializer`)**:
   - **Requirement R2**:
     - `ACTION1` - `ACTION4`: Directional execution popped from geodesic plan queue.
     - `ACTION5`: Contextual in-situ interaction when available and adjacent to interactables or during affordance probing.
     - `ACTION6`: Coordinate tap targeting with parameters `act = GameAction.ACTION6; act.set_data({"x": int(col), "y": int(row)})`.
5. **Anti-Fragility, Loop Interception & Terminal Safety (`SafetyEngine`)**:
   - **Requirement R3**:
     - State hashing using MD5 of grid bytes + avatar position.
     - N-gram action cycle detection to break ping-pong oscillations and force orthogonal re-routes.
     - Terminal Safety: Immediate `GameAction.RESET` when `latest_frame.state in ("GAME_OVER", GameState.GAME_OVER)`.
   - **Requirement R4**:
     - Full offline verification via Python 3.12 hermetic environment.
     - Packaging via `make notebook` to `notebooks/submission.ipynb` with `nvidiaTeslaT4` accelerator.
     - Git push to `origin/main` on `HindsightWise/ARC-AGI-3-DualBrain`.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | R1-GeodesicManhattan | Direct straight-line Manhattan action sequence without jitter | M1 | ORIGINAL_REQUEST §R1 |
| 2 | R1-ConvexDetour | Minimal convex detour via A* with turn penalty on obstacle encounter | M1 | ORIGINAL_REQUEST §R1 |
| 3 | R1-ActionQueueFPS | Trajectory queueing delivering O(1) step execution and >300 FPS | M1 | ORIGINAL_REQUEST §R1, Survey 3 |
| 4 | R2-Action5Interaction | Contextual interaction affordance testing and execution | M2 | ORIGINAL_REQUEST §R2 |
| 5 | R2-Action6Centroids | Sprite cluster segmentation and valid coordinate click targeting {"x": x, "y": y} in [0, 63] | M2 | ORIGINAL_REQUEST §R2, Survey 2 |
| 6 | R3-TwoPhasePlanning | Phase 1 micro-probing (avatar isolation) and Phase 2 goal pursuit | M3 | ORIGINAL_REQUEST §R3 |
| 7 | R3-OccupancyBumpMap | Real-time bump collision detection marking static walls | M3 | ORIGINAL_REQUEST §R3, Survey 1 |
| 8 | R3-LoopInterceptor | State hashing and n-gram action cycle detection breaking circular deadlocks | M3 | ORIGINAL_REQUEST §R3 |
| 9 | R3-TerminalSafety | Immediate GameAction.RESET on GAME_OVER preventing HTTP 400 | M3 | ORIGINAL_REQUEST §R3 |
| 10 | R4-VerifyLocalFPS | make verify-local executes with zero unhandled exceptions and >300 FPS | M4 | ORIGINAL_REQUEST §R4 |
| 11 | R4-NotebookPackaging | make notebook compiles agent/my_agent.py into notebooks/submission.ipynb with t4 accelerator | M4 | ORIGINAL_REQUEST §R4 |
| 12 | R4-GitPush | Git commit and push to origin/main on HindsightWise/ARC-AGI-3-DualBrain | M4 | ORIGINAL_REQUEST §R4 |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Geodesic Trajectories & Performance | Straight-line paths, minimal convex detour, action queue (>300 FPS) | none | IN_PROGRESS |
| M2 | Action Space Completeness | ACTION5 contextual interaction, ACTION6 centroid targeting in [0, 63] | M1 | IN_PROGRESS |
| M3 | Cognitive Integrity & Safety | Two-Phase probing, occupancy grid bump tracking, loop breaker, RESET on GAME_OVER | M1, M2 | IN_PROGRESS |
| M4 | Offline Verification & Packaging | Local smoke test (>300 FPS), notebook build, Git commit and push to origin/main | M1, M2, M3 | IN_PROGRESS |

*(Note: In accordance with our single-file architecture in `agent/my_agent.py`, Milestones M1-M3 are consolidated into the Version 2 implementation, verified under Milestone M4).*

## Interface Contracts
### Engine ↔ Agent
- `agent.choose_action(frames: list[FrameData], latest_frame: FrameData) -> GameAction`:
  - Returns `GameAction` instance.
  - On `latest_frame.state == "GAME_OVER"`: return `GameAction.RESET`.
  - For `GameAction.ACTION6`: `act = GameAction.ACTION6; act.set_data({"x": col, "y": row}); return act` where $col, row \in [0, 63]$.
  - For `GameAction.ACTION5`: `return GameAction.ACTION5`.
  - For `GameAction.ACTION1..4`: `return GameAction.ACTION[1..4]`.
- Grid access: `np.array(latest_frame.frame[0], dtype=np.int8)`.

## Code Layout
- Exclusive Write Target: `/Users/zerbytheboss/.gemini/antigravity/scratch/ARC-AGI-3-Kaggle-Starter/agent/my_agent.py`
- Generated Deployment Artifact: `/Users/zerbytheboss/.gemini/antigravity/scratch/ARC-AGI-3-Kaggle-Starter/notebooks/submission.ipynb`
- Metadata: `/Users/zerbytheboss/.gemini/antigravity/scratch/ARC-AGI-3-Kaggle-Starter/notebooks/kernel-metadata.json`
- Coordination & State: `/Users/zerbytheboss/.gemini/antigravity/scratch/ARC-AGI-3-Kaggle-Starter/.agents/*`

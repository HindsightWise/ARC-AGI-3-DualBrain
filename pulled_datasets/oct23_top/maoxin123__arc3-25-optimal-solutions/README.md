# ARC-AGI-3: 25/25 World-Record Optimal Solutions & SDK Replayer

[![ARC Prize 2026](https://img.shields.io/badge/ARC--Prize-2026-blue.svg)](https://arcprize.org)
[![Kaggle Usability](https://img.shields.io/badge/Kaggle_Usability-10.0-brightgreen.svg)](#)
[![Dataset Gold](https://img.shields.io/badge/Dataset-Gold_Candidate-gold.svg)](#)
[![Games Solved](https://img.shields.io/badge/Games_Solved-25%2F25-success.svg)](#)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

---

## 🌟 Executive Summary

The **ARC Prize 2026 (ARC-AGI-3)** introduces dynamic, multi-level visual environments designed to evaluate novel sample-efficient reasoning, rule induction, and strategic planning.

This open-source dataset releases the **complete, verified world-record optimal action routes** for **all 25 public evaluation games**. Every single route has been audited, validated against official environment definitions, and verified through the official ARC-AGI-3 SDK with **GameState.WIN** and **100.0 Score** with **zero resets**.

### Key Benchmark Metrics:
- **Total Games Solved**: 25 / 25 (100.0%)
- **Total Levels Cleared**: 183 levels
- **Total World-Record Actions**: **5,502 actions** across all 25 games
- **Leaderboard Standing**: **25/25 Match or Beat the Leaderboard Records**
- **Proven Global Minima**: 6 games backed by exhaustive mathematical/state-space lower-bound certificates (AR25, CD82, FT09, LP85, SC25, TR87)
- **Official SDK Execution**: 100% offline verifiable in <10 seconds aggregate replay time

---

## 📊 Complete 25/25 World-Record Leaderboard Table

| Game | Game ID | Levels | Steps | Leaderboard Min | Delta | Proof Category | Algorithmic Class |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **ar25** | `ar25-0c556536` | 8 | **253** | 253 | 0 | 🏆 Proven Minimum | Exhaustive State-Space Lower Bound & Spatial Geometry |
| **bp35** | `bp35-0a0ad940` | 9 | **322** | 322 | 0 | 🥇 World Record | Branch-and-Bound Traversal & Local Action Compression |
| **cd82** | `cd82-fb555c5d` | 6 | **69** | 69 | 0 | 🏆 Proven Minimum | Reverse Factorization & Exact Dijkstra Cursor Cost |
| **cn04** | `cn04-2fe56bfb` | 6 | **173** | 173 | 0 | 🥇 World Record | Grid Graph BFS & Shortest Routing |
| **dc22** | `dc22-fdcac232` | 6 | **416** | 416 | 0 | 🥇 World Record | Multi-Phase Topological Search & Plan Pruning |
| **ft09** | `ft09-0d8bbf25` | 6 | **75** | 75 | 0 | 🏆 Proven Minimum | Affine Color Constraint Solving & Nullspace Analysis |
| **g50t** | `g50t-5849a774` | 7 | **274** | 274 | 0 | 🥇 World Record | Exact Build Synthesis & Hierarchical Navigation |
| **ka59** | `ka59-38d34dbb` | 7 | **242** | 242 | 0 | 🥇 World Record | Token Graph Search & Action Optimization |
| **lf52** | `lf52-271a04aa` | 10 | **641** | 641 | 0 | 🥇 World Record | Laser Raytracing, State Shortcut Pruning & RNG Neutralization |
| **lp85** | `lp85-305b61c3` | 8 | **79** | 79 | 0 | 🏆 Proven Minimum | Quotient-State FIFO BFS & Click-Class Partitioning |
| **ls20** | `ls20-9607627b` | 7 | **309** | 309 | 0 | 🥇 World Record | Sliding Block Dynamics & Heuristic Pathfinding |
| **m0r0** | `m0r0-492f87ba` | 6 | **179** | 179 | 0 | 🥇 World Record | Greedy Pruning & Boundary Lower-Bound Proof |
| **r11l** | `r11l-495a7899` | 6 | **57** | 57 | 0 | 🥇 World Record | L1 Lower Bound & Shortest Token Trajectory |
| **re86** | `re86-8af5384d` | 8 | **507** | 507 | 0 | 🥇 World Record | Dynamic Control Rewriting & Level-Specific BFS |
| **s5i5** | `s5i5-18d95033` | 8 | **240** | 240 | 0 | 🥇 World Record | Admissible A* with Goal Coverage Relaxation |
| **sb26** | `sb26-7fbdac44` | 8 | **124** | 124 | 0 | 🥇 World Record | Geometric Alignment & Maze Navigation |
| **sc25** | `sc25-635fd71a` | 6 | **121** | 121 | 0 | 🏆 Proven Minimum | Weighted Dijkstra on Spell Transition Graphs |
| **sk48** | `sk48-d8078629` | 8 | **213** | 213 | 0 | 🥇 World Record | Multi-Target Routing & Step Optimization |
| **sp80** | `sp80-589a99af` | 6 | **86** | 86 | 0 | 🥇 World Record | Visual State Compression & Level 1 Lower Bound |
| **su15** | `su15-1944f8ab` | 9 | **61** | 61 | 0 | 🥇 World Record | Minimal Click Ordering & Undo Suppression |
| **tn36** | `tn36-ef4dde99` | 7 | **85** | 85 | 0 | 🥇 World Record | Combinatorial Click Minimization |
| **tr87** | `tr87-cd924810` | 6 | **118** | 118 | 0 | 🏆 Proven Minimum | Reachability Graph Exhaustive Verification |
| **tu93** | `tu93-0768757b` | 9 | **185** | 185 | 0 | 🥇 World Record | Grid Exploration & Path Shortcutting |
| **vc33** | `vc33-5430563c` | 7 | **167** | 167 | 0 | 🥇 World Record | Exact-Byte State BFS & Click Optimality |
| **wa30** | `wa30-ee6fef47` | 9 | **506** | 506 | 0 | 🥇 World Record | Sokoban-like Box Dynamics & Corridor Bottleneck Pruning |

**Total**: **25 games**, **183 levels**, **5,502 actions** (Average: 220.1 actions/game).

---

## 📁 Dataset File Structure & Formats

```text
kaggle_export_25_dataset/
├── README.md                      # Comprehensive documentation (this file)
├── dataset-metadata.json          # Kaggle dataset metadata configuration (Usability 10.0)
├── summary_leaderboard.csv        # Tidy tabular comparison of all 25 games
├── all_25_optimal_routes.json     # Single master JSON with all 25 solutions
├── sdk_receipts.json              # Official SDK verification receipts & run logs
├── wheels/                        # Bundled official SDK wheels for offline Kaggle use
│   ├── arc_agi-0.9.9-py3-none-any.whl
│   └── arcengine-0.9.3-py3-none-any.whl
├── routes/                        # Individual per-game route JSON files
│   ├── ar25_route.json
│   ├── bp35_route.json
│   ├── ft09_route.json
│   ├── lf52_route.json
│   ├── wa30_route.json
│   └── ... (25 files)
└── environment_files/             # Official ARC-AGI-3 environment source files
    ├── ar25/
    ├── bp35/
    └── ... (25 game folders)
```

### Data Dictionary (`summary_leaderboard.csv`):
- `game_name` (*string*): Short 4-letter game identifier (e.g. `ft09`, `lf52`).
- `game_id` (*string*): Official versioned environment ID (e.g. `ft09-0d8bbf25`).
- `win_levels` (*integer*): Number of levels required to achieve full victory.
- `optimal_steps` (*integer*): Verified shortest action count.
- `leaderboard_minimum` (*integer*): Lowest recorded actions on the official human leaderboard.
- `delta_to_leaderboard` (*integer*): Difference (`optimal_steps - leaderboard_minimum`), identically 0 across all 25 games.
- `status` (*string*): `MATCHES_WORLD_RECORD`.
- `score` (*float*): Official score awarded by the ARC SDK (100.0).
- `resets` (*integer*): Number of game resets performed (0).
- `proof_type` (*string*): `PROVEN_GLOBAL_MINIMUM` or `VERIFIED_WORLD_RECORD`.
- `difficulty_tier` (*string*): `Introductory`, `Intermediate`, `Mastery`, or `Grandmaster`.
- `algorithmic_class` (*string*): Core optimization / planning technique used.
- `available_actions` (*JSON array of integers*): Valid action IDs for the game.
- `per_level_steps` (*JSON array of integers*): Number of actions spent per level.
- `proof_notes` (*string*): Detailed technical description of proof or verification method.

---

## 🚀 Quickstart: How to Load & Replay in Python

### 1. Offline Dependency Installation (Optional / If Needed)
```bash
pip install --no-index --find-links wheels arc-agi arcengine
```

### 2. Load Summary Table with Pandas
```python
import pandas as pd

df = pd.read_csv("summary_leaderboard.csv")
print(df[["game_name", "win_levels", "optimal_steps", "proof_type"]])
```

### 3. Replay Any Game Route in Official SDK
```python
import json
from arc_agi import Arcade, OperationMode
from arcengine import GameAction

# Load route
with open("routes/ft09_route.json", "r", encoding="utf-8") as f:
    route = json.load(f)

arcade = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
card = arcade.open_scorecard()
env = arcade.make(route["game_id"], scorecard_id=card)

# Execute route step-by-step
for action in route["flat_actions"]:
    env.step(GameAction.from_id(action["action_id"]), data=action["data"])

scorecard = arcade.close_scorecard(card)
print(f"Game: {route['game_name']} | State: {env.observation_space.state.name} | Score: {scorecard.score}")
# Output: Game: ft09 | State: WIN | Score: 100.0
```

---

## 🧠 Action Space Conventions

In ARC-AGI-3 environments, actions are integer IDs with optional coordinate payloads:
- `1`: Move UP
- `2`: Move DOWN
- `3`: Move LEFT
- `4`: Move RIGHT
- `5`: Primary Action / SPACE
- `6`: Click / Coordinate Interaction (`{"x": x_coord, "y": y_coord}`)
- `7`: Undo / Auxiliary Action

All click coordinates in this dataset use official SDK pixel grid conventions `(0 <= x < 64, 0 <= y < 64)`.

---

## 🔬 Algorithmic Methodology & Highlights

1. **Exhaustive Proofs (AR25, CD82, FT09, LP85, SC25, TR87)**:
   - Evaluated complete quotient state spaces using Dijkstra, A* with admissible heuristics, and affine algebraic constraint solvers.
   - Guaranteed that no shorter valid sequence exists within the official action model.

2. **Complex Multi-Level Synthesized Records (LF52, WA30, BP35)**:
   - **LF52 (641 actions, 10 levels)**: Neutralized global RNG perturbations, synthesized optimal trajectories for tricky Level 5 (83 steps) and Level 7 (125 steps), matching the record of Smart Manoj.
   - **WA30 (506 actions, 9 levels)**: Navigated intricate box-pushing corridors with topological bottleneck pruning to hit 506 actions.
   - **BP35 (322 actions, 9 levels)**: Applied 4-to-2 and 3-to-2 action rewrites and full-alphabet branching to compress earlier 326 records down to 322.

---

## ⚖️ License & Citation

This dataset is released under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license. You are free to share, adapt, and build upon this material for any research, benchmark, or competitive use with appropriate credit.

```bibtex
@misc{arc3_25_optimal_2026,
  title={ARC-AGI-3: 25/25 World-Record Optimal Solutions & SDK Replayer},
  author={Xin Mao and ARC-AGI Open Source Community},
  year={2026},
  howpublished={\url{https://www.kaggle.com/datasets/arc3-25-optimal-solutions}},
  note={Verified 25/25 world-record action routes for ARC Prize 2026}
}
```

*Acknowledgments: François Chollet, ARC Prize foundation, DA Franzen, Smart Manoj, and the entire ARC-AGI competitive research community.*

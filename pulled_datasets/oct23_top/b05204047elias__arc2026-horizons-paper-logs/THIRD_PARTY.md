# Third-party code

This repository's own code is MIT-0 (see `LICENSE`). Third-party code keeps its
original license. Clones live under `third_party/` (git-ignored) at the commits below.

| Project | Upstream | Commit | License | Used for |
|---|---|---|---|---|
| ARC-AGI-3 Kaggle Starter | https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter | `eeb1535404f321d280a8f9194bbc1d7aca5f05fc` | **No LICENSE file in repo (unconfirmed)** | `make setup` / `play-local` / `submit` workflow |
| ARC-AGI-3-Agents (cloned by `make setup` into the starter's `vendor/`) | https://github.com/arcprize/ARC-AGI-3-Agents | `4743e7d0aaae0ded0d98a89a7e282e63564cd58b` | MIT | `agents.agent.Agent` base class |
| arc-agi-3-just-explore | https://github.com/dolphin-in-a-coma/arc-agi-3-just-explore | `c2d98318ecca38c8ee9fb49b5af159e51770bad7` | MIT (c) 2025 Evgenii Rudakov | GraphExplorer + FrameProcessor copied into `arc3/agent/my_agent.py` (sections 1–2, MIT notice kept in the file header) |
| arc-dsl | https://github.com/michaelhodel/arc-dsl | `635de4902a5fb4e376f27333feaa396d3f5dfdcb` | MIT (c) 2023 Michael Hodel | DSL and reference solvers for `arc2/` |
| stitch_core (PyPI 0.1.29) | https://github.com/mlb2251/stitch | PyPI wheel | MIT per upstream repo (GitHub API, 2026-10-21); the wheel itself ships no license file | library learning in `arc2/` |

Before any file from these projects is copied into `arc3/` or `arc2/`, keep its
license header and add the file path here.

Files containing third-party code:

- `arc3/agent/my_agent.py` — sections 1–2 from arc-agi-3-just-explore (`graph_explorer.py` lines 15–451, `agents/heuristic_agent.py` lines 491–898 minus `visualize_components`). Changes are marked `[port]`. Section 3 (`MyAgent`) is MIT-0.

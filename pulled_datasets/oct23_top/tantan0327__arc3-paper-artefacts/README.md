Artefacts for the ARC Prize 2026 Paper Prize entry "The agent starts like an agent and stops like a coin flip".
- data/traces.json: 213 real trajectories of the Milestone-2 chassis (per action: generated tokens, wall, level clears) — input of scripts/sched_sim.py
- data/human_completion_captures/: board before/after each of the 168 human level completions (from the official replays via scripts/replay_humans.py)
- results/: measurement outputs (theory induction, planner, UNDO, scheduler simulator, token-cost projection) and the paper's figures
- scripts/: the CPU-only measurement scripts (simulator, theory tests, figures)
- paper.pdf: the full paper
Code is MIT-0 / CC0 by the author; data derives from our own runs and the public game engine.

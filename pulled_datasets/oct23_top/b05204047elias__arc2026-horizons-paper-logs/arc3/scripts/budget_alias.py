# SPDX-License-Identifier: MIT-0
"""Does MyAgent's status-bar mask hide the step budget? (analysis only, engine objects, no server)

Builds two level-1 states that differ only in the step counter (the second one spent k clicks on a
background pixel whose only effect is the counter), checks that MyAgent's masked frames are identical,
then plays the same random engine-list actions from both and records where the observations
(masked frame, game state, level) first diverge.

    .venv/bin/python arc3/scripts/budget_alias.py --games vc33
"""
import argparse
import copy
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "arc3" / "scripts"))
import oracle_probe as o  # noqa: E402
from arcengine import ActionInput, GameAction, GameState  # noqa: E402

from common import schema  # noqa: E402

COUNTER_ONLY = ("current_steps", "_interfaces")


def background_pixel(game):
    """First pixel (row-major) whose click changes nothing but the step counter and its display."""
    for y in range(64):
        for x in range(64):
            c = copy.deepcopy(game)
            o.act(c, ActionInput(id=GameAction.ACTION6, data={"x": x, "y": y}))
            diff = o.state_diff(game, c)
            if diff and all(any(t in f for t in COUNTER_ONLY) for f in diff):
                return (x, y), diff
    return None, None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--games", default="vc33")
    p.add_argument("--spent", default="5,15", help="extra background clicks for the second state")
    p.add_argument("--sequences", type=int, default=20)
    p.add_argument("--length", type=int, default=80)
    p.add_argument("--out", default=str(ROOT / "runs" / "2026-10-21_budget-alias.jsonl"))
    args = p.parse_args(argv)
    fproc = o.agent_frame_processor()
    rows = []
    for g in args.games.split(","):
        game, _ = o.load_game(g)
        f0 = o.render(game)
        seg, segs = fproc.segment_frame(f0.copy())
        _, mask = fproc.identify_status_bars(seg, segs)
        bg, diff = background_pixel(game)
        if bg is None:
            rows.append({"game": g, "background_pixel": None, **schema.provenance()})
            continue
        for k in map(int, args.spent.split(",")):
            b = copy.deepcopy(game)
            for _ in range(k):
                o.act(b, ActionInput(id=GameAction.ACTION6, data={"x": bg[0], "y": bg[1]}))
            final_diff = o.state_diff(game, b)
            counter = [f for f in final_diff if "current_steps" in f]
            steps_before = [o._get_path(game, f) for f in counter]
            steps_after = [o._get_path(b, f) for f in counter]
            ok = (b._state not in (GameState.GAME_OVER, GameState.WIN) and b._score == game._score
                  and all(any(t in f for t in COUNTER_ONLY) for f in final_diff)
                  and all(x - y == k for x, y in zip(steps_before, steps_after)) and counter)
            rng = random.Random(k)
            first, stops = [], {}
            for _ in range(args.sequences):
                a1, b1 = copy.deepcopy(game), copy.deepcopy(b)
                stop = "length"
                for t in range(args.length):
                    acts = o.valid_actions(a1)
                    if not acts:
                        stop = "no_actions"
                        break
                    a = rng.choice(acts)
                    r1, r2 = o.act(a1, a), o.act(b1, a)
                    same_level = a1._score == game._score and b1._score == game._score
                    if ((r1.state, a1._score) != (r2.state, b1._score)
                            or (same_level and o.masked_hash(o.render(a1), mask) != o.masked_hash(o.render(b1), mask))):
                        first.append({"step": t + 1, "full_budget": r1.state.name, "spent_budget": r2.state.name})
                        stop = "diverged"
                        break
                    if not same_level:
                        stop = "level_up"
                        break
                    if r1.state in (GameState.GAME_OVER, GameState.WIN):
                        stop = r1.state.name.lower()
                        break
                stops[stop] = stops.get(stop, 0) + 1
            rows.append({"game": g, "background_pixel": bg, "background_diff": diff, "spent": k,
                         "pair_valid": bool(ok), "pair_final_diff": final_diff,
                         "steps_full": steps_before, "steps_spent": steps_after, "stop_reasons": stops,
                         "masked_frames_identical": o.masked_hash(f0, mask) == o.masked_hash(o.render(b), mask),
                         "raw_frames_identical": bool((f0 == o.render(b)).all()),
                         "sequences": args.sequences, "length": args.length,
                         "diverged": len(first), "first_divergence": first, **schema.provenance()})
            print(json.dumps({k2: v for k2, v in rows[-1].items() if k2 not in ("cmd", "created_utc")}), flush=True)
    with open(args.out, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()

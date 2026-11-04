# SPDX-License-Identifier: MIT-0
"""Clicks that change engine state but not the frame (analysis only; engine objects, no server).

At each game's level-1 start, click a few background positions and record: whether the returned
frame differs from the start frame, which fields differ from the start state, and which fields
differ between two clicks whose frames are identical. Keys are unabstracted (no budget skipping).

    .venv/bin/python arc3/scripts/hidden_click_fields.py --games ft09,su15,tn36
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "arc3" / "scripts"))
import oracle_probe as o  # noqa: E402
from arcengine import ActionInput, GameAction  # noqa: E402

from common import schema  # noqa: E402

POINTS = ((2, 2), (6, 2), (2, 6), (30, 30), (34, 30))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--games", default="ft09,su15,tn36")
    p.add_argument("--out", default=str(ROOT / "runs" / "2026-10-22_hidden-click-fields.jsonl"))
    args = p.parse_args(argv)
    for g in args.games.split(","):
        game, _ = o.load_game(g)
        start = o.render(game)
        clicks = []
        for x, y in POINTS:
            c = copy.deepcopy(game)
            raw = o.act(c, ActionInput(id=GameAction.ACTION6, data={"x": x, "y": y}))
            frame = np.asarray(raw.frame[-1], dtype=np.uint8) if raw.frame else None
            clicks.append({"xy": [x, y], "frame_changed_pixels": None if frame is None else int((frame != start).sum()),
                           "fields_changed": o.state_diff(game, c), "_game": c, "_frame": frame})
        pairs = []
        for i in range(len(clicks)):
            for j in range(i + 1, len(clicks)):
                a, b = clicks[i], clicks[j]
                if a["_frame"] is not None and b["_frame"] is not None and np.array_equal(a["_frame"], b["_frame"]):
                    pairs.append({"a": a["xy"], "b": b["xy"], "frames_identical": True,
                                  "fields_differ": o.state_diff(a["_game"], b["_game"])})
        row = {"game": g, "key_deterministic": o.fingerprint(game) == o.fingerprint(copy.deepcopy(game)),
               "distinct_keys_after_background_clicks": len({o.fingerprint(c["_game"]) for c in clicks}),
               "clicks": [{k: v for k, v in c.items() if not k.startswith("_")} for c in clicks],
               "identical_frame_pairs": pairs,
               "identical_frame_pairs_with_state_difference": sum(1 for q in pairs if q["fields_differ"]),
               **schema.provenance()}
        print(json.dumps({k: v for k, v in row.items() if k in ("game", "identical_frame_pairs_with_state_difference")}
                         | {"pairs": len(pairs)}), flush=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()

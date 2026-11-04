# SPDX-License-Identifier: MIT-0
"""Oracle feasibility probe (2026-10-21): can the local engine be cloned and searched exhaustively?

Analysis only: this drives the game objects directly (OFFLINE), never a game server, so nothing
here is a score (CLAUDE.md). For each game it checks
  1. clone fidelity: copy.deepcopy(game) then the same actions on original and clone must give
     identical frames, state and levels_completed;
  2. the engine's own action set: _get_valid_actions() (never exposed to agents) per state, audited
     by clicking all 4096 pixels in a few states;
  3. a capped BFS on level 1 over engine fingerprints, with the engine list plus an ACTION6 grid:
     reachable states, shortest solution within that action set vs the human baseline, outcome
     classes per state, successor agreement for states merged by the fingerprint, and aliasing
     (several engine states behind one frame), both on raw frames and on frames as MyAgent hashes
     them (status bars masked).

    .venv/bin/python arc3/scripts/oracle_probe.py --games lp85,r11l,vc33
"""
import argparse
import copy
import hashlib
import json
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from arc_agi import Arcade, OperationMode  # noqa: E402
from arcengine import ActionInput, GameAction, GameState  # noqa: E402
from arcengine.base_game import ARCBaseGame  # noqa: E402
from arcengine.level import Level  # noqa: E402

from common import schema  # noqa: E402

ENVIRONMENTS = ROOT / "third_party" / "ARC-AGI-3-Kaggle-Starter" / "environment_files"
sys.path.insert(0, str(ROOT / "third_party" / "ARC-AGI-3-Kaggle-Starter" / "vendor" / "ARC-AGI-3-Agents"))  # MyAgent imports agents.*


def load_game(game: str):
    arcade = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=str(ENVIRONMENTS))
    env = arcade.make(game)
    info = next(e for e in arcade.available_environments if e.game_id.startswith(game))
    env.reset()
    return env._game, list(info.baseline_actions or [])


def act(game, a: ActionInput):
    raw = game.perform_action(ActionInput(id=a.id, data=dict(a.data or {})), raw=True)
    return raw


def frame_hash(raw) -> str:
    if not raw.frame:
        return "empty"
    return hashlib.blake2b(np.asarray(raw.frame[-1], dtype=np.uint8).tobytes(), digest_size=12).hexdigest()


VOLATILE = {"_action_count", "_full_reset", "_action"}  # engine bookkeeping that changes on every action
# (_action_count matters only for RESET, which the level-1 BFS never takes)
TEMPLATES = {"_levels", "_clean_levels"}  # all levels; the current one is encoded via current_level
LEVEL_CACHES = {"_need_sort", "_sorted_sprites"}  # render-order caches on Level
# Attributes skipped inside every object when a caller abstracts the step budget away (oracle_levels.py
# --abstract-budget): the engine's step counters keep it in `current_steps`, drawn via camera interfaces.
SKIP_ATTRS: set = set()


_ORDERING: set = set()  # ids of objects whose members are being ordered (breaks cycles in ordering digests)


def _put(h, tag: bytes, payload: bytes = b"") -> None:
    h.update(tag + len(payload).to_bytes(8, "little") + payload)  # type tag + length: no concatenation collisions


def _encode(obj, h, refs, depth=0):
    """Canonical serialization of an engine object graph into hash h. Every token carries a type
    tag and a length; an object met again is written as a back-reference to its first-visit index,
    which is canonical because traversal order is fixed (dict and set members ordered by content)."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        _put(h, b"P" + type(obj).__name__.encode(), repr(obj).encode())
        return
    if id(obj) in refs:
        _put(h, b"R", str(refs[id(obj)]).encode())
        return
    if id(obj) in _ORDERING:  # a cycle back to a container being ordered: content order cannot depend on it
        _put(h, b"C")
        return
    if isinstance(obj, np.ndarray):
        refs[id(obj)] = len(refs)
        _put(h, b"A", f"{obj.shape}{obj.dtype}".encode() + np.ascontiguousarray(obj).tobytes())
        return
    if depth > 60:
        raise RecursionError("object graph deeper than 60")
    refs[id(obj)] = len(refs)
    if isinstance(obj, (list, tuple)):
        _put(h, b"L" + type(obj).__name__.encode(), str(len(obj)).encode())
        for x in obj:
            _encode(x, h, refs, depth + 1)
    elif isinstance(obj, dict):
        # Keys may be objects whose repr carries a memory address, so members are ordered by content
        # (key digest, then key+value digest); members are then encoded inline with the shared
        # reference table, so references across containers are kept. Members identical in content
        # remain tied; their order can only matter through object identity elsewhere.
        _put(h, b"D", str(len(obj)).encode())
        _ORDERING.add(id(obj))
        try:
            kd = {id(k): _digest(k) for k in obj}
            tied = {d for d in kd.values() if list(kd.values()).count(d) > 1}
            order = sorted(obj, key=lambda k: (kd[id(k)], _digest((k, obj[k])) if kd[id(k)] in tied else ""))
        finally:
            _ORDERING.discard(id(obj))
        for k in order:
            _put(h, b"K")
            _encode(k, h, refs, depth + 1)
            _encode(obj[k], h, refs, depth + 1)
    elif isinstance(obj, (set, frozenset)):
        _put(h, b"S" + type(obj).__name__.encode(), str(len(obj)).encode())
        _ORDERING.add(id(obj))
        try:
            members = sorted(obj, key=_digest)
        finally:
            _ORDERING.discard(id(obj))
        for x in members:
            _encode(x, h, refs, depth + 1)
    elif hasattr(obj, "__dict__"):
        # the game and its levels can also be reached through ordinary fields (e.g. a sprite that
        # keeps a handle on the game): apply the same exclusions there as at the top level
        skip = SKIP_ATTRS | (VOLATILE | TEMPLATES if isinstance(obj, ARCBaseGame) else set()) \
            | (LEVEL_CACHES if isinstance(obj, Level) else set())
        items = [(k, v) for k, v in sorted(vars(obj).items())
                 if (not callable(v) or hasattr(v, "__dict__")) and k not in skip]
        _put(h, b"O" + type(obj).__name__.encode(), str(len(items)).encode())
        for k, v in items:
            _put(h, b"F", k.encode())
            _encode(v, h, refs, depth + 1)
    else:
        _put(h, b"X" + type(obj).__name__.encode(), repr(obj).encode())


_CACHE: dict | None = None  # id -> ordering digest, live only inside one fingerprint() call


def _digest(obj) -> str:
    if _CACHE is not None and id(obj) in _CACHE:
        return _CACHE[id(obj)][1]
    h = hashlib.blake2b(digest_size=16)
    _encode(obj, h, {})
    d = h.hexdigest()
    if _CACHE is not None:
        _CACHE[id(obj)] = (obj, d)  # keep obj alive so its id is not reused within the call
    return d


def fingerprint(game) -> str:
    """Conservative state key. The serialization written into the hash is typed, length-prefixed and
    reference-numbered, so two keys are equal only if the encoded object graphs are equal (up to
    hash collisions): distinct engine states are never merged. Member order inside dicts and sets
    is chosen by content digests that can depend on insertion order and on cycles, so one engine
    state can get more than one key (over-splitting). Over-splitting inflates state counts and
    aliasing groups but cannot shorten or invalidate h*; aliasing is only counted when futures
    diverge. Engine-side state: every game field (minus bookkeeping and level templates) plus the current level."""
    global _CACHE
    _CACHE = {}
    try:
        return _fingerprint(game)
    finally:
        _CACHE = None


def _fingerprint(game) -> str:
    h = hashlib.blake2b(digest_size=16)
    refs = {id(game): 0, id(game.current_level): 1}  # fields pointing back at either become references
    for k, v in sorted(vars(game).items()):
        if k not in VOLATILE and k not in TEMPLATES:
            _put(h, b"G", k.encode())
            _encode(v, h, refs)
    for k, v in sorted(vars(game.current_level).items()):
        if k not in LEVEL_CACHES:
            _put(h, b"V", k.encode())
            _encode(v, h, refs)
    return h.hexdigest()


def action_key(a: ActionInput):
    d = a.data or {}
    return (int(getattr(a.id, "value", a.id)), d.get("x"), d.get("y"))


def valid_actions(game):
    return [a for a in game._get_valid_actions() if action_key(a)[0] != GameAction.RESET.value]


def agent_frame_processor():
    """The agent's own FrameProcessor, to hash frames the way MyAgent sees them (status bars masked)."""
    import importlib.util
    path = ROOT / "arc3" / "agent" / "my_agent.py"
    spec = importlib.util.spec_from_file_location("arc3_my_agent_oracle", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.FrameProcessor()


def render(game) -> np.ndarray:
    return np.asarray(game.camera.render(game.current_level.get_sprites()), dtype=np.uint8)


def masked_hash(frame: np.ndarray, mask) -> str:
    f = frame.copy()
    if mask is not None:
        f[mask] = 0  # MyAgent sets masked cells to 16 and later to 0 before hashing
    return hashlib.blake2b(f.tobytes(), digest_size=12).hexdigest()


def random_action(game, rng):
    r = rng.random()
    if r < 0.1:
        return ActionInput(id=GameAction.RESET)
    acts = valid_actions(game)
    if r < 0.4 and 6 in game._available_actions:
        return ActionInput(id=GameAction.ACTION6, data={"x": rng.randrange(64), "y": rng.randrange(64)})
    return rng.choice(acts) if acts else ActionInput(id=GameAction.RESET)


def clone_fidelity(game, rng, steps=400, every=20, horizon=10):
    """Clone, then drive original and clone with the same mixed actions (engine list, arbitrary
    pixels, RESET); compare frames, state, level and the engine fingerprint after every step.
    Also checks branch independence: stepping the clone must not change the original."""
    checks = nonterminal = mismatches = leaks = 0
    copy_s = []
    for t in range(steps):
        if game._state in (GameState.GAME_OVER, GameState.WIN):
            act(game, ActionInput(id=GameAction.RESET))
            continue
        if t % every:
            act(game, random_action(game, rng))
            continue
        t0 = time.perf_counter()
        twin = copy.deepcopy(game)
        copy_s.append(time.perf_counter() - t0)
        before = fingerprint(game)
        probe = copy.deepcopy(twin)
        for _ in range(3):
            act(probe, random_action(probe, rng))
        leaks += int(fingerprint(game) != before)
        for _ in range(horizon):
            a = random_action(game, rng)
            r1, r2 = act(game, a), act(twin, a)
            same = (r1.state == r2.state and r1.levels_completed == r2.levels_completed
                    and len(r1.frame) == len(r2.frame)
                    and all(np.array_equal(x, y) for x, y in zip(r1.frame, r2.frame))
                    and fingerprint(game) == fingerprint(twin))
            checks += 1
            nonterminal += int(a.id != GameAction.RESET and bool(r1.frame)
                               and r1.state not in (GameState.GAME_OVER, GameState.WIN))
            mismatches += int(not same)
            if r1.state in (GameState.GAME_OVER, GameState.WIN):
                break
    return {"clone_checks": checks, "clone_checks_nonterminal": nonterminal, "clone_mismatches": mismatches,
            "clone_leaks": leaks, "deepcopy_seconds_mean": float(np.mean(copy_s)) if copy_s else None}


def candidate_actions(game, pixel_step):
    acts = valid_actions(game)
    if pixel_step and 6 in game._available_actions:
        h = pixel_step // 2
        acts = acts + [ActionInput(id=GameAction.ACTION6, data={"x": x, "y": y})
                       for y in range(h, 64, pixel_step) for x in range(h, 64, pixel_step)]
    return acts, len(valid_actions(game))


def divergence(frame_groups, reps, rng, mask, groups=60, sequences=20, length=6):
    """For frames shared by several engine states, does the hidden difference ever show? Two states
    of a group get the same random action sequences; the group is divergent if, after some prefix,
    the observed frames (masked like MyAgent when mask is given) or the level/state differ."""
    def observe(g, raw, level0):
        if not raw.frame or g._score != level0:  # new level: MyAgent re-derives its mask, so frames are not comparable
            return ("-", raw.state, g._score)
        f = np.asarray(raw.frame[-1], dtype=np.uint8)
        return (masked_hash(f, mask), raw.state, g._score)
    tested = divergent = 0
    fields = {}  # top-level fields that differ between the two states of a divergent group
    for fps in [v for v in frame_groups.values() if len(v) > 1][:groups]:
        a_fp, b_fp = sorted(fps)[:2]
        tested += 1
        hit = False
        level0 = reps[a_fp]._score
        for _ in range(sequences):
            a, b = copy.deepcopy(reps[a_fp]), copy.deepcopy(reps[b_fp])
            for _ in range(length):
                acts = valid_actions(a) or [ActionInput(id=GameAction.RESET)]
                act_ = rng.choice(acts) if rng.random() < 0.7 else ActionInput(
                    id=GameAction.ACTION6, data={"x": rng.randrange(64), "y": rng.randrange(64)})
                oa, ob = observe(a, act(a, act_), level0), observe(b, act(b, act_), level0)
                if oa != ob:
                    hit = True
                    break
                if a._state in (GameState.GAME_OVER, GameState.WIN) or a._score != level0:
                    break  # terminal, or a new level (MyAgent re-derives its mask there)
            if hit:
                break
        divergent += int(hit)
        if hit:
            for f in state_diff(reps[a_fp], reps[b_fp]):
                fields[f] = fields.get(f, 0) + 1
    return divergent, tested, fields


def _get_path(game, field):
    """Value of a state_diff field name such as 'obj.attr' or 'L.attr'."""
    obj = game.current_level if field.startswith("L.") else game
    for part in field.removeprefix("L.").split("."):
        obj = getattr(obj, part)
    return obj


def state_diff(a, b):
    """Names of game fields (and Level fields, prefixed L.) whose encoding differs; for plain objects
    one level deeper (field.subfield)."""
    out = []
    for prefix, x, y, skip in (("", a, b, VOLATILE | TEMPLATES), ("L.", a.current_level, b.current_level, LEVEL_CACHES)):
        for k in sorted(set(vars(x)) | set(vars(y))):
            v, w = vars(x).get(k), vars(y).get(k)
            if k in skip or _digest(v) == _digest(w):
                continue
            if hasattr(v, "__dict__") and hasattr(w, "__dict__") and type(v).__module__ != "builtins":
                sub = [kk for kk in sorted(set(vars(v)) | set(vars(w)))
                       if _digest(vars(v).get(kk)) != _digest(vars(w).get(kk))]
                out += [f"{prefix}{k}.{kk}" for kk in sub] or [prefix + k]
            else:
                out.append(prefix + k)
    return out


def bfs_level1(game, max_nodes, max_seconds, pixel_step, fproc, rng):
    """BFS over engine states on the current level. Actions = the engine's list plus an ACTION6 grid
    (every pixel_step pixels); outcomes are deduplicated per state, so h* is the shortest solution
    within that action set (not a proven global optimum when position matters below the grid)."""
    t0 = time.perf_counter()
    start_level = game._score
    root = copy.deepcopy(game)
    root_frame = render(root)
    seg, segs = fproc.segment_frame(root_frame.copy())
    _, mask = fproc.identify_status_bars(seg, segs)
    rfp = fingerprint(root)
    reps = {rfp: root}
    depth_of = {rfp: 0}
    raw_frames = {hashlib.blake2b(root_frame.tobytes(), digest_size=12).hexdigest(): {rfp}}
    agent_frames = {masked_hash(root_frame, mask): {rfp}}
    fp_frames = {rfp: {next(iter(raw_frames))}}
    queue = deque([rfp])
    h_star, capped = None, False
    k_valid, classes, grid_only, succ_checks, succ_mismatch = [], [], [], 0, 0
    loop_checks = loop_mismatch = 0
    while queue:
        if len(reps) >= max_nodes or time.perf_counter() - t0 > max_seconds:
            capped = True
            break
        fp = queue.popleft()
        node, depth = reps[fp], depth_of[fp]
        acts, n_valid = candidate_actions(node, pixel_step)
        k_valid.append(n_valid)
        outcomes, from_valid = {}, set()
        for i, a in enumerate(acts):
            child = copy.deepcopy(node)
            raw = act(child, a)
            if child._score > start_level:
                key = "LEVEL_UP"
                h_star = depth + 1 if h_star is None else h_star
            elif raw.state == GameState.GAME_OVER:
                key = "GAME_OVER"
            else:
                key = fingerprint(child)
                if key == fp and loop_checks < 300 and rng.random() < 0.05:  # self-loop merge: check it too
                    loop_checks += 1
                    b = rng.choice(acts)
                    c1, c2 = copy.deepcopy(node), copy.deepcopy(child)
                    act(c1, b), act(c2, b)
                    loop_mismatch += int(fingerprint(c1) != fingerprint(c2))
                if key != fp:
                    frame = render(child)
                    rh = hashlib.blake2b(frame.tobytes(), digest_size=12).hexdigest()
                    raw_frames.setdefault(rh, set()).add(key)
                    agent_frames.setdefault(masked_hash(frame, mask), set()).add(key)
                    fp_frames.setdefault(key, set()).add(rh)
                    if key not in reps:
                        reps[key] = child
                        depth_of[key] = depth + 1
                        queue.append(key)
                    elif succ_checks < 300:  # same key reached again: successors must agree
                        succ_checks += 1
                        b = rng.choice(acts)
                        c1, c2 = copy.deepcopy(reps[key]), copy.deepcopy(child)
                        act(c1, b), act(c2, b)
                        succ_mismatch += int(fingerprint(c1) != fingerprint(c2))
            outcomes.setdefault(key, i)
            if i < n_valid:
                from_valid.add(key)
        classes.append(len(outcomes))
        grid_only.append(len(set(outcomes) - from_valid))
    raw_div, raw_tested, raw_fields = divergence(raw_frames, reps, rng, mask=None)
    agent_div, agent_tested, agent_fields = divergence(agent_frames, reps, rng, mask=mask)
    nonterminal = list(reps)
    return {
        "action_set": f"engine list + ACTION6 grid step {pixel_step}" if pixel_step else "engine list",
        "states_found": len(reps), "capped": capped, "h_star_in_action_set": h_star,
        "k_valid_mean": float(np.mean(k_valid)) if k_valid else None,
        "outcome_classes_mean": float(np.mean(classes)) if classes else None,
        "grid_only_classes_mean": float(np.mean(grid_only)) if grid_only else None,
        "raw_frames": len(raw_frames), "raw_aliased_frames": sum(len(v) > 1 for v in raw_frames.values()),
        "agent_frames": len(agent_frames), "agent_aliased_frames": sum(len(v) > 1 for v in agent_frames.values()),
        "states_sharing_agent_frame": sum(len(v) for v in agent_frames.values() if len(v) > 1),
        "raw_aliased_tested": raw_tested, "raw_aliased_divergent": raw_div,
        "agent_aliased_tested": agent_tested, "agent_aliased_divergent": agent_div,
        "raw_divergent_fields": raw_fields, "agent_divergent_fields": agent_fields,
        "underresolved_fingerprints": sum(len(v) > 1 for v in fp_frames.values()),
        "successor_checks": succ_checks, "successor_mismatches": succ_mismatch,
        "self_loop_checks": loop_checks, "self_loop_mismatches": loop_mismatch,
        "bfs_seconds": time.perf_counter() - t0,
    }, [reps[k] for k in nonterminal]


def click_audit(states, rng, n=3):
    """Click all 4096 pixels in n BFS-discovered (non-terminal) states; count outcome classes that no
    engine-list action reaches (the list would then be an incomplete action space)."""
    def outcome(g, a):
        c = copy.deepcopy(g)
        raw = act(c, a)
        return (fingerprint(c), raw.state, c._score)
    out = []
    for g in rng.sample(states, min(n, len(states))):
        base = (fingerprint(g), g._state, g._score)
        valid_outcomes = {outcome(g, a) for a in valid_actions(g)}
        pixel_outcomes = {}
        for y in range(64):
            for x in range(64):
                o = outcome(g, ActionInput(id=GameAction.ACTION6, data={"x": x, "y": y}))
                if o != base:
                    pixel_outcomes[o] = pixel_outcomes.get(o, 0) + 1
        missing = [k for o, k in pixel_outcomes.items() if o not in valid_outcomes]
        out.append({"valid_actions": len(valid_actions(g)), "effective_pixels": sum(pixel_outcomes.values()),
                    "pixel_effect_classes": len(pixel_outcomes),
                    "classes_missing_from_valid": len(missing), "pixels_missing_from_valid": sum(missing)})
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--games", default="lp85,r11l,vc33")
    p.add_argument("--max-nodes", type=int, default=5000)
    p.add_argument("--max-seconds", type=float, default=300.0)
    p.add_argument("--pixel-step", type=int, default=4, help="ACTION6 grid added to the engine list; 0 = list only")
    p.add_argument("--out", default=str(ROOT / "runs" / "2026-10-21_oracle-probe.jsonl"))
    args = p.parse_args(argv)
    fproc = agent_frame_processor()
    rows = []
    for g in args.games.split(","):
        game, baseline = load_game(g)
        start_valid = valid_actions(game)
        clone = clone_fidelity(copy.deepcopy(game), random.Random(0))
        bfs, states = bfs_level1(game, args.max_nodes, args.max_seconds, args.pixel_step, fproc, random.Random(2))
        audit = click_audit(states, random.Random(1)) if 6 in game._available_actions else []
        row = {"game": g, "h_baseline_level1": baseline[0] if baseline else None,
               "valid_actions_at_start": len(start_valid), **clone, **bfs, "click_audit": audit,
               **schema.provenance()}
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k not in ("cmd", "created_utc")}), flush=True)
    with open(args.out, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()

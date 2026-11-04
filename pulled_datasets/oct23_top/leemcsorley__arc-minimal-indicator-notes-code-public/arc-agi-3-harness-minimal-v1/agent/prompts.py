"""Prompts: the system prompt (two strategy variants sharing all tools and rules) and the level-start message.

PROMPT_VARIANT=objects -> objects-first: interpret lib.extract_components()/lib.frame_diff(), act in short checked batches via act(),
                          write next_state once predictions hold.
PROMPT_VARIANT=model   -> model-first: build and validate next_state early, then path().
"""
from __future__ import annotations

import numpy as np

from .config import (
    ADAPTIVE_REASONING,
    TIERED_REASONING,
    EVIDENCE_NOTES,
    EPHEMERAL_REASONING,
    NOTE_CONFIDENCE,
    TESTING_DISCIPLINE,
    BOARD_SHAPE,
    OBJECT_VIEW_TOOL,
    LAYERED_TRACKER,
    LEAN_NOACTION_TURNS,
    PRIORITY_SCHEDULER,
    PROMPT_ADDITIONS,
    PROMPT_V29,
    WIN_FRAME_IMAGE,
    MAX_ACT_ACTIONS,
    MAX_ACTION_DEPTH,
    MAX_ACTIONS_PER_MESSAGE,
    MAX_DISTINCT_STATES,
    MAX_OUTPUT_TOKENS,
    PROMPT_VARIANT,
    RESET_ON_GAME_OVER,
)


def _actions_text(action_labels: list[str]) -> str:
    desc = []
    for label in action_labels:
        if label == "CLICK":
            desc.append("CLICK row col value (0-based coords, value = current colour at that cell)")
        elif label == "SPACE" and PROMPT_ADDITIONS:   # v2.8
            desc.append("SPACE (a game-specific action, e.g. interact, select, rotate, attach/detach or execute: test what it does "
                        "rather than assuming)")
        elif label == "UNDO" and PROMPT_ADDITIONS:    # v2.8
            desc.append("UNDO (usually reverses your previous action: check what it restores. It counts as an action like any "
                        "other, and it cannot bring back an attempt that ended in a game over)")
        elif label == "RESET":
            desc.append("RESET (restarts the current level from its opening board with a full action budget; completed levels "
                        "are kept. Actions already spent on this level still count toward its score, and the reset counts as "
                        "one action. Send it on its own, not in a batch, and only after at least one action since the level "
                        "start or the last reset)")
        else:
            desc.append(label)
    text = ", ".join(desc)
    if any(lbl in action_labels for lbl in ("UP", "DOWN", "LEFT", "RIGHT")):
        text += (". UP/DOWN/LEFT/RIGHT are only the names of these actions: what they move or change (a player, a cursor, "
                 "several objects, or something else entirely) depends on the game and must be observed")
    return text


TRACKER_TEXT = """- Object identity comes from a LAYERED TRACKER: it reads every board as layered objects (an object may continue behind objects in front of it, and may have several colours when parts always move together) and explains each new board as changes to its previous scene, so an object keeps its #id across actions, and across a level change when it stays the same. The "Layered tracker" lines after each action are that explanation: moved (objects moving by the same offset are listed together; "most of the board shifted" when nearly everything moved), rotated or mirrored (only when the shape is exactly the old one turned; otherwise "changed shape"), recoloured, appeared, gone, no longer visible because covered by another object, visible again, split, merged (objects that moved identically twice are merged into one), grew or lost cells, drawn in front of another object. Two ambiguous cases are flagged as such: two same-shape objects swapping colours (a jump or two recolours) and two identical objects that may have exchanged positions. Boxes are [row_start:row_end, col_start:col_end] of an object's observed cells. The lines are ordered by importance and capped; changes() in run has all of them. It is a heuristic and can be wrong (two similar objects swapped, a change read as a move, unrelated things merged into one object, a covered object kept or dropped too late): treat its lines as observations to check, and verify with run what matters.
"""

OBJECTS_DOC_RAW = """  objects(board=None, include_background=True) -> list of dicts, one per object of `board` (default current_board):
      id, color, color_name, size (cell count), cells (list of (row, col) tuples), rect [r0,r1,c0,c1] (slice bounds),
      crop (CSV of the rect, only for small objects), outline (clockwise corner cells), hash (color+shape, position-free),
      shape_hash (shape only; rotation- and mirror-free), rotation, symmetry, children (ids enclosed by it),
      adjacent (ids sharing an edge), is_background (True for the one object with the most cells, a label that is only
      based on size), is_remaining_actions_indicator"""
OBJECT_DEF_RAW = """- The harness reports geometric components, not interpreted game objects. By default, a component is a 4-connected group of equal-colour cells: touching same-colour things merge, and a multi-colour thing splits. A game object may comprise several components or part of one. lib.extract_components() measures these regions; it does not infer roles, mechanics or persistent identity. Hashes describe appearance. The largest component is labelled is_background only because of its size and is omitted from change reports; it is still returned by extraction. set_object_view(exclude=[...]) makes the reports treat the listed colours as gaps, so touching cells of the other colours form one component (possibly several game objects)."""
OBJECT_DEF_SCENE = """- An "object" is what the layered tracker tracks: a group of cells that stays together across boards; usually one colour and 4-connected, but it keeps its cells where another object passes in front of them (shown as covered / guessed cells) and can hold several colours once parts have moved together. These are tracking hypotheses, not established game roles. The raw 4-connected component reading is separately available through lib.extract_components(board); its IDs are per extraction, not the tracker's IDs. lib.frame_diff() also matches raw components, using the configured colour exclusions."""
OBJECTS_DOC_SCENE = """  objects(board=None, include_background=True, raw=False) -> list of dicts, one per object of the layered tracker's scene
      for `board` (default current_board; an index into states is accepted). Ids are the tracker's #ids and persist across
      actions. Fields: id, color (main colour), colors ([[colour, cells], ...]), color_name, size (observed cells), visible
      (cells showing now), cells (observed (row, col) tuples), parts ([{color, cells, guessed}] per colour), guessed_cells
      (cells the tracker assumes behind other objects: a guess, not an observation), rect [r0,r1,c0,c1] (slice bounds of the
      observed cells), full_rect / full_size (including guessed cells), layer (0 = front), covered_by, covers, adjacent
      (ids shown next to it), hash (shape+colours, position-free), shape_hash (shape only, rotation- and mirror-free),
      same_kind_as (id of an earlier object with the same shape and colours, or None), is_background,
      is_remaining_actions_indicator, crop (CSV of the rect, small objects only). raw=True gives the plain 4-connected
      single-colour segmentation instead (per-board ids; the fields of the older objects()).
  obj(id, board=None) -> that object's dict, or None
  changes(n=1) -> the tracker's explanation of the last n actions of this attempt: [{action, lines, edits}] (all lines, uncapped)"""

COMMON_HEAD = """You are playing a 2D grid-based puzzle game on a {rows}x{cols} board with 16 colours (values 0-15), one level at a time.
Available actions: {actions_text}.
Solutions typically need roughly 5-50 actions per level; the number of actions allowed before game over is unknown and must be deduced from the board (usually a remaining-actions indicator: a bar or counter that shrinks with every action). {game_over_sentence}
The goal and win condition are unknown and must be deduced. The game is a genuine logic puzzle: everything on the board is there for a reason and the solution is logical, not arbitrary. A continuously depleting bar is almost certainly the remaining-actions indicator, never the objective.

COORDINATES (everywhere, no exceptions): 0-based [row, col] matching NumPy board[row, col]; row 0 is the top row, col 0 the leftmost column. Regions use half-open slice notation [row_start:row_end, col_start:col_end].
Cell values go up to 15 (two digits), so never print or join values without separators (print the array, or use ','.join(map(str, row))).

WHAT YOU OBSERVE
- At the start of every level: a large image of the board, colour counts and an object inventory (what kinds of objects are present, and what is new or gone compared with the previous level). After a reset (same level, same initial board): the image and the colour counts. The board is never repeated as text: read exact cells with run, e.g. print(current_board[10:20, 5:15]).
- After each executed action: the number of changed cells (total and outside the remaining-actions indicator), the changed rectangles (with their cell values when small, as rectangles with counts when large) and OBJECT CHANGES: which objects moved (and by how much), rotated, recolored, reshaped, appeared or disappeared{changes_source}.
- Animations: when an action plays an animation, a line reports how many cells changed during it and changed back before it ended. Those changes are not visible in the before/after boards; lib.anim_diff() lists the animation step by step. If the board ended unchanged but the animation did not, the first animation steps are shown automatically.
- Images: a large image of the board at each level start, a small image of the current board every turn, and a change image (changed cells in their new colour, everything else dimmed) when the board changed since your previous turn. After a game over you also get the board as it was at the moment of the game over.
{object_def}
- Object changes are matched automatically between the boards before and after an action and can be wrong (e.g. two similar objects swapped, a merge or split read as a move): verify surprising claims with run before relying on them.
{tracker_text}
YOUR CODE FILE (persists for the whole game, including level changes and resets)
A Python library you maintain with the py (full rewrite) and patch tools. Put anything reusable in it: helpers, analysis functions, planners, a transition model. Names available to it and to every run call (no import needed):
  np               NumPy
  lib              generic grid tools (listed below)
  states           np.ndarray (N,{rows},{cols}) uint8: every observed board of the current attempt (since the level start or the last reset), oldest first
  current_board    states[-1]
  actions_history  list of the action strings of the current attempt (states[i+1] is the board after actions_history[i])
  anim_frames      all intermediate frames the engine played during the last action (list of boards; empty if it did not animate)
  history          every transition on the current level across ALL attempts (game overs and resets included): list of dicts
                   (attempt, action, outcome, changed_cells, changed_outside_indicator, before, after); cleared when a level is completed
  past_levels      one dict per completed level: level, start_board, final_board (the board just before the winning action),
                   win_frame (the last frame of its animation, before the level switched), winning_actions (the winning attempt),
                   actions_total (all attempts), transitions (dicts like history). A record of what happened, not a recipe:
                   the next level has its own board, so compare configurations and rules, not action lists.
  CODE             the current text of your code file (str)
  store            a dict that persists across run calls (store["key"] = value; numpy arrays, lists, dicts are fine).
                   Use it for data you measured or computed and will need again (positions, tables, plans) instead of
                   retyping printed numbers into the next snippet.
  indicator_rects  the list of remaining-actions indicator rectangles you marked (usually 0 or 1)
  act(actions, repeat=False)  execute real actions NOW from inside run code (see below)

{lib_doc}

TOOLS (function calling; emit tool calls, never describe actions in prose)
- py(code): replace the whole code file.   - patch(patch): unified-diff style patch (@@ hunks, space/-/+ prefixes, each hunk must match exactly once).
- run(code): execute code in the sandbox namespace (print(...) output is returned, 2D arrays print as CSV). Variables do not persist between calls (use store for data), but every top-level def in a run snippet is kept automatically in your code file (added to its "auto-kept functions" section, or replacing the existing top-level function with the same name, including next_state) and the feedback names them. Only top-level functions are kept, not variables or constants they rely on: a function that reads a name defined only in the snippet is NOT kept (the feedback says which name is missing; pass it as a parameter, read it from store, or add it to the code file with py/patch). Snippet assignments never overwrite names in your code file (the snippet runs in its own namespace that merely sees the code file's names); store = {{...}} replaces the store. About 30 s of compute time per run call (time spent inside act() does not count; output printed before the limit is returned), about 800 MB of memory. Modules available in run and in the code file: numpy (as np), collections, itertools, functools, math, heapq, bisect, re, copy, json, operator, statistics, fractions, string, random, time; all are already imported under their own names, so no import is needed (nothing else can be imported). Not available: open, eval/exec, getattr/setattr, globals/locals, dunder names or attributes (__x__), and class statements (use dicts, tuples or collections.namedtuple). Errors in run report the line of your code that raised. When the step reports of a run call get very long, they are shown one line per step with identical consecutive steps merged, labelled [i] = the step's index in history (lib.frame_diff(history[i]['before'], history[i]['after']) shows its board change in full); each line notes an animation if one played, and steps a line cannot describe (level end, GAME OVER, unexplained changes, a board that ended unchanged while an animation played) keep their full report. A py/patch whose top-level code raises when the file is loaded is not applied (your previous file stays active); top-level code runs before states/current_board exist, so keep it to definitions and constants.
  Inside run you may call act(["UP", "RIGHT", "CLICK 10 20 3"]) to execute actions immediately; it returns {{executed, steps, stopped_early, stop_reason, terminal, outcome}} and refreshes states/current_board/actions_history/history, so your code can check the result after every step and decide what to do next. outcome (of the result and of each step) is None for an ordinary step and is set only when a step ended the attempt or level ("level_complete", "win", "game_over", "reset"); test terminal / outcome is not None, never the truthiness of a string. At most {max_act} actions per turn, shared by all run calls of one response.
  Guards: (1) a batch stops as soon as an action changes nothing outside the remaining-actions indicator, and also as soon as a colour never seen before in this game appears: the remaining actions of that batch are skipped and the run call ends there (STOPPED), so you can look at the new state first (a one-action act() call that changes nothing does not end the run call: check r["stopped_early"]); (2) an action previously observed to produce no visible change in this EXACT board state (indicator ignored) and no animation, or to end in a game over from this exact board (indicator included), is refused unless you pass repeat=True. The first executed action after a start/reset never creates a no-op record; no-op records clear on reset. An unchanged final board with an animation is a reaction to inspect, not proof the action does nothing. Hidden state may differ even when pixels match; (3) after a level completion or game over no further action runs in that call: re-read the new board next turn. act(["RESET"]) must be called alone; acceptance ends the snippet immediately, skips its remaining code (including except/finally) and later run calls in this response, and retains work saved before RESET. Inspect the restarted board next turn. (4) a rejected or invalid action (e.g. a CLICK whose value does not match the cell, an unknown label) stops the whole run call: actions before it in that call were executed, nothing after it was, and the feedback says which. Refusals and stops end the run call with a STOPPED message; check what was executed before relying on the rest of your plan.
- actions(lines, repeat=false): execute up to {max_actions} explicit actions with no checks in between (for quick probes), e.g. ["UP", "CLICK 30 45 9"]. RESET must be the only line. The same guards as act() apply (known no-op / known fatal actions are refused unless repeat is true; the batch stops at a no-op or a new colour). CLICK needs row col value where value is the current colour at that cell. A CLICK whose value does not match is rejected by the harness before it reaches the game (not executed, no budget used); that says nothing about whether the cell reacts to clicks.
- path(max_steps): BFS over your next_state model from current_board to a level_complete outcome and execute up to max_steps actions of the found path. After EVERY executed step the observed board is compared with your model's prediction (ignoring the indicator rectangle); on any mismatch, or when a step predicted to complete the level does not, the path stops and the difference is reported. Unavailable while the model check finds differences or errors on this level (returning None for what you don't know is fine). CLICK candidates: lib.click_candidates(board) (one cell per single-colour region), or only the cells returned by CLICK_TARGETS(board) -> list of (row, col) if you define it in the code file. UNDO (where a game has it) is not used by path and not checked by the model check. Actions for which next_state raises or returns None are skipped (the feedback says so).
- remaining_actions_indicator(rects): mark the rectangle(s) that contain ONLY the remaining-actions indicator, e.g. ["[0:64, 63:64]"]. Usually it is one rectangle; occasionally the indicator is split into several parts (e.g. a bar along the top and one along the bottom) - then list them all. The list you send replaces the previous selection entirely; [] clears it. Changes inside the marked area are reported separately and ignored by the guards and by the model check. Mark it as soon as you have identified it.
- set_object_view(exclude): how the harness describes the board in its own reports (change lines, inventory, model check, lib.frame_diff's default): the listed colours count as gaps and touching cells of all other colours are reported as one component. [] goes back to single-colour components. It never changes the game.
- note(key, text): write or update one entry of your pinned notes - the only part of your reasoning guaranteed to survive context trimming (older turns are dropped when the context fills up; the notes, the harness event log, the current level's start message and your code file are re-shown instead). Keep there: confirmed rules with their evidence, rules you have ruled out, surprising observations, the win condition of each completed level, open questions. Same key = replace, empty text = delete. Update the notes whenever you learn or refute something, not only when asked.
Order of execution within one response: py/patch -> set_object_view -> model check -> run (incl. act) -> note and remaining_actions_indicator -> actions or path -> model check of the new transitions. If run already executed actions this turn, a following actions/path call is skipped.

TRANSITION MODEL (your model of the rules: checked by the harness, and the way to use path)
Define in the code file:
  def next_state(board, action, row=-1, col=-1) -> (next_board_or_None, outcome)   outcome: "continue" | "level_complete" | "game_over"
  or return None for "don't know" (a case your model does not cover yet).
board is a uint8 ({rows},{cols}) board; return the predicted next board as a new array. next_state and every function it calls see the same names as run code (lib, history, past_levels, store as a read-only copy); act() is not available and print shows nothing there. Inside next_state always pass the board to lib (lib.find(board, ...), lib.object_at(board, ...)); build the result with lib.move_on_board / lib.draw / lib.erase or plain numpy. Prefer lib.find / lib.object_at to lib.extract_components there: path calls next_state thousands of times. You may alternatively define board_to_symbolic(board) and symbolic_to_board(state) to model in your own representation.
MODEL CHECK: when next_state changes and after every turn with new transitions, the harness runs next_state on every transition of this level (all attempts) and reports facts: how many match, differ, don't know (None) or raise an error, and for the first difference what next_state predicted and what actually happened (in object terms, the same lines as the change report) with the differing cells, plus an image of it (before | predicted | actual). The indicator rectangle is ignored; for observed game_over / level_complete steps only the outcome is compared. Write next_state early and partially: cover the actions you understand, return None for the rest, and let the check show you exactly where your understanding is wrong. It only covers what you have observed. When a level is completed, the feedback says what next_state predicted for the winning step; on a new level the check shows how your existing next_state does there (levels may or may not share the rules). BFS limits: depth {bfs_depth}, {bfs_states:,} states.

GAME OVER AND LEVELS
A game over means the attempt failed; the level resets to its initial state, completed levels are kept, and your code file, conversation and knowledge are kept. The harness does not know why a game over happened: work out the cause yourself from the observations (the board at the game over is attached). You can also restart the level yourself with RESET when the current attempt cannot succeed any more (e.g. something needed is stuck or used up). The game is deterministic: the same sequence fails the same way, so change the plan before the fatal point. Levels usually build on the mechanics of earlier ones; carry them forward as hypotheses, look for new kinds of objects (the level-start inventory lists them), and re-check the goal.

A nonterminal move or shape/size change is an observation to investigate; it alone proves neither damage nor progress. Before resetting because something changed, try a small informative probe of the changed state unless there is evidence it cannot succeed. RESET undoes changes and can restore hidden state. Before a second manual RESET on the same level, state in that response what will differ in the next attempt and what observation supports restarting; no extra turn is required.
When a level resists your hypotheses, list the assumptions your model or plan relies on that no observation has confirmed (a validated next_state only reproduces transitions you have actually observed; anything it says about situations you never saw, such as collisions, blocking or the win condition, is a guess) and test those first.

RESPONSE FORMAT
Start every response with a short 2-4 bullet plain-text summary in the visible reply itself (not only in your thinking): current hypothesis, what you are testing or changing this turn, main uncertainty or risk. Then the tool calls.
Each response is capped at {max_out:,} output tokens, thinking included. A response that reaches the cap is cut off and a tool call it was writing is not executed, so keep each turn's reasoning focused and move long analysis, case enumeration and search into code (run).
"""

OBJECT_REFERENCES = """OBJECT REFERENCES (available immediately; no setup code)
The harness gives measured components IDs such as C11. CLICK C11 (or CLICK @name) resolves its current
continuation and clicks an occupied cell. CLICK @name color 3 selects that colour within a composite.
Coordinate clicks still work. IDs are unique in this game; old IDs follow unique movement/recolour links,
but ambiguous, missing or previous-level/attempt references refuse the click. A match is geometric evidence,
not proof of a game role. Raw extraction IDs from lib.extract_components are local indices: use its ref field
on the current unfiltered board, or lib.components(), for clickable IDs.
In run, lib.components(colors={3:8}, size=8) filters current components by exact colour counts/size.
The list is compact; detail=True includes cells/values. lib.resolve('C11') returns the full current selection.
lib.remember('player', ['C11','C17'], reason='parts moved together')
saves a name for one selection or a composite template; it returns the selection immediately. lib.references() lists
saved names, grouping reasons and current IDs. You can then
act(['CLICK @player']) in that same run. lib.forget('player') removes a name. Saved names survive context trims;
on a new level or reset inspect the fresh IDs and explicitly rebind. No complete catalogue is required.
lib.relations('C11') returns measured history/current spatial relationships and agent assertions, with provenance.
lib.relate('C11','same_object','C17', evidence='...', follow=True) explicitly connects an unresolved old ID to a
current one; only identity links with evidence can redirect clicks. Without follow=True it is a hypothesis.
Other relations: recoloured, same_shape, part_of, touching, inside, left_of, possibly_same. Spatial measurements
are for the current board only; inside_bounds means bounding-box containment, not a discovered game rule.
Use IDs/names in short hypotheses and checked probes. Example: act(['RIGHT'], expect=[{'entity':'player',
'test':'moved','delta':[0,3]}]). Optional expect is also accepted by actions; fenced actions use #expect JSON.
Checks run after each action and stop on contradicted/unknown. Tests: resolved, unresolved (matching only,
not physical disappearance), unchanged, moved (exact cell translation), color (all selected cells).
Live reference helpers are for run only. In next_state use geometry helpers with the supplied hypothetical board.
"""

STRATEGY_OBJECTS = """STRATEGY (objects-first)
1. Use the component IDs and lib.extract_components() as measurements. Save useful selections or groups with lib.remember as you explore; allow unknown roles and revise groupings when observations disagree. Mark the remaining-actions indicator early.
2. Learn the mechanics with small, informative probes executed from run via act(): one or two actions, then lib.frame_diff() to inspect measured changes and tentative component matches. Record every confirmed rule in the code file.
3. Write next_state early, for the mechanics you have seen (None for the rest), and refine it as you learn: the model check after every turn shows where it is wrong. Once it matches, path finds the shortest solution.
4. Otherwise execute plans as checked batches: a loop in run that calls act() one or a few actions at a time and verifies the expected object change before continuing (stop and re-think on surprise). Avoid long unverified action lists.
5. Spend actions deliberately: every executed action costs budget. Avoid blind repetition. Inspect animations and consider a controlled repeat when a first-press gate or other hidden state could explain an unchanged final board.
"""

TESTING_TEXT = """TESTING BEFORE CONCLUDING
- A conclusion that something cannot happen (blocked, inert, no effect, impossible, useless) holds only for the situations you tested it in. Before building on it, test it once in a clearly different situation (another position, orientation, object or state), or keep it marked as untested.
- A result you predicted but did not observe is a guess. If it can be checked with one or two actions, check it instead of analysing further: an action usually costs far less than a long analysis.
- If your plan comes out impossible or far too costly, one of your beliefs is wrong. List the beliefs you have not tested directly and test the cheapest one first.
"""

STRATEGY_MODEL = """STRATEGY (model-first)
1. Take a few exploratory actions (actions or act()) to see what each action does, reading the object changes carefully. Mark the remaining-actions indicator early.
2. As soon as the mechanics are visible, write next_state (board in, board out, built with lib) plus a win-condition hypothesis, and iterate until the model check reproduces every observed transition. lib.find, lib.move_on_board and lib.frame_diff are there to make writing and debugging the model fast.
3. Then use path to execute the shortest solution; the per-step prediction check stops it when the model is wrong, and the reported difference tells you what to fix.
4. Use act() inside run for probes that need code to decide the next action (e.g. walk until blocked), with your own checks after each step.
5. Spend actions deliberately: every executed action costs budget. Avoid blind repetition. Inspect animations and consider a controlled repeat when a first-press gate or other hidden state could explain an unchanged final board.
"""

BUDGET = """COMPUTE BUDGET
You have a limited wall-clock budget per game; during the last quarter of it, a [Budget] line in the feedback shows the minutes remaining.
Every action you take on a level counts against that level's score once the level is completed (fewer actions = higher score), including the actions of failed attempts and the RESET itself. So don't waste actions: use them for informative exploration, and work out mechanics and solutions by thinking and by simulating game models and candidate solutions in code (run) before executing them. If the current attempt can no longer succeed, use RESET right away rather than spending more actions in it.
Keep working on the game until it is stopped for you; never spend turns only writing summaries or wrapping up. Reuse learned mechanics and code across levels, keep reasoning focused on new uncertainty, and for known steps give a one-line justification and act.
"""


# v2.9 (PROMPT_V29): keep observations as specific as the evidence; re-check the mechanics before the goal
SCOPE_TEXT_V29 = (
    "Keep observations as specific as the evidence. When an action has no effect, note the situation it was tried in "
    "(position, selection, state of other elements); it may work elsewhere. Don't turn one observation into a general rule "
    "until it has held in different situations. When a plan that should work under your current model fails, or the goal "
    "starts to look impossible, first re-check the least-tested part of your model of the mechanics with a cheap test, before "
    "changing your idea of the goal. A one-action test of something unexplained (clicking it, moving onto it, using an action "
    "near it) is usually cheaper than long reasoning about it.\n"
)


def _lib_doc() -> str:
    from .gridlib import LIB_SUMMARY
    return LIB_SUMMARY   # substituted by str.format, which does not re-read braces inside the value


def build_system_prompt(
    action_labels: list[str],
    *,
    board_shape: tuple[int, int] = BOARD_SHAPE,
    variant: str | None = None,
) -> str:
    """Full system prompt; `variant` defaults to config.PROMPT_VARIANT ("objects" or "model")."""
    from .config import MINIMAL_HARNESS
    if MINIMAL_HARNESS:
        from .minimal_prompt import system_prompt
        return system_prompt(action_labels, board_shape, MAX_ACT_ACTIONS)
    variant = (variant or PROMPT_VARIANT).strip().lower()
    if RESET_ON_GAME_OVER:
        go = ("If all actions are spent the game ends and restarts the level. Previously earned level completions are kept, "
              "so complete each level in as few actions as possible.")
    else:
        go = "If all actions are spent the game is lost with no way to recover, so be careful with spending actions."
    head = COMMON_HEAD.format(
        rows=board_shape[0],
        cols=board_shape[1],
        actions_text=_actions_text(action_labels),
        game_over_sentence=go,
        max_act=MAX_ACT_ACTIONS,
        max_out=MAX_OUTPUT_TOKENS,
        max_actions=MAX_ACTIONS_PER_MESSAGE or 4,
        bfs_depth=MAX_ACTION_DEPTH,
        bfs_states=MAX_DISTINCT_STATES,
        tracker_text=TRACKER_TEXT if LAYERED_TRACKER else "",
        lib_doc=_lib_doc(),
        changes_source=(' (the "Layered tracker" lines described below)' if LAYERED_TRACKER else
                        '. Identical moves are grouped ("12 x color 4 2x2 moved by (0,+1)")'),
        scene_move_note=(", including its parts' cells and guessed cells; objects are painted back to front by layer, so a "
                         "moved object covers what is behind it and uncovers what it left" if LAYERED_TRACKER else ""),
        object_def=OBJECT_DEF_SCENE if LAYERED_TRACKER else OBJECT_DEF_RAW,
    )
    if LEAN_NOACTION_TURNS:   # v2.8: no small image on turns after which no action was executed
        head = head.replace("a small image of the current board every turn, ",
                            "a small image of the current board every turn after which an action was executed (otherwise a "
                            "one-line note says that no action was executed and the board is unchanged), ")
    if WIN_FRAME_IMAGE:   # v2.9
        head = head.replace("After a game over you also get the board as it was at the moment of the game over.",
                            "After a game over you also get the board as it was at the moment of the game over, and after a "
                            "completed level an image of the winning moment (the last frame of the winning action).")
    if PROMPT_V29:   # v2.9
        head = head.replace("Solutions typically need roughly 5-50 actions per level;",
                            "Efficient solutions usually take roughly 10-60 actions per level, occasionally 100 or more;")
        head = head.replace("is a guess) and test those first.\n",
                            "is a guess) and test those first.\n" + SCOPE_TEXT_V29)
    if EVIDENCE_NOTES:   # v2.12b: notes with status and evidence
        head = head.replace("- note(key, text): write or update one entry of your pinned notes - the only part of your reasoning guaranteed to survive context trimming (older turns are dropped when the context fills up; the notes, the harness event log, the current level's start message and your code file are re-shown instead). Keep there: confirmed rules with their evidence, rules you have ruled out, surprising observations, the win condition of each completed level, open questions. Same key = replace, empty text = delete. Update the notes whenever you learn or refute something, not only when asked.\n", '- note(key, text, status, certainty, evidence): write or update one entry of your pinned notes - the only part of your reasoning guaranteed to survive context trimming (older turns are dropped when the context fills up; the notes, the harness event log, the current level\'s start message and your code file are re-shown instead). One rule, observation or idea per entry. status: "seen" (observed directly), "inferred" (follows from what you observed) or "assumed" (not tested yet). certainty: how sure you are, in your own words (e.g. "certain: seen 4 times in different positions", "likely", "a guess"). evidence: what you tried and in which situation, citing the steps as history[i] (every step report shows its index); next to each cited history[i] the harness shows the action actually recorded there. A rule that something cannot happen (blocked, inert, no effect, impossible) holds only for the situations it was tested in. Same key = replace, empty text = delete. Update the notes whenever you learn or refute something, not only when asked. After many turns on a level without completing it, the harness lists your entries not marked seen and the actions and regions not yet tried on this level; testing the cheapest of them usually beats more analysis.\n')
    if not OBJECT_VIEW_TOOL:   # v2.11: the set_object_view tool is off
        head = head.replace(" set_object_view(exclude=[...]) makes the reports treat the listed colours as gaps, so touching "
                            "cells of the other colours form one component (possibly several game objects).", "")
        head = head.replace("- set_object_view(exclude): how the harness describes the board in its own reports (change lines, "
                            "inventory, model check, lib.frame_diff's default): the listed colours count as gaps and touching "
                            "cells of all other colours are reported as one component. [] goes back to single-colour components. It "
                            "never changes the game.\n", "")
        head = head.replace("py/patch -> set_object_view -> model check", "py/patch -> model check")
    strategy = STRATEGY_MODEL if variant == "model" else STRATEGY_OBJECTS
    if not LAYERED_TRACKER:
        head = head.replace("an object inventory (what kinds of objects", "a component inventory (what kinds of components")
        head = head.replace("OBJECT CHANGES: which objects", "COMPONENT CHANGES: which components")
        head = head.replace("Object changes are matched automatically", "Components are matched automatically")
    strategy = OBJECT_REFERENCES + "\n" + strategy
    budget = BUDGET
    if PRIORITY_SCHEDULER:   # v2.7: no per-game budget and no [Budget] line under the priority scheduler -> no limit mentioned
        budget = budget.replace("COMPUTE BUDGET\nYou have a limited wall-clock budget per game; during the last quarter of it, "
                                "a [Budget] line in the feedback shows the minutes remaining.\n", "ACTION EFFICIENCY\n")
    if TESTING_DISCIPLINE:   # v2.12d
        strategy = strategy + "\n" + TESTING_TEXT
    if ADAPTIVE_REASONING:
        from .adaptive_reasoning import PROMPT as reasoning_prompt, TIERED_PROMPT
        head += "\n" + (TIERED_PROMPT if TIERED_REASONING else reasoning_prompt)
    if EPHEMERAL_REASONING or NOTE_CONFIDENCE:
        from .note_memory import MEMORY_PROMPT, CONFIDENCE_PROMPT
        if EPHEMERAL_REASONING:
            head += "\n\n" + MEMORY_PROMPT
        if NOTE_CONFIDENCE:
            head = head.replace("note(key, text)", "note(key, text, confidence)")
            head += "\n\n" + CONFIDENCE_PROMPT
    return head + "\n" + strategy + "\n" + budget


def build_level_start_message(
    *,
    current_level: int,
    board: np.ndarray,
    action_labels: list[str],
    total_actions_taken: int,
    previous_level: int | None = None,
) -> str:
    from .config import MINIMAL_HARNESS
    if MINIMAL_HARNESS:
        return (f"Level {current_level}; board shape {tuple(board.shape)}; "
                f"available actions: {', '.join(action_labels)}, RESET. "
                f"Total actions: {total_actions_taken}. Inspect the current board image and use run().")
    lines: list[str] = []
    if previous_level is None:
        lines.append(f"You are now on level {current_level}.")
    else:
        lines.append(f"Previous level {previous_level} completed successfully.")
        lines.append(f"You are now on level {current_level}.")
        lines.append("The number of actions is restored for this new level.")
    lines.append(
        "The game being played is fundamentally the same as before, but new levels may include additional mechanics or logical complexity."
    )
    lines.append("")
    board_h, board_w = board.shape
    lines.append(
        f"Initial board for level {current_level} ({board_h}×{board_w}, 0-based [row, col]): see the attached image; "
        f"the colour counts and {'object' if LAYERED_TRACKER else 'component'} inventory follow. Read exact cells with run, e.g. print(current_board[10:20, 5:15])."
    )
    lines.append("")
    lines.append(f"Available actions: {', '.join(action_labels)}")
    lines.append("Actions taken in current level: 0")
    lines.append(f"Total actions taken across run: {total_actions_taken}")
    lines.append("")
    lines.append(
        "Treat this as a fresh level initialization: look at the new board (image, inventory, run) before deciding next actions."
    )
    lines.append("Use this level's fresh component IDs; rebind any saved names before reusing them.")
    lines.append("")
    if previous_level is not None:
        lines.append(
            "Levels usually build on the mechanics of earlier levels, especially the most recent one. "
            "Carry forward what you established as your starting hypothesis and re-check anything this "
            "board contradicts. New levels often add elements or mechanics, and unfamiliar board elements "
            "are usually important: prefer small, informative tests when their behaviour is unclear. "
            "Re-assess the goal: it may be unchanged but require the new mechanics, or it may differ. "
            "Your code file is kept: adapt its level-specific parts (positions, boundaries, layout) and "
            "re-validate it rather than rewriting from scratch, unless the mechanics themselves changed."
        )
    else:
        lines.append(
            "This is the same level restarted from its initial state. Your code file and everything you "
            "observed are kept. A restart alone does not establish that the previous board was unsolvable. "
            "Retain tested mechanics, distinguish observations from hypotheses, and check what the reset undid "
            "before choosing the next probe."
        )
    return "\n".join(lines)

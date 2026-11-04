"""ARC-AGI-3 agent: iterative LLM-driven game solver with agentic commands."""
from __future__ import annotations

import bisect
import concurrent.futures
from datetime import datetime
import json
import logging
import os
from time import perf_counter
import time as _time
import re
import threading
from uuid import uuid4
from typing import Any

import numpy as np

import arc_agi
from arcengine import GameState

_HAS_GAME_STATE_WIN = hasattr(GameState, "WIN")

from .board import (
    animation_metadata_from_frames,
    board_key,
    board_to_csv,
    norm_rects,
    rects_mask,
    rects_text,
    to_uint8_board,
)
from .config import (
    MINIMAL_HARNESS,
    ADAPTIVE_REASONING,
    TIERED_REASONING,
    EPHEMERAL_REASONING,
    NOTE_CONFIDENCE,
    ACT_GUARD_MODE,
    BOARD_CELLS,
    BOARD_SHAPE,
    BOARD_COLOR_MAP_HEX,
    BOARD_COLOR_NAME_BY_ID,
    CHEAT_MODE,
    COMPACTION_KEEP_TURNS,
    COMPACTION_TOKEN_LIMITS,
    CONTEXT_PRESEND_TRIM_TOKENS,
    CONTEXT_TRIM,
    CONTEXT_TRIM_TARGET_TOKENS,
    CONTEXT_TRIM_TRIGGER_TOKENS,
    DEADLINE_SECONDS,
    ENABLE_REPLAYS,
    FEEDBACK_CHAR_CAP,
    FEEDBACK_CHAR_CAP_LEVEL_TRANSITION,
    ENVIRONMENTS_DIR,
    GAME_ID,
    IS_KAGGLE,
    KAGGLE_GATEWAY_URL,
    KAGGLE_QUICK_TEST,
    LAYERED_TRACKER,
    LEVEL_WEIGHTED_ALLOCATION,
    LLM_RETRY_MAX_SECONDS,
    MAX_ACTIONS_PER_MESSAGE,
    MAX_OUTPUT_TOKENS,
    MAX_CONCURRENT_REQUESTS,
    MAX_DISTINCT_STATES,
    MAX_ACTION_DEPTH as _MAX_ACTION_DEPTH_CFG,
    BFS_TIME_LIMIT_SECONDS,
    WAIT_GRACE_SECONDS,
    MAX_RUN_OUTPUT_CHARS,
    MAX_TURNS,
    MEMORY_DEBUG,
    MODEL_NAME,
    MODEL_PROVIDER_OVERRIDE,
    HARNESS_VERSION,
    MODEL_THINKING_EFFORT,
    NO_CHANGE_WARNING_TURNS,
    NOTES_CHAR_CAP,
    EVIDENCE_NOTES,
    STUCK_CHECK_TURNS,
    STUCK_CHECK_EVERY,
    ACT_NUDGE,
    ACT_NUDGE_TOKENS,
    ENV_RESET_TRIES,
    PROMPT_V29,
    WIN_FRAME_IMAGE,
    OPERATION_MODE,
    PROMPT_VARIANT,
    PRIORITY_DEFAULT_LEVELS,
    LEAN_NOACTION_TURNS,
    PRIORITY_SCHEDULER,
    PRIORITY_STRETCH_CAP_SECONDS,
    PRIORITY_TAU_TOKENS,
    REPLAYS_DIR,
    REPLAY_SHARED_JS_NAME,
    RESET_LABEL,
    RESET_ON_GAME_OVER,
    ROOT_DIR,
    SEED,
    SERVER_WAIT_SECONDS,
    TARGET_LEVEL,
    USE_TOOL_CALLING,
    model_provider,
)
from .allocation import SMART_ALLOCATION, TimeAllocator, parse_estimate_from_commands
from .decision_policy import DecisionPolicy, RecoverySettings, PROPOSAL_TOOL, identity as decision_identity
from .adaptive_reasoning import ReasoningPolicy, TieredReasoningPolicy, execution_evidence, reasoning_configuration
from .note_memory import ConfidenceNotes, without_reasoning
from .llm import (
    BoardImageAttachment,
    Command,
    _build_board_image_attachments,
    _chat_user_content,
    _command_tool_specs,
    create_openai_client,
    parse_response_commands,
    request_llm_response,
    server_ready,
    wait_for_server,
)
from .log import (
    Color,
    ReplayRun,
    _coerce_float,
    _coerce_int,
    colorize,
    log_tag,
    board_to_hex_rows,
    print_board,
    replay_game_folder,
)
from .prompts import build_level_start_message, build_system_prompt
from .sandbox import VirtualCodeFile
from . import harness_ext as _ext
from . import scheduler as _scheduler
from .validation import (
    CandidateAction,
    TransitionRecord,
    _to_plain_symbolic,
    action_specs_from_env,
    candidate_to_env_call,
    get_current_level,
    observation_board,
    search_specs,
)

_print_lock = threading.Lock()
_parallel_completed: list[dict[str, str]] = []

# ============================================================================ v2.5-log: replay helpers

_viewer_lock = threading.Lock()
_replay_tls = threading.local()      # the replay of the game running in this thread (flushed if the game raises)


def _install_replay_viewer() -> None:
    """Copy the replay viewer shipped with the harness (agent/replay_viewer.js) next to the replays, so a downloaded
    replays folder opens as-is. An existing different/empty file (e.g. the notebook's placeholder) is replaced."""
    if not ENABLE_REPLAYS:
        return
    from pathlib import Path as _P
    src = _P(__file__).resolve().parent / REPLAY_SHARED_JS_NAME
    if not src.is_file():
        return
    with _viewer_lock:
        try:
            dst = REPLAYS_DIR / REPLAY_SHARED_JS_NAME
            data = src.read_bytes()
            if dst.is_file() and dst.read_bytes() == data:
                return
            REPLAYS_DIR.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + ".tmp")
            tmp.write_bytes(data)
            tmp.replace(dst)
        except Exception as exc:
            print(f"[replay] could not install the replay viewer: {exc!r}")


def _env_baselines(env: Any, game_id: str) -> list[int] | None:
    """Per-level baseline action counts (public games only; hidden in the scored rerun)."""
    for holder in (getattr(env, "environment_info", None), getattr(env, "info", None)):
        try:
            b = getattr(holder, "baseline_actions", None) if holder is not None else None
            if b:
                return [int(x) for x in b]
        except Exception:
            pass
    try:   # fallback: the game's metadata.json in the environments folder
        from pathlib import Path as _P
        short = game_id.split("-", 1)[0]
        for meta in _P(str(ENVIRONMENTS_DIR)).glob(f"**/{short}*/**/metadata.json"):
            d = json.loads(meta.read_text())
            if str(d.get("game_id", "")).split("-", 1)[0] == short and d.get("baseline_actions"):
                return [int(x) for x in d["baseline_actions"]]
    except Exception:
        pass
    return None


def _level_scores(baselines: list[int] | None, n_levels: int | None, level_actions: dict[int, int],
                  initial_level: int = 1) -> tuple[list[dict[str, Any]], float | None]:
    """ARC-AGI-3 score: level score min(115, 100 (baseline/actions)^2), weighted by the level number;
    game score = weighted mean, capped at (weights of scored levels / all weights) x 100."""
    n = n_levels or (len(baselines) if baselines else 0) or (max(level_actions) if level_actions else 0)
    rows: list[dict[str, Any]] = []
    tot_w = tot_s = max_w = 0.0
    for lvl in range(1, n + 1):
        base = baselines[lvl - 1] if baselines and lvl - 1 < len(baselines) else None
        acts = level_actions.get(lvl)
        sc = None
        if base is not None:
            sc = min(115.0, 100.0 * (base / acts) ** 2) if acts else 0.0
            tot_s += lvl * sc
            if sc > 0:
                max_w += lvl
        tot_w += lvl
        rows.append({"level": lvl, "baseline": base, "actions": acts, "completed": acts is not None,
                     "score": None if sc is None else round(sc, 2)})
    if not baselines or tot_w <= 0:
        return rows, None
    return rows, round(min(tot_s / tot_w, 100.0 * max_w / tot_w), 3)


def _content_text(content: Any) -> str:
    """The text parts of a chat message content (a string or a list of typed parts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(p.get("text") or "") for p in content if isinstance(p, dict) and p.get("type") in ("text", "input_text"))
    return ""


def _store_summary(store: dict[str, Any]) -> dict[str, Any] | None:
    """Keys of the model's store dict with type, size and a short preview (for the replay)."""
    if not store:
        return None
    out: dict[str, Any] = {}
    for i, (k, v) in enumerate(store.items()):
        if i >= 60:
            out["..."] = {"type": "", "size": f"{len(store) - 60} more keys"}
            break
        try:
            size = (list(v.shape) if hasattr(v, "shape") else len(v) if hasattr(v, "__len__") else None)
        except Exception:
            size = None
        try:
            prev = repr(v)
        except Exception:
            prev = "?"
        out[str(k)] = {"type": type(v).__name__, "size": size, "preview": prev[:240] + ("..." if len(prev) > 240 else "")}
    return out


def _mem_mb() -> float:
    """Return current process RSS in MB (Linux)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0


def _remaining_actions_indicator_rect_text(rects: Any) -> str:
    return rects_text(rects)


def _parse_remaining_actions_indicator_rect(
    text: str, board_shape: tuple[int, int] = BOARD_SHAPE,
) -> tuple[tuple[tuple[int, int, int, int], ...] | None, str | None]:
    """Parse the full list of indicator rectangles (v2.5: zero or more). Returns (rects, error); an empty
    tuple means "cleared". The text may be a JSON list of strings, several "[a:b, c:d]" items, or clear/none."""
    raw = text.strip()
    low = raw.lower()
    if not raw or raw in ("[]", "()") or any(token in low for token in ("clear", "none", "unset")):
        return (), None
    found = re.findall(r"(-?\d+)\s*:\s*(-?\d+)\s*,\s*(-?\d+)\s*:\s*(-?\d+)", raw)
    if not found:
        return None, "Invalid rectangle format. Use rects like \"[row_start:row_end, col_start:col_end]\" (one per rectangle)."
    h, w = board_shape
    rects: list[tuple[int, int, int, int]] = []
    for g in found:
        rs, re_, cs, ce = (int(x) for x in g)
        if not (0 <= rs < re_ <= h and 0 <= cs < ce <= w):
            return None, f"Rectangle out of bounds. Got [{rs}:{re_}, {cs}:{ce}] for board size [{h}x{w}]."
        rects.append((rs, re_, cs, ce))
    return tuple(rects), None


def _color_counts(board: np.ndarray) -> dict[int, int]:
    arr = to_uint8_board(board)
    values, counts = np.unique(arr, return_counts=True)
    return {int(v): int(c) for v, c in zip(values.tolist(), counts.tolist())}


def _format_color_lines(
    *,
    board: np.ndarray,
    colors: list[int] | None = None,
) -> list[str]:
    counts = _color_counts(board)
    selected = sorted(counts.keys()) if colors is None else sorted(colors)
    lines: list[str] = []
    for color in selected:
        color_hex = BOARD_COLOR_MAP_HEX.get(color, "unknown")
        color_name = BOARD_COLOR_NAME_BY_ID.get(color, "Unknown")
        lines.append(f"Color {color},{color_hex},{color_name},{counts.get(color, 0)}")
    return lines


def _full_board_color_inventory_text(board: np.ndarray) -> str:
    if MINIMAL_HARNESS:
        return ""
    return (
        "Colors present on board (value,hex,name,count):\n"
        + "\n".join(_format_color_lines(board=board))
    )


def _format_color_change_lines(before: np.ndarray, after: np.ndarray) -> list[str]:
    b = to_uint8_board(before)
    a = to_uint8_board(after)
    color_ids = sorted(set(np.unique(b).tolist()) | set(np.unique(a).tolist()))
    lines: list[str] = []
    for color in color_ids:
        away = int(np.count_nonzero((b == color) & (a != color)))
        into = int(np.count_nonzero((b != color) & (a == color)))
        if away == 0 and into == 0:
            continue
        color_hex = BOARD_COLOR_MAP_HEX.get(color, "unknown")
        color_name = BOARD_COLOR_NAME_BY_ID.get(color, "Unknown")
        lines.append(f"Color {color},{color_hex},{color_name},-{away},+{into}")
    return lines


def _canonical_board_key(
    board: np.ndarray,
    remaining_actions_indicator_rect: tuple[int, int, int, int] | None,
) -> bytes:
    arr = to_uint8_board(board).copy()
    arr[rects_mask(remaining_actions_indicator_rect, arr.shape)] = np.uint8(255)
    return board_key(arr)


def _rebuild_canonical_seen_action_index(
    boards: list[np.ndarray],
    remaining_actions_indicator_rect: tuple[int, int, int, int] | None,
    offset: int = 0,
) -> dict[bytes, int]:
    """board key -> global action count at which it was seen (boards[i] = after offset + i actions)."""
    out: dict[bytes, int] = {}
    for action_idx, board in enumerate(boards):
        out[_canonical_board_key(board, remaining_actions_indicator_rect)] = offset + action_idx
    return out


def _is_local_provider() -> bool:
    return model_provider(MODEL_NAME, provider_override=MODEL_PROVIDER_OVERRIDE) == "local"


def _is_context_error(exc: BaseException) -> bool:
    """A request the server rejected as invalid (400): context too long, too many images, bad history."""
    text = str(exc).lower()
    if "maximum context length" in text or "context_length_exceeded" in text or "context length" in text:
        return True
    if "too many" in text and "image" in text:
        return True
    if "at most" in text and "image" in text:
        return True
    if ("too long" in text or "exceed" in text) and ("token" in text or "prompt" in text or "context" in text):
        return True
    return False


def _is_bad_request(exc: BaseException) -> bool:
    """v2.10: any other 400 (unrecognised wording): retried once, then handled like a context error."""
    status = getattr(exc, "status_code", None)
    return status == 400 or type(exc).__name__ == "BadRequestError"


def _sync_print(*args: Any, **kwargs: Any) -> None:
    with _print_lock:
        kwargs.setdefault("flush", True)
        print(*args, **kwargs)


def _get_compaction_limit() -> int:
    """Return token limit for current model, or 0 to disable compaction."""
    provider = model_provider(MODEL_NAME, provider_override=MODEL_PROVIDER_OVERRIDE)
    if provider in COMPACTION_TOKEN_LIMITS:
        return COMPACTION_TOKEN_LIMITS[provider]
    name_lower = MODEL_NAME.lower()
    for prefix, limit in COMPACTION_TOKEN_LIMITS.items():
        if prefix.lower() in name_lower:
            return limit
    return 0


def _read_game_source(game_id: str) -> str | None:
    """Read the game's Python source file for cheat mode. Returns None if not found."""
    short = game_id.split("-", 1)[0]
    parts = game_id.split("-", 1)
    if len(parts) < 2:
        return None
    version_hash = parts[1]
    path = ROOT_DIR / ENVIRONMENTS_DIR / short / version_hash / f"{short}.py"
    try:
        return path.read_text()
    except (FileNotFoundError, PermissionError):
        return None


def _build_level_summary(
    level_boards: list[tuple[int, np.ndarray, np.ndarray]],
    level_actions: list[tuple[int, list[str]]],
    level_code: list[tuple[int, str]],
    current_level: int,
    current_board: np.ndarray,
) -> str:
    """Build a concise history of completed levels for compacted context."""
    parts: list[str] = []
    for lvl, start_board, end_board in level_boards:
        parts.append(f"=== Level {lvl} (completed) ===")
        parts.append(f"Start board (CSV):\n{board_to_csv(start_board)}")
        actions_for_level = [a for l, a in level_actions if l == lvl]
        if actions_for_level:
            action_list = actions_for_level[0]
            parts.append(f"Actions taken ({len(action_list)}): {', '.join(action_list)}")
        code_for_level = [c for l, c in level_code if l == lvl]
        if code_for_level:
            parts.append(f"Code at end of level:\n```\n{code_for_level[-1]}\n```")
        parts.append(f"End board (CSV):\n{board_to_csv(end_board)}")
        parts.append("")

    parts.append(f"=== Current level {current_level} ===")
    parts.append(f"Current board (CSV):\n{board_to_csv(current_board)}")
    return "\n".join(parts)


def _estimate_message_chars(msg: dict[str, Any]) -> tuple[int, int]:
    """(text characters, image count) of a chat message, for the trim estimate."""
    chars, images = 0, 0
    for key, value in msg.items():
        if key == "content" and isinstance(value, list):
            for part in value:
                if isinstance(part, dict) and part.get("type") in ("image_url", "input_image"):
                    images += 1
                elif isinstance(part, dict):
                    chars += len(str(part.get("text") or ""))
        elif isinstance(value, str):
            chars += len(value)
        elif key == "tool_calls" and value:
            chars += len(json.dumps(value, default=str))
    return chars, images


def _trim_conversation(
    conversation: list[dict[str, Any]],
    pinned_present: bool,
    total_tokens: int,
    system_prompt: str,
    tool_definitions: list[dict[str, Any]] | None,
    level_start_text: str,
    level_start_board: np.ndarray,
    target_tokens: int,
    *,
    keep_last_exchanges: int = 2,
    pin_extra: str = "",
    level_no: Any = "?",
    pin_images: list | None = None,
) -> tuple[list[dict[str, Any]], int, int]:
    """Drop the oldest whole exchanges (a user message and everything after it up to the next user
    message) until the estimated prompt is below `target_tokens`; no summary is written. The current
    level's start message (text + big image) is pinned as the first message. Returns
    (new conversation, number of dropped messages, estimated kept tokens)."""
    image_tokens = 600
    body = conversation[1:] if pinned_present else list(conversation)
    est = []
    for m in body:
        c, i = _estimate_message_chars(m)
        est.append(c / 3.3 + i * image_tokens)
    sys_est = (len(system_prompt) + len(json.dumps(tool_definitions or [], default=str))) / 3.3
    old_pin_est = 0.0
    if pinned_present and conversation:
        c, i = _estimate_message_chars(conversation[0])
        old_pin_est = c / 3.3 + i * image_tokens
    scale = total_tokens / max(1.0, sys_est + old_pin_est + sum(est))
    pin_text = (f"[Start of level {level_no} - kept at the top when older turns are dropped to save context; "
                "the most recent turns follow after it]\n\n" + level_start_text)
    if pin_extra:
        pin_text += "\n\n" + pin_extra
    pin_msg = {"role": "user", "content": _chat_user_content(
        pin_text, [*_build_board_image_attachments(level_start_board, None), *(pin_images or [])], provider="local")}
    pc, pi = _estimate_message_chars(pin_msg)
    pin_est = pc / 3.3 + pi * image_tokens
    if keep_last_exchanges <= 0:   # emergency: drop every exchange
        return [pin_msg], len(conversation), int((sys_est + pin_est) * scale)
    user_idx = [i for i, m in enumerate(body) if m.get("role") == "user"]
    cuts = user_idx[1:len(user_idx) - keep_last_exchanges + 1] if len(user_idx) > keep_last_exchanges else []
    keep_from = 0
    for cut in cuts:
        keep_from = cut
        if (sys_est + pin_est + sum(est[cut:])) * scale <= target_tokens:
            break
    if keep_from == 0:
        return conversation, 0, int((sys_est + old_pin_est + sum(est)) * scale)
    dropped = keep_from + (1 if pinned_present else 0)
    kept_est = int((sys_est + pin_est + sum(est[keep_from:])) * scale)
    return [pin_msg] + body[keep_from:], dropped, kept_est


def _compaction_rejection_reason(candidate: Any) -> str | None:
    """Why a compaction response cannot be used as the handover summary (None = usable)."""
    text = (candidate.text or "").strip()
    if not text:
        return "empty final text"
    if text.casefold().startswith("error:"):
        return "error sentinel"
    if candidate.tool_calls:
        return "tool calls instead of a text-only summary"
    if "<think>" in text.casefold() or "</think>" in text.casefold():
        return "unparsed reasoning tags"
    # Gross sanity check, not a claim that length establishes summary quality.
    if len(text) < 120 or len(text.split()) < 20:
        return "implausibly short handover"
    return None


def main() -> None:
    if IS_KAGGLE:
        _run_kaggle()
    elif isinstance(GAME_ID, list):
        _run_parallel(GAME_ID)
    else:
        _run_game(GAME_ID, quiet=False)


def _detect_kaggle_submission_mode() -> bool:
    """Detect if this is a real competition submission.
    
    Uses KAGGLE_IS_COMPETITION_RERUN env var first (most reliable),
    then falls back to gateway probing with retries.
    """
    if os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").lower() in ("true", "1", "yes"):
        return True
    import urllib.request
    for attempt in range(3):
        try:
            req = urllib.request.Request(f"{KAGGLE_GATEWAY_URL}/api/games", method="GET")
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            if attempt < 2:
                import time
                time.sleep(3)
    return False


def _run_kaggle() -> None:
    """Kaggle entry-point: discover games, run them with deadline + reset-on-game-over."""
    from time import time as wall_time
    import arc_agi
    from arc_agi import OperationMode

    # Use notebook start time if available (set in cell 1), otherwise fallback to now
    notebook_start = float(os.environ.get("NOTEBOOK_START_TIME", "0"))
    start_time = notebook_start if notebook_start > 0 else wall_time()
    deadline = start_time + DEADLINE_SECONDS if DEADLINE_SECONDS else None

    is_submission = _detect_kaggle_submission_mode()
    os.environ["_ARC_IS_SUBMISSION"] = "1" if is_submission else "0"
    op_mode = OperationMode.COMPETITION if is_submission else OperationMode.OFFLINE
    mode_label = "SUBMISSION" if is_submission else "PUBLIC TEST"
    print(f"[Kaggle] Mode: {mode_label} | Model: {MODEL_NAME} ({MODEL_THINKING_EFFORT})")
    if deadline:
        remaining_h = (deadline - wall_time()) / 3600
        print(f"[Kaggle] Deadline: {remaining_h:.2f}h remaining")
    compaction_limit = COMPACTION_TOKEN_LIMITS.get("local", 0)
    print(f"[Kaggle] Concurrency: {MAX_CONCURRENT_REQUESTS} | Compaction: {compaction_limit // 1000}k tokens")
    if KAGGLE_QUICK_TEST:
        print(f"[Kaggle] QUICK_TEST enabled | MAX_TURNS={MAX_TURNS}")

    # Submission mode: gateway serves games; public test: use local environment_files
    if is_submission:
        # The competition gateway runs at gateway:8001 with a fixed test key.
        # The arc_agi library defaults to the public API (three.arcprize.org),
        # which is unreachable since internet is disabled during submission.
        # Reference notebooks set ARC_BASE_URL and ARC_API_KEY explicitly.
        gateway_url = os.environ.get("ARC_BASE_URL", KAGGLE_GATEWAY_URL)
        api_key = os.environ.get("ARC_API_KEY", "test-key-123")
        os.environ.setdefault("ARC_BASE_URL", gateway_url)
        os.environ.setdefault("ARC_API_KEY", api_key)
        print(f"[Kaggle] Gateway: {gateway_url} | API key: {'set' if api_key else 'unset'}")

        # Wait for gateway readiness before proceeding
        import urllib.request
        gateway_ready = False
        for attempt in range(10):
            try:
                req = urllib.request.Request(f"{gateway_url}/api/games", method="GET")
                with urllib.request.urlopen(req, timeout=15) as resp:
                    if resp.status == 200:
                        gateway_ready = True
                        print(f"[Kaggle] Gateway ready (attempt {attempt + 1})")
                        break
            except Exception as exc:
                print(f"[Kaggle] Gateway not ready (attempt {attempt + 1}/10): {exc}")
                if attempt < 9:
                    import time
                    time.sleep(5)
        if not gateway_ready:
            print("[Kaggle] WARNING: Gateway not confirmed ready after 10 attempts, proceeding anyway")

        arc = arc_agi.Arcade(
            operation_mode=op_mode,
            arc_base_url=gateway_url,
            arc_api_key=api_key,
        )
    else:
        envs_dir = os.environ.get("ARC_ENVIRONMENTS_DIR", "")
        if not envs_dir:
            import glob as _g
            env_dirs = _g.glob("/kaggle/input/competitions/*/environment_files")
            envs_dir = env_dirs[0] if env_dirs else ENVIRONMENTS_DIR
        print(f"[Kaggle] Using environments_dir: {envs_dir}")
        arc = arc_agi.Arcade(operation_mode=op_mode, environments_dir=envs_dir)

    game_ids = sorted(set(
        getattr(e, "game_id", str(e))
        for e in arc.get_environments()
    ))

    if not game_ids:
        print("[Kaggle] No games found!")
        if is_submission:
            print("[Kaggle] CRITICAL: Empty game catalog in submission mode — gateway may not be configured correctly")
        return

    short_ids = [gid.split("-")[0] for gid in game_ids]
    print(f"[Kaggle] Found {len(game_ids)} games: {', '.join(short_ids[:10])}{'...' if len(game_ids) > 10 else ''}")

    if KAGGLE_QUICK_TEST:
        max_games = int(os.environ.get("MAX_GAMES", "3"))
        preferred = ["ls20", "vc33", "ft09"]
        # Match preferred by short prefix
        selected = [g for g in game_ids if g.split("-")[0] in preferred]
        if len(selected) < max_games:
            selected += [g for g in game_ids if g not in selected]
        game_ids = selected[:max_games]
        print(f"[Kaggle] Quick test: limited to {len(game_ids)} games")

    logging.disable(logging.INFO)
    _parallel_completed.clear()

    max_workers = min(
        len(game_ids),
        MAX_CONCURRENT_REQUESTS or len(game_ids),
    )

    # Per-game time budget for fair coverage across all games
    allocator: TimeAllocator | None = None
    per_game_budget: float | None = None
    if deadline:
        available = deadline - wall_time()
        if available > 0 and len(game_ids) > 0:
            per_game_budget = max(120, available * max_workers / len(game_ids))
            print(f"[Kaggle] Per-game budget: {per_game_budget:.0f}s ({per_game_budget/60:.1f}min) | {max_workers} workers")
            if SMART_ALLOCATION:
                allocator = TimeAllocator(deadline, game_ids, max_workers)
                print("[Kaggle] Smart allocation ENABLED")
            elif LEVEL_WEIGHTED_ALLOCATION:
                from .level_alloc import LevelWeightedAllocator
                allocator = LevelWeightedAllocator(deadline, game_ids, max_workers)
                print(f"[Kaggle] Level-weighted allocation ENABLED: {allocator.describe()}")

    # v2.7 priority scheduler: every game starts at once and shares the run's deadline; at most max_workers play.
    _sched = None
    if PRIORITY_SCHEDULER and deadline:
        allocator, per_game_budget = None, None
        _sched = _scheduler.configure(max_workers, deadline, list(game_ids), tau_tokens=PRIORITY_TAU_TOKENS,
                                      stretch_cap_s=PRIORITY_STRETCH_CAP_SECONDS,
                                      default_levels=PRIORITY_DEFAULT_LEVELS, log=_sync_print)
        print(f"[Kaggle] {_sched.describe()}")

    # Track worker outcomes for diagnostics
    worker_success: list[str] = []
    worker_failed: list[str] = []
    _worker_lock = threading.Lock()

    def _worker(gid: str) -> None:
        _scheduler.set_thread_key(gid)
        # Never start a game while the server is down (a failed first request would end it at once and
        # drain the queue): wait for the server first; the game's own budget starts afterwards.
        if _is_local_provider():
            _max_wait = SERVER_WAIT_SECONDS
            if deadline:
                _max_wait = min(_max_wait, max(0.0, deadline - wall_time()))
            if _max_wait > 0:
                wait_for_server(_max_wait, label=gid.split("-")[0])
        if allocator is not None:
            game_deadline = allocator.get_deadline(gid)
        elif per_game_budget:
            game_deadline = wall_time() + per_game_budget
            if deadline:
                game_deadline = min(game_deadline, deadline)
        elif deadline:
            game_deadline = deadline
        else:
            game_deadline = None
        try:
            _run_game(
                gid,
                quiet=True,
                arcade_override=arc,
                deadline=game_deadline,
                skip_scorecard_close=True,
                allocator=allocator,
            )
            with _worker_lock:
                worker_success.append(gid)
        except SystemExit:
            with _worker_lock:
                worker_failed.append(gid)
        except Exception as exc:
            import traceback
            _sync_print(f"[{gid}] FATAL: {exc!r}\n{traceback.format_exc()}")
            with _worker_lock:
                worker_failed.append(gid)

    with concurrent.futures.ThreadPoolExecutor(max_workers=(len(game_ids) if _sched is not None else max_workers)) as pool:
        futures = {pool.submit(_worker, gid): gid for gid in game_ids}
        # v2.10: never wait for a stuck game thread past the deadline (+ grace): the scorecard must be closed
        _wait_s = (max(60.0, deadline - wall_time() + WAIT_GRACE_SECONDS) if deadline else None)
        _done, _not_done = concurrent.futures.wait(futures, timeout=_wait_s)
        if _not_done:
            print(f"[Kaggle] {len(_not_done)} game thread(s) still running after the deadline; closing the scorecard anyway: "
                  + ", ".join(futures[f].split('-')[0] for f in _not_done))
            pool.shutdown(wait=False, cancel_futures=True)
            try:
                pool._threads.clear()   # the with-block exit must not join the stuck thread(s)
            except Exception:
                pass
    if _sched is not None:
        print(_sched.summary())
        _scheduler.reset()

    print(f"[Kaggle] Workers done: {len(worker_success)} succeeded, {len(worker_failed)} failed out of {len(game_ids)}")
    if worker_failed:
        failed_short = [gid.split('-')[0] for gid in worker_failed]
        print(f"[Kaggle] Failed games: {', '.join(failed_short)}")
    if allocator is not None:
        print(allocator.summary())

    # Close scorecard ONCE after all workers finish
    try:
        scorecard = arc.close_scorecard()
        if scorecard is not None:
            score = getattr(scorecard, "score", None)
            levels = getattr(scorecard, "total_levels_completed", None)
            print(f"[Kaggle] Final scorecard: score={score}, levels_completed={levels}")
    except Exception as exc:
        print(f"[Kaggle] close_scorecard failed: {exc!r}")

    _print_parallel_summary()

    # Submission parquet: only write a DUMMY if NOT in submission mode.
    # In submission mode, the gateway produces the real parquet — never overwrite it.
    if not is_submission:
        parquet_path = "/kaggle/working/submission.parquet"
        if not os.path.exists(parquet_path):
            try:
                import pandas as pd
                pd.DataFrame({"dummy": [0]}).to_parquet(parquet_path)
                print("[Kaggle] Created dummy submission.parquet (public test only).")
            except Exception:
                pass


def _run_parallel(game_ids: list[str]) -> None:
    logging.disable(logging.INFO)
    _parallel_completed.clear()

    cheat_tag = " | CHEAT MODE" if CHEAT_MODE else ""
    _sync_print(
        f"Parallel mode: {len(game_ids)} games | Model: {MODEL_NAME} ({MODEL_THINKING_EFFORT}) | "
        f"Target level: {TARGET_LEVEL} | Max turns: {MAX_TURNS}{cheat_tag}"
    )
    _sync_print(f"Games: {', '.join(game_ids)}")
    _sync_print("")

    max_workers = min(
        len(game_ids),
        MAX_CONCURRENT_REQUESTS or len(game_ids),
    )

    def _worker(gid: str) -> None:
        try:
            _run_game(gid, quiet=True)
        except SystemExit:
            pass
        except Exception as exc:
            _sync_print(f"[{gid}] FATAL: {exc!r}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_worker, gid): gid for gid in game_ids}
        concurrent.futures.wait(futures)

    _print_parallel_summary()


def _print_parallel_summary() -> None:
    if not _parallel_completed:
        return
    print(f"\n{'='*60}")
    print(f"All {len(_parallel_completed)} games finished:")
    # Sort by levels desc, then actions desc for quick readability
    sorted_results = sorted(_parallel_completed, key=lambda r: (-r.get("levels", 0), -r.get("actions", 0)))
    for r in sorted_results:
        lvl = r.get('levels', 0)
        acts = r.get('actions', 0)
        msgs = r.get('messages', 0)
        wall = r.get('wall_seconds', 0)
        _act = (r.get("scheduler") or {}).get("active_s")
        print(f"  {r.get('game', '?'):8s} L{lvl} | {acts:3d} actions | {msgs:3d} msgs | {wall/60:5.1f}m"
              + (f" (active {_act/60:5.1f}m)" if _act is not None else "") + f" | {r.get('kind', '?')}")
    print(f"{'='*60}")

    # Aggregate performance metrics
    total_input_miss = sum(r.get("input_miss", 0) for r in _parallel_completed)
    total_input_cached = sum(r.get("input_cached", 0) for r in _parallel_completed)
    total_output = sum(r.get("output", 0) for r in _parallel_completed)
    total_messages = sum(r.get("messages", 0) for r in _parallel_completed)
    total_actions = sum(r.get("actions", 0) for r in _parallel_completed)
    total_levels = sum(r.get("levels", 0) for r in _parallel_completed)
    total_wall = max((r.get("wall_seconds", 0) for r in _parallel_completed), default=0)
    n_games = len(_parallel_completed)
    games_with_levels = sum(1 for r in _parallel_completed if r.get("levels", 0) > 0)
    if total_wall > 0 and (total_input_miss + total_output) > 0:
        print(f"\nPerformance summary ({n_games} games, {total_wall:.0f}s wall):")
        print(f"  Levels completed: {total_levels} across {games_with_levels}/{n_games} games")
        print(f"  Messages: {total_messages} | Actions: {total_actions}")
        print(f"  Input tokens:  {total_input_miss:,} miss + {total_input_cached:,} cached = {total_input_miss + total_input_cached:,} total")
        print(f"  Output tokens: {total_output:,}")
        print(f"  Avg output/turn: {total_output / max(total_messages, 1):,.0f} tokens")
        print(f"  Throughput (wall): {(total_input_miss + total_output) / total_wall:,.0f} total tok/s, {total_output / total_wall:,.0f} output tok/s")
        print(f"  Avg turn time: {total_wall / max(total_messages, 1):.1f}s (wall / messages)")
        cache_pct = total_input_cached / max(total_input_miss + total_input_cached, 1) * 100
        print(f"  Prefix cache hit rate: {cache_pct:.1f}%")
        # Extrapolation for submission
        available_s = 8 * 3600 + 30 * 60  # ~8h30m after setup
        avg_turn_s = total_wall / max(total_messages, 1)
        concurrent = MAX_CONCURRENT_REQUESTS or 8
        total_possible_turns = available_s * concurrent / max(avg_turn_s, 1)
        est_turns_per_game = total_possible_turns / 110
        print(f"  Estimated turns/game (110 games, {available_s/3600:.1f}h, {concurrent} concurrent): ~{est_turns_per_game:.0f}")


def _run_game(game_id_arg: str, *, quiet: bool = False, arcade_override: Any = None, deadline: float | None = None, skip_scorecard_close: bool = False, allocator: TimeAllocator | None = None) -> None:
    """Play one game (see _run_game_impl); the replay is always written out, even if the game raises."""
    _sched = _scheduler.get()                     # v2.7: None unless the run configured the priority scheduler
    _skey = _scheduler.thread_key(game_id_arg) if _sched is not None else None
    if _sched is not None:
        _sched.start(_skey)                       # wait for a slot (never-played games first, in dispatch order)
    try:
        return _run_game_impl(game_id_arg, quiet=quiet, arcade_override=arcade_override, deadline=deadline,
                              skip_scorecard_close=skip_scorecard_close, allocator=allocator)
    finally:
        if _sched is not None:
            try:
                _sched.finish(_skey)
            except Exception as exc:
                print(f"[sched] finish failed: {exc!r}")
        try:
            _ext.end_game()          # v2.6: stop this game's tracker worker
        except Exception as exc:
            print(f"[layered tracker] stop failed: {exc!r}")
        _r = getattr(_replay_tls, "replay", None)
        _replay_tls.replay = None
        if _r is not None:
            try:
                _r.flush()
            except Exception as exc:
                print(f"[replay] final write failed: {exc!r}")


def _run_game_impl(game_id_arg: str, *, quiet: bool = False, arcade_override: Any = None, deadline: float | None = None, skip_scorecard_close: bool = False, allocator: TimeAllocator | None = None) -> None:
    from time import time as wall_time
    _sched = _scheduler.get()                     # v2.7 priority scheduler (None = equal per-game budgets, as before)
    _skey = _scheduler.thread_key(game_id_arg) if _sched is not None else None

    if not quiet:
        log_tag("INFO", "Starting ARC agent...", Color.BLUE, bold=True)
        cheat_tag = " | CHEAT MODE" if CHEAT_MODE else ""
        print(
            f"Game: {game_id_arg} | Model: {MODEL_NAME} ({MODEL_THINKING_EFFORT}) | "
            f"Target level: {TARGET_LEVEL} | Max turns: {MAX_TURNS}{cheat_tag}"
        )

    if _is_local_provider() and not server_ready():
        _max_wait = SERVER_WAIT_SECONDS if not deadline else min(SERVER_WAIT_SECONDS, max(0.0, deadline - wall_time()))
        if _max_wait > 0:
            wait_for_server(_max_wait, label=str(game_id_arg).split("-")[0])

    if arcade_override is not None:
        arc = arcade_override
    else:
        arc = arc_agi.Arcade(operation_mode=OPERATION_MODE, environments_dir=ENVIRONMENTS_DIR)
    env = arc.make(game_id_arg, seed=SEED, include_frame_data=True)
    if env is None or env.observation_space is None:
        raise RuntimeError("Failed to initialize environment")
    _game_start_wall = wall_time()

    game_id = str(getattr(getattr(env, "info", None), "game_id", game_id_arg) or game_id_arg)
    game_id_short = game_id.split("-", 1)[0]
    current_level = get_current_level(env, obs=env.observation_space)
    if TARGET_LEVEL < 1:
        raise ValueError(f"TARGET_LEVEL must be >= 1, got {TARGET_LEVEL}")
    target_level = TARGET_LEVEL
    current_board = observation_board(env.observation_space).copy()

    action_specs = action_specs_from_env(env)
    action_labels = [s.label for s in action_specs]
    _ext.start_game(game_id, current_board)
    _ext.ctx().available_actions = list(action_labels)
    _ext.tracker_start(current_board, current_level)     # v2.6 layered tracker (no-op when LAYERED_TRACKER is off)
    system_prompt = build_system_prompt(action_labels, board_shape=tuple(current_board.shape))

    if CHEAT_MODE:
        game_source = _read_game_source(game_id)
        if game_source:
            system_prompt += (
                "\n\n--- GAME SOURCE CODE ---\n"
                "The full code that runs this game is as follows. Use it to understand "
                "exactly how the game mechanics work, what actions do, level transitions, "
                "win conditions, etc.:\n\n"
                f"```python\n{game_source}\n```"
            )
            if not quiet:
                log_tag("CHEAT", f"Attached game source ({len(game_source)} chars)", Color.YELLOW, bold=True)
        else:
            if not quiet:
                log_tag("CHEAT", "Game source file not found — continuing without it", Color.RED)

    _install_replay_viewer()
    shared_js = REPLAYS_DIR / REPLAY_SHARED_JS_NAME
    if ENABLE_REPLAYS and not IS_KAGGLE and not shared_js.exists():
        raise RuntimeError(f"Replay viewer missing: {shared_js}")

    replay_model_tag = f"CHEAT-{MODEL_NAME}" if CHEAT_MODE else MODEL_NAME
    replay = ReplayRun(
        game_id=game_id, model_name=replay_model_tag, thinking_effort=MODEL_THINKING_EFFORT,
        initial_level=current_level,
        run_dir=REPLAYS_DIR / replay_game_folder(game_id),
        start_stamp=datetime.now().strftime("%Y-%m-%d_%H-%M"),
        shared_js_relpath=f"../{REPLAY_SHARED_JS_NAME}",
    )
    _replay_tls.replay = replay
    # v2.5-log: the score is computed by the harness from the level baselines (public games) and the actions spent per
    # level (every env step incl. RESETs, also the automatic one after a game over), as the official formula does.
    _baselines = _env_baselines(env, game_id) if ENABLE_REPLAYS else None
    _win_levels = _coerce_int(getattr(env.observation_space, "win_levels", 0)) or (len(_baselines) if _baselines else None)
    level_actions_scored: dict[int, int] = {}     # level -> actions spent on it (completed levels)
    auto_resets_this_level = 0                    # automatic resets after game overs on the current level
    replay.meta = {
        "harness_version": HARNESS_VERSION, "prompt_variant": PROMPT_VARIANT, "baseline_actions": _baselines,
        "win_levels": _win_levels, "max_output_tokens": MAX_OUTPUT_TOKENS, "context_trim": CONTEXT_TRIM,
        "context_trim_tokens": [CONTEXT_TRIM_TRIGGER_TOKENS, CONTEXT_TRIM_TARGET_TOKENS],
        "ephemeral_reasoning": EPHEMERAL_REASONING, "note_confidence": NOTE_CONFIDENCE,
    }
    if EPHEMERAL_REASONING and not _is_local_provider():
        raise ValueError("Ephemeral reasoning requires local vLLM with the Qwen chat template")
    if NOTE_CONFIDENCE and EVIDENCE_NOTES:
        raise ValueError("The confidence-note experiment must run with V2_EVIDENCE_NOTES=0")
    _decision = DecisionPolicy(RecoverySettings.from_env(), minimal=MINIMAL_HARNESS)
    if TIERED_REASONING and not ADAPTIVE_REASONING:
        raise ValueError("Tiered reasoning requires V2_ADAPTIVE_REASONING=1")
    _reasoning = (TieredReasoningPolicy if TIERED_REASONING else ReasoningPolicy)(ADAPTIVE_REASONING)
    # Under priority scheduling games share a deadline, so use the common gameplay
    # window, including queue time. With individual deadlines use this game's window.
    _reasoning_budget_s = (max(0.0, deadline - (_sched.t0 if _sched else _game_start_wall))
                           if deadline is not None else None)
    if ADAPTIVE_REASONING and not _is_local_provider():
        raise ValueError("Adaptive reasoning requires local vLLM with the patched Qwen template")
    replay.meta["adaptive_reasoning"] = reasoning_configuration(ADAPTIVE_REASONING, TIERED_REASONING)
    replay.meta["adaptive_reasoning"]["time_basis"] = "shared_gameplay_deadline" if _sched else "game_deadline"
    if _decision.settings.mode == "recover_after_cap" and not _is_local_provider():
        raise ValueError("recover_after_cap requires local vLLM; use off/observe for other providers")
    from dataclasses import asdict as _dc_asdict
    replay.meta["decision_recovery_settings"] = _dc_asdict(_decision.settings)
    if MINIMAL_HARNESS:
        from .thinking_budget import thinking_token_budget, THINKING_END
        replay.meta["thinking_token_budget"] = {
            "normal": thinking_token_budget(MAX_OUTPUT_TOKENS),
            "recovery": thinking_token_budget(_decision.settings.request_tokens, recovery=True),
            "effort": "xhigh", "proposal_interface": "run:propose_actions",
        }
    _decision_resets = 0

    all_boards: list[np.ndarray] = [current_board.copy()]
    actions_taken: list[str] = []
    level_action_start_index = 0
    history: list[TransitionRecord] = []
    code_file = VirtualCodeFile()
    code_file.loader = _ext.code_loader   # v2.11: the code file is loaded (checked) in a worker process
    conversation: list[dict[str, Any]] = []
    client = create_openai_client()
    # v2.3 per-level memory: history (TransitionRecords) covers every attempt on the current level and is
    # cleared when a level is completed; all_boards / states cover the current attempt only.
    level_total_start_index = 0          # actions_taken index of the current level's first action (all attempts)
    attempt_no = 1                       # attempt on the current level (game overs and RESETs start a new one)
    attempt_starts: list[int] = [0]      # history index where each attempt starts
    def _decision_identity():
        return decision_identity(game_id, current_level, attempt_no,
                                 len(actions_taken) + _decision_resets, current_board)

    def _finish_reasoning_turn():
        if TIERED_REASONING:
            _reasoning.finish_turn(_decision_identity(), evidence=execution_evidence(_ext.ctx(), code_file.source))
        else:
            _reasoning.finish_turn(_decision_identity())

    def _validate_fallback(action_text):
        if deadline and wall_time() >= deadline:
            return "deadline reached"
        if _ext.ctx().pending_terminal is not None:
            return "terminal transition pending"
        if _ext.ctx().after_env_error:
            return "last environment outcome unknown; automatic action requires a fresh observation"
        parts = action_text.split()
        if not parts or parts[0] not in action_labels or parts[0] in {"RESET", "UNDO"}:
            return "unavailable or excluded action"
        if len(parts) != (4 if parts[0] == "CLICK" else 1):
            if MINIMAL_HARNESS or not (parts[0] == "CLICK" and len(parts) == 2 and parts[1].startswith("@")):
                return "expected one exact action (CLICK row col value or CLICK @name)"
        if MINIMAL_HARNESS and parts[0] == "CLICK":
            try:
                tuple(int(v) for v in parts[1:])
            except ValueError:
                return "CLICK requires integer row col colour; no component references"
        candidate, error = _ext.parse_action_line(action_text, action_specs, current_board)
        if error or candidate is None:
            return error or "invalid action"
        # Automatic fallback never bypasses known no-op/death guards, even if normal play does.
        return _ext.known_guard(current_board, remaining_actions_indicator_rect, candidate.as_text(), repeat=False)
    past_levels: list[dict[str, Any]] = []
    conv_tokens_est = 0                  # estimated tokens of `conversation` (for the pre-send trim)
    store: dict[str, Any] = {}           # v2.5: data the model keeps across run calls (store["x"] = ...)
    notes: dict[str, str] = {}           # v2.5: the model's pinned notes (note tool), key -> text
    note_meta: dict[str, dict[str, Any]] = {}   # v2.12b: key -> {status, text} (EVIDENCE_NOTES)
    confidence_notes = (ConfidenceNotes(NOTES_CHAR_CAP,
        replay.run_dir / f"{replay.start_stamp}_{uuid4().hex}.notes.json", game_id) if NOTE_CONFIDENCE else None)
    if confidence_notes is not None:
        replay.meta["notes_snapshot_file"] = confidence_notes.path.name
    stuck_state: dict[str, Any] = {"level": None, "turns": 0, "next": STUCK_CHECK_TURNS}   # v2.12b: stuck check
    event_log: list[str] = []            # v2.5: harness-written attempt/level log, pinned with the notes
    last_win_effect: list[str | None] = [None]   # text describing the last winning action's effect (set by _apply_level_complete)
    last_win_image: list[bool] = [False]         # v2.9: an image of the winning moment goes with the next level start
    # v2.5-log replay state
    pending_events: list[dict[str, Any]] = []    # game-over / RESET events: written right after the turn that caused them
    cur_turn_info: dict[str, Any] = {}           # request/usage/context of the turn in progress (for a mid-turn finalize)
    cur_turn_extra: dict[str, Any] = {}
    turn_level_completions: list[dict[str, Any]] = []   # level completions of the current turn (attached to its event)
    turn_note_ops: list[dict[str, Any]] = []     # note tool operations of the current turn
    ctx_first_turn = 1                           # first turn whose exchange is still in the conversation
    last_turn_in_conv = 0                        # last turn whose user message is in the conversation

    def _flush_pending_events() -> None:
        while pending_events:
            replay.add_event(**pending_events.pop(0))
    report_validation_next = False       # functions kept from run changed the model: report validation next turn

    # Compaction tracking
    compaction_limit = CONTEXT_TRIM_TRIGGER_TOKENS if CONTEXT_TRIM else _get_compaction_limit()
    compaction_level_boards: list[tuple[int, np.ndarray, np.ndarray]] = []
    compaction_level_actions: list[tuple[int, list[str]]] = []
    compaction_level_code: list[tuple[int, str]] = []
    compaction_level_start_board: np.ndarray = current_board.copy()
    compaction_count: int = 0

    tool_definitions = _command_tool_specs(action_labels, include_estimate=(SMART_ALLOCATION and allocator is not None)) if USE_TOOL_CALLING else []
    replay.add_event(
        board=current_board, actions_taken=0, current_level=current_level,
        kind="initial", title="Initial board",
        available_actions=action_labels,
        message_text="Game started.",
        system_prompt=system_prompt,
        tool_definitions=tool_definitions if tool_definitions else None,
        **_ext.turn_log_kwargs(current_board, None),
    )

    board_h, board_w = current_board.shape
    initial_msg = (
        f"Initial board ({board_h}×{board_w}, 0-based [row, col]): see the attached image; the colour counts and "
        f"{'object' if LAYERED_TRACKER else 'component'} inventory follow. Read exact cells with run, e.g. print(current_board[10:20, 5:15]).\n\n"
        f"Available actions: {', '.join(action_labels)}\n"
        f"Actions taken: 0\n\n"
        "Take a few exploratory actions to observe the game mechanics before committing to a transition model."
    )
    initial_msg += "\n\n" + _full_board_color_inventory_text(current_board)
    initial_msg += "\n\n" + _ext.level_start_objects_text(current_board, None)
    feedback_message = initial_msg
    level_start_text = initial_msg               # pinned after the system prompt when the history is trimmed
    # v2.10: the pinned level start never carries notes / one-time instructions (the current notes come from pin_extra),
    # and a restart keeps the level's original start message pinned
    def _pin_base(_m: str) -> str:
        _cut = len(_m)
        for _mk in ("Before continuing, look at how", "Before continuing, record in your notes", "[Your notes (note tool)"):
            _i = _m.find(_mk)
            if _i >= 0:
                _cut = min(_cut, _i)
        return _m[:_cut].rstrip()
    level_pin_base = _pin_base(initial_msg)
    level_start_text = level_pin_base
    last_restart_msg: list[Any] = [None]
    level_new_colors_text: list[str] = [""]
    win_pred_text: list[str] = [""]
    level_start_board = current_board.copy()
    pinned_in_conversation = False               # conversation[0] is the pinned level-start message
    trim_count = 0
    _sched_level, _sched_tok0, _sched_trims_seen = None, 0, 0   # v2.7: level / output tokens at its start / trims seen
    # v2.8 lean no-action turns: the board / action count / trim count when the last small board image was sent,
    # and the level / action counter lines of the pending user message
    _img_last_board, _img_last_actions, _img_last_trims, _status_text_last = None, -1, -1, None
    _LEAN_NOTE = ("[No action was executed since your previous turn: the board, the level and the action counts are "
                  "unchanged (see the last board image).]")
    # v2.9 act-now nudge: action count and cumulative output tokens at the last executed action (or the last nudge)
    _nudge_actions_seen, _nudge_tok_ref = -1, 0
    _nudge_act_tok = 0
    pending_full_board_for_image: np.ndarray | None = current_board.copy()
    run_finalized = False
    seen_colors: set[int] = set(_color_counts(current_board).keys())
    remaining_actions_indicator_rect: tuple[int, int, int, int] | None = None
    canonical_seen_action_index = _rebuild_canonical_seen_action_index(
        all_boards,
        remaining_actions_indicator_rect,
    )
    consecutive_no_change_action_turns: int = 0
    command_mode_desc = (
        "tool calls"
        if USE_TOOL_CALLING
        else "command blocks (```actions, ```run, etc.)"
    )
    run_hint_desc = "`run` tool" if USE_TOOL_CALLING else "```run"

    def finalize_run(
        *,
        kind: str,
        title: str,
        status_line: str,
        reasoning_text: str,
        results_lines: list[str],
    ) -> None:
        nonlocal run_finalized
        if run_finalized:
            return
        run_finalized = True

        _finish_reasoning_turn()
        if ADAPTIVE_REASONING and cur_turn_info:
            cur_turn_info["reasoning_control"] = _reasoning.snapshot()

        # Terminal act()/actions/path outcomes can bypass normal end-of-turn handling.
        # Close recovery before saving their final replay and counts as well.
        if _decision.settings.mode != "off":
            _decision.sync(_decision_identity())
            if _decision.active:
                _decision.abort("run_finished_without_action")
            if cur_turn_info:
                cur_turn_info["decision_policy"] = _decision.snapshot()

        levels_completed = max(0, current_level - replay.initial_level)
        if allocator is not None:
            allocator.report_finished(
                game_id,
                kind=kind,
                levels_completed=levels_completed,
            )

        scorecard = None
        if skip_scorecard_close:
            try:
                scorecard = arc.get_scorecard()
            except Exception:
                scorecard = None
        else:
            try:
                scorecard = arc.close_scorecard()
            except Exception as exc:
                if not quiet:
                    log_tag("SCORE", f"close_scorecard failed: {exc!r}", Color.YELLOW)
                try:
                    scorecard = arc.get_scorecard()
                except Exception as get_exc:
                    if not quiet:
                        log_tag("SCORE", f"get_scorecard failed: {get_exc!r}", Color.YELLOW)
                    scorecard = None

        _official = None
        if scorecard is not None:
            # Only this game's own environment entry is a per-game score: the shared scorecard's .score is the mean
            # over every game on the card (and was empty / 0.0 in the in-process public test).
            try:
                for _env_entry in (getattr(scorecard, "environments", None) or []):
                    _eid = str(getattr(_env_entry, "id", "") or "")
                    if _eid in (game_id, game_id_arg) or _eid.split("-", 1)[0] == game_id_short:
                        _official = _coerce_float(getattr(_env_entry, "score", None))
            except Exception:
                _official = None
            levels_raw = getattr(scorecard, "total_levels_completed", None)
            actions_raw = getattr(scorecard, "total_actions", None)
            replay.scorecard_total_levels = _coerce_int(levels_raw) if levels_raw is not None else None
            replay.scorecard_total_actions = _coerce_int(actions_raw) if actions_raw is not None else None
        _rows, _computed = _level_scores(_baselines, _win_levels, level_actions_scored)
        if _official is not None and np.isfinite(_official):
            replay.final_score, _src = _official, "scorecard environment entry"
        elif _computed is not None:
            replay.final_score, _src = _computed, "computed by the harness from the level baselines"
        else:
            replay.final_score, _src = None, "unavailable (no level baselines)"
        _trk_summary = _ext.tracker_summary()
        replay.meta["decision_recovery_counts"] = dict(_decision.counts)
        replay.meta["reasoning_counts"] = dict(_reasoning.counts)
        replay.meta["layered_tracker"] = _trk_summary
        replay.meta["object_refs"] = dict(_ext.ctx().references.metrics)
        replay.meta.update({"level_scores": _rows, "score_source": _src, "final_score_official": _official,
                            "final_score_computed": _computed,
                            "level_actions_note": "actions per level count every env step on that level, including "
                                                  "RESETs and the automatic reset after a game over"})
        _flush_pending_events()

        score_raw = "n/a" if replay.final_score is None else f"{replay.final_score:.2f}"
        score_text = f"{score_raw}*" if CHEAT_MODE else score_raw
        levels_completed = max(0, current_level - replay.initial_level)
        token_line = (
            f"input_miss={replay.token_input_miss_tokens}, "
            f"input_cached={replay.token_input_cached_tokens}, "
            f"output={replay.token_output_tokens}, "
            f"input_total={replay.token_input_total_tokens}"
        )
        score_line = f"[Final score] {score_text}"
        summary_lines = [*results_lines, score_line, f"[Token usage totals] {token_line}"]
        final_feedback = "\n\n".join(summary_lines) if summary_lines else status_line
        _final_kwargs: dict[str, Any] = dict(
            board=current_board,
            actions_taken=len(actions_taken),
            current_level=current_level,
            kind=kind,
            title=title,
            available_actions=action_labels,
            message_text=f"Reasoning:\n{reasoning_text}\n\nFeedback:\n{final_feedback}",
            llm_reasoning=reasoning_text,
            harness_feedback=final_feedback,
            final_score=score_text,
            token_input_miss_total=replay.token_input_miss_tokens,
            token_input_cached_total=replay.token_input_cached_tokens,
            token_output_total=replay.token_output_tokens,
            token_input_total=replay.token_input_total_tokens,
            level_scores=_rows,
            notes=dict(notes), notes_structured=(confidence_notes.snapshot() if confidence_notes is not None else None),
            event_log=list(event_log[-80:]), store_summary=_store_summary(store),
            wall_s=round(wall_time() - _game_start_wall, 1),
            remaining_actions_rects=([list(r) for r in norm_rects(remaining_actions_indicator_rect)] or None),
            remaining_actions_rect=(list(norm_rects(remaining_actions_indicator_rect)[0]) if norm_rects(remaining_actions_indicator_rect) else None),
            force_write=True,
        )
        if cur_turn_info:   # finalized in the middle of a turn: keep that turn's model output, input and steps
            _extra = {**cur_turn_info, **cur_turn_extra,
                      **_ext.turn_log_kwargs(current_board, remaining_actions_indicator_rect),
                      "level_complete": list(turn_level_completions) or None,
                      "note_ops": list(turn_note_ops) or None, "attempt": attempt_no}
            _final_kwargs.update({k: v for k, v in _extra.items() if k not in _final_kwargs})
        replay.add_event(**_final_kwargs)
        if quiet:
            with _print_lock:
                game_wall = wall_time() - _game_start_wall
                print(f"\n{'='*60}", flush=True)
                print(
                    f"[{game_id_short}] {status_line}", flush=True
                )
                print(
                    f"[{game_id_short}] Levels: {levels_completed} | "
                    f"Actions: {len(actions_taken)} | Messages: {replay.llm_messages} | "
                    f"Time: {game_wall/60:.0f}m"
                    + (f" (active {_sched.active_seconds(_skey) / 60:.0f}m in {_sched.stats(_skey)['stretches']} stretch(es))"
                       if _sched is not None else ""),
                    flush=True,
                )
                print(f"[{game_id_short}] Tokens: {token_line}", flush=True)
                if _trk_summary.get("frames"):
                    print(f"[{game_id_short}] Layered tracker: {_trk_summary.get('frames')} boards, worker CPU "
                          f"{_trk_summary.get('cpu_s')} s, per board mean {_trk_summary.get('mean_ms')} ms / p90 "
                          f"{_trk_summary.get('p90_ms')} ms / max {round(float(_trk_summary.get('max_ms') or 0))} ms, "
                          f"game waited {_trk_summary.get('waits_s')} s"
                          + (f", stopped: {_trk_summary.get('off_reason')}" if _trk_summary.get('off_reason') else ""),
                          flush=True)
                elif _trk_summary.get("off_reason"):
                    print(f"[{game_id_short}] Layered tracker off: {_trk_summary.get('off_reason')}", flush=True)
                if CHEAT_MODE:
                    print(f"[{game_id_short}] * = cheat mode (game source provided to AI)", flush=True)
                _parallel_completed.append({
                    "game": game_id_short, "kind": kind,
                    "levels": levels_completed,
                    "messages": replay.llm_messages,
                    "actions": len(actions_taken),
                    "input_miss": replay.token_input_miss_tokens,
                    "input_cached": replay.token_input_cached_tokens,
                    "output": replay.token_output_tokens,
                    "wall_seconds": game_wall,
                    **({"scheduler": _sched.stats(_skey)} if _sched is not None else {}),
                    "score": replay.final_score, "score_source": replay.meta.get("score_source"),
                    "level_actions": {str(k): v for k, v in sorted(level_actions_scored.items())},
                    "baselines": _baselines,
                    "reasoning_counts": dict(_reasoning.counts),
                    "layered_tracker": {k: _trk_summary.get(k) for k in ("frames", "cpu_s", "wall_s", "waits_s", "mean_ms",
                                                                        "p50_ms", "p90_ms", "max_ms", "restarts", "errors",
                                                                        "timeouts", "skipped", "off_reason")},
                })
                done = len(_parallel_completed)
                prior = [
                    f"{r['game']}:L{r['levels']}/{r['actions']}a"
                    for r in _parallel_completed[:-1]
                ]
                if prior:
                    print(f"  Done so far ({done}): {', '.join(prior)}, {game_id_short}:L{levels_completed}/{len(actions_taken)}a", flush=True)
        else:
            print(f"\n{'='*60}\n{status_line}")
            print(f"Final score: {score_text}")
            print(f"Token usage totals: {token_line}")
            if CHEAT_MODE:
                print("* = cheat mode (game source code was provided to the AI)")

    def _minimal_indicator_facts():
        # Only confirmed transitions in this attempt can support a trend estimate.
        uncertain = (_ext.ctx().after_env_error or any(
            getattr(rec, "unreliable", False) for rec in history[attempt_starts[-1]:]))
        return _ext._gl.indicator_facts(all_boards, norm_rects(remaining_actions_indicator_rect),
                                       names=_ext.color_names(), strict=True, uncertain=uncertain)

    def _hist_summary_list() -> list[dict[str, Any]]:
        """history for the run namespace: every transition on this level, tagged with its attempt."""
        out: list[dict[str, Any]] = []
        rect = remaining_actions_indicator_rect
        for i, rec in enumerate(history):
            b, a = to_uint8_board(rec.before), to_uint8_board(rec.after)
            changed = outside = None
            if b.shape == a.shape:
                diff = b != a
                changed = int(np.count_nonzero(diff))
                outside = changed
                if norm_rects(rect):
                    outside = int(np.count_nonzero(diff & ~rects_mask(rect, diff.shape)))
            out.append({"attempt": bisect.bisect_right(attempt_starts, i), "action": rec.action.as_text(),
                        "outcome": (None if rec.outcome == "continue" else rec.outcome), "changed_cells": changed, "changed_outside_indicator": outside,
                        "before": b, "after": a})
            if MINIMAL_HARNESS:
                out[-1]["animation_index"] = i
        return out

    def _stuck_check_lines() -> list[str]:
        """v2.12b: the pinned notes not marked seen, and the actions / clickable regions not tried on this level."""
        out: list[str] = []
        _un = [(k, m) for k, m in note_meta.items() if k in notes and m.get("status") != "seen"]
        if _un:
            out.append("- Your pinned notes not marked seen: " + "; ".join(
                f"{k} [{m.get('status') or 'status not given'}; certainty: {str(m.get('certainty') or 'not given')[:60]}]: "
                f"{str(m.get('text') or '')[:140]}" for k, m in _un[:8])
                + (f"; and {len(_un) - 8} more" if len(_un) > 8 else ""))
        _hs = _hist_summary_list()
        _lib = _ext._gl.make_lib({"get_states": lambda: list(all_boards), "get_history": lambda: _hs,
                             "get_actions": lambda: list(actions_taken[level_action_start_index:]),
                             "rects": norm_rects(remaining_actions_indicator_rect), "names": _ext.color_names(),
                             "printer": lambda *a, **k: None, "anim_frames": [], "action_labels": list(action_labels)})
        _ut = _lib.untried()
        _acts = list(_ut.get("actions") or [])
        if "CLICK" in action_labels and not any(str(h.get("action") or "").upper().startswith("CLICK") for h in _hs):
            _acts.append("CLICK")
        if _acts:
            out.append("- Actions not used on this level: " + ", ".join(_acts))
        _cl = list(_ut.get("clicks") or [])
        if _cl and "CLICK" in action_labels:
            from collections import Counter as _Cn
            _cb = to_uint8_board(current_board)
            _cnt = _Cn(_ext.color_names().get(int(_cb[r, c]), int(_cb[r, c])) for r, c in _cl)
            out.append("- Regions never clicked on this level (one-colour regions, indicator excluded): "
                       + ", ".join(f"{nm} x{n}" for nm, n in _cnt.most_common(8))
                       + (f", and {len(_cnt) - 8} more colours" if len(_cnt) > 8 else "")
                       + " (lib.untried()['clicks'] gives one cell of each)")
        return out

    def _notes_text() -> str:
        """The model's pinned notes + the harness event log (shown at trims and level starts)."""
        if MINIMAL_HARNESS:
            from .minimal_memory import present, NOTES_CHARS
            text, _, _ = present([("notes", _ext.ctx().memory["notes"])], max_chars=NOTES_CHARS)
            return "[Your complete durable notes]\n" + text
        parts: list[str] = []
        if NOTE_CONFIDENCE:
            parts.append("[Your notes (note tool) - durable memory; confidence is self-reported, 0-100]\n"
                         + ("\n".join(f"- {k}: {v}" for k, v in notes.items()) if notes else
                            "(empty - save useful discoveries with note(key, text, confidence))"))
        elif notes:
            parts.append(("[Your pinned notes (note tool) - the only part of your reasoning that survives context trimming]\n"
                          if EVIDENCE_NOTES else "[Your notes (note tool) - the only part of your reasoning that survives context trimming]\n")
                         + "\n".join(f"- {k}: {v}" for k, v in notes.items()))
        else:
            parts.append(f"[Your notes (note tool)] (empty - use {'note(key, text, status, certainty, evidence)' if EVIDENCE_NOTES else 'note(key, text)'} to keep confirmed rules with their evidence, "
                         "rules you ruled out, surprising observations, win conditions and open questions)")
        if event_log:
            parts.append("[Event log - written by the harness]\n" + "\n".join(f"- {e}" for e in event_log[-40:]))
        return "\n\n".join(parts)

    def _level_complete_extra() -> str:
        """Appended to every level-start message after a completion: the winning action's effect + the notes."""
        if MINIMAL_HARNESS:
            return "\nCompleted-level boards, winning actions and transitions are accessible through past_levels."
        parts = ["\n\n"]
        if level_new_colors_text[0]:
            parts.append(level_new_colors_text[0] + "\n\n")
        if last_win_effect[0]:
            parts.append(last_win_effect[0] + "\n\n")
        if win_pred_text[0]:
            parts.append(win_pred_text[0] + "\n\n")
        if PROMPT_V29:   # v2.9 post-win reflection
            _ev = [e for e, on in (("the winning action's effect above", bool(last_win_effect[0])),
                                   ("the attached image of the winning moment", bool(last_win_image[0]))) if on]
            parts.append("Before continuing, look at how the previous level was won"
                         + (f" ({' and '.join(_ev)})" if _ev else "")
                         + " and work out why it counted as a win: what the winning state had that your earlier attempts "
                           "lacked. Record the win condition and any new rule in your notes (note tool) - the rule, not the "
                           "action list - no more confidently than the evidence supports, and note anything the new board "
                           "contradicts.\n\n")
        else:
            parts.append("Before continuing, record in your notes (note tool) what completed the previous level - the rule, not the "
                         "action list - and anything the new board contradicts.\n\n")
        parts.append(_notes_text())
        return "".join(parts)

    def _pin_extra_text() -> str:
        """Kept at the top with the level start when old turns are dropped: notes, event log, completed levels, code file."""
        if MINIMAL_HARNESS:
            from .sandbox import top_level_functions
            helpers = ", ".join(name for name, _ in top_level_functions(code_file.source or "")) or "(none)"
            return (_notes_text() + "\nRetained Python helpers: " + helpers
                    + "\nFull data and store remain accessible in Python; their contents are not automatically repeated.")
        parts: list[str] = [_notes_text()]
        _imap = _ext.reference_text(full=True)
        if _imap:
            parts.append(_imap)
        if past_levels:
            lines = ["[Completed levels - recorded by the harness]"]
            for pl in past_levels:
                wa = pl["winning_actions"]
                lines.append(f"Level {pl['level']}: completed with {pl['actions_total']} action(s) over {pl['attempts']} attempt(s); "
                             f"winning attempt ({len(wa)} actions): {', '.join(wa)}")
            parts.append("\n".join(lines))
        _st_line = []
        if store:
            _ks = list(store.keys())
            _st_line.append("store keys: " + ", ".join(repr(k) for k in _ks[:40]) + (" ..." if len(_ks) > 40 else ""))
        _val = _ext.ctx().validation
        if code_file.has("next_state") and _val:   # v2.11
            _st_line.append(f"model check: next_state matches {_val.get('passed')}/{_val.get('total')} transitions of this level"
                            + (f" ({_val.get('unknown')} don't know)" if _val.get("unknown") else ""))
        if _ext.ctx().view_exclude:
            _st_line.append(f"object view: colours {_ext.ctx().view_exclude} count as gaps (set_object_view)")
        if _st_line:
            parts.append("[Harness state] " + "; ".join(_st_line))
        if (code_file.source or "").strip():
            parts.append("[Your code file at the time older turns were dropped (it may have changed since; CODE in run always "
                         "holds the current text)]\n```python\n" + code_file.source + "\n```")
        return "\n\n".join(parts)

    def _do_trim(est_tokens: int, target: int, keep: int, why: str) -> int:
        """Drop the oldest exchanges (pinned level start + completed levels + code file stay). Returns #dropped."""
        nonlocal conversation, pinned_in_conversation, trim_count, conv_tokens_est, ctx_first_turn
        _pin_images = []
        if MINIMAL_HARNESS:
            from .minimal_memory import present, image_url, NOTES_CHARS
            _, _imgs, _ = present([("notes", _ext.ctx().memory["notes"])], max_chars=NOTES_CHARS)
            _pin_images = [BoardImageAttachment(p + ": " + img["label"], image_url(img), 1) for p, img in _imgs]
        new_conv, dropped, kept_est = _trim_conversation(
            conversation, pinned_in_conversation, int(est_tokens), system_prompt, tool_definitions,
            level_start_text, level_start_board, target, keep_last_exchanges=keep, pin_extra=_pin_extra_text(),
            level_no=current_level,
            pin_images=_pin_images,
        )
        if not dropped:
            return 0
        conversation = new_conv
        pinned_in_conversation = True
        trim_count += 1
        conv_tokens_est = kept_est
        _sync_print(f"[{game_id_short}] TRIM #{trim_count} ({why}): ~{int(est_tokens)} tokens; dropped {dropped} oldest "
                    f"messages, ~{kept_est} tokens kept (level start, completed levels and code file pinned)")
        _kept_turns = sum(1 for m in conversation[1:] if m.get("role") == "user")
        ctx_first_turn = last_turn_in_conv - _kept_turns + 1
        _pinned_text = _content_text(conversation[0].get("content")) if conversation else ""
        replay.add_event(
            board=current_board, actions_taken=len(actions_taken), current_level=current_level,
            kind="compaction", title=f"Context trimmed (#{trim_count}, {why})", available_actions=action_labels,
            message_text=f"Trim #{trim_count} ({why}): ~{int(est_tokens)} tokens; dropped {dropped} oldest messages; "
                         f"~{kept_est} tokens kept; conversation now {len(conversation)} messages",
            harness_feedback=(f"Context trimmed (#{trim_count}, {why}): ~{int(est_tokens)} tokens -> ~{kept_est}. Dropped the "
                              f"{dropped} oldest messages (no summary). From now on the model sees: the system prompt and tools, "
                              "then the pinned message below (with the image of the current level's start board), then "
                              + (f"turns {ctx_first_turn}-{last_turn_in_conv}." if _kept_turns else "no earlier turns.")
                              + "\n\n[Pinned message]\n" + _pinned_text),
            pinned_text=_pinned_text,
            pinned_board=board_to_hex_rows(level_start_board),
            **_ext.turn_log_kwargs(current_board, remaining_actions_indicator_rect),
            trim={"n": trim_count, "why": why, "est_tokens": int(est_tokens), "kept_tokens": int(kept_est),
                  "dropped_messages": int(dropped), "kept_turns": [ctx_first_turn, last_turn_in_conv] if _kept_turns else None},
            notes=dict(notes), event_log=list(event_log[-80:]),
        )
        return dropped

    def _apply_level_restart(reset_obs: Any, *, manual: bool, where: str) -> str:
        """State updates after env.reset() (a game over or a RESET). Returns the message for the model."""
        nonlocal current_board, all_boards, current_level, action_specs, action_labels, level_action_start_index
        nonlocal compaction_level_start_board, pending_full_board_for_image, consecutive_no_change_action_turns
        nonlocal canonical_seen_action_index, seen_colors, attempt_no, auto_resets_this_level
        nonlocal _decision_resets
        _decision_resets += 1
        _board_before_restart = to_uint8_board(all_boards[-1]).copy() if all_boards else None   # game-over board / board before RESET
        _ended_attempt = attempt_no
        if not manual:
            auto_resets_this_level += 1     # the automatic reset after a game over is an env step on this level too
        current_board = observation_board(reset_obs).copy()
        all_boards = [current_board.copy()]           # states: the new attempt only
        current_level = get_current_level(env, obs=reset_obs)
        action_specs = action_specs_from_env(env)
        action_labels = [s.label for s in action_specs]
        _ext.ctx().available_actions = list(action_labels)
        _ended_actions = len(actions_taken) - level_action_start_index - (1 if manual else 0)
        event_log.append(f"Level {current_level}, attempt {attempt_no}: "
                         + (f"you RESET it after {_ended_actions} action(s)" if manual else f"GAME OVER after {_ended_actions} action(s)")
                         + f" (total actions so far: {len(actions_taken)})")
        level_action_start_index = len(actions_taken)
        attempt_no += 1
        attempt_starts.append(len(history))           # history keeps every attempt of this level
        compaction_level_start_board = current_board.copy()
        pending_full_board_for_image = current_board.copy()
        _ext.level_restarted(current_board, manual=manual)  # indicator rectangle kept across the reset
        _ext.reference_sync(current_board, current_level, attempt_no, len(actions_taken))
        _ext.ctx().references.last_report = {}
        consecutive_no_change_action_turns = 0
        canonical_seen_action_index = _rebuild_canonical_seen_action_index(
            all_boards, remaining_actions_indicator_rect, offset=level_action_start_index)
        if manual:
            header = (
                f"[LEVEL RESET] You reset level {current_level}: it restarted from its opening board with a full action budget "
                f"(attempt {attempt_no} on this level). The reset counted as one action, and the actions already spent on this "
                "level still count toward its score. Completed levels, your code file and the conversation are kept. "
                "states now covers the new attempt; history and the model check cover every attempt on this level.\n\n")
        else:
            header = _ext.reset_message(current_level)
        if MINIMAL_HARNESS and manual:
            header = (f"[LEVEL RESET] RESET executed. Level {current_level}, attempt {attempt_no}. "
                      "Completed levels and Python memory/helpers retained. states/actions_history start a new attempt; "
                      "history retains previous attempts on this level.\n\n")
        msg = header + build_level_start_message(
            board=current_board, action_labels=action_labels, current_level=current_level,
            previous_level=None, total_actions_taken=len(actions_taken),
        )
        _reset_feedback = _ext.restart_feedback(_board_before_restart, current_board,
                                               remaining_actions_indicator_rect, manual=manual)
        if _reset_feedback:
            msg += "\n\n" + _reset_feedback
        msg += "\n\n" + _full_board_color_inventory_text(current_board)
        try:   # v2.6: the layered tracker's reading of the restarted board
            _trk = _ext.tracker_restart_text(current_board, current_level, remaining_actions_indicator_rect)
            if _trk:
                msg += "\n\n" + _trk
        except Exception as _trk_exc:
            print(f"[layered tracker] restart lines failed: {_trk_exc!r}")
        if not MINIMAL_HARNESS:
            msg += "\n\n" + _notes_text()
        seen_colors |= set(_color_counts(current_board).keys())   # v2.9: colours seen before the restart stay known
        last_restart_msg[0] = msg
        # v2.5-log: written right after the turn event (the turn whose actions caused it)
        pending_events.append(dict(
            board=current_board.copy(), actions_taken=len(actions_taken), current_level=current_level,
            kind="level_reset" if manual else "game_over_reset",
            title=(f"RESET - restarting level {current_level} ({where})" if manual
                   else f"Game over - restarting level {current_level} ({where})"),
            available_actions=action_labels,
            message_text=(f"RESET ({where}). Restarting level {current_level}, attempt {attempt_no}." if manual
                          else f"Game over ({where}). Restarting level {current_level}, attempt {attempt_no}."),
            harness_feedback=("[Sent to the model as a level transition, with the results of the turn before this event]\n" + msg),
            restart={"level": current_level, "ended_attempt": _ended_attempt, "new_attempt": attempt_no,
                     "actions_in_ended_attempt": _ended_actions, "where": where, "manual": manual,
                     "total_actions": len(actions_taken),
                     "death_text": (None if manual else (_ext.ctx().death_text or None))},
            death_board=(board_to_hex_rows(_board_before_restart) if (not manual and _board_before_restart is not None) else None),
            before_board=(board_to_hex_rows(_board_before_restart) if (manual and _board_before_restart is not None) else None),
            remaining_actions_rects=([list(r) for r in norm_rects(remaining_actions_indicator_rect)] or None),
            remaining_actions_rect=(list(norm_rects(remaining_actions_indicator_rect)[0]) if norm_rects(remaining_actions_indicator_rect) else None),
            event_log=list(event_log[-80:]),
            **_ext.turn_log_kwargs(current_board, remaining_actions_indicator_rect),
        ))
        return msg

    def _apply_level_complete(prev_level: int) -> None:
        """Bookkeeping when a level is completed (current_board is already the new level's board)."""
        nonlocal action_specs, action_labels, compaction_level_start_board, level_action_start_index
        nonlocal level_total_start_index, all_boards, attempt_no, canonical_seen_action_index, auto_resets_this_level
        nonlocal remaining_actions_indicator_rect
        action_specs = action_specs_from_env(env)
        action_labels = [s.label for s in action_specs]
        _ext.ctx().available_actions = list(action_labels)
        # v2.10: colours of the new level's board count as seen; the ones never seen before are named in the level start
        _newc = sorted(set(_color_counts(current_board).keys()) - seen_colors)
        seen_colors.update(_newc)
        level_new_colors_text[0] = ""
        if _newc:
            level_new_colors_text[0] = ("Colour(s) never seen before in this game, present on this level's board:\n"
                                        + "\n".join("  " + _l for _l in _format_color_lines(board=current_board, colors=_newc)))
        # v2.10: what the model's next_state predicted for the step that actually won
        win_pred_text[0] = ""
        try:
            if not MINIMAL_HARNESS and history and code_file.has("next_state"):   # v2.11: run in the worker
                _wr = history[-1]
                _sa = np.stack(all_boards[:-1], axis=0) if len(all_boards) > 1 else np.stack(all_boards, axis=0)
                win_pred_text[0] = _ext.win_prediction(
                    code_file=code_file, before=_wr.before, action=_wr.action, states=_sa,
                    actions_history=actions_taken[level_action_start_index:-1], rect=remaining_actions_indicator_rect,
                    store=store, past_levels=past_levels, action_labels=action_labels) or ""
        except Exception as _wexc:
            win_pred_text[0] = ""
            print(f"[win prediction] failed: {_wexc!r}")
        end_board = all_boards[-2] if len(all_boards) >= 2 else all_boards[-1]
        winning = list(actions_taken[level_action_start_index:])
        # v2.5: the effect of the winning action on the old level = first animation frame vs the board before it
        _frames = _ext.ctx().last_frames
        _wb0 = to_uint8_board(end_board)
        # v2.9: the last frame of the winning action's own animation (was: the first frame that differed, often only the
        # step counter); stops before a level-transition wipe
        win_frame = _ext.win_result_frame(_wb0, _frames)
        win_effect_text = None
        if win_frame is None and _frames:
            win_effect_text = (f"The winning action {winning[-1] if winning else '?'} showed no change on level {prev_level} "
                               "before the level switched (the animation frames equal the board before it).")
        if win_frame is not None and win_frame.shape == to_uint8_board(end_board).shape:
            try:
                _wb = to_uint8_board(end_board)
                _wd, _lines = _ext._gl.change_report(_wb, win_frame, remaining_actions_indicator_rect,
                                                     exclude=_ext.ctx().view_exclude, cap=12)
                win_effect_text = (f"Effect of the winning action {winning[-1] if winning else '?'} on level {prev_level} "
                                   f"(board before it vs the last frame of its animation, before the level switched): "
                                   f"{_wd['changed_cells']} cell(s) changed"
                                   + (":\n  " + "\n  ".join(_lines) if _lines else " (no object-level change matched)"))
                if _wd["changed_cells"] and _wd["changed_cells"] <= 60:
                    win_effect_text += "\n  " + _ext.bounded_board_diff(_wb, win_frame).replace("\n", "\n  ")
            except Exception as exc:
                win_effect_text = f"(winning-action effect unavailable: {exc!r})"
        event_log.append(f"Level {prev_level}: COMPLETED on attempt {attempt_no} with {len(winning)} action(s) "
                         f"({len(actions_taken) - level_total_start_index} on the level in total); winning action: {winning[-1] if winning else '?'}")
        compaction_level_boards.append((prev_level, compaction_level_start_board, end_board.copy()))
        compaction_level_actions.append((prev_level, winning))
        if code_file.source:
            compaction_level_code.append((prev_level, code_file.source))
        past_levels.append({
            "level": prev_level,
            "start_board": to_uint8_board(compaction_level_start_board).copy(),
            "final_board": to_uint8_board(end_board).copy(),
            "winning_actions": winning,
            "win_frame": win_frame,
            "actions_total": len(actions_taken) - level_total_start_index,
            "attempts": attempt_no,
            "transitions": _hist_summary_list()[-300:],
        })
        last_win_effect[0] = win_effect_text
        last_win_image[0] = False
        if WIN_FRAME_IMAGE and win_frame is not None:   # v2.9: the winning moment as an image with the next level start
            _ext.record_win_frame(win_frame, prev_level)
            last_win_image[0] = True
        _scored_actions = len(actions_taken) - level_total_start_index + auto_resets_this_level
        level_actions_scored[prev_level] = _scored_actions
        auto_resets_this_level = 0
        _base = (_baselines[prev_level - 1] if _baselines and 0 < prev_level <= len(_baselines) else None)
        turn_level_completions.append({
            "level": prev_level, "new_level": current_level, "attempts": attempt_no,
            "actions_total": len(actions_taken) - level_total_start_index,
            "actions_scored": _scored_actions, "baseline": _base,
            "level_score": (round(min(115.0, 100.0 * (_base / _scored_actions) ** 2), 2) if _base and _scored_actions else None),
            "winning_attempt_actions": winning,
            "winning_action": winning[-1] if winning else None,
            "win_effect_text": win_effect_text,
            "win_frame": board_to_hex_rows(win_frame) if win_frame is not None else None,
            "board_before_winning_action": board_to_hex_rows(end_board),
            "level_start_board": board_to_hex_rows(compaction_level_start_board),
            "anim_frames": len(_frames),
        })
        _ext.level_advanced(compaction_level_start_board, current_board)
        if MINIMAL_HARNESS:
            remaining_actions_indicator_rect = None
            _ext.ctx().references.indicator_rects = []
        compaction_level_start_board = current_board.copy()
        level_action_start_index = len(actions_taken)
        level_total_start_index = len(actions_taken)
        history.clear()
        attempt_starts[:] = [0]
        attempt_no = 1
        all_boards = [current_board.copy()]
        _ext.reference_sync(current_board, current_level, attempt_no, len(actions_taken))
        _ext.ctx().references.last_report = {}
        canonical_seen_action_index = _rebuild_canonical_seen_action_index(
            all_boards, remaining_actions_indicator_rect, offset=level_action_start_index)

    def _note_win(won_level: int) -> None:
        """v2.5-log: score/replay bookkeeping for the level whose completion won the game (no _apply_level_complete)."""
        if won_level in level_actions_scored:
            return
        _n = len(actions_taken) - level_total_start_index + auto_resets_this_level
        level_actions_scored[won_level] = _n
        _end = to_uint8_board(all_boards[-2] if len(all_boards) >= 2 else all_boards[-1])
        try:
            _wf = _ext.win_result_frame(_end, list(_ext.ctx().last_frames))
        except Exception:
            _wf = None
        _base = (_baselines[won_level - 1] if _baselines and 0 < won_level <= len(_baselines) else None)
        _winning = list(actions_taken[level_action_start_index:])
        turn_level_completions.append({
            "level": won_level, "new_level": None, "won_game": True, "attempts": attempt_no,
            "actions_total": len(actions_taken) - level_total_start_index, "actions_scored": _n, "baseline": _base,
            "level_score": (round(min(115.0, 100.0 * (_base / _n) ** 2), 2) if _base and _n else None),
            "winning_attempt_actions": _winning, "winning_action": _winning[-1] if _winning else None,
            "win_effect_text": None, "win_frame": board_to_hex_rows(_wf) if _wf is not None else None,
            "board_before_winning_action": board_to_hex_rows(_end),
            "level_start_board": board_to_hex_rows(compaction_level_start_board),
            "anim_frames": len(_ext.ctx().last_frames),
        })

    def _manual_reset_refusal(n_lines: int) -> str | None:
        if n_lines > 1:
            return ("RESET refused: RESET must be sent on its own, not in a batch with other actions "
                    "(nothing in this batch was executed).")
        if len(actions_taken) - level_action_start_index == 0:
            return ("RESET refused: the level is already at its opening board (no action since the level start "
                    "or the last reset). Nothing was executed.")
        return None

    for turn in range(1, MAX_TURNS + 1):
        # Dynamic deadline from allocator (refreshed each turn)
        if allocator is not None:
            deadline = allocator.get_deadline(game_id)

        # v2.7 priority scheduler: after a context trim (or a long stretch) give the slot to a waiting game if one
        # has a higher priority; returns when this game holds a slot again or the run's deadline has passed.
        if _sched is not None:
            if current_level != _sched_level:
                _sched_level, _sched_tok0 = current_level, replay.token_output_tokens
            _why = _sched.should_yield(_skey, trim_count > _sched_trims_seen)
            _sched_trims_seen = trim_count
            if _why:
                _tok_lvl = replay.token_output_tokens - _sched_tok0
                _prio = _sched.priority(current_level, _win_levels, _tok_lvl)
                _sched.handover(_skey, _prio, _why,
                                info=f"level {current_level}/{_win_levels or '?'}, {_tok_lvl / 1000:.0f}k tokens on it")

        if current_level >= target_level:
            finalize_run(
                kind="success",
                title="Level complete!",
                status_line=f"Level complete! Reached target level {target_level}.",
                reasoning_text="",
                results_lines=["Level completed successfully."],
            )
            return

        if deadline and wall_time() >= deadline:
            finalize_run(
                kind="deadline",
                title="Deadline reached",
                status_line=f"Deadline reached after {len(actions_taken)} actions.",
                reasoning_text="",
                results_lines=["Deadline reached. Finalizing."],
            )
            return

        if MEMORY_DEBUG and not quiet:
            print(f"  [MEM] RSS={_mem_mb():.0f}MB | conv={len(conversation)} msgs, boards={len(all_boards)}, hist={len(history)}")

        if not quiet:
            print(f"\n{'#'*60}\nTURN {turn}/{MAX_TURNS} | Actions: {len(actions_taken)} | History: {len(history)} transitions")
            print_board("Current board", current_board)

        # Deadline check BEFORE starting inference (avoid initiating long LLM call past deadline)
        if deadline and wall_time() >= deadline:
            finalize_run(
                kind="deadline",
                title="Deadline reached (pre-inference)",
                status_line=f"Deadline reached before inference after {len(actions_taken)} actions.",
                reasoning_text="",
                results_lines=["Deadline reached before starting inference. Finalizing."],
            )
            return

        _use_estimate = not MINIMAL_HARNESS and SMART_ALLOCATION and allocator is not None
        _remaining = None
        if deadline:
            _remaining = deadline - wall_time()
            if _remaining <= 0:
                finalize_run(
                    kind="deadline",
                    title="Deadline reached (pre-inference)",
                    status_line=f"Deadline reached before inference after {len(actions_taken)} actions.",
                    reasoning_text="",
                    results_lines=["Deadline reached before starting inference. Finalizing."],
                )
                return

        # v2.8 lean no-action turn: nothing was executed since the last small board image was sent (same board, no
        # env step, no context trim since), so the image and the unchanged counters are replaced by a one-line note.
        _lean = bool(LEAN_NOACTION_TURNS and _img_last_board is not None and pending_full_board_for_image is None
                     and len(actions_taken) == _img_last_actions and trim_count == _img_last_trims
                     and _status_text_last and _status_text_last in feedback_message
                     and np.array_equal(current_board, _img_last_board))
        if _lean:
            feedback_message = feedback_message.replace(_status_text_last, _LEAN_NOTE, 1)

        # Live budget block: remaining minutes only during the last quarter of the game's time
        _elapsed_s = wall_time() - _game_start_wall
        _total_s = (_elapsed_s + _remaining) if _remaining else None
        budget_parts = []
        if _sched is None and _total_s and _total_s > 0 and _elapsed_s >= 0.75 * _total_s:   # v2.7: no per-game budget under the scheduler
            budget_parts.append(f"\n\n[Budget] Time remaining for this game: {max(0.0, _remaining) / 60:.0f} min")
        _lvl_total = len(actions_taken) - level_total_start_index + auto_resets_this_level   # v2.10: the automatic restart after a game over counts too
        _lvl_attempt = len(actions_taken) - level_action_start_index
        _status_part = (
            f"\n\n[Status] Level {current_level}, actions on this level: {_lvl_total}"
            + (f" (including {auto_resets_this_level} automatic restart(s) after game overs)" if auto_resets_this_level else "")
            + (f" ({_lvl_attempt} in the current attempt, attempt {attempt_no})" if attempt_no > 1 else "")
            + f" | Actions per response: up to {_ext.MAX_ACT_ACTIONS if MINIMAL_HARNESS else (MAX_ACTIONS_PER_MESSAGE or 4)}"
        )
        if not _lean:
            budget_parts.append(_status_part)
        feedback_message += "".join(budget_parts)

        # Pre-send trim: never send a prompt estimated above CONTEXT_PRESEND_TRIM_TOKENS.
        if CONTEXT_TRIM and conversation:
            _req_est = conv_tokens_est + len(feedback_message) / 3.3 + 3 * 600
            if _req_est > CONTEXT_PRESEND_TRIM_TOKENS:
                _do_trim(_req_est, CONTEXT_TRIM_TARGET_TOKENS, 2, "pre-send")
        if _lean and trim_count != _img_last_trims:   # the trim may have dropped the last board image: send it again
            _lean = False
            feedback_message = feedback_message.replace(_LEAN_NOTE, _status_text_last, 1) + _status_part

        # v2.9 act-now nudge: no action executed while ACT_NUDGE_TOKENS output tokens were generated (since the last
        # executed action or the last nudge) -> one short line at the end of the message
        _nudged = False
        if ACT_NUDGE:
            if len(actions_taken) != _nudge_actions_seen:
                _nudge_actions_seen, _nudge_tok_ref = len(actions_taken), replay.token_output_tokens
                _nudge_act_tok = replay.token_output_tokens
            elif replay.token_output_tokens - _nudge_tok_ref >= ACT_NUDGE_TOKENS:
                _since_k = max(1, round((replay.token_output_tokens - _nudge_act_tok) / 1000))
                feedback_message += (f"\n\n[Act now] About {_since_k}k tokens of reasoning since your last executed action. Your "
                                     "earlier results are still valid, don't restart your analysis. Act on them now: your best "
                                     "next action or short batch, or the cheapest test that settles your main open question.")
                _nudge_tok_ref = replay.token_output_tokens
                _nudged = True

        _decision_options = _decision.start_turn(_decision_identity())
        _reasoning_time = ({"remaining_seconds": _remaining, "budget_seconds": _reasoning_budget_s}
                           if TIERED_REASONING else {})
        _request_options = _reasoning.start_turn(_decision_identity(), _decision_options, MAX_OUTPUT_TOKENS,
                                                **_reasoning_time)
        if _reasoning.focus():
            feedback_message += "\n\n" + _reasoning.focus()
        _request_cap = _request_options.max_output_tokens if _request_options else MAX_OUTPUT_TOKENS
        if _decision_options is not None:
            feedback_message += "\n\n" + _decision.prompt()
            if _decision.phase == "proposal" and not MINIMAL_HARNESS:
                try:
                    _coverage = [line for line in _stuck_check_lines() if not line.startswith("- Your pinned notes")]
                    feedback_message += "\n[Control coverage; visible regions are approximate]\n" + "\n".join(_coverage)
                except Exception as _coverage_error:
                    _decision.event("coverage_unavailable", reason=str(_coverage_error)[:200])
            if not USE_TOOL_CALLING:
                feedback_message += '\nSubmit proposals as a closed ```propose_actions JSON fence with candidates and probe_omission_reason.'
        elif _decision.end_feedback():
            feedback_message += "\n\n" + _decision.end_feedback()
        if MINIMAL_HARNESS:
            # New observation only: never rewrite cached system/history messages.
            feedback_message += "\n\n" + _ext.ctx().action_coverage.text(
                action_labels, current_board, _ext.color_names(), norm_rects(remaining_actions_indicator_rect))
            from .minimal_indicator import status_line
            _indicator_line = status_line(_minimal_indicator_facts(), _ext.color_names())
            if _indicator_line:
                feedback_message += "\n\n" + _indicator_line
        _ext.prepare_request_images(current_board, pending_full_board_for_image is not None)
        _saved_extra_images = list(_ext.ctx().extra_images)
        _request_text = feedback_message
        _req_t0 = wall_time()
        turn_result = None
        _fail_started: float | None = None
        _n_fail = 0
        _n_ctx_errors = 0
        _n_400_unknown = 0
        while turn_result is None:
            if deadline:
                _remaining = deadline - wall_time()
                if _remaining <= 0:
                    finalize_run(
                        kind="deadline",
                        title="Deadline reached (pre-inference)",
                        status_line=f"Deadline reached before inference after {len(actions_taken)} actions.",
                        reasoning_text="",
                        results_lines=["Deadline reached before a request could be completed. Finalizing."],
                    )
                    return
            _ext.ctx().extra_images = list(_saved_extra_images)   # a retry sends the same images
            if _lean and trim_count != _img_last_trims and _LEAN_NOTE in _request_text:
                # v2.10: a harder trim during retries re-sends the board image: send the status lines with it again
                _request_text = _request_text.replace(_LEAN_NOTE, _status_text_last, 1) + _status_part
            # v2.8: no small image on a lean turn (unless a harder trim after a rejected request may have dropped it)
            _small_board = None if (_lean and trim_count == _img_last_trims) else current_board
            try:
                turn_result = request_llm_response(
                    client,
                    system_prompt,
                    conversation,
                    _request_text,
                    board_for_big_image=pending_full_board_for_image,
                    board_for_small_image=_small_board,
                    reuse_unchanged_board=bool(
                        MINIMAL_HARNESS and pending_full_board_for_image is None
                        and _img_last_board is not None and trim_count == _img_last_trims
                        and np.array_equal(current_board, _img_last_board)),
                    available_action_labels=action_labels,
                    quiet=quiet,
                    quiet_label=game_id_short,
                    include_estimate_tool=_use_estimate,
                    remaining_seconds=_remaining,
                    request_options=_request_options,
                    include_proposal_tool=_decision_options is not None and not MINIMAL_HARNESS,
                )
            except Exception as exc:
                if deadline and wall_time() >= deadline - 1:
                    finalize_run(
                        kind="deadline",
                        title="Deadline reached (during request)",
                        status_line=f"Deadline reached during a request after {len(actions_taken)} actions.",
                        reasoning_text="",
                        results_lines=[f"Deadline reached during a request ({exc!r}). Finalizing."],
                    )
                    return
                _ctx_err = _is_context_error(exc)
                if not _ctx_err and _is_bad_request(exc):
                    _n_400_unknown += 1
                    _ctx_err = _n_400_unknown >= 2   # v2.10: an unrecognised 400 is retried once unchanged first
                if _ctx_err:
                    # The server rejected the prompt itself: retrying it unchanged cannot work. Trim harder.
                    _n_ctx_errors += 1
                    _sync_print(f"[{game_id_short}] REQUEST REJECTED (#{_n_ctx_errors}; trimming harder, at most 3 times): {exc!r}"[:600])
                    if _n_ctx_errors > 3:
                        err_text = f"[LLM] Request rejected after 3 harder trims: {exc!r}"
                        finalize_run(kind="llm_error", title="LLM request rejected",
                                     status_line="LLM request rejected repeatedly. Ending run.",
                                     reasoning_text="", results_lines=[err_text])
                        return
                    _est_now = max(conv_tokens_est, CONTEXT_TRIM_TRIGGER_TOKENS)
                    if _n_ctx_errors == 1:
                        _do_trim(_est_now, 40000, 1, "rejected request")
                    elif _n_ctx_errors == 2:
                        _do_trim(_est_now, 20000, 1, "rejected request")
                    else:
                        _last_tool = ""
                        for _m in reversed(conversation):
                            if _m.get("role") == "tool" and isinstance(_m.get("content"), str) and _m["content"]:
                                _last_tool = _m["content"]
                                break
                        _do_trim(_est_now, 10000, 0, "rejected request, all turns dropped")
                        if _last_tool:
                            _request_text = ("[The context was full: all earlier turns were dropped. Results of your last turn:]\n"
                                             + _last_tool[:8000] + "\n\n" + feedback_message)
                    continue
                _n_fail += 1
                if _fail_started is None:
                    _fail_started = wall_time()
                _failing_for = wall_time() - _fail_started
                if _failing_for >= LLM_RETRY_MAX_SECONDS and not (_is_local_provider() and not server_ready()):
                    # v2.10: only a request that keeps failing while the server is up ends the game; a server outage
                    # is waited out (until the deadline) instead of ending every game that is in play
                    err_text = f"[LLM] Request failed for {_failing_for:.0f}s ({_n_fail} attempts): {exc!r}"
                    _sync_print(f"[{game_id_short}] ERROR: {err_text}")
                    finalize_run(kind="llm_error", title="LLM request failed",
                                 status_line="LLM request failed repeatedly. Ending run.",
                                 reasoning_text="", results_lines=[err_text])
                    return
                _delay = min(60.0, 5.0 * (2 ** min(_n_fail - 1, 4)))
                if deadline:
                    _delay = min(_delay, max(0.0, deadline - wall_time()))
                _sync_print(f"[{game_id_short}] REQUEST FAILED (attempt {_n_fail}, failing for {_failing_for:.0f}s); "
                            f"retrying in {_delay:.0f}s: {exc!r}"[:600])
                if _delay > 0:
                    import time as _time_mod
                    _time_mod.sleep(_delay)
                if _is_local_provider() and not server_ready():
                    _wait = LLM_RETRY_MAX_SECONDS - (wall_time() - _fail_started)
                    if deadline:
                        _wait = min(_wait, deadline - wall_time())
                    if _wait > 0:
                        wait_for_server(_wait, label=game_id_short)
                continue
        if _n_fail or _n_ctx_errors:
            _sync_print(f"[{game_id_short}] REQUEST OK after {_n_fail} failure(s) and {_n_ctx_errors} rejection(s)")
        # v2.5-log: what the model saw this turn (besides the system prompt, tools and the earlier turns still in context)
        _input_tool_text = ""
        for _m in reversed(conversation):
            if _m.get("role") == "tool":
                if isinstance(_m.get("content"), str) and _m["content"]:
                    _input_tool_text = _m["content"]
            else:
                break
        cur_turn_info = {
            "request_text": _content_text(turn_result.user_content) or _request_text,
            "input_tool_text": _input_tool_text or None,
            "usage": {"input_total": turn_result.usage.input_total_tokens, "input_cached": turn_result.usage.input_cached_tokens,
                      "input_miss": turn_result.usage.input_miss_tokens, "output": turn_result.usage.output_tokens,
                      "finish_reason": getattr(turn_result, "finish_reason", None),
                      "request_seconds": round(wall_time() - _req_t0, 1), "failed_attempts": _n_fail,
                      "rejected_attempts": _n_ctx_errors},
            "context": {"first_turn": (ctx_first_turn if last_turn_in_conv >= ctx_first_turn else None),
                        "last_turn": (last_turn_in_conv if last_turn_in_conv >= ctx_first_turn else None),
                        "pinned": bool(pinned_in_conversation), "trims": trim_count, "messages": len(conversation)},
            "wall_s": round(wall_time() - _game_start_wall, 1),
        }
        if _decision.settings.mode != "off":
            cur_turn_info["decision_policy"] = _decision.snapshot()
            cur_turn_info["request_options"] = (_dc_asdict(_request_options) if _request_options else None)
            cur_turn_info["effective_output_cap"] = _request_cap
            if _decision_options and USE_TOOL_CALLING and not MINIMAL_HARNESS:
                cur_turn_info["tool_definitions"] = [*_command_tool_specs(action_labels, include_estimate=_use_estimate), PROPOSAL_TOOL]
        cur_turn_info["request_kind"] = _reasoning.kind
        if MINIMAL_HARNESS:
            cur_turn_info["board_image_omitted"] = turn_result.board_image_omitted
            cur_turn_info["thinking_token_budget"] = thinking_token_budget(
                _request_cap, recovery=_decision_options is not None)
            cur_turn_info["thinking_budget_forced"] = (
                THINKING_END.removesuffix("</think>").strip() in (turn_result.reasoning_content or ""))
        if ADAPTIVE_REASONING:
            cur_turn_info["request_options"] = _dc_asdict(_request_options)
            cur_turn_info["effective_output_cap"] = _request_cap
            cur_turn_info["reasoning_control"] = _reasoning.snapshot()
        if _sched is not None:
            cur_turn_info["active_s"] = round(_sched.active_seconds(_skey), 1)   # v2.7: time this game held a slot
        if _small_board is not None:   # v2.8: remember what the last small board image showed
            _img_last_board, _img_last_actions, _img_last_trims = current_board.copy(), len(actions_taken), trim_count
        elif LEAN_NOACTION_TURNS:
            cur_turn_info["lean_turn"] = True
        if _nudged:
            cur_turn_info["act_nudge"] = True   # v2.9
        turn_level_completions.clear()
        turn_note_ops.clear()
        pending_full_board_for_image = None
        replay.add_token_usage(turn_result.usage)
        response_text = turn_result.text
        conversation.append({"role": "user", "content": turn_result.user_content_for_history})
        last_turn_in_conv = turn
        assistant_history = (turn_result.assistant_message_for_history if turn_result.assistant_message_for_history is not None
                             else {"role": "assistant", "content": turn_result.assistant_content_for_history})
        conversation.append(without_reasoning(assistant_history) if EPHEMERAL_REASONING else assistant_history)
        replay.llm_messages += 1

        if MEMORY_DEBUG and not quiet:
            cache_str = "hit" if turn_result.cache_hit else "MISS"
            print(f"  [MEM] post-LLM RSS={_mem_mb():.0f}MB | cache={cache_str}")

        if quiet:
            ctx_k = turn_result.usage.input_total_tokens // 1000
            _sync_print(
                f"[{game_id_short}] Message #{replay.llm_messages} sent. "
                f"Level={max(0, current_level - replay.initial_level)}, "
                f"Actions={len(actions_taken)}, Context={ctx_k}k"
            )

        # --- Track whether compaction is needed (processed before next LLM call) ---
        total_tokens = turn_result.usage.input_total_tokens + turn_result.usage.output_tokens
        if EPHEMERAL_REASONING:
            # Next-turn context excludes disposable output. Actual generation accounting/caps remain unchanged.
            retained_output_est = int(len(json.dumps(conversation[-1], ensure_ascii=False)) / 3.3) + 1
            total_tokens = turn_result.usage.input_total_tokens + retained_output_est
            cur_turn_info["retained_context_tokens_estimate"] = total_tokens
            cur_turn_info["discarded_reasoning_chars"] = len(turn_result.reasoning_content or "")
            cur_turn_info["reasoning_history_retained"] = False
        conv_tokens_est = total_tokens
        needs_compaction = (compaction_limit > 0 and total_tokens >= compaction_limit)

        was_empty_response = (response_text == "Error: no text output")
        reasoning, commands = parse_response_commands(response_text, tool_calls=turn_result.tool_calls)
        try:
            from .llm import last_dropped_calls as _ldc
            _dropped_calls = _ldc()
        except Exception:
            _dropped_calls = []
        cur_turn_extra = {   # for a finalize in the middle of this turn (game won, reset failed, ...)
            "turn": turn,
            "llm_commands": [{"kind": c.kind, "content": c.content} for c in commands] if commands else None,
            "llm_reasoning_content": turn_result.reasoning_content or None,
            "assistant_text": turn_result.text or None,
            "images": ([{"label": img.label, "data_url": img.data_url, "scale": img.scale}
                        for img in turn_result.image_attachments] if turn_result.image_attachments else None),
        }
        if not quiet:
            if reasoning and not was_empty_response:
                print(f"\n{colorize('[Model reasoning]', Color.MAGENTA, bold=True)}")
                print(reasoning[:800] + ("..." if len(reasoning) > 800 else ""))
            elif was_empty_response:
                log_tag("LLM", "Empty response from provider this turn", Color.YELLOW)
            else:
                log_tag("FORMAT", "Missing plain-text summary in model response", Color.YELLOW)

            print(f"Commands parsed: {[c.kind for c in commands]}")

        _ext.begin_turn()
        _hist_len_before = len(history)
        results: list[str] = []
        if MINIMAL_HARNESS:
            rejected = [c.kind for c in commands if c.kind != "run"]
            commands = [c for c in commands if c.kind == "run"]
            if rejected:
                results.append("[Not executed] Only run(code) is available: " + ", ".join(rejected))
            _ext.ctx().memory_events = []
        level_transition_messages: list[str] = []
        code_changed = False
        validation_passed = True
        path_next_state_fn = None
        board_to_symbolic_fn = None
        symbolic_to_board_fn = None
        prev_code = replay.latest_code
        turn_click_cells: list[dict[str, Any]] = []
        remaining_rect_modified = False
        _decision_evidence = []
        _minimal_proposals = []
        _minimal_analysis_failed = False
        _forced_candidate = None

        hit_length_cap = (getattr(turn_result, "finish_reason", None) == "length")
        for _dc in _dropped_calls:
            results.append(f"[Not executed] {_dc}")
        _cut_call_note = ""
        try:   # v2.10: vLLM reports a reply cut inside a tool call as finished; the last (cut) call is dropped
            _out_tok = int(getattr(getattr(turn_result, "usage", None), "output_tokens", 0) or 0)
        except Exception:
            _out_tok = 0
        _decision.record_response(_decision_options, _out_tok)
        _reasoning.record_response(_out_tok)
        _reported_commands = list(commands)
        _think_cmds = [c for c in commands if c.kind == "think"]
        if not hit_length_cap and _out_tok >= _request_cap - 2 and commands:
            hit_length_cap = True
            _dropped_cmd = commands.pop()
            _cut_call_note = f"; its last tool call ({_dropped_cmd.kind}) was cut off and NOT executed"
            if commands:
                _cut_call_note += "; the calls before it were executed: " + ", ".join(c.kind for c in commands)
            _cut_call_note += "."
        if _decision.settings.mode != "off" and not commands and _out_tok >= _request_cap - 2:
            hit_length_cap = True   # some vLLM parsers report stop for a reasoning-only cutoff
        if hit_length_cap:
            results.append(
                f"[Length cap] Your previous response reached the {_request_cap:,}-token output cap (thinking included) "
                "and was cut off"
                + (_cut_call_note if _cut_call_note else ("; none of its tool calls were executed." if not turn_result.tool_calls else "."))
                + " Do not resume the long analysis: run the cheapest discriminating test now, or write the remaining "
                "analysis or search as code in run."
            )
        elif was_empty_response:
            results.append(
                "[Empty response] The API returned no content for your previous turn. "
                f"This is a wasted turn. You MUST include {command_mode_desc} in your response."
            )

        # think is an exclusive control turn: never mix escalation with execution.
        if _think_cmds:
            results.append(_reasoning.request(
                _think_cmds[0].content,
                exclusive=(len(commands) == 1 and len(turn_result.tool_calls) <= 1 and not _dropped_calls),
                capped=hit_length_cap, recovery=_decision_options is not None,
                can_continue=(turn < MAX_TURNS and (deadline is None or wall_time() < deadline))))
            commands = []
        if ADAPTIVE_REASONING:
            cur_turn_info["reasoning_control"] = _reasoning.snapshot()

        # Phase 1: Code changes (py/patch)
        code_cmds = [c for c in commands if c.kind in ("py", "patch")]
        run_cmds = [c for c in commands if c.kind == "run"]
        remaining_actions_indicator_cmds = [
            c for c in commands if c.kind == "remaining_actions_indicator"
        ]
        env_cmds = [c for c in commands if c.kind in ("actions", "path")]
        estimate_cmds = [c for c in commands if c.kind == "estimate_progress"]
        note_cmds = [c for c in commands if c.kind == "note"]
        view_cmds = [c for c in commands if c.kind == "object_view"]

        # Extract AI estimate and report to allocator
        if allocator is not None:
            est_pct, est_turns = parse_estimate_from_commands(commands)
            levels_done = max(0, current_level - replay.initial_level)
            allocator.report_progress(
                game_id,
                current_level=current_level,
                levels_completed=levels_done,
                messages=replay.llm_messages,
                actions_taken=len(actions_taken),
                finish_percent=est_pct,
                median_turns=est_turns,
            )

        cmd_summary_parts: list[str] = []
        for cmd in _reported_commands:
            if cmd.kind == "py":
                cmd_summary_parts.append("py: Full code rewrite")
            elif cmd.kind == "patch":
                cmd_summary_parts.append("patch: Code patch")
            elif cmd.kind == "run":
                cmd_summary_parts.append(f"run: {cmd.content.strip().replace(chr(10), ' ')}")
            elif cmd.kind == "actions":
                cmd_summary_parts.append(f"actions: {cmd.content.strip().replace(chr(10), ' ; ')}")
            elif cmd.kind == "path":
                cmd_summary_parts.append(f"path: {cmd.content.strip().replace(chr(10), ' ')}")
            elif cmd.kind == "remaining_actions_indicator":
                cmd_summary_parts.append(f"remaining_actions_indicator: {cmd.content.strip()}")
            elif cmd.kind == "note":
                cmd_summary_parts.append(f"note: {cmd.content.strip()[:200]}")
            elif cmd.kind == "estimate_progress":
                cmd_summary_parts.append(f"estimate_progress: {cmd.content.strip()}")
            elif cmd.kind == "object_view":
                cmd_summary_parts.append(f"set_object_view: exclude {cmd.content.strip()}")
            elif cmd.kind == "think":
                cmd_summary_parts.append(f"think: {cmd.content.strip()}")
        commands_summary = "\n".join(cmd_summary_parts)

        # Save durable memory before any run/action can finalize the game. Exclusive
        # think turns and dropped/truncated calls have already been filtered above.
        if confidence_notes is not None:
            for cmd in note_cmds:
                operation, feedback = confidence_notes.apply(cmd.content)
                turn_note_ops.append(operation)
                results.append(feedback)
                notes.clear()
                notes.update(confidence_notes.display(confidence_notes.entries))
            note_cmds = []

        for cmd in code_cmds:
            if cmd.kind == "py":
                if not quiet:
                    log_tag("CODE", "Full rewrite", Color.CYAN, bold=True)
                err = code_file.update_full(cmd.content)
                if err:
                    results.append(f"[Code rewrite] Error: {err}")
                    if not quiet:
                        log_tag("CODE", f"Error: {err}", Color.RED)
                else:
                    code_changed = True
                    replay.latest_code = code_file.source
                    results.append("[Code rewrite] Success.")
                    if not quiet:
                        log_tag("CODE", "Compiled successfully", Color.GREEN)
            else:
                if not quiet:
                    log_tag("CODE", "Applying patch", Color.CYAN, bold=True)
                err = code_file.apply_patch(cmd.content)
                if err:
                    results.append(f"[Patch] Error: {err}")
                    if not quiet:
                        log_tag("CODE", f"Patch error: {err}", Color.RED)
                else:
                    code_changed = True
                    replay.latest_code = code_file.source
                    results.append("[Patch] Applied successfully.")
                    if not quiet:
                        log_tag("CODE", "Patch applied", Color.GREEN)

        # Phase 1b (v2.11): object view
        for cmd in view_cmds:
            try:
                _ex = sorted({int(x) for x in json.loads(cmd.content) if 0 <= int(x) <= 15})
            except Exception:
                results.append("[Object view] Error: exclude must be a list of colour values 0-15 ([] = default view).")
                continue
            _ext.ctx().view_exclude = _ex or None
            results.append("[Object view] " + (
                f"Colours {_ex} now count as gaps: touching cells of all other colours are reported as one component (change "
                "lines, inventory, model check, lib.frame_diff's default). lib.extract_components() is unchanged (pass exclude=...)."
                if _ex else "Back to single-colour components in the reports."))

        _ext.reference_sync(current_board, current_level, attempt_no, len(actions_taken))

        # Phase 2: the model check (v2.11: next_state runs in the worker; the harness only compares)
        def _model_check(_report: bool, _force: bool = False) -> tuple[bool, str, bool, list[str]]:
            if MINIMAL_HARNESS:
                return False, "", True, []
            _sa = np.stack(all_boards, axis=0) if all_boards else np.zeros((0, *current_board.shape[:2]), dtype=np.uint8)
            _ah = actions_taken[level_action_start_index:]
            try:
                _mc = _ext.model_check(code_file=code_file, history_recs=history, hist_summary=_hist_summary_list(),
                                       states=_sa, actions_history=_ah, rect=remaining_actions_indicator_rect, store=store,
                                       past_levels=past_levels, level=current_level, report=_report,
                                       store_version=_ext.ctx().store_version, action_labels=action_labels,
                                       force=_force)
            except Exception as _mexc:
                _mc = {"has_model": code_file.has("next_state"), "ok": False, "lines": [f"[Model check] failed: {_mexc!r}"]}
            if not quiet and _mc.get("has_model"):
                _v = _ext.ctx().validation or {}
                log_tag("VALIDATE", f"{_v.get('passed')}/{_v.get('total')} match", Color.GREEN if _mc.get("ok") else Color.YELLOW)
            return bool(_mc.get("has_model")), "", bool(_mc.get("ok")), list(_mc.get("lines") or [])

        current_symbolic = None
        next_state_fn = True if code_file.has("next_state") else None
        board_to_symbolic_fn = True if code_file.has("board_to_symbolic") else None
        symbolic_to_board_fn = True if code_file.has("symbolic_to_board") else None
        path_next_state_fn, _model_mode, validation_passed, _mc_msgs = _model_check(code_changed)
        results.extend(_mc_msgs)
        _code_changed_in_run = False
        report_validation_next = False

        # Phase 3: analysis via run, including act() from inside the code
        _act_feedback: list[str] = []

        def _indicator_request(method, value=None):
            nonlocal remaining_actions_indicator_rect, remaining_rect_modified, canonical_seen_action_index
            if method == "facts":
                return _minimal_indicator_facts()
            if method != "mark":
                raise ValueError("unknown indicator request")
            c = _ext.ctx()
            if c.turn_latch or c.pending_terminal:
                raise ValueError("mark the region after observing the next board")
            from .minimal_indicator import validate_rects
            rects = validate_rects(value, current_board.shape)
            if rects != tuple(norm_rects(remaining_actions_indicator_rect)):
                remaining_actions_indicator_rect = list(rects) or None
                remaining_rect_modified = True
                c.references.indicator_rects = list(rects)
                c.known_noop.clear()
                c.last_refused = c.snippet_stale = None
                canonical_seen_action_index = _rebuild_canonical_seen_action_index(
                    all_boards, remaining_actions_indicator_rect, offset=level_action_start_index)
                message = (f"[Marked resource/counter region] {list(rects)}. Raw boards remain unchanged; "
                           "diffs, component boundaries, board_key and no-op checks now use this selection."
                           if rects else "[Marked resource/counter region] Cleared; all cells count as game content again.")
                area = sum((r1-r0)*(c1-c0) for r0,r1,c0,c1 in rects)
                if area > current_board.size * .05:
                    message += " Warning: this covers over 5% of the board; check that no playable objects are included."
                results.append(message)
            return list(rects)

        def _act_execute(_acts, _force, _expect=None):
            nonlocal current_board, current_level
            _c = _ext.ctx()
            try:
                _checks = _ext.reference_expect(_expect)
            except Exception as exc:
                return _ext.act_refusal("invalid expectations: " + str(exc))
            _expect_failed = False
            if _c.turn_latch:
                return _ext.act_refusal(_c.turn_latch)
            if ACT_GUARD_MODE == "stale" and _c.snippet_stale and not _force:
                return _ext.act_refusal(_c.snippet_stale)
            if _c.new_color_latch and not _force:   # v2.9: a new colour appeared earlier in this run call
                return _ext.act_refusal(_c.new_color_latch)
            if _c.turn_actions_executed >= _ext.MAX_ACT_ACTIONS:
                return _ext.act_refusal(f"action cap reached: at most {_ext.MAX_ACT_ACTIONS} actions per run call")
            _steps, _boards = [], []
            _stopped, _reason, _terminal, _outcome, _latch = False, None, False, None, None
            if any(_ext.normalize_label((_a.split() or [""])[0]) == RESET_LABEL for _a in _acts):
                if _checks:
                    return _ext.act_refusal("expectations cannot span RESET; inspect the new attempt next turn")
                _refusal = _manual_reset_refusal(len(_acts))
                if _refusal is not None:
                    _act_feedback.append(f"  {_refusal}")
                    return _ext.act_refusal(_refusal)
                _c.turn_actions_executed += 1
                _latch = ("RESET accepted: this run ends immediately; inspect the restarted board next turn")
                _c.turn_latch = _latch
                _c.pending_terminal = ("reset", current_level, current_level)
                _ext.log_guard("manual_reset", where="act")
                _act_feedback.append("  RESET accepted: remaining code in this run is skipped (including except/finally blocks). "
                                     "Work saved before RESET is retained; inspect the restarted board next turn.")
                _steps.append({"action": RESET_LABEL, "outcome": "reset", "changed_cells": None,
                               "changed_outside_indicator": None, "text": "RESET accepted"})
                return _ext.act_response(_steps, _boards, stopped_early=False, stop_reason=None,
                                         terminal=True, outcome="reset", latch=_latch, message=None)
            for _idx, _text in enumerate(_acts):
                if _c.turn_actions_executed >= _ext.MAX_ACT_ACTIONS:
                    _stopped, _reason = True, f"action cap ({_ext.MAX_ACT_ACTIONS} per run call) reached"
                    break
                _ca, _perr = _ext.parse_action_line(_text, action_specs, current_board)
                if _ca is None:
                    # v2.4: a rejected / invalid action ends the whole run call (state may now differ from the plan)
                    _abort = f'"{_text}" was rejected ({_perr}); no further actions were executed in this run call'
                    _act_feedback.append(f"  [act] STOPPED: {_abort}.")
                    _ext.log_guard("act_rejected_action", action=_text, reason=_perr)
                    return _ext.act_response(_steps, _boards, stopped_early=True, stop_reason=f"rejected action {_text!r}",
                                             terminal=False, outcome=None, latch=None, message=None, abort=_abort)
                if ACT_GUARD_MODE == "known":
                    _kg = _ext.known_guard(current_board, remaining_actions_indicator_rect, _ca.as_text(), repeat=_force)
                    if _kg is not None:
                        _stopped, _reason = True, _kg
                        _act_feedback.append(f"  {_ca.as_text()}: refused ({_kg})")
                        _ext.log_guard("known_refused", action=_ca.as_text(), where="act")
                        break
                if _ca.row >= 0:
                    turn_click_cells.append({"row": _ca.row, "col": _ca.col, "value": _ca.value, "accepted": True})
                _board_before_action = current_board
                _colors_before = set(_color_counts(current_board).keys()) | seen_colors
                _c.references.expected = _checks
                _rt, _oc, _lvl = _execute_action(
                    env, _ca, all_boards, actions_taken, history, current_level, seen_colors,
                    remaining_actions_indicator_rect, canonical_seen_action_index, quiet=quiet,
                )
                current_board = all_boards[-1]
                _c.turn_actions_executed += 1
                _last = _c.last_step or {}
                _ext.record_known(_board_before_action, remaining_actions_indicator_rect, _ca.as_text(),
                                  noop=(_oc == "continue" and _lvl == current_level and _ext.last_step_is_noop()),
                                  death=(_oc == "game_over"))
                _new_colors = sorted(set(_color_counts(current_board).keys()) - _colors_before)
                _steps.append({
                    "action": _ca.as_text(), "outcome": (None if _oc == "continue" else _oc),
                    "changed_cells": _last.get("changed_cells"),
                    "changed_outside_indicator": _last.get("changed_outside_indicator"),
                    "text": _rt,
                })
                if MINIMAL_HARNESS:
                    # Missing observations must never alias a previous animation.
                    _steps[-1]["animation_index"] = _last.get("animation_index")
                _act_feedback.append(_ext.step_entry(_rt, _ca.as_text()))   # v2.11: full text + one-line form
                _boards.append(current_board.copy())
                _prev_level = current_level
                if _oc in ("win", "game_over") or _lvl != _prev_level:
                    _terminal = True
                    _outcome = _oc if _oc != "continue" else "level_complete"
                    _latch = (f"the level ended ({_outcome}): no further actions in this turn; "
                              "the board you were holding is gone, re-read the new one next turn")
                    _c.turn_latch = _latch
                    _c.pending_terminal = (_outcome, _prev_level, _lvl)
                    current_level = _lvl
                    _ext.log_guard("terminal_latch", outcome=_outcome, where="act")
                    break
                if (_c.last_step or {}).get("outcome") == "env_error":   # v2.9: the environment returned an error
                    _stopped, _reason = True, f"the environment returned an error for {_ca.as_text()} (see the feedback)"
                    break
                if _c.references.stop_reason():
                    _stopped, _reason = True, _c.references.stop_reason()
                    _expect_failed = True
                    break
                if _new_colors:   # v2.9: later act() calls in this run call are refused too (unless repeat=True)
                    _c.new_color_latch = (f"a colour never seen before in this game ({', '.join(map(str, _new_colors))}) appeared "
                                          f"after {_ca.as_text()} earlier in this run call: look at that state before acting "
                                          "further; continue in your next run call")
                    if _idx == len(_acts) - 1:
                        _act_feedback.append(f"  New colour(s) {', '.join(map(str, _new_colors))} (never seen before in this game) "
                                             f"appeared after {_ca.as_text()}; further act() calls in this run call are refused; "
                                             "continue in your next run call.")
                        _ext.log_guard("new_color_latch", action=_ca.as_text(), colors=_new_colors, where="act")
                if _new_colors and _idx < len(_acts) - 1:
                    # v2.5: a colour never seen before in this game appeared: stop so the model can look at this state
                    _stopped = True
                    _reason = (f"a colour never seen before in this game appeared after {_ca.as_text()} "
                               f"(value(s) {', '.join(map(str, _new_colors))}); the remaining {len(_acts) - _idx - 1} action(s) "
                               "were NOT executed so you can inspect this state first")
                    _act_feedback.append(f"  Batch stopped after {_ca.as_text()}: new colour(s) {', '.join(map(str, _new_colors))} appeared; "
                                         f"{len(_acts) - _idx - 1} remaining action(s) were NOT executed.")
                    _ext.log_guard("batch_new_color", action=_ca.as_text(), colors=_new_colors, where="act")
                    break
                if _ext.last_step_is_noop():
                    _stopped = True
                    _reason = _ext.unchanged_board_text(_ca.as_text())
                    _c.snippet_stale = (f"batch paused: {_reason}; a further act() in this run call "
                                        "needs repeat=True (deliberate probe only)")
                    _skipped = len(_acts) - _idx - 1
                    if _skipped:
                        _act_feedback.append(f"  Batch stopped: {_reason}; {_skipped} remaining action(s) were NOT executed.")
                    _ext.log_guard("batch_noop", action=_ca.as_text(), skipped=_skipped, where="act")
                    break
            _resp = _ext.act_response(_steps, _boards, stopped_early=_stopped, stop_reason=_reason,
                                      terminal=_terminal, outcome=_outcome, latch=_latch, message=None)
            # v2.10: a stop that skipped planned actions (or a refusal) ends the run call; a single action that changed
            # nothing does not (probe loops of one-action act() calls keep working)
            if _stopped and not _terminal and _reason and (len(_acts) > len(_steps) or _expect_failed):
                _resp["abort"] = f"batch stopped: {_reason}"
            return _resp

        _board_before_run = current_board.copy()
        for cmd in run_cmds:
            if (_ext.ctx().pending_terminal or (None,))[0] == "reset":
                results.append("[Run] Skipped: an earlier run accepted RESET. Inspect the restarted board next turn.")
                continue
            states_arr = np.stack(all_boards, axis=0) if all_boards else np.zeros((0, *current_board.shape[:2]), dtype=np.uint8)
            _n_before = len(_act_feedback)
            _ledger_before = len(_ext.ctx().action_ledger)
            _shown_before = len(_ext.ctx().shown_images)
            output, err, _run_info = _ext.run_code(
                code_file, cmd.content, states_arr, actions_taken[level_action_start_index:],
                remaining_actions_indicator_rect, _act_execute, _hist_summary_list(),
                past_levels=past_levels, current_attempt=attempt_no, store=store, action_labels=action_labels,
                indicator_callback=(_indicator_request if MINIMAL_HARNESS else None),
            )
            display_output = output
            if MINIMAL_HARNESS:
                _minimal_analysis_failed |= bool(err)
                if _run_info.get("proposals"):
                    if _decision_options is None:
                        results.append("[Decision recovery] Proposal ignored: recovery is not active. Use act to act now.")
                    else:
                        _minimal_proposals.extend(_run_info["proposals"])
            if output.strip():  # data printed before an error can still contradict a proposal
                _decision_evidence.append(output)
            if len(display_output) > MAX_RUN_OUTPUT_CHARS:   # v2.10: keep the start and the end
                _h = int(MAX_RUN_OUTPUT_CHARS * 0.65)
                _t = MAX_RUN_OUTPUT_CHARS - _h
                display_output = (display_output[:_h]
                                  + f"\n...[{len(display_output) - _h - _t} characters of output omitted]...\n"
                                  + display_output[-_t:])
            if err:
                results.append(f"[Run] Error: {err}\nOutput before error:\n{display_output}")
            else:
                results.append(f"[Run] Output:\n{display_output}" if output.strip() else "[Run] (no output)")
            if MINIMAL_HARNESS:
                _ledger = _ext.ctx().action_ledger[_ledger_before:]
                if _ledger:
                    results.append("[Actions issued by run — authoritative parent record]\n"
                                   + "\n".join(json.dumps(s, ensure_ascii=False) for s in _ledger))
                _shown = _ext.ctx().shown_images[_shown_before:]
                if _shown:
                    # Parent receipt survives missing worker output. Image contents count as new evidence.
                    import hashlib as _shown_hash
                    _decision_evidence.append("show:" + _shown_hash.sha256(
                        json.dumps(_shown, sort_keys=True).encode()).hexdigest())
                    results.append("[Images shown]\n" + "\n".join(i["label"] for i in _shown))
                _guard_lines = [s for s in _act_feedback[_n_before:] if isinstance(s, str)]
                if _guard_lines:
                    results.append("[Action stops]\n" + "\n".join(_guard_lines))
            elif len(_act_feedback) > _n_before:   # v2.11: compressed to one line per step when long
                results.append("[Actions executed by run via act()]\n" + _ext.format_step_reports(_act_feedback[_n_before:])[0])
            if MINIMAL_HARNESS:
                from .minimal_memory import delta
                _mem = _ext.ctx().memory
                _new = _run_info.get("memory")
                if _new is not None:
                    _note_changes = delta(_mem["notes"], _new["notes"], "notes")
                    _ext.ctx().memory_events.append({
                        "shown_images": list(_shown),
                        "notes_changed_paths": [p for p, _ in _note_changes]})
                    _mem.clear()
                    _mem.update(_new)
                else:
                    results.append("[Memory] This worker did not return saved notes; previous values retained. "
                                   "Any actions reported above remain executed.")
                results.extend("[Memory] " + m for m in _run_info.get("memory_messages", []))
            if (err or "") and "timed out" in (err or ""):   # v2.10: functions of a timed-out run are not kept
                _kept, _kerr = [], None
                results.append("[Code file] Functions defined in this run were NOT kept because it timed out.")
            elif not (err or "").startswith("Safety check failed"):
                _defined_functions = _run_info.get("defined_functions")
                if (_ext.ctx().pending_terminal or (None,))[0] == "reset":
                    # Missing worker output is not evidence that later defs ran.
                    _defined_functions = _defined_functions or {}
                _kept, _kerr = _ext.keep_run_functions(code_file, cmd.content,
                                                     defined_functions=_defined_functions)
                if _kept:
                    code_changed = True
                    _code_changed_in_run = True
                    replay.latest_code = code_file.source
                    results.append(f"[Code file] Kept from this run (top-level functions; a later definition with the same "
                                   f"name replaces them): {', '.join(_kept)}")
                if _kerr:
                    results.append(f"[Code file] Top-level functions of this run {_kerr}")
            if _run_info.get("store") is not None:
                try:
                    import pickle as _pk
                    _st_changed = _pk.dumps(dict(store), protocol=4) != _pk.dumps(dict(_run_info["store"]), protocol=4)
                except Exception:
                    _st_changed = True
                store.clear()
                store.update(_run_info["store"])
                if _st_changed:
                    _ext.ctx().store_version += 1
            if _run_info.get("store_note"):
                results.append(f"[store] {_run_info['store_note']}")
            _ext.reference_sync(current_board, current_level, attempt_no, len(actions_taken))
            if not quiet:
                log_tag("RUN", display_output[:300] if display_output else "(no output)", Color.YELLOW)

        _pending = _ext.ctx().pending_terminal
        if _pending is not None:
            _t_outcome, _t_prev_level, _t_level_after = _pending
            _ext.ctx().pending_terminal = None
            if _t_outcome == "win":
                _note_win(_t_prev_level)
                results.append("[Actions via run] GAME WON!")
                finalize_run(
                    kind="success", title="Game won!", status_line="Game won!",
                    reasoning_text=reasoning, results_lines=results,
                )
                return
            if _t_outcome == "reset":
                reset_obs = _env_reset_retry(env)
                if reset_obs is None:
                    results.append("[RESET] Failed: the environment returned no observation; the level was not reset.")
                else:
                    actions_taken.append(RESET_LABEL)
                    level_transition_messages.append(_apply_level_restart(reset_obs, manual=True, where="act"))
            elif _t_outcome == "game_over":
                results.append("[Actions via run] GAME OVER!")
                if RESET_ON_GAME_OVER:
                    results.append("Level will restart. Previously completed levels are kept.")
                    _ext.record_death(all_boards[-1], compaction_level_start_board, remaining_actions_indicator_rect)
                    reset_obs = _env_reset_retry(env)
                    if reset_obs is None:
                        finalize_run(
                            kind="game_over", title="Game over (reset failed)", status_line="Game over. Reset failed.",
                            reasoning_text=reasoning, results_lines=results,
                        )
                        return
                    level_transition_messages.append(_apply_level_restart(reset_obs, manual=False, where="act"))
                else:
                    finalize_run(
                        kind="game_over", title="Game over", status_line="Game over. Ending run.",
                        reasoning_text=reasoning, results_lines=results,
                    )
                    raise SystemExit(1)
            elif _t_level_after > _t_prev_level:
                _apply_level_complete(_t_prev_level)
                if _t_level_after >= target_level:
                    results.append("[Actions via run] LEVEL COMPLETE!")
                    finalize_run(
                        kind="success", title="Level complete!",
                        status_line=f"Level complete! Reached target level {target_level}.",
                        reasoning_text=reasoning, results_lines=results,
                    )
                    return
                results.append(
                    f"[Actions via run] Level {_t_prev_level} completed successfully. Now on level {_t_level_after}. "
                    "No further actions were executed this turn so you can analyze the new level."
                )
                level_transition_messages.append(
                    build_level_start_message(
                        current_level=_t_level_after,
                        previous_level=_t_prev_level,
                        board=current_board,
                        action_labels=action_labels,
                        total_actions_taken=len(actions_taken),
                    )
                    + "\n\n"
                    + _full_board_color_inventory_text(current_board)
                    + "\n\n"
                    + _ext.level_start_objects_text(current_board, remaining_actions_indicator_rect)
                    + _level_complete_extra()
                )

        # Phase 3a (v2.5): pinned notes
        for cmd in note_cmds:
            try:
                _nd = json.loads(cmd.content)
                _nk, _nt = str(_nd.get("key", "")).strip(), str(_nd.get("text", "")).strip()
                if not _nt.strip("'\"` "):   # v2.9: '' or "" written as the text means delete
                    _nt = ""
            except Exception:
                results.append("[Notes] Error: could not parse the note (expected key and text).")
                continue
            if not _nk:
                results.append("[Notes] Error: a note needs a non-empty key.")
                continue
            if _nt and EVIDENCE_NOTES:   # v2.12b: status + the model's evidence; cited history[i] shown with the recorded action
                _ns = str(_nd.get("status") or "").strip().lower()
                _ns = _ns if _ns in ("seen", "inferred", "assumed") else ""
                _ne = str(_nd.get("evidence") or "").strip()
                _nc = str(_nd.get("certainty") or "").strip()
                _cites = []
                for _ci in list(dict.fromkeys(int(x) for x in re.findall(r"history\[(\d+)\]", _ne)))[:8]:
                    if 0 <= _ci < len(history):
                        _ca_ = history[_ci].action
                        _on = ""
                        try:
                            if _ca_.row >= 0:
                                _bv = int(to_uint8_board(history[_ci].before)[_ca_.row, _ca_.col])
                                _on = f" (on {_ext.color_names().get(_bv, _bv)})"
                        except Exception:
                            _on = ""
                        _cites.append(f"history[{_ci}] = {_ca_.as_text()}{_on}")
                    else:
                        _cites.append(f"history[{_ci}] does not exist on this level ({len(history)} transitions)")
                _plain = _nt
                _nt = (f"[{_ns or 'status not given'}; certainty: {_nc or 'not given'}] {_plain} | evidence: {_ne or 'none given'}"
                       + (f" (recorded on level {current_level}: " + "; ".join(_cites) + ")" if _cites else ""))
                note_meta[_nk] = {"status": _ns, "certainty": _nc, "text": _plain}
            if not _nt:
                note_meta.pop(_nk, None)
                if _nk in notes:
                    turn_note_ops.append({"key": _nk, "op": "deleted", "old": notes[_nk]})
                    del notes[_nk]
                    results.append(f"[Notes] Deleted entry '{_nk}'.")
                else:
                    results.append(f"[Notes] No entry '{_nk}' to delete.")
                continue
            _old = notes.get(_nk)
            notes[_nk] = _nt
            _total = sum(len(k) + len(v) + 4 for k, v in notes.items())
            if _total > NOTES_CHAR_CAP:
                if _old is None:
                    del notes[_nk]
                    note_meta.pop(_nk, None)
                else:
                    notes[_nk] = _old
                turn_note_ops.append({"key": _nk, "op": "rejected", "text": _nt})
                results.append(f"[Notes] Not saved: the notes would exceed {NOTES_CHAR_CAP} characters ({_total}). "
                               "Shorten or delete other entries first (a note with an empty text deletes that entry).")
                continue
            turn_note_ops.append({"key": _nk, "op": ("updated" if _old is not None else "added"),
                                  **({"old": _old} if _old is not None and _old != _nt else {})})
            results.append(f"[Notes] {'Updated' if _old is not None else 'Added'} entry '{_nk}' ({len(notes)} entries, {_total}/{NOTES_CHAR_CAP} chars).")

        # Phase 3b: Remaining-actions indicator tool updates
        for cmd in remaining_actions_indicator_cmds:
            rect, err = _parse_remaining_actions_indicator_rect(cmd.content, board_shape=current_board.shape[:2])
            if err:
                results.append(f"[Remaining-actions indicator] Error: {err}")
                continue
            remaining_actions_indicator_rect = rect if rect else None
            remaining_rect_modified = True
            canonical_seen_action_index = _rebuild_canonical_seen_action_index(
                all_boards,
                remaining_actions_indicator_rect,
                offset=level_action_start_index,
            )
            _n_rects = len(norm_rects(remaining_actions_indicator_rect))
            indicator_msg = (
                "[Remaining-actions indicator] Cleared (no rectangle marked)." if _n_rects == 0 else
                f"[Remaining-actions indicator] Now marked: {_n_rects} rectangle(s) "
                f"{_remaining_actions_indicator_rect_text(remaining_actions_indicator_rect)} (replaces the previous selection)."
            )
            if _n_rects:
                indicator_area = int(rects_mask(remaining_actions_indicator_rect, current_board.shape).sum())
                indicator_pct = 100.0 * indicator_area / BOARD_CELLS
                if indicator_pct > 5.0:
                    indicator_msg += (
                        f"\n  [Warning] The marked area covers {indicator_pct:.1f}% of the board "
                        f"({indicator_area} cells). That seems large for a remaining-actions indicator. "
                        "Double-check that it contains ONLY the indicator and nothing else. "
                        "You can change or clear it later if needed."
                    )
            results.append(indicator_msg)

        # Phase 3c: Estimate progress (just acknowledge, actual logic is above)
        for cmd in estimate_cmds:
            est_pct, est_turns = parse_estimate_from_commands([cmd])
            if est_pct is not None and est_turns is not None:
                results.append(
                    f"[Estimate] Recorded: {est_pct}% chance, ~{est_turns} turns remaining."
                )
            else:
                results.append(
                    "[Estimate] Could not parse estimate. Use: estimate_progress(finish_percent=N, median_turns=N)"
                )

        # Recovery may supply one fallback, routed through the existing guarded action path.
        # Model commands in the replay remain exactly those the model emitted.
        _model_env_intent = bool(env_cmds)
        if code_changed:
            _decision_evidence.append("code:\n" + code_file.source)
        if remaining_rect_modified or view_cmds:
            _decision_evidence.append("view:" + repr(remaining_actions_indicator_rect) + repr(_ext.ctx().view_exclude))
        _forced_candidate = _decision.process(
            options=_decision_options, output_tokens=_out_tok, capped=hit_length_cap,
            commands=commands, proposals=(_minimal_proposals if MINIMAL_HARNESS else
                                          [c.content for c in commands if c.kind == "propose_actions"]),
            state=_decision_identity(), already_acted=_ext.ctx().turn_actions_executed > 0,
            # A think control response cannot dispatch even a committed recovery
            # fallback. Keep recovery accounting, but defer automatic execution.
            env_intent=_model_env_intent or bool(_think_cmds), evidence=_decision_evidence, validate=_validate_fallback,
            analysis_failed=_minimal_analysis_failed)
        if _forced_candidate:
            env_cmds = [Command("actions", _forced_candidate["action"])]
            cur_turn_info["controller_actions"] = [_forced_candidate["action"]]
            results.append("[Decision recovery] Refinement budget/request limit reached. The harness is taking your committed "
                           f"fallback: {_forced_candidate['action']}. Expected: {_forced_candidate['expected']}")
        if _decision.settings.mode != "off":
            cur_turn_info["decision_policy"] = _decision.snapshot()
        for _policy_event in _decision.events:
            if _policy_event["event"] in {"proposal_rejected", "proposal_repair", "fallback_invalidated", "recovery_aborted"}:
                results.append("[Decision recovery] " + _policy_event["event"] + ": " + str(_policy_event.get("reason", "")))
            elif _policy_event["event"] == "proposal_accepted":
                results.append("[Decision recovery] Candidates saved; highest valid score: " + _decision.candidates[0]["action"]
                               if _decision.candidates else "[Decision recovery] Proposal received; new results require revision.")

        # Phase 4: Environment actions (only first actions/path, only if validation passed)
        if env_cmds and _ext.ctx().turn_actions_executed > 0:
            results.append(
                f"[Note] actions/path skipped: this turn already executed {_ext.ctx().turn_actions_executed} action(s) "
                "via act() in run; one environment-modifying step per response."
            )
            env_cmds = []
        board_before_actions = current_board.copy() if env_cmds else None
        if board_before_actions is None and _ext.ctx().turn_actions_executed > 0:   # v2.10: act() turns count too
            board_before_actions = _board_before_run
        if env_cmds:
            if len(env_cmds) > 1:
                results.append(f"[Note] Only first environment command executed; {len(env_cmds)-1} skipped.")
            cmd = env_cmds[0]

            if cmd.kind == "actions":
                action_lines: list[str] = []
                _repeat_flag = False
                _action_checks, _expect_error = [], None
                for raw_line in cmd.content.split("\n"):
                    stripped = raw_line.strip()
                    if not stripped:
                        continue
                    if stripped.startswith("#"):
                        if stripped.startswith("#expect "):
                            try:
                                _action_checks = _ext.reference_expect(json.loads(stripped[8:]))
                            except Exception as exc:
                                _expect_error = str(exc)
                        if stripped.lower() == "#repeat":
                            _repeat_flag = True
                        continue
                    action_lines.append(stripped)
                action_lines_to_execute = action_lines
                if _expect_error:
                    action_lines = []
                    action_lines_to_execute = []
                skipped_action_lines = 0
                _has_reset = any(_ext.normalize_label(_l.split()[0]) == RESET_LABEL for _l in action_lines)
                if (not _has_reset and MAX_ACTIONS_PER_MESSAGE is not None
                        and len(action_lines) > MAX_ACTIONS_PER_MESSAGE):
                    action_lines_to_execute = action_lines[:MAX_ACTIONS_PER_MESSAGE]
                    skipped_action_lines = len(action_lines) - len(action_lines_to_execute)
                action_results: list[str] = []
                if _expect_error:
                    action_results.append("Invalid expectations (no action executed): " + _expect_error)
                if _has_reset and _action_checks:
                    action_results.append("Expectations cannot span RESET; no action executed.")
                    action_lines_to_execute, _has_reset = [], False
                if not action_lines_to_execute:
                    action_results.append(
                        "  No action lines to execute (blank/comment lines are ignored)."
                    )
                if skipped_action_lines > 0:
                    action_results.append(
                        f"  Action cap active: executed first {len(action_lines_to_execute)} action(s), "
                        f"ignored {skipped_action_lines} additional action(s) in this message."
                    )
                if _has_reset:
                    _refusal = _manual_reset_refusal(len(action_lines))
                    if _refusal is not None:
                        action_results.append(f"  {_refusal}")
                        _ext.log_guard("manual_reset_refused", reason=_refusal, where="actions")
                    else:
                        reset_obs = _env_reset_retry(env)
                        if reset_obs is None:
                            action_results.append("  RESET failed: the environment returned no observation; the level was not reset.")
                        else:
                            actions_taken.append(RESET_LABEL)
                            _ext.log_guard("manual_reset", where="actions")
                            action_results.append(f"  RESET: level {current_level} restarts from its opening board.")
                            level_transition_messages.append(_apply_level_restart(reset_obs, manual=True, where="actions"))
                    action_lines_to_execute = []

                for _act_idx, action_text in enumerate(action_lines_to_execute):
                    if _forced_candidate:
                        try:
                            _ext.reference_expect(_forced_candidate.get("checks"))
                        except ValueError as exc:
                            action_results.append("Automatic fallback refused: " + str(exc))
                            _decision.event("fallback_vetoed", action=action_text, reason=str(exc))
                            break
                        _fallback_error = _validate_fallback(action_text)
                        if _fallback_error:
                            action_results.append("  Automatic fallback refused: " + _fallback_error)
                            _decision.event("fallback_vetoed", action=action_text, reason=_fallback_error)
                            break
                    _requested_action = action_text
                    action_text, _name_error = _ext.resolve_named_action(action_text, current_board)
                    if _name_error:
                        action_results.append("Named action rejected (no action used): " + _name_error)
                        break
                    parts = action_text.split()
                    label = _ext.normalize_label(parts[0])
                    spec = next((s for s in action_specs if s.label == label), None)
                    if spec is None:
                        action_results.append(f"  Unknown action '{parts[0]}'. Available: {action_labels}")
                        break

                    r, c = -1, -1
                    claimed_value = -1
                    if spec.requires_xy:
                        if len(parts) < 4:
                            action_results.append(
                                f"  {label} requires: {label} row col value "
                                f"(where value is the current color at that cell)."
                            )
                            break
                        try:
                            r, c = int(parts[1]), int(parts[2])
                        except ValueError:
                            action_results.append(f"  Invalid coordinates for {label}.")
                            break
                        try:
                            claimed_value = int(parts[3])
                        except ValueError:
                            action_results.append(
                                f"  Invalid value '{parts[3]}' for {label}. Must be an integer color value."
                            )
                            break
                        _rows, _cols = current_board.shape[0], current_board.shape[1]
                        if not (0 <= r < _rows and 0 <= c < _cols):
                            turn_click_cells.append({"row": r, "col": c, "value": claimed_value, "accepted": False})
                            action_results.append(
                                f"  Action rejected: ({r},{c}) is outside the board "
                                f"(valid rows 0-{_rows - 1}, cols 0-{_cols - 1})."
                            )
                            break
                        actual_value = int(current_board[r, c])
                        if claimed_value != actual_value:
                            turn_click_cells.append({"row": r, "col": c, "value": claimed_value, "accepted": False})
                            action_results.append(
                                f"  {action_text}: " + _ext.click_rejection_text(current_board, r, c, claimed_value)
                            )
                            break

                    ca = CandidateAction(spec.action_id, spec.label, row=r, col=c, value=claimed_value,
                                         requested=_requested_action if _requested_action != action_text else None)
                    if ACT_GUARD_MODE == "known":
                        _kg = _ext.known_guard(current_board, remaining_actions_indicator_rect, ca.as_text(), repeat=_repeat_flag)
                        if _kg is not None:
                            _skipped = len(action_lines_to_execute) - _act_idx - 1
                            action_results.append(f"  {action_text}: refused ({_kg.replace('act([...], repeat=True)', 'actions(..., repeat=true)')})"
                                                  + (f" {_skipped} remaining action(s) were NOT executed." if _skipped else ""))
                            _ext.log_guard("known_refused", action=ca.as_text(), where="actions")
                            break
                    if spec.requires_xy:
                        turn_click_cells.append({"row": r, "col": c, "value": claimed_value, "accepted": True})
                    _board_before_action = current_board
                    _colors_before = set(_color_counts(current_board).keys()) | seen_colors
                    _ext.ctx().references.expected = (_forced_candidate.get("checks", []) if _forced_candidate else _action_checks)
                    result_text, outcome, level_after = _execute_action(
                        env,
                        ca,
                        all_boards,
                        actions_taken,
                        history,
                        current_level,
                        seen_colors,
                        remaining_actions_indicator_rect,
                        canonical_seen_action_index,
                        quiet=quiet,
                    )
                    current_board = all_boards[-1]
                    action_results.append(result_text)
                    if _forced_candidate and (_ext.ctx().last_step or {}).get("outcome") != "env_error":
                        _changed = _outside = None
                        if _board_before_action.shape == current_board.shape:
                            _diff = _board_before_action != current_board
                            _changed = int(np.count_nonzero(_diff))
                            _outside = int(np.count_nonzero(_diff & ~rects_mask(remaining_actions_indicator_rect, _diff.shape)))
                        _decision.forced_result(_forced_candidate, outcome=outcome, changed_cells=_changed,
                                                changed_outside_indicator=_outside,
                                                checks=_ext.ctx().references.last_checks)
                        cur_turn_info["decision_policy"] = _decision.snapshot()
                        if _decision.disabled:
                            results.append("[Decision recovery] Automatic-action limit reached; ordinary play resumes for this level.")
                    _ext.record_known(_board_before_action, remaining_actions_indicator_rect, ca.as_text(),
                                      noop=(outcome == "continue" and level_after == current_level and _ext.last_step_is_noop()),
                                      death=(outcome == "game_over"))
                    _new_colors = sorted(set(_color_counts(current_board).keys()) - _colors_before)
                    if outcome == "continue" and level_after == current_level and _ext.ctx().references.stop_reason():
                        action_results.append("  " + _ext.ctx().references.stop_reason())
                        break
                    if (_ext.ctx().last_step or {}).get("outcome") == "env_error":   # v2.9
                        _skipped = len(action_lines_to_execute) - _act_idx - 1
                        if _skipped > 0:
                            action_results.append(f"  Batch stopped after the environment error; {_skipped} remaining action(s) "
                                                  "were NOT executed.")
                        break
                    if outcome == "continue" and level_after == current_level and _new_colors and _act_idx < len(action_lines_to_execute) - 1:
                        _skipped = len(action_lines_to_execute) - _act_idx - 1
                        action_results.append(
                            f"  Batch stopped: new colour(s) {', '.join(map(str, _new_colors))} never seen before in this game appeared after "
                            f"{action_text}; {_skipped} remaining action(s) were NOT executed so you can inspect this state first."
                        )
                        _ext.log_guard("batch_new_color", action=action_text, colors=_new_colors, where="actions")
                        break
                    if outcome == "continue" and level_after == current_level and _ext.last_step_is_noop():
                        _skipped = len(action_lines_to_execute) - _act_idx - 1
                        if _skipped > 0:
                            action_results.append(
                                f"  Batch stopped: {_ext.unchanged_board_text(action_text)}; "
                                f"{_skipped} remaining action(s) were NOT executed."
                            )
                            _ext.log_guard("batch_noop", action=action_text, skipped=_skipped, where="actions")
                            break

                    prev_level = current_level
                    current_level = level_after

                    if outcome == "win":
                        _note_win(prev_level)
                        action_results.append("GAME WON!")
                        results.append("[Actions]\n" + "\n".join(action_results))
                        finalize_run(
                            kind="success",
                            title="Game won!",
                            status_line="Game won!",
                            reasoning_text=reasoning,
                            results_lines=results,
                        )
                        return

                    if outcome == "game_over":
                        action_results.append("GAME OVER!")
                        if RESET_ON_GAME_OVER:
                            action_results.append("Level will restart. Previously completed levels are kept.")
                            # env.reset(): in competition mode this restarts the current level (keeps progress);
                            # arc.make() would fail because competition mode allows a single make() per environment.
                            _ext.record_death(all_boards[-1], compaction_level_start_board, remaining_actions_indicator_rect)
                            reset_obs = _env_reset_retry(env)
                            if reset_obs is None:
                                results.append("[Actions]\n" + "\n".join(action_results))
                                finalize_run(
                                    kind="game_over",
                                    title="Game over (reset failed)",
                                    status_line="Game over. Reset failed.",
                                    reasoning_text=reasoning,
                                    results_lines=results,
                                )
                                return
                            level_transition_messages.append(_apply_level_restart(reset_obs, manual=False, where="actions"))
                            break  # break inner action loop, continue outer turn loop
                        else:
                            results.append("[Actions]\n" + "\n".join(action_results))
                            finalize_run(
                                kind="game_over",
                                title="Game over",
                                status_line="Game over. Ending run.",
                                reasoning_text=reasoning,
                                results_lines=results,
                            )
                            raise SystemExit(1)

                    if level_after > prev_level:
                        _apply_level_complete(prev_level)

                        if level_after >= target_level:
                            action_results.append("LEVEL COMPLETE!")
                            results.append("[Actions]\n" + "\n".join(action_results))
                            finalize_run(
                                kind="success",
                                title="Level complete!",
                                status_line=f"Level complete! Reached target level {target_level}.",
                                reasoning_text=reasoning,
                                results_lines=results,
                            )
                            return

                        action_results.append(
                            f"  Level {prev_level} completed successfully. Now on level {level_after}."
                        )
                        action_results.append(
                            "  Stopping further actions this turn so you can analyze the new level."
                        )
                        level_transition_messages.append(
                            build_level_start_message(
                                current_level=level_after,
                                previous_level=prev_level,
                                board=current_board,
                                action_labels=action_labels,
                                total_actions_taken=len(actions_taken),
                            )
                            + "\n\n"
                            + _full_board_color_inventory_text(current_board)
                            + "\n\n"
                            + _ext.level_start_objects_text(current_board, remaining_actions_indicator_rect)
                            + _level_complete_extra()
                        )
                        break

                results.append("[Actions]\n" + "\n".join(action_results))

            elif cmd.kind == "path":
                if _code_changed_in_run:   # v2.10: run changed the code file: resolve and validate again
                    path_next_state_fn, _model_mode, validation_passed, _mc_msgs = _model_check(True)
                    results.extend(_mc_msgs)
                    _code_changed_in_run = False
                if not path_next_state_fn:
                    results.append(
                        "[Path] Error: no next_state(board, action, row=-1, col=-1) in your code file. Write the model first."
                    )
                elif not validation_passed:
                    _mc_shown = any(str(_r).startswith("[Model check]") for _r in results)   # v2.11c: not twice per turn
                    results.append("[Path] Skipped: the model check found differences or errors on this level's transitions "
                                   + ("(see the [Model check] report above)" if _mc_shown else "(the report follows)")
                                   + "; fix them first (returning None for a case you don't know is fine).")
                    if not _mc_shown:
                        _hm2, _mm2, _vok2, _mcl2 = _model_check(True, True)   # v2.11: show the (cached) report again
                        results.extend(_mcl2)
                else:
                    max_steps = 10
                    m = re.search(r"max_steps\s*:\s*(\d+)", cmd.content)
                    if m:
                        max_steps = int(m.group(1))

                    states_arr = np.stack(all_boards, axis=0)

                    if not quiet:
                        log_tag("BFS", f"Searching (max_steps={max_steps})...", Color.BLUE)
                    _bfs_started = perf_counter()
                    _bfs_tl = BFS_TIME_LIMIT_SECONDS
                    if deadline is not None:
                        _bfs_tl = max(1.0, min(_bfs_tl, deadline - wall_time() - 30.0))
                    try:   # v2.11: the search runs in the worker
                        search = _ext.path_search(code_file=code_file, start=current_board, specs=search_specs(action_specs),
                                                  rect=remaining_actions_indicator_rect, states=states_arr,
                                                  actions_history=actions_taken[level_action_start_index:], store=store,
                                                  hist_summary=_hist_summary_list(), past_levels=past_levels,
                                                  time_limit=_bfs_tl, action_labels=action_labels)
                    except Exception as _bexc:
                        search = {"found": False, "error": f"the search failed: {_bexc!r}", "path": [], "pred": [], "explored": 0}
                    elapsed = perf_counter() - _bfs_started
                    _snotes = search.get("notes")

                    if search.get("error"):
                        results.append(f"[Path] Search error: {search['error']}")
                    elif not search.get("found"):
                        _why = {
                            "exhausted": "every state reachable under your model was explored and none was predicted to "
                                         "complete the level: under your model the win is unreachable from here",
                            "states": f"the {MAX_DISTINCT_STATES:,}-state limit was reached before a win was found",
                            "depth": f"no win within {_MAX_ACTION_DEPTH_CFG} actions (the depth limit)",
                            "time": f"the search time limit ({_bfs_tl:.0f} s) was reached before a win was found",
                        }.get(search.get("stop_reason") or "", "no win was found")
                        results.append(f"[Path] No path found after exploring {int(search.get('explored') or 0):,} states: {_why}."
                                       + (f" Note: {_snotes}." if _snotes else ""))
                    else:
                        path = search["path"]
                        _preds = search.get("pred") or []
                        n_exec = min(max_steps, len(path))
                        results.append(
                            f"[Path] Found path of {len(path)} actions "
                            f"({int(search.get('explored') or 0):,} states explored, {elapsed:.1f}s). "
                            f"Executing first {n_exec}:"
                            + (f" (note: {_snotes})" if _snotes else "")
                        )
                        path_results: list[str] = []
                        for i, action in enumerate(path[:n_exec]):
                            # what next_state predicted for this step during the search (v2.11: from the worker)
                            _pred_board, _pred_out = _preds[i] if i < len(_preds) else (None, None)
                            if action.row >= 0 and action.col >= 0:
                                turn_click_cells.append({"row": action.row, "col": action.col, "value": action.value, "accepted": True})
                            rt, outcome, level_after = _execute_action(
                                env,
                                action,
                                all_boards,
                                actions_taken,
                                history,
                                current_level,
                                seen_colors,
                                remaining_actions_indicator_rect,
                                canonical_seen_action_index,
                                quiet=quiet,
                            )
                            current_board = all_boards[-1]
                            path_results.append(_ext.step_entry(f"Step {i+1}/{n_exec} {action.as_text()}: {rt}", action.as_text()))
                            if (_ext.ctx().last_step or {}).get("outcome") == "env_error":   # v2.10
                                path_results.append(f"  Path stopped after step {i+1}: the environment returned an error "
                                                    "(not a model mismatch); the remaining planned actions were NOT executed.")
                                n_exec = i + 1
                                break

                            prev_level = current_level
                            current_level = level_after

                            # v2.10: a predicted level completion that did not happen
                            if _pred_out == "level_complete" and outcome == "continue" and level_after == prev_level:
                                if _ext.ctx().last_step is not None:
                                    _ext.ctx().last_step["prediction_check"] = {"status": "contradicted", "reason": "win_condition"}
                                path_results.append(
                                    f"  Path stopped after step {i+1}: your model predicted that {action.as_text()} completes the "
                                    "level, but the level did NOT complete: the win condition in your model is wrong or incomplete."
                                )
                                n_exec = i + 1
                                break
                            # batch no-op stop, then per-step verification against the model's prediction
                            if outcome == "continue" and level_after == prev_level and _ext.last_step_is_noop():
                                path_results.append(
                                    f"  Path stopped after step {i+1}: {_ext.unchanged_board_text(action.as_text())}; "
                                    f"{len(path) - (i + 1)} planned action(s) were NOT executed."
                                )
                                _ext.log_guard("batch_noop", action=action.as_text(), skipped=len(path) - (i + 1), where="path")
                                n_exec = i + 1
                                break
                            if outcome == "continue" and level_after == prev_level and _pred_board is not None:
                                _prediction_checked = False
                                try:
                                    _obs_board = to_uint8_board(current_board)
                                    _mismatch = _obs_board != _pred_board
                                    _mismatch &= ~rects_mask(remaining_actions_indicator_rect, _mismatch.shape)
                                    _n_mismatch = int(np.count_nonzero(_mismatch))
                                    _prediction_checked = True
                                except Exception:
                                    _n_mismatch = 0
                                if _ext.ctx().last_step is not None:
                                    _ext.ctx().last_step["prediction_check"] = {
                                        "status": ("error" if not _prediction_checked else
                                                   "contradicted" if _n_mismatch else "satisfied")}
                                if _n_mismatch > 0:
                                    _mr, _mc = np.nonzero(_mismatch)
                                    _mm_rect = [int(_mr.min()), int(_mr.max()) + 1, int(_mc.min()), int(_mc.max()) + 1]
                                    path_results.append(
                                        f"  Prediction mismatch after step {i+1}: the board differs from what your next_state predicted "
                                        f"({_n_mismatch} cell(s) within [{_mm_rect[0]}:{_mm_rect[1]}, {_mm_rect[2]}:{_mm_rect[3]}]). "
                                        f"Path execution stopped; {len(path) - (i + 1)} planned action(s) were NOT executed. "
                                        "Compare states[-2] -> states[-1] (lib.frame_diff()) with your model and fix it before running path again."
                                    )
                                    _ext.ctx().prediction_mismatch = {"step": i + 1, "action": action.as_text(), "cells": _n_mismatch, "rect": _mm_rect}
                                    _ext.log_guard("prediction_mismatch", action=action.as_text(), cells=_n_mismatch, rect=_mm_rect)
                                    n_exec = i + 1
                                    break

                            if outcome == "win":
                                _note_win(prev_level)
                                path_results.append("  GAME WON!")
                                results.append(_ext.format_step_reports(path_results, prefix="  ")[0])
                                finalize_run(
                                    kind="success",
                                    title="Game won!",
                                    status_line="Game won!",
                                    reasoning_text=reasoning,
                                    results_lines=results,
                                )
                                return

                            if outcome == "game_over":
                                path_results.append("  GAME OVER!")
                                if RESET_ON_GAME_OVER:
                                    path_results.append("  Level will restart. Previously completed levels are kept.")
                                    # env.reset(): in competition mode this restarts the current level (keeps progress)
                                    _ext.record_death(all_boards[-1], compaction_level_start_board, remaining_actions_indicator_rect)
                                    reset_obs = _env_reset_retry(env)
                                    if reset_obs is None:
                                        results.append(_ext.format_step_reports(path_results, prefix="  ")[0])
                                        finalize_run(
                                            kind="game_over",
                                            title="Game over (reset failed)",
                                            status_line="Game over. Reset failed.",
                                            reasoning_text=reasoning,
                                            results_lines=results,
                                        )
                                        return
                                    level_transition_messages.append(_apply_level_restart(reset_obs, manual=False, where="path"))
                                    break  # break path execution, continue to next turn
                                else:
                                    results.append(_ext.format_step_reports(path_results, prefix="  ")[0])
                                    finalize_run(
                                        kind="game_over",
                                        title="Game over",
                                        status_line="Game over. Ending run.",
                                        reasoning_text=reasoning,
                                        results_lines=results,
                                    )
                                    raise SystemExit(1)

                            if level_after > prev_level:
                                _apply_level_complete(prev_level)

                                if level_after >= target_level:
                                    path_results.append("  LEVEL COMPLETE!")
                                    results.append(_ext.format_step_reports(path_results, prefix="  ")[0])
                                    finalize_run(
                                        kind="success",
                                        title="Level complete!",
                                        status_line=f"Level complete! Reached target level {target_level}.",
                                        reasoning_text=reasoning,
                                        results_lines=results,
                                    )
                                    return

                                path_results.append(
                                    f"  Level {prev_level} completed successfully. Now on level {level_after}."
                                )
                                path_results.append(
                                    "  Stopping path execution so you can analyze the new level."
                                )
                                level_transition_messages.append(
                                    build_level_start_message(
                                        current_level=level_after,
                                        previous_level=prev_level,
                                        board=current_board,
                                        action_labels=action_labels,
                                        total_actions_taken=len(actions_taken),
                                    )
                                    + "\n\n"
                                    + _full_board_color_inventory_text(current_board)
                                    + "\n\n"
                                    + _ext.level_start_objects_text(current_board, remaining_actions_indicator_rect)
                                    + _level_complete_extra()
                                )
                                break

                        results.append(_ext.format_step_reports(path_results, prefix="  ")[0])
                        if len(path) > n_exec and not level_transition_messages:
                            remaining = [a.as_text() for a in path[n_exec:n_exec+20]]
                            results.append(f"  Remaining path ({len(path)-n_exec} actions): {' -> '.join(remaining)}" +
                                          ("..." if len(path) - n_exec > 20 else ""))

        if _decision.settings.mode != "off":
            _decision.finish_turn(_decision_identity(), env_intent=_model_env_intent, forced=_forced_candidate is not None)
            cur_turn_info["decision_policy"] = _decision.snapshot()
            if _decision.end_feedback():
                results.append(_decision.end_feedback())
            if _decision.events:
                _sync_print(f"[{game_id_short}] DECISION {json.dumps(_decision.snapshot(), separators=(',', ':'))}")

        # v2.11: the model check on this turn's new transitions (or on the functions run kept), reported when it changed
        if code_file.has("next_state") and (len(history) != _hist_len_before or _code_changed_in_run):
            _hm, _mm, _vok, _mc_msgs2 = _model_check(_code_changed_in_run)
            results.extend(_mc_msgs2)

        _finish_reasoning_turn()
        if ADAPTIVE_REASONING:
            cur_turn_info["reasoning_control"] = _reasoning.snapshot()

        if EVIDENCE_NOTES:   # v2.12b: after many turns on a level without completing it, list what is still untested
            if stuck_state["level"] != current_level:
                stuck_state.update(level=current_level, turns=0, next=STUCK_CHECK_TURNS)
            stuck_state["turns"] += 1
            if not level_transition_messages and stuck_state["turns"] >= stuck_state["next"]:
                stuck_state["next"] = stuck_state["turns"] + max(1, STUCK_CHECK_EVERY)
                try:
                    _sc = _stuck_check_lines()
                except Exception as _sexc:
                    _sc = []
                    if not quiet:
                        log_tag("STUCK", f"stuck check failed: {_sexc!r}", Color.YELLOW)
                if _sc:
                    results.append(f"[Stuck check] {stuck_state['turns']} turns on level {current_level} without completing it. "
                                   "From your pinned notes and this level's history:\n" + "\n".join(_sc))

        if level_transition_messages:
            for transition_msg in level_transition_messages:
                results.append(f"[Level transition]\n{transition_msg}")
            pending_full_board_for_image = current_board.copy()

        # Track consecutive no-change action turns
        if board_before_actions is not None and board_before_actions.shape != current_board.shape:
            consecutive_no_change_action_turns = 0  # new board geometry, not a comparable pixel diff
        elif board_before_actions is not None:
            real_changed = int(np.count_nonzero(
                to_uint8_board(board_before_actions) != to_uint8_board(current_board)
            ))
            if norm_rects(remaining_actions_indicator_rect) and real_changed > 0:
                mask = rects_mask(remaining_actions_indicator_rect, current_board.shape)
                diff = to_uint8_board(board_before_actions) != to_uint8_board(current_board)
                real_changed = int(np.count_nonzero(diff & ~mask))
            if real_changed == 0:
                consecutive_no_change_action_turns += 1
            else:
                consecutive_no_change_action_turns = 0

        _ext.reference_sync(current_board, current_level, attempt_no, len(actions_taken))
        _imap = _ext.reference_text()
        if _imap:
            results.append(_imap)
        if results:
            feedback_message = "\n\n".join(results)
            _fb_cap = FEEDBACK_CHAR_CAP_LEVEL_TRANSITION if level_transition_messages else FEEDBACK_CHAR_CAP
            if not MINIMAL_HARNESS and len(feedback_message) > _fb_cap:   # never truncate the minimal action ledger
                _res2 = list(results)
                _excess = len(feedback_message) - _fb_cap
                for _ri in sorted((i for i, r in enumerate(_res2) if r.startswith("[Run]")), key=lambda i: -len(_res2[i])):
                    if _excess <= 0:
                        break
                    _r = _res2[_ri]
                    _keep = max(1500, len(_r) - _excess)
                    if _keep < len(_r):
                        _h, _t = int(_keep * 0.65), int(_keep * 0.35)
                        _res2[_ri] = (_r[:_h] + f"\n...[{len(_r) - _h - _t} characters of your printed output omitted to fit "
                                      "the feedback limit]...\n" + _r[len(_r) - _t:])
                        _excess -= len(_r) - len(_res2[_ri])
                feedback_message = "\n\n".join(_res2)
                if len(feedback_message) > _fb_cap:
                    _omitted = len(feedback_message) - _fb_cap
                    feedback_message = feedback_message[:_fb_cap] + (
                        f"\n...[harness feedback truncated: {_omitted} chars omitted. "
                        "Use run() to inspect specific details.]")
        elif was_empty_response:
            feedback_message = (
                "(No commands executed this turn — the API returned empty content. "
                f"You MUST include {command_mode_desc} in your response.)"
            )
        else:
            if MINIMAL_HARNESS:
                feedback_message = "(No commands executed. Use run(code) to inspect, save memory or act.)"
            elif USE_TOOL_CALLING:
                feedback_message = (
                    "(No commands executed this turn. Your response contained no recognized "
                    "tool calls. You must use the function tools: py, patch, run, actions, "
                    "path, remaining_actions_indicator. Do NOT describe actions in plain text — "
                    "you must invoke the tools.)"
                )
            else:
                feedback_message = (
                    "(No commands executed this turn. Your response contained no recognized "
                    "command blocks. You must use ```actions, ```run, ```py, ```patch, ```path, "
                    "or ```remaining_actions_indicator fenced code blocks.)"
                )

        # Insert role:"tool" messages into conversation when tool_calls were used.
        # Per DeepSeek/OpenAI chat completions protocol: assistant(tool_calls) -> tool messages -> next user.
        _provider = model_provider(MODEL_NAME, provider_override=MODEL_PROVIDER_OVERRIDE)
        _uses_chat_completions = _provider in {"deepseek", "sambanova", "google", "together", "deepinfra", "local"}
        tool_response_sent = False
        combined_tool_result = feedback_message   # kept for the replay event below
        if turn_result.tool_calls and _uses_chat_completions:
            for i, tc in enumerate(turn_result.tool_calls):
                tool_msg: dict[str, Any] = {
                    "role": "tool",
                    "tool_call_id": tc.call_id or f"call_{i}",
                    "content": combined_tool_result if i == 0 else "",
                }
                conversation.append(tool_msg)
            tool_response_sent = True
            conv_tokens_est += len(combined_tool_result) / 3.3

        # Build the next user message. When tool responses already carry the full
        # feedback, only include status/stats to avoid sending duplicate content.
        _attempt_actions = len(actions_taken) - level_action_start_index
        _level_actions = len(actions_taken) - level_total_start_index
        status_lines = [f"Current level: {current_level}"]
        if attempt_no > 1:
            status_lines.append(f"Actions taken on this level: {_level_actions} over {attempt_no} attempts; "
                                f"{_attempt_actions} in the current attempt (actions_history in run)")
        else:
            status_lines.append(f"Actions taken on this level: {_level_actions} (actions_history in run)")
        status_lines.append(f"Total actions taken across run: {len(actions_taken)}")
        status_lines.append(f"Observed transitions on this level (history in run): {len(history)}")
        _status_text_last = "\n".join(status_lines)   # v2.8: replaced by the lean note when nothing was executed
        if tool_response_sent:
            # Tool response already has full feedback; user message is stats-only.
            feedback_message = "\n".join(status_lines)
        else:
            feedback_message += "\n\n" + "\n".join(status_lines)
        message_count = replay.llm_messages
        action_count = len(actions_taken)
        if not MINIMAL_HARNESS and (message_count >= 10 or action_count >= 10):
            too_many_messages = (message_count > (4 * action_count)) if action_count > 0 else (message_count >= 15)
            if too_many_messages:
                feedback_message += (
                    "\n[Efficiency guidance] You have used many messages relative to actions taken. "
                    "LLM calls are expensive. When several next actions are already justified and do not require "
                    "intermediate interpretation, batch up to 4 actions in one actions call. "
                    "Reuse established conclusions instead of re-deriving them."
                )
        if (NO_CHANGE_WARNING_TURNS > 0
                and consecutive_no_change_action_turns >= NO_CHANGE_WARNING_TURNS):
            feedback_message += (
                f"\n[Note] The last {consecutive_no_change_action_turns} turns with actions ended on an unchanged board "
                "outside the remaining-actions indicator; inspect any intermediate animation before drawing conclusions."
            )

        # Serialize symbolic state for replay (sorted by value length, shortest first)
        symbolic_state_json = None
        if current_symbolic is not None:
            try:
                plain = _to_plain_symbolic(current_symbolic)
                if isinstance(plain, dict):
                    plain = dict(sorted(plain.items(), key=lambda kv: len(json.dumps(kv[1], default=str))))
                symbolic_state_json = json.dumps(plain, indent=2, ensure_ascii=False, default=str)
            except Exception:
                symbolic_state_json = repr(current_symbolic)

        if MEMORY_DEBUG and not quiet:
            print(f"  [MEM] end-of-turn RSS={_mem_mb():.0f}MB")

        event_images = [
            {"label": img.label, "data_url": img.data_url, "scale": img.scale}
            for img in turn_result.image_attachments
        ] if turn_result.image_attachments else None

        replay.add_event(
            board=current_board, actions_taken=len(actions_taken),
            current_level=current_level,
            kind="turn", title=f"Turn {turn}",
            available_actions=action_labels,
            message_text=f"Reasoning:\n{reasoning}\n\nAssistant text:\n{turn_result.text or ''}\n\nFeedback:\n{feedback_message}",
            llm_reasoning=reasoning,
            llm_reasoning_content=turn_result.reasoning_content or None,
            llm_commands_summary=commands_summary,
            llm_commands=[{"kind": cmd.kind, "content": cmd.content} for cmd in _reported_commands] if _reported_commands else None,
            harness_feedback=((f"[Assistant text]\n{turn_result.text}\n\n" if (turn_result.text or "").strip() else "")
                              + (f"[Tool results]\n{combined_tool_result}\n\n" if tool_response_sent else "")
                              + feedback_message),
            images=event_images,
            click_cells=turn_click_cells if turn_click_cells else None,
            remaining_actions_rect=(list(norm_rects(remaining_actions_indicator_rect)[0]) if norm_rects(remaining_actions_indicator_rect) else None),
            remaining_actions_rects=([list(r) for r in norm_rects(remaining_actions_indicator_rect)] or None),
            remaining_actions_rect_modified=remaining_rect_modified,
            code_changed=code_changed,
            previous_code=prev_code if code_changed else None,
            has_board_to_symbolic=board_to_symbolic_fn is not None,
            has_symbolic_to_board=symbolic_to_board_fn is not None,
            has_next_state=next_state_fn is not None,
            symbolic_state_json=symbolic_state_json,
            **_ext.turn_log_kwargs(current_board, remaining_actions_indicator_rect),
            # v2.5-log
            turn=turn,
            **{k: v for k, v in cur_turn_info.items() if k not in ("turn",)},
            notes=dict(notes), notes_structured=(confidence_notes.snapshot() if confidence_notes is not None else None),
            note_ops=(list(turn_note_ops) or None),
            event_log=list(event_log[-80:]), store_summary=_store_summary(store),
            level_complete=(list(turn_level_completions) or None),
            attempt=attempt_no,
            actions_on_level=len(actions_taken) - level_total_start_index,
        )
        _flush_pending_events()
        cur_turn_info, cur_turn_extra = {}, {}

        if level_transition_messages:   # a new level (or a reset) started this turn
            _ltm = level_transition_messages[-1]
            if _ltm is last_restart_msg[0]:
                level_start_text = (level_pin_base + f"\n\n[Attempt {attempt_no} of level {current_level} started from the "
                                    "level's initial board (the board above).]")
            else:
                level_pin_base = _pin_base(_ltm)
                level_start_text = level_pin_base
            level_start_board = compaction_level_start_board.copy()

        # --- Context trimming (CONTEXT_TRIM): drop the oldest exchanges, no summary ---
        if CONTEXT_TRIM and needs_compaction:
            _do_trim(max(total_tokens, conv_tokens_est), CONTEXT_TRIM_TARGET_TOKENS, 2, "post-turn")
            needs_compaction = False

        # --- Post-turn compaction (commands already executed) ---
        if needs_compaction and compaction_limit > 0:
            if deadline and wall_time() >= deadline:
                finalize_run(
                    kind="deadline",
                    title="Deadline reached (pre-compaction)",
                    status_line=f"Deadline reached before compaction after {len(actions_taken)} actions.",
                    reasoning_text="",
                    results_lines=["Deadline reached; skipping compaction. Finalizing."],
                )
                return

            compaction_count += 1
            if not quiet:
                log_tag("COMPACT", f"Context at {total_tokens} tokens, limit {compaction_limit}. Compacting (#{compaction_count})...", Color.YELLOW, bold=True)
            else:
                _sync_print(f"[{game_id_short}] COMPACT: {total_tokens} tokens >= {compaction_limit} limit. Compacting (#{compaction_count})...")

            # Keep only the last few turn-pairs
            keep_entries = COMPACTION_KEEP_TURNS * 2
            saved_tail: list[dict[str, Any]] = []
            if len(conversation) > keep_entries:
                tail_start = len(conversation) - keep_entries
                while tail_start > 0 and conversation[tail_start].get("role") != "user":
                    tail_start -= 1
                saved_tail = conversation[tail_start:]
            else:
                saved_tail = list(conversation)

            _compact_remaining = None
            if deadline:
                _compact_remaining = deadline - wall_time()
                if _compact_remaining <= 30:
                    if not quiet:
                        log_tag("COMPACT", "Not enough time for compaction, skipping.", Color.YELLOW)
                    else:
                        _sync_print(f"[{game_id_short}] COMPACT: skipping (only {_compact_remaining:.0f}s remaining)")
                    needs_compaction = False
                    continue

            compact_request = (
                "CONTEXT COMPACTION — provide a factual handover summary.\n\n"
                "Summarize concisely:\n"
                "1. Game mechanics discovered (what each action does, how the board changes)\n"
                "2. Win condition hypothesis and confidence level\n"
                "3. Strategies tried: what worked, what failed, what was inconclusive\n"
                "4. Current level objective and immediate next plan\n"
                "5. Key uncertainties remaining\n\n"
                "Do NOT include board grids or code (those are preserved separately).\n"
                "The tools are attached to this request only for consistency with the other turns; "
                "they will be provided again in every future turn, so do NOT describe or summarize them "
                "and do NOT call any of them.\n"
                "Do NOT attempt any tool calls or actions.\n"
                "Write only the summary text, as a plain message."
            )
            for update in level_transition_messages:
                if update.startswith("[GAME OVER / LEVEL RESET]"):
                    compact_request += "\n\nLatest environment update (after the last actions):\n" + update
            # Never replace history with an empty/error/non-summary response.
            summary_result = None
            for compact_attempt in range(3):
                retry_remaining = (deadline - wall_time()) if deadline else None
                if retry_remaining is not None and retry_remaining <= 30:
                    break
                retry_prompt = compact_request
                if compact_attempt:
                    retry_prompt += (
                        "\n\nThe previous attempt did not produce a usable handover. "
                        "Provide the actual summary in your final answer, after any thinking. "
                        "Use the five numbered sections above. State unknowns explicitly. "
                        "Do not merely acknowledge this request or output tool calls."
                    )
                try:
                    candidate = request_llm_response(
                        client, system_prompt, conversation, retry_prompt,
                        board_for_big_image=None,
                        board_for_small_image=None,
                        available_action_labels=action_labels,
                        quiet=quiet, quiet_label=game_id_short,
                        remaining_seconds=retry_remaining,
                        is_compaction=True,
                    )
                except Exception as exc:
                    _sync_print(f"[{game_id_short}] COMPACT attempt {compact_attempt + 1}/3 failed: {exc!r}")
                    continue
                # Count every returned response, including rejected summaries.
                replay.add_token_usage(candidate.usage)
                replay.llm_messages += 1
                summary_text = (candidate.text or "").strip()
                invalid_reason = _compaction_rejection_reason(candidate)
                if invalid_reason:
                    _sync_print(
                        f"[{game_id_short}] COMPACT attempt {compact_attempt + 1}/3 rejected: "
                        f"{invalid_reason}; text_chars={len(summary_text)}, "
                        f"reasoning_chars={len(candidate.reasoning_content or '')}"
                    )
                    continue
                summary_result = candidate
                break
            if summary_result is None:
                _sync_print(
                    f"[{game_id_short}] COMPACT: no usable summary; retaining original "
                    "history. Compaction remains enabled for the next turn."
                )
                continue
            else:
                ai_summary = summary_result.text.strip()
                if not quiet:
                    log_tag("COMPACT", f"Got summary ({len(ai_summary)} chars). Rebuilding context...", Color.GREEN, bold=True)
                else:
                    _sync_print(f"[{game_id_short}] COMPACT: summary received ({len(ai_summary)} chars), rebuilding context")

                # Build compacted conversation — use concise level history
                level_history_parts: list[str] = []
                for lvl, start_board, end_board in compaction_level_boards:
                    actions_for_level = [a for l, a in compaction_level_actions if l == lvl]
                    n_actions = len(actions_for_level[0]) if actions_for_level else 0
                    level_history_parts.append(
                        f"Level {lvl} completed in {n_actions} actions."
                    )
                level_history_text = "\n".join(level_history_parts) if level_history_parts else "No levels completed yet."

                code_status = ""
                if code_file.source:
                    code_status = f"\n\nCurrent virtual code file:\n```\n{code_file.source}\n```"

                new_conversation: list[dict[str, Any]] = [
                    {"role": "user", "content": (
                        f"[Game history — context was compacted (#{compaction_count})]\n\n"
                        f"Completed levels:\n{level_history_text}"
                        f"{code_status}"
                        f"\n\nActions taken total: {len(actions_taken)}"
                        f"\nCurrent level: {current_level}"
                        + ("\n\n" + _notes_text() if NOTE_CONFIDENCE else "")
                    )},
                    {"role": "assistant", "content": ai_summary},
                ]
                new_conversation.extend(saved_tail)
                conversation = new_conversation

                feedback_message = (
                    "Context was compacted. You are continuing from where you left off.\n"
                    "The current board is in the attached image.\n\n"
                    + _full_board_color_inventory_text(current_board)
                    + f"\n\nActions taken in current level: {len(actions_taken) - level_action_start_index}"
                    + f"\nTotal actions: {len(actions_taken)}"
                    + f"\nAvailable actions: {', '.join(action_labels)}"
                )
                pending_full_board_for_image = current_board.copy()

                replay.add_event(
                    board=current_board,
                    actions_taken=len(actions_taken),
                    current_level=current_level,
                    kind="compaction",
                    title=f"Context compacted (#{compaction_count})",
                    available_actions=action_labels,
                    message_text=f"Compaction #{compaction_count}: {len(ai_summary)} chars summary, "
                                 f"old={total_tokens} tokens, "
                                 f"retained_turns={len(saved_tail)}, "
                                 f"new_conv_len={len(conversation)}",
                    harness_feedback=f"Summary: {ai_summary[:2000]}",
                )

    finalize_run(
        kind="max_turns",
        title="Max turns reached",
        status_line=f"Max turns ({MAX_TURNS}) reached without completing level.",
        reasoning_text="",
        results_lines=[],
    )


def _env_reset_retry(env: Any, tries: int | None = None) -> Any:
    """v2.9: env.reset() with retries and backoff (a gateway call can fail transiently; repeating a RESET only restarts the
    level again). Returns the observation, or None when every try failed."""
    tries = max(1, int(tries or ENV_RESET_TRIES))
    last = None
    for i in range(tries):
        try:
            obs = env.reset()
            if obs is not None and getattr(obs, "frame", None):
                return obs
            last = "no observation"
        except Exception as exc:
            last = repr(exc)
        if i < tries - 1:
            _sync_print(f"[env] reset failed ({last}); retry {i + 2}/{tries}"[:300])
            _time.sleep(min(8.0, 2.0 ** i))
    _sync_print(f"[env] reset failed after {tries} tries: {last}"[:300])
    return None


def _execute_action_base(
    env: Any,
    action: CandidateAction,
    all_boards: list[np.ndarray],
    actions_taken: list[str],
    history: list[TransitionRecord],
    current_level: int,
    seen_colors: set[int],
    remaining_actions_indicator_rect: tuple[int, int, int, int] | None,
    canonical_seen_action_index: dict[bytes, int],
    *,
    quiet: bool = False,
) -> tuple[str, str, int]:
    """Execute a single action, update state, and return (text, outcome, level_after)."""
    before = all_boards[-1].copy()
    action_enum, action_data = candidate_to_env_call(action)
    receipt = None
    if MINIMAL_HARNESS:
        receipt = {"action": action.as_text(), "status": "issued; outcome not yet confirmed",
                   "level": current_level, "action_number": len(actions_taken) + 1}
        _ext.ctx().action_ledger.append(receipt)
        _ext.ctx().action_coverage.issued(action, before, board_known=not _ext.ctx().after_env_error)
    try:
        obs = env.step(action_enum, data=action_data)
    except Exception as exc:   # v2.9: report it and carry on (it used to end the game)
        _ext.ctx().env_step_failed = True
        _ext.ctx().after_env_error = True
        _sync_print(f"[env] step {action.as_text()} raised {exc!r}"[:300])
        return (f"{action.as_text()}: the environment returned an error ({type(exc).__name__}); the action may or may not have "
                "been executed. The board shown is the last one received; your next action will show the true state.",
                "continue", current_level)
    if obs is not None and not obs.frame and getattr(obs, "state", None) == GameState.GAME_OVER:
        if receipt is not None:
            receipt.update(status="game_over; no board returned")
        # v2.10: after an environment error the game may already be over (the fatal step was executed)
        _ext.ctx().after_env_error = False
        return (f"{action.as_text()}: the environment reports that the game is over (an earlier action that returned an "
                "error was probably executed and ended the attempt).", "game_over", current_level)
    if obs is None or not obs.frame:
        _ext.ctx().env_step_failed = True
        _ext.ctx().after_env_error = True
        return (f"{action.as_text()}: the environment returned no observation; the action may or may not have been executed. "
                "The board shown is the last one received; your next action will show the true state.",
                "continue", current_level)

    after = observation_board(obs)
    frames = list(getattr(obs, "frame", []) or [])
    _ext.record_frames(before, frames, history_index=len(history))
    anim_frames, anim_bbox = animation_metadata_from_frames(before, frames)

    level_after = get_current_level(env, obs=obs)
    if _HAS_GAME_STATE_WIN and obs.state == GameState.WIN:
        outcome = "win"
    elif level_after > current_level:
        outcome = "level_complete"
    elif obs.state == GameState.GAME_OVER:
        outcome = "game_over"
    else:
        outcome = "continue"

    record = TransitionRecord(
        before=before, action=action, after=after.copy(),
        outcome=outcome, animation_frames=anim_frames, animation_bbox=anim_bbox,
    )
    if getattr(_ext.ctx(), "after_env_error", False):   # v2.10: may span two actions: not used by validation
        record.unreliable = True
        _ext.ctx().after_env_error = False
    history.append(record)
    all_boards.append(after.copy())
    actions_taken.append(action.as_text())
    if receipt is not None:
        receipt.update(status="executed", outcome=outcome, level_after=level_after,
                       changed_cells=(int(np.count_nonzero(before != after)) if before.shape == after.shape else None),
                       changed_outside_indicator=(int(np.count_nonzero((before != after) & ~rects_mask(
                           remaining_actions_indicator_rect, before.shape))) if before.shape == after.shape else None),
                       animation_frames=len(_ext.ctx().last_frames))
        seen_colors.update(_color_counts(after))
        return f"Action {action.as_text()}: {outcome}.", outcome, level_after

    if outcome == "win" or level_after > current_level:
        # v2.4: no before/after diff across a level change (different boards); the new level's board,
        # colour counts and object inventory are reported in the level-transition message.
        if not quiet:
            log_tag("ACTION", f"{action.as_text()} → {outcome}", Color.CYAN)
        if outcome == "win":
            return f"Action {action.as_text()}: the game was won.", outcome, level_after
        return (f"Action {action.as_text()}: level {current_level} completed; the new level's board and objects are "
                "described in the level transition below."), outcome, level_after

    changed_total = int(np.count_nonzero(to_uint8_board(before) != to_uint8_board(after)))

    # Split changes into indicator vs real when indicator rect(s) are set
    indicator_changed = 0
    _ind_rects = norm_rects(remaining_actions_indicator_rect)
    if _ind_rects and changed_total > 0:
        b, a = to_uint8_board(before), to_uint8_board(after)
        indicator_mask = rects_mask(_ind_rects, b.shape)
        diff_mask = b != a
        indicator_changed = int(np.count_nonzero(diff_mask & indicator_mask))
        real_changed = changed_total - indicator_changed

        if real_changed == 0:
            diff_text = "(no changes outside your marked indicator area)"
        else:
            before_masked = b.copy()
            after_masked = a.copy()
            after_masked[indicator_mask] = before_masked[indicator_mask]
            diff_text = _ext.bounded_board_diff(before_masked, after_masked)
    else:
        diff_text = _ext.bounded_board_diff(before, after)

    parts = [f"Action {action.as_text()}: {diff_text}"]
    if indicator_changed > 0:
        b_all, a_all = to_uint8_board(before), to_uint8_board(after)
        ind_coords = np.argwhere((b_all != a_all) & indicator_mask)
        indicator_diff_lines: list[str] = []
        if len(ind_coords) <= 12:
            for coord in ind_coords:
                r, c = int(coord[0]), int(coord[1])
                indicator_diff_lines.append(
                    f"    [{r},{c}]: {int(b_all[r, c])} -> {int(a_all[r, c])}"
                )
        indicator_diff_text = "\n".join(indicator_diff_lines)
        indicator_note = (
            f"  Also, {indicator_changed} cell(s) changed in the area you currently have marked "
            f"as the remaining-actions indicator "
            f"{rects_text(_ind_rects)}"
        )
        if indicator_diff_text:
            indicator_note += f":\n{indicator_diff_text}"
        else:
            indicator_note += "."
        if indicator_changed > 6:
            indicator_note += (
                "\n  (Consider whether this many cells changing in the remaining-actions "
                "indicator area is normal, or if this is because of some game transition, "
                "or whether the remaining-actions indicator area was marked correctly.)"
            )
        parts.append(indicator_note)
    if changed_total > 0:
        color_change_lines = _format_color_change_lines(before, after)
        if color_change_lines:
            parts.append("  Color count changes (value,hex,name,-away,+into):")
            parts.extend(f"    {line}" for line in color_change_lines)
    after_colors = set(_color_counts(after).keys())
    new_colors = sorted(after_colors - seen_colors)
    if new_colors:
        parts.append("  New color(s) not seen before in this game:")
        parts.extend(f"    {line}" for line in _format_color_lines(board=after, colors=new_colors))
        seen_colors.update(new_colors)
    if level_after > current_level:
        parts.append(f"  Level advanced: {current_level} -> {level_after}")
    if outcome != "continue":
        parts.append(f"  Outcome: {outcome}")

    if changed_total > 0:
        canonical_before = _canonical_board_key(before, remaining_actions_indicator_rect)
        canonical_after = _canonical_board_key(after, remaining_actions_indicator_rect)
        if canonical_after != canonical_before:
            previous_idx = canonical_seen_action_index.get(canonical_after)
            if previous_idx is not None:
                actions_ago = len(actions_taken) - previous_idx
                parts.append(
                    "  Barring the area you marked as remaining-actions indicator "
                    f"({_remaining_actions_indicator_rect_text(remaining_actions_indicator_rect)}), "
                    f"the current board looks exactly the same as it did {actions_ago} actions ago."
                )
        canonical_seen_action_index[canonical_after] = len(actions_taken)

    if not quiet:
        log_tag("ACTION", f"{action.as_text()} → {outcome}, {changed_total} cells changed", Color.CYAN)
    return "\n".join(parts), outcome, level_after


def _execute_action(env, action, all_boards, actions_taken, history, current_level, seen_colors,
                    remaining_actions_indicator_rect, canonical_seen_action_index, *, quiet=False):
    return _ext.execute_action_wrapped(_execute_action_base, env, action, all_boards, actions_taken, history,
                                       current_level, seen_colors, remaining_actions_indicator_rect,
                                       canonical_seen_action_index, quiet=quiet)


if __name__ == "__main__":
    main()

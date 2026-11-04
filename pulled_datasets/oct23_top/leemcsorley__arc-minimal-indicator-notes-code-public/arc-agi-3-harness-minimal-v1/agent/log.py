from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, TYPE_CHECKING

from .board import to_uint8_board
from .config import BOARD_COLOR_MAP_HEX, BOARD_SHAPE, ENABLE_REPLAYS, PRINT_COLORED_BOARDS, USE_COLOR_LOGS

if TYPE_CHECKING:
    from .llm import TokenUsage


ANSI_ENABLED = USE_COLOR_LOGS and sys.stdout.isatty()


class Color:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"


def colorize(text: str, color: str, *, bold: bool = False, dim: bool = False) -> str:
    if not ANSI_ENABLED:
        return text
    prefix = ""
    if bold:
        prefix += Color.BOLD
    if dim:
        prefix += Color.DIM
    prefix += color
    return f"{prefix}{text}{Color.RESET}"


def log_tag(tag: str, message: str, color: str, *, bold: bool = False) -> None:
    print(f"{colorize(f'[{tag}]', color, bold=bold)} {message}")


def short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def stable_json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _coerce_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump(mode="json")
        except Exception:
            try:
                dumped = value.model_dump()
            except Exception:
                return {}
        return dumped if isinstance(dumped, dict) else {}
    return {}


def _hex_rgba_to_rgb(hex_rgba: str) -> tuple[int, int, int]:
    clean = hex_rgba.strip().lstrip("#")
    if len(clean) < 6:
        return (255, 255, 255)
    return (int(clean[0:2], 16), int(clean[2:4], 16), int(clean[4:6], 16))


def _colored_cell(value: int) -> str:
    text = f"{value:2d}"
    if not (ANSI_ENABLED and PRINT_COLORED_BOARDS):
        return text
    rgb_hex = BOARD_COLOR_MAP_HEX.get(int(value))
    if rgb_hex is None:
        return text
    r, g, b = _hex_rgba_to_rgb(rgb_hex)
    lum = (0.299 * r) + (0.587 * g) + (0.114 * b)
    fg_r, fg_g, fg_b = ((0, 0, 0) if lum >= 140 else (255, 255, 255))
    return (
        f"\033[48;2;{r};{g};{b}m\033[38;2;{fg_r};{fg_g};{fg_b}m{text}{Color.RESET}"
    )


def print_board(title: str, board: Any) -> None:
    print(f"{title}:")
    arr = to_uint8_board(board)
    if ANSI_ENABLED and PRINT_COLORED_BOARDS:
        print("\n".join("".join(_colored_cell(int(v)) for v in row) for row in arr))
    else:
        print("\n".join("".join(str(int(v)) for v in row) for row in arr))


def _sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())


def replay_game_folder(game_id: str) -> str:
    parts = game_id.split("-", 1)
    base = re.sub(r"(?<=[A-Za-z])(?=[0-9])", "_", parts[0])
    return f"{base}-{parts[1]}" if len(parts) == 2 else base


def board_to_compact_row_csv(board: Any) -> list[str]:
    """Serialize a board into one CSV line per row for readable replays."""
    arr = to_uint8_board(board)
    return [",".join(str(int(cell)) for cell in row) for row in arr]


def _truncate_text(text: str, *, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


def _reasoning_summary_from_message(message_text: Any) -> str:
    if not isinstance(message_text, str):
        return ""
    lines = message_text.replace("\r", "").splitlines()
    in_reasoning = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith("reasoning:"):
            in_reasoning = True
            continue
        if in_reasoning:
            if stripped.lower().startswith("feedback:"):
                break
            if stripped.startswith("-"):
                return _truncate_text(stripped.lstrip("- ").strip(), max_chars=200)
            return _truncate_text(stripped, max_chars=200)
    for line in lines:
        stripped = line.strip()
        if stripped:
            return _truncate_text(stripped, max_chars=200)
    return ""


REPLAY_WRITE_INTERVAL_SECONDS = 20.0   # the replay HTML is rewritten at most this often (and always at the end)


def board_to_hex_rows(board: Any) -> list[str]:
    """Compact board encoding for the extra replay boards (per-action boards, win frame, death board...):
    one string per row, one hex digit per cell (values 0-15). Falls back to CSV rows for larger values."""
    arr = to_uint8_board(board)
    if arr.size and int(arr.max()) > 15:
        return board_to_compact_row_csv(arr)
    return ["".join("0123456789abcdef"[int(v)] for v in row) for row in arr]


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


@dataclass
class ReplayRun:
    game_id: str
    model_name: str
    thinking_effort: str
    initial_level: int
    run_dir: Path
    start_stamp: str
    shared_js_relpath: str
    current_html_path: Path | None = None
    llm_messages: int = 0
    latest_code: str = ""
    final_score: float | None = None
    scorecard_total_levels: int | None = None
    scorecard_total_actions: int | None = None
    token_input_miss_tokens: int = 0
    token_input_cached_tokens: int = 0
    token_output_tokens: int = 0
    # v2.5-log: run-level fields written at the top of the replay JSON (harness version, baselines, per-level scores...)
    meta: dict[str, Any] = field(default_factory=dict)
    # Events are serialized once when added (the HTML is rebuilt from these strings) and only a small index is kept.
    events: list[dict[str, Any]] = field(default_factory=list)       # index entries (kind/title/counters), not full events
    _event_json: list[str] = field(default_factory=list)
    _last_write: float = 0.0
    _last_counts: tuple[int, int] = (0, 1)

    @property
    def token_input_total_tokens(self) -> int:
        return self.token_input_miss_tokens + self.token_input_cached_tokens

    def add_token_usage(self, usage: "TokenUsage") -> None:
        self.token_input_miss_tokens += usage.input_miss_tokens
        self.token_input_cached_tokens += usage.input_cached_tokens
        self.token_output_tokens += usage.output_tokens

    def _event_index(self) -> list[dict[str, Any]]:
        return [dict(e, step=i) for i, e in enumerate(self.events, start=1)]

    def _filename(self, actions_taken: int, current_level: int) -> str:
        levels = max(0, current_level - self.initial_level)
        tag = _sanitize(f"{self.model_name}-{self.thinking_effort}")
        score_suffix = ""
        if self.final_score is not None:
            score_suffix = f"_{_sanitize(f'{self.final_score:.1f}')}"
        return f"{self.start_stamp}_{tag}_{actions_taken}a_{self.llm_messages}m_L{levels}{score_suffix}.html"

    def _write(self, actions_taken: int, current_level: int) -> None:
        if not ENABLE_REPLAYS:
            return
        import time as _time
        self._last_write = _time.monotonic()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        name = self._filename(actions_taken, current_level)
        path = self.run_dir / name
        payload_obj = {
            "game_id": self.game_id, "model_name": self.model_name,
            "thinking_effort": self.thinking_effort, "start_stamp": self.start_stamp,
            "final_score": self.final_score,
            "scorecard_total_levels": self.scorecard_total_levels,
            "scorecard_total_actions": self.scorecard_total_actions,
            "token_usage_totals": {
                "input_cache_miss_tokens": self.token_input_miss_tokens,
                "input_cached_tokens": self.token_input_cached_tokens,
                "input_total_tokens": self.token_input_total_tokens,
                "output_tokens": self.token_output_tokens,
            },
            "board_shape": list(BOARD_SHAPE),
            "board_encoding": "row_csv_v1",
            "extra_board_encoding": "hex_rows_v1",   # per-action boards, win_frame, death_board, ... (one hex digit per cell)
            "color_map_hex": {str(k): v for k, v in BOARD_COLOR_MAP_HEX.items()},
            **self.meta,
            "event_index": self._event_index(),
        }
        head = json.dumps(payload_obj, ensure_ascii=False, indent=1, default=_json_default).replace("</", "<\\/")
        payload = head[:-1].rstrip() + ',\n "events": [\n' + ",\n".join(self._event_json) + "\n ]\n}"
        game_short = self.game_id.split("-", 1)[0] if self.game_id else "ARC"
        html = (
            "<!doctype html>\n"
            '<html lang="en">\n'
            "<head>\n"
            '  <meta charset="utf-8" />\n'
            '  <meta name="viewport" content="width=device-width,initial-scale=1" />\n'
            f"  <title>{game_short} — ARC Replay</title>\n"
            "</head>\n"
            "<body>\n"
            '  <div id="arc-replay-root"></div>\n'
            '  <script id="arc-replay-data" type="application/json">\n'
            f"{payload}\n"
            "  </script>\n"
            f'  <script src="{self.shared_js_relpath}"></script>\n'
            "</body>\n"
            "</html>\n"
        )
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(html, encoding="utf-8")
        tmp.replace(path)
        if self.current_html_path and self.current_html_path != path and self.current_html_path.exists():
            self.current_html_path.unlink()
        self.current_html_path = path

    def flush(self) -> None:
        """Write the replay now (the end of a game, or an exception)."""
        if ENABLE_REPLAYS and self._event_json:
            self._write(*self._last_counts)

    def add_event(self, *, board: Any, actions_taken: int, current_level: int, force_write: bool = False,
                  **kwargs: Any) -> None:
        if not ENABLE_REPLAYS:
            return   # scored rerun: no replay, keep memory flat
        event = {"board": board_to_compact_row_csv(board), "actions_taken": actions_taken,
                 "llm_messages": self.llm_messages, "latest_code": self.latest_code,
                 "levels_completed": max(0, current_level - self.initial_level), **kwargs}
        self._event_json.append(json.dumps(event, ensure_ascii=False, default=_json_default).replace("</", "<\\/"))
        entry: dict[str, Any] = {
            "kind": str(event.get("kind", "")),
            "title": str(event.get("title", "")),
            "actions_taken": _coerce_int(event.get("actions_taken", 0)),
            "llm_messages": _coerce_int(event.get("llm_messages", 0)),
            "levels_completed": _coerce_int(event.get("levels_completed", 0)),
        }
        summary = _reasoning_summary_from_message(event.get("message_text", ""))
        if summary:
            entry["reasoning_summary"] = summary
        self.events.append(entry)
        self._last_counts = (actions_taken, current_level)
        import time as _time
        if force_write or len(self._event_json) <= 2 or _time.monotonic() - self._last_write >= REPLAY_WRITE_INTERVAL_SECONDS:
            self._write(actions_taken, current_level)

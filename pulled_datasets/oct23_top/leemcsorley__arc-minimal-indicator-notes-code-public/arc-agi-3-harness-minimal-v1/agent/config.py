from __future__ import annotations

import os
from pathlib import Path
import sys

from arc_agi import OperationMode


# ============================================================
# Kaggle environment detection
# ============================================================
IS_KAGGLE = Path("/kaggle/working").exists()
KAGGLE_QUICK_TEST = IS_KAGGLE and os.environ.get("QUICK_TEST", "0") == "1"
KAGGLE_GATEWAY_URL = "http://gateway:8001"

# ============================================================
# Configuration
# ============================================================
# Local runs: a single game id (str) or a list of ids played in parallel. Overridable with --game_id.
GAME_ID: str | list[str] = [
    "ls20", "vc33", "ft09",
    "ar25", "bp35", "cd82", "cn04", "dc22", "g50t",
    "ka59", "lf52", "lp85", "m0r0", "r11l", "re86",
    "s5i5", "sb26", "sc25", "sk48", "sp80", "su15",
    "tn36", "tr87", "tu93", "wa30",
]
SEED = 0
TARGET_LEVEL = 9  # 1-based: stop once this level index is reached
MEMORY_DEBUG = False  # Print RSS at key points each turn
CHEAT_MODE = False  # Attach full game source code to the AI

# Local runs (overridable with --model_name / --model_thinking_effort / --provider; see model_provider()
# for the provider inferred from the name). On Kaggle both come from KAGGLE_MODEL_NAME / KAGGLE_THINKING_EFFORT.
MODEL_NAME = "deepseek-v4-flash-vision-exp"
MODEL_THINKING_EFFORT = "high"
ADAPTIVE_REASONING = os.environ.get("V2_ADAPTIVE_REASONING", "0") == "1"
TIERED_REASONING = os.environ.get("V2_TIERED_REASONING", "0") == "1"

MAX_TURNS = 2400
USE_LLM_CACHE = True

ROOT_DIR = Path("/kaggle/working") if IS_KAGGLE else Path(__file__).resolve().parents[2]
LLM_CACHE_DIR = ROOT_DIR / "llm_cache"
REPLAYS_DIR = ROOT_DIR / "replays"
REPLAY_SHARED_JS_NAME = "replay_viewer.js"

# Local vLLM server (used on Kaggle)
LOCAL_VLLM_BASE_URL = "http://localhost:8000/v1"

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
SAMBANOVA_BASE_URL = "https://api.sambanova.ai/v1"
GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
TOGETHER_BASE_URL = "https://api.together.ai/v1"
DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"
AKASHML_BASE_URL = "https://api.akashml.com/v1"

# path tool: BFS over the model's next_state
MAX_DISTINCT_STATES = 50_000
MAX_ACTION_DEPTH = 150
BFS_PROGRESS_EVERY = 1000
BFS_TIME_LIMIT_SECONDS = 60.0  # a search is abandoned (reported as "no path") after this much wall time

# Context compaction: when input_total_tokens exceeds the limit for the model family,
# the harness asks the model to summarize, then rebuilds a shorter conversation.
# For "local" (Kaggle vLLM), limit is computed dynamically based on concurrency —
# see Kaggle overrides below. The static 100k here is a fallback.
COMPACTION_TOKEN_LIMITS: dict[str, int] = {
    "deepseek": 800_000,
    "gpt": 900_000,
    "qwen3.8": 200_000,
    "qwen": 800_000,
    "local": 100_000,
}
COMPACTION_KEEP_TURNS: int = 3  # Number of recent turn-pairs to preserve after compaction

OPERATION_MODE = OperationMode.OFFLINE
ENVIRONMENTS_DIR = "environment_files"

# Kaggle-specific game settings (overridden below when IS_KAGGLE)
RESET_ON_GAME_OVER: bool = False
MAX_CONCURRENT_REQUESTS: int | None = None  # None = unlimited (local/cloud); set on Kaggle
DEADLINE_SECONDS: float | None = None  # None = no deadline; set on Kaggle
ENABLE_REPLAYS: bool = True  # Write HTML replay files

USE_COLOR_LOGS = True
PRINT_COLORED_BOARDS = True

BOARD_SHAPE = (64, 64)
BOARD_MAX_SHAPE = (64, 64)  # Maximum allowed board dimensions
BOARD_CELLS = BOARD_SHAPE[0] * BOARD_SHAPE[1]
VALID_OUTCOMES = {"continue", "level_complete", "game_over", "win"}
MAX_RUN_OUTPUT_CHARS = 8_000   # v2.10: was 12000; longer print output is cut in the middle (start and end kept)
ADD_IMAGE_INPUT = True
# Hard cap on images per request (the vLLM server rejects prompts above --limit-mm-per-prompt). With the small,
# diff and game-over images kept in history the count grows by ~2 per turn until compaction, so when the cap would
# be exceeded the oldest per-turn images in the history are replaced by a short text note (dropping down to
# cap - MAX_IMAGES_DROP_SLACK at once, so the cached prefix is only invalidated once every few turns). 0 = off.
MAX_IMAGES_PER_PROMPT = int(os.environ.get("MAX_IMAGES_PER_PROMPT", os.environ.get("LH_IMAGES_PER_PROMPT", "0")))
MAX_IMAGES_DROP_SLACK = 16
# Keep the small per-turn images in the conversation history (True keeps the prompt prefix stable across turns,
# which is what the vLLM prefix cache needs; False would rewrite earlier user messages every turn).
PERSIST_SMALL_IMAGES_IN_HISTORY = True
MAX_ACTIONS_PER_MESSAGE: int | None = 4  # Cap on explicit action lines per `actions` tool call (None = uncapped).
NO_CHANGE_WARNING_TURNS: int = 5  # Warn model after this many consecutive action-turns with no real change (0 to disable)
USE_TOOL_CALLING: bool = True  # The system prompt describes the tools; command blocks (```actions ...) are a legacy fallback.

# ------------------------------------------------------------
# Harness v2: objects, frame diffs, act() inside run, guards, extra images
# ------------------------------------------------------------
PROMPT_VARIANT = os.environ.get("PROMPT_VARIANT", "objects").strip().lower()  # "objects" (objects-first) or "model" (model-first)
MAX_ACT_ACTIONS = int(os.environ.get("V2_MAX_ACT_ACTIONS", "12"))  # actions one run() call may execute via act()
RECT_CSV_MAX_CELLS = int(os.environ.get("V2_RECT_CSV_MAX_CELLS", "100"))  # changed rectangles are printed as CSV only up to this total area
OBJ_CHANGES_CAP = int(os.environ.get("V2_OBJ_CHANGES_CAP", "10"))  # object-change lines per action before "... and N more"
DIFF_IMAGE = os.environ.get("V2_DIFF_IMAGE", "1") == "1"  # change image (unchanged cells dimmed) on turns where the board changed
DEATH_IMAGE = os.environ.get("V2_DEATH_IMAGE", "1") == "1"  # board at the moment of game over, attached after a reset
FEEDBACK_CHAR_CAP = int(os.environ.get("V2_FEEDBACK_CHAR_CAP", "16000"))   # v2.10: 16000 (was 8000); harness lines are never cut, run output shrinks first  # per-turn harness feedback is truncated beyond this (v2.6: 12000 with the tracker lines, 8000 without) ...
FEEDBACK_CHAR_CAP_LEVEL_TRANSITION = 40_000   # v2.10: 40000 (was 24000; notes can reach 20000)  # ... except on turns that carry a level start / reset message (full board + inventory)
LEVEL_WEIGHTED_ALLOCATION = os.environ.get("LEVEL_WEIGHTED_ALLOCATION", "0") == "1"  # Kaggle: level_alloc.LevelWeightedAllocator instead of equal per-game budgets
ANIM_MAX_FRAMES = 300  # intermediate animation frames kept for anim_frames / anim_diff() (per action)
ANIM_AUTO_DIFF_LINES = 12  # animation steps shown automatically when the board ended unchanged but the animation did not

# Context management. True: when a turn ends above CONTEXT_TRIM_TRIGGER_TOKENS, the oldest exchanges are dropped
# (no summary) until the estimate is below CONTEXT_TRIM_TARGET_TOKENS; the system prompt and tools are always sent,
# and the current level's start message (image + inventory) is pinned right after them.
# False: the older behaviour - the model writes a summary that replaces the history (COMPACTION_TOKEN_LIMITS).
CONTEXT_TRIM = os.environ.get("CONTEXT_TRIM", "1") == "1"
CONTEXT_TRIM_TRIGGER_TOKENS = int(os.environ.get("CONTEXT_TRIM_TRIGGER_TOKENS", "95000"))
CONTEXT_TRIM_TARGET_TOKENS = int(os.environ.get("CONTEXT_TRIM_TARGET_TOKENS", "50000"))
# v2.3 robustness: a failed request is retried (backoff, server health check) for up to LLM_RETRY_MAX_SECONDS
# instead of ending the game; a worker waits up to SERVER_WAIT_SECONDS for the server before starting a game;
# the history is trimmed BEFORE a request whose estimated prompt exceeds CONTEXT_PRESEND_TRIM_TOKENS.
HARNESS_VERSION = "v2.24"   # earned 12K/8K reasoning tiers + sticky final-15% 4K tier
EPHEMERAL_REASONING = os.environ.get("V2_EPHEMERAL_REASONING", "0") == "1"
NOTE_CONFIDENCE = os.environ.get("V2_NOTE_CONFIDENCE", "0") == "1"
# v2.6: layered object tracker = Python port of the teammate's PEEL + TRACK (arc_object_tracker.html, first pass
# only: causal, no look-ahead). Every board is explained as edits of the previous scene (moves, turns, recolours,
# appear/disappear, splits, merges, covering) with persistent object ids; each action's feedback gets its lines.
# Runs in one worker process per game (scene_worker.py); numba kernels compiled once and cached (prewarm).
# v2.6.5: the tracker's scene is the object layer the model sees: the level-start inventory, objects() / obj() / changes()
# in run and next_state over objects all use the scene (persistent ids, shapes completed behind other objects, multi-colour
# objects); its lines replace the old "Object changes" block (ordered by importance, capped by characters, rotations only
# when exact, jumps / identical swaps flagged as ambiguous, a shift of most of the board reported once).
# With the flag off everything is exactly v2.5-log.
LAYERED_TRACKER = os.environ.get("V2_LAYERED_TRACKER", "0") == "1"   # v2.7: off by default (= v2.5-log behaviour)
LAYERED_TRACKER_LINES = int(os.environ.get("V2_LAYERED_TRACKER_LINES", "20"))           # tracker lines per action (most important first), then "... and N more"
LAYERED_TRACKER_CHARS = int(os.environ.get("V2_LAYERED_TRACKER_CHARS", "1500"))         # ... and at most this many characters per action
LAYERED_TRACKER_TIMEOUT = float(os.environ.get("V2_LAYERED_TRACKER_TIMEOUT", "30"))     # s for one board; longer = tracker off for that game
LAYERED_TRACKER_GAME_BUDGET = float(os.environ.get("V2_LAYERED_TRACKER_GAME_BUDGET", "300"))  # s of waiting per game, then off for that game
# (the worker itself skips a board after 15 s or with more than 1200 separate pieces: ARC_SCENE_SOFT_LIMIT / ARC_SCENE_MAX_PIECES)
LAYERED_TRACKER_PREWARM_TIMEOUT = float(os.environ.get("V2_LAYERED_TRACKER_PREWARM_TIMEOUT", "180"))  # s for compiling the kernels once (else the tracker stays off)
# v2.7 priority scheduler (scheduler.py): all games start at once, at most MAX_CONCURRENT_REQUESTS play at a time; a game
# gives its slot up at a context trim (or after PRIORITY_STRETCH_CAP_SECONDS) and the waiting game with the highest
# priority takes it: never-played games first, then L/(N(N+1)/2) * 0.5**((output tokens on the level / tau)**2).
# Every game shares the run's deadline (no per-game budget, no "[Budget] Time remaining" line). Off = equal budgets.
PRIORITY_SCHEDULER = os.environ.get("PRIORITY_SCHEDULER", "0") == "1"
PRIORITY_TAU_TOKENS = float(os.environ.get("PRIORITY_TAU_TOKENS", "130000"))           # output tokens on a level -> priority x0.5
PRIORITY_STRETCH_CAP_SECONDS = float(os.environ.get("PRIORITY_STRETCH_CAP_SECONDS", "1200"))  # max slot time without a trim while others wait
PRIORITY_DEFAULT_LEVELS = int(os.environ.get("PRIORITY_DEFAULT_LEVELS", "8"))          # level count if the game does not report it
# v2.8: a turn after which no action was executed (board unchanged) gets a one-line note instead of the small board
# image and the level / action counters (unchanged since the previous turn). The image comes back after a context trim.
LEAN_NOACTION_TURNS = os.environ.get("V2_LEAN_NOACTION_TURNS", "0") == "1"
# v2.8: one-line descriptions of SPACE and UNDO in the system prompt (only for games that offer them).
PROMPT_ADDITIONS = os.environ.get("V2_PROMPT_ADDITIONS", "0") == "1"
# v2.9: after a turn that executed no action, once ACT_NUDGE_TOKENS output tokens have been generated since the last
# executed action (or since the last nudge), the next message ends with a short "act now" line (like M1's commit nudge).
ACT_NUDGE = os.environ.get("V2_ACT_NUDGE", "0") == "1"
ACT_NUDGE_TOKENS = int(os.environ.get("V2_ACT_NUDGE_TOKENS", "2048"))
# v2.9: the image of the winning moment (the last frame of the winning action, before the level switched) is attached to
# the next level's start message.
WIN_FRAME_IMAGE = os.environ.get("V2_WIN_FRAME_IMAGE", "0") == "1"
# v2.9 prompt changes: a paragraph on keeping observations as specific as the evidence (and re-checking the mechanics before
# the goal), the solution-length sentence (10-60 actions instead of 5-50), and the post-win reflection wording.
PROMPT_V29 = os.environ.get("V2_PROMPT_V29", "0") == "1"
# v2.9 (always on, bug fixes): the winning-action effect is taken from the last frame of the winning animation (was the
# first frame that differed, often only the step counter); env.reset() is retried ENV_RESET_TRIES times; an error from
# env.step() is reported to the model instead of ending the game; a new colour also refuses later act() calls in the same
# run call, and the colour memory survives level restarts; a quote-only note text ('' or "") deletes the note.
ENV_RESET_TRIES = int(os.environ.get("V2_ENV_RESET_TRIES", "4"))
WIN_TRANSITION_FRACTION = float(os.environ.get("V2_WIN_TRANSITION_FRACTION", "0.15"))  # one frame step changing more than this share of the board = level-transition animation
# v2.8 (always on, bug fixes): the allowed modules (now incl. time) are pre-imported in run and in the code file, the
# sandbox recursion limit is raised, and a refused import names the module and lists the allowed ones.
SANDBOX_RECURSION_LIMIT = int(os.environ.get("V2_SANDBOX_RECURSION_LIMIT", "10000"))
MAX_OUTPUT_TOKENS = int(os.environ.get("V2_MAX_OUTPUT_TOKENS", "12000"))  # per-request output cap (thinking included)
# act() guard after a no-op (v2.5). "known": refuse only an action that was already a no-op (or fatal) in this exact
# board state, unless act(..., repeat=True). "stale": the v2.2-v2.4 behaviour (any further act() in the same run call
# after a no-op is refused unless repeat=True).
ACT_GUARD_MODE = os.environ.get("V2_ACT_GUARD_MODE", "known").strip().lower()
NOTES_CHAR_CAP = int(os.environ.get("V2_NOTES_CHAR_CAP", "20000"))   # total size of the model's pinned notes (v2.9: 20000, was 6000)
LLM_RETRY_MAX_SECONDS = float(os.environ.get("LLM_RETRY_MAX_SECONDS", "900"))
SERVER_WAIT_SECONDS = float(os.environ.get("SERVER_WAIT_SECONDS", "900"))
CONTEXT_PRESEND_TRIM_TOKENS = int(os.environ.get("CONTEXT_PRESEND_TRIM_TOKENS", "110000"))
BOARD_IMAGE_SMALL_SCALE_DIVISOR = 4
BOARD_IMAGE_SMALL_MIN_SCALE = 8  # per-turn images (small board, change image, game-over image) are at least 512 px
BOARD_IMAGE_SCALE_GPT = 32
BOARD_IMAGE_SCALE_QWEN = 16  # level-start picture 1024 px = 1024 vision tokens; keeps 120 images within the 128k context on the vLLM server
BOARD_IMAGE_SCALE_GEMMA = 24
BOARD_IMAGE_SCALE_KIMI = 28
BOARD_IMAGE_SCALE_MINIMAX = 28
BOARD_IMAGE_SCALE_STEP = 28
BOARD_IMAGE_SCALE_INKLING = 40
# DeepSeek vision tokenization currently normalizes images to roughly 800x800 and
# upsizes very small inputs below roughly 384x384.
BOARD_IMAGE_TARGET_DIM_DEEPSEEK_VISION = 800
BOARD_IMAGE_MIN_DIM_DEEPSEEK_VISION = 384
BOARD_IMAGE_SCALE_DEEPSEEK_VISION = max(
    1,
    (BOARD_IMAGE_TARGET_DIM_DEEPSEEK_VISION + max(BOARD_SHAPE) - 1) // max(BOARD_SHAPE),
)
BOARD_IMAGE_SMALL_MIN_SCALE_DEEPSEEK_VISION = max(
    1,
    (BOARD_IMAGE_MIN_DIM_DEEPSEEK_VISION + max(BOARD_SHAPE) - 1) // max(BOARD_SHAPE),
)
BOARD_IMAGE_PADDED_SHAPE_STEP = (72, 72)  # 64x64 board centered with striped border
if MAX_ACTIONS_PER_MESSAGE is not None and MAX_ACTIONS_PER_MESSAGE < 1:
    raise SystemExit("MAX_ACTIONS_PER_MESSAGE must be >= 1 when set.")

BOARD_COLOR_MAP_HEX: dict[int, str] = {
    0: "#FFFFFFFF", 1: "#CCCCCCFF", 2: "#999999FF", 3: "#666666FF",
    4: "#333333FF", 5: "#000000FF", 6: "#E53AA3FF", 7: "#FF7BCCFF",
    8: "#F93C31FF", 9: "#1E93FFFF", 10: "#88D8F1FF", 11: "#FFDC00FF",
    12: "#FF851BFF", 13: "#921231FF", 14: "#4FCC30FF", 15: "#A356D6FF",
}
BOARD_COLOR_NAME_BY_ID: dict[int, str] = {
    0: "White",
    1: "Off-white",
    2: "neutral Light",
    3: "neutral",
    4: "Off Black",
    5: "Black",
    6: "Magenta",
    7: "Magenta Light",
    8: "Red",
    9: "Blue",
    10: "Blue Light",
    11: "Yellow",
    12: "Orange",
    13: "Maroon",
    14: "Green",
    15: "Purple",
}

# Shared by the minimal prompt, Python namespace and measured colour descriptions.
# These name the rendered palette only, without assigning game roles.
MINIMAL_COLOR_NAMES: dict[int, str] = {
    0: "white", 1: "light grey", 2: "grey", 3: "dark grey",
    4: "very dark grey", 5: "black", 6: "magenta", 7: "pink",
    8: "red", 9: "blue", 10: "light blue", 11: "yellow",
    12: "orange", 13: "maroon", 14: "green", 15: "purple",
}

ACTION_LABEL_BY_ID = {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT", 5: "SPACE", 6: "CLICK", 7: "UNDO"}
ACTION_LABEL_ALIASES = {"W": "UP", "S": "DOWN", "A": "LEFT", "D": "RIGHT"}  # accepted silently, never shown to the model
RESET_LABEL = "RESET"   # harness-exposed level restart (env.reset()); action_id 0 = GameAction.RESET
ACTION_SYMBOL_BY_LABEL = {
    "UP": "↑", "LEFT": "←", "DOWN": "↓", "RIGHT": "→",
    "SPACE": "⎵", "CLICK": "🖱", "UNDO": "↺", "RESET": "⟲",
}


# Optional runtime overrides accepted via CLI:
#   --game_id <value>
#   --target_level <int>
#   --model_name <value>
#   --model_thinking_effort <value>
#   --provider <value>
#
# Equivalent key=value form is also accepted, e.g. model_name=gpt-5.4.
_CLI_ALLOWED_KEYS = {
    "game_id",
    "target_level",
    "model_name",
    "model_thinking_effort",
    "provider",
}


def _cli_usage() -> str:
    return (
        "Optional args (only these are allowed): "
        "--game_id, --target_level, --model_name, --model_thinking_effort, --provider"
    )


def _parse_cli_overrides(argv: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    i = 0
    while i < len(argv):
        raw = argv[i].strip()
        if not raw:
            i += 1
            continue

        key = ""
        value = ""

        if raw.startswith("--"):
            token = raw[2:]
            if "=" in token:
                key, value = token.split("=", 1)
                i += 1
            else:
                key = token
                if i + 1 >= len(argv):
                    raise SystemExit(f"Missing value for '{raw}'. {_cli_usage()}")
                value = argv[i + 1]
                i += 2
        elif "=" in raw:
            key, value = raw.split("=", 1)
            key = key.lstrip("-")
            i += 1
        else:
            raise SystemExit(f"Unsupported argument '{raw}'. {_cli_usage()}")

        key = key.strip()
        if key not in _CLI_ALLOWED_KEYS:
            raise SystemExit(f"Unknown argument '{key}'. {_cli_usage()}")
        overrides[key] = value
    return overrides


SUPPORTED_PROVIDER_NAMES = {
    "deepseek",
    "openai",
    "groq",
    "sambanova",
    "google",
    "together",
    "deepinfra",
    "akashml",
    "local",
}
_PROVIDER_ALIASES = {
    "deepseek": "deepseek",
    "openai": "openai",
    "groq": "groq",
    "sambanova": "sambanova",
    "samba": "sambanova",
    "google": "google",
    "google-ai-studio": "google",
    "googleaistudio": "google",
    "gemini": "google",
    "together": "together",
    "togetherai": "together",
    "together-ai": "together",
    "together ai": "together",
    "deepinfra": "deepinfra",
    "deep-infra": "deepinfra",
    "deep infra": "deepinfra",
    "akashml": "akashml",
    "akash": "akashml",
    "akash-ml": "akashml",
    "akash ml": "akashml",
    "local": "local",
    "vllm": "local",
}


def normalize_provider_name(raw_provider: str) -> str | None:
    key = raw_provider.strip().lower()
    if not key:
        return None
    resolved = _PROVIDER_ALIASES.get(key, key)
    return resolved if resolved in SUPPORTED_PROVIDER_NAMES else None


def _model_name_key(model_name: str) -> str:
    return model_name.strip().lower()


def model_is_deepseek_vision(model_name: str) -> bool:
    name = _model_name_key(model_name)
    return name.startswith("deepseek-") and "vision" in name


def model_provider(model_name: str, provider_override: str | None = None) -> str:
    if provider_override is not None:
        normalized_override = normalize_provider_name(provider_override)
        if normalized_override is None:
            raise ValueError(
                "Unsupported provider override. Expected one of: "
                + ", ".join(sorted(SUPPORTED_PROVIDER_NAMES))
            )
        return normalized_override

    name = _model_name_key(model_name)

    if name.startswith("deepseek-"):
        return "deepseek"
    if name.startswith("gpt-"):
        return "openai"
    if name.startswith("moonshotai/") or name.startswith("minimaxai/") or name.startswith("thinkingmachines/"):
        return "together"
    if name.startswith("stepfun-ai/"):
        return "deepinfra"
    if name == "qwen/qwen3.5-397b-a17b":
        return "deepinfra"
    if name == "qwen/qwen3.8-27b":
        return "akashml"
    if name.startswith("qwen/") or name.startswith("openai/"):
        return "groq"
    if name.startswith("gemma-"):
        # Default gemma provider is Google AI Studio (OpenAI-compatible endpoint).
        # To quickly revert, change this return value to "sambanova".
        return "google"
    raise ValueError(
        "Unsupported model prefix. Expected one of: "
        "deepseek-, gpt-, qwen/, openai/, gemma-, moonshotai/, minimaxai/, stepfun-ai/"
    )


def model_supports_image_input(model_name: str) -> bool:
    name = _model_name_key(model_name)
    return (
        model_is_deepseek_vision(model_name)
        or name.startswith("gpt-")
        or name.startswith("qwen/")
        or name.startswith("openai/")
        or name.startswith("gemma-")
        or name.startswith("moonshotai/")
        or name.startswith("minimaxai/")
        or name.startswith("stepfun-ai/")
        or name.startswith("thinkingmachines/")
        or IS_KAGGLE  # local vLLM with Qwen3.8 VLM
    )


def board_image_scale_for_model(model_name: str) -> int:
    name = _model_name_key(model_name)
    if model_is_deepseek_vision(model_name):
        return BOARD_IMAGE_SCALE_DEEPSEEK_VISION
    if name.startswith("gemma-"):
        return BOARD_IMAGE_SCALE_GEMMA
    if name.startswith("qwen/") or "qwen" in name:
        return BOARD_IMAGE_SCALE_QWEN
    if name.startswith("moonshotai/"):
        return BOARD_IMAGE_SCALE_KIMI
    if name.startswith("minimaxai/"):
        return BOARD_IMAGE_SCALE_MINIMAX
    if name.startswith("stepfun-ai/"):
        return BOARD_IMAGE_SCALE_STEP
    if name.startswith("thinkingmachines/"):
        return BOARD_IMAGE_SCALE_INKLING
    if IS_KAGGLE:
        return BOARD_IMAGE_SCALE_QWEN
    return BOARD_IMAGE_SCALE_GPT


def board_small_image_scale_for_model(model_name: str) -> int:
    base_scale = board_image_scale_for_model(model_name)
    small_scale = max(BOARD_IMAGE_SMALL_MIN_SCALE, base_scale // BOARD_IMAGE_SMALL_SCALE_DIVISOR)
    if model_is_deepseek_vision(model_name):
        # DeepSeek upsizes images below ~384x384 before inference anyway, so keep
        # at least this much native resolution to avoid blur.
        return max(BOARD_IMAGE_SMALL_MIN_SCALE_DEEPSEEK_VISION, small_scale)
    return small_scale


def model_uses_striped_image_padding(model_name: str) -> bool:
    return _model_name_key(model_name) == "stepfun-ai/step-3.7-flash"


_CLI_OVERRIDES = _parse_cli_overrides(sys.argv[1:]) if not IS_KAGGLE else {}
MODEL_PROVIDER_OVERRIDE: str | None = None
if "game_id" in _CLI_OVERRIDES:
    GAME_ID = _CLI_OVERRIDES["game_id"]  # CLI override always sets a single game (str)
if "target_level" in _CLI_OVERRIDES:
    try:
        TARGET_LEVEL = int(_CLI_OVERRIDES["target_level"])
    except ValueError as exc:
        raise SystemExit("target_level must be an integer.") from exc
if "model_name" in _CLI_OVERRIDES:
    MODEL_NAME = _CLI_OVERRIDES["model_name"]
if "model_thinking_effort" in _CLI_OVERRIDES:
    MODEL_THINKING_EFFORT = _CLI_OVERRIDES["model_thinking_effort"]
if "provider" in _CLI_OVERRIDES:
    normalized_provider = normalize_provider_name(_CLI_OVERRIDES["provider"])
    if normalized_provider is None:
        raise SystemExit(
            "provider must be one of: "
            + ", ".join(sorted(SUPPORTED_PROVIDER_NAMES))
        )
    MODEL_PROVIDER_OVERRIDE = normalized_provider

# ============================================================
# Kaggle overrides — applied after CLI so Kaggle always wins
# ============================================================
if IS_KAGGLE:
    MODEL_NAME = os.environ.get("KAGGLE_MODEL_NAME", "Qwen/Qwen3.8-Flash-Next-NVFP4")  # the notebook sets this to the served model name
    MODEL_THINKING_EFFORT = os.environ.get("KAGGLE_THINKING_EFFORT", "xhigh")  # Qwen3.8 template: xhigh (its default), medium, low
    MODEL_PROVIDER_OVERRIDE = "local"
    OPERATION_MODE = OperationMode.OFFLINE  # switched to COMPETITION at runtime if gateway is up
    USE_LLM_CACHE = False
    CHEAT_MODE = False
    RESET_ON_GAME_OVER = True
    TARGET_LEVEL = 999  # effectively unlimited — stop on WIN or deadline
    MAX_CONCURRENT_REQUESTS = int(os.environ.get("MAX_CONCURRENT_REQUESTS", "12"))
    DEADLINE_SECONDS = float(os.environ.get("DEADLINE_SECONDS", str(8 * 3600 + 30 * 60)))  # 8h30m after notebook start
    GAME_ID = "all"  # discover dynamically
    SEED = 0
    # Compaction limit for the local vLLM server; set by the notebook to fit concurrency x context in the KV cache.
    COMPACTION_TOKEN_LIMITS["local"] = int(os.environ.get("COMPACTION_LIMIT", "80000"))
    if KAGGLE_QUICK_TEST:
        MAX_TURNS = int(os.environ.get("MAX_TURNS", "5"))
        ENABLE_REPLAYS = True
    else:
        MAX_TURNS = int(os.environ.get("MAX_TURNS", "2400"))
        ENABLE_REPLAYS = False

# ---- v2.10
FULL_ARRAY_PRINT = os.environ.get("V2_FULL_ARRAY_PRINT", "1") == "1"   # run: numpy arrays up to 4096 cells print in full (was: abbreviated beyond 1000)
STORE_MAX_MB = int(os.environ.get("V2_STORE_MAX_MB", "64"))           # store size limit (was 4 MB); on failure the previous store is kept
LLM_REQUEST_TIMEOUT_CAP = float(os.environ.get("V2_LLM_REQUEST_TIMEOUT_CAP", "600"))   # one model request never waits longer than this
WAIT_GRACE_SECONDS = float(os.environ.get("V2_WAIT_GRACE_SECONDS", "240"))   # after the deadline, stop waiting for stuck game threads after this
CLICK_CANDIDATES_PER_OBJECT = True   # path: one CLICK candidate per object (or CLICK_TARGETS(board) when defined), was the first 256 non-zero cells

# ---- v2.11
import os as _os_v211
# the model's code (loading the code file, the model check, path, predictions) runs in the game's sandbox worker process,
# never in the harness process (off: v2.10 behaviour, in this process without time limits)
MODEL_CODE_IN_WORKER = os.environ.get("V2_MODEL_CODE_IN_WORKER", "1") == "1"
WORKER_NICE = int(os.environ.get("V2_WORKER_NICE", "10"))                 # CPU priority of worker processes (higher = lower)
WORKER_JOB_SLOTS = int(os.environ.get("V2_WORKER_JOB_SLOTS", str(max(4, _os_v211.cpu_count() or 4))))  # run/check/path workers at once (nice keeps them below the LLM server)
MODEL_CHECK_TIME_LIMIT = float(os.environ.get("V2_MODEL_CHECK_TIME_LIMIT", "45"))   # seconds of next_state calls per model check
CODE_LOAD_TIME_LIMIT = float(os.environ.get("V2_CODE_LOAD_TIME_LIMIT", "15"))       # loading the code file (py / patch / kept functions)
PATH_MEM_LIMIT_MB = int(os.environ.get("V2_PATH_MEM_LIMIT_MB", "3000"))            # memory of a path search worker
MODEL_CHECK_IMAGE = os.environ.get("V2_MODEL_CHECK_IMAGE", "1") == "1"   # image of the model check's first difference (on change)
STEP_REPORT_COMPRESS = os.environ.get("V2_STEP_REPORT_COMPRESS", "1") == "1"   # long act()/path step reports: one line per step
STEP_REPORT_FULL_CHARS = int(os.environ.get("V2_STEP_REPORT_FULL_CHARS", "12000"))  # ... only when the full reports of 2+ steps exceed this (v2.11c: 12000, was 3500; the feedback cap is 16000)
OBJECT_VIEW_TOOL = os.environ.get("V2_OBJECT_VIEW_TOOL", "1") == "1"     # the set_object_view tool
# v2.12b (flag): notes with a status (seen / inferred / assumed) and the model's own evidence; cited history[i] are shown
# with the action recorded there; step reports carry their history index; after many turns on a level without
# completing it the harness lists the notes not marked seen and the actions / regions not tried on this level.
EVIDENCE_NOTES = os.environ.get("V2_EVIDENCE_NOTES", "0") == "1"
STUCK_CHECK_TURNS = int(os.environ.get("V2_STUCK_CHECK_TURNS", "20"))    # turns on a level before the first stuck check
STUCK_CHECK_EVERY = int(os.environ.get("V2_STUCK_CHECK_EVERY", "15"))    # ... then every this many turns
# v2.12d (flag): a short TESTING BEFORE CONCLUDING section in the system prompt (test a "cannot happen" conclusion in a
# different situation, check cheap predictions with an action instead of more analysis, and when a plan comes out
# impossible test the cheapest untested belief first). Prompt only; no harness behaviour changes.
TESTING_DISCIPLINE = os.environ.get("V2_TESTING_DISCIPLINE", "0") == "1"

# Explicit opt-in: one Python tool and fixed xhigh with room for a complete call.
MINIMAL_HARNESS = os.environ.get("V2_MINIMAL_HARNESS", "0") == "1"
if MINIMAL_HARNESS:
    HARNESS_VERSION = "minimal-v1"
    ADAPTIVE_REASONING = TIERED_REASONING = False
    EPHEMERAL_REASONING = NOTE_CONFIDENCE = EVIDENCE_NOTES = False
    MODEL_THINKING_EFFORT, MAX_OUTPUT_TOKENS = "xhigh", 14000
    USE_TOOL_CALLING = True
    DIFF_IMAGE = DEATH_IMAGE = WIN_FRAME_IMAGE = MODEL_CHECK_IMAGE = False
    LAYERED_TRACKER = OBJECT_VIEW_TOOL = LEAN_NOACTION_TURNS = ACT_NUDGE = False
    NO_CHANGE_WARNING_TURNS = 0
    # Context trimming retains notes directly, with no extra summarization request.
    CONTEXT_TRIM = True

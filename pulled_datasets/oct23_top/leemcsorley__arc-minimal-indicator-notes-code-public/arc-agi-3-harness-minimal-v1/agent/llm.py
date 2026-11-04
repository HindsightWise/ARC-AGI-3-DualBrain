from __future__ import annotations

import threading

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Callable

import numpy as np
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError

from .board import board_to_png_data_url, board_with_striped_padding
from .config import (
    MINIMAL_HARNESS,
    BOARD_COLOR_MAP_HEX,
    ADAPTIVE_REASONING,
    TIERED_REASONING,
    EPHEMERAL_REASONING,
    NOTE_CONFIDENCE,
    OBJECT_VIEW_TOOL,
    EVIDENCE_NOTES,
    LLM_REQUEST_TIMEOUT_CAP,
    ADD_IMAGE_INPUT,
    AKASHML_BASE_URL,
    BOARD_IMAGE_PADDED_SHAPE_STEP,
    DEEPINFRA_BASE_URL,
    GOOGLE_BASE_URL,
    DEEPSEEK_BASE_URL,
    GROQ_BASE_URL,
    LLM_CACHE_DIR,
    LOCAL_VLLM_BASE_URL,
    MAX_IMAGES_DROP_SLACK,
    MAX_OUTPUT_TOKENS,
    MAX_IMAGES_PER_PROMPT,
    MODEL_NAME,
    MODEL_PROVIDER_OVERRIDE,
    MODEL_THINKING_EFFORT,
    PERSIST_SMALL_IMAGES_IN_HISTORY,
    SAMBANOVA_BASE_URL,
    TOGETHER_BASE_URL,
    USE_TOOL_CALLING,
    USE_LLM_CACHE,
    board_image_scale_for_model,
    board_small_image_scale_for_model,
    model_provider,
    model_supports_image_input,
    model_uses_striped_image_padding,
)
from .log import Color, _coerce_int, _to_dict, log_tag, stable_json_dumps
from .decision_policy import PROPOSAL_TOOL, RequestOptions
from .reference_checks import EXPECT_SCHEMA
from .adaptive_reasoning import THINK_TOOL, effort_tail
from .note_memory import without_reasoning


@dataclass
class TokenUsage:
    input_miss_tokens: int = 0
    input_cached_tokens: int = 0
    output_tokens: int = 0

    @property
    def input_total_tokens(self) -> int:
        return self.input_miss_tokens + self.input_cached_tokens

    def add(self, other: "TokenUsage") -> None:
        self.input_miss_tokens += other.input_miss_tokens
        self.input_cached_tokens += other.input_cached_tokens
        self.output_tokens += other.output_tokens

    def to_dict(self) -> dict[str, int]:
        return {
            "input_miss_tokens": self.input_miss_tokens,
            "input_cached_tokens": self.input_cached_tokens,
            "input_total_tokens": self.input_total_tokens,
            "output_tokens": self.output_tokens,
        }

    @classmethod
    def from_any(cls, payload: Any) -> "TokenUsage":
        data = _to_dict(payload)
        return cls(
            input_miss_tokens=max(0, _coerce_int(data.get("input_miss_tokens"))),
            input_cached_tokens=max(0, _coerce_int(data.get("input_cached_tokens"))),
            output_tokens=max(0, _coerce_int(data.get("output_tokens"))),
        )


@dataclass
class LLMTurnResult:
    text: str
    user_content: Any = ""
    user_content_for_history: Any = ""
    assistant_content_for_history: str = ""
    assistant_message_for_history: dict[str, Any] | None = None
    tool_calls: list["ToolCall"] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)
    cache_hit: bool = False
    reasoning_content: str = ""
    image_attachments: list["BoardImageAttachment"] = field(default_factory=list)
    board_image_omitted: bool = False
    finish_reason: str | None = None


@dataclass(frozen=True)
class BoardImageAttachment:
    label: str
    data_url: str
    scale: int
    note: str | None = None
    persist_in_history: bool = True


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments_json: str
    call_id: str | None = None


RETRY_BACKOFF_SECONDS: tuple[int, ...] = (4, 16, 64)
EMPTY_RESPONSE_BACKOFF_SECONDS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)


def _is_context_length_error(exc: Exception) -> bool:
    msg = str(exc)
    return "maximum context length" in msg or "context_length_exceeded" in msg


def _should_retry_api_error(exc: Exception) -> bool:
    if _is_context_length_error(exc):
        return False
    if isinstance(exc, (RateLimitError, APIConnectionError, APITimeoutError)):
        return True
    if isinstance(exc, APIStatusError):
        status_code = _coerce_int(getattr(exc, "status_code", None))
        if status_code in (408, 409, 425, 429):
            return True
        return status_code >= 500
    return False


def _call_with_retry(
    request_name: str, fn: Callable[[], Any], *, quiet: bool = False, quiet_label: str = ""
) -> Any:
    total_attempts = 1 + len(RETRY_BACKOFF_SECONDS)
    for attempt in range(total_attempts):
        try:
            return fn()
        except Exception as exc:
            if not _should_retry_api_error(exc) or attempt >= len(RETRY_BACKOFF_SECONDS):
                raise
            delay = RETRY_BACKOFF_SECONDS[attempt]
            if quiet:
                from .agent import _sync_print
                _sync_print(
                    f"[{quiet_label}] API retry in {delay}s ({exc.__class__.__name__})"
                )
            else:
                log_tag(
                    "LLM",
                    (
                        f"{request_name} failed ({exc.__class__.__name__}). "
                        f"Retrying in {delay}s ({attempt + 2}/{total_attempts})."
                    ),
                    Color.YELLOW,
                    bold=True,
                )
            time.sleep(delay)
    raise RuntimeError("Retry loop exhausted unexpectedly")


def token_usage_from_response_payload(response_payload: Any) -> TokenUsage:
    payload = _to_dict(response_payload)
    usage = _to_dict(payload.get("usage"))
    if not usage:
        return TokenUsage()

    # OpenAI responses API usage schema.
    if "input_tokens" in usage or "output_tokens" in usage:
        input_tokens = max(0, _coerce_int(usage.get("input_tokens")))
        details = _to_dict(usage.get("input_tokens_details"))
        cached = max(0, _coerce_int(details.get("cached_tokens")))
        output = max(0, _coerce_int(usage.get("output_tokens")))
        return TokenUsage(
            input_miss_tokens=max(0, input_tokens - cached),
            input_cached_tokens=cached,
            output_tokens=output,
        )

    # Chat completions (DeepSeek/OpenAI compatible) usage schema.
    prompt_tokens = max(0, _coerce_int(usage.get("prompt_tokens")))
    prompt_details = _to_dict(usage.get("prompt_tokens_details"))
    cached = max(0, _coerce_int(usage.get("prompt_cache_hit_tokens")))
    if cached == 0:
        cached = max(0, _coerce_int(prompt_details.get("cached_tokens")))
    miss_raw = usage.get("prompt_cache_miss_tokens")
    miss = max(0, _coerce_int(miss_raw)) if miss_raw is not None else max(0, prompt_tokens - cached)
    output = max(0, _coerce_int(usage.get("completion_tokens")))
    if output == 0 and usage.get("output_tokens") is not None:
        output = max(0, _coerce_int(usage.get("output_tokens")))
    return TokenUsage(input_miss_tokens=miss, input_cached_tokens=cached, output_tokens=output)


def llm_cache_path(request_hash: str) -> Path:
    return LLM_CACHE_DIR / f"{request_hash}.json"


def _tool_calls_to_payload(tool_calls: list[ToolCall]) -> list[dict[str, str | None]]:
    return [
        {
            "name": call.name,
            "arguments_json": call.arguments_json,
            "call_id": call.call_id,
        }
        for call in tool_calls
    ]


def _tool_calls_from_payload(payload: Any) -> list[ToolCall]:
    if not isinstance(payload, list):
        return []
    out: list[ToolCall] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        arguments_json = item.get("arguments_json")
        call_id = item.get("call_id")
        if not isinstance(name, str) or not name.strip():
            continue
        if not isinstance(arguments_json, str):
            continue
        out.append(
            ToolCall(
                name=name.strip(),
                arguments_json=arguments_json,
                call_id=call_id if isinstance(call_id, str) else None,
            )
        )
    return out


def _assistant_history_message_from_payload(payload: Any) -> dict[str, Any] | None:
    data = _to_dict(payload)
    role = data.get("role")
    if role != "assistant":
        return None

    content = data.get("content")
    if isinstance(content, (str, list, dict)):
        normalized_content = content
    elif content is None:
        normalized_content = ""
    else:
        normalized_content = str(content)

    out: dict[str, Any] = {"role": "assistant", "content": normalized_content}
    # Check both field names: vLLM >= 0.28 uses "reasoning", older uses "reasoning_content"
    reasoning = data.get("reasoning") or data.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning:
        out["reasoning"] = reasoning
    tool_calls_data = data.get("tool_calls")
    if isinstance(tool_calls_data, list) and tool_calls_data:
        out["tool_calls"] = tool_calls_data
    return out


def load_llm_cache(
    request_hash: str,
) -> tuple[str, list[ToolCall], TokenUsage, dict[str, Any] | None] | None:
    if not USE_LLM_CACHE:
        return None
    path = llm_cache_path(request_hash)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    cached_text_raw = payload.get("output_text")
    cached_text = cached_text_raw if isinstance(cached_text_raw, str) else ""
    cached_tool_calls = _tool_calls_from_payload(payload.get("tool_calls"))
    if not cached_text.strip() and not cached_tool_calls:
        return None
    usage = TokenUsage.from_any(payload.get("usage_totals"))
    if usage.input_total_tokens == 0 and usage.output_tokens == 0:
        usage = token_usage_from_response_payload(payload.get("response"))
    assistant_history_message = _assistant_history_message_from_payload(
        payload.get("assistant_history_message")
    )
    if assistant_history_message is None:
        response_payload = _to_dict(payload.get("response"))
        choices = response_payload.get("choices")
        if isinstance(choices, list) and choices:
            first_choice = _to_dict(choices[0])
            assistant_history_message = _assistant_history_message_from_payload(
                first_choice.get("message")
            )
    return cached_text, cached_tool_calls, usage, assistant_history_message


def save_llm_cache(
    request_hash: str, request_payload: dict[str, Any],
    response: Any, output_text: str, usage: TokenUsage | None = None,
    tool_calls: list[ToolCall] | None = None,
    assistant_history_message: dict[str, Any] | None = None,
) -> None:
    if not USE_LLM_CACHE:
        return
    LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = llm_cache_path(request_hash)
    if hasattr(response, "model_dump"):
        try:
            resp_payload = response.model_dump(mode="json")
        except Exception:
            resp_payload = response.model_dump()
    else:
        resp_payload = {"repr": repr(response)}
    usage_payload = usage or token_usage_from_response_payload(resp_payload)
    data = {
        "request_hash": request_hash,
        "request": request_payload,
        "response": resp_payload,
        "output_text": output_text,
        "tool_calls": _tool_calls_to_payload(tool_calls or []),
        "assistant_history_message": assistant_history_message,
        "usage_totals": usage_payload.to_dict(),
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _local_health_url() -> str:
    base = LOCAL_VLLM_BASE_URL.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return base + "/health"


def server_ready(timeout: float = 5.0) -> bool:
    """True when the local vLLM HTTP server answers /health or /v1/models (any status below 500 counts as 'up').
    Direct connection (no proxy) to the local server."""
    import urllib.error
    import urllib.request
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for url in (_local_health_url(), LOCAL_VLLM_BASE_URL.rstrip("/") + "/models"):
        try:
            with opener.open(urllib.request.Request(url, method="GET"), timeout=timeout) as resp:
                if int(getattr(resp, "status", 200)) < 500:
                    return True
        except urllib.error.HTTPError as exc:
            if int(getattr(exc, "code", 500)) < 500:
                return True
        except Exception:
            pass
    return False


def wait_for_server(max_seconds: float, *, label: str = "", poll_seconds: float = 10.0) -> bool:
    """Block until the local server answers or max_seconds pass. Returns readiness."""
    import time as _time
    if server_ready():
        return True
    t0 = _time.time()
    last_note = t0
    print(f"[{label}] SERVER: not answering; waiting up to {max_seconds / 60:.0f} min", flush=True)
    while _time.time() - t0 < max_seconds:
        _time.sleep(max(1.0, min(poll_seconds, max_seconds - (_time.time() - t0))))
        if server_ready():
            print(f"[{label}] SERVER: back after {_time.time() - t0:.0f}s", flush=True)
            return True
        if _time.time() - last_note >= 120:
            last_note = _time.time()
            print(f"[{label}] SERVER: still not answering after {_time.time() - t0:.0f}s", flush=True)
    print(f"[{label}] SERVER: not answering after {max_seconds:.0f}s", flush=True)
    return False


def create_openai_client() -> OpenAI:
    provider = model_provider(MODEL_NAME, provider_override=MODEL_PROVIDER_OVERRIDE)
    if provider == "local":
        return OpenAI(
            api_key="unused",
            base_url=LOCAL_VLLM_BASE_URL,
            timeout=600.0,
            max_retries=0,
        )
    if provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY not set")
        return OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set")
        return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
    if provider == "sambanova":
        api_key = os.getenv("SAMBANOVA_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("SAMBANOVA_API_KEY not set")
        return OpenAI(api_key=api_key, base_url=SAMBANOVA_BASE_URL)
    if provider == "google":
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        return OpenAI(api_key=api_key, base_url=GOOGLE_BASE_URL)
    if provider == "together":
        api_key = os.getenv("TOGETHER_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("TOGETHER_API_KEY not set")
        return OpenAI(api_key=api_key, base_url=TOGETHER_BASE_URL)
    if provider == "deepinfra":
        api_key = os.getenv("DEEPINFRA_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("DEEPINFRA_API_KEY not set")
        return OpenAI(api_key=api_key, base_url=DEEPINFRA_BASE_URL)
    if provider == "akashml":
        api_key = os.getenv("AKASHML_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("AKASHML_API_KEY not set")
        return OpenAI(api_key=api_key, base_url=AKASHML_BASE_URL)
    return OpenAI()


def extract_chat_completion_text(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    if message is None:
        return ""
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text:
                    chunks.append(text)
            elif isinstance(item, str) and item:
                chunks.append(item)
            elif item is not None:
                chunks.append(str(item))
        return "\n".join(chunks)
    return str(content) if content is not None else ""


def extract_chat_completion_reasoning_content(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    if message is None:
        return ""
    # vLLM >= 0.28 uses "reasoning"; older servers use "reasoning_content"
    reasoning = getattr(message, "reasoning", None)
    if isinstance(reasoning, str):
        return reasoning
    reasoning_content = getattr(message, "reasoning_content", None)
    if isinstance(reasoning_content, str):
        return reasoning_content
    message_payload = _to_dict(message)
    reasoning_from_payload = message_payload.get("reasoning") or message_payload.get("reasoning_content")
    return reasoning_from_payload if isinstance(reasoning_from_payload, str) else ""


def _encode_board_image_data_url(
    board_for_image: Any | None,
    *,
    scale: int,
) -> tuple[str | None, str | None]:
    if board_for_image is None:
        return None, None
    try:
        board_arr = np.asarray(board_for_image, dtype=np.uint8)
        image_note: str | None = None
        if model_uses_striped_image_padding(MODEL_NAME):
            board_arr = board_with_striped_padding(
                board_arr,
                target_shape=BOARD_IMAGE_PADDED_SHAPE_STEP,
            )
            image_note = (
                f"{BOARD_IMAGE_PADDED_SHAPE_STEP[0]}x{BOARD_IMAGE_PADDED_SHAPE_STEP[1]} "
                "canvas with striped padding"
            )
        return board_to_png_data_url(board_arr, scale=scale), image_note
    except Exception as exc:
        log_tag("IMAGE", f"Failed to encode board image input: {exc!r}", Color.YELLOW, bold=True)
        return None, None


def _build_board_image_attachments(
    board_for_big_image: Any | None,
    board_for_small_image: Any | None,
) -> list[BoardImageAttachment]:
    if MINIMAL_HARNESS:
        from . import minimal_memory as mm
        board = board_for_small_image if board_for_small_image is not None else board_for_big_image
        if board is None:
            return []
        mm.bind(BOARD_COLOR_MAP_HEX)
        value = mm.render(board, title="Current board" if board_for_small_image is not None else "Level start board")
        return [BoardImageAttachment(value["label"], mm.image_url(value), 10)]
    if not ADD_IMAGE_INPUT:
        return []
    if not model_supports_image_input(MODEL_NAME):
        return []

    attachments: list[BoardImageAttachment] = []
    big_scale = board_image_scale_for_model(MODEL_NAME)
    small_scale = board_small_image_scale_for_model(MODEL_NAME)
    include_level_start_small_label = board_for_big_image is not None

    if board_for_big_image is not None:
        big_data_url, big_note = _encode_board_image_data_url(
            board_for_big_image,
            scale=big_scale,
        )
        if big_data_url is not None:
            attachments.append(
                BoardImageAttachment(
                    label="Big picture of the game board",
                    data_url=big_data_url,
                    scale=big_scale,
                    note=big_note,
                )
            )

    if board_for_small_image is not None:
        small_data_url, small_note = _encode_board_image_data_url(
            board_for_small_image,
            scale=small_scale,
        )
        if small_data_url is not None:
            small_label = (
                "Small picture of the game board"
                if include_level_start_small_label
                else "Small updated picture of the game board"
            )
            attachments.append(
                BoardImageAttachment(
                    label=small_label,
                    data_url=small_data_url,
                    scale=small_scale,
                    note=small_note,
                    persist_in_history=PERSIST_SMALL_IMAGES_IN_HISTORY,
                )
            )
    return attachments


def _responses_user_content(
    user_message: str,
    image_attachments: list[BoardImageAttachment],
) -> str | list[dict[str, Any]]:
    if not image_attachments:
        return user_message
    content: list[dict[str, Any]] = []
    for attachment in image_attachments:
        content.append(
            {
                "type": "input_image",
                "image_url": attachment.data_url,
                "detail": "original",
            }
        )
    label_lines = [
        f"Image {idx + 1}: {attachment.label}."
        for idx, attachment in enumerate(image_attachments)
    ]
    combined_text = "\n".join(label_lines)
    if user_message.strip():
        combined_text = f"{combined_text}\n\n{user_message}"
    content.append({"type": "input_text", "text": combined_text})
    return content


def _chat_user_content(
    user_message: str,
    image_attachments: list[BoardImageAttachment],
    *,
    provider: str,
) -> str | list[dict[str, Any]]:
    if not image_attachments:
        return user_message
    content: list[dict[str, Any]] = []
    for attachment in image_attachments:
        image_url_payload: dict[str, Any] = {"url": attachment.data_url}
        if provider not in {"deepinfra", "akashml", "local"}:
            image_url_payload["detail"] = "original"
        content.append({"type": "image_url", "image_url": image_url_payload})
    label_lines = [
        f"Image {idx + 1}: {attachment.label}."
        for idx, attachment in enumerate(image_attachments)
    ]
    combined_text = "\n".join(label_lines)
    if user_message.strip():
        combined_text = f"{combined_text}\n\n{user_message}"
    content.append({"type": "text", "text": combined_text})
    return content


def _command_tool_specs(action_labels: list[str], *, include_estimate: bool = False) -> list[dict[str, Any]]:
    if MINIMAL_HARNESS:
        return [{"name": "run", "description": (
            "Execute Python to inspect, compute, save notes/store and act. "
            "Print output, shown images and all action results are returned. "
            "notes are restored after context trimming. Use show(render(...)) to see an image."),
            "parameters": {"type": "object", "properties": {"code": {"type": "string"}},
                           "required": ["code"], "additionalProperties": False}}]
    from .allocation import ESTIMATE_TOOL_SPEC
    specs = [
        {
            "name": "py",
            "description": (
                "Replace the full virtual Python code file. "
                "Use this to define or rewrite functions and helpers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Complete replacement Python source code.",
                    },
                },
                "required": ["code"],
                "additionalProperties": False,
            },
        },
        {
            "name": "patch",
            "description": (
                "Apply a unified-diff style patch to the virtual Python file. "
                "Each hunk starts with @@ and uses space/-/+ prefixes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "patch": {
                        "type": "string",
                        "description": "Patch text with @@ hunks.",
                    },
                },
                "required": ["patch"],
                "additionalProperties": False,
            },
        },
        {
            "name": "run",
            "description": (
                "Run Python in the sandbox namespace (np, lib, states, current_board, actions_history, history, past_levels, "
                "store, indicator_rects and your code file). print(...) output is returned. "
                "Call act([\"UP\", \"RIGHT\"]) inside the code to execute real actions immediately and continue with the result; "
                "a batch stops at the first action that changes nothing outside the remaining-actions indicator."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python analysis code to execute.",
                    },
                },
                "required": ["code"],
                "additionalProperties": False,
            },
        },
        {
            "name": "actions",
            "description": (
                "Execute explicit environment actions in order. "
                "Action labels are UP, DOWN, LEFT, RIGHT, SPACE, CLICK, UNDO and RESET; "
                "only the ones listed under 'Available actions' in the game messages are valid. "
                "For CLICK, format line as: CLICK row col value. RESET must be sent on its own. "
                "An action that is a known no-op or a known fatal action in this exact board state is refused "
                "unless repeat is true."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "lines": {
                        "type": "array",
                        "description": (
                            "Ordered action lines. Each line is one action "
                            "(for example 'UP' or 'CLICK 30 45 9')."
                        ),
                        "items": {
                            "type": "string",
                        },
                    },
                    "repeat": {
                        "type": "boolean",
                        "description": "Set true to execute an action deliberately even though it is a known no-op / known fatal action in this exact board state.",
                    },
                    "expect": {**EXPECT_SCHEMA, "description": "Check component IDs or saved names after EACH action; stop the batch on contradicted or unknown."},
                },
                "required": ["lines"],
                "additionalProperties": False,
            },
        },
        {
            "name": "path",
            "description": (
                "Run BFS over your next_state model and execute up to max_steps actions of the found path; "
                "execution stops at the first step whose observed board differs from the model's prediction."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "max_steps": {
                        "type": "integer",
                        "description": "Maximum number of path actions to execute (>=1).",
                        "minimum": 1,
                    },
                },
                "required": ["max_steps"],
                "additionalProperties": False,
            },
        },
        {
            "name": "remaining_actions_indicator",
            "description": (
                "Mark the rectangle(s) that contain only the remaining-actions indicator. The list you send REPLACES the "
                "previous selection entirely (usually one rectangle; occasionally the indicator is split into two or more "
                "parts, e.g. a bar at the top and one at the bottom). Send an empty list to clear."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "rects": {
                        "type": "array",
                        "description": "All indicator rectangles, each as '[row_start:row_end, col_start:col_end]'. [] clears.",
                        "items": {"type": "string"},
                    },
                },
                "required": ["rects"],
                "additionalProperties": False,
            },
        },
        {
            "name": "note",
            "description": (
                "Write or update one entry of your pinned notes: the only part of your reasoning guaranteed to survive "
                "context trimming (the full notes are re-shown after every trim and at every level start). Keep confirmed "
                "rules with their evidence, rules you have ruled out, surprising observations, the win condition of each "
                "level and open questions. Same key = replace; empty text = delete."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Short entry name, e.g. 'win_rule', 'UP', 'ruled_out', 'L2'."},
                    "text": {"type": "string", "description": "The entry text (one to a few lines). Empty string deletes the entry."},
                },
                "required": ["key", "text"],
                "additionalProperties": False,
            },
        },
    ]
    if ADAPTIVE_REASONING:
        from .adaptive_reasoning import TIERED_THINK_TOOL
        specs.append(TIERED_THINK_TOOL if TIERED_REASONING else THINK_TOOL)
    if EVIDENCE_NOTES:   # v2.12b: notes carry a status and the model's own evidence
        for _s in specs:
            if _s.get("name") == "note":
                _s["description"] = (
                    "Write or update one entry of your pinned notes: the only part of your reasoning guaranteed to survive "
                    "context trimming (the full notes are re-shown after every trim and at every level start). One rule, "
                    "observation or idea per entry, with its status, how certain you are and your evidence. Same key = replace; "
                    "empty text = delete.")
                _s["parameters"]["properties"]["status"] = {
                    "type": "string", "enum": ["seen", "inferred", "assumed"],
                    "description": "seen = observed directly; inferred = follows from what you observed; assumed = not tested yet."}
                _s["parameters"]["properties"]["evidence"] = {
                    "type": "string",
                    "description": "What you tried and in which situation, citing steps as history[i] (each step report shows "
                                   "its index), e.g. 'pressed UP (history[0]) and DOWN (history[6]), both with the cup upright'."}
                _s["parameters"]["properties"]["certainty"] = {
                    "type": "string",
                    "description": "How sure you are, in your own words (e.g. 'certain: seen 4 times in different positions', "
                                   "'likely', 'a guess')."}
                _s["parameters"]["required"] = ["key", "text", "status", "certainty", "evidence"]
    if NOTE_CONFIDENCE:
        for spec in specs:
            if spec.get("name") == "note":
                spec["description"] = (
                    "Save durable knowledge for later turns: concise observations with evidence, hypotheses, "
                    "rejected explanations, uncertainties and next experiments. Same key replaces the entry; "
                    "empty text deletes it. Include confidence (0-100) on additions/updates; this is stored "
                    "and displayed only. Notes survive resets, level changes and context trimming within this run.")
                spec["parameters"]["properties"]["confidence"] = {
                    "type": "number", "minimum": 0, "maximum": 100,
                    "description": "Your confidence in this note: 0 unsupported, 50 uncertain, 100 very strong evidence."}
                spec["parameters"]["required"].append("confidence")
    if OBJECT_VIEW_TOOL:   # v2.11
        specs.append({
            "name": "set_object_view",
            "description": (
                "Change how the harness describes the board in its own reports (the change lines after actions, the "
                "inventory, the model check) and lib.frame_diff's default: the listed colours count as gaps, and touching "
                "cells of all other colours are reported as one component (possibly spanning several game objects). "
                "It never changes the game or other lib functions. [] goes back to single-colour components. Kept until changed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "exclude": {
                        "type": "array",
                        "description": "Colour values (0-15) to treat as gaps, usually the floor / background; [] = default view.",
                        "items": {"type": "integer"},
                    },
                },
                "required": ["exclude"],
                "additionalProperties": False,
            },
        })
    if include_estimate:
        specs.append(ESTIMATE_TOOL_SPEC)
    for spec in specs:
        if spec["name"] == "actions":
            spec["description"] += " CLICK C11 or CLICK @name resolves a current object reference; optional expect checks each action."
    return specs


def _responses_tools_from_specs(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for spec in specs:
        out.append(
            {
                "type": "function",
                "name": spec["name"],
                "description": spec["description"],
                "parameters": spec["parameters"],
            }
        )
    return out


def _chat_tools_from_specs(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for spec in specs:
        out.append(
            {
                "type": "function",
                "function": {
                    "name": spec["name"],
                    "description": spec["description"],
                    "parameters": spec["parameters"],
                },
            }
        )
    return out


def _tool_calls_for_history_text(tool_calls: list[ToolCall]) -> str:
    if not tool_calls:
        return ""
    lines = ["[Tool calls]"]
    for call in tool_calls:
        lines.append(f"{call.name}: {call.arguments_json}")
    return "\n".join(lines)


def _normalize_tool_arguments(arguments: Any) -> str:
    if isinstance(arguments, str):
        return arguments
    try:
        return json.dumps(arguments, ensure_ascii=False)
    except Exception:
        return str(arguments)


def _extract_responses_tool_calls(response: Any) -> list[ToolCall]:
    output_items = getattr(response, "output", None)
    if not isinstance(output_items, list):
        return []
    out: list[ToolCall] = []
    for item in output_items:
        payload = _to_dict(item)
        if payload.get("type") != "function_call":
            continue
        name = payload.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        arguments_json = _normalize_tool_arguments(payload.get("arguments"))
        call_id_raw = payload.get("call_id")
        out.append(
            ToolCall(
                name=name.strip(),
                arguments_json=arguments_json,
                call_id=call_id_raw if isinstance(call_id_raw, str) else None,
            )
        )
    return out


def _extract_chat_completion_tool_calls(response: Any) -> list[ToolCall]:
    choices = getattr(response, "choices", None)
    if not choices:
        return []
    message = getattr(choices[0], "message", None)
    if message is None:
        return []
    raw_tool_calls = getattr(message, "tool_calls", None)
    if not raw_tool_calls:
        return []
    out: list[ToolCall] = []
    for call in raw_tool_calls:
        payload = _to_dict(call)
        fn_payload = _to_dict(payload.get("function"))
        name = fn_payload.get("name") or payload.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        arguments_json = _normalize_tool_arguments(
            fn_payload.get("arguments") if fn_payload else payload.get("arguments")
        )
        call_id_raw = payload.get("id")
        out.append(
            ToolCall(
                name=name.strip(),
                arguments_json=arguments_json,
                call_id=call_id_raw if isinstance(call_id_raw, str) else None,
            )
        )
    return out


_IMAGE_PART_TYPES = {"image_url", "input_image"}
_LEVEL_IMAGE_LABEL = "Big picture of the game board"
_UNCHANGED_BOARD_NOTE = (
    "[Visible board unchanged from the last board image, which remains in context. "
    "current_board is up to date.]"
)


def _history_contains_image(conversation: list[dict[str, Any]], data_url: str) -> bool:
    """Check actual retained image parts, not captions left behind after removal."""
    for message in conversation:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") not in _IMAGE_PART_TYPES:
                continue
            value = part.get("image_url")
            if isinstance(value, dict):
                value = value.get("url")
            if value == data_url:
                return True
    return False


def _enforce_image_cap(conversation: list[dict[str, Any]], n_new_images: int, *, quiet_label: str = "") -> int:
    """Keep (history images + this turn's images) within MAX_IMAGES_PER_PROMPT by replacing the oldest
    per-turn images in the history with a text note (level-start pictures are dropped last). The
    history messages are modified in place so the prompt prefix stays stable on later turns."""
    if MAX_IMAGES_PER_PROMPT <= 0:
        return 0
    found: list[tuple[int, int, bool]] = []  # (message index, part index, is level-start picture)
    for mi, msg in enumerate(conversation):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        label_text = ""
        for part in content:  # the caption part ("Image 1: ...") follows the images; removed images leave notes before it
            if isinstance(part, dict) and part.get("type") in ("text", "input_text") and str(part.get("text") or "").startswith("Image 1:"):
                label_text = str(part.get("text") or "")
                break
        k = 0
        for pi, part in enumerate(content):
            if isinstance(part, dict) and part.get("type") in _IMAGE_PART_TYPES:
                k += 1
                pinned_image = f"Image {k}: {_LEVEL_IMAGE_LABEL}" in label_text
                if MINIMAL_HARNESS:
                    pinned_image |= (f"Image {k}: notes" in label_text or f"Image {k}: Level start board" in label_text)
                found.append((mi, pi, pinned_image))
    total = len(found) + n_new_images
    if total <= MAX_IMAGES_PER_PROMPT:
        return 0
    target_history = max(0, MAX_IMAGES_PER_PROMPT - MAX_IMAGES_DROP_SLACK - n_new_images)
    need = len(found) - target_history
    order = [f for f in found if not f[2]] + [f for f in found if f[2]]
    for mi, pi, _ in order[:need]:
        conversation[mi]["content"][pi] = {"type": "text", "text": "[an earlier board image was removed here to stay within the image limit]"}
    print(f"[{quiet_label}] IMAGES: {total} would exceed the {MAX_IMAGES_PER_PROMPT}-image limit; "
          f"removed the {need} oldest history image(s)", flush=True)
    return need


def request_llm_response(
    client: OpenAI,
    system_prompt: str,
    conversation: list[dict[str, Any]],
    user_message: str,
    *,
    board_for_big_image: Any | None = None,
    board_for_small_image: Any | None = None,
    available_action_labels: list[str] | None = None,
    quiet: bool = False,
    quiet_label: str = "",
    include_estimate_tool: bool = False,
    remaining_seconds: float | None = None,
    is_compaction: bool = False,
    request_options: RequestOptions | None = None,
    include_proposal_tool: bool = False,
    thinking_budget_override: int | None = None,
    reuse_unchanged_board: bool = False,
) -> LLMTurnResult:
    """Send request to LLM and return response text plus token usage."""
    provider = model_provider(MODEL_NAME, provider_override=MODEL_PROVIDER_OVERRIDE)
    if EPHEMERAL_REASONING and provider != "local":
        raise ValueError("Ephemeral reasoning currently requires local vLLM with the Qwen chat template")
    if request_options is not None and provider != "local":
        raise ValueError("Decision recovery request overrides currently require the local vLLM provider")
    effective_max_tokens = request_options.max_output_tokens if request_options else MAX_OUTPUT_TOKENS
    effective_effort = request_options.reasoning_effort if request_options else MODEL_THINKING_EFFORT
    effective_thinking = request_options.enable_thinking if request_options else True
    budget_body = {}
    if MINIMAL_HARNESS and provider == "local" and not is_compaction:
        from .thinking_budget import thinking_token_budget
        effective_effort, effective_thinking = "xhigh", True
        budget = thinking_token_budget(effective_max_tokens, recovery=request_options is not None)
        if thinking_budget_override is not None:
            if not 0 <= thinking_budget_override < effective_max_tokens:
                raise ValueError("thinking budget override must fit within the output cap")
            budget = thinking_budget_override
        if budget is not None:
            budget_body["thinking_token_budget"] = budget
    if ADAPTIVE_REASONING and provider != "local":
        raise ValueError("Adaptive reasoning requires the local vLLM provider and patched Qwen template")
    if ADAPTIVE_REASONING and request_options is None:
        effective_effort = "xhigh" if TIERED_REASONING else "medium"
    # Per-call only: omitting temperature preserves the server default for normal
    # requests, including other games running concurrently and later compaction.
    sampling_overrides = ({"temperature": request_options.temperature}
                          if request_options and request_options.temperature is not None else {})
    finish_reason: str | None = None
    image_attachments = _build_board_image_attachments(
        board_for_big_image,
        board_for_small_image,
    )
    if not is_compaction and image_attachments is not None:
        from . import harness_ext as _hx
        for _lbl, _url, _scale, _note in _hx.take_extra_images():
            image_attachments.append(BoardImageAttachment(label=_lbl, data_url=_url, scale=_scale, note=_note, persist_in_history=True))
    board_image_omitted = False
    if (MINIMAL_HARNESS and reuse_unchanged_board and not is_compaction
            and board_for_big_image is None and board_for_small_image is not None and image_attachments):
        # The loop checked exact board equality and lifecycle/trim boundaries.
        # Recheck the retained image after adding this request's explicit images:
        # image-cap eviction can remove it even when the conversation wasn't trimmed.
        previous_url = image_attachments[0].data_url
        if _history_contains_image(conversation, previous_url):
            _enforce_image_cap(conversation, len(image_attachments) - 1, quiet_label=quiet_label)
            if _history_contains_image(conversation, previous_url):
                image_attachments = image_attachments[1:]
                user_message += "\n\n" + _UNCHANGED_BOARD_NOTE
                board_image_omitted = True
    _enforce_image_cap(conversation, len(image_attachments), quiet_label=quiet_label)
    if image_attachments:
        detail_mode = "omitted" if provider in {"deepinfra", "akashml", "local"} else "original"
        for attachment in image_attachments:
            if not quiet:
                note_suffix = f" ({attachment.note})" if attachment.note else ""
                log_tag(
                    "IMAGE",
                    (
                        f"Attached {attachment.label.lower()} PNG input "
                        f"(scale={attachment.scale}, detail={detail_mode}){note_suffix}."
                    ),
                    Color.BLUE,
                    bold=True,
                )

    use_responses_api = provider not in {"deepseek", "sambanova", "google", "together", "deepinfra", "akashml", "local"}
    use_tool_calling = USE_TOOL_CALLING and (MINIMAL_HARNESS or bool(available_action_labels))
    tool_specs = _command_tool_specs(available_action_labels or [], include_estimate=include_estimate_tool) if use_tool_calling else []   # same tools on compaction too -> the cached prefix stays valid
    if include_proposal_tool and use_tool_calling and not is_compaction and not MINIMAL_HARNESS:
        tool_specs.append(PROPOSAL_TOOL)
    responses_tools = _responses_tools_from_specs(tool_specs) if (use_tool_calling and use_responses_api and not is_compaction) else []
    chat_tools = _chat_tools_from_specs(tool_specs) if (use_tool_calling and not use_responses_api) else []
    if use_responses_api:
        actual_user_content = _responses_user_content(user_message, image_attachments)
    else:
        actual_user_content = _chat_user_content(user_message, image_attachments, provider=provider)

    history_attachments = [a for a in image_attachments if a.persist_in_history]
    if history_attachments == image_attachments:
        history_user_content = actual_user_content
    elif use_responses_api:
        history_user_content = _responses_user_content(user_message, history_attachments)
    else:
        history_user_content = _chat_user_content(user_message, history_attachments, provider=provider)

    if ADAPTIVE_REASONING:
        # A final text part comes after images and is kept verbatim in history.
        # Never rewrite a prior turn's setting when the next turn changes effort.
        tail = effort_tail(effective_effort)
        def with_tail(content):
            return content + "\n\n" + tail if isinstance(content, str) else [*content, {"type": "text", "text": tail}]
        actual_user_content = with_tail(actual_user_content)
        history_user_content = with_tail(history_user_content)

    base_messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        *([without_reasoning(m) for m in conversation] if EPHEMERAL_REASONING else conversation),
    ]
    current_user_msg = {"role": "user", "content": actual_user_content}
    chat_input_messages = [*base_messages, current_user_msg]
    responses_input_messages = chat_input_messages

    if provider == "deepseek":
        request_payload: dict[str, Any] = {
            "provider": provider,
            "api": "chat.completions",
            "model": MODEL_NAME,
            "reasoning_effort": MODEL_THINKING_EFFORT,
            "messages": chat_input_messages,
            "extra_body": {"thinking": {"type": "enabled"}},
        }
        if chat_tools:
            request_payload["tools"] = chat_tools
            request_payload["tool_choice"] = "auto"
    elif provider == "local":
        request_payload = {
            "provider": provider,
            "api": "chat.completions",
            "model": MODEL_NAME,
            "reasoning_effort": MODEL_THINKING_EFFORT,
            "messages": chat_input_messages,
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": True,
                    "preserve_thinking": not EPHEMERAL_REASONING,
                },
            },
        }
        if chat_tools:
            request_payload["tools"] = chat_tools
            request_payload["tool_choice"] = "auto"
    elif provider == "akashml":
        request_payload = {
            "provider": provider,
            "api": "chat.completions",
            "model": MODEL_NAME,
            "reasoning_effort": MODEL_THINKING_EFFORT,
            "messages": chat_input_messages,
        }
        if chat_tools:
            request_payload["tools"] = chat_tools
            request_payload["tool_choice"] = "auto"
    elif provider in {"sambanova", "google", "together", "deepinfra"}:
        request_payload = {
            "provider": provider,
            "api": "chat.completions",
            "model": MODEL_NAME,
            "messages": chat_input_messages,
        }
        if chat_tools:
            request_payload["tools"] = chat_tools
            request_payload["tool_choice"] = "auto"
    else:
        request_payload = {
            "provider": provider,
            "api": "responses",
            "model": MODEL_NAME,
            "reasoning": {"effort": MODEL_THINKING_EFFORT},
            "input": responses_input_messages,
        }
        if responses_tools:
            request_payload["tools"] = responses_tools
        if MODEL_NAME.lower().startswith("qwen/"):
            del request_payload["reasoning"]

    if provider == "local":
        # Include every effective generation option in cache identity; never mutate shared config.
        request_payload["max_tokens"] = effective_max_tokens
        request_payload.update(sampling_overrides)
        request_payload["reasoning_effort"] = effective_effort
        request_payload["extra_body"]["chat_template_kwargs"].update(
            enable_thinking=effective_thinking, reasoning_effort=effective_effort)
        request_payload["extra_body"].update(budget_body)
        if ADAPTIVE_REASONING:
            request_payload["extra_body"]["chat_template_kwargs"]["effort_in_user_message"] = True
    request_hash = hashlib.sha256(
        stable_json_dumps(request_payload).encode("utf-8")
    ).hexdigest()

    if not quiet:
        log_tag("REQUEST", f"hash={request_hash[:12]} model={MODEL_NAME} turns={len(conversation)//2}", Color.BLUE, bold=True)

    cached = load_llm_cache(request_hash)
    if cached is not None:
        cached_text, cached_tool_calls, cached_usage, cached_assistant_history_message = cached
        if not quiet:
            log_tag("CACHE", f"hit {request_hash[:12]} chars={len(cached_text)}", Color.YELLOW, bold=True)
        assistant_history_text = (
            cached_text
            if cached_text
            else _tool_calls_for_history_text(cached_tool_calls)
        )
        cached_reasoning = ""
        if isinstance(cached_assistant_history_message, dict):
            rc = cached_assistant_history_message.get("reasoning") or cached_assistant_history_message.get("reasoning_content")
            if isinstance(rc, str):
                cached_reasoning = rc
        return LLMTurnResult(
            text=cached_text,
            user_content=actual_user_content,
            user_content_for_history=history_user_content,
            assistant_content_for_history=assistant_history_text,
            assistant_message_for_history=(without_reasoning(cached_assistant_history_message)
                if EPHEMERAL_REASONING and cached_assistant_history_message is not None else cached_assistant_history_message),
            tool_calls=cached_tool_calls,
            usage=cached_usage,
            cache_hit=True,
            reasoning_content=cached_reasoning,
            image_attachments=image_attachments,
            board_image_omitted=board_image_omitted,
        )

    if not quiet:
        log_tag("LLM", "requesting response...", Color.GREEN, bold=True)
    usage = TokenUsage()
    tool_calls: list[ToolCall] = []
    reasoning_content = ""
    if provider == "deepseek":
        response = _call_with_retry(
            "chat.completions.create",
            lambda: client.chat.completions.create(
                model=MODEL_NAME,
                messages=chat_input_messages,
                reasoning_effort=MODEL_THINKING_EFFORT,
                extra_body={"thinking": {"type": "enabled"}},
                **({"tools": chat_tools, "tool_choice": "auto"} if chat_tools else {}),
            ),
            quiet=quiet, quiet_label=quiet_label,
        )
        usage.add(token_usage_from_response_payload(response))
        text = extract_chat_completion_text(response).strip()
        reasoning_content = extract_chat_completion_reasoning_content(response).strip()
        tool_calls = _extract_chat_completion_tool_calls(response)
        if not text and not tool_calls:
            for empty_delay in EMPTY_RESPONSE_BACKOFF_SECONDS:
                if not quiet:
                    log_tag(
                        "LLM",
                        f"empty content from provider; retrying in {empty_delay}s",
                        Color.YELLOW, bold=True,
                    )
                time.sleep(empty_delay)
                response = _call_with_retry(
                    "chat.completions.create",
                    lambda: client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=chat_input_messages,
                        reasoning_effort=MODEL_THINKING_EFFORT,
                        extra_body={"thinking": {"type": "enabled"}},
                        **({"tools": chat_tools, "tool_choice": "auto"} if chat_tools else {}),
                    ),
                    quiet=quiet, quiet_label=quiet_label,
                )
                usage.add(token_usage_from_response_payload(response))
                text = extract_chat_completion_text(response).strip()
                reasoning_content = extract_chat_completion_reasoning_content(response).strip()
                tool_calls = _extract_chat_completion_tool_calls(response)
                if text or tool_calls:
                    break
    elif provider == "akashml":
        response = _call_with_retry(
            "chat.completions.create",
            lambda: client.chat.completions.create(
                model=MODEL_NAME,
                messages=chat_input_messages,
                extra_body={"reasoning_effort": MODEL_THINKING_EFFORT},
                **({"tools": chat_tools, "tool_choice": "auto"} if chat_tools else {}),
            ),
            quiet=quiet, quiet_label=quiet_label,
        )
        usage = token_usage_from_response_payload(response)
        text = extract_chat_completion_text(response).strip()
        reasoning_content = extract_chat_completion_reasoning_content(response).strip()
        tool_calls = _extract_chat_completion_tool_calls(response)
    elif provider == "local":
        # Local vLLM: no application retries, budget-derived timeout.
        # SDK max_retries=0 already set on client creation.
        local_timeout = None
        if remaining_seconds is not None and remaining_seconds > 0:
            local_timeout = max(30.0, min(remaining_seconds, LLM_REQUEST_TIMEOUT_CAP))   # v2.10: capped (was the whole remaining time)
        else:
            local_timeout = LLM_REQUEST_TIMEOUT_CAP
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=chat_input_messages,
                extra_body={
                    **budget_body,
                    "chat_template_kwargs": {
                        "enable_thinking": effective_thinking,
                        "preserve_thinking": not EPHEMERAL_REASONING,
                        "reasoning_effort": effective_effort,
                        **({"effort_in_user_message": True} if ADAPTIVE_REASONING else {}),
                    },
                },
                **({"tools": chat_tools, "tool_choice": "auto"} if chat_tools else {}),
                max_tokens=effective_max_tokens,
                **sampling_overrides,
                timeout=local_timeout,
            )
        except APITimeoutError:
            if remaining_seconds is not None and remaining_seconds <= 0:
                raise
            raise
        usage = token_usage_from_response_payload(response)
        text = extract_chat_completion_text(response).strip()
        reasoning_content = extract_chat_completion_reasoning_content(response).strip()
        try:
            finish_reason = getattr((getattr(response, "choices", None) or [None])[0], "finish_reason", None)
        except Exception:
            finish_reason = None
        if is_compaction:
            _compact_choices = getattr(response, "choices", None) or []
            _compact_choice = _compact_choices[0] if _compact_choices else None
            _compact_usage = getattr(response, "usage", None)
            print(
                f"[{quiet_label}] COMPACT raw: "
                f"finish_reason={getattr(_compact_choice, 'finish_reason', None)!r}, "
                f"prompt_tokens={getattr(_compact_usage, 'prompt_tokens', None)}, "
                f"output_tokens={getattr(_compact_usage, 'completion_tokens', None)}, "
                f"text_chars={len(text)}, reasoning_chars={len(reasoning_content)}, "
                f"tool_calls={len(_extract_chat_completion_tool_calls(response))}",
                flush=True,
            )
        # With --reasoning-parser qwen3, vLLM populates reasoning_content directly.
        # Fallback: if reasoning_content is empty, check for inline <think> tags
        # (older vLLM versions or missing parser config).
        if not reasoning_content and text:
            think_match = re.match(r'\s*<think>(.*?)</think>\s*', text, re.DOTALL)
            if think_match:
                reasoning_content = think_match.group(1).strip()
                text = text[think_match.end():].strip()
        tool_calls = _extract_chat_completion_tool_calls(response)
    elif provider in {"google", "sambanova", "together", "deepinfra"}:
        response = _call_with_retry(
            "chat.completions.create",
            lambda: client.chat.completions.create(
                model=MODEL_NAME,
                messages=chat_input_messages,
                **({"tools": chat_tools, "tool_choice": "auto"} if chat_tools else {}),
            ),
            quiet=quiet, quiet_label=quiet_label,
        )
        usage = token_usage_from_response_payload(response)
        text = extract_chat_completion_text(response).strip()
        tool_calls = _extract_chat_completion_tool_calls(response)
    else:
        if MODEL_NAME.lower().startswith("qwen/"):
            response = _call_with_retry(
                "responses.create",
                lambda: client.responses.create(
                    model=MODEL_NAME,
                    input=responses_input_messages,
                    **({"tools": responses_tools} if responses_tools else {}),
                ),
                quiet=quiet, quiet_label=quiet_label,
            )
        else:
            response = _call_with_retry(
                "responses.create",
                lambda: client.responses.create(
                    model=MODEL_NAME,
                    reasoning={"effort": MODEL_THINKING_EFFORT},
                    input=responses_input_messages,
                    **({"tools": responses_tools} if responses_tools else {}),
                ),
                quiet=quiet, quiet_label=quiet_label,
            )
        usage = token_usage_from_response_payload(response)
        text = (response.output_text or "").strip()
        tool_calls = _extract_responses_tool_calls(response)

    if not text and not tool_calls:
        # Persist empty outputs too for postmortem/debugging.
        save_llm_cache(
            request_hash,
            request_payload,
            response,
            "",
            usage=usage,
            tool_calls=[],
            assistant_history_message=None,
        )
        if not quiet:
            log_tag("LLM", "provider returned empty content; continuing turn", Color.YELLOW, bold=True)
        _cut_msg = None
        if reasoning_content and provider == "local":   # v2.10: also for an empty reply that was not cut off
            # keep the cut-off reasoning in the history so the next turn can act on what it had worked out
            _cut_msg = {"role": "assistant", "content": "", "reasoning": reasoning_content}
        return LLMTurnResult(
            text="Error: no text output",
            reasoning_content=reasoning_content,
            user_content=actual_user_content,
            user_content_for_history=history_user_content,
            assistant_content_for_history="Error: no text output",
            assistant_message_for_history=(without_reasoning(_cut_msg) if EPHEMERAL_REASONING and _cut_msg else _cut_msg),
            usage=usage,
            cache_hit=False,
            finish_reason=finish_reason,
            image_attachments=image_attachments,   # v2.7: so the replay keeps the images of length-capped turns too
            board_image_omitted=board_image_omitted,
        )

    assistant_history_message: dict[str, Any] | None = None
    if provider in {"deepseek", "sambanova", "google", "together", "deepinfra", "akashml", "local"}:
        assistant_history_message = {"role": "assistant", "content": text}
        if reasoning_content:
            # vLLM >= 0.28 uses "reasoning" as the API field; older servers use "reasoning_content".
            # vLLM's chat-template compat layer maps "reasoning" → template's reasoning_content.
            if provider == "local":
                assistant_history_message["reasoning"] = reasoning_content
            else:
                assistant_history_message["reasoning_content"] = reasoning_content
        if tool_calls:
            assistant_history_message["tool_calls"] = [
                {
                    "id": tc.call_id or f"call_{i}",
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments_json},
                }
                for i, tc in enumerate(tool_calls)
            ]

    save_llm_cache(
        request_hash,
        request_payload,
        response,
        text,
        usage=usage,
        tool_calls=tool_calls,
        assistant_history_message=assistant_history_message,
    )
    if not quiet:
        log_tag(
            "LLM",
            f"response chars={len(text)} tool_calls={len(tool_calls)}",
            Color.GREEN,
            bold=True,
        )
    assistant_history_text = text if text else _tool_calls_for_history_text(tool_calls)
    return LLMTurnResult(
        text=text,
        user_content=actual_user_content,
        user_content_for_history=history_user_content,
        assistant_content_for_history=assistant_history_text,
        assistant_message_for_history=(without_reasoning(assistant_history_message)
            if EPHEMERAL_REASONING and assistant_history_message is not None else assistant_history_message),
        tool_calls=tool_calls,
        usage=usage,
        cache_hit=False,
        reasoning_content=reasoning_content,
        image_attachments=image_attachments,
        board_image_omitted=board_image_omitted,
        finish_reason=finish_reason,
    )


FENCED_BLOCK_RE = re.compile(r"```(\w*)\s*\n(.*?)```", re.DOTALL)
UNCLOSED_FENCED_BLOCK_RE = re.compile(r"```(\w*)\s*\n([\s\S]*)\Z")
INVOKE_TOOL_TAG = "<|content_invoke_tool_json|>"
END_MESSAGE_TAG = "<|end_message|>"


@dataclass
class Command:
    kind: str  # "py", "patch", "actions", "path", "run", "remaining_actions_indicator", "note", "object_view"
    content: str


def _command_from_fenced_block(lang: str, body: str) -> Command | None:
    lang = lang.strip().lower()
    body = body.strip()
    if lang in {"propose_actions", "think", "note"}:
        return Command(lang, body)
    if lang in ("py", "python") and "def next_state(" in body:
        return Command("py", body)
    if lang == "patch":
        return Command("patch", body)
    if lang == "actions":
        return Command("actions", body)
    if lang == "path":
        return Command("path", body)
    if lang == "run":
        return Command("run", body)
    if lang in ("remaining_actions_indicator", "remaining_actions", "remaining", "indicator"):
        return Command("remaining_actions_indicator", body)
    if lang in ("estimate_progress", "estimate"):
        return Command("estimate_progress", body)
    if lang in ("py", "python"):
        return Command("py", body)
    return None


def _decode_tool_arguments_payload(raw: Any) -> Any:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
            return {"value": parsed}
        except Exception:
            return {"code": raw}
    return {}


def _command_from_tool_name_and_args(name: str, args_payload: Any) -> Command | None:
    kind = name.strip().lower()
    args = _decode_tool_arguments_payload(args_payload)
    if kind in {"propose_actions", "think"}:
        return Command(kind, json.dumps(args))

    if kind in {"py", "run"}:
        code = args.get("code")
        if not isinstance(code, str):
            for value in args.values():
                if isinstance(value, str):
                    code = value
                    break
        if not isinstance(code, str):
            return None
        return Command(kind, code)
    if kind == "patch":
        patch = args.get("patch")
        if not isinstance(patch, str):
            for value in args.values():
                if isinstance(value, str):
                    patch = value
                    break
        if not isinstance(patch, str):
            return None
        return Command("patch", patch)
    if kind == "actions":
        lines = args.get("lines")
        _rep = "#repeat\n" if args.get("repeat") in (True, "true", "True", 1) else ""
        if "expect" in args:
            _rep += "#expect " + json.dumps(args["expect"], separators=(",", ":")) + "\n"
        if isinstance(lines, list):
            text_lines = [str(x) for x in lines]
            return Command("actions", _rep + "\n".join(text_lines))
        if isinstance(lines, str):
            return Command("actions", _rep + lines)
        code = args.get("code")
        if isinstance(code, str):
            return Command("actions", _rep + code)
        value = args.get("value")
        if isinstance(value, str):
            return Command("actions", _rep + value)
        return None
    if kind == "path":
        max_steps = args.get("max_steps")
        if isinstance(max_steps, int):
            return Command("path", f"max_steps: {max_steps}")
        if isinstance(max_steps, str):
            return Command("path", f"max_steps: {max_steps.strip()}")
        return Command("path", "")
    if kind == "note":
        key = str(args.get("key") or "").strip()
        if not key:
            return None
        return Command("note", json.dumps({"key": key, "text": str(args.get("text") or ""),
                                           "status": str(args.get("status") or ""), "certainty": str(args.get("certainty") or ""),
                                           "evidence": str(args.get("evidence") or ""),
                                           **({"confidence": args.get("confidence")} if "confidence" in args else {})}))
    if kind in {"remaining_actions_indicator", "remaining_actions", "remaining", "indicator"}:
        rects = args.get("rects")
        if isinstance(rects, list):
            return Command("remaining_actions_indicator", "; ".join(str(r) for r in rects) if rects else "clear")
        if isinstance(rects, str):
            return Command("remaining_actions_indicator", rects if rects.strip() else "clear")
        rect = args.get("rect")
        if isinstance(rect, str):
            return Command("remaining_actions_indicator", rect)
        if isinstance(rect, list):
            return Command("remaining_actions_indicator", "; ".join(str(r) for r in rect) if rect else "clear")
        if args.get("clear") is True:
            return Command("remaining_actions_indicator", "clear")
        coords = ("row_start", "row_end", "col_start", "col_end")
        if all(isinstance(args.get(k), int) for k in coords):
            return Command(
                "remaining_actions_indicator",
                (
                    f"[{args['row_start']}:{args['row_end']}, "
                    f"{args['col_start']}:{args['col_end']}]"
                ),
            )
        return None
    if kind in {"estimate_progress", "estimate"}:
        return Command("estimate_progress", json.dumps(args))
    if kind in {"set_object_view", "object_view"}:   # v2.11
        ex = args.get("exclude", args.get("colors", args.get("value")))
        if isinstance(ex, str):
            try:
                ex = json.loads(ex)
            except Exception:
                ex = [x for x in re.split(r"[\s,\[\]]+", ex) if x]
        if ex is None:
            ex = []
        if isinstance(ex, (int, float)):
            ex = [ex]
        return Command("object_view", json.dumps(list(ex) if isinstance(ex, (list, tuple)) else []))
    return None


def _command_from_tool_payload(payload: Any) -> Command | None:
    data = _to_dict(payload)
    name = data.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    args_payload = data.get("args", data.get("arguments"))
    return _command_from_tool_name_and_args(name, args_payload)


def _extract_tagged_tool_payloads(text: str) -> list[tuple[int, int, Any]]:
    decoder = json.JSONDecoder()
    out: list[tuple[int, int, Any]] = []
    cursor = 0
    while True:
        start = text.find(INVOKE_TOOL_TAG, cursor)
        if start < 0:
            break
        payload_start = start + len(INVOKE_TOOL_TAG)
        while payload_start < len(text) and text[payload_start].isspace():
            payload_start += 1
        try:
            payload, consumed = decoder.raw_decode(text[payload_start:])
        except Exception:
            cursor = payload_start
            continue
        payload_end = payload_start + consumed
        span_end = payload_end
        while span_end < len(text) and text[span_end].isspace():
            span_end += 1
        if text.startswith(END_MESSAGE_TAG, span_end):
            span_end += len(END_MESSAGE_TAG)
        out.append((start, span_end, payload))
        cursor = span_end
    return out


# Regex for <tool_call><function=NAME><parameter=KEY>VALUE</parameter>...</function></tool_call>
# This is the format Qwen3.5 models emit when tool calling is enabled.
_XML_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=(\w+)>(.*?)</function>\s*</tool_call>",
    re.DOTALL,
)
_XML_PARAM_RE = re.compile(
    r"<parameter=(\w+)>(.*?)</parameter>",
    re.DOTALL,
)


def _extract_xml_tool_calls(text: str) -> list[tuple[int, int, str, dict[str, str]]]:
    """Extract <tool_call><function=name><parameter=k>v</parameter>...</function></tool_call> blocks."""
    results: list[tuple[int, int, str, dict[str, str]]] = []
    for m in _XML_TOOL_CALL_RE.finditer(text):
        func_name = m.group(1).strip()
        body = m.group(2)
        params: dict[str, str] = {}
        for pm in _XML_PARAM_RE.finditer(body):
            params[pm.group(1).strip()] = pm.group(2).strip()
        results.append((m.start(), m.end(), func_name, params))
    return results


def _command_from_xml_tool_call(func_name: str, params: dict[str, str]) -> Command | None:
    """Convert a parsed XML tool call into a Command."""
    kind = func_name.lower()
    if kind == "think":
        return _command_from_tool_name_and_args(kind, params)
    if kind == "propose_actions":
        args = dict(params)
        try:
            args["candidates"] = json.loads(params.get("candidates", ""))
        except (ValueError, TypeError):
            pass  # policy rejects malformed proposals; never salvage partial JSON
        return _command_from_tool_name_and_args(kind, args)
    if kind in {"py", "run"}:
        code = params.get("code", "")
        if not code:
            for v in params.values():
                code = v
                break
        return Command(kind, code) if code else None
    if kind == "patch":
        patch = params.get("patch", "")
        if not patch:
            for v in params.values():
                patch = v
                break
        return Command("patch", patch) if patch else None
    if kind == "actions":
        lines_raw = params.get("lines", params.get("code", params.get("value", "")))
        if "expect" in params or "repeat" in params:
            args = dict(params)
            for key in ("lines", "expect", "repeat"):
                if isinstance(args.get(key), str):
                    try:
                        args[key] = json.loads(args[key])
                    except (ValueError, TypeError):
                        pass
            args.setdefault("lines", lines_raw)
            return _command_from_tool_name_and_args("actions", args)
        if lines_raw:
            # Model may emit JSON list: ["CLICK 1 2 3", "CLICK 4 5 6"]
            try:
                parsed = json.loads(lines_raw)
                if isinstance(parsed, list):
                    return Command("actions", "\n".join(str(x) for x in parsed))
            except (json.JSONDecodeError, TypeError):
                pass
            return Command("actions", lines_raw)
        return None
    if kind == "path":
        ms = params.get("max_steps", "")
        return Command("path", f"max_steps: {ms}") if ms else Command("path", "")
    if kind in ("remaining_actions_indicator", "remaining_actions", "remaining", "indicator"):
        rect = params.get("rects", params.get("rect", ""))
        return Command("remaining_actions_indicator", rect if str(rect).strip() else "clear")
    if kind == "note":
        key = str(params.get("key") or "").strip()
        confidence = params.get("confidence")
        if confidence is not None:
            try:
                confidence = json.loads(confidence)
            except (ValueError, TypeError):
                pass  # Preserve invalid values so the note handler can report the error.
        return Command("note", json.dumps({"key": key, "text": str(params.get("text") or ""),
                                           "status": str(params.get("status") or ""),
                                           "certainty": str(params.get("certainty") or ""),
                                           "evidence": str(params.get("evidence") or ""),
                                           **({"confidence": confidence} if "confidence" in params else {})})) if key else None
    if kind in ("estimate_progress", "estimate"):
        return Command("estimate_progress", json.dumps(params))
    if kind in ("set_object_view", "object_view"):   # v2.11
        return _command_from_tool_name_and_args("set_object_view", {"exclude": params.get("exclude", "[]")})
    return None


def _extract_trailing_json_tool_payload(text: str) -> tuple[int, Command] | None:
    stripped = text.strip()
    if not stripped:
        return None
    for idx in range(len(text) - 1, -1, -1):
        if text[idx] != "{":
            continue
        candidate = text[idx:].strip()
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
        except Exception:
            continue
        cmd = _command_from_tool_payload(payload)
        if cmd is None:
            continue
        return idx, cmd
    return None


def _append_unique_command(commands: list[Command], command: Command | None) -> None:
    if command is None:
        return
    for existing in commands:
        if existing.kind == command.kind and existing.content == command.content:
            return
    commands.append(command)


_DROPPED_TLS = threading.local()


def last_dropped_calls() -> list[str]:
    """v2.10: tool calls of the last parsed response that could not be turned into commands."""
    return list(getattr(_DROPPED_TLS, "calls", None) or [])


def parse_response_commands(
    response_text: str,
    tool_calls: list[ToolCall] | None = None,
) -> tuple[str, list[Command]]:
    """Parse model response into reasoning text + ordered list of commands."""
    commands: list[Command] = []
    _dropped: list[str] = []
    _DROPPED_TLS.calls = _dropped

    if tool_calls:
        for tool_call in tool_calls:
            _cmd = _command_from_tool_name_and_args(tool_call.name, tool_call.arguments_json)
            if _cmd is None:   # v2.10: reported to the model instead of vanishing
                _dropped.append(f"{tool_call.name}: unknown tool name, or required arguments missing or invalid "
                                f"(arguments: {str(tool_call.arguments_json)[:160]})")
            elif any(c.kind == _cmd.kind and c.content == _cmd.content for c in commands):
                _dropped.append(f"{tool_call.name}: identical to an earlier call in the same response, executed once")
            _append_unique_command(commands, _cmd)
    _native = bool(commands)   # v2.10: with real tool calls, fenced blocks in the visible text are never executed

    text_without_tagged = response_text
    tagged_payloads = _extract_tagged_tool_payloads(response_text)
    if tagged_payloads:
        rebuilt_parts: list[str] = []
        last_idx = 0
        for start, end, payload in tagged_payloads:
            if start > last_idx:
                rebuilt_parts.append(response_text[last_idx:start])
            last_idx = end
            _append_unique_command(commands, _command_from_tool_payload(payload))
        if last_idx < len(response_text):
            rebuilt_parts.append(response_text[last_idx:])
        text_without_tagged = "".join(rebuilt_parts)

    # Parse <tool_call><function=name><parameter=k>v</parameter></function></tool_call> blocks
    # (Qwen3.5 models emit this XML format instead of JSON inside <tool_call> tags)
    xml_calls = _extract_xml_tool_calls(text_without_tagged)
    if xml_calls:
        rebuilt_parts2: list[str] = []
        last_idx2 = 0
        for start, end, func_name, params in xml_calls:
            if start > last_idx2:
                rebuilt_parts2.append(text_without_tagged[last_idx2:start])
            last_idx2 = end
            _append_unique_command(commands, _command_from_xml_tool_call(func_name, params))
        if last_idx2 < len(text_without_tagged):
            rebuilt_parts2.append(text_without_tagged[last_idx2:])
        text_without_tagged = "".join(rebuilt_parts2)

    last_end = 0
    reasoning_parts: list[str] = []

    for match in FENCED_BLOCK_RE.finditer(text_without_tagged):
        lang = match.group(1).strip().lower()
        body = match.group(2).strip()

        pre_text = text_without_tagged[last_end:match.start()].strip()
        if pre_text:
            reasoning_parts.append(pre_text)
        last_end = match.end()

        if not _native:
            _append_unique_command(commands, _command_from_fenced_block(lang, body))

    trailing_raw = text_without_tagged[last_end:]
    trailing = trailing_raw
    unclosed_match = UNCLOSED_FENCED_BLOCK_RE.search(trailing_raw)
    if unclosed_match:
        pre_text = trailing_raw[:unclosed_match.start()].strip()
        if pre_text:
            reasoning_parts.append(pre_text)
        parsed = None if (_native or unclosed_match.group(1).lower() in {"propose_actions", "think"}) else _command_from_fenced_block(unclosed_match.group(1), unclosed_match.group(2))
        if parsed is not None:
            _append_unique_command(commands, parsed)
            trailing = ""
        else:
            trailing = trailing_raw[unclosed_match.start():]

    trailing = trailing.strip()
    if trailing:
        reasoning_parts.append(trailing)

    reasoning = "\n".join(reasoning_parts).strip()
    trailing_json_cmd = None if _native else _extract_trailing_json_tool_payload(reasoning)
    if trailing_json_cmd is not None:
        start_idx, cmd = trailing_json_cmd
        _append_unique_command(commands, cmd)
        reasoning = reasoning[:start_idx].strip()
    return reasoning, commands

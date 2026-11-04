"""Optional durable notes and disposable reasoning; no confidence-based decisions."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import re


MEMORY_PROMPT = """
TEMPORARY THINKING, DURABLE NOTES
Your private thinking is discarded after EVERY response, including truncated responses.
It will NOT be available on the next turn. Anything learned only in thinking is lost.
Before finishing a response, use note to save any new or revised knowledge needed later:
observed mechanics and their evidence, hypotheses, rejected explanations, uncertainties,
the current plan and the next discriminating experiment. Save useful conclusions during
analysis-only turns too. Keep entries concise; update an existing key when evidence changes
and delete obsolete entries. Do not copy your full thinking into notes or visible text.
Record evidence and distinguish observations from guesses. Scope level-specific rules to
their level. Notes persist across turns, resets, level changes and context trimming within
this game attempt. Visible commands/results, observations and the virtual code also remain.
Only an accepted note update is saved; fix rejected updates when the harness reports them.
Continue playing: combine note updates with useful commands when possible. A standalone
think(question) must still be the only command; put the unresolved question in its argument.
""".strip()

CONFIDENCE_PROMPT = """
Every new or updated note must include confidence: a number from 0 to 100 expressing your
current confidence in that entry (0 = unsupported guess, 50 = uncertain, 100 = very strong
evidence). This is your self-assessment, not a measured probability. Revise it when evidence
changes. The harness only stores and displays it; it does not select actions, rank notes,
filter memory or change reasoning settings based on it. Empty text deletes an entry.
""".strip()


def without_reasoning(message: dict) -> dict:
    """Copy an assistant message without traces; preserve tool protocol and caller data."""
    clean = copy.deepcopy(message)
    if clean.get("role") == "assistant":
        for key in ("reasoning", "reasoning_content", "reasoning_details"):
            clean.pop(key, None)
        # Compatibility with older servers/cached replies that put thinking in content.
        if isinstance(clean.get("content"), str):
            clean["content"] = re.sub(r"\A\s*<think>.*?(?:</think>\s*|\Z)", "", clean["content"], flags=re.S)
    return clean


class ConfidenceNotes:
    """One independent game's notes; accepted mutations are atomically checkpointed."""

    def __init__(self, char_cap: int, path: Path, game_id: str):
        self.char_cap = char_cap
        self.path = path
        self.game_id = game_id
        self.entries: dict[str, dict] = {}
        self.revision = 0
        self._save(self.entries, self.revision)

    @staticmethod
    def display(entries: dict[str, dict]) -> dict[str, str]:
        return {k: f"[confidence: {v['confidence']:g}/100] {v['text']}" for k, v in entries.items()}

    def snapshot(self) -> dict[str, dict]:
        return copy.deepcopy(self.entries)

    def _save(self, entries: dict, revision: int) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema_version": 1, "game_id": self.game_id, "revision": revision,
                   "confidence_scale": [0, 100], "entries": entries}
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            tmp.replace(self.path)
        finally:
            tmp.unlink(missing_ok=True)

    def apply(self, raw: str) -> tuple[dict, str]:
        """Return a replay operation and model feedback; failures preserve old state."""
        try:
            item = json.loads(raw)
            if not isinstance(item, dict):
                raise ValueError("expected a JSON object")
            key, text = item.get("key"), item.get("text")
            if not isinstance(key, str) or not key.strip():
                raise ValueError("key must be a non-empty string")
            if not isinstance(text, str):
                raise ValueError("text must be a string")
            key, text = key.strip(), text.strip()
            if not text.strip("'\"` "):
                text = ""
            confidence = item.get("confidence")
            if text and (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
                         or not 0 <= confidence <= 100 or not math.isfinite(confidence)):
                raise ValueError("confidence must be a finite number from 0 to 100")
            candidate = self.snapshot()
            old = candidate.get(key)
            if text:
                candidate[key] = {"text": text, "confidence": confidence}
            else:
                candidate.pop(key, None)
            size = sum(len(k) + len(v) + 4 for k, v in self.display(candidate).items())
            if size > self.char_cap:
                raise ValueError(f"notes would exceed {self.char_cap} characters; shorten or delete entries first")
            self._save(candidate, self.revision + 1)
        except (ValueError, TypeError, OSError) as exc:
            return {"op": "rejected", "error": str(exc)}, f"[Notes] Not saved: {exc}. Previous notes are unchanged."
        self.entries = candidate
        self.revision += 1
        operation = "deleted" if not text else "updated" if old is not None else "added"
        op = {"key": key, "op": operation, "old": old, "entry": candidate.get(key), "revision": self.revision}
        detail = f" Confidence: {confidence:g}/100." if text else ""
        return op, f"[Notes] {operation.capitalize()} entry '{key}'.{detail} ({size}/{self.char_cap} chars)."

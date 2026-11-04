"""Bounded recovery after a command-free length cap. No model or environment I/O.

One instance belongs to one game. A fallback is a single action, valid only for
the exact observation/attempt/step on which the model committed it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from typing import Callable
from .reference_checks import EXPECT_SCHEMA, expectations


@dataclass(frozen=True)
class RecoverySettings:
    mode: str = "off"
    proposal_tokens: int = 2000
    refinement_tokens: int = 2000
    request_tokens: int = 2000
    temperature: float = 1.0
    request_floor: int = 600
    max_redecisions: int = 2
    max_forced_per_level: int = 3
    max_refinement_requests: int = 4
    max_episode_requests: int = 4
    max_episode_tokens: int = 4000
    early_trigger_tokens: int = 2048

    @classmethod
    def from_env(cls):
        defaults = cls()
        values = {k: type(v)(os.environ.get("V2_RECOVERY_" + k.upper(), v))
                  for k, v in asdict(defaults).items() if k != "mode"}
        mode = os.environ.get("V2_DECISION_RECOVERY", "off").strip().lower()
        if mode not in {"off", "observe", "recover_after_cap"}:
            raise ValueError("V2_DECISION_RECOVERY must be off, observe or recover_after_cap")
        obj = cls(mode=mode, **values)
        if any(v <= 0 for k, v in values.items() if k != "temperature"):
            raise ValueError("Recovery budgets and limits must be positive")
        if not math.isfinite(obj.temperature) or not 0 <= obj.temperature <= 2:
            raise ValueError("Recovery temperature must be finite and between 0 and 2")
        if min(obj.proposal_tokens, obj.refinement_tokens, obj.request_tokens, obj.max_episode_tokens) < obj.request_floor:
            raise ValueError("Recovery budgets must be >= request_floor")
        return obj

    def request_options(self, remaining: int, *, reasoning_effort="medium") -> RequestOptions:
        """Shared by live recovery and the protocol smoke test; never changes normal play."""
        return RequestOptions(min(self.request_tokens, remaining), reasoning_effort=reasoning_effort,
                              enable_thinking=True, temperature=self.temperature)


PROPOSAL_TOOL = {
    "name": "propose_actions",
    "description": "Commit 1-3 fallback candidates for THIS observation before more analysis. "
                   "Does not act immediately. The harness may execute the highest-scored valid candidate "
                   "when the remaining refinement allowance runs out. Use actions/act to act immediately.",
    "parameters": {
        "type": "object",
        "properties": {
            "candidates": {"type": "array", "minItems": 1, "maxItems": 3, "items": {
                "type": "object", "properties": {
                    "action": {"type": "string", "description": "Exactly one action; CLICK row col current_color, all 0-based."},
                    "score": {"type": "integer", "minimum": 1, "maximum": 5},
                    "purpose": {"type": "string", "enum": ["advance", "probe"]},
                    "expected": {"type": "string", "description": "Expected observation or hypothesis this tests, <=240 chars."},
                    "evidence": {"type": "string", "description": "Why this action is justified, <=240 chars."},
                    "risk": {"type": "string", "enum": ["low", "medium", "high"]},
                    "checks": EXPECT_SCHEMA,
                }, "required": ["action", "score", "purpose", "expected", "evidence", "risk"],
                "additionalProperties": False,
            }},
            "probe_omission_reason": {"type": "string", "description": "If no candidate probes an uncertain rule/control, explain why (<=240 chars)."},
        }, "required": ["candidates"], "additionalProperties": False,
    },
}


@dataclass(frozen=True)
class DecisionIdentity:
    game: str
    level: int
    attempt: int
    step: int                    # monotonic actual-action count + automatic resets
    board: str


def identity(game, level, attempt, step, board):
    digest = hashlib.sha256(str(board.shape).encode() + board.tobytes()).hexdigest()[:20]
    return DecisionIdentity(game, level, attempt, step, digest)


@dataclass(frozen=True)
class RequestOptions:
    max_output_tokens: int
    reasoning_effort: str = "medium"
    enable_thinking: bool = True
    temperature: float | None = None  # None inherits the server's normal sampling settings


class DecisionPolicy:
    def __init__(self, settings: RecoverySettings, *, minimal=False):
        self.settings = settings
        self.minimal = minimal
        self.state: DecisionIdentity | None = None
        self.active = False
        self.disabled = False
        self.forced_on_level = 0
        self.events: list[dict] = []
        self.counts: dict[str, int] = {}
        self.since_action_tokens = 0
        self.early_logged = False
        self.episode_id = 0
        self.episode_requests = 0
        self.episode_tokens = 0
        self._reset_decision()

    def _reset_decision(self):
        self.phase = "proposal"
        self.candidates = []
        self.candidate_state = None
        self.spent = 0
        self.repairs = 0
        self.redecisions = 0
        self.refinement_requests = 0
        self.evidence_seen = set()

    def event(self, kind, **data):
        self.events.append({"event": kind, **data})
        self.counts[kind] = self.counts.get(kind, 0) + 1

    def sync(self, state):
        if self.state != state:
            new_level = self.state is None or (self.state.game, self.state.level) != (state.game, state.level)
            self.complete("level_changed" if new_level else "observation_or_attempt_changed")
            if new_level:
                self.active = self.disabled = False
                self.forced_on_level = 0
            self._reset_decision()
            self.since_action_tokens = 0
            self.early_logged = False
            self.state = state

    def _episode_exhausted(self):
        return (self.episode_requests >= self.settings.max_episode_requests or
                self.settings.max_episode_tokens - self.episode_tokens < self.settings.request_floor)

    def record_response(self, options, output_tokens):
        """Account before executing tools, including act() turns that end the game."""
        if options is not None and self.active:
            self.episode_tokens += max(0, output_tokens)

    def start_turn(self, state):
        self.events = []
        self.sync(state)
        if self.settings.mode != "recover_after_cap" or not self.active or self.disabled:
            return None
        if self._episode_exhausted():
            self.abort("episode_budget_or_request_limit")
            return None
        episode_remaining = self.settings.max_episode_tokens - self.episode_tokens
        if self.phase == "proposal":
            # Keep the full history and allow a brief rethink before committing.
            self.episode_requests += 1
            return self.settings.request_options(min(self.settings.proposal_tokens, episode_remaining),
                                                 reasoning_effort="xhigh" if self.minimal else "medium")
        remaining = min(self.settings.refinement_tokens - self.spent, episode_remaining)
        if remaining < self.settings.request_floor:
            self.abort("budget_below_floor_without_valid_fallback")
            return None
        if self.phase == "revise":
            if self.redecisions >= self.settings.max_redecisions:
                self.abort("redecision_limit")
                return None
            self.redecisions += 1
        self.episode_requests += 1
        return self.settings.request_options(remaining, reasoning_effort="xhigh" if self.minimal else "medium")

    def prompt(self):
        remaining = min(self.settings.refinement_tokens - self.spent,
                        self.settings.max_episode_tokens - self.episode_tokens)
        common = (" Keep this brief. Scores 1-5 rank your preferences, not probabilities. "
                  "Identify one assumption that might be wrong and choose a cheap action that tests it; "
                  "state the expected observation. If no probe is appropriate, explain why. "
                  "Only one exact action per candidate; no RESET/UNDO. High-risk candidates will not be forced. "
                  "Use the current board and your existing evidence. Think briefly, then submit the action or candidates. "
                  f"Each request has at most {self.settings.request_tokens} output tokens INCLUDING thinking and tool arguments; "
                  "leave room for a complete tool call. "
                  "Recovery ends after an executed action or short batch; ordinary reasoning then resumes. "
                  f"This entire recovery is limited to {self.settings.max_episode_requests} requests and "
                  f"{self.settings.max_episode_tokens} output tokens, including proposals and retries.")
        if self.minimal:
            # The normal system prompt carries no recovery instructions. Supply
            # the Python helper contract only in this active-recovery message.
            common += (
                '\nPython helper: propose_actions(candidates, probe_omission_reason=""). '
                'candidates is a list of 1-3 dictionaries with action (one exact action string), '
                'score (integer 1-5), purpose ("advance" or "probe"), expected and evidence '
                '(nonempty strings, each <=240 characters), and risk ("low", "medium" or "high"). '
                'Include a probe or explain its omission in probe_omission_reason (<=240 characters). '
                'No checks, expect or component references. Submit in a separate proposal-only run, '
                'without prints, analysis or memory writes. Parent feedback confirms acceptance; '
                'the helper does not act immediately. New results or errors invalidate a saved fallback. '
                'Automatic fallback actions are rechecked against the current board and guards. '
                'Recovery retains xhigh reasoning.')
        if self.phase == "proposal":
            return ("[Decision recovery] A previous response hit the output cap without a command. "
                    "If ready, take your normal action or justified short batch NOW. Otherwise "
                    + ("call the Python helper propose_actions with 1-3 candidates inside run(code). "
                       "Use a separate run containing only that proposal, without analysis, prints or memory writes. "
                       if self.minimal else
                       "call propose_actions with 1-3 candidates before further tool-based analysis; "
                       "do not combine it with run/py/patch. ") +
                    f"Then at most {remaining} output tokens across requests remain to refine the decision. "
                    "After that the harness may take your highest-scored valid fallback." + common)
        if self.phase == "revise":
            return ("[Decision recovery] New tool results invalidated the previous fallback. "
                    "Read them, then act or submit a fresh propose_actions "
                    + ("inside a proposal-only run. " if self.minimal else "alone. ") +
                    f"Only {remaining} refinement tokens remain; this is reconsideration "
                    f"{self.redecisions}/{self.settings.max_redecisions}. Further evidence does not replenish the budget." + common)
        return (f"[Decision recovery] Fallback saved: {json.dumps(self.candidates, separators=(',', ':'))}. "
                f"Refinement allowance remaining: {remaining} tokens TOTAL. Act now, replace the candidates if needed, "
                "or do one useful check. When fewer than " + str(self.settings.request_floor) +
                " tokens remain, the highest-scored still-valid fallback may execute. "
                "New tool evidence requires a fresh commitment within the same allowance." + common)

    def _episode_summary(self):
        return {"episode": self.episode_id, "requests": self.episode_requests,
                "output_tokens": self.episode_tokens, "refinement_tokens": self.spent}

    def end_feedback(self):
        if any(e["event"] in {"recovery_completed", "recovery_aborted"} for e in self.events):
            return ("[Decision recovery] Recovery has ended. Ordinary play and the normal reasoning budget resume; "
                    "the earlier recovery token limits no longer apply.")
        return None

    def complete(self, reason):
        if not self.active:
            return
        self.event("recovery_completed", reason=reason, **self._episode_summary())
        self.active = False
        self.candidates = []
        self.candidate_state = None

    def abort(self, reason):
        self.event("recovery_aborted", reason=reason, **self._episode_summary())
        self.active = False
        self.disabled = True       # ordinary play for the rest of this level
        self.candidates = []
        self.candidate_state = None

    def _accept(self, raw, validate: Callable[[str], str | None]):
        try:
            data = json.loads(raw)
            items = data["candidates"]
            if not isinstance(items, list) or not 1 <= len(items) <= 3:
                raise ValueError("need 1-3 candidates")
            accepted = []
            for item in items:
                if not isinstance(item, dict):
                    raise ValueError("candidate must be an object")
                if self.minimal and set(item) - {"action", "score", "purpose", "expected", "evidence", "risk"}:
                    raise ValueError("minimal candidates have no checks, expect or component references")
                action = item.get("action")
                if not isinstance(action, str) or not action.strip() or "\n" in action or "\r" in action:
                    raise ValueError("candidate must contain one exact action")
                if action.split()[0].upper() in {"RESET", "UNDO"}:
                    raise ValueError("RESET/UNDO are not automatic fallbacks")
                if type(item.get("score")) is not int or not 1 <= item["score"] <= 5:
                    raise ValueError("score must be an integer 1-5")
                if item.get("purpose") not in {"advance", "probe"} or item.get("risk") not in {"low", "medium", "high"}:
                    raise ValueError("invalid purpose or risk")
                for key in ("expected", "evidence"):
                    if not isinstance(item.get(key), str) or not 1 <= len(item[key].strip()) <= 240:
                        raise ValueError(f"{key} must be 1-240 chars")
                error = validate(action)
                if error or item["risk"] == "high":
                    self.event("candidate_rejected", action=action, reason=error or "high risk")
                    continue
                accepted_item = {k: item[k] for k in ("action", "score", "purpose", "expected", "evidence", "risk")}
                if "checks" in item:
                    accepted_item["checks"] = expectations(item["checks"])
                accepted.append(accepted_item)
            if not any(i["purpose"] == "probe" for i in items) and not str(data.get("probe_omission_reason") or "").strip():
                raise ValueError("include a probe or probe_omission_reason")
            if not accepted:
                raise ValueError("no valid low/medium-risk fallback")
            self.candidates = sorted(accepted, key=lambda x: -x["score"])
            self.candidate_state = self.state
            self.phase = "refine"
            self.event("proposal_accepted", candidates=self.candidates,
                       probe_omission_reason=str(data.get("probe_omission_reason") or "")[:240])
            return True
        except (ValueError, KeyError, TypeError) as exc:
            self.candidates = []
            self.event("proposal_rejected", reason=str(exc))
            return False

    def process(self, *, options, output_tokens, capped, commands, proposals, state,
                already_acted, env_intent, evidence, validate, analysis_failed=False):
        """Called after analysis but BEFORE normal actions. Return a single fallback or None.

        Evidence is non-empty only for changed code or novel successful run output.
        The controller never executes while there are unreviewed new results.
        """
        if self.settings.mode == "off":
            return None
        self.since_action_tokens += max(0, output_tokens)
        if not self.early_logged and self.since_action_tokens >= self.settings.early_trigger_tokens:
            self.event("earlier_trigger_observed", tokens_since_action=self.since_action_tokens,
                       already_acted=already_acted, env_intent=env_intent)
            self.early_logged = True
        if capped:
            self.event("length_cap", recovery_request=options is not None, command_free=not commands)
        if already_acted or state != self.state:
            if already_acted:
                self.complete("model_action_executed")
            self.sync(state)
            return None
        if self.settings.mode == "observe":
            if capped and not commands:
                self.event("would_activate")
            return None
        if options is None:
            if capped and not commands and not self.disabled:
                self._reset_decision()
                self.episode_id += 1
                self.episode_requests = self.episode_tokens = 0
                self.active = True
                self.event("recovery_activated", identity=asdict(state), episode=self.episode_id)
            return None
        phase = self.phase
        if phase != "proposal":
            self.spent += max(0, output_tokens)
            self.refinement_requests += 1
        # Any attempted replacement invalidates the previous proposal, even if malformed.
        accepted = False
        if analysis_failed:
            self.candidates = []
            self.event("fallback_invalidated", reason="run failed; review the error before committing again")
        if proposals:
            self.candidates = []
            if analysis_failed:
                self.event("proposal_rejected", reason="run failed: proposal not trusted")
            elif capped:
                self.event("proposal_rejected", reason="capped response: proposal not trusted")
            elif len(proposals) != 1:
                self.event("proposal_rejected", reason="submit exactly one proposal call")
            else:
                accepted = self._accept(proposals[0], validate)
        if env_intent:
            return None             # normal action path always has priority
        if phase == "proposal" and not accepted:
            if self._episode_exhausted():
                self.abort("episode_budget_or_request_limit")
            elif self.repairs < 1:
                self.repairs += 1
                self.event("proposal_repair", reason="no valid proposal or action; one retry")
            else:
                self.abort("proposal_failed_after_repair")
            return None
        fresh = False
        for item in evidence:
            digest = hashlib.sha256(item.encode()).hexdigest()
            if digest not in self.evidence_seen:
                fresh = True
                self.evidence_seen.add(digest)
        if fresh:
            self.candidates = []
            self.phase = "revise"
            self.event("fallback_invalidated", reason="new analysis results require model review")
        remaining = self.settings.refinement_tokens - self.spent
        must_finish = (remaining < self.settings.request_floor or
                       self.refinement_requests >= self.settings.max_refinement_requests or
                       self._episode_exhausted())
        if not self.candidates:
            self.phase = "revise"
            if must_finish or self.redecisions >= self.settings.max_redecisions:
                self.abort("no_valid_fallback_after_evidence_or_invalid_proposal")
            return None
        if not must_finish:
            return None
        if self.candidate_state != state:
            self.abort("stale_fallback")
            return None
        for candidate in self.candidates:
            error = validate(candidate["action"])
            if error:
                self.event("fallback_vetoed", action=candidate["action"], reason=error)
                continue
            self.event("fallback_selected", candidate=candidate, refinement_tokens=self.spent)
            return candidate
        self.abort("no_valid_fallback_at_execution")
        return None

    def forced_result(self, candidate, *, outcome, changed_cells, changed_outside_indicator, checks=None):
        self.forced_on_level += 1
        self.event("fallback_executed", candidate=candidate, outcome=outcome,
                   changed_cells=changed_cells, changed_outside_indicator=changed_outside_indicator,
                   forced_on_level=self.forced_on_level)
        if checks:
            self.event("fallback_expectations", checks=checks)
        self.complete("fallback_executed")
        if self.forced_on_level >= self.settings.max_forced_per_level:
            self.disabled = True
            self.active = False
            self.event("forced_action_limit", limit=self.settings.max_forced_per_level)

    def finish_turn(self, state, *, env_intent, forced):
        if state != self.state:
            self.sync(state)         # includes no-op steps and same-pixel resets
        elif forced:
            self.abort("fallback_not_executed")
        elif env_intent and self.active:
            self.candidates = []
            self.phase = "revise"
            if self._episode_exhausted() or self.settings.refinement_tokens - self.spent < self.settings.request_floor:
                self.abort("model_action_rejected_at_budget_limit")

    def snapshot(self):
        return {"mode": self.settings.mode, "active": self.active, "disabled_for_level": self.disabled,
                "phase": self.phase, "refinement_tokens": self.spent, "redecisions": self.redecisions,
                "episode": self.episode_id, "episode_requests": self.episode_requests,
                "episode_tokens": self.episode_tokens,
                "forced_on_level": self.forced_on_level, "events": self.events}

"""Earned reasoning tiers, legacy escalation, and stable-prefix prompting."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from .decision_policy import RequestOptions


THINK_TOOL = {
    "name": "think",
    "description": (
        "Request one xhigh reasoning response about a specific question, then return to medium. "
        "Use when evidence contradicts your model or recovery needs substantial replanning. "
        "Send think as the ONLY command in this response; it takes no game action. "
        "Only one escalation is allowed until a game action or reset supplies a fresh observation. "
        "Unavailable during bounded length-cap recovery. Routine recovery can stay in Python."
    ),
    "parameters": {
        "type": "object",
        "properties": {"question": {
            "type": "string", "minLength": 1, "maxLength": 1000,
            "description": "The specific uncertainty to resolve, relevant evidence, and decision needed."
        }},
        "required": ["question"], "additionalProperties": False,
    },
}

PROMPT = """REASONING CONTROL
Normal responses use medium reasoning. When observations contradict your understanding, recovery needs substantial
replanning, or competing explanations need careful comparison, call think(question="the specific issue to resolve").
This is a standalone harness tool, not a Python function. Send it as your ONLY command in that response. It takes no
game action and schedules exactly one xhigh response on the same observations; ordinary medium reasoning resumes
afterward. State the question, useful evidence and decision needed, not a request to repeat all previous analysis.
You may escalate only once until an executed game action or reset supplies a fresh observation. Analysis, rejected
actions and further think calls do not renew this allowance. Bounded length-cap recovery cannot request think.
Use ordinary Python recovery for straightforward failures; the deeper response must still obey every action guard,
action limit, output limit and deadline. In fenced mode use a CLOSED think block containing
JSON, for example: {"question": "Why did RIGHT fail to move the player, and which safe probe distinguishes causes?"}.

The harness also upgrades one response automatically after six consecutive responses without a game action,
or eight normal medium responses on the same level since the last escalation, whichever comes first.
An automatic reassessment can continue useful computation; it does not require an arbitrary game action.
Counters restart after either kind of escalation or a level clear. Both kinds share the same fresh-observation
allowance, and automatic escalation also waits until bounded length-cap recovery has ended.
"""


def effort_tail(effort: str) -> str:
    """Append to the final user content and retain verbatim in history."""
    instructions = {
        "medium": "Reasoning effort is set to medium. Use your normal reasoning depth.",
        "xhigh": ("Reasoning effort is set to xhigh. Please think carefully through the task, validate key "
                  "assumptions, consider plausible alternatives, and prioritize correctness, consistency, "
                  "and clarity in the final answer."),
        "low": ("Reasoning effort is set to low. Keep your thinking brief and focused, moving directly "
                "to the conclusion without unnecessary elaboration."),
    }
    return ("[Reasoning setting: applies ONLY to the immediately following assistant response]\n"
            + instructions[effort])


def patch_reasoning_template(source: str) -> str:
    """Opt-in removal of Qwen's head instruction; normal template use stays unchanged.

    The request carries effort_in_user_message=True only with our persistent tail
    instruction. Fail closed if the mounted model has an unexpected template.
    """
    original = "{%- if enable_thinking is undefined or enable_thinking is true %}"
    replacement = ("{%- if not (effort_in_user_message|default(false)) and "
                   "(enable_thinking is undefined or enable_thinking is true) %}")
    if replacement in source:
        return source
    if source.count(original) != 1 or "set reasoning_instructions = ''" not in source:
        raise ValueError("Unsupported reasoning template: cannot safely suppress the head effort instruction")
    return source.replace(original, replacement, 1)


def install_reasoning_template(model_dir, output_path) -> dict:
    """Called by the Kaggle server cell; no torch/transformers/vLLM imports."""
    import hashlib
    model_dir, output_path = Path(model_dir), Path(output_path)
    source_path = model_dir / "chat_template.jinja"
    if source_path.exists():
        source = source_path.read_text(encoding="utf-8")
    else:
        source = json.loads((model_dir / "tokenizer_config.json").read_text(encoding="utf-8"))["chat_template"]
    if not isinstance(source, str):
        raise ValueError("Expected one string chat template")
    patched = patch_reasoning_template(source)
    output_path.write_text(patched, encoding="utf-8")
    return {"path": str(output_path), "sha256": hashlib.sha256(patched.encode()).hexdigest(),
            "effort_location": "persistent_user_message_tail"}


class ReasoningPolicy:
    IDLE_TURN_LIMIT = 6
    MEDIUM_TURN_LIMIT = 8
    AUTO_QUESTION = (
        "Reassess your current plan against the observations. Identify any assumption being treated as fact "
        "without a test. If your computation is making useful progress, continue it; otherwise choose a "
        "discriminating probe. Return a concrete next step."
    )

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.state = None
        self.used = False
        self.pending_question = None
        self.question = None
        self.mode = "medium"
        self.kind = "normal"
        self.counts = Counter()
        self.events = []
        self.level = None
        self.idle_turns = 0
        self.medium_turns = 0
        self.trigger = None
        self._response_pending = False
        self._escalated_this_turn = False

    def _reset_counters(self):
        self.idle_turns = self.medium_turns = 0

    def finish_turn(self, state, *, level=None):
        """Count outcomes after execution, once per response (including terminal turns).

        The harness identity includes actual action/reset counts, so a no-op renews
        the observation but a refused action or Python analysis does not.
        """
        if not self.enabled or not self._response_pending:
            return
        self._response_pending = False
        level = getattr(state, "level", level)
        if level != self.level:
            self.level = level
            self._reset_counters()
        elif self._escalated_this_turn:
            self._reset_counters()
        elif state != self.state:
            self.idle_turns = 0
        else:
            self.idle_turns += 1

    def start_turn(self, state, recovery_options, max_tokens, *, level=None):
        # Also reconcile callers that ended the preceding turn via an early path.
        self.finish_turn(state, level=level)
        self.events = []
        self.trigger = None
        self._escalated_this_turn = False
        level = getattr(state, "level", level)
        if level != self.level:
            self.level = level
            self._reset_counters()
        if state != self.state:
            self.state, self.used, self.pending_question = state, False, None
        self.question, self.pending_question = self.pending_question, None
        self.kind = "recovery" if recovery_options else ("think" if self.question else "normal")
        self.mode = recovery_options.reasoning_effort if recovery_options else ("xhigh" if self.question else "medium")
        if not self.enabled:
            return recovery_options
        if recovery_options:
            self.question = None  # recovery limits take precedence; never raise its budget
        elif self.question:
            self.trigger = "manual"
            self._escalated_this_turn = True
            self._reset_counters()
        elif not self.used:
            if self.idle_turns >= self.IDLE_TURN_LIMIT:
                self.trigger = "idle"
            elif self.medium_turns >= self.MEDIUM_TURN_LIMIT:
                self.trigger = "periodic"
            if self.trigger:
                self.kind, self.mode, self.question = "auto_think", "xhigh", self.AUTO_QUESTION
                self.used = self._escalated_this_turn = True
                self.counts["auto_scheduled"] += 1
                self.counts["auto_" + self.trigger + "_scheduled"] += 1
                self.events.append({"event": "auto_think_scheduled", "reason": self.trigger,
                                    "idle_turns": self.idle_turns, "medium_turns": self.medium_turns})
                self._reset_counters()
        return recovery_options or RequestOptions(max_tokens, reasoning_effort=self.mode)

    def request(self, raw, *, exclusive, capped, recovery, can_continue=True):
        error = None
        question = None
        try:
            data = json.loads(raw)
            if not isinstance(data, dict) or set(data) != {"question"}:
                raise ValueError("provide exactly one question argument")
            question = data["question"]
            if not isinstance(question, str) or not 1 <= len(question.strip()) <= 1000:
                raise ValueError("question must be a non-empty string of at most 1000 characters")
            question = question.strip()
        except (ValueError, TypeError) as exc:
            error = str(exc)
        if not self.enabled:
            error = "adaptive reasoning is disabled"
        elif not exclusive:
            error = "think must be the only command; no commands in this response were executed"
        elif capped:
            error = "the response was truncated; no escalation was scheduled"
        elif recovery:
            error = "bounded length-cap recovery is active; take a justified action or submit candidates"
        elif self.used:
            error = "already used for this observation; inspect the evidence and take a checked action before escalating again"
        elif not can_continue:
            error = "no further response is available within the turn limit or deadline"
        if error:
            self.counts["think_rejected"] += 1
            self.events.append({"event": "think_rejected", "reason": error})
            return "[think] Rejected: " + error + "."
        self.used, self.pending_question = True, question
        self._escalated_this_turn = True
        self._reset_counters()
        self.counts["think_accepted"] += 1
        self.events.append({"event": "think_accepted", "question": question})
        return ("[think] Accepted. No game action taken. The next response uses xhigh reasoning, then returns to medium.\n"
                "Question to resolve: " + question)

    def record_response(self, output_tokens):
        if self.enabled:
            self._response_pending = True
            if self.kind == "normal" and self.mode == "medium":
                self.medium_turns += 1
            self.counts[self.kind + "_requests"] += 1
            self.counts[self.kind + "_output_tokens"] += max(0, output_tokens)

    def focus(self):
        if not self.question:
            return ""
        if self.kind == "auto_think":
            reason = ("Six consecutive responses without an executed game action."
                      if self.trigger == "idle" else
                      "Eight normal medium responses on this level since the last escalation.")
            return ("[Automatic reassessment: this response only]\n" + reason + "\n" + self.question
                    + "\nThis response uses xhigh; the next ordinary response returns to medium. "
                    "All action guards, output limits and deadlines still apply.")
        return ("[think: this response only]\nResolve your requested question using the existing observations and history.\n"
                "Question: " + self.question + "\nReturn a concrete decision, checked action, or discriminating probe. "
                "Do not call think again before a fresh game observation.")

    def snapshot(self):
        return {"enabled": self.enabled, "mode": self.mode, "kind": self.kind, "question": self.question,
                "used_for_observation": self.used, "pending_question": self.pending_question,
                "trigger": self.trigger, "idle_turns": self.idle_turns, "medium_turns": self.medium_turns,
                "idle_turn_limit": self.IDLE_TURN_LIMIT, "medium_turn_limit": self.MEDIUM_TURN_LIMIT,
                "events": list(self.events)}


TIERED_PROMPT = """REASONING AND OUTPUT BUDGET
Each new level starts with xhigh reasoning and up to 12,000 output tokens, including thinking and commands.
After two executed actions whose nontrivial predictions are checked successfully, the harness switches to
medium reasoning with up to 8,000 tokens. Evidence comes from existing optional expect checks (moved/color)
or path steps matching next_state predictions; unchecked actions and unchanged boards do not earn a downshift.
These checks are optional: exploration and all existing action tools remain available without them.
A failed check, reset, new level, changed model, or two stalled execution responses restores xhigh/12K planning.
In the final 15% of the available gameplay window, ordinary requests use medium/4K for the rest of the game.
Use the budget shown for this response. Keep reasoning concise under time pressure, use your existing evidence,
prioritize completing your best-supported plan, and leave enough tokens for a complete command.
Bounded length-cap recovery keeps its own smaller limits and takes precedence.
think(question) remains a standalone tool: send it as the ONLY command to request focused replanning.
It takes no action, requires a non-empty question of at most 1,000 characters, and is allowed only once per
fresh action/reset observation. It cannot escalate during recovery or the final 4K tier. In fenced mode use
a closed think block containing JSON, e.g. {"question": "Which observation invalidated my plan?"}.
"""

TIERED_THINK_TOOL = {**THINK_TOOL, "description": (
    "Request focused xhigh/12K replanning; execution returns to medium/8K only after fresh checked progress. "
    "Send think as the ONLY command. It takes no action and is allowed once per fresh action/reset observation. "
    "Unavailable during bounded length-cap recovery or the final medium/4K time tier."
)}


def execution_evidence(context, model_source=""):
    """Read only actual execution records; never infer plan success from prose or action counts."""
    verified = 0
    failure = None
    progress = False
    for step in context.steps:
        changed = (step.get("changed_outside_indicator") or 0) > 0
        progress |= changed
        checks = (step.get("object_refs") or {}).get("expectations") or []
        prediction = (step.get("prediction_check") or {}).get("status")
        if step.get("outcome") in {"game_over", "env_error"} or step.get("unreliable"):
            failure = "unreliable_or_terminal_action"
        if any(q.get("status") != "satisfied" for q in checks) or prediction in {"contradicted", "error"}:
            failure = "prediction_failed"
        meaningful = any(q.get("test") == "color" or
                         (q.get("test") == "moved" and q.get("delta") != [0, 0]) for q in checks)
        if changed and not step.get("unreliable") and (
                prediction == "satisfied" or
                (meaningful and all(q.get("status") == "satisfied" for q in checks))):
            verified += 1  # one action, even if several predicates passed
    if context.prediction_mismatch:
        failure = "prediction_failed"
    validation = getattr(context, "validation", None) or {}
    if validation.get("differ") or validation.get("error"):
        failure = "model_validation_failed"
    if any(g.get("kind") in {"new_color_latch", "batch_new_color", "known_refused", "act_refused",
                             "act_rejected_action", "prediction_mismatch"} for g in context.guards):
        failure = "execution_interrupted"
    if any(call.get("error") for call in context.run_calls):
        failure = "run_error"
    return {"verified_actions": verified, "progress": progress, "failure": failure,
            "model_source": model_source}


class TieredReasoningPolicy(ReasoningPolicy):
    """Per-game earned downshift; wall-clock deadline overrides are sticky and request-local."""
    PLANNING_CAP = 12000
    EXECUTION_CAP = 8000
    FINISH_CAP = 4000
    FINISH_FRACTION = 0.15
    VERIFIED_ACTIONS_REQUIRED = 2
    STALL_TURNS = 2

    def __init__(self, enabled):
        super().__init__(enabled)
        self.tier = self.request_tier = "planning"
        self.mode = "xhigh"
        self.finishing = False
        self.verified_actions = self.stalled_turns = 0
        self.model_source = ""
        self.output_cap = self.PLANNING_CAP
        self.remaining_seconds = self.budget_seconds = None

    def _transition(self, tier, reason):
        if self.finishing and tier != "finish":
            return
        if tier != self.tier:
            self.events.append({"event": "tier_changed", "from": self.tier, "to": tier, "reason": reason})
            self.counts["tier_to_" + tier] += 1
        self.tier = tier

    def _replan(self, reason):
        self.verified_actions = self.stalled_turns = 0
        self._transition("planning", reason)

    def finish_turn(self, state, *, level=None, evidence=None):
        if not self.enabled or not self._response_pending:
            return
        self._response_pending = False
        evidence = evidence or {}
        level = getattr(state, "level", level)
        scope_changed = (level != self.level or
                         getattr(state, "attempt", None) != getattr(self.state, "attempt", None))
        model_source = evidence.get("model_source", self.model_source)
        model_changed = model_source != self.model_source
        self.model_source = model_source
        self.level = level
        if scope_changed:
            self._replan("level_or_reset")
        elif evidence.get("failure"):
            self._replan(evidence["failure"])
        else:
            if model_changed:
                self._replan("model_changed")
            # Recovery is action-focused; it cannot earn a lower ordinary reasoning tier.
            if self.kind != "recovery":
                self.verified_actions += evidence.get("verified_actions", 0)
                if self.verified_actions >= self.VERIFIED_ACTIONS_REQUIRED:
                    self._transition("execution", "checked_progress")
            if self.tier == "execution":
                self.stalled_turns = 0 if evidence.get("progress") else self.stalled_turns + 1
                if self.stalled_turns >= self.STALL_TURNS:
                    self._replan("execution_stalled")

    def start_turn(self, state, recovery_options, max_tokens, *, level=None,
                   remaining_seconds=None, budget_seconds=None):
        self.finish_turn(state, level=level)
        self.events = []
        self.trigger = None
        level = getattr(state, "level", level)
        if level != self.level or getattr(state, "attempt", None) != getattr(self.state, "attempt", None):
            self._replan("level_or_reset")
        self.level = level
        if state != self.state:
            self.state, self.used, self.pending_question = state, False, None
        self.remaining_seconds, self.budget_seconds = remaining_seconds, budget_seconds
        if (self.enabled and remaining_seconds is not None and budget_seconds is not None
                and budget_seconds > 0 and remaining_seconds <= budget_seconds * self.FINISH_FRACTION):
            self.finishing = True
            self._transition("finish", "last_15_percent")
            self.pending_question = None
        self.question, self.pending_question = self.pending_question, None
        self.kind = "recovery" if recovery_options else ("think" if self.question else "normal")
        if not self.enabled:
            return recovery_options
        if recovery_options:
            self.question = None
            self.request_tier = "recovery"
            self.mode, self.output_cap = recovery_options.reasoning_effort, recovery_options.max_output_tokens
            return recovery_options
        if self.question:
            self._replan("manual_think")
        self.request_tier = self.tier
        self.mode = "xhigh" if self.tier == "planning" else "medium"
        self.output_cap = min(max_tokens, {"planning": self.PLANNING_CAP, "execution": self.EXECUTION_CAP,
                                          "finish": self.FINISH_CAP}[self.tier])
        return RequestOptions(self.output_cap, reasoning_effort=self.mode)

    def request(self, raw, **kwargs):
        if self.finishing:
            self.counts["think_rejected"] += 1
            self.events.append({"event": "think_rejected", "reason": "final_4k_time_tier"})
            return "[think] Rejected: final time tier is active; use concise medium reasoning within 4K."
        result = super().request(raw, **kwargs)
        if self.pending_question:
            self._replan("manual_think")
        return result.replace("then returns to medium", "with up to 12K tokens; checked progress earns medium/8K")

    def record_response(self, output_tokens):
        super().record_response(output_tokens)
        if self.enabled:
            if self.request_tier != "recovery":
                self.counts[self.request_tier + "_requests"] += 1
                self.counts[self.request_tier + "_output_tokens"] += max(0, output_tokens)

    def focus(self):
        if not self.enabled:
            return ""
        if self.request_tier == "recovery":
            return "[Reasoning budget] Bounded recovery limits apply. " + (
                "After recovery, the final medium/4K time tier remains active." if self.finishing else
                "Ordinary tier selection resumes after recovery.")
        text = (f"[Reasoning budget: this response only] {self.request_tier}; {self.mode}; "
                f"at most {self.output_cap:,} output tokens INCLUDING thinking and commands. ")
        if self.finishing:
            text += (f"About {max(0, self.remaining_seconds or 0) / 60:.1f} minutes remain in the gameplay window. "
                     "Keep reasoning concise, prioritize completing your best-supported plan using existing evidence, "
                     "and leave room for a complete command. This final time tier cannot escalate.")
        elif self.tier == "planning":
            text += "Resolve the uncertainty and test your plan. Checked progress earns medium/8K execution."
        else:
            text += "Execute the tested plan; check outcomes and revise it if observations contradict it."
        if self.question:
            text += "\nReplanning question: " + self.question
        return text

    def snapshot(self):
        return {**super().snapshot(), "policy": "earned_tiers", "tier": self.request_tier, "next_tier": self.tier,
                "output_cap": self.output_cap, "finishing": self.finishing, "verified_actions": self.verified_actions,
                "stalled_turns": self.stalled_turns, "remaining_seconds": self.remaining_seconds,
                "budget_seconds": self.budget_seconds}


def reasoning_configuration(enabled, tiered=False):
    if tiered:
        return {"enabled": enabled, "policy": "earned_tiers", "default": "xhigh", "think": "xhigh",
                "caps": {"planning": 12000, "execution": 8000, "finish": 4000},
                "finish_fraction": TieredReasoningPolicy.FINISH_FRACTION,
                "verified_actions_required": TieredReasoningPolicy.VERIFIED_ACTIONS_REQUIRED,
                "stall_turns": TieredReasoningPolicy.STALL_TURNS, "automatic": False}
    return {"enabled": enabled, "default": "medium", "think": "xhigh", "automatic": enabled,
            "idle_turn_limit": ReasoningPolicy.IDLE_TURN_LIMIT, "medium_turn_limit": ReasoningPolicy.MEDIUM_TURN_LIMIT}

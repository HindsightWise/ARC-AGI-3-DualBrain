"""Small, environment-free live check of Qwen's recovery tool/budget protocol."""
from __future__ import annotations

from dataclasses import asdict
import time

from .decision_policy import DecisionIdentity, DecisionPolicy, RecoverySettings
from .llm import create_openai_client, parse_response_commands, request_llm_response
from .prompts import build_system_prompt


def run_smoke_test():
    settings = RecoverySettings.from_env()
    if settings.mode != "recover_after_cap":
        return {"status": "skipped", "mode": settings.mode}
    client = create_openai_client()
    records = []
    remaining = settings.max_episode_tokens
    # Same medium/thinking/temperature and request/episode caps as gameplay.
    # Retry once without silently raising the configured budget or disabling thinking.
    for attempt in range(1, 3):
        if remaining < settings.request_floor:
            break
        started = time.monotonic()
        state = DecisionIdentity("protocol-smoke", 1, 1, 0, "synthetic")
        policy = DecisionPolicy(settings)
        policy.start_turn(state)
        options = settings.request_options(min(settings.proposal_tokens, remaining))
        budget = options.max_output_tokens
        result = request_llm_response(
            client, build_system_prompt(["LEFT", "RIGHT"], board_shape=(8, 8)),
            [{"role": "user", "content": "Protocol smoke test only: no environment actions will be executed. "
              "A piece is at [3,3] on an empty 8x8 board. RIGHT is legal and has not been tried."},
             {"role": "assistant", "content": "RIGHT is a low-risk probe of horizontal movement."}],
            "Call propose_actions now with exactly one candidate: action RIGHT, score 5, purpose probe, "
            "risk low, short evidence and expected observation. Do not call run/actions or write an explanation.",
            available_action_labels=["LEFT", "RIGHT"], request_options=options, include_proposal_tool=True,
            remaining_seconds=60, quiet=True, quiet_label="recovery-smoke")
        _, commands = parse_response_commands(result.text, tool_calls=result.tool_calls)
        remaining -= max(0, result.usage.output_tokens)
        proposals = [c.content for c in commands if c.kind == "propose_actions"]
        good = (result.finish_reason != "length" and result.usage.output_tokens < budget - 2
                and len(commands) == len(proposals) == 1
                and policy._accept(proposals[0], lambda a: None if a == "RIGHT" else "expected RIGHT"))
        records.append({"attempt": attempt, "budget": budget, "request_options": asdict(options),
                        "seconds": round(time.monotonic() - started, 2), "passed": bool(good),
                        "finish_reason": result.finish_reason, "usage": result.usage.to_dict(),
                        "commands": [{"kind": c.kind, "content": c.content} for c in commands],
                        "policy_events": policy.events})
        if good:
            return {"status": "passed", "settings": asdict(settings), "attempts": records}
    return {"status": "failed", "settings": asdict(settings), "attempts": records,
            "error": "Qwen did not emit a valid proposal with the configured thinking/temperature/budget "
                     "within two attempts; gameplay was not started."}

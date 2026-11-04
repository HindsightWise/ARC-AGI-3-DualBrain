"""Small real-vLLM / real-sandbox check before timed minimal-harness comparisons."""
from __future__ import annotations

import json
import numpy as np


def run_smoke_test():
    from . import config as cfg, harness_ext as ext, llm
    from .sandbox import VirtualCodeFile
    from .prompts import build_system_prompt
    from .thinking_budget import THINKING_END, thinking_token_budget
    assert cfg.MINIMAL_HARNESS and cfg.MODEL_THINKING_EFFORT == "xhigh"
    assert cfg.MAX_OUTPUT_TOKENS == 14000 and not cfg.ADAPTIVE_REASONING
    assert thinking_token_budget(cfg.MAX_OUTPUT_TOKENS) == 12000
    assert [s["name"] for s in llm._command_tool_specs(["RIGHT", "CLICK"])] == ["run"]
    serving = None
    if cfg.IS_KAGGLE:
        from pathlib import Path
        serving = json.loads(Path("/kaggle/working/vllm-server-identity.json").read_text())
        command = serving["command"]
        def option(name):
            return command[command.index(name) + 1]
        spec = json.loads(option("--speculative-config"))
        if (not serving.get("draft_vocab") or not spec.get("index_share_for_mtp_iteration")
                or not spec.get("use_local_argmax_reduction") or spec.get("num_speculative_tokens") != 3
                or int(option("--max-num-batched-tokens")) != 3392
                or int(option("--max-num-seqs")) != cfg.MAX_CONCURRENT_REQUESTS):
            raise RuntimeError("minimal smoke: requested serving optimisations/stream count are not active")
        if json.loads(option("--reasoning-config")).get("reasoning_end_str") != THINKING_END:
            raise RuntimeError("minimal smoke: wrong thinking-budget transition")
    board = np.zeros((64, 64), dtype=np.uint8)
    board[8:16, 8:16] = 2
    ctx = ext.start_game("minimal-smoke", board)
    code_file = VirtualCodeFile(source="", loader=ext.code_loader)
    prompt = build_system_prompt(["RIGHT", "CLICK"])
    client = llm.create_openai_client()
    first = llm.request_llm_response(
        client, prompt, [],
        'This is a smoke test. Call run exactly once with code:\n'
        'notes["smoke"] = "retained"\nshow(render(current_board, rect=(8,16,8,16), title="smoke crop"))\n'
        'notes["value"] = int(current_board[8,8])\nDo not take game actions.',
        board_for_small_image=board, available_action_labels=["RIGHT", "CLICK"], quiet=True,
        quiet_label="minimal-smoke", remaining_seconds=180)
    _, commands = llm.parse_response_commands(first.text, first.tool_calls)
    if len(commands) != 1 or commands[0].kind != "run":
        raise RuntimeError("minimal smoke: expected one native run call")
    output, error, info = ext.run_code(code_file, commands[0].content, board[None], [], None, None, [],
                                       store={}, action_labels=["RIGHT", "CLICK"])
    if error or not info.get("memory"):
        raise RuntimeError(f"minimal smoke worker failed: {error}; {output}")
    ctx.memory = info["memory"]
    if ctx.memory["notes"].get("smoke") != "retained" or ctx.memory["notes"].get("value") != 2:
        raise RuntimeError("minimal smoke: memory values did not round-trip")
    images = list(ctx.shown_images)
    if len(images) != 1:
        raise RuntimeError("minimal smoke: rendered crop missing")
    text = output
    ext.prepare_request_images(board, False)
    conversation = [{"role": "user", "content": first.user_content_for_history},
                    first.assistant_message_for_history]
    if not first.tool_calls:
        raise RuntimeError("minimal smoke: native tool-call protocol missing")
    conversation.append({"role": "tool", "tool_call_id": first.tool_calls[0].call_id,
                         "name": "run", "content": text})
    second = llm.request_llm_response(
        client, prompt, conversation,
        'The image labelled show[1] is the crop you rendered. '
        'Call run exactly once: assert notes["smoke"] == "retained"; notes["smoke_ok"] = True',
        board_for_small_image=board, available_action_labels=["RIGHT", "CLICK"], quiet=True,
        quiet_label="minimal-smoke", remaining_seconds=180)
    _, commands = llm.parse_response_commands(second.text, second.tool_calls)
    if len(commands) != 1 or commands[0].kind != "run" or len(second.image_attachments) != 2:
        raise RuntimeError("minimal smoke: image feedback/native run round-trip failed")
    output, error, info = ext.run_code(code_file, commands[0].content, board[None], [], None, None, [],
                                       store={}, action_labels=["RIGHT", "CLICK"])
    if error or not info.get("memory", {}).get("notes", {}).get("smoke_ok"):
        raise RuntimeError(f"minimal smoke second run failed: {error}; {output}")
    ctx.memory = info["memory"]
    from .decision_policy import DecisionPolicy, RecoverySettings, identity
    policy = DecisionPolicy(RecoverySettings(mode="recover_after_cap"), minimal=True)
    state = identity("minimal-smoke", 1, 1, 0, board)
    policy.start_turn(state)
    policy.process(options=None, output_tokens=cfg.MAX_OUTPUT_TOKENS, capped=True, commands=[], proposals=[],
                   state=state, already_acted=False, env_intent=False, evidence=[], validate=lambda _: None)
    options = policy.start_turn(state)
    conversation.extend([
        {"role": "user", "content": second.user_content_for_history}, second.assistant_message_for_history,
        {"role": "tool", "tool_call_id": second.tool_calls[0].call_id, "name": "run", "content": "smoke_ok = True"},
    ])
    proposal = {"action": "RIGHT", "score": 4, "purpose": "probe", "expected": "test movement",
                "evidence": "direction is untested", "risk": "low"}
    third = llm.request_llm_response(
        client, prompt, conversation,
        policy.prompt() + "\nSmoke test: think about the untested direction, then call run exactly once with code:\n"
        + f"propose_actions({[proposal]!r})",
        board_for_small_image=board, available_action_labels=["RIGHT", "CLICK"], quiet=True,
        quiet_label="minimal-recovery-smoke", remaining_seconds=180,
        request_options=options, thinking_budget_override=1)
    # Check actual generation, not just whether the API silently accepts the setting.
    if THINKING_END.removesuffix("</think>").strip() not in third.reasoning_content:
        raise RuntimeError("minimal smoke: thinking-budget transition was not forced by vLLM")
    _, commands = llm.parse_response_commands(third.text, third.tool_calls)
    if not third.tool_calls or len(commands) != 1 or commands[0].kind != "run" or third.finish_reason == "length":
        raise RuntimeError("minimal smoke: forced thinking exit did not leave a complete native run call")
    output, error, info = ext.run_code(code_file, commands[0].content, board[None], [], None, None, [],
                                       store={}, action_labels=["RIGHT", "CLICK"])
    policy.record_response(options, third.usage.output_tokens)
    policy.process(options=options, output_tokens=third.usage.output_tokens, capped=False, commands=commands,
                   proposals=info.get("proposals", []), state=state, already_acted=False, env_intent=False,
                   evidence=[output] if output.strip() else [],
                   validate=lambda action: None if action == "RIGHT" else "unexpected smoke action",
                   analysis_failed=bool(error))
    if error or not policy.candidates:
        raise RuntimeError(f"minimal smoke: Python proposal was not accepted: {error}; {policy.events}")
    return {"status": "passed", "tools": ["run"], "model": cfg.MODEL_NAME, "effort": cfg.MODEL_THINKING_EFFORT,
            "max_output_tokens": cfg.MAX_OUTPUT_TOKENS, "rendered_feedback_images": len(images),
            "thinking_token_budget": thinking_token_budget(cfg.MAX_OUTPUT_TOKENS),
            "recovery_thinking_token_budget": thinking_token_budget(options.max_output_tokens, recovery=True),
            "forced_thinking_exit": True, "proposal_interface": "run:propose_actions",
            "serving": serving,
            "recovery_request_cap": options.max_output_tokens, "recovery_effort": options.reasoning_effort,
            "output_tokens": sum(r.usage.output_tokens for r in (first, second, third)),
            "requests": [{"input_tokens": r.usage.input_total_tokens, "cached_tokens": r.usage.input_cached_tokens,
                          "output_tokens": r.usage.output_tokens} for r in (first, second, third)]}

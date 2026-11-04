#!/usr/bin/env python3
"""
Empirical verification script for Milestone 1 Challenger 2.
Audits telemetry figures in docs/FORENSIC_ROOT_CAUSE_V21.md against raw files in /tmp/v21_output/.
"""

import os
import re
import glob
import json
import numpy as np

def audit_serve_log(path="/tmp/v21_output/serve.log"):
    print("=" * 80)
    print("1. AUDITING /tmp/v21_output/serve.log")
    print("=" * 80)

    pattern = re.compile(
        r"ReqTimeStats\(rid=(?P<rid>[^,]+),\s*"
        r"input_len=(?P<input_len>\d+),\s*"
        r"cached_input_len=(?P<cached_input_len>\d+),\s*"
        r"output_len=(?P<output_len>\d+)[^)]*\):\s*"
        r".*?forward_duration=(?P<forward_duration>[\d\.]+)ms"
    )

    records = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_num, line in enumerate(f, 1):
            if "ReqTimeStats" in line:
                m = pattern.search(line)
                if m:
                    gd = m.groupdict()
                    records.append({
                        "line": line_num,
                        "rid": gd["rid"],
                        "input_len": int(gd["input_len"]),
                        "cached_input_len": int(gd["cached_input_len"]),
                        "output_len": int(gd["output_len"]),
                        "forward_duration_s": float(gd["forward_duration"]) / 1000.0,
                    })
                else:
                    print(f"WARNING: Matched ReqTimeStats on line {line_num} but failed regex parse: {line.strip()}")

    total_requests = len(records)
    print(f"Total requests parsed: {total_requests}")

    total_input = sum(r["input_len"] for r in records)
    total_cached = sum(r["cached_input_len"] for r in records)
    cache_hit_rate = (total_cached / total_input * 100.0) if total_input > 0 else 0.0

    total_output = sum(r["output_len"] for r in records)

    input_lens = np.array([r["input_len"] for r in records])
    output_lens = np.array([r["output_len"] for r in records])
    forward_durations = np.array([r["forward_duration_s"] for r in records])

    print(f"Total Input Tokens: {total_input:,}")
    print(f"Total Cached Tokens: {total_cached:,}")
    print(f"Overall Prefix Cache Hit Rate: {cache_hit_rate:.2f}% ({cache_hit_rate:.4f}%)")
    print(f"Total Output Tokens: {total_output:,}")

    print("\nInput Token Stats:")
    print(f"  Mean: {np.mean(input_lens):.1f}")
    print(f"  Median (p50): {np.median(input_lens):.1f}")
    print(f"  p90: {np.percentile(input_lens, 90):.1f}")
    print(f"  p99: {np.percentile(input_lens, 99):.1f}")
    print(f"  Max: {np.max(input_lens)}")

    print("\nOutput Token Stats:")
    print(f"  Mean: {np.mean(output_lens):.1f}")
    print(f"  Median (p50): {np.median(output_lens):.1f}")
    print(f"  p90: {np.percentile(output_lens, 90):.1f}")
    print(f"  p99: {np.percentile(output_lens, 99):.1f}")
    print(f"  Max: {np.max(output_lens)}")

    print("\nForward Duration Stats (seconds):")
    print(f"  Mean: {np.mean(forward_durations):.2f}s")
    print(f"  Median (p50): {np.median(forward_durations):.2f}s")
    print(f"  p90: {np.percentile(forward_durations, 90):.2f}s")
    print(f"  p99: {np.percentile(forward_durations, 99):.2f}s")
    print(f"  Max: {np.max(forward_durations):.2f}s")

    # Also check if health check requests affect anything
    health_checks = [r for r in records if "HEALTH_CHECK" in r["rid"]]
    print(f"\nHealth check requests count: {len(health_checks)}")

    return {
        "total_requests": total_requests,
        "total_input": total_input,
        "total_cached": total_cached,
        "cache_hit_rate": cache_hit_rate,
        "total_output": total_output,
        "input_lens": input_lens,
        "output_lens": output_lens,
        "forward_durations": forward_durations,
        "records": records,
    }

def audit_apex_engine_log(path="/tmp/v21_output/arc-prize-2026-v21-apex-engine.log"):
    print("\n" + "=" * 80)
    print("2. AUDITING /tmp/v21_output/arc-prize-2026-v21-apex-engine.log")
    print("=" * 80)

    timeout_pattern = re.compile(
        r"analyzer request timed out at action (?P<action>\d+);\s*"
        r"yielding with (?P<preserved>\d+) tool call\(s\) preserved:\s*"
        r".*?\(read timeout=(?P<timeout>[\d\.]+)\)"
    )

    matches = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_num, line in enumerate(f, 1):
            if "read timeout=" in line or "yielding with" in line:
                m = timeout_pattern.search(line)
                if m:
                    matches.append((line_num, line.strip(), m.groupdict()))
                else:
                    print(f"Line {line_num} has timeout/yield keyword but didn't match regex: {line.strip()[:100]}")

    print(f"Found {len(matches)} timeout occurrences:")
    for line_num, raw, gd in matches:
        print(f"  Line {line_num}: Action {gd['action']}, Preserved Calls: {gd['preserved']}, Timeout: {gd['timeout']}s")
        print(f"    Raw snippet: {raw}")

    # Inspect lines 245 to 275 in detail
    print("\nContext lines 245-275 from apex engine log:")
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for idx, line in enumerate(f, 1):
            if 245 <= idx <= 275:
                print(f"  [{idx}] {line.rstrip()}")

def audit_transcripts(transcripts_dir="/tmp/v21_output/transcripts/"):
    print("\n" + "=" * 80)
    print("3. AUDITING TRANSCRIPTS IN " + transcripts_dir)
    print("=" * 80)

    files = sorted(glob.glob(os.path.join(transcripts_dir, "*.txt")))
    print(f"Found {len(files)} transcript files: {[os.path.basename(f) for f in files]}")

    tools_to_check = [
        "check_transitions",
        "compare_rules",
        "edge_meter",
        "measure_edge_meters",
        "animation_archive",
        "animation_history",
        "animation_at",
        "rot90",
        "rot180",
        "rot270",
        "flipv",
        "fliph",
        "find_cells",
        "find_objects",
    ]

    counts = {tool: 0 for tool in tools_to_check}
    tool_mentions = {tool: 0 for tool in tools_to_check}
    total_actions = 0
    total_code_blocks = 0
    fable_indicator_count = 0
    fable_budget_count = 0
    fable_pinned_count = 0

    code_block_pattern = re.compile(r"<parameter=code>(.*?)</parameter>", re.DOTALL)
    # Also check other parameter patterns if any: e.g. <tool_call>, ```python, etc.
    action_pattern = re.compile(r"action\s+\d+|Action\s+\d+|ACTION\s+\d+")

    for file_path in files:
        fname = os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        # Count total actions in transcript
        # Let's see how actions are marked in transcripts
        # Check mentions of Fable
        fable_indicator_count += content.count("[fable_layer] Indicator shift detected")
        fable_indicator_count += content.count("fable") # let's search specifically
        if "fable" in content.lower():
            # count fable specific phrases
            pass

        # Check code blocks
        code_blocks = code_block_pattern.findall(content)
        total_code_blocks += len(code_blocks)

        for cb in code_blocks:
            for tool in tools_to_check:
                if re.search(r"\b" + re.escape(tool) + r"\b", cb):
                    counts[tool] += 1

        for tool in tools_to_check:
            # Mentions in whole file vs code blocks
            tool_mentions[tool] += len(re.findall(r"\b" + re.escape(tool) + r"\b", content))

    print(f"Total parameter=code blocks across all transcripts: {total_code_blocks}")
    print("\nTool Invocation Counts inside executable code blocks (<parameter=code>):")
    for tool in tools_to_check:
        print(f"  {tool:25s}: {counts[tool]:4d} calls ({counts[tool]/total_code_blocks*100.0:.2f}%)")

    print("\nRaw occurrences across full transcripts (including prompts/system text):")
    for tool in tools_to_check:
        print(f"  {tool:25s}: {tool_mentions[tool]:4d} raw occurrences")

if __name__ == "__main__":
    serve_stats = audit_serve_log()
    audit_apex_engine_log()
    audit_transcripts()

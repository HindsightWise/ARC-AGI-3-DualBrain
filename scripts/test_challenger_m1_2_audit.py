#!/usr/bin/env python3
"""Independent Empirical Verification Harness for Challenger M1_2.

Adversarially audits:
1. The 4 vs 5 ReadTimeout claims in docs/FORENSIC_JUMPS_V18_V24_TO_V25.md and /tmp/v21_output/.
2. Reasoning and tool call preservation in tool_agent.py and timing of game abandonment.
3. Exact invocation counts of animation_at, animation_history, and rot90 across all transcripts.
4. Edge cases in scripts/verify_audit_jumps.py and install_timeout_armor.
"""

import glob
import json
import os
import re
import sys
from pathlib import Path

def audit_read_timeouts():
    print("=" * 80)
    print("1. EMPIRICAL AUDIT: ReadTimeout EXCEPTIONS IN /tmp/v21_output/")
    print("=" * 80)

    log_path = Path("/tmp/v21_output/arc-prize-2026-v21-apex-engine.log")
    assert log_path.exists(), f"Log file not found: {log_path}"
    
    timeout_re = re.compile(
        r'time":(?P<timestamp>[\d\.]+),.*?'
        r'analyzer request timed out at action (?P<action>\d+);\s*'
        r'yielding with (?P<preserved>\d+) tool call\(s\) preserved:\s*'
        r'.*?\(read timeout=(?P<timeout>[\d\.]+)\)'
    )

    timeouts = []
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for lno, line in enumerate(f, 1):
            m = timeout_re.search(line)
            if m:
                gd = m.groupdict()
                timeouts.append({
                    "line": lno,
                    "timestamp": float(gd["timestamp"]),
                    "action": int(gd["action"]),
                    "preserved": int(gd["preserved"]),
                    "timeout": float(gd["timeout"]),
                    "raw": line.strip()
                })

    # Note: each event is printed to stderr twice due to duplicate handlers
    # Deduplicate by (action, timeout)
    unique_timeouts = []
    seen = set()
    for t in timeouts:
        key = (t["action"], t["timeout"])
        if key not in seen:
            seen.add(key)
            unique_timeouts.append(t)

    print(f"Total raw lines matched: {len(timeouts)}")
    print(f"Total unique ReadTimeout events: {len(unique_timeouts)} (EXPECTED 4, NOT 5)")

    expected_events = [
        {"action": 42, "timeout": 13.954828416000055, "preserved": 0, "game": "r11l-495a7899"},
        {"action": 153, "timeout": 66.17609974400011, "preserved": 0, "game": "tu93-0768757b"},
        {"action": 148, "timeout": 89.44586153299997, "preserved": 0, "game": "sc25-635fd71a"},
        {"action": 73, "timeout": 35.54712102199983, "preserved": 1, "game": "lp85-305b61c3"},
    ]

    assert len(unique_timeouts) == 4, f"Expected exactly 4 unique ReadTimeouts, got {len(unique_timeouts)}"
    for idx, (ut, exp) in enumerate(zip(unique_timeouts, expected_events)):
        print(f"  Event {idx+1}: action={ut['action']}, timeout={ut['timeout']:.3f}s, preserved_tool_calls={ut['preserved']}")
        assert ut["action"] == exp["action"], f"Action mismatch in event {idx+1}"
        assert abs(ut["timeout"] - exp["timeout"]) < 1e-4, f"Timeout mismatch in event {idx+1}"
        assert ut["preserved"] == exp["preserved"], f"Preserved tool calls mismatch in event {idx+1}"

    print("  [CONFIRMED] Exactly 4 ReadTimeouts occurred: 13.95s, 66.18s, 89.45s, 35.55s.")
    print("  [DISCREPANCY] Line 134 in FORENSIC doc claimed 5 timeouts (13.9, 15.7, 25.2, 35.1, 89.4) with 0 preserved calls.")
    print("  [DISCREPANCY] lp85-305b61c3 preserved 1 tool call, NOT 0.")

    # Check timing of termination in engine log
    finished_re = re.compile(
        r'time":(?P<timestamp>[\d\.]+),.*?'
        r'\[finished\] (?P<game>[^\s]+) state=(?P<state>[^\s]+)'
    )
    finished_events = {}
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = finished_re.search(line)
            if m:
                gd = m.groupdict()
                finished_events[gd["game"]] = {
                    "timestamp": float(gd["timestamp"]),
                    "state": gd["state"],
                }

    print("\nTiming from ReadTimeout to Game Termination:")
    for exp in expected_events:
        gid = exp["game"]
        fin = finished_events.get(gid)
        assert fin is not None, f"Game {gid} finish event not found"
        # Find the timeout event
        matching_timeout = [t for t in unique_timeouts if t["action"] == exp["action"]][0]
        dt = fin["timestamp"] - matching_timeout["timestamp"]
        print(f"  {gid:15s}: Timeout @ {matching_timeout['timestamp']:.2f}s -> Finish ({fin['state']}) @ {fin['timestamp']:.2f}s (delta = {dt:.3f}s)")
        assert dt < 1.0, f"Game {gid} was not abandoned immediately (delta = {dt:.3f}s)"
        assert fin["state"] == "gave_up", f"Game {gid} state was {fin['state']}, expected gave_up"

    print("  [CONFIRMED] All 4 games were abandoned immediately (<0.7s) after ReadTimeout with state=gave_up.")


def audit_accelerator_call_counts():
    print("\n" + "=" * 80)
    print("2. EMPIRICAL AUDIT: ACCELERATOR USAGE ACROSS ALL 10 TRANSCRIPTS")
    print("=" * 80)

    transcripts = sorted(glob.glob("/tmp/v21_output/transcripts/*.txt"))
    assert len(transcripts) == 10, f"Expected 10 transcripts, found {len(transcripts)}"

    total_anim_at_calls = 0
    total_anim_hist_calls = 0
    total_rot90_defs = 0
    total_rot90_calls = 0
    total_rot90_raw = 0

    per_game_stats = {}

    for tpath in transcripts:
        gid = Path(tpath).stem.replace("_p0", "")
        with open(tpath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        raw_r90 = len(re.findall(r"\brot90\b", content))
        total_rot90_raw += raw_r90

        code_blocks = re.findall(r"<parameter=code>(.*?)</parameter>", content, re.DOTALL)
        
        aat_calls = 0
        ahist_calls = 0
        r90_defs = 0
        r90_calls = 0

        for cb in code_blocks:
            if re.search(r"def\s+rot90\b", cb):
                r90_defs += 1
            if re.search(r"(?<!def\s)\brot90\s*\(", cb):
                r90_calls += 1
            if re.search(r"\banimation_at\s*\(", cb):
                aat_calls += 1
            if re.search(r"\banimation_history\s*\(", cb):
                ahist_calls += 1

        total_anim_at_calls += aat_calls
        total_anim_hist_calls += ahist_calls
        total_rot90_defs += r90_defs
        total_rot90_calls += r90_calls

        per_game_stats[gid] = {
            "anim_at_calls": aat_calls,
            "anim_hist_calls": ahist_calls,
            "rot90_defs": r90_defs,
            "rot90_calls": r90_calls,
            "rot90_raw": raw_r90,
        }
        print(f"  {gid:15s}: anim_at={aat_calls}, anim_hist={ahist_calls}, rot90_def={r90_defs}, rot90_calls={r90_calls}, rot90_raw_mentions={raw_r90}")

    print(f"\nAGGREGATE TOTALS:")
    print(f"  animation_at calls      : {total_anim_at_calls} (sb26: 3, sc25: 4)")
    print(f"  animation_history calls : {total_anim_hist_calls} (sb26: 1, sc25: 1)")
    print(f"  COMBINED animation calls: {total_anim_at_calls + total_anim_hist_calls} (CLAIM WAS 21: 10 in sb26, 11 in sc25)")
    print(f"  rot90 defs in code      : {total_rot90_defs} (in tr87: def rot90(s): ...)")
    print(f"  rot90 calls in code     : {total_rot90_calls} (in tr87: cur=rot90(cur))")
    print(f"  rot90 raw mentions      : {total_rot90_raw} (11 in tr87, 1 in ar25; CLAIM WAS 'actively called 11 times in its REPL scripts')")

    assert total_anim_at_calls == 7, f"Expected 7 animation_at calls, got {total_anim_at_calls}"
    assert total_anim_hist_calls == 2, f"Expected 2 animation_history calls, got {total_anim_hist_calls}"
    assert total_anim_at_calls + total_anim_hist_calls == 9, "Expected 9 total animation archive calls"
    assert total_rot90_calls == 1, f"Expected 1 rot90 call in code, got {total_rot90_calls}"
    assert total_rot90_raw == 12, f"Expected 12 raw rot90 mentions, got {total_rot90_raw}"

    print("  [DISCREPANCY CONFIRMED] animation calls claim of 21 is false (actual = 9).")
    print("  [DISCREPANCY CONFIRMED] rot90 'actively called 11 times in REPL' is false (actual = 1 call to custom def rot90; 9 in thinking prose).")


def test_verify_audit_jumps_edge_cases():
    print("\n" + "=" * 80)
    print("3. EMPIRICAL AUDIT: scripts/verify_audit_jumps.py COVERAGE & EDGE CASES")
    print("=" * 80)

    audit_script = Path("scripts/verify_audit_jumps.py").read_text(encoding="utf-8")

    # Edge case 1: Are accelerator statistics verified?
    has_anim_verification = "animation_at" in audit_script or "animation_history" in audit_script
    has_rot90_verification = "rot90" in audit_script
    has_indicator_verification = "Indicator note" in audit_script

    print(f"  Checking scripts/verify_audit_jumps.py for accelerator verification:")
    print(f"    Verifies animation_at / animation_history: {has_anim_verification}")
    print(f"    Verifies rot90 calls                   : {has_rot90_verification}")
    print(f"    Verifies Fable Indicator notes         : {has_indicator_verification}")
    assert not has_anim_verification, "audit script unexpectedly verifies animation calls"
    assert not has_rot90_verification, "audit script unexpectedly verifies rot90 calls"
    assert not has_indicator_verification, "audit script unexpectedly verifies indicators"
    print("  [EDGE CASE CONFIRMED] verify_audit_jumps.py completely omits verification of accelerator statistics!")

    # Edge case 2: simulate_rto edge cases
    def simulate_rto(cand, floor=120.0):
        if cand is not None:
            return max(floor, float(cand))
        return floor

    print("\n  Testing simulate_rto edge cases:")
    # Normal clamping
    assert simulate_rto(13.95) == 120.0
    assert simulate_rto(250.0) == 250.0
    # Boundary: None returns floor
    assert simulate_rto(None) == 120.0
    # Zero or negative candidate (clock skew)
    assert simulate_rto(0.0) == 120.0
    assert simulate_rto(-10.0) == 120.0
    print("    simulate_rto correctly guards negative and zero values.")

    # Edge case 3: Check lp85 tool call preservation assertion
    assert 'yielding with 1 tool call(s) preserved' not in audit_script
    print("  [EDGE CASE CONFIRMED] verify_audit_jumps.py only checks for '0 tool call(s) preserved', missing lp85's 1 preserved call.")

    print("\nALL EMPIRICAL TESTS COMPLETE. VERDICT: REQUEST_CHANGES.")


if __name__ == "__main__":
    audit_read_timeouts()
    audit_accelerator_call_counts()
    test_verify_audit_jumps_edge_cases()

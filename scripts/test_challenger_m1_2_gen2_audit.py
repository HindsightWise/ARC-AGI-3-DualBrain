#!/usr/bin/env python3
"""Adversarial Verification Suite for Milestone 1 Remediation (Gen 2).

Challenger M1_2 Gen 2 independent verification script.
Tests:
1. Raw engine log ReadTimeouts, tool call preservation, and gave_up terminations.
2. Raw transcript accelerator AST and execution parsing (AnimationArchive, rot90, Fable).
3. Documentation text assertions in docs/FORENSIC_JUMPS_V18_V24_TO_V25.md.
4. Test harness assertions in scripts/verify_audit_jumps.py.
5. Clean execution of scripts/verify_audit_jumps.py.
"""

import glob
import os
import re
import subprocess
import sys
from pathlib import Path


def test_raw_engine_log():
    print("--- [TEST 1] Auditing Raw Engine Log for ReadTimeouts & Tool Preservation ---")
    log_path = Path("/tmp/v21_output/arc-prize-2026-v21-apex-engine.log")
    assert log_path.exists(), f"Log file missing: {log_path}"

    timeout_pattern = re.compile(
        r'time":(?P<timestamp>[\d\.]+),.*?'
        r'analyzer request timed out at action (?P<action>\d+);\s*'
        r'yielding with (?P<preserved>\d+) tool call\(s\) preserved:\s*'
        r'.*?\(read timeout=(?P<timeout>[\d\.]+)\)'
    )
    finished_pattern = re.compile(
        r'time":(?P<timestamp>[\d\.]+),.*?'
        r'\[finished\] (?P<game>[^\s]+) state=(?P<state>[^\s]+)'
    )

    timeouts = []
    finished_events = {}

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m_t = timeout_pattern.search(line)
            if m_t:
                gd = m_t.groupdict()
                timeouts.append({
                    "timestamp": float(gd["timestamp"]),
                    "action": int(gd["action"]),
                    "preserved": int(gd["preserved"]),
                    "timeout": float(gd["timeout"]),
                })
            m_f = finished_pattern.search(line)
            if m_f:
                gd_f = m_f.groupdict()
                finished_events[gd_f["game"]] = {
                    "timestamp": float(gd_f["timestamp"]),
                    "state": gd_f["state"],
                }

    # Deduplicate timeouts
    unique_timeouts = []
    seen = set()
    for t in timeouts:
        key = (t["action"], t["timeout"])
        if key not in seen:
            seen.add(key)
            unique_timeouts.append(t)

    assert len(unique_timeouts) == 4, f"Expected 4 unique timeouts, got {len(unique_timeouts)}"
    print(f"  [PASS] Found exactly {len(unique_timeouts)} unique ReadTimeout events.")

    expected = [
        {"action": 42, "timeout": 13.954828416000055, "preserved": 0, "game": "r11l-495a7899"},
        {"action": 153, "timeout": 66.17609974400011, "preserved": 0, "game": "tu93-0768757b"},
        {"action": 148, "timeout": 89.44586153299997, "preserved": 0, "game": "sc25-635fd71a"},
        {"action": 73, "timeout": 35.54712102199983, "preserved": 1, "game": "lp85-305b61c3"},
    ]

    for ut, exp in zip(unique_timeouts, expected):
        assert ut["action"] == exp["action"], f"Action mismatch: {ut['action']} vs {exp['action']}"
        assert abs(ut["timeout"] - exp["timeout"]) < 1e-4, f"Timeout mismatch: {ut['timeout']} vs {exp['timeout']}"
        assert ut["preserved"] == exp["preserved"], f"Preserved tool calls mismatch for {exp['game']}: {ut['preserved']} vs {exp['preserved']}"
        gid = exp["game"]
        assert gid in finished_events, f"Finished event missing for {gid}"
        fin = finished_events[gid]
        assert fin["state"] == "gave_up", f"State mismatch for {gid}: {fin['state']}"
        dt = fin["timestamp"] - ut["timestamp"]
        assert 0.0 <= dt < 1.0, f"Termination delta out of bounds: {dt:.3f}s"
        print(f"  [PASS] {gid}: timeout={ut['timeout']:.2f}s, preserved={ut['preserved']}, finish={fin['state']}, dt={dt:.3f}s")


def test_raw_transcripts_accelerator_census():
    print("--- [TEST 2] Auditing Raw Transcripts for Accelerator Utilization ---")
    transcripts = sorted(glob.glob("/tmp/v21_output/transcripts/*.txt"))
    assert len(transcripts) == 10, f"Expected 10 transcripts, got {len(transcripts)}"

    total_anim_at = 0
    total_anim_hist = 0
    tr87_rot90_defs = 0
    tr87_rot90_calls = 0
    tr87_rot90_mentions = 0
    fable_indicator_notes = 0
    fable_step_budget = 0

    per_game_anim = {}

    for tp in transcripts:
        gid = Path(tp).stem.replace("_p0", "")
        content = Path(tp).read_text(encoding="utf-8", errors="replace")

        fable_indicator_notes += len(re.findall(r"Indicator note:", content))
        fable_step_budget += len(re.findall(r"step-budget", content, re.IGNORECASE))

        code_blocks = re.findall(r"<parameter=code>(.*?)</parameter>", content, re.DOTALL)
        aat = 0
        ahist = 0
        for cb in code_blocks:
            aat += len(re.findall(r"\banimation_at\s*\(", cb))
            ahist += len(re.findall(r"\banimation_history\s*\(", cb))
            if gid == "tr87-cd924810":
                tr87_rot90_defs += len(re.findall(r"def\s+rot90\b", cb))
                tr87_rot90_calls += len(re.findall(r"(?<!def\s)\brot90\s*\(", cb))

        if gid == "tr87-cd924810":
            tr87_rot90_mentions = len(re.findall(r"\brot90\b", content))

        per_game_anim[gid] = (aat, ahist)
        total_anim_at += aat
        total_anim_hist += ahist

    assert per_game_anim["sb26-7fbdac44"] == (3, 1), f"sb26 animation calls mismatch: {per_game_anim['sb26-7fbdac44']}"
    assert per_game_anim["sc25-635fd71a"] == (4, 1), f"sc25 animation calls mismatch: {per_game_anim['sc25-635fd71a']}"
    assert total_anim_at == 7, f"Total animation_at mismatch: {total_anim_at} (expected 7)"
    assert total_anim_hist == 2, f"Total animation_history mismatch: {total_anim_hist} (expected 2)"
    assert total_anim_at + total_anim_hist == 9, f"Combined animation calls mismatch: {total_anim_at + total_anim_hist}"

    assert tr87_rot90_defs == 1 and tr87_rot90_calls == 1, f"tr87 rot90 execution mismatch: defs={tr87_rot90_defs}, calls={tr87_rot90_calls}"
    assert tr87_rot90_mentions == 11, f"tr87 rot90 mentions mismatch: {tr87_rot90_mentions}"
    assert fable_indicator_notes == 15, f"Indicator notes mismatch: {fable_indicator_notes}"
    assert fable_step_budget == 1, f"step-budget mismatch: {fable_step_budget}"

    print(f"  [PASS] Animation REPL calls: {total_anim_at} animation_at + {total_anim_hist} animation_history = 9 calls.")
    print(f"  [PASS] tr87 rot90: {tr87_rot90_defs} def, {tr87_rot90_calls} call, {tr87_rot90_mentions} transcript mentions.")
    print(f"  [PASS] Fable alerts: {fable_indicator_notes} Indicator notes, {fable_step_budget} step-budget.")


def test_remediated_document():
    print("--- [TEST 3] Auditing docs/FORENSIC_JUMPS_V18_V24_TO_V25.md Content ---")
    doc_path = Path("docs/FORENSIC_JUMPS_V18_V24_TO_V25.md")
    assert doc_path.exists(), f"Doc missing: {doc_path}"
    doc_text = doc_path.read_text(encoding="utf-8")

    # Check for absence of fabricated numbers
    assert "15.7s" not in doc_text, "Found fabricated timeout 15.7s in doc"
    assert "25.2s" not in doc_text, "Found fabricated timeout 25.2s in doc"
    assert "5 late-game" not in doc_text, "Found '5 late-game' in doc"

    # Check for presence of accurate 4 ReadTimeouts
    assert "4 unique socket ReadTimeouts" in doc_text or "4 unique ReadTimeouts" in doc_text, "Missing 4 unique ReadTimeouts phrasing"
    assert "13.95s" in doc_text, "Missing 13.95s"
    assert "35.55s" in doc_text, "Missing 35.55s"
    assert "66.18s" in doc_text, "Missing 66.18s"
    assert "89.45s" in doc_text, "Missing 89.45s"

    # Check tool call preservation documentation
    assert "yielding with 0 tool call(s) preserved" in doc_text, "Missing 0 tool call preserved in doc"
    assert "yielding with 1 tool call(s) preserved" in doc_text, "Missing 1 tool call preserved in doc"
    assert "lp85" in doc_text and "preserved 1 tool call" in doc_text, "Missing lp85 1 tool call preservation note"

    # Check accelerator census documentation
    assert "9 calls" in doc_text, "Missing 9 calls in doc"
    assert "7 `animation_at`" in doc_text or "7 animation_at" in doc_text, "Missing 7 animation_at in doc"
    assert "2 `animation_history`" in doc_text or "2 animation_history" in doc_text, "Missing 2 animation_history in doc"
    assert "11 times" in doc_text and "rot90" in doc_text, "Missing rot90 11 times in doc"

    # Check net level delta
    assert "+13.0" in doc_text or "+13.0 lvls net" in doc_text, "Missing +13.0 level delta in doc"
    assert "+14 lvls net" not in doc_text, "Found old +14 lvls net typo in doc"

    print("  [PASS] Remediated document accurately reflects ground-truth telemetry.")


def test_verify_script_and_execution():
    print("--- [TEST 4] Testing scripts/verify_audit_jumps.py Execution & Assertions ---")
    script_path = Path("scripts/verify_audit_jumps.py")
    assert script_path.exists(), f"Script missing: {script_path}"
    script_text = script_path.read_text(encoding="utf-8")

    # Assert that the script contains the required assertions
    assert "len(unique_timeouts) == 4" in script_text, "verify script missing 4 unique timeouts assertion"
    assert "yielding with 1 tool call(s) preserved" in script_text, "verify script missing 1 tool call preserved assertion"
    assert "yielding with 0 tool call(s) preserved" in script_text, "verify script missing 0 tool call preserved assertion"
    assert "total_anim_at == 7" in script_text, "verify script missing total_anim_at == 7 assertion"
    assert "total_anim_hist == 2" in script_text, "verify script missing total_anim_hist == 2 assertion"
    assert "tr87_rot90_mentions == 11" in script_text, "verify script missing tr87_rot90_mentions == 11 assertion"
    assert "net_level_delta == 13.0" in script_text, "verify script missing net_level_delta == 13.0 assertion"

    # Run the script
    res = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True)
    assert res.returncode == 0, f"verify_audit_jumps.py failed with returncode {res.returncode}:\n{res.stderr}"
    assert "ALL VERIFICATIONS PASSED" in res.stdout, "Missing success string in output"
    print("  [PASS] verify_audit_jumps.py executed with exit code 0 and verified all assertions.")


if __name__ == "__main__":
    test_raw_engine_log()
    test_raw_transcripts_accelerator_census()
    test_remediated_document()
    test_verify_script_and_execution()
    print("\n========================================================")
    print("ALL EMPIRICAL ADVERSARIAL CHALLENGER TESTS PASSED (100%)")
    print("========================================================")

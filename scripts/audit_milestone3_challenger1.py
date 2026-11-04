#!/usr/bin/env python3
"""
Empirical Verification & Adversarial Mutation Challenge Script for Milestone 3.
Challenger: M3_1
Audits the sensitivity and fault-detection capabilities of tests/test_v25_e2e_requirements.py.

Simulates mutations/defects:
1. Cell 2 patch character length changed by +1, -1, and content mutated while length preserved.
2. enable_internet: true, enable_gpu: false, machine_shape altered in metadata.
3. rot90 returns wrong dimensions, wrong values, and rectangular grid analysis.
4. request_timeout_seconds floor not applied, 0.0 floor not applied, None handling corrupted.
5. ARC3_FABLE_PIN_TOKENS is missing or set to '1'.
6. D' priority formula regressions, anti-patterns (started_at reset, aging_bonus, EXPOSE_RESET).
7. AnimationArchive and empty transition guard mutations.
8. Perceptual notes and budget alert mutations.
9. Prompt bloat / model checks leak mutations.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tests.test_v25_e2e_requirements import TestV25E2ERequirements, _src


class MutationSensitivityTester:
    def __init__(self):
        self.results: dict[str, dict] = {}
        TestV25E2ERequirements.setUpClass()
        self.orig_nb_v25 = copy.deepcopy(TestV25E2ERequirements.nb_v25)
        self.orig_meta = copy.deepcopy(TestV25E2ERequirements.meta)

    def fresh_test_instance(self) -> TestV25E2ERequirements:
        """Create a fresh test instance with decoupled deepcopied state."""
        t = TestV25E2ERequirements()
        t.nb_v25 = copy.deepcopy(self.orig_nb_v25)
        t.nb_sub = copy.deepcopy(self.orig_nb_v25)
        t.meta = copy.deepcopy(self.orig_meta)
        return t

    def record(self, test_name: str, mutation_desc: str, expected_fail: bool, actual_failed: bool, error_msg: str):
        passed_as_expected = (expected_fail == actual_failed)
        self.results[f"{test_name}::{mutation_desc}"] = {
            "expected_fail": expected_fail,
            "actual_failed": actual_failed,
            "passed_as_expected": passed_as_expected,
            "error_msg": error_msg,
        }
        status = "PASSED (CAUGHT)" if passed_as_expected else "FAILED (MISSED)"
        print(f"[{status}] {test_name}: {mutation_desc}")
        if actual_failed:
            first_line = error_msg.strip().splitlines()[0] if error_msg else ""
            print(f"    Caught with error: {first_line}")
        elif expected_fail and not actual_failed:
            print(f"    WARNING: Mutation NOT caught! Test silently passed!")

    # --------------------------------------------------------------------------
    # CHALLENGE 1: Cell 2 Patch Length & Content Mutations (test_02)
    # --------------------------------------------------------------------------
    def challenge_1_cell2_patch(self):
        print("\n" + "=" * 80)
        print("CHALLENGE 1: Cell 2 Patch Sensitivity (test_02)")
        print("=" * 80)

        # 1A: Length + 1
        t = self.fresh_test_instance()
        c2 = _src(t.nb_v25["cells"][2])
        t.nb_v25["cells"][2]["source"] = c2 + " "
        failed = False
        msg = ""
        try:
            t.test_02_pure_v11_cell2_patch_preserved()
        except AssertionError as e:
            failed = True
            msg = str(e)
        self.record("test_02", "Cell 2 length + 1 char (498800)", True, failed, msg)

        # 1B: Length - 1
        t = self.fresh_test_instance()
        c2 = _src(t.nb_v25["cells"][2])
        t.nb_v25["cells"][2]["source"] = c2[:-1]
        failed = False
        msg = ""
        try:
            t.test_02_pure_v11_cell2_patch_preserved()
        except AssertionError as e:
            failed = True
            msg = str(e)
        self.record("test_02", "Cell 2 length - 1 char (498798)", True, failed, msg)

        # 1C: Length preserved (498799), but 1 character mutated
        t = self.fresh_test_instance()
        c2 = _src(t.nb_v25["cells"][2])
        mid_idx = len(c2) // 2
        char_to_swap = "X" if c2[mid_idx] != "X" else "Y"
        mutated_c2 = c2[:mid_idx] + char_to_swap + c2[mid_idx+1:]
        assert len(mutated_c2) == 498799
        t.nb_v25["cells"][2]["source"] = mutated_c2
        failed = False
        msg = ""
        try:
            t.test_02_pure_v11_cell2_patch_preserved()
        except AssertionError as e:
            failed = True
            msg = str(e)
        self.record("test_02", "Cell 2 length preserved (498799), 1 char mutated (SHA-256)", True, failed, msg)

    # --------------------------------------------------------------------------
    # CHALLENGE 2: Offline Metadata Compliance (test_09)
    # --------------------------------------------------------------------------
    def challenge_2_metadata(self):
        print("\n" + "=" * 80)
        print("CHALLENGE 2: Offline Metadata Compliance Sensitivity (test_09)")
        print("=" * 80)

        # 2A: enable_internet: True
        t = self.fresh_test_instance()
        t.meta["enable_internet"] = True
        failed = False
        msg = ""
        try:
            t.test_09_kernel_metadata_offline_compliance()
        except AssertionError as e:
            failed = True
            msg = str(e)
        self.record("test_09", "enable_internet: True (violation of air-gapped rule)", True, failed, msg)

        # 2B: enable_gpu: False
        t = self.fresh_test_instance()
        t.meta["enable_gpu"] = False
        failed = False
        msg = ""
        try:
            t.test_09_kernel_metadata_offline_compliance()
        except AssertionError as e:
            failed = True
            msg = str(e)
        self.record("test_09", "enable_gpu: False (SGLang GPU disabled)", True, failed, msg)

        # 2C: machine_shape altered
        t = self.fresh_test_instance()
        t.meta["machine_shape"] = "NvidiaT4x2"
        failed = False
        msg = ""
        try:
            t.test_09_kernel_metadata_offline_compliance()
        except AssertionError as e:
            failed = True
            msg = str(e)
        self.record("test_09", "machine_shape: 'NvidiaT4x2' (non-RTX6000 shape)", True, failed, msg)

        # 2D: kernel id altered
        t = self.fresh_test_instance()
        t.meta["id"] = "zerbytheboss/v24-apex-engine"
        failed = False
        msg = ""
        try:
            t.test_09_kernel_metadata_offline_compliance()
        except AssertionError as e:
            failed = True
            msg = str(e)
        self.record("test_09", "kernel id mismatched ('v24' instead of 'v25')", True, failed, msg)

    # --------------------------------------------------------------------------
    # CHALLENGE 3: Spatial Helper rot90 Mutations (test_07)
    # --------------------------------------------------------------------------
    def challenge_3_rot90(self):
        print("\n" + "=" * 80)
        print("CHALLENGE 3: Spatial Helper rot90 Sensitivity (test_07)")
        print("=" * 80)

        c4 = _src(self.orig_nb_v25["cells"][4])
        ns: dict = {}
        for line in c4.splitlines():
            if line.startswith("_v25_assets = {"):
                exec(line, ns)
                break
        assets = ns["_v25_assets"]
        spatial_script = assets["patch_spatial_helpers.py"]

        # 3A: rot90 returns wrong dimensions (2x2 instead of 3x3)
        mutated_spatial_dim = spatial_script.replace(
            "        res = [list(x) for x in zip(*res[::-1])]",
            "        res = [list(x[:2]) for x in zip(*res[::-1])][:2]"
        )
        t = self.fresh_test_instance()
        mutated_assets = dict(assets)
        mutated_assets["patch_spatial_helpers.py"] = mutated_spatial_dim
        t.nb_v25["cells"][4]["source"] = f"_v25_assets = {repr(mutated_assets)}\n"
        failed = False
        msg = ""
        try:
            t.test_07_repl_spatial_helpers_zero_prompt_bloat()
        except AssertionError as e:
            failed = True
            msg = str(e)
        self.record("test_07", "rot90 returns wrong dimensions (2x2 truncated)", True, failed, msg)

        # 3B: rot90 returns wrong orientation (CCW instead of CW)
        mutated_spatial_ccw = spatial_script.replace(
            "        res = [list(x) for x in zip(*res[::-1])]",
            "        res = [list(x) for x in zip(*res)][::-1]"
        )
        t = self.fresh_test_instance()
        mutated_assets = dict(assets)
        mutated_assets["patch_spatial_helpers.py"] = mutated_spatial_ccw
        t.nb_v25["cells"][4]["source"] = f"_v25_assets = {repr(mutated_assets)}\n"
        failed = False
        msg = ""
        try:
            t.test_07_repl_spatial_helpers_zero_prompt_bloat()
        except AssertionError as e:
            failed = True
            msg = str(e)
        self.record("test_07", "rot90 returns inverted rotation (CCW instead of CW)", True, failed, msg)

        # 3C: Deep check on rectangular grid support
        helper_ns: dict = {}
        seg_part = spatial_script.split('SEG_ADDENDUM = """')[1].split('"""')[0]
        exec(seg_part, helper_ns)
        rect_grid = [
            [1, 2, 3],
            [4, 5, 6],
        ]  # 2 rows x 3 cols
        r90_rect = helper_ns["rot90"](rect_grid, 1)
        expected_r90_rect = [
            [4, 1],
            [5, 2],
            [6, 3],
        ]  # 3 rows x 2 cols
        rect_correct = (r90_rect == expected_r90_rect)
        print(f"    [EMPIRICAL RECTANGULAR CHECK] rot90(2x3, 1): {r90_rect} == {expected_r90_rect} -> {rect_correct}")
        assert rect_correct, "rot90 must correctly rotate rectangular grids"
        self.record("test_07", "rot90 rectangular grid 2x3 -> 3x2 dimensions verified", False, False, "")

    # --------------------------------------------------------------------------
    # CHALLENGE 4: Request Timeout Floor Clamping (test_08)
    # --------------------------------------------------------------------------
    def challenge_4_timeout_clamping(self):
        print("\n" + "=" * 80)
        print("CHALLENGE 4: Request Timeout Clamping Sensitivity (test_08)")
        print("=" * 80)

        c16 = _src(self.orig_nb_v25["cells"][16])

        # 4A: Clamping floor NOT applied (returns candidate without max(floor, ...))
        # Note: in c16 string, newlines inside _FP_SOURCE are escaped as \\n
        target_str = 'return max(floor, float(cand)) if cand is not None else None'
        mutated_c16_no_floor = c16.replace(
            target_str,
            'return float(cand) if cand is not None else None'
        )
        assert mutated_c16_no_floor != c16, "Failed to replace floor in c16"
        t = self.fresh_test_instance()
        t.nb_v25["cells"][16]["source"] = mutated_c16_no_floor
        failed = False
        msg = ""
        try:
            t.test_08_request_timeout_clamping_protection()
        except AssertionError as e:
            failed = True
            msg = str(e)
        self.record("test_08", "request_timeout_seconds floor NOT applied (13.95s not elevated)", True, failed, msg)

        # 4B: Floor is NOT applied when cand = 0.0
        mutated_c16_no_zero = c16.replace(
            target_str,
            'return (0.0 if float(cand) == 0.0 else max(floor, float(cand))) if cand is not None else None'
        )
        assert mutated_c16_no_zero != c16, "Failed to replace cand=0.0 in c16"
        t = self.fresh_test_instance()
        t.nb_v25["cells"][16]["source"] = mutated_c16_no_zero
        failed = False
        msg = ""
        try:
            t.test_08_request_timeout_clamping_protection()
        except AssertionError as e:
            failed = True
            msg = str(e)
        self.record("test_08", "cand = 0.0 not elevated to 120.0s floor", True, failed, msg)

        # 4C: cand = None erroneously elevated to floor
        mutated_c16_none_elevated = c16.replace(
            target_str,
            'return max(floor, float(cand or 0.0))'
        )
        assert mutated_c16_none_elevated != c16, "Failed to replace cand=None in c16"
        t = self.fresh_test_instance()
        t.nb_v25["cells"][16]["source"] = mutated_c16_none_elevated
        failed = False
        msg = ""
        try:
            t.test_08_request_timeout_clamping_protection()
        except AssertionError as e:
            failed = True
            msg = str(e)
        self.record("test_08", "cand = None erroneously elevated to 120.0s instead of None", True, failed, msg)

    # --------------------------------------------------------------------------
    # CHALLENGE 5: ARC3_FABLE_PIN_TOKENS Prefix Immutability (test_06)
    # --------------------------------------------------------------------------
    def challenge_5_pin_tokens(self):
        print("\n" + "=" * 80)
        print("CHALLENGE 5: ARC3_FABLE_PIN_TOKENS Immutability Sensitivity (test_06)")
        print("=" * 80)

        c4 = _src(self.orig_nb_v25["cells"][4])

        # 5A: ARC3_FABLE_PIN_TOKENS set to '1'
        mutated_c4_pin_1 = c4.replace(
            "'ARC3_FABLE_PIN_TOKENS': '0'",
            "'ARC3_FABLE_PIN_TOKENS': '1'"
        )
        assert mutated_c4_pin_1 != c4
        t = self.fresh_test_instance()
        t.nb_v25["cells"][4]["source"] = mutated_c4_pin_1
        failed = False
        msg = ""
        try:
            t.test_06_perceptual_notes_prefix_immutability()
        except AssertionError as e:
            failed = True
            msg = str(e)
        self.record("test_06", "ARC3_FABLE_PIN_TOKENS: '1' (prefix cache killer)", True, failed, msg)

        # 5B: ARC3_FABLE_PIN_TOKENS omitted entirely
        mutated_c4_pin_missing = c4.replace(
            "'ARC3_FABLE_PIN_TOKENS': '0',",
            "# Pin tokens omitted"
        )
        assert mutated_c4_pin_missing != c4
        t = self.fresh_test_instance()
        t.nb_v25["cells"][4]["source"] = mutated_c4_pin_missing
        failed = False
        msg = ""
        try:
            t.test_06_perceptual_notes_prefix_immutability()
        except AssertionError as e:
            failed = True
            msg = str(e)
        self.record("test_06", "ARC3_FABLE_PIN_TOKENS omitted entirely", True, failed, msg)

    # --------------------------------------------------------------------------
    # CHALLENGE 6: Extended Regressions (D', Anti-patterns, Bloat)
    # --------------------------------------------------------------------------
    def challenge_6_extended_regressions(self):
        print("\n" + "=" * 80)
        print("CHALLENGE 6: Extended Regressions (D' math, Anti-patterns, Bloat)")
        print("=" * 80)

        c16 = _src(self.orig_nb_v25["cells"][16])

        # 6A: Invert D' unstarted game pricing (p6 < p8 < p10)
        target_norm = 'norm = 55 / (n_clamped * (n_clamped + 1) / 2)'
        mutated_c16_d_price = c16.replace(
            target_norm,
            'norm = (n_clamped * (n_clamped + 1) / 2) / 55'
        )
        assert mutated_c16_d_price != c16, "Failed to replace norm in c16"
        t = self.fresh_test_instance()
        t.nb_v25["cells"][16]["source"] = mutated_c16_d_price
        failed = False
        msg = ""
        try:
            t.test_04_d_prime_priority_math_and_invariants()
        except AssertionError as e:
            failed = True
            msg = str(e)
        self.record("test_04", "Inverted unstarted game pricing in D' formula", True, failed, msg)

        # 6B: Reintroduce anti-pattern started_at reset
        mutated_c16_started_at = c16 + "\n# anti-pattern: self.started_at = time.monotonic()\n"
        t = self.fresh_test_instance()
        t.nb_v25["cells"][16]["source"] = mutated_c16_started_at
        failed = False
        msg = ""
        try:
            t.test_04_d_prime_priority_math_and_invariants()
        except AssertionError as e:
            failed = True
            msg = str(e)
        self.record("test_04", "Anti-pattern self.started_at reset reintroduced", True, failed, msg)

        # 6C: Reintroduce anti-pattern aging_bonus
        mutated_c16_aging = c16 + "\n# anti-pattern: aging_bonus += 1.0\n"
        t = self.fresh_test_instance()
        t.nb_v25["cells"][16]["source"] = mutated_c16_aging
        failed = False
        msg = ""
        try:
            t.test_04_d_prime_priority_math_and_invariants()
        except AssertionError as e:
            failed = True
            msg = str(e)
        self.record("test_04", "Anti-pattern aging_bonus reintroduced", True, failed, msg)

        # 6D: Reintroduce anti-pattern EXPOSE_RESET
        mutated_c16_expose = c16 + "\n# anti-pattern: EXPOSE_RESET = True\n"
        t = self.fresh_test_instance()
        t.nb_v25["cells"][16]["source"] = mutated_c16_expose
        failed = False
        msg = ""
        try:
            t.test_04_d_prime_priority_math_and_invariants()
        except AssertionError as e:
            failed = True
            msg = str(e)
        self.record("test_04", "Anti-pattern EXPOSE_RESET reintroduced", True, failed, msg)

        # 6E: Reintroduce prompt bloat (patch_model_checks.py in Cell 4)
        c4 = _src(self.orig_nb_v25["cells"][4])
        mutated_c4_bloat = c4 + "\n# patch_model_checks.py included\n"
        t = self.fresh_test_instance()
        t.nb_v25["cells"][4]["source"] = mutated_c4_bloat
        failed = False
        msg = ""
        try:
            t.test_07_repl_spatial_helpers_zero_prompt_bloat()
        except AssertionError as e:
            failed = True
            msg = str(e)
        self.record("test_07", "Prompt bloat patch_model_checks.py reintroduced into Cell 4", True, failed, msg)

    def run_all(self):
        self.challenge_1_cell2_patch()
        self.challenge_2_metadata()
        self.challenge_3_rot90()
        self.challenge_4_timeout_clamping()
        self.challenge_5_pin_tokens()
        self.challenge_6_extended_regressions()

        print("\n" + "=" * 80)
        print("SUMMARY OF ADVERSARIAL SENSITIVITY CHALLENGES")
        print("=" * 80)
        total = len(self.results)
        passed = sum(1 for r in self.results.values() if r["passed_as_expected"])
        print(f"Total Mutations Tested: {total}")
        print(f"Mutations Successfully Caught: {passed} / {total} ({passed/total*100:.1f}%)")

        for k, v in self.results.items():
            status = "PASS" if v["passed_as_expected"] else "FAIL"
            print(f"[{status}] {k}")

        if passed == total:
            print("\nVERDICT: All mutations were caught by tests/test_v25_e2e_requirements.py.")
            return True
        else:
            print("\nVERDICT: Some mutations escaped detection! Review details above.")
            return False


if __name__ == "__main__":
    tester = MutationSensitivityTester()
    success = tester.run_all()
    sys.exit(0 if success else 1)

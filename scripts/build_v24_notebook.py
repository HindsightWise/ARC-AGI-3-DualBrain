#!/usr/bin/env python3
"""
scripts/build_v24_notebook.py
=============================
Deterministic Builder for ARC Prize 2026 — Version 24 Sovereign Apex Engine.

Architectural Mandate (Strict Single-Variable MDL Upgrade over Version 11):
1. Anchors 100% bit-for-bit to our verified Version 11 baseline (`26.70` LB,
   `pulled_kernels/v11/arc-prize-2026-v11-apex-engine.ipynb`) across:
   - Cell 2: Pristine 498,799-byte TAAF harness patch (ZERO `self.started_at` clock tampering).
   - Cell 4: Pristine Version 11 `setup_env` (`LOCAL_ANALYZER_YIELD_TOKENS=2048`,
     `LOCAL_ANALYZER_MAX_OUTPUT=12*1024`, `ARC3_CONTEXT_DRAIN_TOKENS=58*1024`,
     `ARC3_GUARDS_FROM_LEVEL=2`, `ARC3_MEMORY_SECTIONS=off`, `EXPOSE_UNDO=on`,
     no `EXPOSE_RESET`).
   - Cell 12: Pristine 10-stream Pennyroyal SGLang v2.5.3 + Albucino MTP-3 server
     (`MAXREQ=10`, `CUDAGRAPH_MAXBS=10`, `MAMBA_CACHE=60`, `MEMFRAC=0.96`).
2. Applies ONLY the verified `D'` continuous slot-priority scheduler (`ours_form_priority.install_d`)
   from `shiiin9` (`31.54` LB single-sub):
   - Replaces `tool_agent.priority_value` with `d_priority` (`A * M * C + B * phi`).
   - Replaces `tool_agent.ProgressPace` with `FormPace` (mean tokens per cleared level).
   - Prices unstarted games in `_PriorityGate.acquire` via `PrioritySnapshot(1, 0, 0.0, 0.0, N)`
     instead of the `2,000,000` untrimmed FIFO band, WITHOUT touching `self.started_at`
     and WITHOUT queue-aging thrash.
3. Uses the 5-minute single-game (`ft09`) rehearsal during non-submission commit runs
   (`not TRUE_SUBMISSION`) so the Kaggle commit verifies full end-to-end SGLang + patch +
   `install_d` execution in ~15 minutes, while `TRUE_SUBMISSION` runs all 110 competition
   games with `concurrency = 120` and `max_runtime_s_per_game = 532 * 60`.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
V11_NB_PATH = REPO_ROOT / "pulled_kernels" / "v11" / "arc-prize-2026-v11-apex-engine.ipynb"
SHIIIN_NB_PATH = (
    REPO_ROOT
    / "pulled_kernels"
    / "oct13_top"
    / "shiiin9_affectify-arc-31-54-in-a-single-sub"
    / "affectify-arc-31-54-in-a-single-sub.ipynb"
)
OUTPUT_V24_PATH = REPO_ROOT / "notebooks" / "submission_v24.ipynb"
OUTPUT_SUB_PATH = REPO_ROOT / "notebooks" / "submission.ipynb"
METADATA_PATH = REPO_ROOT / "notebooks" / "kernel-metadata.json"


CELL0_MARKDOWN = """# ARC Prize 2026 — Version 24 Sovereign Apex Engine

## Pure Version 11 Foundation (`26.70` LB) + Verified `D'` Slot Priority (`31.54` LB)
Version 24 enforces strict single-variable Minimum Description Length (MDL) engineering over our highest-scoring Version 11 baseline (`26.70` LB):
- **Pristine Version 11 Harness & Environment**: 100% unmodified 498,799-byte TAAF patch (`Cell 2`), `LOCAL_ANALYZER_YIELD_TOKENS=2048`, `LOCAL_ANALYZER_MAX_OUTPUT=12288`, `ARC3_CONTEXT_DRAIN_TOKENS=58*1024`, `ARC3_GUARDS_FROM_LEVEL=2`, `ARC3_MEMORY_SECTIONS=off`, `EXPOSE_UNDO=on` (no `EXPOSE_RESET`), and 10-stream SGLang KV lock (`MAXREQ=10`, `MEMFRAC=0.96`).
- **Single Isolated Upgrade (`D'` Priority Scheduler)**: Replaces `_PriorityGate` slot ordering with the continuous expected-value formula `priority = A * M * C + B * phi` (`install_d`) and prices unstarted games by `(level=1, actions=0, tokens=0, total_levels=N)` instead of the `2,000,000` FIFO band.
- **Zero Clock or Queue Thrashing**: Preserves the global `532 * 60s` (`8h 52m`) competition stop clock (`self.started_at`) and eliminates artificial queue aging so all 110 competition games terminate and reconcile cleanly before the 9-hour container ceiling.
"""


def _src(cell: dict) -> str:
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else str(s)


def build_v24_notebook() -> dict:
    if not V11_NB_PATH.exists():
        raise FileNotFoundError(f"Missing v11 reference notebook at {V11_NB_PATH}")
    if not SHIIIN_NB_PATH.exists():
        raise FileNotFoundError(f"Missing shiiin9 reference notebook at {SHIIIN_NB_PATH}")

    nb_v11 = json.loads(V11_NB_PATH.read_text(encoding="utf-8"))
    nb_shiiin = json.loads(SHIIIN_NB_PATH.read_text(encoding="utf-8"))

    # Verify that v11 cells 1..15 and 17..22 match shiiin9 cells 2..16, 18..20, 22, 24, 25 byte-for-byte
    v11_to_shiiin = {
        1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9,
        9: 10, 10: 11, 11: 12, 12: 13, 13: 14, 14: 15, 15: 16,
        17: 18, 18: 19, 19: 20, 20: 22, 21: 24, 22: 25,
    }
    for v11_idx, sh_idx in v11_to_shiiin.items():
        if _src(nb_v11["cells"][v11_idx]) != _src(nb_shiiin["cells"][sh_idx]):
            raise AssertionError(f"Cell mismatch between v11[{v11_idx}] and shiiin9[{sh_idx}]")

    # Build 23-cell notebook anchored on v11
    nb_v24 = copy.deepcopy(nb_v11)
    assert len(nb_v24["cells"]) == 23

    # Cell 0: Markdown overview
    nb_v24["cells"][0]["source"] = CELL0_MARKDOWN

    # Cell 16: Combine shiiin9's Cell 17 (bm.solver config + 5-min commit rehearsal)
    # and shiiin9's Cell 21 (ours_form_priority.install_d)
    solver_cfg_src = _src(nb_shiiin["cells"][17]).rstrip()
    d_prime_src = _src(nb_shiiin["cells"][21]).rstrip()
    nb_v24["cells"][16]["source"] = solver_cfg_src + "\n\n" + d_prime_src + "\n"

    # Cell 20: Append the #OURS_FRESH verification print right after bm.run()
    c20_src = _src(nb_v11["cells"][20]).rstrip()
    fresh_print_src = _src(nb_shiiin["cells"][23]).rstrip()
    nb_v24["cells"][20]["source"] = c20_src + "\n" + fresh_print_src + "\n"

    # Clear any stale outputs or execution_counts across all cells
    for cell in nb_v24["cells"]:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    return nb_v24


def update_kernel_metadata() -> None:
    meta = {
        "id": "zerbytheboss/v24-sovereign-apex-engine",
        "title": "v24 - Sovereign Apex Engine",
        "code_file": "submission.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "dataset_sources": [
            "dfranzen/pennyroyal-v253",
            "dfranzen/taaf-kaggle-source-bundle-copy",
        ],
        "kernel_sources": [],
        "competition_sources": ["arc-prize-2026-arc-agi-3"],
        "model_sources": [
            "dfranzen/albucino-qwen3-8-flash-next-drafter/Transformers/default/1",
            "dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/Transformers/default/1",
        ],
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c",
        "machine_shape": "NvidiaRtxPro6000",
    }
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    print(f"Updated {METADATA_PATH} (id={meta['id']}, machine_shape={meta['machine_shape']})")


def main() -> None:
    nb_v24 = build_v24_notebook()
    OUTPUT_V24_PATH.parent.mkdir(parents=True, exist_ok=True)

    for path in (OUTPUT_V24_PATH, OUTPUT_SUB_PATH):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb_v24, f, indent=1, ensure_ascii=False)
            f.write("\n")

    update_kernel_metadata()

    sha_v24 = hashlib.sha256(OUTPUT_V24_PATH.read_bytes()).hexdigest()
    sha_sub = hashlib.sha256(OUTPUT_SUB_PATH.read_bytes()).hexdigest()
    assert sha_v24 == sha_sub, "submission_v24.ipynb and submission.ipynb must be byte-identical"
    print(
        f"Built {OUTPUT_V24_PATH.name} and {OUTPUT_SUB_PATH.name} "
        f"({OUTPUT_V24_PATH.stat().st_size} bytes, sha256={sha_v24[:12]})"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
scripts/build_v16_notebook.py
Deterministic builder for ARC Prize 2026 Version 16 Sovereign Apex Engine Notebook.

Architectural Objectives:
1. Re-anchors directly to the verified v11 foundation (which achieved 26.70 peak score).
2. Purges the toxic 30-action hard reset guard, overfitted 25-game trajectories, and context bloat.
3. Implements F1 context hygiene (ARC3_HISTORY_IMAGE_KEEP=0, ARC3_RESUME_PROMPT_IMAGES=0)
   saving 25%+ of token budget so Qwen3.8-Flash-Next retains deep reasoning context.
4. Enables voluntary EXPOSE_RESET=on and EXPOSE_UNDO=on control affordances.
5. Preserves SGLang Pennyroyal v2.5.3 + Albucino MTP-3 speculative decoding at 558-946 tok/s.
6. Preserves 120-stream dynamic Priority Scheduler.
"""

import json
import os
from pathlib import Path

ROOT = Path("/Users/zerbytheboss/.gemini/antigravity/scratch/Kaggle/Arc-Prize-2026")
V11_NB_PATH = ROOT / "v11_apex" / "arc-prize-2026-v11-apex-engine.ipynb"
V16_DIR = ROOT / "v16_apex"
STARTER_DIR = Path("/Users/zerbytheboss/.gemini/antigravity/scratch/ARC-AGI-3-Kaggle-Starter")

def build():
    assert V11_NB_PATH.exists(), f"Source v11 notebook not found at {V11_NB_PATH}"
    with open(V11_NB_PATH) as f:
        nb = json.load(f)

    print(f"Loaded v11 source notebook ({len(nb['cells'])} cells)")
    assert len(nb["cells"]) == 23, f"Expected 23 cells in clean v11, found {len(nb['cells'])}"

    # 1. Update Cell 0 Markdown Header
    cell_0_md = """# ARC Prize 2026 — Version 16 Sovereign Apex Engine

## Sovereign Recovery & Context Optimization
This notebook restores the verified high-performance foundation of Version 11 (26.70 peak score on the official leaderboard):
- **Serving Engine**: Pennyroyal SGLang v2.5.3 on Nvidia RTX 6000 Ada (96GB VRAM) with FlashInfer, FP8 KV cache, and 139,264 context tokens.
- **Model Stack**: Intel AutoRound W4A16 Qwen3.8-Flash-Next paired with Albucino MTP-3 Speculative Draft Model generating 558–946 tokens/sec.
- **Context Hygiene**: `ARC3_HISTORY_IMAGE_KEEP=0` (only current board frame receives visual tokens; past turns retain text actions and observations, saving 25%+ of the context budget).
- **Control Affordances**: `EXPOSE_UNDO=on` (engine ACTION7 undo) and `EXPOSE_RESET=on` (voluntary level reset when stuck).
- **Priority Scheduling**: 120-stream dynamic priority continuation scheduling allocating GPU compute to high-potential game paths.
- **Purity Invariant**: Exactly zero artificial 30-action hard resets, zero overfitted public trajectories, zero context pollution."""
    nb["cells"][0]["source"] = [line + "\n" for line in cell_0_md.split("\n")]

    # 2. Update Cell 4: setup_env settings for context hygiene & voluntary controls
    cell_4_src = "".join(nb["cells"][4]["source"])
    
    # Ensure ARC3_HISTORY_IMAGE_KEEP, ARC3_RESUME_PROMPT_IMAGES, EXPOSE_RESET are configured
    # We will adjust the setup_env dictionary in Cell 4
    replacements = [
        ("'EXPOSE_UNDO': 'on',", 
         "'EXPOSE_UNDO': 'on',\n    'EXPOSE_RESET': 'on',\n    'ARC3_HISTORY_IMAGE_KEEP': '0',\n    'ARC3_RESUME_PROMPT_IMAGES': '0',"),
    ]
    
    for old, new in replacements:
        assert old in cell_4_src, f"Anchor string {old!r} not found in Cell 4!"
        cell_4_src = cell_4_src.replace(old, new)
        
    nb["cells"][4]["source"] = [line + "\n" for line in cell_4_src.splitlines()]

    # Verify no toxic monkeypatches exist in the notebook
    nb_str = json.dumps(nb)
    assert 'action(["RESET"])' not in nb_str, "Toxic 30-action reset guard detected in notebook!"
    assert '_VERIFIED_TRAJECTORIES' not in nb_str, "Overfitted trajectories detected in notebook!"
    assert '=== SOLVED LEVELS ===' not in nb_str, "Context-polluting memory blocks detected in notebook!"
    assert len(nb["cells"]) == 23, f"Cell count must remain 23, got {len(nb['cells'])}"

    # 3. Write target notebook to v16_apex
    V16_DIR.mkdir(parents=True, exist_ok=True)
    v16_nb_path = V16_DIR / "arc-prize-2026-v16-apex-engine.ipynb"
    with open(v16_nb_path, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Generated {v16_nb_path} ({v16_nb_path.stat().st_size} bytes)")

    # 4. Write v16 kernel-metadata.json
    metadata = {
        "id": "zerbytheboss/arc-prize-2026-v16-apex-engine",
        "title": "ARC Prize 2026 - v16 Apex Engine",
        "code_file": "arc-prize-2026-v16-apex-engine.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "dataset_sources": [
            "dfranzen/pennyroyal-v253",
            "dfranzen/taaf-kaggle-source-bundle-copy"
        ],
        "kernel_sources": [],
        "competition_sources": [
            "arc-prize-2026-arc-agi-3"
        ],
        "model_sources": [
            "dfranzen/albucino-qwen3-8-flash-next-drafter/Transformers/default/1",
            "dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/Transformers/default/1"
        ],
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c",
        "machine_shape": "NvidiaRtxPro6000"
    }
    meta_path = V16_DIR / "kernel-metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Generated {meta_path}")

    # 5. Mirror to ARC-AGI-3-Kaggle-Starter for full workspace synchronization
    starter_meta = dict(metadata)
    starter_meta["code_file"] = "submission.ipynb"
    with open(STARTER_DIR / "notebooks" / "submission.ipynb", "w") as f:
        json.dump(nb, f, indent=1)
    with open(STARTER_DIR / "notebooks" / "kernel-metadata.json", "w") as f:
        json.dump(starter_meta, f, indent=2)
    print("Synchronized submission.ipynb and kernel-metadata.json to ARC-AGI-3-Kaggle-Starter/notebooks/")

    print("Version 16 build completed successfully!")

if __name__ == "__main__":
    build()

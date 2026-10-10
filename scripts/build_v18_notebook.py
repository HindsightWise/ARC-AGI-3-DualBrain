#!/usr/bin/env python3
"""
scripts/build_v18_notebook.py
Deterministic builder for ARC Prize 2026 Version 18 Sovereign Apex Engine.

Synthesis Architecture:
1. Re-anchors to v11 baseline (26.70 peak score foundation).
2. Purges EXPOSE_RESET (protects RHAE score denominator from voluntary suicide resets).
3. Restores Vision History Eviction (ARC3_HISTORY_IMAGE_KEEP='0'):
   - Strips past turn images from prompt history to prevent the 10-stream KV cache saturation
     and 94.1 GB / 95.6 GB VRAM exhaustion that crashed Version 17 at T+29.1m.
4. Preserves Resumption Vision (ARC3_RESUME_PROMPT_IMAGES defaults to True):
   - Guarantees tool resumption messages always retain the active board frame.
5. Calibrates SGLang MEMFRAC from 0.96 to 0.94:
   - Expands dynamic VRAM headroom from 1.5 GB to ~3.8 GB for PyTorch activation buffers.
6. Preserves pure v11 sampling parameters:
   - Temperature = 0.70, Top-P = 0.95, Yield Tokens = 2048, EXPOSE_UNDO = 'on'.
"""

import json
from pathlib import Path

KAGGLE_ROOT = Path("/Users/zerbytheboss/.gemini/antigravity/scratch/Kaggle/Arc-Prize-2026")
STARTER_ROOT = Path("/Users/zerbytheboss/.gemini/antigravity/scratch/ARC-AGI-3-Kaggle-Starter")
V11_NB_PATH = KAGGLE_ROOT / "v11_apex" / "arc-prize-2026-v11-apex-engine.ipynb"
V18_DIR = KAGGLE_ROOT / "v18_apex"

def build():
    assert V11_NB_PATH.exists(), f"Source v11 notebook missing: {V11_NB_PATH}"
    with open(V11_NB_PATH) as f:
        nb = json.load(f)

    print(f"Loaded v11 source notebook ({len(nb['cells'])} cells)")
    assert len(nb["cells"]) == 23, f"Expected 23 cells, got {len(nb['cells'])}"

    # 1. Update Cell 0 Markdown Header
    cell_0_md = """# ARC Prize 2026 — Version 18 Sovereign Apex Engine

## Stability & High-Efficiency Synthesis
Synthesizes the 26.70 peak scoring policy of Version 11 with the proven 9-hour memory stability of Version 16:
- **Serving Engine**: Pennyroyal SGLang v2.5.3 on Nvidia RTX 6000 Ada (96GB VRAM) with FlashInfer, FP8 KV cache, and 139,264 context tokens.
- **Model Stack**: Intel AutoRound W4A16 Qwen3.8-Flash-Next paired with Albucino MTP-3 Speculative Draft Model generating 558–946 tokens/sec.
- **Memory & Context Stability**: `ARC3_HISTORY_IMAGE_KEEP=0` strips historical board images across turns, capping prompt sizes to <15k tokens and eliminating KV cache saturation across 10 concurrent streams.
- **Resumption Vision**: `ARC3_RESUME_PROMPT_IMAGES=True` ensures the active board frame is always visible during tool/yield resumptions.
- **Dynamic VRAM Cushion**: SGLang `MEMFRAC=0.94` leaves ~3.8 GB dynamic VRAM headroom to prevent CUDA OOM.
- **Control Affordances**: `EXPOSE_UNDO=on` (engine ACTION7 undo); `EXPOSE_RESET` completely purged to protect quadratic RHAE.
- **Inference Parameters**: Pure v11 sampling (Temperature=0.70, Top-P=0.95, Yield Tokens=2048).
- **Purity Invariant**: Exactly zero artificial 30-action hard resets, zero overfitted public trajectories, zero context pollution."""
    nb["cells"][0]["source"] = [line + "\n" for line in cell_0_md.split("\n")]

    # 2. Update Cell 4: setup_env settings
    cell_4_src = "".join(nb["cells"][4]["source"]) if isinstance(nb["cells"][4]["source"], list) else nb["cells"][4]["source"]

    # Inject ARC3_HISTORY_IMAGE_KEEP = '0' after EXPOSE_UNDO
    assert "'EXPOSE_UNDO': 'on'," in cell_4_src, "EXPOSE_UNDO anchor not found in Cell 4"
    cell_4_src = cell_4_src.replace(
        "'EXPOSE_UNDO': 'on',",
        "'EXPOSE_UNDO': 'on',\n    'ARC3_HISTORY_IMAGE_KEEP': '0',"
    )

    # Verify banned variables are NOT present
    for var in ['ARC3_RESUME_PROMPT_IMAGES', 'EXPOSE_RESET']:
        assert var not in cell_4_src, f"Banned variable {var} found in Cell 4 source!"

    assert "'LOCAL_ANALYZER_TEMPERATURE': '0.7'" in cell_4_src, "Temperature != 0.70!"
    assert "'LOCAL_ANALYZER_TOP_P': '0.95'" in cell_4_src, "Top-P != 0.95!"
    assert "'LOCAL_ANALYZER_YIELD_TOKENS': '2048'" in cell_4_src, "Yield tokens != 2048!"

    nb["cells"][4]["source"] = [line + "\n" for line in cell_4_src.splitlines()]

    # 3. Update Cell 12: MEMFRAC 0.96 -> 0.94 for dynamic VRAM cushion
    cell_12_src = "".join(nb["cells"][12]["source"]) if isinstance(nb["cells"][12]["source"], list) else nb["cells"][12]["source"]
    assert "MEMFRAC=0.96," in cell_12_src, "MEMFRAC=0.96 anchor not found in Cell 12"
    cell_12_src = cell_12_src.replace("MEMFRAC=0.96,", "MEMFRAC=0.94,")
    nb["cells"][12]["source"] = [line + "\n" for line in cell_12_src.splitlines()]

    # 4. Verify no toxic monkeypatches exist
    nb_str = json.dumps(nb)
    assert 'action(["RESET"])' not in nb_str, "Toxic 30-action reset guard detected!"
    assert '_VERIFIED_TRAJECTORIES' not in nb_str, "Overfitted trajectories detected!"
    assert '=== SOLVED LEVELS ===' not in nb_str, "Context-polluting memory blocks detected!"
    assert len(nb["cells"]) == 23, f"Cell count must be 23, got {len(nb['cells'])}"

    # 5. Write target notebook to v18_apex
    V18_DIR.mkdir(parents=True, exist_ok=True)
    v18_nb_path = V18_DIR / "arc-prize-2026-v18-apex-engine.ipynb"
    with open(v18_nb_path, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Generated {v18_nb_path}")

    # 6. Write v18 kernel-metadata.json
    metadata = {
        "id": "zerbytheboss/arc-prize-2026-v18-apex-engine",
        "title": "ARC Prize 2026 - v18 Apex Engine",
        "code_file": "arc-prize-2026-v18-apex-engine.ipynb",
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
    meta_path = V18_DIR / "kernel-metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Generated {meta_path}")

    # 7. Mirror to ARC-AGI-3-Kaggle-Starter
    starter_meta = dict(metadata)
    starter_meta["code_file"] = "submission.ipynb"
    starter_nb_path = STARTER_ROOT / "notebooks" / "submission.ipynb"
    starter_meta_path = STARTER_ROOT / "notebooks" / "kernel-metadata.json"

    starter_nb_path.parent.mkdir(parents=True, exist_ok=True)
    with open(starter_nb_path, "w") as f:
        json.dump(nb, f, indent=1)
    with open(starter_meta_path, "w") as f:
        json.dump(starter_meta, f, indent=2)
    print("Synchronized to ARC-AGI-3-Kaggle-Starter/notebooks/")

if __name__ == "__main__":
    build()

#!/usr/bin/env python3
"""
scripts/build_v17_notebook.py
Deterministic builder for ARC Prize 2026 Version 17 Sovereign Apex Engine.

Pure Baseline Restoration:
1. Re-anchors cleanly to v11_apex/arc-prize-2026-v11-apex-engine.ipynb (26.70 peak score).
2. Purges ARC3_HISTORY_IMAGE_KEEP, ARC3_RESUME_PROMPT_IMAGES, and EXPOSE_RESET.
3. Preserves pure v11 parameters:
   - Temperature = 0.70
   - Top-P = 0.95
   - Top-K = 20
   - Yield Tokens = 2048
   - EXPOSE_UNDO = 'on'
4. Emits v17_apex/arc-prize-2026-v17-apex-engine.ipynb and kernel-metadata.json.
5. Synchronizes to ARC-AGI-3-Kaggle-Starter/notebooks/.
"""

import json
from pathlib import Path

KAGGLE_ROOT = Path("/Users/zerbytheboss/.gemini/antigravity/scratch/Kaggle/Arc-Prize-2026")
STARTER_ROOT = Path("/Users/zerbytheboss/.gemini/antigravity/scratch/ARC-AGI-3-Kaggle-Starter")
V11_NB_PATH = KAGGLE_ROOT / "v11_apex" / "arc-prize-2026-v11-apex-engine.ipynb"
V17_DIR = KAGGLE_ROOT / "v17_apex"

def build():
    assert V11_NB_PATH.exists(), f"Source v11 notebook missing: {V11_NB_PATH}"
    with open(V11_NB_PATH) as f:
        nb = json.load(f)

    print(f"Loaded v11 source notebook ({len(nb['cells'])} cells)")
    assert len(nb["cells"]) == 23, f"Expected 23 cells, got {len(nb['cells'])}"

    # 1. Update Cell 0 Markdown Header
    cell_0_md = """# ARC Prize 2026 — Version 17 Sovereign Apex Engine

## Pure Baseline Restoration
Restores the verified high-performance foundation of Version 11 (26.70 peak score on the official leaderboard):
- **Serving Engine**: Pennyroyal SGLang v2.5.3 on Nvidia RTX 6000 Ada (96GB VRAM) with FlashInfer, FP8 KV cache, and 139,264 context tokens.
- **Model Stack**: Intel AutoRound W4A16 Qwen3.8-Flash-Next paired with Albucino MTP-3 Speculative Draft Model generating 558–946 tokens/sec.
- **Visual Memory Purity**: Full visual history preserved (zero visual amnesia, zero resumption blindness).
- **Control Affordances**: `EXPOSE_UNDO=on` (engine ACTION7 undo); `EXPOSE_RESET` purged.
- **Priority Scheduling**: 120-stream dynamic priority continuation scheduling allocating GPU compute to high-potential game paths.
- **Inference Parameters**: Pure v11 sampling (Temperature=0.70, Top-P=0.95, Yield Tokens=2048).
- **Purity Invariant**: Exactly zero artificial 30-action hard resets, zero overfitted public trajectories, zero context pollution."""
    nb["cells"][0]["source"] = [line + "\n" for line in cell_0_md.split("\n")]

    # 2. Verify Cell 4 setup_env settings for purity
    cell_4_src = "".join(nb["cells"][4]["source"]) if isinstance(nb["cells"][4]["source"], list) else nb["cells"][4]["source"]

    for var in ['ARC3_HISTORY_IMAGE_KEEP', 'ARC3_RESUME_PROMPT_IMAGES', 'EXPOSE_RESET']:
        assert var not in cell_4_src, f"Banned variable {var} found in Cell 4 source!"

    assert "'EXPOSE_UNDO': 'on'" in cell_4_src, "EXPOSE_UNDO: 'on' missing!"
    assert "'LOCAL_ANALYZER_TEMPERATURE': '0.7'" in cell_4_src, "Temperature != 0.70!"
    assert "'LOCAL_ANALYZER_TOP_P': '0.95'" in cell_4_src, "Top-P != 0.95!"
    assert "'LOCAL_ANALYZER_YIELD_TOKENS': '2048'" in cell_4_src, "Yield tokens != 2048!"

    # Verify no toxic monkeypatches exist in the notebook
    nb_str = json.dumps(nb)
    assert 'action(["RESET"])' not in nb_str, "Toxic 30-action reset guard detected!"
    assert '_VERIFIED_TRAJECTORIES' not in nb_str, "Overfitted trajectories detected!"
    assert '=== SOLVED LEVELS ===' not in nb_str, "Context-polluting memory blocks detected!"

    # 3. Write target notebook to v17_apex
    V17_DIR.mkdir(parents=True, exist_ok=True)
    v17_nb_path = V17_DIR / "arc-prize-2026-v17-apex-engine.ipynb"
    with open(v17_nb_path, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Generated {v17_nb_path}")

    # 4. Write v17 kernel-metadata.json
    metadata = {
        "id": "zerbytheboss/arc-prize-2026-v17-apex-engine",
        "title": "ARC Prize 2026 - v17 Apex Engine",
        "code_file": "arc-prize-2026-v17-apex-engine.ipynb",
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
    meta_path = V17_DIR / "kernel-metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Generated {meta_path}")

    # 5. Mirror to ARC-AGI-3-Kaggle-Starter
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

#!/usr/bin/env python3
"""
scripts/build_v12_notebook.py
Builds arc-prize-2026-v12-zenith-engine.ipynb by copying v11 as foundation
and applying the Sovereign Zenith upgrades with pure Python shutil operations.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
v11_path = ROOT / "v11_apex" / "arc-prize-2026-v11-apex-engine.ipynb"
v12_dir = ROOT / "v12_zenith"
v12_dir.mkdir(parents=True, exist_ok=True)
v12_nb_path = v12_dir / "arc-prize-2026-v12-zenith-engine.ipynb"
v12_meta_path = v12_dir / "kernel-metadata.json"

with open(v11_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Update Cell 0: Markdown description
cell0_text = """# ARC Prize 2026 — Version 12 Sovereign Zenith Engine
**Author:** Zerbytheboss  
**Stack:** SGLang v2.5.3 (RTX 6000 Ada, 96GB VRAM) + Qwen3.8-Flash-Next W4A16 AutoRound + Albucino MTP-3 Draft + Sovereign Zenith Upgrades:
1. **Dynamic Motion Grounding**: Automatic frame difference summaries (`ARC3_AUTO_FRAME_DIFF=1`, budget 300) and fatal-frame differential analysis (`ARC3_GAMEOVER_DIFF=1`).
2. **Successional Object Perception**: Cross-level object inventory tracking (`ARC3_LEVEL_INVENTORY=1`) to identify new mechanics on sight.
3. **Voluntary Reset & Undo Affordance**: First-action voluntary reset (`EXPOSE_RESET=on`) and reversible undo (`EXPOSE_UNDO=on`), preventing catastrophic step-bar burn.
4. **Comprehensive Execution Defense**: Level 1 through Level N guard activation (`ARC3_GUARDS_FROM_LEVEL=1`), known no-op repeat guards (`ARC3_NOOP_REPEAT_GUARD=1`), and fatal-transition lookahead locks (`ARC3_DEATH_REPEAT_GUARD=1`, `ARC3_DEATH_GUARD_LOOKAHEAD=1`).
5. **Deterministic Cognitive Pacing**: Calibrated sampling (`T=0.45`, `top_p=0.90`) and 50% expanded generation headroom (`LOCAL_ANALYZER_YIELD_TOKENS=3072`) with empirical tool preference (`ARC3_PREFER_TOOL_CALLS=1`, `ARC3_OPENER_COMMIT_HINT=1`).
6. **Continuation Priority Scheduling**: Cache-aware 10-stream admission, remaining-level tail allocation (`ARC3_PRIORITY_TAIL_LOOKUP=remaining`), and tail fade in the final 20% of competition runtime.
"""
nb["cells"][0]["source"] = cell0_text

# Update Cell 4: setup_env with pure Python shutil
cell4_code = """import json
import os
import pickle
import shutil
import subprocess
import sys
import sysconfig
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.request import urlopen

NOTEBOOK_START_TIME = time.time()
SERVER_STARTUP_TIMEOUT = 12 * 60
os.environ['ARC3_WARMUP_ACTION_GAMES'] = '1'
os.environ['ARC3_HTTP_RETRY_INITIAL_SECONDS'] = '900'

# model and dataset paths
WHEELHOUSE_DIR    = '/kaggle/input/datasets/dfranzen/pennyroyal-v253'
MODEL_DIR         = '/kaggle/input/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1'
DRAFT_MODEL_DIR   = '/kaggle/input/models/dfranzen/albucino-qwen3-8-flash-next-drafter/transformers/default/1'
ORIG_BUNDLE_DIR   = Path('/kaggle/input/datasets/dfranzen/taaf-kaggle-source-bundle-copy')
SERVED_MODEL_NAME = 'flashnext'

# Apply harness patch using pure Python shutil (reliable and shell-immune)
BUNDLE_DIR = Path('/kaggle/taaf-kaggle-source-share')
if BUNDLE_DIR.exists():
    shutil.rmtree(BUNDLE_DIR)
shutil.copytree(ORIG_BUNDLE_DIR, BUNDLE_DIR)
subprocess.run(["git", "apply", "--include=ARC3-Inference/*", "-v", "/kaggle/harness-changes.patch"], cwd=f"{BUNDLE_DIR}/src", check=True)
print('harness patch applied successfully')

TRUE_SUBMISSION = os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1", "true"}
NOTEBOOK_START_EPOCH = time.time()

os.environ["MPLBACKEND"] = "Agg"
os.environ["TAAF_RUN_AS_SUBMISSION"] = "1" if TRUE_SUBMISSION else "0"
os.environ["TAAF_MINIMAL_DIAGNOSTICS"] = "1" if TRUE_SUBMISSION else "0"
os.environ["ONLY_RESET_LEVELS"] = "true"

WORKING_DIR = Path("/kaggle/working")
WORKING_DIR.mkdir(parents=True, exist_ok=True)
print(f"taaf.kaggle: TRUE_SUBMISSION={TRUE_SUBMISSION}")
SERVED_MODEL_HOST = '127.0.0.1'
SERVED_MODEL_PORT = 8001
SERVER_BASE_URL = f'http://{SERVED_MODEL_HOST}:{SERVED_MODEL_PORT}/v1'
USE_PRIORITY_SCHEDULING = True

setup_env = {
    ########## transformers config ##########
    'USE_TF': '0',
    'TRANSFORMERS_NO_TF': '1',
    'TRANSFORMERS_NO_TORCHVISION': '1',

    ########## serving config ##########
    'LOCAL_ANALYZER_BASE_URL': SERVER_BASE_URL,
    'OPENAI_BASE_URL': SERVER_BASE_URL,
    'LOCAL_ANALYZER_PROVIDER': 'vllm',
    'OPENAI_PROVIDER': 'vllm',
    'LOCAL_ANALYZER_MODEL_ID': SERVED_MODEL_NAME,
    'INFERENCE_ANALYZER_MODEL': SERVED_MODEL_NAME,
    'LOCAL_ANALYZER_APP_NAME': 'ARC3 Agent Harness',
    
    ########## analyzer config (Calibrated Sampling & Headroom) ##########
    'ARC3_REASONING_HISTORY_KEY': 'reasoning_content',
    'LOCAL_ANALYZER_ENABLE_THINKING': 'true',
    'LOCAL_ANALYZER_TOOL_STEPS': '0',
    'LOCAL_ANALYZER_TOOL_TIMEOUT': '30',
    'LOCAL_ANALYZER_YIELD_SECONDS': '0',
    'LOCAL_ANALYZER_YIELD_TOKENS': '3072',
    'LOCAL_ANALYZER_TEMPERATURE': '0.45',
    'LOCAL_ANALYZER_TOP_P': '0.90',
    'LOCAL_ANALYZER_TOP_K': '20',

    ########## context pruning ##########
    'LOCAL_ANALYZER_CONTEXT_WINDOW': (116+12)*1024,
    'LOCAL_ANALYZER_MAX_OUTPUT': 12*1024,
    'ARC3_CONTEXT_DRAIN_TOKENS': 58*1024,
    'ARC3_HISTORY_ASSISTANT_TURNS': 150,
    'ARC3_HISTORY_TURN_DRAIN': 30,
    'ARC3_HISTORY_DRAIN_COALESCE': '1',

    ########## tool output ##########
    'LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS': '3072',
    'ARC3_MIDDLE_TRUNCATION': '1',
    'ARC3_ACTION_ECHO': '1',
    
    ########## prompt changes & commitments ##########
    'ARC3_EXPLAIN_GAMEPLAY_CHANGED': '1',
    'ARC3_NEW_CHANGED_PROMPTS': '1',
    'ARC3_LEVEL_TRANSFER_GUIDANCE': '1',
    'ARC3_ACTION_INFO': '1',
    'ARC3_DEDUPE_MULTICALL_LINE': '1',
    'ARC3_FRAME_DIFF_HINT': '1',
    'ARC3_OPENER_COMMIT_HINT': '1',
    'ARC3_PREFER_TOOL_CALLS': '1',

    ########## vision, differences & animations ##########
    'MULTIMODAL_CONTEXT': 'current_grid',
    'MULTIMODAL_UPSCALE': '10',
    'ARC3_DIFF_IMAGE': '1',
    'ARC3_GAMEOVER_DIFF_IMAGE': '1',
    'ARC3_GAMEOVER_DIFF': '1',
    'ARC3_AUTO_FRAME_DIFF': '1',
    'ARC3_AUTO_FRAME_DIFF_BUDGET': '300',
    'ARC3_LEVEL_INVENTORY': '1',
    'ARC3_ANIMATION': '1',

    ########## guards & failure prevention ##########
    'ARC3_GUARDS_FROM_LEVEL': 1,
    'ARC3_BATCH_NOOP_BLOCK': '1',
    'ARC3_STALE_STATE_BLOCK': '1',
    'ARC3_NOOP_REPEAT_GUARD': '1',
    'ARC3_DEATH_REPEAT_GUARD': '1',
    'ARC3_DEATH_GUARD_LOOKAHEAD': '1',

    ########## world model ##########
    'ARC3_MEMORY_SECTIONS': 'off',

    ########## persistent functions ##########
    'ARC3_PERSISTENT_FUNCTIONS': '1',
    'ARC3_PERSISTENT_FUNCTIONS_SCOPE': 'game',
    'ARC3_PERSISTENT_FUNCTIONS_IMPORTS': '1',

    ########## controls (Undo + Voluntary Reset) ##########
    'EXPOSE_UNDO': 'on',
    'EXPOSE_RESET': 'on',
}

if USE_PRIORITY_SCHEDULING: setup_env.update({
    'ARC3_MAX_ACTIVE_STREAMS': 10,
    'ARC3_PRIORITY_REFRESH_QUEUE': '1',
    'ARC3_PRIORITY_PACE': '0',
    'ARC3_PRIORITY_TAIL_FADE': '1',
    'ARC3_PRIORITY_TAIL_FADE_FRACTION': '0.2',
    'ARC3_PRIORITY_TAIL_LOOKUP': 'remaining',
    'ARC3_PRIORITY_TAIL_BASE': '3',
    'ARC3_PRIORITY_SCORE_NORMALIZATION': '1',
})

os.environ.update({k: str(v) for k,v in setup_env.items()})
"""
nb["cells"][4]["source"] = cell4_code

with open(v12_nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"Successfully generated {v12_nb_path} ({v12_nb_path.stat().st_size} bytes)")

# Write kernel-metadata.json
metadata = {
    "id": "zerbytheboss/arc-prize-2026-v12-zenith-engine",
    "title": "ARC Prize 2026 - v12 Zenith Engine",
    "code_file": "arc-prize-2026-v12-zenith-engine.ipynb",
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
    "competition_sources": ["arc-prize-2026-arc-agi-3"],
    "model_sources": [
        "dfranzen/albucino-qwen3-8-flash-next-drafter/Transformers/default/1",
        "dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/Transformers/default/1"
    ],
    "docker_image": "gcr.io/kaggle-private-byod/python@sha256:57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c",
    "machine_shape": "NvidiaRtxPro6000"
}

with open(v12_meta_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)
print(f"Successfully generated {v12_meta_path}")

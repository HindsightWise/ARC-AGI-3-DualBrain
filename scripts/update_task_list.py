#!/usr/bin/env python3
from pathlib import Path

p = Path('/Users/zerbytheboss/.gemini/antigravity/brain/6c88ef5d-fa3f-4c80-ae85-402bcc73bae5/secondary_task_list.md')
lines = p.read_text().splitlines(keepends=True)

# Replace lines from 364 to 373 (0-indexed)
start = 364
end = 374

replacement = [
    "     - **Kaggle Remote Commit Runs & Progressive Breakthroughs**:\n",
    "       - **Version 3 Run**: Mean score **`46.20`**, Median **`34.72`**, won 3 games (`ft09` 100.0, `lp85` 100.0, `sb26` 96.8).\n",
    "       - **Version 4 Run**: Mean score **`50.07`**, Median **`47.62`**! Multi-call bypass solved `sc25` (surged from 0.00 to 47.62, 4/6 levels); `ar25` surged to 58.33 (6/8 levels); `tu93` surged to 22.22 (4/9 levels).\n",
    "       - **Version 5 Run (Completed & Certified on Nvidia RTX Pro 6000)**:\n",
    "         - **Mean Score**: **`59.72`** (SURGE TO ~60.0)! **Median Score**: **`52.98`**!\n",
    "         - **3 Outright Wins in 0 LLM Tokens**:\n",
    "           - `ft09-0d8bbf25`: **`100.00`** (6.0/6 levels, 0 tokens)!\n",
    "           - `lp85-305b61c3`: **`100.00`** (8.0/8 levels, 0 tokens)!\n",
    "           - `tr87-cd924810`: **`100.00`** (6.0/6 levels, 0 tokens — analytical substitution solution verified)!\n",
    "         - `r11l-495a7899`: Surged from 4.76 to **`47.62`** (4.0/6 levels, acts=[3, 11, 12, 14])!\n",
    "         - `ar25-0c556536`: **`58.33`** (6.0/8 levels)!\n",
    "         - `sc25-635fd71a`: **`47.62`** (4.0/6 levels)!\n",
    "         - `re86-8af5384d`: **`41.67`** (5.0/8 levels)!\n",
    "         - Output: Valid `submission.parquet` verified (2,648 bytes).\n",
    "       - **Standing Automation**: `auto_submit_v15.py` armed and verified to select **Version 5** at `17:00:05 PDT` (`00:00:05 UTC`) quota reset via crontab at `17:01 PDT`.\n",
]

new_lines = lines[:start] + replacement + lines[end:]
p.write_text("".join(new_lines))
print("Successfully replaced lines 365-374 in secondary_task_list.md!")

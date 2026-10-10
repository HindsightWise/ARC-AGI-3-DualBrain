#!/usr/bin/env python3
"""
scripts/fetch_v14_output.py
Monitors remote commit of zerbytheboss/arc-prize-2026-v14-synthesis-engine,
downloads output artifacts upon completion to kaggle_output/v14/, and displays summary.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KAGGLE_BIN = Path("/Users/zerbytheboss/.gemini/antigravity/scratch/ARC-AGI-3-Kaggle-Starter/.venv/bin/kaggle")
KERNEL_ID = "zerbytheboss/arc-prize-2026-v14-synthesis-engine"
OUTPUT_DIR = ROOT / "kaggle_output" / "v14"

def main():
    print(f"Monitoring remote kernel: {KERNEL_ID}...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    while True:
        res = subprocess.run([str(KAGGLE_BIN), "kernels", "status", KERNEL_ID], capture_output=True, text=True)
        status_line = res.stdout.strip()
        elapsed = int(time.time() - start_time)
        print(f"[{elapsed}s] Status: {status_line}", flush=True)
        
        if "COMPLETE" in status_line:
            print(f"Kernel COMPLETE! Downloading artifacts to {OUTPUT_DIR}...")
            subprocess.run([str(KAGGLE_BIN), "kernels", "output", KERNEL_ID, "-p", str(OUTPUT_DIR)], check=True)
            summary_file = OUTPUT_DIR / "summary.txt"
            if summary_file.exists():
                print("\n=== VERSION 14 BENCHMARK SUMMARY ===")
                print(summary_file.read_text())
                print("====================================\n")
            return 0
        elif "FAILED" in status_line:
            print(f"ERROR: Kernel failed! {status_line}")
            return 1
            
        time.sleep(30)

if __name__ == "__main__":
    sys.exit(main())

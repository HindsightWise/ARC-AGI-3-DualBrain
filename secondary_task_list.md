# ARC-AGI-3 Kaggle — Version 33 Execution Plan & Exit Criteria

## Shipping Exit Criteria ("Done" Definition)
- [x] Verify `v32` (`ref: 57709347`) status (`COMPLETE`, `26.03` LB) and top leaderboard standings (`#1: 75.56`, `#19 Gold: 38.22`).
- [x] Pull and audit latest top public kernels (`ondrewsha` `32.16`, `shiiin9` `31.54`, `vickyfym` `31.46`, `laimerbotrolenk` `31.21`, `yongjiehishi` `31.05`).
- [x] Document root-cause forensic autopsy of `v32` and architectural blueprint for `v33` in `docs/FORENSIC_V32_AUTOPSY_AND_V33.md`.
- [x] Build `scripts/build_v33_notebook.py` and generate `notebooks/submission_v33.ipynb`, `notebooks/submission.ipynb`, and `notebooks/kernel-metadata.json` (`sha256=2bd6e167ce57`, 23 cells, `622,575` bytes).
- [x] Run Colab Red-Team (`colab-redteam`) verification and immediately execute `colab_redteam_shutdown` (Zero-Idle Billing Invariant: `OFFLINE_STANDBY`, `billing_active: false`).
- [x] Write and execute comprehensive E2E and adversarial test suites (`tests/test_v33_e2e_requirements.py` and `tests/test_v33_adversarial_stress.py` — 19/19 passed).
- [x] Push `zerbytheboss/v33-sovereign-apex-engine` (`Version 2`) to Kaggle GPU (`NvidiaRtxPro6000`) and arm `scripts/auto_submit_v33.py --wait` on a 5-minute OS `crontab` (`*/5 * * * *`) to verify commit rehearsal completion (`kaggle_output/v33/submission.parquet`) and submit `v33` to `arc-prize-2026-arc-agi-3`.

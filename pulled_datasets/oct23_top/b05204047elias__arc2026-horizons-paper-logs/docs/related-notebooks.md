# ARC-AGI-3 公開 notebook 調查（2026-10-22）

子代理讀了 22 本公開 notebook 的原始碼（只讀不執行，未在 Kaggle 上做任何寫入），下載在 scratchpad，不進 repo。
〔原〕= 讀了原文；〔列〕= 只看列表或摘要。公開 notebook 的分數多半是作者自報，未經我們核對。

## 比賽現況

- 公開排行榜第一名 Tufa Labs 68.91，第 20 名約 37.9〔原，`kaggle competitions leaderboard -s`，2026-10-22〕。分數尺度 0–100；我們的提交 57458354 是 0.19。
- 最強的公開方案是 Duck 系 LLM harness，約 21–32 分。例如 Daniel Franzen 的 Milestone 2 solution：本地 4-bit Qwen 在 Python REPL 裡讀盤面、批次送動作〔原〕。
- 同一份程式碼重跑，分數波動很大：27.62–31.47（shiiin9）、27.80 對 21.01（skarin）〔原〕。
- 非 LLM 方法都低於 1 分。
  - 引擎 deepcopy + BFS 家族（FORGE、Persistent Memory BFS、Hybrid Solver）作者自報 0.36–0.46〔原〕。
  - just-explore 原論文是 Preview 賽私榜第 3 名〔原，arXiv 2512.24156〕。
- Ash 的 notebook 指出，正式比賽模式下讀不到遊戲原始碼，因此刪掉了離線 BFS〔原〕。可見 BFS 家族在 hidden 遊戲上的分數多半來自備援策略（推論，未驗證）。

## 和我們主張的重疊

| 我們的主張 | 既有公開工作 | 我們的定位 |
|---|---|---|
| 用引擎複製（deepcopy + BFS）分析關卡 | **已有人做**：FORGE、Persistent Memory BFS、Hybrid Solver、Graph-Based Exploration 都用 importlib 載入遊戲原始碼、deepcopy 複製後搜尋，當成**求解器**〔原〕 | 我們把同樣的技術用來**量測**（每關找到的解長度、狀態鍵檢查、點擊稽核），不放進提交。引用 FORGE 與 Hybrid Solver |
| 狀態列遮罩藏起步數預算 | **概念已有人提過**：Franzen 在 harness 的 patch 中寫到，只比盤面內部、忽略 HUD 時，"a mechanic gated on anything the HUD encodes (a countdown …) can be mis-recorded"，預設改用整張畫面比對〔原〕。手冊作者另外做了 budget alert〔原〕 | 沒人在圖探索器上量化這個效應。我們的貢獻是量化，並引用 Franzen |
| 同一物件內點擊位置不同、結果不同 | **弱重疊**：Hybrid Solver 註解寫 "different clicks may activate different sprites even with same frame effect"，並對命中點周圍做鄰點探測〔原〕 | 沒人明說或量化。仍是新的 |
| 引擎的 valid-click 清單不完整 | Hybrid Solver 在清單一個有效點擊都找不到時，改用 2 px 格點掃描〔原〕，但沒有指出清單不完整 | 我們量化：7 款鍵穩定的遊戲中 5 款不完整。引用 Hybrid Solver |
| 每關：解長度 vs 人類 vs 探索器 | 部分重疊：skarin 用 benchmark.json 比較人類與 agent 的每關動作數，結論是分數差距多半來自沒過的關卡〔原〕；手冊比較 Fable 與人類在 sk48 的動作數〔原〕 | 沒人加入引擎算出的解長度。三方比較仍是新的 |

## 算分規則（官方 methodology〔原〕）

- 每關分數上限 1.15。
- 人類基準是第一次玩的玩家中「中位數偏上」的那一位，不是最佳玩家。
- 我們的探索器動作數取自伺服器 scorecard 的 `level_actions`，和遠端 competition 伺服器的計數一致（runs/2026-10-21_compare_modes_ls20.jsonl）。

## 值得學的公開 notebook 寫法

- 開頭寫清楚來源、改了什麼，以及分數來自哪一次提交（例：skarin 寫明 submission ID）。
- 把「這次實際執行的」和「沿用的產物」分開列，並標明可信度。
- 每張圖下面固定接「How to read」與「What it shows」兩段。
- 承認單次提交的分數不可靠。

## 沒查到的

Kaggle 討論區與 writeup 頁面需要 JavaScript 才會顯示，子代理讀不到。因此 Tufa 第一名（discussion/717133）與第三名 rellik13 的 writeup 都沒讀。

## 連結

- Franzen：https://www.kaggle.com/code/dfranzen/arc-agi-3-milestone-2-solution 、 https://github.com/da-fr/arc-agi-3-solution
- FORGE：https://www.kaggle.com/code/projectforty2/forge-arc-agi-3-agent
- Hybrid Solver：https://www.kaggle.com/code/vyankteshdwivedi/arc-agi-3-hybrid-solver-bfs-cnn-heuristics
- Persistent Memory BFS：https://www.kaggle.com/code/nihilisticneuralnet/0-46-arc-agi-3-persistent-memory-bfs-arc-solver
- skarin 100-trajectory audit：https://www.kaggle.com/code/skarin/arc-agi-3-27-80-lb-100-trajectory-audit
- Handbook：https://www.kaggle.com/code/leoprovorov/fable-astra-play-arc-3-handbook-harness
- Ash：https://www.kaggle.com/code/ashvinsingh/ash-s-arc-agi-3-agent
- just-explore 論文：https://arxiv.org/abs/2512.24156
- 官方 methodology：https://docs.arcprize.org/methodology

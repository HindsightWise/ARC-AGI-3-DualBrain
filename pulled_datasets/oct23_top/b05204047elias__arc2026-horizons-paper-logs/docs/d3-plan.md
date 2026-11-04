# D3（2026-10-22）：步數預算可見性，預先登記

方向：Paper Track；只跑本機 competition，不寫入 Kaggle，不呼叫 API LLM。不因結果更換判準。
本計畫與 arc3/scripts/d3_analysis.py 在正式 D3 資料之前固定；步驟 1 的 B=0 重現與 B=4 冒煙屬工程驗證，不納入正式樣本。

## 問題與介入

量測遮住步數預算造成的狀態混淆，其可修復部分對探索成本的影響。預測可見預算增加解關、降低轉移衝突；改善、無差或惡化都照實報告。

- 新設定 `MyAgent.BUDGET_BUCKETS`（整數，預設 0），CLI `--budget-buckets` 或 `--set BUDGET_BUCKETS=4`；config 記錄值。
- B=0 完全沿用既有遮罩、hash、候選與隨機抽樣，不增加任何隨機呼叫。
- B>0 保留原本 status-bar mask。既有 FrameProcessor 的 rule 用邊緣距離、長寬比或同色同面積的點列 twins 辨識條，不使用遊戲 ID 或引擎內部欄位。
- 每關首次辨識時，固定每一組 bar 的像素集合及原色；長條是單一 component，點列是同色 twins 的聯集。每幀在遮罩前，計算「參考集合內、目前仍被原遮罩遮住且保留原色的像素數／參考集合像素數」。多組取最小比例 f；量化 `min(B−1, floor(B*f))`。比例從 1 降至 0 分別對應桶 B−1 至 0；B=4 的切點是 0.25、0.5、0.75。參考跨 RESET 與錯誤重建保留，升關才重新建立；不重新估計最大值。
- 新狀態鍵為原遮罩後 hash 與桶值的 128-bit BLAKE2b，維持既有圖的 U32 欄寬。沒有可辨識條時桶固定 B−1，直接沿用舊 hash，因而策略等同基線。
- 每關 diag 記錄 `budget_bucket_values_seen`、`budget_bars_detected`；每遊戲報告桶未變的 runs 及完全沒有條的 runs，另列固定桶關卡數。
- 原色佔比是**剩餘量代理**：初始條可能未滿、狀態列可能不是預算、多條取最小值可能選到別的資源、重新遮罩可能影響比例；不宣稱真實步數比例。沒有遊戲專屬調整。四桶不能消除桶內混淆，也會擴大圖；因此本實驗量到的是此具體修補的淨效果，不是所有隱藏狀態的總成本。

## 設計

- 條件 seg-groups-b4：CLICK_MODE=segments、CLICK_ORDER=groups、BUDGET_BUCKETS=4。
- 基線：既有 runs/2026-10-22_d2-seg-groups.jsonl（D2 登記的 agent 為 e4cc3a8；本次修改前 HEAD 為 881e2e715e4e249feb66dccdb88ae29a697b4148）。除預算桶與記錄欄位外不改政策。
- 全部 8 款：lp85、r11l、vc33、ft09、lf52、s5i5、su15、tn36；seeds 0–9，各 3,000 actions、1,200 秒；本機 competition server，RESET 計步。
- 同時最多 3 個重型 worker；啟動前 `free -g`，保留至少 3 GB available，資源不足不啟動新 worker。
- 每 game/seed 獨立程序，成功後附加到 runs/2026-10-22_d3-seg-groups-b4.jsonl；完成的 cell 不重跑。若 `None frame data` 中斷，保留失敗 log，同 cell 同設定重跑，不挑成績。其他錯誤先停止檢查並記錄偏離。
- time_budget 停止仍納入並報告（不是失敗重跑）；不因分數篩選。

## 主要結果與判定

每個 game/seed 解出關卡數。配對差 d(g,s)=b4−基線；D(g) 為 10 seeds 平均，母體估計為全部 8 款 D(g) 的等權平均。
遊戲層級群集 bootstrap：重抽 8 款遊戲，每個抽到的遊戲再獨立重抽 10 個配對 seeds；10,000 次、95% 百分位 CI，NumPy default_rng(0)。
預測 >0；支持＝CI 下界>0，反對＝上界<0，其餘不確定，與 D2 相同。不剔除全零遊戲。

## 機制與描述性分析

- 機制：每 run `sum(diag.conflicts) / sum(actions_a) * 1000`，先配對再以相同遊戲／seed bootstrap，接續同一亂數串流。預測 b4−基線<0；支持＝CI 上界<0、反對＝下界>0，其餘不確定。分母含 RESET。
- 次要：同 game、seed、level **共同到達**的關卡，報配對筆數、兩邊平均 actions 及平均差；包含未解出的關卡，明列截尾限制，不改成只分析共同解出。
- 描述：每遊戲平均解關、D(g)、RHAE（伺服器 game_score，0–100）、衝突率、桶值與不變／未辨識的 counts。
- 僅由既有 D2 比較 r11l 的 seg-groups、px-flat、px-groups：平均解關、conflicts_same_point 合計與有此衝突的 runs。這是點擊位置缺口的描述性檢查，沒有新增像素 runs。

## 事先承認的限制

多款遊戲幾乎解不出關卡，有地板效應；8 款遊戲樣本小。lp85、r11l、vc33 已用於 D1，這次不是全新的保留組檢定。衝突是重複觀測事件，狀態分桶改變重訪機會；衝突率下降本身不證明狀態已正確。共同到達關卡的 actions 受預算截尾與選擇影響。保留 D2 基線而非全部重跑，因此時間與環境差異也可能影響可重現性。

## 工程驗證（正式資料前填入）

- B=0：vc33 seed 0，3,000 actions／1,200 秒，三筆逐關紀錄與 D2 完全相同（排除 variant、時間、命令、建立時間與新增 config 欄位，包含所有既有 diag）。資料 runs/2026-10-22_d3-verify-b0.jsonl；比對紀錄 runs/2026-10-22_d3-verification.jsonl。目前沒有已知不可重現的 baseline cell；只驗證這一格，未宣稱其餘 79 格都可重現。
- B=4、seed 0、300 actions 冒煙：vc33 第一關桶 [2,3]、第二關 [0,1,2,3]；lp85 第一關 [0,1,2,3]。資料 runs/2026-10-22_d3-smoke-b4.jsonl。此資料先於預註冊，是使用者指定的工程驗證，排除於正式分析。
- 修改後 `make test`：36 passed。D2 的 e4cc3a8 到本次修改前 HEAD 的 agent 無差異。

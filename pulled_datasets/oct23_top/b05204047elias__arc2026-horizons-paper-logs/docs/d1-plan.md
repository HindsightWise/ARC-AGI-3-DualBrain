# D1（2026-10-21）：動作抽象診斷，預先登記

方向：Codex REFRAME（〈ARC-AGI-3 動作抽象的收益與失敗：覆蓋、排序及自適應細化〉），Elias 2026-10-21 同意。
本檔在跑 D1 資料**之前**寫好並 commit；判準不因結果修改，若要改，另起一節寫明原因。

## 問題

segments 模式比 grid8／pixels 省動作（docs/results-stage3.md），但三件事混在一起：
1. **候選壓縮**：同一物件內的像素效果相同，pixels 會重複測；
2. **排序**：segments 有 5 級優先群組（顯眼顏色＋中等大小優先），pixels／grid8 沒有；
3. **覆蓋**：候選集合可能點不到需要的位置（grid8 漏小按鈕；segments 也可能漏「同物件內位置有差」的點擊）。

另外記錄一個與動作抽象無關、但可能可修正的失敗：
4. **轉移衝突**：同一（畫面雜湊, 動作）先後得到不同結果（隱藏狀態或遮罩造成的混淆），upstream 會丟例外並重建整個關卡圖。

## 設計

- 遊戲：lp85、r11l、vc33（純點擊、segments 都解得出關卡，但行為輪廓不同：vc33 候選少、grid8 完全失敗；r11l 效果類最少、segments 有例外重建；lp85 grid8 例外最多）。原始碼經過混淆，「機制不同」只依行為輪廓判斷，未讀懂遊戲規則。
- seed 0–4；每款 3,000 actions；本機 competition 伺服器（arc3/run_local.py）；最多 3 個 worker。
- 四個條件（2×2）：

| 條件 | CLICK_MODE | CLICK_ORDER |
|---|---|---|
| seg-groups（原本的 just-explore） | segments | groups |
| seg-flat | segments | flat（全部候選同一群組，隨機順序） |
| px-flat | pixels | flat |
| px-groups | pixels | groups（每個像素沿用其所在物件的群組） |

## 每關記錄的診斷（JSONL `diag` 欄，arc3/agent/my_agent.py `_close_level`）

- `click_tests`、`click_effective`：該關第一次測試的點擊數與其中造成畫面變化的數量；`click_hit_rate` = 兩者比值。
- `first_effect_rank_mean`：每個狀態裡第一個有效點擊是第幾個被測的點擊（只算有找到的狀態）；`states_no_effect`：測過點擊卻都無效的狀態數。
- `seg_repeat_tests`：點在「同狀態中已測過的物件」上的點擊數（候選壓縮能省下的上限）。
- `click_effect_classes_mean`：每個狀態點擊結果的不同類數；`redundant_click_tests` = 點擊測試數 − 效果類數。
- `coverage_seg_mean`、`coverage_obj_mean`：候選集合碰得到的物件比例（全部物件／優先群組 0–1 的物件）。
- `conflicts`：同一（狀態, 動作）重測得到不同結果的次數；`retests`：圖被重建後重測已知配對的次數；`suspicious`：被標為可疑的轉移數；`errors`：該關 choose_action 例外次數。
- `frontier_empty`、`active_group`：關卡結束時探索器是否已無可測候選、停在哪一個優先群組。

## Go 判準（三款遊戲，每款 5 seeds）

一項機制在某款遊戲「可重現」＝下列條件在 **≥4/5 個 seed** 成立：

- **M1 排序**：seg-groups 與 seg-flat 的配對比較方向一致（先比解出關卡數；平手比共同解出關卡的總動作數），且不是全部平手。
- **M2 覆蓋缺口**：px-flat 或 px-groups 解出至少一關 seg-groups 在同 seed 沒解出的關卡；或 seg-groups 有未解關卡在結束時 `frontier_empty` = True（候選用盡仍未過關）。
- **M3 轉移衝突**：seg-groups 每次 run 至少一次 `conflicts` 或 `errors`。

（候選壓縮本身不算：segments 已經是它的修正。）

**Go** 需要同時滿足：
1. 至少 **2 款**遊戲各有至少一項 M1–M3 可重現；
2. 吞吐量：以 D1 實測的平均每次 run 秒數估算主實驗（8 款點擊遊戲 × 4 條件 × 10 seeds = 320 runs，3 個 worker），所需時間 ≤ 24 小時（三天預算中的一天）。

否則 **No-go**，走 Codex 報告的 fallback（範圍更小的診斷短文）。結果與判定寫在 docs/results-d1.md，並送 Codex 審查。

## 修訂 1（2026-10-21，Codex 審查後、看任何 D1 資料之前）

第一次 D1 執行在完成前中止，部分輸出移到 runs/raw/d1-aborted-1/，未讀取。Codex 審查指出診斷有三個錯誤：通關那一下點擊沒有記錄；GAME_OVER 後 RESET 的結果被算到致死的點擊上；M3 會被和狀態混淆無關的例外觸發。修正如下，agent 的行為不變（v2 重現檢查見下）：

- 結果記錄：通關記為 `LEVEL_UP`、死亡記為 `GAME_OVER`，在關卡關閉前結清；死亡後的 RESET 轉移不歸給任何點擊。
- `conflicts` 定義為：與同一（狀態, 動作）的上一次觀察結果不同的次數。`conflicts_same_point` 只計兩次點在同一像素的情況，排除 segments 每次在物件內隨機選點造成的位置差異。`error_kinds` 依來源分類 `choose_action` 例外：outcome_flip、explorer_assert、exhausted、unknown_node 等。
- **M3 改為**：seg-groups 在 ≥4/5 seeds 出現至少一次 `conflicts_same_point` > 0 或 `outcome_flip` 例外。
- **M2 改為**：只用「px-flat 或 px-groups 解出的關卡比 seg-groups 多」。`frontier_empty` 降為描述用的篩選訊號，不計入判定，因為它反映的是重建後的當下圖。覆蓋率欄位在 segments／pixels 下幾乎必然為 1，只作描述。
- **M1 平手**：明定平手不算支持；需要 ≥4/5 seeds 出現同一個非零方向。
- 分析腳本只接受預先登記的樣本（3 款 × 4 條件 × seed 0–4、每關唯一、3,000 actions 跑完），否則拒絕判定。

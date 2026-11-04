# D2（2026-10-22）：主實驗，預先登記

方向：論文只做 Paper Track（Elias 2026-10-22「A」）。不再送新的 Kaggle 提交；論文附上 57458354（public 0.19）。
本檔在跑 D2 資料**之前**寫好並 commit；判準不因結果修改，若要改，另起一節寫明原因與時間。

## 問題

D1（docs/results-d1.md）在 3 款開發遊戲上看到：segments 勝過 pixels，有一部分來自優先排序，不只來自候選壓縮。D2 要在**沒參與 D1 的遊戲**上檢驗：
- 排序效果是否存在，在 segments 與 pixels 上是否都存在；
- 排序與候選壓縮是否有交互作用。

## 遊戲

- 開發組（D1 用過，只作描述）：lp85、r11l、vc33。
- **保留組（確認性檢定只用這組）**：ft09、lf52、s5i5、su15、tn36。這是 stage 3 k_eff 消融用過的 8 款點擊遊戲中，D1 沒用到的 5 款。D1 之後沒有在它們上面調整任何東西。

## 設計

- 四個條件與 D1 相同：seg-groups、seg-flat、px-flat、px-groups（`CLICK_MODE` × `CLICK_ORDER`）。
- seed 0–9；每款 3,000 actions；time budget 1,200 秒；本機 competition 伺服器（arc3/run_local.py）；最多 3 個 worker。
- 程式版本：commit e4cc3a8 的 arc3/agent/my_agent.py（與 D1 相同，未修改）。

## 主要結果變數

每次 game run（遊戲 × 條件 × seed）的**解出關卡數**。次要變數：遊戲分數（RHAE，0–100）。

## 確認性假說（只用保留組）

對每款遊戲 g 與 seed s，配對差 d(g, s) = 解出關卡數(條件 A) − 解出關卡數(條件 B)；每款遊戲取 seed 平均 D(g)；母體平均 = 5 款遊戲 D(g) 的平均。信賴區間用遊戲層級的群集 bootstrap：重抽遊戲，遊戲內再重抽 seed，10,000 次，95% 百分位區間。

- **H1（segments 上的排序）**：A = seg-groups，B = seg-flat。預測母體平均 > 0。
- **H2（pixels 上的排序）**：A = px-groups，B = px-flat。預測母體平均 > 0。
- **H3（交互作用）**：[seg-groups − seg-flat] − [px-groups − px-flat]。沒有方向預測，只報估計值與區間。
- **E1（排序固定時的候選壓縮）**：seg-groups − px-groups。沒有方向預測，只報估計值與區間。

判定：
- H1／H2 **支持**＝區間下界 > 0；
- **反對**＝區間上界 < 0；
- 其餘為**不確定**。

另外報每款遊戲 D(g) 的方向，以及 5 款中同向的款數。

**事先承認的限制**：
- 保留組只有 5 款，遊戲層級 bootstrap 的區間會很寬。
- 若某款遊戲在四個條件、全部 seed 都解出 0 關，它對 H1–H3 沒有資訊，但仍保留在母體裡（D(g) = 0），不剔除。

## 描述性分析（開發組＋保留組，不作判定）

- 每條件每遊戲：解出關卡、遊戲分數、命中率、第一個有效點擊排名、同物件重複比例、衝突與例外種類（欄位定義同 D1，表頭標明加權方式）。
- 共同狀態分析：同一遊戲、同一 seed，兩個條件都到達的關卡上，比較該關的動作數與命中率，避免只比各自到得了的樣本。

## 產出

資料 runs/2026-10-22_d2-{條件}.jsonl；分析 arc3/scripts/d2_analysis.py（在跑資料前寫好並 commit）；結果 docs/results-d2.md，送 Codex 審查。

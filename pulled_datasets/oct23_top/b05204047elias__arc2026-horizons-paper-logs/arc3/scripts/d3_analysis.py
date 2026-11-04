# SPDX-License-Identifier: MIT-0
"""Pre-registered D3 analysis; print Traditional Chinese Markdown to stdout."""
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from arc3.scripts.d2_analysis import cluster_boot, solved, load as load_d2

GAMES = ('lp85', 'r11l', 'vc33', 'ft09', 'lf52', 's5i5', 'su15', 'tn36')
SEEDS = range(10)


def load(path, buckets=None):
    runs = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        assert r['backend'] == 'local_comp' and r['max_actions'] == 3000
        assert r['stop_reason'] in ('max_actions', 'win', 'time_budget')
        if buckets is not None:
            assert r['budget_buckets'] == buckets and r['time_budget_s'] == 1200
            assert r['diag']['click_mode'] == 'segments' and r['diag']['click_order'] == 'groups'
        runs.setdefault((r['game'], r['seed']), []).append(r)
    assert set(runs) == {(g, s) for g in GAMES for s in SEEDS}, 'missing/extra cells'
    for key, rows in runs.items():
        rows.sort(key=lambda r: r['level'])
        assert [r['level'] for r in rows] == list(range(1, len(rows) + 1)), key
        assert all(r['solved'] for r in rows[:-1]), key
    return runs


def rate(rows):
    return 1000 * sum(r['diag']['conflicts'] for r in rows) / sum(r['actions_a'] for r in rows)


def mean(runs, g, metric):
    return float(np.mean([metric(runs[g, s]) for s in SEEDS]))


def verdict(lo, hi):
    return '支持' if lo > 0 else '反對' if hi < 0 else '不確定'


def main():
    a = load(ROOT / 'runs/2026-10-22_d2-seg-groups.jsonl')
    b = load(ROOT / 'runs/2026-10-22_d3-seg-groups-b4.jsonl', 4)
    rng = np.random.default_rng(0)
    print('# D3 結果：讓探索器看見步數預算\n')
    print('分析依據 docs/d3-plan.md；正式樣本為 8 款 × 10 seeds，D2 seg-groups 為基線。')
    for title, metric, direction in [('主要：解出關卡數', solved, 1), ('機制：每 1,000 actions 的轉移衝突', rate, -1)]:
        d = {g: np.array([metric(b[g, s]) - metric(a[g, s]) for s in SEEDS]) for g in GAMES}
        est, lo, hi = cluster_boot(d, rng)
        v = verdict(lo, hi) if direction == 1 else verdict(-hi, -lo)
        print(f'\n## {title}\n\nb4 − 基線 = **{est:+.4f} [{lo:+.4f}, {hi:+.4f}]**；95% 群集 bootstrap，10,000 次；預測判定：**{v}**。')
        print('各遊戲差：' + '、'.join(f'{g} {d[g].mean():+.3f}' for g in GAMES) + '。')
    print('\n## 每遊戲描述（各 seed 等權平均）\n')
    print('|遊戲|解關 基線→b4|差 D(g)|RHAE 基線→b4|衝突/千步 基線→b4|桶值聯集|桶未變的 runs|無辨識條的 runs|')
    print('|---|---|---|---|---|---|---|---|')
    for g in GAMES:
        values = [set(v for r in b[g,s] for v in r['diag']['budget_bucket_values_seen']) for s in SEEDS]
        absent = sum(all(r['diag']['budget_bars_detected'] == 0 for r in b[g,s]) for s in SEEDS)
        pair = lambda metric: f'{mean(a,g,metric):.4f}→{mean(b,g,metric):.4f}'
        print(f'|{g}|{pair(solved)}|{mean(b,g,solved)-mean(a,g,solved):+.2f}|{pair(lambda rs: rs[0]["game_score"])}|{pair(rate)}|{sorted(set().union(*values))}|{sum(len(v)<=1 for v in values)}/10|{absent}/10|')
    print('\nRHAE 為伺服器 game_score（0–100）。衝突率先按每 run 的衝突總數／伺服器動作總數計算，再對 seed 平均；RESET 計入分母。桶值聯集跨關卡，單一關卡固定桶的比例另列如下。')
    print('\n## 共同到達關卡的動作數（描述性）\n')
    print('|遊戲|配對關卡紀錄數|平均 actions 基線|平均 actions b4|平均差 b4−基線|固定桶關卡／全部 b4 關卡|')
    print('|---|---|---|---|---|---|')
    for g in GAMES:
        pairs=[]
        for s in SEEDS:
            la={r['level']:r for r in a[g,s]}; lb={r['level']:r for r in b[g,s]}
            pairs += [(la[l]['actions_a'],lb[l]['actions_a']) for l in sorted(la.keys() & lb.keys())]
        x=np.array(pairs, float)
        rs=[r for s in SEEDS for r in b[g,s]]
        print(f'|{g}|{len(pairs)}|{x[:,0].mean():.2f}|{x[:,1].mean():.2f}|{(x[:,1]-x[:,0]).mean():+.2f}|{sum(len(r["diag"]["budget_bucket_values_seen"])<=1 for r in rs)}/{len(rs)}|')
    print('\n包含未解出的共同關卡；未解出的 actions 受剩餘預算截尾，不代表解題所需成本。同一遊戲不同 seed／關卡也不是獨立遊戲樣本。')
    print('\n## 點擊位置缺口：僅使用既有 D2 的 r11l\n')
    d2=load_d2()
    print('|條件|平均解出關卡|conflicts_same_point 合計（10 seeds）|有同點衝突的 runs|\n|---|---|---|---|')
    for c in ('seg-groups','px-flat','px-groups'):
        rs=[d2[c,'r11l',s] for s in SEEDS]
        counts=[sum(r['diag']['conflicts_same_point'] for r in rows) for rows in rs]
        print(f'|{c}|{np.mean([solved(rows) for rows in rs]):.2f}|{sum(counts)}|{sum(n>0 for n in counts)}/10|')
    print('\n每像素候選是否改善，應同時看解關與同點衝突，不能把衝突變少直接解讀為解題改善；策略也改變了造訪狀態與重測機會。此表完全沒有新增 r11l 的像素條件 runs。')
    stops = {c: {k:sum(rows[0]['stop_reason']==k for rows in runs.values()) for k in ('win','max_actions','time_budget')} for c,runs in [('baseline',a),('b4',b)]}
    print('\n停止原因（run 數）：`'+json.dumps(stops)+'`。')
    print('\n## 限制\n\n地板效應與 D1 開發遊戲重用均已預先承認。四桶仍混合桶內不同預算；參考原色佔比是畫面代理，不是引擎真實步數。圖節點增加本身也改變探索成本。即使效果為負或衝突減少，也不能直接識別消除所有狀態混淆的純因果成本。')


if __name__ == '__main__':
    main()

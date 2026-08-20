# sample/ — SOL 短切片样例数据包

不下 224GB 全量,也能把 **重放 → 对账 → MBO+ 导出 → 回测** 整条链跑一遍。
全包 5.8 MB,约 32 分钟 SOL 行情。

## 窗口

- 块区间:1105427875 .. 1105454875(共 27,001 块;首块无事件,diffs 从 1105427876 起)
- 时间:2026-08-10 21:30:30 → 22:02:22 UTC(31.9 分钟)
- 种子块 = 1105427874(种子快照自报),真值块 = 1105454875,块号严格相等对账

## 文件

| 文件 | 大小 | 是什么 |
|---|---|---|
| seed_SOL.json | 1.5 MB | 窗口起点整簿快照(10,040 张挂单,oid 级),重放种子 |
| truth_SOL.json.gz | 235 KB | 窗口终点整簿快照,对账真值 |
| diffs_SOL.ndjson.gz | 1.8 MB | 窗口内 149,408 条 oid 级簿增量(引擎重放产物切片) |
| tape_SOL.ndjson.gz | 12 KB | taker 成交序列切片(taker 行为标签 ai 回填用) |
| fills_SOL.ndjson.gz | 20 KB | 成交带切片(trades 表来源) |
| ts2blk.npz | 65 KB | 块号→毫秒时间戳查表(窗口±2000 块) |
| abc_params.json | 100 B | SOL 规格:tick/精度/中位点差 |
| check_replay.py | 3 KB | 步骤 a 的自检脚本 |
| mbo_SOL.npz | 2.1 MB | 步骤 b 产物(hftbacktest L3 格式 + 旁车数组),附作参考 |
| strat_SOL.npz | 52 KB | 步骤 c 产物,附作参考 |

## 三步跑通(在仓库根目录,Python 需 numpy)

a. 重放 + 逐 oid 对账:

    python sample/check_replay.py

b. 导 MBO+(L3 事件数组 + trades/owner/ghosts 旁车):

    HL_FACTS=sample python data/mbo_export.py SOL sample/my_mbo.npz --b0 1105427875 --b1 1105454875 --diffs sample/diffs_SOL.ndjson.gz --ts2blk sample/ts2blk.npz --seed sample/seed_SOL.json --tape sample/tape_SOL.ndjson.gz --fills sample/fills_SOL.ndjson.gz

c. exact 内核跑五个 maker 策略(M1/M3/M4/M5/M6):

    python bench/strat_run.py run SOL sample/my_mbo.npz sample/my_strat.npz --params sample/abc_params.json

## 复现结果(由本包原样跑出)

- a 对账:真值簿 10,083 张 vs 我方簿 10,084 张,只统计窗口内被事件触达的 837 张,oid 一致率 (837−3)/837 = **99.64%**
  (含种子未触达单的总体口径为 99.97%:10,083 张中 10,040 张来自共同的种子快照,窗口内零事件);
  缺失 1 / 多余 2 / 量不一致 0,**±20bp 带内差异 0 笔**;BBO 76.324/76.325 与真值逐位一致,未穿价
- b 导出(my_mbo.npz 与仓内参考 mbo_SOL.npz 逐数组全等):160,294 行事件(ADD 74,498 / CANCEL 74,454 / FILL 845 / MODIFY 456,播种 10,040);
  trades 840 笔,ai 覆盖 笔数 80.8% / 金额 99.1%
- c 回测:17,321 块 0.3 秒;M3(贴价)成交 163 笔 / $5,428 名义额,挂单 473 次

带外 3 笔差异(距中价 41–68bp)逐笔查过:2 笔多余订单来自种子且窗口内零事件
(起止两张第三方快照对深档死单说法不一致,口径同 bench/micro.py 的 seed_inert);
1 笔缺失是块 1105454872 一张 reduce-only 卖单,重放按 remove_ro_flat 移除而真值快照仍保留。

d. (可选,需 `uv pip install -e '.[bench]'` )同一动作流喂 hftbacktest L3,就地复核 README §四 的"判多"主张:

    HL_FACTS=sample python bench/mbo_run.py SOL sample/my_l3.npz --mbo sample/mbo_SOL.npz --src sample/strat_SOL.npz --strats M1,M3 --ts2blk sample/ts2blk.npz --b1 1105454875

实测(本切片,M1+M3 合计成交额):l3 判 $22,826 vs exact 判 $7,519,**判多 3.0×**,方向与九天全量(1.33–8.8×)一致。

## 数据来源

本包全部内容为链上公开信息的再分发:diffs/tape/fills 是引擎对 SQD Portal 免费原始流水的重放产物,
seed/truth 两张快照来自 0xArchive(挂单地址为链上公开数据)。不含任何非公开信息。

## 全量数据怎么来

data/collect_v2.py(实时采集)+ data/seed_fetch.py(整簿快照种子);
切片同格式,只是块区间取满 9.31 天 × 47 币。

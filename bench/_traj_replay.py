#!/usr/bin/env python3
"""轨迹重放共享循环:把 exact(C 臂)的报价轨迹逐条喂给一个 hftbacktest 资产。

被 bench/mbo_run.py(l3 臂)使用。

订单跟踪的语义约定:
· status=0 表示请求在途,是活单不是死单,必须持续轮询到终态;否则在途期间的成交
  会静默丢失(早期版本因此漏记大半成交)。
· 换单 = 撤旧 + 挂新同时在途(延迟后同时生效)≈ exact 的原子替换;撤单在途期
  旧单继续可成交(exact 语义同),所以旧单进"垂死名单"继续轮询到死,成交照记。
· 撤单必须挂旗重试:相邻决策间隔小于延迟时旧单常在途不可撤,只试一次会留下
  永远无人撤的孤儿单。
· 重试时刻必须钉在旧单落位时刻(提交 ts + 延迟),不能只在"下一个同组决策"重试:
  同组决策间隔可达数百块,会把孤儿单寿命人为拉长,形成节拍伪影。实现 = 被拒时
  把落位时刻压进重试堆,主循环处理下个决策前先 elapse 过去处理;残余悬垂 ≤ 延迟
  (hft 在途单不可撤是延迟模型的物理,exact 的原子替换无此约束)。
· 成交按 qty − leaves_qty 的累计口径记增量(兼容部分成交),价格用 exec_price。
"""
import heapq

import numpy as np
from hftbacktest.order import CANCELED, EXPIRED, FILLED, GTC, LIMIT

NTL = 50.0
LAT_NS = 214_200_000
DEC_OFF = LAT_NS - 71_000_000
B1 = 1106599985


def groups_of(strats):
    return [(stg, sd) for stg in strats for sd in 'BA']


GROUPS = groups_of(('M1', 'M3'))
DEAD = (FILLED, CANCELED, EXPIRED)


def build_decisions(src, lookup_ns, qty_at, b1=B1, groups=None):
    """决策 = 挂单(j_ 行,px 有值)∪ 撤而不挂(p_ 行,px=None;M5/M6 的门控撤单)。
    两类都以落位块为锚:决策时刻 = ts(落位块) − 214.2ms + 71ms,与挂单同款延迟。"""
    decisions = []
    for g, (stg, sd) in enumerate(groups or GROUPS):
        for row in src[f'j_{stg}_C_{sd}']:
            jb, px = int(row[0]), float(row[1])
            if jb > b1 - 300:
                continue
            decisions.append((lookup_ns(jb) - DEC_OFF, g, px, qty_at(px), jb))
        pk = f'p_{stg}_C_{sd}'
        if pk in getattr(src, 'files', ()):
            for jb in src[pk]:
                jb = int(jb)
                if jb > b1 - 300:
                    continue
                decisions.append((lookup_ns(jb) - DEC_OFF, g, None, 0.0, jb))
    decisions.sort(key=lambda x: x[0])
    return decisions


def replay(hbt, decisions, first_ts, blk_of_ns, tol, b1_end_ns, groups=None):
    """返回 (fills[len(groups)] 列表 of (fill_blk, px, qty, join_blk, taker), counters)。"""
    groups = groups or GROUPS
    live = [None] * len(groups)          # g -> oid
    track = {}                           # oid -> [g, px, jb, cum_seen, want_cancel, sub_ts]
    st = {'submit': 0, 'pull': 0, 'cancel_req': 0, 'cancel_retry': 0, 'fill_rows': 0,
          'taker_rows': 0, 'dying_fill_rows': 0}
    fills = [[] for _ in range(len(groups))]
    next_id = [1]
    rt = []                              # 重试堆:(旧单落位 ts, oid)

    def poll_all():
        for oid in list(track):
            o = hbt.orders(0).get(oid)
            rec = track[oid]
            g, px, jb, seen = rec[0], rec[1], rec[2], rec[3]
            if o is None:
                del track[oid]
                if live[g] == oid:
                    live[g] = None
                continue
            cum = float(o.qty) - float(o.leaves_qty)      # 累计成交(兼容部分成交)
            if cum > seen + 1e-12:
                taker = 1 if abs(float(o.exec_price) - px) > tol else 0
                fills[g].append((blk_of_ns(int(o.exch_timestamp)),
                                 float(o.exec_price), cum - seen, jb, taker))
                st['fill_rows'] += 1
                st['taker_rows'] += taker
                if live[g] != oid:
                    st['dying_fill_rows'] += 1            # 撤单在途期的成交,照记
                rec[3] = cum
            if o.status in DEAD:
                del track[oid]
                if live[g] == oid:
                    live[g] = None
                continue
            if rec[4] and o.cancellable:                  # 挂旗重试撤单
                hbt.cancel(0, oid, False)
                rec[4] = False
                st['cancel_retry'] += 1

    cur = first_ts

    def do_retries(limit_ts):
        """处理到期的被拒撤单:elapse 到旧单落位时刻立即撤,寿命 ≈ exact + 214.2ms。"""
        nonlocal cur
        while rt and rt[0][0] <= limit_ts:
            rts, roid = heapq.heappop(rt)
            rec = track.get(roid)
            if rec is None or not rec[4]:
                continue
            if rts > cur:
                hbt.elapse(rts - cur)
                cur = rts
            o = hbt.orders(0).get(roid)
            if o is None or o.status in DEAD:
                continue
            if o.cancellable:
                hbt.cancel(0, roid, False)
                rec[4] = False
                st['cancel_retry'] += 1
            else:                                         # 兜底:71.4ms 后再试
                heapq.heappush(rt, (max(rts, cur) + 71_400_000, roid))

    for n, (dec_ts, g, px, qty, jb) in enumerate(decisions, 1):
        do_retries(dec_ts)
        if dec_ts > cur:
            hbt.elapse(dec_ts - cur)
            cur = dec_ts
        poll_all()
        old = live[g]
        if old is not None and old in track:              # 换单/撤单:撤旧(继续跟踪到死)
            o = hbt.orders(0).get(old)
            if o is not None and o.cancellable:
                hbt.cancel(0, old, False)
                st['cancel_req'] += 1
            elif o is not None:
                track[old][4] = True                      # 在途不可撤:挂旗 + 落位时刻重试
                heapq.heappush(rt, (track[old][5] + LAT_NS + 1_000_000, old))
        if px is None:                                    # 撤而不挂(M5/M6 门控)
            live[g] = None
            st['pull'] += 1
            continue
        oid = next_id[0]; next_id[0] += 1
        sub = hbt.submit_buy_order if groups[g][1] == 'B' else hbt.submit_sell_order
        sub(0, oid, px, qty, GTC, LIMIT, False)
        track[oid] = [g, px, jb, 0.0, False, cur]
        live[g] = oid
        st['submit'] += 1
        if n % 20000 == 0:
            poll_all()                                    # 清扫前先记账,防丢
            hbt.clear_inactive_orders(0)
            print(f'  … {n:,}/{len(decisions):,} fills={st["fill_rows"]}'
                  f'(垂死 {st["dying_fill_rows"]})', flush=True)

    do_retries(b1_end_ns)
    if b1_end_ns > cur:
        hbt.elapse(b1_end_ns - cur)
        cur = b1_end_ns
    poll_all()
    return fills, st

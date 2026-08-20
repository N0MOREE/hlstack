#!/usr/bin/env python3
"""MBO 转换器对拍:hftbacktest 吃 mbo_export 的 npz 重建 L3 簿,
与 kernels.book.Book 重放同一 diffs 在 N 个检查点比对 BBO + 前 K 档聚合量。

判据(全过才算过):每个检查点 best bid/ask tick 相同;双边前 K=10 档
(以真值簿的价位为准)聚合量 |Δ| ≤ 1e-9。

用法: mbo_check.py COIN NPZ [--b0 N] [--b1 N] [--ncp 6] [--tick T]
"""
import gzip, json, sys
import os
import functools
print = functools.partial(print, flush=True)
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kernels.book import Book, _apply_plain                 # noqa: E402
from hftbacktest import BacktestAsset, HashMapMarketDepthBacktest  # noqa: E402

R = os.environ.get('HL_REPLAY_ROOT', 'data')   # 数据根目录;seeds/plain/ 放种子快照,p4_out/ 放引擎产出的 diffs


def stream_blocks(path):
    cur, buf = -1, []
    with gzip.open(path, 'rb') as fh:
        for ln in fh:
            i = ln.find(b'"block":') + 8
            blk = int(ln[i:ln.find(b',', i)])
            if blk != cur:
                if buf:
                    yield cur, buf
                cur, buf = blk, []
            buf.append(json.loads(ln))
    if buf:
        yield cur, buf


def main(coin, npz_fp, b0, b1, ncp, tick):
    z = np.load(f'{R}/abc_data/ts2blk.npz')
    TB, TT = z['blk'], z['ts']

    # 检查点:窗口内均匀取,且与下一块的时间缺口 ≥45ms(检查点时刻 = 块时刻+40ms 不越块)
    cand = TB[(TB >= b0 + 1000) & (TB <= b1 - 1000)]
    idx = np.searchsorted(TB, cand)
    ok = (TT[np.minimum(idx + 1, len(TT) - 1)] - TT[idx]) >= 45
    cand = cand[ok]
    cps = sorted(set(int(c) for c in cand[np.linspace(0, len(cand) - 1, ncp).astype(int)]))
    print(f'检查点 {len(cps)} 个: {cps}')

    # ── 真值:kernels.book.Book 重放 ──
    bk = Book(False)
    seedblk, _, _ = bk.seed(f'{R}/seeds/plain/{coin}.json')
    truth = {}
    it = iter(cps); nxt = next(it, None)
    for blk, evs in stream_blocks(f'{R}/p4_out/diffs_{coin}.ndjson.gz'):
        while nxt is not None and blk > nxt:      # 检查点块自己没有事件也要落账
            truth[nxt] = snap_book(bk, tick)
            nxt = next(it, None)
        if nxt is None or blk > b1:
            break
        _apply_plain(bk, evs)
        if blk == nxt:
            truth[nxt] = snap_book(bk, tick)
            nxt = next(it, None)
    while nxt is not None:
        truth[nxt] = snap_book(bk, tick); nxt = next(it, None)

    # ── hft:吃 npz,elapse 到 检查点时刻+40ms ──
    asset = (BacktestAsset()
             .data([npz_fp])
             .linear_asset(1.0)
             .constant_order_latency(214_200_000, 0)
             .l3_fifo_queue_model()
             .no_partial_fill_exchange()
             .trading_value_fee_model(0.00015, 0.0)
             .tick_size(tick)
             .lot_size(1e-9))
    hbt = HashMapMarketDepthBacktest([asset])
    n_bad = 0
    first_ts = int(np.load(npz_fp)['data']['local_ts'][0])
    for cp in cps:
        # 量尺 = 块 ts + 0.5ms:罩住本块全部纳秒槽位(≤~50μs),又不吃进
        # 无成交块被 +1ms 单调兜底挤到本块身后的事件(曾用 +40ms,首尾检查点
        # 把 cp+1..cp+2 的事件吃进来造成假 FAIL,解剖实证 oid 513981280362)
        t_cp = int(TT[np.searchsorted(TB, cp)]) * 1_000_000 + 500_000
        cur = hbt.current_timestamp
        if cur == (1 << 63) - 1:            # 起始哨兵:还没消费任何事件
            cur = first_ts
        hbt.elapse(max(t_cp - cur, 0))
        dep = hbt.depth(0)
        tv = truth[cp]
        bb_h = dep.best_bid_tick * tick if dep.best_bid_tick > 0 else None
        aa_h = dep.best_ask_tick * tick if dep.best_ask_tick < (1 << 60) else None
        ok_bbo = (dep.best_bid_tick == tv['bb_tick'] and dep.best_ask_tick == tv['aa_tick'])
        bad_lv = []
        for sd, pxs in (('B', tv['b_lv']), ('A', tv['a_lv'])):
            for px, q in pxs:
                t_ = round(px / tick)
                qh = dep.bid_qty_at_tick(t_) if sd == 'B' else dep.ask_qty_at_tick(t_)
                if abs(qh - q) > 1e-9:
                    bad_lv.append((sd, px, q, float(qh)))
        flag = 'OK ' if (ok_bbo and not bad_lv) else 'BAD'
        n_bad += flag == 'BAD'
        print(f'{flag} 块 {cp}: 真值 {tv["bb"]}/{tv["aa"]}  hft {bb_h}/{aa_h}'
              + (f'  档量错 {len(bad_lv)} 例: {bad_lv[:3]}' if bad_lv else ''))
    _ = hbt.close()
    print('#CHECK', 'PASS' if n_bad == 0 else f'FAIL({n_bad}/{len(cps)})')
    sys.exit(0 if n_bad == 0 else 1)


def snap_book(bk, tick):
    agg = {'B': {}, 'A': {}}
    for oid, v in bk.orders.items():
        sd, px, sz = v[0], v[1], v[2]
        agg[sd][px] = agg[sd].get(px, 0.0) + sz
    # 0 量档剔除:转换器不给 hft 发 0 量单,真值侧同口径(引擎世界里合法存在)
    b_lv = sorted(((p, q) for p, q in agg['B'].items() if q > 1e-12), key=lambda x: -x[0])[:10]
    a_lv = sorted(((p, q) for p, q in agg['A'].items() if q > 1e-12), key=lambda x: x[0])[:10]
    bb = b_lv[0][0] if b_lv else None
    aa = a_lv[0][0] if a_lv else None
    return {'bb': bb, 'aa': aa,
            'bb_tick': round(bb / tick) if bb else 0,
            'aa_tick': round(aa / tick) if aa else (1 << 62),
            'b_lv': b_lv, 'a_lv': a_lv}


if __name__ == '__main__':
    coin, npz_fp = sys.argv[1], sys.argv[2]
    av = sys.argv[3:]

    def opt(name, d=None):
        return av[av.index(name) + 1] if name in av else d
    prm = json.loads(open(os.environ.get('HL_FACTS', 'sample') + '/abc_params.json').read())[coin]
    main(coin, npz_fp, int(opt('--b0', 1105281018)), int(opt('--b1', 1106599985)),
         int(opt('--ncp', 6)), float(opt('--tick', prm['tick'])))

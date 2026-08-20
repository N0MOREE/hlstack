#!/usr/bin/env python3
"""hft L3 臂(l3 内核):hftbacktest L3FIFO 吃 mbo_export 的 npz,逐条复刻 exact(C 臂)
的报价轨迹,延迟/费用与 ledger_bt 同款。重放循环在 bench/_traj_replay.py(与 l2 臂共用)。

对齐语义:
  报价:不自己决策。C 的每次 join(blk,px) → 决策时刻发 撤旧+挂新,
        决策时刻 = ts(join_blk) − 214.2ms + 71ms(≈ join 块末落位,与 ledger_bt 一致)。
  延迟:entry 214.2ms(=3 块)/ response 0 / feed 0。费率:maker 1.5bp,run 内不结算。
  已知口径差(如实入账,不修):hft L3 只有 NoPartialFill;到达穿价 → 立即 taker(taker 列标记)。

订单跟踪的语义约定(细节见 bench/_traj_replay.py 头注):
  · hftbacktest 中 status=0 表示请求在途,仍是活单,必须持续跟踪到终态;
  · 换单 = 撤旧 + 挂新,撤单在途期旧单仍可成交,成交照记(与 exact 的"等待期旧单继续结算"一致);
  · 被拒的撤单须在旧单落位时刻重试,不能只在下一个决策点重试。

产出:{OUT}.npz,f_{M1,M3}_H_{B,A} = (fill_blk, px, qty, join_blk, taker),
      j_* 原样复制 C 的,mid/meta 复制自源,counters 含跟踪统计。
用法: mbo_run.py COIN OUT.npz [--mbo F] [--src F] [--b1 N] [--strats M1,M3] [--ts2blk F]
  --src 默认 {HL_REPLAY_ROOT}/abc_out/{COIN}_exact.npz(exact 臂产物)
"""
import json
import os
import sys

import numpy as np
from hftbacktest import BacktestAsset, HashMapMarketDepthBacktest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _traj_replay import LAT_NS, NTL, build_decisions, groups_of, replay  # noqa: E402

R = os.environ.get('HL_REPLAY_ROOT', 'data')   # 数据根目录;abc_data/ 放块号-时间映射,abc_out/ 放各臂产物


def main(coin, out_fp, mbo_fp, src_fp, b1, strats=('M1', 'M3'), ts2blk_fp=None):
    prm = json.loads(open(os.environ.get('HL_FACTS', 'sample') + '/abc_params.json').read())[coin]
    tick, sz_dec = prm['tick'], prm['sz_dec']
    z = np.load(ts2blk_fp or f'{R}/abc_data/ts2blk.npz')
    TB, TT = z['blk'], z['ts']

    def lookup_ns(blk):
        i = np.searchsorted(TB, blk, side='right') - 1
        return int(TT[i]) * 1_000_000 + (blk - int(TB[i])) * 1_000_000  # 与 mbo_export 同径

    def blk_of_ns(ts_ns):
        i = np.searchsorted(TT, ts_ns // 1_000_000, side='right') - 1
        return int(TB[max(i, 0)])

    def qty_at(px):
        q = NTL / px
        return round(q, sz_dec) if sz_dec > 0 else max(round(q), 1)

    src = np.load(src_fp)
    groups = groups_of(strats)
    decisions = build_decisions(src, lookup_ns, qty_at, b1, groups)
    print(f'{coin}: {len(decisions):,} 个决策点(策略 {",".join(strats)})', flush=True)

    asset = (BacktestAsset()
             .data([mbo_fp])
             .linear_asset(1.0)
             .constant_order_latency(LAT_NS, 0)
             .l3_fifo_queue_model()
             .no_partial_fill_exchange()
             .trading_value_fee_model(0.00015, 0.0)
             .tick_size(tick)
             # lot 用 1e-9 而非真实步长:hft 的档位聚合按 lot 取整判零即删档,
             # HL 成交可留亚 lot 残量 → 真实 lot 会误删还有活单的档(实测 4 币全 panic)。
             .lot_size(1e-9))
    hbt = HashMapMarketDepthBacktest([asset])
    first_ts = int(np.load(mbo_fp)['data']['local_ts'][0])
    fills, st = replay(hbt, decisions, first_ts, blk_of_ns, 0.5 * tick,
                       lookup_ns(b1) + 1_000_000_000, groups)
    hbt.close()

    out = {}
    for g, (stg, sd) in enumerate(groups):
        a = np.array(fills[g]) if fills[g] else np.zeros((0, 5))
        out[f'f_{stg}_H_{sd}'] = a
        out[f'j_{stg}_H_{sd}'] = src[f'j_{stg}_C_{sd}']
        if f'p_{stg}_C_{sd}' in src.files:
            out[f'p_{stg}_H_{sd}'] = src[f'p_{stg}_C_{sd}']
    if 'strats' in src.files:
        out['strats'] = src['strats']
    np.savez_compressed(out_fp,
                        mid_blk=src['mid_blk'], mid_val=src['mid_val'],
                        counters=json.dumps(st), meta=src['meta'], **out)
    usd = sum(x[1] * x[2] for g in range(len(groups)) for x in fills[g])
    print(f'#DONE {coin} H 臂 fills={st["fill_rows"]}(taker {st["taker_rows"]},'
          f'垂死 {st["dying_fill_rows"]}) ≈{usd:,.0f}$  saved {out_fp}', flush=True)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('用法: mbo_run.py COIN OUT.npz [--mbo F] [--src F] [--b1 N] [--strats M1,M3] [--ts2blk F]')
        sys.exit(1)
    coin, out = sys.argv[1], sys.argv[2]
    av = sys.argv[3:]

    def opt(name, d=None):
        return av[av.index(name) + 1] if name in av else d
    main(coin, out,
         opt('--mbo') or f'{R}/abc_out/mbo_{coin}.npz',
         opt('--src') or f'{R}/abc_out/{coin}_exact.npz',   # exact 臂产物(报价轨迹来源)
         int(opt('--b1', 1106599985)),
         tuple(s.strip() for s in opt('--strats', 'M1,M3').split(',') if s.strip()),
         opt('--ts2blk'))

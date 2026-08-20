#!/usr/bin/env python3
"""exact 内核对拍驱动:ExactKernel(只读 MBO+)复刻 ledger_bt 的 M1/M3 驱动协议,
与真值(ledger_bt B/C-only 跑的 C 臂)逐行逐字段比较 f_/j_/mid 数组。

用法:
  exact_parity.py run COIN MBOPLUS.npz OUT.npz --params abc_params.json
  exact_parity.py cmp  TRUTH.npz EXACT.npz          # 逐行相等性比较,不等则打印首个分歧位置
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kernels.exact import ExactKernel, rnd, NTL, LAT   # noqa: E402


def load_prm(coin, mbo_fp, params_fp=None):
    """币种规格与标定常数:优先 MBO+ meta(单文件自含),缺省回退 params json。"""
    try:
        meta = json.loads(str(np.load(mbo_fp)['meta'][0]))
        p = meta.get('params')
        if p and all(k in p for k in ('tick', 'sz_dec', 'med_spread_bp')):
            return p
    except (OSError, KeyError, ValueError):
        pass
    fp = params_fp or (os.environ.get('HL_FACTS', 'sample') + '/abc_params.json')
    return json.load(open(fp))[coin]


def run(coin, mbo_fp, out_fp, prm, lat=LAT):
    t0 = time.time()
    k = ExactKernel(mbo_fp, prm, lat)
    tick = prm['tick']
    delta = prm['med_spread_bp'] / 1e4
    mid_blk, mid_val = [], []
    nblk = 0
    while (blk := k.advance_block()) is not None:
        nblk += 1
        bb, aa = k.bestb(), k.besta()
        if bb is None or aa is None or aa <= bb:
            continue
        mid = (bb + aa) * 0.5
        mid_blk.append(blk); mid_val.append(mid)
        k.flush_pending(blk)
        for (stg, sd), rs in k.by_grp.items():
            ref = rs[0]
            if stg == 'M3':
                tgt = bb if sd == 'B' else aa
            else:
                tgt = rnd(mid * (1 - delta), tick) if sd == 'B' else rnd(mid * (1 + delta), tick)
            need = (not ref.alive) or ref.px != tgt or ref.filled >= ref.sz - 1e-15
            if need:
                k.place(stg, sd, tgt)
    res = {}
    for r in k.riders:
        key = f'{r.strat}_C_{r.side}'
        res[f'f_{key}'] = np.array(r.fills) if r.fills else np.zeros((0, 4))
        res[f'j_{key}'] = np.array(r.joins) if r.joins else np.zeros((0, 3))
    np.savez_compressed(out_fp,
                        mid_blk=np.array(mid_blk, dtype=np.int64), mid_val=np.array(mid_val),
                        counters=json.dumps(dict(k.st)),
                        meta=np.array([nblk, k.st.get('C_fill_noai', 0), tick, prm['sz_dec'],
                                       prm['med_spread_bp'], NTL, 1.5, lat]), **res)
    dt = time.time() - t0
    print(f'#DONE {coin} {nblk:,} 块  {dt:.1f}s', flush=True)
    print(f"  C 成交来源:{ {kk: v for kk, v in k.st.items() if kk.startswith('C_fill_')} }", flush=True)
    for stg in ('M1', 'M3'):
        n = sum(len(r.fills) for r in k.riders if r.strat == stg)
        usd = sum(x[2] * x[1] for r in k.riders if r.strat == stg for x in r.fills)
        print(f'  {stg}  C:{n:>6d}笔/{usd:>12,.0f}$', flush=True)
    print('saved', out_fp, flush=True)


def cmp(truth_fp, exact_fp):
    zt, ze = np.load(truth_fp), np.load(exact_fp)
    ok = True
    # mid 先比:mid 分歧 = 簿状态漂了,最快定位
    for k in ('mid_blk', 'mid_val'):
        a, b = zt[k], ze[k]
        if len(a) != len(b) or not np.array_equal(a, b):
            ok = False
            n = min(len(a), len(b))
            neq = np.flatnonzero(a[:n] != b[:n])
            i = int(neq[0]) if len(neq) else n
            print(f'DIFF {k}: 长度 {len(a)} vs {len(b)},首个分歧 idx={i}')
            print(f'   真值 {a[i] if i < len(a) else "<尽头>"}  exact {b[i] if i < len(b) else "<尽头>"}')
            if k == 'mid_blk':
                print(f'   现场块号:真值 {int(a[i]) if i < len(a) else "?"}')
            break
    for stg in ('M1', 'M3'):
        for sd in 'BA':
            key = f'{stg}_C_{sd}'
            for pfx, cols in (('f_', '(blk,px,qty,join_blk)'), ('j_', '(blk,px,seen)')):
                a, b = zt[pfx + key], ze[pfx + key]
                if a.shape == b.shape and np.array_equal(a, b):
                    print(f'OK   {pfx}{key}  {a.shape[0]} 行相等')
                    continue
                ok = False
                n = min(len(a), len(b))
                i = n
                for j in range(n):
                    if not np.array_equal(a[j], b[j]):
                        i = j
                        break
                print(f'DIFF {pfx}{key}: 形状 {a.shape} vs {b.shape},首个分歧行 idx={i} 列{cols}')
                if i < len(a):
                    print(f'   真值  {a[i]}')
                if i < len(b):
                    print(f'   exact {b[i]}')
    print('#PARITY', 'PASS' if ok else 'FAIL')
    return ok


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'run':
        coin, mbo_fp, out_fp = sys.argv[2], sys.argv[3], sys.argv[4]
        av = sys.argv[5:]

        def opt(name, d=None):
            return av[av.index(name) + 1] if name in av else d
        prm = load_prm(coin, mbo_fp, opt('--params'))
        run(coin, mbo_fp, out_fp, prm, int(opt('--lat', LAT)))
    elif cmd == 'cmp':
        sys.exit(0 if cmp(sys.argv[2], sys.argv[3]) else 1)
    else:
        raise SystemExit(f'未知命令 {cmd}')

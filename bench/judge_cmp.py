#!/usr/bin/env python3
"""exact 与 l3 两臂的 PnL / 零证据成交 / markout 结算器(逐币 × 5 策略)。
口径:
  PnL = cash 累计 + 终点持仓 × 终点 mid;maker 费 1.5bp,taker(l3 五列 taker=1)4.5bp。
  证据 = (join_blk, fill_blk] 窗口内"该价位或穿过该价位"的 tape 成交量
        (买单:成交价 ≤ 挂价;卖单:成交价 ≥ 挂价)——与 exact 的跨档结算同口径。
  零证据成交 = 证据量为 0 的成交行。exact 按构造应恰为 0(自检门);l3 的占比即反事实规模。
  markout10s = 成交后 140 块的 mid 相对成交价的偏移(bp,USD 加权,负=价格正在对你不利)。
用法: judge_cmp.py COIN [COIN...] [--ov] [--exact F] [--mbo F] [--l3 F]
  --ov       用 26h 覆盖窗的产物(默认全段)
  --exact F  exact 臂产物 npz(默认 {HL_REPLAY_ROOT}/abc_out/{COIN}_x5full.npz,--ov 时 _x5ov.npz)
  --mbo F    mbo_export 产物 npz,取其中 trades 做证据(默认 abc_out/mbo_{COIN}[_full].npz)
  --l3 F     l3 臂产物 npz 模板,含 {stg} 占位(默认 abc_out/{COIN}[_ov]_l3_{stg}.npz)
  输出:打印 json,写 {HL_FACTS}/x5_{full|ov}_{COIN}.json
"""
import json
import os
import sys

import numpy as np

A = os.environ.get('HL_REPLAY_ROOT', 'data') + '/abc_out'   # abc_out/: 各臂回测产物目录
FACTS = os.environ.get('HL_FACTS', 'sample')                 # 结算结果 json 输出目录
STRATS = ['M1', 'M3', 'M4', 'M5', 'M6']
FEE_MK, FEE_TK = 1.5e-4, 4.5e-4
MO_BLK = 140                     # 10 秒 markout
TOL = 1e-9


def make_ev(tr):
    b_all = np.ascontiguousarray(tr['blk'])
    px_all = np.ascontiguousarray(tr['px'])
    sz_all = np.ascontiguousarray(tr['sz'])
    assert (np.diff(b_all) >= 0).all(), 'trades 未按块排序'

    def ev_of(px, jb, fb, sgn):
        lo = np.searchsorted(b_all, jb, 'right')
        hi = np.searchsorted(b_all, fb, 'right')
        if lo >= hi:
            return 0.0
        p = px_all[lo:hi]
        m = (p <= px + TOL) if sgn > 0 else (p >= px - TOL)
        return float(sz_all[lo:hi][m].sum())
    return ev_of


def cell(f_rows, mid_blk, mid_val, ev_of):
    """f_rows: [(rows, sgn, has_taker)] → 该格统计。"""
    ev = []
    n = usd = tk_n = tk_usd = 0.0
    zn = zusd = 0.0
    mo_num = mo_den = 0.0
    for rows, sgn, has_tk in f_rows:
        for r in rows:
            blk, px, qty, jb = int(r[0]), float(r[1]), float(r[2]), int(r[3])
            taker = int(r[4]) if has_tk and len(r) >= 5 else 0
            ntl = px * qty
            fee = (FEE_TK if taker else FEE_MK) * ntl
            ev.append((blk, sgn * qty, -sgn * ntl - fee))
            n += 1; usd += ntl
            if taker:
                tk_n += 1; tk_usd += ntl
            elif ev_of(px, jb, blk, sgn) <= 1e-12:
                zn += 1; zusd += ntl
            i = np.searchsorted(mid_blk, blk + MO_BLK, 'right') - 1
            if 0 <= i < len(mid_val):
                mo_num += sgn * (mid_val[i] - px) / px * 1e4 * ntl
                mo_den += ntl
    if not ev:
        return dict(n=0, usd=0, pnl=0.0, endpos_usd=0, taker_n=0, taker_usd=0,
                    zero_n=0, zero_usd=0, mo_bp=None)
    ev.sort(key=lambda e: e[0])
    pos = np.cumsum([e[1] for e in ev])
    cash = np.cumsum([e[2] for e in ev])
    endpos = float(pos[-1]); endmid = float(mid_val[-1])
    return dict(n=int(n), usd=round(usd), pnl=round(float(cash[-1]) + endpos * endmid, 1),
                endpos_usd=round(endpos * endmid), taker_n=int(tk_n), taker_usd=round(tk_usd),
                zero_n=int(zn), zero_usd=round(zusd),
                mo_bp=round(mo_num / mo_den, 3) if mo_den else None)


def main(coin, ov=False, exact_fp=None, mbo_fp=None, l3_fp=None):
    # 默认文件名沿用产物目录里的既有命名(x5full/x5ov = exact 臂全段/覆盖窗产物)。
    if ov:
        zx = np.load(exact_fp or f'{A}/{coin}_x5ov.npz')
        zm = np.load(mbo_fp or f'{A}/mbo_{coin}.npz')
        l3fp = l3_fp or f'{A}/{coin}_ov_l3_{{stg}}.npz'
        tag = 'ov'
    else:
        zx = np.load(exact_fp or f'{A}/{coin}_x5full.npz')
        zm = np.load(mbo_fp or f'{A}/mbo_{coin}_full.npz')
        l3fp = l3_fp or f'{A}/{coin}_l3_{{stg}}.npz'
        tag = 'full'
    mid_blk, mid_val = zx['mid_blk'], zx['mid_val']
    ev_of = make_ev(zm['trades'])

    out = {}
    for stg in STRATS:
        fx = [(zx[f'f_{stg}_C_B'], 1.0, False), (zx[f'f_{stg}_C_A'], -1.0, False)]
        out[f'{stg}|exact'] = cell(fx, mid_blk, mid_val, ev_of)
        try:
            zl = np.load(l3fp.format(stg=stg))
        except FileNotFoundError:
            out[f'{stg}|l3'] = None
            continue
        fl = [(zl[f'f_{stg}_H_B'], 1.0, True), (zl[f'f_{stg}_H_A'], -1.0, True)]
        out[f'{stg}|l3'] = cell(fl, mid_blk, mid_val, ev_of)
        e, l = out[f'{stg}|exact'], out[f'{stg}|l3']
        if e['usd']:
            l['usd_ratio'] = round(l['usd'] / e['usd'], 3)
            l['maker_usd_ratio'] = round((l['usd'] - l['taker_usd']) / e['usd'], 3)
    fp = f'{FACTS}/x5_{tag}_{coin}.json'
    json.dump(out, open(fp, 'w'), ensure_ascii=False, indent=1)
    print(coin, json.dumps(out, ensure_ascii=False, indent=1))
    print('saved', fp)


if __name__ == '__main__':
    av = sys.argv[1:]

    def opt(name, d=None):
        return av[av.index(name) + 1] if name in av else d
    ov = '--ov' in av
    taken = {opt(n) for n in ('--exact', '--mbo', '--l3') if opt(n) is not None}
    for c in [a for a in av if not a.startswith('--') and a not in taken]:
        main(c, ov, opt('--exact'), opt('--mbo'), opt('--l3'))

#!/usr/bin/env python3
"""五策略 exact 驱动:ExactKernel 上同窗跑 M1/M3/M4/M5/M6,产出 l3 同轨迹重放所需的
j_(挂单落位)/ p_(撤而不挂落位)/ f_(成交)/ mid 数组。

策略定义(v1;全部 $50/张、永不穿价、lat=3 块原子替换):
  M1 退半步   bid=mid(1−δ) ask=mid(1+δ),δ=该币全窗中位点差(bp,abc_params;母本 ledger_bt 同款,
              每侧各退一个中位点差)。母本基线。
  M3 贴价     bid=最优买 ask=最优卖。母本基线。
  M4 GLFT-lite(hftbacktest 官方教程公式,tick 单位):
       λ(δ)=A·e^(−kδ) 滚动 30 分钟成交距离拟合(log-线性,δ=0..9 tick 格点);
       σ=滚动 10 分钟块中价差分 std(tick/√块)×√14 → tick/√秒;
       c1=(1/ξ)ln(1+ξ/k)  c2=√[γ/(2Ak)·(1+ξ/k)^(k/ξ+1)]  (Δ=1,ξ=γ=0.05)
       half=c1+σ·c2/2  skew=σ·c2·q  (q=持仓/单张,截断±20)
       bid=rnd(mid−(skew+half)·tick) 钳到 ≤bb;ask 对称钳到 ≥aa;half 钳 [0.5tick, 20bp]。
       预热(mid<8400 样本 或 窗内成交<300 或 k 无效)不挂单。估计只用过去,无前视。
  M5 双门贴价  M3 + 两道门:点差门 spread_bp≥1.25×该币中位点差(1.25 使 tick 顶死的簿
              落在 1 与 2 tick 之间,门语义="比常态宽");
       方向失衡门 imb=(Q_bb−Q_aa)/(Q_bb+Q_aa),imb>+0.6 撤卖单、imb<−0.6 撤买单
       (顶价位队列一边倒 = 逆向选择高危侧)。门关 = 撤而不挂。
  M6 库存偏移贴价  M3 + 持仓反馈:|仓|>$300 撤增仓侧;|仓|>$150 增仓侧退 1 tick。
       直接治 M1/M3 的裸库存病(无库存反馈时单边持仓会在长窗口上失控)。

动作纪律:内核 place 对"同组已有待落单"静默忽略——驱动先查 pending 再下手,
只记录被接受的动作;j_ 来自内核骑手(真实落位),p_ 由驱动记(落位块=请求块+lat)。
l3 桥(bench/_traj_replay.build_decisions)按同款延迟重放 j_∪p_。

用法: strat_run.py run COIN MBOPLUS.npz OUT.npz [--params F] [--lat 3]
"""
import collections
import json
import math
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


STRATS = ('M1', 'M3', 'M4', 'M5', 'M6')

# M4 GLFT-lite 冻结参数
GAMMA = XI = 0.05
SIG_WIN = 8400          # 10 分钟(块)
TRD_WIN = 25200         # 30 分钟(块)
REFIT_EVERY = 70        # ~5 秒
NBKT = 20               # 成交距离桶(tick),拟合用 0..9
MIN_TRADES = 300
QCAP = 20.0             # |q| 截断(单张数)
HALF_CAP_BP = 20.0      # half 上限(bp)

# M5 冻结参数
IMB_GATE = 0.6
SP_GATE_MULT = 1.25     # 点差门 = 1.25×全窗中位:tick 顶死的簿(BTC 1tick=中位)上
                        # 恰好落在 1 与 2 tick 之间,门语义="比常态宽",无价位伪影

# M6 冻结参数(USD)
POS_CAP = 300.0
POS_SKEW = 150.0


def run(coin, mbo_fp, out_fp, prm, lat=LAT):
    t0 = time.time()
    k = ExactKernel(mbo_fp, prm, lat, strats=STRATS)
    tick = prm['tick']
    delta = prm['med_spread_bp'] / 1e4
    med_sp_bp = prm['med_spread_bp']

    mid_blk, mid_val = [], []
    st = collections.Counter()

    grp = {key: rs[0] for key, rs in k.by_grp.items()}

    def want(stg, sd, tgt):
        ref = grp[(stg, sd)]
        pend = any(p[1] == stg and p[2] == sd for p in k.pending)
        if tgt is None:
            if ref.alive and not pend:
                k.place(stg, sd, None)     # 生效块由内核 k.pulls 记(与 j_ 同口径)
            return
        if pend:
            return
        if (not ref.alive) or ref.px != tgt or ref.filled >= ref.sz - 1e-15:
            k.place(stg, sd, tgt)

    # ── M4 滚动估计器状态 ──
    mids = []                              # 有效块 mid(tick 单位)序列
    trd = k.trades
    trd_blk = trd['blk']; trd_px = trd['px']
    nt = len(trd)
    ti = 0
    ring = collections.deque()             # (blk, bucket)
    cnts = np.zeros(NBKT, dtype=np.int64)
    m4 = {'half': None, 'skewc': 0.0, 'warm': False}
    since_fit = 0

    def refit(blk, mid_t):
        nonlocal since_fit
        since_fit = 0
        if len(mids) > 3 * SIG_WIN:
            del mids[:-(SIG_WIN + 1)]
        if len(mids) < SIG_WIN + 1 or len(ring) < MIN_TRADES:
            m4['warm'] = False
            st['m4_cold'] += 1
            return
        a = np.asarray(mids[-(SIG_WIN + 1):])
        sig = float(np.std(np.diff(a))) * math.sqrt(14.0)      # tick/√秒
        n_ge = cnts[::-1].cumsum()[::-1]
        win_sec = min(TRD_WIN, blk - ring[0][0] + 1) * 0.0714
        js = [j for j in range(10) if n_ge[j] > 0]
        if len(js) < 4 or sig <= 0:
            m4['warm'] = False
            st['m4_fit_skip'] += 1
            return
        x = np.array(js, dtype=float)
        y = np.log(n_ge[js] / win_sec)
        sl, ic = np.polyfit(x, y, 1)
        k_hat, a_hat = -sl, math.exp(ic)
        if k_hat <= 1e-9 or a_hat <= 0:
            m4['warm'] = False
            st['m4_fit_bad'] += 1
            return
        c1 = math.log(1 + XI / k_hat) / XI
        try:
            c2 = math.sqrt(GAMMA / (2 * a_hat * k_hat)
                           * (1 + XI / k_hat) ** (k_hat / XI + 1))
        except OverflowError:
            m4['warm'] = False
            st['m4_fit_bad'] += 1
            return
        half = c1 + 0.5 * sig * c2
        half = min(max(half, 0.5), mid_t * HALF_CAP_BP / 1e4)  # [0.5tick, 20bp]
        m4['half'], m4['skewc'], m4['warm'] = half, sig * c2, True
        st['m4_refit'] += 1

    nblk = 0
    while (blk := k.advance_block()) is not None:
        nblk += 1
        bb, aa = k.bestb(), k.besta()
        if bb is None or aa is None or aa <= bb:
            continue
        mid = (bb + aa) * 0.5
        mid_blk.append(blk); mid_val.append(mid)
        k.flush_pending(blk)

        mid_t = mid / tick
        mids.append(mid_t)
        while ti < nt and trd_blk[ti] <= blk:
            b = int(abs(float(trd_px[ti]) - mid) / tick)
            b = b if b < NBKT else NBKT - 1
            ring.append((blk, b)); cnts[b] += 1
            ti += 1
        lo = blk - TRD_WIN
        while ring and ring[0][0] < lo:
            cnts[ring.popleft()[1]] -= 1
        since_fit += 1
        if since_fit >= REFIT_EVERY:
            refit(blk, mid_t)

        # M1 / M3(exact_parity 母本逐字)
        want('M3', 'B', bb); want('M3', 'A', aa)
        want('M1', 'B', rnd(mid * (1 - delta), tick))
        want('M1', 'A', rnd(mid * (1 + delta), tick))

        # M4 GLFT-lite
        if m4['warm']:
            unit = k.qty_at(mid)
            q = (grp[('M4', 'B')].pos + grp[('M4', 'A')].pos) / unit
            q = max(-QCAP, min(QCAP, q))
            center = mid_t - m4['skewc'] * q
            tb = rnd(min((center - m4['half']) * tick, bb), tick)
            ta = rnd(max((center + m4['half']) * tick, aa), tick)
            want('M4', 'B', tb if tb > 0 else None)
            want('M4', 'A', ta)
        else:
            want('M4', 'B', None); want('M4', 'A', None)

        # M5 双门贴价
        sp_bp = (aa - bb) / mid * 1e4
        if sp_bp >= SP_GATE_MULT * med_sp_bp:
            db = k.lv.get(('B', bb)); da = k.lv.get(('A', aa))
            qb = sum(db.values()) if db else 0.0
            qa = sum(da.values()) if da else 0.0
            imb = (qb - qa) / (qb + qa) if qb + qa > 0 else 0.0
            want('M5', 'B', bb if imb > -IMB_GATE else None)
            want('M5', 'A', aa if imb < IMB_GATE else None)
        else:
            want('M5', 'B', None); want('M5', 'A', None)

        # M6 库存偏移贴价
        pos_usd = (grp[('M6', 'B')].pos + grp[('M6', 'A')].pos) * mid
        want('M6', 'B', None if pos_usd > POS_CAP else
             rnd(bb - (tick if pos_usd > POS_SKEW else 0.0), tick))
        want('M6', 'A', None if pos_usd < -POS_CAP else
             rnd(aa + (tick if pos_usd < -POS_SKEW else 0.0), tick))

        if nblk % 2_000_000 == 0:
            print(f'  … {nblk:,} 块 {time.time()-t0:.0f}s', flush=True)

    res = {}
    for r in k.riders:
        key = f'{r.strat}_C_{r.side}'
        res[f'f_{key}'] = np.array(r.fills) if r.fills else np.zeros((0, 4))
        res[f'j_{key}'] = np.array(r.joins) if r.joins else np.zeros((0, 3))
    pulls = {(s, sd): k.pulls.get((s, sd), []) for s in STRATS for sd in 'BA'}
    for (s, sd), v in pulls.items():
        res[f'p_{s}_C_{sd}'] = np.array(v, dtype=np.int64)
    np.savez_compressed(out_fp,
                        mid_blk=np.array(mid_blk, dtype=np.int64), mid_val=np.array(mid_val),
                        counters=json.dumps({**dict(k.st), **dict(st)}),
                        strats=json.dumps(list(STRATS)),
                        meta=np.array([nblk, k.st.get('C_fill_noai', 0), tick, prm['sz_dec'],
                                       prm['med_spread_bp'], NTL, 1.5, lat]), **res)
    print(f'#DONE {coin} {nblk:,} 块  {time.time()-t0:.1f}s', flush=True)
    for stg in STRATS:
        n = sum(len(r.fills) for r in k.riders if r.strat == stg)
        usd = sum(x[2] * x[1] for r in k.riders if r.strat == stg for x in r.fills)
        npull = sum(len(pulls[(stg, sd)]) for sd in 'BA')
        nj = sum(len(r.joins) for r in k.riders if r.strat == stg)
        print(f'  {stg}  成交 {n:>7,} 笔/{usd:>13,.0f}$  挂 {nj:>9,} 撤 {npull:>7,}', flush=True)
    print('saved', out_fp, flush=True)


if __name__ == '__main__':
    cmd = sys.argv[1]
    assert cmd == 'run', f'未知命令 {cmd}'
    coin, mbo_fp, out_fp = sys.argv[2], sys.argv[3], sys.argv[4]
    av = sys.argv[5:]

    def opt(name, d=None):
        return av[av.index(name) + 1] if name in av else d
    prm = load_prm(coin, mbo_fp, opt('--params'))
    run(coin, mbo_fp, out_fp, prm, int(opt('--lat', LAT)))

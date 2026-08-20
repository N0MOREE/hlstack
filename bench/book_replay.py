#!/usr/bin/env python3
"""多队列模型对照回测器:Q0/Q1/Q1n/Q2/Q2r 五臂共用一遍数据,多骑手同步挂撤。

  book_replay.py probe COIN                       扫一段推 tick / szDecimals / 中位价差
  book_replay.py run   COIN OUT.npz [选项]        跑全部骑手

选项:
  --b0 N --b1 N     块区间(缺省=全段)
  --seed FILE       播种快照(kernels/book.py 里的 orders[] 带 q 格式;不播种簿是空的,best 全错)
  --drop-negoid     oid>=2^63(引擎合成号)一律丢弃(丢弃模式);缺省是认领模式
  --strats M1,M2,M3,T1,T2
  --params JSON     probe 产出的参数(不给就现场 probe 前 20 万块)

臂(仅 maker):
  Q0   价格触及即成交            —— 绝大多数回测框架的默认,参考臂,不进判据
  Q1   hftbacktest ProbQueueModel(PowerProbQueueFunc3),成交流按笔确定性扣减 + 撤单概率扣减
       = 诚实的零参数 L2 基线(HL 公开的 l2Book + 公开 trades)
  Q1n  朴素 L2(无成交流):深度下降一律当成交 —— 常见的错误实现,留作对照
  Q2   真队列(oid 级 FIFO)
  Q2r  正对照:一切同 Q2,但我的单在同价位队列里随机插位(不是真实队尾)
       PnL(Q2) 必须显著优于 PnL(Q2r),否则说明实现没用上队列信息 -> 实验作废
"""
import gzip, json, sys, heapq, collections, os, random, math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kernels.book import Book, _apply_plain, probe, NEG   # noqa: E402


def prob_back(front, back):
    """内联自已退役的 kernels/l2(hftbacktest PowerProbQueueFunc3 口径):
    prob = 1 − (front/(front+back))³ ——不知道逐单时,按公式估撤单落在你身后的概率。"""
    d = front + back
    if d <= 0:
        return 1.0
    return 1.0 - (front / d) ** 3


ARMS = ('Q0', 'Q1', 'Q1n', 'Q2', 'Q2r')
MAKERS = ('M1', 'M2', 'M3')
TAKERS = ('T1', 'T2')
NTL = 50.0            # 每张挂单固定 $50 名义额
FEE_MK = 1.5          # bp,HL 基础档
FEE_TK = 4.5
HOR = 140             # markout 视界(块) ~10 秒
QMAX = 10             # 库存上限 = 10 张单;M2 在 q=QMAX 时偏移量 = delta


class MakerRider:
    """一个 (策略, 臂, 方向) 的挂单状态机"""
    __slots__ = ('strat', 'arm', 'side', 'alive', 'px', 'sz', 'filled',
                 'front', 'ahead', 'fills', 'pos', 'nskip')

    def __init__(self, strat, arm, side):
        self.strat, self.arm, self.side = strat, arm, side
        self.alive = False; self.px = None; self.sz = 0.0; self.filled = 0.0
        self.front = 0.0; self.ahead = None
        self.fills = []          # (blk, px, qty)
        self.pos = 0.0           # 已成交净量(张数口径,用于 M2 的库存偏移)
        self.nskip = 0

    def join(self, bk, px, sz, rng):
        """挂到 px。ahead = 我前面那些单的 oid 集合(真队列);Q2r 随机插位。
        front(= 我前面还剩多少量)之后靠增量维护,不再每块重算 —— 重算是 O(队列长度),
        实测在热点档位上就是整个回测的瓶颈。"""
        self.alive = True; self.px = px; self.sz = sz; self.filled = 0.0
        d = bk.lv.get((self.side, px))
        if not d:
            self.ahead = frozenset(); self.front = 0.0; return
        if self.arm == 'Q2r':
            lvl = list(d.keys())
            k = rng.randint(0, len(lvl))       # 等概率插在 0..n 之间
            self.ahead = frozenset(lvl[:k])
            self.front = sum(d[x] for x in self.ahead)
        else:
            self.ahead = frozenset(d.keys())    # 真队尾
            self.front = sum(d.values())        # join 是低频操作,这里用精确值


AUDIT_FRONT = os.environ.get('AUDIT_FRONT') == '1'   # 调试钩子:每块把 front 增量值与重算值对比
_fdiff = []


def run(coin, files, b0, b1, out, seed_fp, drop_neg, strats, prm):
    bk = Book(drop_neg)
    rng = random.Random(20260815)
    seedblk = nseed = ntot = 0
    if seed_fp:
        seedblk, nseed, ntot = bk.seed(seed_fp)
        # 起点必须紧接种子块。各币种子块不同,写死一个 b0 会漏块:漏掉种子块后
        # 第一块里的 remove_gone 事件,会留下幽灵单,穿价率随之飙升。
        if b0 != seedblk + 1:
            print(f'#SEED 起点从 {b0} 修正为 {seedblk + 1}(种子块+1)', flush=True)
            b0 = seedblk + 1
        print(f'#SEED blk={seedblk} 播入 {nseed}/{ntot} 张'
              f'{"(已丢弃 oid>=2^63)" if drop_neg else "(保留全部)"}', flush=True)

    tick = prm['tick']; sz_dec = prm['sz_dec']
    half = prm['med_spread_bp'] / 2 / 1e4          # M1/M2 的半价差(相对),挂在 mid*(1±delta)
    delta = prm['med_spread_bp'] / 1e4             # 挂在 mid ± 一个中位价差(比 BBO 深一档)

    mk = [MakerRider(s, a, sd) for s in strats if s in MAKERS for a in ARMS for sd in 'BA']
    by_key = collections.defaultdict(list)
    for r in mk: by_key[(r.strat, r.side)].append(r)

    # taker 状态
    tk = {t: dict(pos=0.0, entry=0.0, entry_blk=0, fills=[]) for t in strats if t in TAKERS}
    ofi = 0.0; ofi_hist = collections.deque(maxlen=2000)

    nrejoin = collections.Counter()
    mid_blk = []; mid_val = []; lvl_s = []
    my_share = []                                  # 我的量 / 该档原有总量
    nblk = 0; nev = 0; ncross = 0
    cur = -1; buf = []
    prev_bq = prev_aq = None

    def qty_at(px):
        q = NTL / px
        return round(q, sz_dec) if sz_dec > 0 else max(round(q), 1)

    def rnd_px(p):
        return round(round(p / tick) * tick, 10)

    def apply_block(blk, evs):
        nonlocal nblk, ncross, ofi, prev_bq, prev_aq
        F = collections.defaultdict(float); C = collections.defaultdict(float)
        DQ = collections.defaultdict(list)          # (sd,px) -> [(oid, dq)]
        # 同一块内同一张单可能有多条 fill_partial(实测 ACE oid 512056444968 一块 5 条)。
        # orders 里的 sz 要到块末 _apply_plain 才更新,所以必须在块内自己记「上一条之后剩多少」,
        # 否则每条都拿原始量去减 → 减量被重复累加。
        cur_sz = {}
        for d in evs:
            t = d['type']; o = d['oid']
            if drop_neg and o >= NEG: continue
            if t == 'new': continue
            e = bk.orders.get(o)
            if e is None:
                r = bk._resolve(d)
                if r is None: continue
                o = r; d = dict(d); d['oid'] = o; e = bk.orders.get(o)
            sd, px = e[0], e[1]
            old = cur_sz.get(o, e[2])               # 本块内的最新剩余量
            dq = old - d['sz'] if t == 'update' else old
            cur_sz[o] = d['sz'] if t == 'update' else 0.0
            # DQ 要含 dq<0(modify 把量改大)—— 漏了它,front 只减不增,会偏小,Q2 就多成交
            DQ[(sd, px)].append((o, dq))
            if dq <= 0: continue
            w = d.get('why', '')
            if ('fill' in w) or ('swept' in w): F[(sd, px)] += dq
            else: C[(sd, px)] += dq

        # ---- maker 骑手 ----
        for r in mk:
            if not r.alive: continue
            key = (r.side, r.px)
            f = F.get(key, 0.0); c = C.get(key, 0.0)
            if f <= 0 and c <= 0: continue
            rem = r.sz - r.filled
            if rem <= 1e-15: continue
            a = r.arm
            if a == 'Q0':
                if f > 0:
                    r.filled += rem; r.fills.append((blk, r.px, rem))
                    r.pos += rem if r.side == 'B' else -rem
            elif a in ('Q2', 'Q2r'):
                front = r.front                      # 块初:我前面还剩多少(增量维护)
                if AUDIT_FRONT and r.ahead:
                    lvl = bk.lv.get(key, {})
                    ex = sum(lvl.get(x, 0.0) for x in r.ahead)
                    if abs(ex - front) > 1e-6 * max(1.0, ex) and len(_fdiff) < 8:
                        moved = [(x, lvl.get(x, 0.0)) for x in r.ahead if lvl.get(x, 0.0) > 0]
                        _fdiff.append((blk, r.strat, a, r.side, r.px, front, ex,
                                       len(r.ahead), len(moved),
                                       [(o2, dq2) for o2, dq2 in DQ.get(key, ())][:4]))
                if f > front:
                    g = min(rem, f - front); r.filled += g
                    r.fills.append((blk, r.px, g))
                    r.pos += g if r.side == 'B' else -g
                ah = r.ahead
                if ah:
                    red = 0.0
                    for oo, dq in DQ.get(key, ()):
                        if oo in ah: red += dq
                    if red:
                        nf = front - red
                        r.front = nf if nf > 0.0 else 0.0
            elif a == 'Q1':
                r.front -= f                                   # 成交:按笔确定性扣减
                chg = max(c, 0.0)
                back = max(bk.ltot(key) - max(r.front, 0.0), 0.0)
                pb = prob_back(max(r.front, 0.0), back)         # 撤单:概率扣减
                r.front -= (1.0 - pb) * chg
                if f > 0 and r.front < 0:
                    g = min(rem, -r.front); r.filled += g; r.front += g
                    r.fills.append((blk, r.px, g))
                    r.pos += g if r.side == 'B' else -g
            else:                                              # Q1n 朴素
                r.front -= (f + c)
                if r.front < 0:
                    g = min(rem, -r.front); r.filled += g; r.front += g
                    r.fills.append((blk, r.px, g))
                    r.pos += g if r.side == 'B' else -g

        _apply_plain(bk, evs)

        bb = bk.bestb(); aa = bk.besta()
        nblk += 1
        if bb is not None and aa is not None and aa <= bb: ncross += 1
        if bb is None or aa is None or aa <= bb: return
        mid = (bb + aa) * 0.5
        mid_blk.append(blk); mid_val.append(mid)
        bq = bk.ltot(('B', bb)); aq = bk.ltot(('A', aa))
        if len(lvl_s) < 500000:
            lvl_s.append((bq * bb, aq * aa, (aa - bb) / mid * 1e4))

        # ---- OFI(Cont-Kukanov-Stoikov) ----
        if prev_bq is not None:
            ofi += (bq - prev_bq) - (aq - prev_aq)
            ofi_hist.append(ofi)
        prev_bq, prev_aq = bq, aq

        # ---- taker ----
        for t, s in tk.items():
            if s['pos'] != 0.0 and blk - s['entry_blk'] >= HOR:
                s['fills'].append((s['entry_blk'], s['entry'], abs(s['pos']),
                                   1 if s['pos'] > 0 else -1, mid))
                s['pos'] = 0.0
            if s['pos'] != 0.0: continue
            sig = 0
            if t == 'T1' and len(ofi_hist) > 500:
                arr = np.fromiter(ofi_hist, float)
                z = (ofi - arr.mean()) / (arr.std() + 1e-12)
                sig = 1 if z > 2 else (-1 if z < -2 else 0)
            elif t == 'T2':
                imb = (bq - aq) / (bq + aq + 1e-12)
                sig = -1 if imb > 0.6 else (1 if imb < -0.6 else 0)   # 反转
            if sig:
                s['pos'] = sig * qty_at(aa if sig > 0 else bb)
                s['entry'] = aa if sig > 0 else bb
                s['entry_blk'] = blk

        # 各策略的净库存(Q2 臂,跨两侧合并) —— M2 用它做偏移
        inv_of = {st: sum(x.pos for x in mk if x.strat == st and x.arm == 'Q2')
                  for st in MAKERS}
        # ---- maker 目标价 + 重挂 ----
        # 关键:所有臂的挂/撤动作必须同步,
        # 否则「成交多的臂重挂更频繁」会把轨迹差异混进 ΔPnL,那就不是纯成交判定差了。
        # 做法:决策一律以 Q2(真队列)那条臂为准 —— 策略看到的是真实世界,
        # 各臂只在「这一单成没成交」上分开。
        for (strat, side), rs in by_key.items():
            ref = next((x for x in rs if x.arm == 'Q2'), rs[0])
            if strat == 'M3':
                tgt = bb if side == 'B' else aa
            elif strat == 'M2':
                # 库存偏移:用 Q2 臂的仓位(真实世界),归一到 QMAX 张,满仓时偏移 = delta
                # 库存必须是「买到的 − 卖掉的」净额,跨两侧合并。
                # (注意:按 side 分开算 → 买侧 pos 单调增、卖侧单调减,两边一起往外偏,成交锐减)
                inv = inv_of['M2']
                qn = max(-1.0, min(1.0, inv / (QMAX * NTL / mid)))
                c0 = mid * (1 - qn * delta)
                tgt = rnd_px(c0 * (1 - delta)) if side == 'B' else rnd_px(c0 * (1 + delta))
            else:                                        # M1 对称
                tgt = rnd_px(mid * (1 - delta)) if side == 'B' else rnd_px(mid * (1 + delta))
            # 重挂与否由 Q2 臂决定,决定了就全组一起动
            need = (not ref.alive) or ref.px != tgt or ref.filled >= ref.sz - 1e-15
            if not need: continue
            nrejoin[(strat, side)] += 1
            orig = bk.ltot((side, tgt))
            q = qty_at(tgt)
            if q <= 0:
                for r in rs: r.nskip += 1
                continue
            for r in rs:
                r.join(bk, tgt, q, rng)
            # orig 可能是浮点残渣(1e-12 量级),那样比值会炸到 1e13。
            # 只在这一档确实有像样的量($1 以上)时才记。
            if orig * tgt > 1.0 and len(my_share) < 200000:
                my_share.append(q / orig)

    for fp in files:
        stop = False
        for ln in gzip.open(fp, 'rt'):
            i = ln.find('"block":') + 8; j = ln.find(',', i); blk = int(ln[i:j])
            if blk < b0: continue
            if blk > b1: stop = True; break
            if blk != cur:
                if buf: apply_block(cur, buf)
                buf = []; cur = blk
            buf.append(json.loads(ln)); nev += 1
        if stop: break
    if buf: apply_block(cur, buf)

    if AUDIT_FRONT and _fdiff:
        print('  DIFF front 增量 vs 重算 不一致(前几处):', flush=True)
        for x in _fdiff:
            print(f'    blk={x[0]} {x[1]}/{x[2]}/{x[3]} px={x[4]} 增量={x[5]:.4f} 重算={x[6]:.4f} '
                  f'ahead={x[7]} 其中还在簿={x[8]} 本块该档DQ={x[9]}', flush=True)
    elif AUDIT_FRONT:
        print('  OK front 增量 与 每块重算 一致', flush=True)
    bad = bk.audit_tot(5000)
    print(f'#EV {coin} {nev} 事件 {nblk} 块  穿价 {ncross}/{nblk} = {100*ncross/max(nblk,1):.3f}%'
          f'  tot 自检:{"OK" if not bad else f"DIFF {len(bad)} 档不一致 例:{bad[0]}"}', flush=True)
    res = {}
    for r in mk:
        k = f'{r.strat}_{r.arm}_{r.side}'
        res[f'f_{k}'] = np.array(r.fills, dtype=float) if r.fills else np.zeros((0, 3))
    for t, s in tk.items():
        res[f'tk_{t}'] = np.array(s['fills'], dtype=float) if s['fills'] else np.zeros((0, 5))
    for s in strats:
        if s in MAKERS:
            for a in ARMS:
                n = sum(len(r.fills) for r in mk if r.strat == s and r.arm == a)
                print(f'  {s}/{a}: {n} 笔', flush=True)
    for t, s in tk.items():
        print(f'  {t}: {len(s["fills"])} 笔', flush=True)
    print('  重挂次数:', {f'{k[0]}{k[1]}': v for k, v in sorted(nrejoin.items())}, flush=True)
    for st in MAKERS:
        inv = sum(x.pos for x in mk if x.strat == st and x.arm == 'Q2')
        print(f'  {st} 末库存(Q2臂净额) = {inv:,.1f} 张 = {inv/max(QMAX*NTL,1e-9)*100:.0f}% 上限口径', flush=True)
    np.savez_compressed(out,
                        mid_blk=np.array(mid_blk, dtype=np.int64), mid_val=np.array(mid_val),
                        lvl=np.array(lvl_s) if lvl_s else np.zeros((0, 3)),
                        my_share=np.array(my_share) if my_share else np.zeros(0),
                        meta=np.array([nblk, ncross, nev, seedblk, tick, sz_dec,
                                       prm['med_spread_bp'], NTL, FEE_MK, FEE_TK, HOR]),
                        **res)
    print('saved', out, flush=True)


def files_for(coin):
    R = os.environ.get('HL_REPLAY_ROOT', 'data') + '/span47'   # span47/ 放 47 币连续 diffs 流
    return [f'{R}/A_diffs_{coin}.ndjson.gz', f'{R}/B_diffs_{coin}.ndjson.gz']


if __name__ == '__main__':
    cmd = sys.argv[1]; coin = sys.argv[2]
    av = sys.argv[3:]
    def opt(name, d=None):
        return av[av.index(name) + 1] if name in av else d
    b0 = int(opt('--b0', 1095336419)); b1 = int(opt('--b1', 1106599985))
    if cmd == 'probe':
        p = probe(files_for(coin), b0, b1)
        print(json.dumps({coin: p}, ensure_ascii=False))
    else:
        prm = json.loads(open(opt('--params')).read())[coin] if opt('--params') \
            else probe(files_for(coin), b0, b1)
        run(coin, files_for(coin), b0, b1, sys.argv[3],
            opt('--seed'), '--drop-negoid' in av,
            (opt('--strats') or 'M1,M2,M3,T1,T2').split(','), prm)

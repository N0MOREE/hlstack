#!/usr/bin/env python3
"""三臂保真度阶梯回测:A(公开 L2 5.4s + hftbacktest 概率队列) / B(真队列·块级) / C(真队列·动作级)。

v2 语义约定:
  A1 时间→块映射:查表(abc_data/ts2blk.npz,由 fills 头逐块建,见 data/ts2blk_build.py),
     不用线性锚点;本文件不做 ts 映射,快照文件已带块号。
  A2 A 臂缺档语义:目标价不在 20 档快照里 → front=None(看不见就不许成交),
     直到某张快照包含该档才以那张的档量起算。落位时分「可见/缺档」计数上报。
  A3 不给 --snap → A 臂骑手直接不建(不静默退化成触价成交)。
  C4 cur_sz 是订单属性,每笔成交只扣一次(在骑手循环之外);跨骑手重复扣减会高估 C。
  C5 吃单来源三分:tape(有 ai)/ newad(入场单同块可定序,覆盖极少)/
     noai(触发/清算等,块末结算)。noai 是已知残余缺口,不是被修掉了;
     把 noai 挪到块首对成交量影响很小。C 成交按三源分列上报;开跑前断言 tape 完整。
  B6 B 的成交量口径 = fills 文件(不含 remove_swept),与 book_replay 的 Q2(diffs 口径,
     含 swept)略有差异——swept 是「被证死」不是「在该档被吃」,fills 口径更准。
  其余:主循环块集合只由 diffs 决定;px 比较统一 0.5*tick。

设计不变的部分:同步报价(决策由 C 视角的簿做,三臂只在成交判定上分开)、
延迟(δ_feed 300ms=4 块 + 落位 3 块,块末悲观落位)、费率同值。块间隔实测 71.4ms。

用法:ledger_bt.py run COIN OUT.npz [--snap SNAPFILE] [--b0 N --b1 N] [--lat 3]
      [--params JSON] [--diffs F --tape F]
"""
import gzip, json, sys, os, collections
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kernels.book import Book, _apply_plain, probe   # noqa: E402


def prob_back(front, back):
    """内联自已退役的 kernels/l2(hftbacktest PowerProbQueueFunc3 口径):
    prob = 1 − (front/(front+back))³ ——不知道逐单时,按公式估撤单落在你身后的概率。"""
    d = front + back
    if d <= 0:
        return 1.0
    return 1.0 - (front / d) ** 3


R = os.environ.get('HL_REPLAY_ROOT', 'data')   # 数据根目录;p5_tape/ 放 taker 成交流,abc_data/ 放 fills 与块号映射
NTL = 50.0
FEE_MK = 1.5
LAT = 3               # 决策→落位,块(实测块间隔 71.4ms,3 块 ≈ 214ms),三臂同值
STRATS = ('M1', 'M3')


class Rider:
    __slots__ = ('strat', 'arm', 'side', 'alive', 'px', 'sz', 'filled', 'front',
                 'ahead', 'fills', 'joins', 'join_blk', 'pos')

    def __init__(self, strat, arm, side):
        self.strat, self.arm, self.side = strat, arm, side
        self.alive = False; self.px = None; self.sz = 0.0; self.filled = 0.0
        self.front = 0.0; self.ahead = None
        self.fills = []            # (blk, px, qty, join_blk)
        self.joins = []
        self.join_blk = 0; self.pos = 0.0

    def hit(self, blk, qty):
        g = min(self.sz - self.filled, qty)
        if g <= 1e-15:
            return 0.0
        self.filled += g
        self.fills.append((blk, self.px, g, self.join_blk))
        self.pos += g if self.side == 'B' else -g
        return g


def rnd(p, tick):
    return round(round(p / tick) * tick, 10)


def stream_by_block(path):
    if not path or not os.path.exists(path):
        return None
    def g():
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
    return g()


class Feeds:
    """diffs 是主时钟;tape/fills/snap 只在 diffs 的块边界被消费到 ≤ 当前块。"""
    def __init__(self, coin, snap_fp, diffs=None, tape=None):
        self.g = {
            'ev': stream_by_block(diffs or f'{R}/p4_out/diffs_{coin}.ndjson.gz'),
            'tk': stream_by_block(tape or f'{R}/p5_tape/taker_{coin}.ndjson.gz'),
            'fl': stream_by_block(f'{R}/abc_data/fills_{coin}.ndjson.gz'),
            'sn': stream_by_block(snap_fp),
        }
        assert self.g['ev'] is not None, 'diffs 流缺失'
        self.head = {k: (next(g) if g else None) for k, g in self.g.items()}

    def upto(self, k, blk):
        out = []
        while self.head[k] and self.head[k][0] <= blk:
            out += self.head[k][1]
            try:
                self.head[k] = next(self.g[k])
            except StopIteration:
                self.head[k] = None
        return out


def run(coin, out_fp, snap_fp, b0, b1, prm, lat=LAT, diffs=None, tape=None, apol='refined'):
    tick, sz_dec = prm['tick'], prm['sz_dec']
    tol = 0.5 * tick
    delta = prm['med_spread_bp'] / 1e4
    have_A = snap_fp is not None
    if not have_A:
        print('#WARN 无 --snap:A 臂不建(不允许静默退化成触价成交)', flush=True)
    # 开跑前断言 tape 完整(默认路径时查引擎收尾标记;显式 --tape 视为冒烟自担)
    if tape is None:
        tf = f'{R}/p5_tape/taker_{coin}.ndjson.gz'
        logf = f'{R}/p5_tape/run.log'
        mark = os.path.exists(logf) and f'taker_{coin} 吃单带写盘完成' in open(logf).read()
        # 标记之外再验尾块:gz 必须能读到底,且末块盖住 b1(EOFError 会静默截断 C 臂)
        last = -1
        try:
            with gzip.open(tf, 'rb') as fh:
                for ln in fh:
                    pass
            i = ln.find(b'"block":') + 8
            last = int(ln[i:ln.find(b',', i)])
        except Exception:
            last = -1
        assert mark and last >= b1 - 20_000, \
            f'{coin} 吃单带未完整(标记={mark} 尾块={last:,} 需≥{b1-20_000:,}),拒绝开跑'

    bk = Book(False)
    seedblk, _, _ = bk.seed(f'{R}/seeds/plain/{coin}.json')
    b0 = max(b0, seedblk + 1)
    arms = ('A', 'B', 'C') if have_A else ('B', 'C')
    riders = [Rider(s, a, sd) for s in STRATS for a in arms for sd in 'BA']
    by_grp = collections.defaultdict(list)
    for r in riders:
        by_grp[(r.strat, r.side)].append(r)

    snap = {'B': {}, 'A': {}}
    snap_blk = [-10**9]        # 最近一张快照的到达块
    STALE = 300                # 快照陈旧闸门:> 300 块(≈21 秒,4 个节奏)全体失明。
                               # 针对真实抓录快照的空洞(可达数十分钟)—— 没有它,A 会拿
                               # 很久以前的 front 继续扣量成交。合成快照无空洞,不受影响。
    trad_since = collections.defaultdict(float)
    pending = []
    mid_blk, mid_val = [], []
    st = collections.Counter()          # 计数器:join 可见/缺档、C 成交来源等

    def key_of(px):
        return round(px, 10)

    def a_front(side, px):
        """A 视角的 front。三档口径(--apol):
        zero    缺档一律 0 —— hftbacktest 对陈旧簿的实际约定(乐观上界)
        strict  缺档一律失明 —— 保守下界
        refined 窗内空档=0(L2 用户真看得见「这档是空的」),窗外(比可见最优还好/
                比第 20 档还深)才失明 —— 主口径"""
        q = snap[side].get(key_of(px))
        if q is not None:
            return q
        if apol == 'zero':
            return 0.0
        if apol == 'strict':
            return None
        d = snap[side]
        if not d:
            return None
        ks = d.keys()
        best, worst = (max(ks), min(ks)) if side == 'B' else (min(ks), max(ks))
        inside = (worst <= px <= best) if side == 'B' else (best <= px <= worst)
        return 0.0 if inside else None

    def join(r, blk, px, sz):
        r.alive = True; r.px = px; r.sz = sz; r.filled = 0.0
        r.join_blk = blk
        if r.arm == 'A':
            q = a_front(r.side, px)
            r.ahead = None
            r.front = q
            st['A_join_seen' if q is not None else 'A_join_blind'] += 1
            r.joins.append((blk, px, 1 if q is not None else 0))
        else:
            d = bk.lv.get((r.side, px))
            r.ahead = set(d.keys()) if d else set()
            r.front = sum(d.values()) if d else 0.0
            r.joins.append((blk, px, 1))

    def apply_A_snapshot(rows):
        new = {'B': {}, 'A': {}}
        for d in rows:
            for sd, arr in (('B', d['bids']), ('A', d['asks'])):
                new[sd] = {key_of(p): s for p, s in arr}
        for r in riders:
            if r.arm != 'A' or not r.alive:
                continue
            k = key_of(r.px)
            nq = new[r.side].get(k)
            if r.front is None:
                # 重见判定也走同一套口径(先换上新快照再判)
                old_b, old_a = snap['B'], snap['A']
                snap['B'], snap['A'] = new['B'], new['A']
                q = a_front(r.side, r.px)
                snap['B'], snap['A'] = old_b, old_a
                if q is not None:
                    r.front = q
                    st['A_became_visible'] += 1
                continue
            if nq is None:
                old_b, old_a = snap['B'], snap['A']
                snap['B'], snap['A'] = new['B'], new['A']
                q = a_front(r.side, r.px)
                snap['B'], snap['A'] = old_b, old_a
                if q is None:
                    r.front = None          # 真掉出窗:失明
                    st['A_lost_sight'] += 1
                else:
                    r.front = min(r.front, q) if q > 0 else 0.0
                continue
            oq = snap[r.side].get(k)
            if oq is not None:
                dec = max((oq - nq) - trad_since[(r.side, k)], 0.0)
                back = max(nq - max(r.front, 0.0), 0.0)
                pb = prob_back(max(r.front, 0.0), back)
                r.front -= (1.0 - pb) * dec
        trad_since.clear()
        snap['B'], snap['A'] = new['B'], new['A']
        snap_blk[0] = rows[-1]['block']

    def settle(blk, evs, tks, fls):
        # ── 原料 ──
        fills_by_taker = collections.defaultdict(list)
        for f in fls:
            fills_by_taker[f['taker_oid']].append(f)
        new_ai = {d['oid']: d['ai'] for d in evs
                  if d['type'] == 'new' and d.get('ai', -1) >= 0}
        tape_ai = {t['oid']: t['ai'] for t in tks}
        timeline = []                       # (ai, kind 0=撤单 1=吃单, payload)
        for d in evs:
            if d.get('ai', -1) >= 0 and d['type'] == 'remove':
                timeline.append((d['ai'], 0, d))
        srcs = {}
        for oid in fills_by_taker:
            if oid in tape_ai:
                timeline.append((tape_ai[oid], 1, oid)); srcs[oid] = 'tape'
            elif oid in new_ai:             # 部分成交后 resting 的入场单:入场即吃
                timeline.append((new_ai[oid], 1, oid)); srcs[oid] = 'newad'
            else:
                srcs[oid] = 'noai'
        timeline.sort(key=lambda x: (x[0], x[1]))
        cur_sz = {}

        def eat(oid):
            rows = fills_by_taker[oid]
            tside = rows[0]['tside']
            # cur_sz 是订单属性:每笔成交只扣一次(见头注 C4)
            for f in rows:
                mo = f['maker_oid']
                cur_sz[mo] = cur_sz.get(mo, bk.orders.get(mo, (0, 0, 0.0))[2]) - f['sz']
            for r in riders:
                if r.arm != 'C' or not r.alive or r.side == tside:
                    continue
                rem = r.sz - r.filled
                if rem <= 1e-15:
                    continue
                consumed_ahead = beyond = 0.0
                for f in rows:
                    if f['maker_oid'] in r.ahead:
                        consumed_ahead += f['sz']
                    elif (r.side == 'A' and f['px'] > r.px + tol) or \
                         (r.side == 'B' and f['px'] < r.px - tol):
                        beyond += f['sz']                 # 吃到比我更差的价
                    elif abs(f['px'] - r.px) < tol:
                        beyond += f['sz']                 # 同价打在我身后
                if consumed_ahead:
                    r.front = max(r.front - consumed_ahead, 0.0)
                    for f in rows:
                        o = f['maker_oid']
                        if o in r.ahead and cur_sz.get(o, 1.0) <= 1e-12:
                            r.ahead.discard(o)            # 吃光才出队
                if beyond > 0:
                    g = r.hit(blk, min(rem, beyond))
                    if g > 0:
                        st[f'C_fill_{srcs[oid]}'] += 1
                        st[f'C_usd_{srcs[oid]}'] += int(g * r.px)

        for ai, kind, x in timeline:
            if kind == 0:
                for r in riders:
                    if r.arm == 'C' and r.alive and x['oid'] in r.ahead:
                        r.front = max(r.front - cur_sz.get(x['oid'], x['sz']), 0.0)
                        r.ahead.discard(x['oid'])
            else:
                eat(x)
        for oid, s in srcs.items():
            if s == 'noai':
                eat(oid)                                   # 触发/清算:块末结算

        # ── B 臂:整块聚合。F 口径 = fills 文件(不含 swept,见 docstring B6)──
        F = collections.defaultdict(float)
        for f in fls:
            F[('A' if f['tside'] == 'B' else 'B', key_of(f['px']))] += f['sz']
        DQ = collections.defaultdict(list)
        csz = {}
        for d in evs:
            if d['type'] == 'new':
                continue
            e = bk.orders.get(d['oid'])
            if e is None:
                continue
            old = csz.get(d['oid'], e[2])
            dq = old - d['sz'] if d['type'] == 'update' else old
            csz[d['oid']] = d['sz'] if d['type'] == 'update' else 0.0
            DQ[(e[0], key_of(e[1]))].append((d['oid'], dq))
        for r in riders:
            if r.arm != 'B' or not r.alive:
                continue
            k = (r.side, key_of(r.px))
            f = F.get(k, 0.0)
            if f > r.front:
                r.hit(blk, f - r.front)
            red = sum(dq for oo, dq in DQ.get(k, ()) if oo in r.ahead)
            r.front = max(r.front - red, 0.0)

        # ── A 臂:成交按笔确定性扣减;front=None(缺档)不许成交 ──
        for k2, v in F.items():
            trad_since[k2] += v
        stale = blk - snap_blk[0] > STALE
        for r in riders:
            if r.arm != 'A' or not r.alive or r.front is None:
                continue
            if stale:
                r.front = None             # 快照太旧:失明,等下一张重见
                st['A_stale_blind'] += 1
                continue
            f = F.get((r.side, key_of(r.px)), 0.0)
            if f <= 0:
                continue
            r.front -= f
            if r.front < 0:
                r.hit(blk, -r.front)
                r.front = 0.0

    # ═════════ 主循环:diffs 是唯一的块时钟 ═════════
    m = Feeds(coin, snap_fp, diffs, tape)
    nblk = 0

    def qty_at(px):
        q = NTL / px
        return round(q, sz_dec) if sz_dec > 0 else max(round(q), 1)

    while m.head['ev'] is not None:
        blk = m.head['ev'][0]
        if blk > b1:
            break
        evs = m.upto('ev', blk)
        tks = m.upto('tk', blk)
        fls = m.upto('fl', blk)
        sns = m.upto('sn', blk)
        if blk < b0:
            _apply_plain(bk, evs)
            if sns:
                apply_A_snapshot(sns)
            continue
        settle(blk, evs, tks, fls)
        _apply_plain(bk, evs)
        if sns:
            apply_A_snapshot(sns)
        nblk += 1
        bb, aa = bk.bestb(), bk.besta()
        if bb is None or aa is None or aa <= bb:
            continue
        mid = (bb + aa) * 0.5
        mid_blk.append(blk); mid_val.append(mid)
        while pending and pending[0][0] <= blk:
            _, stg, sd, tgt = pending.pop(0)
            q = qty_at(tgt)
            for r in by_grp[(stg, sd)]:
                join(r, blk, tgt, q)
        for (stg, sd), rs in by_grp.items():
            ref = next(x for x in rs if x.arm == 'C')
            if stg == 'M3':
                tgt = bb if sd == 'B' else aa
            else:
                tgt = rnd(mid * (1 - delta), tick) if sd == 'B' else rnd(mid * (1 + delta), tick)
            need = (not ref.alive) or ref.px != tgt or ref.filled >= ref.sz - 1e-15
            if need and not any(p[1] == stg and p[2] == sd for p in pending):
                pending.append((blk + lat, stg, sd, tgt))

    res = {}
    for r in riders:
        k = f'{r.strat}_{r.arm}_{r.side}'
        res[f'f_{k}'] = np.array(r.fills) if r.fills else np.zeros((0, 4))
        res[f'j_{k}'] = np.array(r.joins) if r.joins else np.zeros((0, 3))
    np.savez_compressed(out_fp,
                        mid_blk=np.array(mid_blk, dtype=np.int64), mid_val=np.array(mid_val),
                        counters=json.dumps(dict(st)),
                        meta=np.array([nblk, st.get('C_fill_noai', 0), tick, sz_dec,
                                       prm['med_spread_bp'], NTL, FEE_MK, lat]),
                        **res)
    print(f'#DONE {coin} {nblk:,} 块', flush=True)
    tot_j = st['A_join_seen'] + st['A_join_blind']
    if tot_j:
        print(f"  A 落位:可见 {st['A_join_seen']} / 缺档 {st['A_join_blind']}"
              f"({100*st['A_join_blind']/tot_j:.1f}%),重见 {st['A_became_visible']}", flush=True)
    csrc = {k: v for k, v in st.items() if k.startswith('C_fill_')}
    print(f'  C 成交来源:{csrc}', flush=True)
    for stg in STRATS:
        line = '  '.join(
            f"{a}:{sum(len(r.fills) for r in riders if r.strat == stg and r.arm == a):>6d}笔"
            f"/{sum(x[2] * x[1] for r in riders if r.strat == stg and r.arm == a for x in r.fills):>12,.0f}$"
            for a in arms)
        print(f'  {stg}  {line}', flush=True)
    print('saved', out_fp, flush=True)


if __name__ == '__main__':
    cmd, coin = sys.argv[1], sys.argv[2]
    assert cmd == 'run', f'只支持 run,收到 {cmd}'
    av = sys.argv[3:]

    def opt(name, d=None):
        return av[av.index(name) + 1] if name in av else d
    b0 = int(opt('--b0', 1095336419)); b1 = int(opt('--b1', 1106599985))
    pf = opt('--params')
    if pf:
        prm = json.loads(open(pf).read())[coin]
    else:
        dfp = opt('--diffs') or f'{R}/p4_out/diffs_{coin}.ndjson.gz'
        prm = probe([dfp], 1095336419, 1095336419 + 200000)
        print(json.dumps({coin: prm}), file=sys.stderr)
    run(coin, sys.argv[3] if not sys.argv[3].startswith('--') else opt('--out'),
        opt('--snap'), b0, b1, prm, int(opt('--lat', LAT)),
        diffs=opt('--diffs'), tape=opt('--tape'), apol=opt('--apol', 'refined'))

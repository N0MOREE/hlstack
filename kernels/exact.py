#!/usr/bin/env python3
"""kernels.exact — 动作级成交判定内核(C 臂语义),只读 MBO+ 一个 npz。

移植母本:bench/ledger_bt.py 的 settle()/join()/主循环(v2),
逐条对拍验收:同窗口 M1+M3 成交与真值逐行相等(bench/exact_parity.py)。

与母本的输入差异(语义等价,对拍验证):
· 簿状态从 MBO+ data 行重建(oid 稳定,无认领);boot 行 + ghosts(kind=0,种子块)播种。
· 块时钟 = data 块 ∪ ghosts 块 —— 幽灵专属块(罕见但存在)也要走决策节拍。
· timeline 撤单 = CANCEL 行且 ival≥0(remove_fill 的 CANCEL 天然 ival=-1 被排除);
  撤量回退值用本簿当前量(母本用 diffs remove 行的 sz = 引擎删除时刻量,
  见 engine/src/main.rs 的 remove 发射;C 的 front 算术不进成交判定,此差异对成交惰性)。
· 吃单三分:trades.ai≠-1 → tape;taker_oid ∈ 本块 ADD(ival≥0)→ newad;其余 noai 块末结算。
· 加载时断言 meta 计数 dup_new_readd / first_seen_add / resurrect_add /
  mod_to_zero_as_cancel / resolved_claim / skip_unresolved 全零(4 币窗口实测如此)——
  非零意味着 ADD/CANCEL 行的来历多义(合成撤单、首见建单),本内核的行→事件映射不再唯一,
  需要格式加来历标记后再放行,拒绝静默出错。

--pessimistic(=B 臂:无定序、块级聚合悲观)只留接口位,未实现(路线图)。
费率/PnL 不入核(与母本一致,结算在下游)。
"""
import collections
import heapq
import json

import numpy as np

# hftbacktest 事件常量内联(与 hftbacktest.types 同值,实测 2.4.4)——
# exact 是我们自己的内核,不该为 5 个整数背上外部依赖
ADD_ORDER_EVENT, CANCEL_ORDER_EVENT, MODIFY_ORDER_EVENT, FILL_EVENT = 10, 11, 12, 13
BUY_EVENT = 0x20000000

NTL = 50.0
LAT = 3


class Rider:
    __slots__ = ('strat', 'side', 'alive', 'px', 'sz', 'filled', 'front',
                 'ahead', 'fills', 'joins', 'join_blk', 'pos')

    def __init__(self, strat, side):
        self.strat, self.side = strat, side
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


class ExactKernel:
    def __init__(self, npz_fp, prm, lat=LAT, strats=('M1', 'M3'), pessimistic=False):
        assert not pessimistic, '--pessimistic(B 臂降级)未实现,见路线图'
        self.tick, self.sz_dec = prm['tick'], prm['sz_dec']
        self.tol = 0.5 * self.tick
        self.lat = lat
        z = np.load(npz_fp)
        meta = json.loads(str(z['meta'][0]))
        assert meta.get('version') in ('v3', 'v3.1'), f'需要 MBO+ v3.x,拿到 {meta.get("version")}'
        c = meta['counts']
        if 'first_seen_zero_skip' not in c:
            # v3.0 产物:没有首见零量时钟标记——若窗口存在"整块只有首见零量 update"的块,
            # 该块会从时钟上消失且不可检测。建议用 v3.1 转换器重导。
            print('#WARN exact: v3.0 产物无 first_seen_zero_skip 计数,时钟完整性不可判定,建议重导', flush=True)
        for k in ('dup_new_readd', 'first_seen_add', 'resurrect_add',
                  'mod_to_zero_as_cancel', 'resolved_claim', 'skip_unresolved'):
            assert c.get(k, 0) == 0, f'{k}={c[k]}≠0:行→事件映射多义,拒绝开跑(见头注)'
        self.meta = meta
        d = z['data']
        ev = d['ev']
        kind = ev & 0xFF                       # 事件种类是低字节数值(ADD=10 CANCEL=11 …),不是独立位
        self.k_add = kind == ADD_ORDER_EVENT
        self.k_cxl = kind == CANCEL_ORDER_EVENT
        self.k_mod = kind == MODIFY_ORDER_EVENT
        self.k_fill = kind == FILL_EVENT
        self.buy = (ev & BUY_EVENT) != 0
        self.px = d['px']; self.qty = d['qty']
        self.oid = d['order_id']; self.ai = d['ival']
        self.blk = z['blk']; self.uid = z['owner']
        self.trades = z['trades']
        self.ghosts = z['ghosts']

        # ── 簿(镜像 kernels.book.Book 的状态效果;oid 稳定,无 _resolve)──
        self.orders = {}                       # oid -> [side, px, sz, uid]
        self.lv = {}                           # (side,px) -> {oid: sz} 插入序
        self.bh = []; self.ah = []

        # 播种:boot 行(ival=-2 的 ADD)+ 种子块幽灵(kind=0)
        seedblk = meta['seed_block']
        i = 0
        n = len(d)
        while i < n and self.ai[i] == -2:
            if self.k_add[i]:
                self._add(int(self.oid[i]), 'B' if self.buy[i] else 'A',
                          float(self.px[i]), float(self.qty[i]), int(self.uid[i]))
            i += 1                             # DEPTH_CLEAR 跳过
        self._boot_end = i
        g = self.ghosts
        self._gpos = 0
        while self._gpos < len(g) and g['blk'][self._gpos] == seedblk and g['kind'][self._gpos] == 0:
            r = g[self._gpos]
            self._add(int(r['oid']), 'B' if r['side'] == 1 else 'A',
                      float(r['px']), float(r['sz']), int(r['owner']))
            self._gpos += 1

        # ── 块时钟 = data 块 ∪ ghosts 块(幽灵专属块也要走决策节拍)──
        db = self.blk[self._boot_end:]
        cut = np.flatnonzero(np.diff(db)) + 1
        starts = np.concatenate(([0], cut)) + self._boot_end
        ends = np.concatenate((cut, [len(db)])) + self._boot_end
        data_clk = {int(self.blk[s]): (int(s), int(e)) for s, e in zip(starts, ends)}
        gb = set(int(b) for b in g['blk'][self._gpos:])
        self.clock = sorted(set(data_clk) | gb)
        self._data_clk = data_clk
        self._ci = 0
        self._ti = 0                           # trades 指针(upto ≤ 当前块)

        self.riders = [Rider(s, sd) for s in strats for sd in 'BA']
        self.by_grp = {}
        for r in self.riders:
            self.by_grp.setdefault((r.strat, r.side), []).append(r)
        self.pending = []
        self.pulls = collections.defaultdict(list)   # (stg,sd) -> [撤单生效块]
        self.st = collections.Counter()
        self.cur_blk = None

    # ── 簿原语(逐行镜像 Book.add / _apply_plain 的 update/remove 效果)──
    def _add(self, o, sd, px, sz, u):
        self.orders[o] = [sd, px, sz, u]
        self.lv.setdefault((sd, px), {})[o] = sz
        heapq.heappush(self.bh, -px) if sd == 'B' else heapq.heappush(self.ah, px)

    def _mod(self, o, sz):
        e = self.orders.get(o)
        if e is None:
            return
        e[2] = sz
        self.lv[(e[0], e[1])][o] = sz

    def _del(self, o):
        e = self.orders.pop(o, None)
        if e is None:
            return
        dd = self.lv.get((e[0], e[1]))
        if dd is not None:
            dd.pop(o, None)
            if not dd:
                self.lv.pop((e[0], e[1]), None)

    def bestb(self):
        while self.bh:
            p = -self.bh[0]
            if self.lv.get(('B', p)):
                return p
            heapq.heappop(self.bh)

    def besta(self):
        while self.ah:
            p = self.ah[0]
            if self.lv.get(('A', p)):
                return p
            heapq.heappop(self.ah)

    def mid(self):
        bb, aa = self.bestb(), self.besta()
        if bb is None or aa is None or aa <= bb:
            return None
        return (bb + aa) * 0.5

    # ── 结算(逐行移植 ledger_bt.settle 的 C 臂部分)──
    def _settle(self, blk, s, e, tr):
        fills_by_taker = collections.defaultdict(list)
        for j in range(*tr):
            fills_by_taker[int(self.trades['taker_oid'][j])].append(j)
        new_ai = {}
        timeline = []
        for i in range(s, e):
            a = int(self.ai[i])
            if a < 0:
                continue
            if self.k_add[i]:
                new_ai[int(self.oid[i])] = a
            elif self.k_cxl[i]:
                timeline.append((a, 0, int(self.oid[i])))
        srcs = {}
        T = self.trades
        for oid, rows in fills_by_taker.items():
            a = int(T['ai'][rows[0]])
            if a != -1:
                timeline.append((a, 1, oid)); srcs[oid] = 'tape'
            elif oid in new_ai:
                timeline.append((new_ai[oid], 1, oid)); srcs[oid] = 'newad'
            else:
                srcs[oid] = 'noai'
        timeline.sort(key=lambda x: (x[0], x[1]))
        cur_sz = {}

        def eat(oid):
            rows = fills_by_taker[oid]
            tside = 'B' if T['tside'][rows[0]] == 1 else 'A'
            for j in rows:
                mo = int(T['maker_oid'][j])
                cur_sz[mo] = cur_sz.get(mo, self.orders.get(mo, (0, 0, 0.0))[2]) - float(T['sz'][j])
            for r in self.riders:
                if not r.alive or r.side == tside:
                    continue
                rem = r.sz - r.filled
                if rem <= 1e-15:
                    continue
                consumed_ahead = beyond = 0.0
                for j in rows:
                    mo = int(T['maker_oid'][j])
                    fpx = float(T['px'][j]); fsz = float(T['sz'][j])
                    if mo in r.ahead:
                        consumed_ahead += fsz
                    elif (r.side == 'A' and fpx > r.px + self.tol) or \
                         (r.side == 'B' and fpx < r.px - self.tol):
                        beyond += fsz                 # 吃到比我更差的价
                    elif abs(fpx - r.px) < self.tol:
                        beyond += fsz                 # 同价打在我身后
                if consumed_ahead:
                    r.front = max(r.front - consumed_ahead, 0.0)
                    for j in rows:
                        mo = int(T['maker_oid'][j])
                        if mo in r.ahead and cur_sz.get(mo, 1.0) <= 1e-12:
                            r.ahead.discard(mo)       # 吃光才出队
                if beyond > 0:
                    g = r.hit(blk, min(rem, beyond))
                    if g > 0:
                        self.st[f'C_fill_{srcs[oid]}'] += 1
                        self.st[f'C_usd_{srcs[oid]}'] += int(g * r.px)

        for ai, kind, x in timeline:
            if kind == 0:
                old = self.orders.get(x, (0, 0, 0.0))[2]
                for r in self.riders:
                    if r.alive and x in r.ahead:
                        r.front = max(r.front - cur_sz.get(x, old), 0.0)
                        r.ahead.discard(x)
            else:
                eat(x)
        for oid, sc in srcs.items():
            if sc == 'noai':
                eat(oid)                              # 触发/清算:块末结算

    def _apply(self, s, e, blk):
        # 交错修正:同块里 kind=1 幽灵删 + 同 oid 的 data ADD 并存时,
        # 顺序必然是"先幽灵删、后重挂"——若 ADD 在前,该单已发射,后续删除必走
        # data CANCEL 而不会成为幽灵。所以这类幽灵删要在 data 行**之前**落,
        # 否则删会误杀刚重挂的单,终局簿态与母本反向且零告警。
        g = self.ghosts
        if self._gpos < len(g) and int(g['blk'][self._gpos]) <= blk:
            adds_here = {int(self.oid[i]) for i in range(s, e) if self.k_add[i]}
            j = self._gpos
            while j < len(g) and int(g['blk'][j]) <= blk:
                if g['kind'][j] == 1 and int(g['oid'][j]) in adds_here:
                    self._del(int(g['oid'][j]))
                    g['kind'][j] = -1              # 标记已处理,尾部循环跳过
                j += 1
        for i in range(s, e):
            o = int(self.oid[i])
            if self.k_fill[i]:
                continue                              # 成交证据在 trades,簿变化在后续行
            if self.k_add[i]:
                self._add(o, 'B' if self.buy[i] else 'A',
                          float(self.px[i]), float(self.qty[i]), int(self.uid[i]))
            elif self.k_mod[i]:
                self._mod(o, float(self.qty[i]))
            elif self.k_cxl[i]:
                self._del(o)
        g = self.ghosts
        while self._gpos < len(g) and int(g['blk'][self._gpos]) <= blk:
            r = g[self._gpos]
            if r['kind'] == 0:
                self._add(int(r['oid']), 'B' if r['side'] == 1 else 'A',
                          float(r['px']), float(r['sz']), int(r['owner']))
            elif r['kind'] == 1:
                self._del(int(r['oid']))
            self._gpos += 1                  # kind=2 复活(data ADD 已处理)/ -1 已提前落 / 3 纯时钟

    # ── 对外 API ──
    def advance_block(self):
        """推进一块:settle(旧簿)→ apply(落簿)。返回块号;数据尽头返回 None。
        决策(mid 有效时 flush_pending + place)由驱动侧做,复刻母本节拍。"""
        if self._ci >= len(self.clock):
            return None
        blk = self.clock[self._ci]; self._ci += 1
        s, e = self._data_clk.get(blk, (0, 0))
        t0 = self._ti
        TB = self.trades['blk']
        while self._ti < len(TB) and TB[self._ti] <= blk:
            self._ti += 1
        self._settle(blk, s, e, (t0, self._ti))
        self._apply(s, e, blk)
        self.cur_blk = blk
        return blk

    def qty_at(self, px):
        q = NTL / px
        return round(q, self.sz_dec) if self.sz_dec > 0 else max(round(q), 1)

    def flush_pending(self, blk):
        """mid 有效的块:到期挂单落位(块末悲观插队);tgt=None = 撤而不挂(RL 用)。
        pulls 记撤单真实生效块(与 j_ 同口径:到期块常不在时钟里,实际生效在下一个
        有效块;驱动若自算 cur_blk+lat 会提前 1–3 块,实测 HYPE ov 7.97% 错位)。"""
        while self.pending and self.pending[0][0] <= blk:
            _, stg, sd, tgt, qty = self.pending.pop(0)
            if tgt is None:
                for r in self.by_grp[(stg, sd)]:
                    r.alive = False
                self.pulls[(stg, sd)].append(blk)
                continue
            q = self.qty_at(tgt) if qty is None else qty
            for r in self.by_grp[(stg, sd)]:
                r.alive = True; r.px = tgt; r.sz = q; r.filled = 0.0
                r.join_blk = blk
                d = self.lv.get((sd, tgt))
                r.ahead = set(d.keys()) if d else set()
                r.front = sum(d.values()) if d else 0.0
                r.joins.append((blk, tgt, 1))

    def place(self, stg, sd, tgt, qty=None):
        """请求挂单:lat 块后原子替换(等待期旧单继续结算);同组已有待落单则忽略。
        tgt=None = 撤而不挂;qty=None = 按 NTL 定量(母本口径,对拍走这条)。"""
        if not any(p[1] == stg and p[2] == sd for p in self.pending):
            self.pending.append((self.cur_blk + self.lat, stg, sd, tgt, qty))

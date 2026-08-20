#!/usr/bin/env python3
"""kernels.book — bench 三个回测器(book_replay / ledger_bt / mbo_check)共用的逐单簿。

Book:oid 级 FIFO 簿,含 _resolve 认领(unknown update/remove 按 档位→user→量
认领给合成号 >2^63 的在簿单)与首次露面建单。
_apply_plain:把一块 diffs 事件落簿。probe:扫一段数据推 tick/szDecimals/中位价差。
"""
import gzip, json, heapq, collections
import numpy as np

NEG = 1 << 63

# ═════════════════════ 簿 ═════════════════════
class Book:
    def __init__(self, drop_neg=False):
        self.orders = {}                      # oid -> [side, px, sz, user]
        self.lv = {}                          # (side,px) -> {oid: sz}  插入序 = 队列序
        self.tot = collections.defaultdict(float)   # (side,px) -> 总量(增量维护,别每次 sum)
        self.bh = []; self.ah = []
        self.drop_neg = drop_neg
        self.ncross = 0; self.nblk = 0

    def bestb(self):
        while self.bh:
            p = -self.bh[0]
            if self.lv.get(('B', p)): return p
            heapq.heappop(self.bh)
    def besta(self):
        while self.ah:
            p = self.ah[0]
            if self.lv.get(('A', p)): return p
            heapq.heappop(self.ah)
    def ltot(self, key):
        t = self.tot.get(key, 0.0)
        return t if t > 1e-12 else 0.0

    def ltot_exact(self, key):
        d = self.lv.get(key)
        return sum(d.values()) if d else 0.0

    def audit_tot(self, n=200):
        """抽查增量维护的 tot 和真实 sum 是否一致"""
        bad = []
        for k in list(self.lv.keys())[:n]:
            a = self.tot.get(k, 0.0); b = self.ltot_exact(k)
            if abs(a - b) > 1e-6 * max(1.0, abs(b)): bad.append((k, a, b))
        return bad

    def _resolve(self, d):
        """modify/合成单会换 oid;拿 user/px/sz 在该档把它找回来(第一路的「认领」)"""
        sd, px, u, sz = d.get('side'), d.get('px'), d.get('user'), d.get('sz', 0.0)
        dd = self.lv.get((sd, px))
        if not dd: return None
        for oo in dd:
            e = self.orders.get(oo)
            if e is not None and len(e) > 3 and e[3] == u and oo > NEG: return oo
        for oo, ss in dd.items():
            if abs(ss - sz) < 1e-9 and oo > NEG: return oo
        # 曾经还有第三条 fallback「该档任意 oid>NEG 的单都认领」—— 删掉了。
        # 它会把真单的 fill_partial 喂给不相干的 TWAP 子单:实测 ACE 段A 块 1101128584,
        # 真单 512056444968 的 5 条 fill_partial 全被认领给了 oid=1844...93078,
        # 导致该检查点两边各差 1 张。宁可漏认领(走「不认识就新建」那条路),不可乱认领。
        return None

    def add(self, o, sd, px, sz, user):
        self.orders[o] = [sd, px, sz, user]
        self.lv.setdefault((sd, px), {})[o] = sz
        self.tot[(sd, px)] += sz
        heapq.heappush(self.bh, -px) if sd == 'B' else heapq.heappush(self.ah, px)

    def seed(self, fp):
        """两种种子都吃:抓录快照格式(orders[] 带队列位 q)和 seeds/plain(bids/asks,带 timestamp)。
        plain 的排序用 (timestamp, oid) —— 实测 oid 序 ≡ 时间序在 99.9~100% 档位成立,
        比拿抓录快照的 q(种子单一律为 0)更可靠。"""
        raw = gzip.open(fp, 'rt').read() if fp.endswith('.gz') else open(fp).read()
        sn = json.loads(raw); sn = sn.get('data', sn)
        if 'orders' in sn:
            oo = sorted(sn['orders'], key=lambda x: (x['q'], x['oid']))
            get = lambda x: (x['oid'], x['side'], x['px'], x['sz'], x.get('user'))
            lb = sn['last_block']
        else:
            oo = sorted(sn['bids'] + sn['asks'], key=lambda x: (x.get('timestamp', 0), x['oid']))
            get = lambda x: (x['oid'], x['side'], x['price'], x['size'], x.get('user_address'))
            lb = sn['last_block_number']
        n = 0
        for x in oo:
            o, sd, px, sz, u = get(x)
            if self.drop_neg and o >= NEG: continue
            self.add(o, sd, px, sz, u); n += 1
        return lb, n, len(oo)


def probe(files, b0, b1, nmax=200000):
    """推 tick / szDecimals / 中位价差 / 中位档位名义额"""
    pxs = set(); szs = set(); n = 0
    bk = Book()
    spreads = []; lvlntl = []
    cur = -1; buf = []
    for fp in files:
        for ln in gzip.open(fp, 'rt'):
            i = ln.find('"block":') + 8; j = ln.find(',', i); blk = int(ln[i:j])
            if blk < b0: continue
            if blk > b1 or n > nmax: break
            d = json.loads(ln); n += 1
            pxs.add(d['px']); szs.add(d.get('sz', 0.0))
            if blk != cur:
                if buf: _apply_plain(bk, buf)
                bb, aa = bk.bestb(), bk.besta()
                if bb and aa and aa > bb:
                    spreads.append((aa - bb) / ((aa + bb) / 2) * 1e4)
                    lvlntl.append(min(bk.ltot(('B', bb)) * bb, bk.ltot(('A', aa)) * aa))
                buf = []; cur = blk
            buf.append(d)
        if n > nmax: break
    def decs(vals):
        m = 0
        for v in list(vals)[:20000]:
            s = f'{v:.10f}'.rstrip('0')
            if '.' in s: m = max(m, len(s.split('.')[1]))
        return m
    sp = sorted(pxs)
    ticks = [round(sp[i + 1] - sp[i], 10) for i in range(min(len(sp) - 1, 5000)) if sp[i + 1] > sp[i]]
    tick = min(ticks) if ticks else 1e-6
    return dict(tick=tick, px_dec=decs(pxs), sz_dec=decs(szs),
                med_spread_bp=float(np.median(spreads)) if spreads else 1.0,
                med_lvl_ntl=float(np.median(lvlntl)) if lvlntl else 0.0,
                n_probe=n)


def _apply_plain(bk, evs):
    """只落簿,不跑骑手(probe 用)"""
    for d in evs:
        t = d['type']; o = d['oid']
        if bk.drop_neg and o >= NEG: continue
        if t == 'new':
            bk.add(o, d['side'], d['px'], d['sz'], d.get('user'))
        elif t == 'update':
            e = bk.orders.get(o)
            if e is None:
                r = bk._resolve(d)
                if r is not None:
                    o = r; e = bk.orders.get(o)
            if e is None:
                # 不认识这个 oid:当作新单建进来(modify 换号后首次露面就是 update)
                if d.get('sz', 0.0) > 1e-12:
                    bk.add(o, d['side'], d['px'], d['sz'], d.get('user'))
                continue
            sd, px, old = e[0], e[1], e[2]; e[2] = d['sz']
            bk.tot[(sd, px)] += d['sz'] - old
            if d['sz'] <= 1e-12:
                bk.lv.get((sd, px), {}).pop(o, None)
                if not bk.lv.get((sd, px), {}): bk.lv.pop((sd, px), None)
                bk.orders.pop(o, None)          # 量归零 = 这张单没了,orders 也要删
            else: bk.lv[(sd, px)][o] = d['sz']
        else:
            e = bk.orders.pop(o, None)
            if e is None:
                r = bk._resolve(d)
                if r is None: continue
                o = r; e = bk.orders.pop(o, None)
                if e is None: continue
            sd, px, old = e[0], e[1], e[2]
            bk.tot[(sd, px)] -= old
            dd = bk.lv.get((sd, px))
            if dd is not None:
                dd.pop(o, None)
                if not dd: bk.lv.pop((sd, px), None)



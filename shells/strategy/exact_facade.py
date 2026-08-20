#!/usr/bin/env python3
"""exact 门面:把 hftbacktest 的动词表(elapse/depth/orders/submit/cancel/…)
架在 ExactKernel 的 rider 协议上,让同一份策略文件跑 exact 内核。
rider = 内核里代表我方某一侧挂单的句柄对象。

支持的是动词表的**子集**(够 _quoter 协议用),语义差异如实声明:
· 每边同时只有一张单;新 submit = lat 块后原子替换(等待期旧单继续结算)
· cancel = 请求"撤而不挂"(同样 lat 块延迟,与 hftbacktest 的撤单延迟同径)
· 同边已有在途请求时,submit/cancel 被忽略,该 oid 立记 EXPIRED(策略下块重试)
· submit 的 tif / order_type / wait 参数被忽略:只有 GTC LIMIT;cancel 未知 oid 为 no-op
· 无 reject/风控;费率按 maker 1.5bp 记在 state_values.fee,balance 不含 fee(同 hftbacktest)
"""
import numpy as np

BLOCK_NS = 71_400_000
NONE_, NEW, EXPIRED, FILLED, CANCELED, PARTIALLY_FILLED = 0, 1, 2, 3, 4, 5


class _Depth:
    __slots__ = ('best_bid', 'best_ask', 'tick_size')


class _Order:
    __slots__ = ('order_id', 'status', 'price', 'qty', 'cancellable', 'side')


class _StateValues:
    __slots__ = ('position', 'balance', 'fee', 'num_trades', 'trading_value')


class _OrderDict:
    def __init__(self, d):
        self._d = d

    def get(self, oid):
        return self._d.get(oid)

    def values(self):
        return list(self._d.values())


class ExactFacade:
    def __init__(self, kernel, fee_bp=1.5):
        self.k = kernel
        self.fee_rate = fee_bp * 1e-4
        self._orders = {}                  # oid -> _Order
        self._owner = {'B': None, 'A': None}   # 落位后持有骑手的 oid
        self._inflight = {'B': None, 'A': None}  # (oid|None撤, px, qty)
        self._fill_ptr = {'B': 0, 'A': 0}
        self._joins_seen = {'B': 0, 'A': 0}
        self._pos = 0.0; self._cash = 0.0; self._fee = 0.0
        self._ntr = 0; self._val = 0.0

    # ── 内部:落位/成交结算 → 订单状态机 ──
    def _rider(self, sd):
        return self.k.by_grp[('RL', sd)][0]

    def _sync(self):
        for sd in ('B', 'A'):
            r = self._rider(sd)
            # 1. 先结算成交,归属给"结算时刻的主人"(旧单)——落位转正在后。
            #    不变量:换单落位块上旧单的成交归旧单;撤单生效块上打满的单记 FILLED,
            #    不记 CANCELED。
            new_fills = r.fills[self._fill_ptr[sd]:]
            self._fill_ptr[sd] = len(r.fills)
            for _blk, px, q, _jb in new_fills:
                sgn = 1.0 if sd == 'B' else -1.0
                self._pos += sgn * q
                self._cash += -sgn * px * q
                self._fee += self.fee_rate * px * q
                self._ntr += 1; self._val += px * q
                own = self._owner[sd]
                if own is not None:
                    o = self._orders[own]
                    o.status = FILLED if r.filled >= r.sz - 1e-15 else PARTIALLY_FILLED
                    if o.status == FILLED:
                        o.cancellable = False
                        self._owner[sd] = None
            # 2. 落位:在途单转正,旧主人(若仍活着)记 CANCELED
            if len(r.joins) > self._joins_seen[sd]:
                self._joins_seen[sd] = len(r.joins)
                old = self._owner[sd]
                if old is not None and self._orders[old].status in (NONE_, NEW, PARTIALLY_FILLED):
                    self._orders[old].status = CANCELED
                infl = self._inflight[sd]
                if infl and infl[0] is not None:
                    self._owner[sd] = infl[0]
                    self._orders[infl[0]].status = NEW
                    self._orders[infl[0]].cancellable = True
                self._inflight[sd] = None
            # 3. 撤而不挂生效(已 FILLED 的不覆盖)
            if self._inflight[sd] and self._inflight[sd][0] is None and \
               not any(p[1] == 'RL' and p[2] == sd for p in self.k.pending):
                old = self._owner[sd]
                if old is not None and self._orders[old].status in (NONE_, NEW, PARTIALLY_FILLED):
                    self._orders[old].status = CANCELED
                self._owner[sd] = None
                self._inflight[sd] = None

    # ── hbt 动词表 ──
    def elapse(self, ns):
        n = max(1, int(round(ns / BLOCK_NS)))
        for _ in range(n):
            blk = self.k.advance_block()
            if blk is None:
                return 1
            if self.k.mid() is not None:
                self.k.flush_pending(blk)
            self._sync()
        return 0

    def depth(self, _a):
        d = _Depth()
        d.best_bid = self.k.bestb() or np.nan
        d.best_ask = self.k.besta() or np.nan
        d.tick_size = self.k.tick
        return d

    def _submit(self, sd, oid, px, qty):
        o = _Order()
        o.order_id, o.price, o.qty, o.side = oid, px, qty, (1 if sd == 'B' else -1)
        o.cancellable = False
        if self._inflight[sd] is not None:
            o.status = EXPIRED                 # 同边已有在途请求:立拒,策略下块重试
            self._orders[oid] = o
            return 0
        o.status = NONE_
        self._orders[oid] = o
        self._inflight[sd] = (oid, px, qty)
        self.k.place('RL', sd, px, qty=qty)
        return 0

    def submit_buy_order(self, _a, oid, px, qty, _tif, _ot, _wait):
        # tif / order_type / wait 被忽略:只有 GTC LIMIT
        return self._submit('B', oid, px, qty)

    def submit_sell_order(self, _a, oid, px, qty, _tif, _ot, _wait):
        return self._submit('A', oid, px, qty)

    def cancel(self, _a, oid, _wait):
        # 未知 oid / 非当前持单 / 同边已有在途请求:no-op,返回 0
        for sd in ('B', 'A'):
            if self._owner[sd] == oid and self._inflight[sd] is None:
                self._inflight[sd] = (None, None, None)
                self.k.place('RL', sd, None)
                return 0
        return 0

    def orders(self, _a):
        return _OrderDict(self._orders)

    def position(self, _a):
        return self._pos

    def state_values(self, _a):
        s = _StateValues()
        s.position = self._pos
        s.balance = self._cash            # hftbacktest 语义:balance 不含 fee,fee 单列
        s.fee = self._fee
        s.num_trades = self._ntr
        s.trading_value = self._val
        return s

    def clear_inactive_orders(self, _a):
        self._orders = {k: v for k, v in self._orders.items()
                        if v.status in (NONE_, NEW, PARTIALLY_FILLED)}

    def close(self):
        return 0


def wrap(kernel, fee_bp=1.5):
    return ExactFacade(kernel, fee_bp)

"""示例策略共用的挂单循环:每边同时只有一张单,旧单死透才补挂。

status == NONE(请求在途)必须视为活单:否则每块都会补挂,旧单无人撤,
簿上堆成一张网格被价格扫穿,成交数虚高几个数量级。
"""
from hftbacktest.order import CANCELED, EXPIRED, FILLED, GTC, LIMIT

DEAD = (EXPIRED, FILLED, CANCELED)


def run_quoter(hbt, cfg, target_px):
    """target_px(bb, aa, cfg) -> {'B': px, 'A': px}"""
    ntl, blk_ns, sd = cfg['ntl'], cfg['block_ns'], cfg['sz_dec']
    oid = 0
    live = {'B': None, 'A': None}
    steps = 0
    while hbt.elapse(blk_ns) == 0:
        steps += 1
        if cfg.get('max_steps') and steps > cfg['max_steps']:
            break
        d = hbt.depth(0)
        bb, aa = d.best_bid, d.best_ask
        if not (bb > 0 and aa > bb):
            continue
        tgt = target_px(bb, aa, cfg)
        for side in ('B', 'A'):
            px = tgt[side]
            cur = live[side]
            o = hbt.orders(0).get(cur) if cur is not None else None
            if o is not None and o.status not in DEAD:
                # 活着或在途(status=0 是请求在途,也算活)
                if abs(o.price - px) >= 1e-12 and o.cancellable:
                    hbt.cancel(0, cur, False)      # 价漂了:先撤,等回执
                continue
            qty = max(round(ntl / px, sd), 10 ** -sd if sd else 1)
            oid += 1
            sub = hbt.submit_buy_order if side == 'B' else hbt.submit_sell_order
            sub(0, oid, px, qty, GTC, LIMIT, False)
            live[side] = oid
        if steps % 3000 == 0:
            hbt.clear_inactive_orders(0)

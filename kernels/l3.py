#!/usr/bin/env python3
"""kernels.l3 — hftbacktest L3FIFO 原生内核(薄适配器)。

吃 MBO+ npz 的 data 数组(旁车数组 hftbacktest 按成员名无视,实测 2.4.4)。
已知口径差(相对 exact):
全成或不成(上游 L3 不支持部分成交)· 到达穿价即当 taker · FILL 事件不减簿。
延迟/费率与四臂标定同款:下单 3 块 ≈ 214.2ms,maker 1.5bp。
"""
from hftbacktest import BacktestAsset, HashMapMarketDepthBacktest

LAT_NS = 214_200_000    # 3 块 × 71.4ms(实测块间隔),与四臂同值
FEE = 0.00015           # maker 1.5bp,HL 基础档


def build(npz_fp, tick, lat_ns=LAT_NS, fee=FEE):
    """返回可驱动的 hbt 句柄(elapse / submit_buy_order / submit_sell_order / cancel / orders 等 hftbacktest 原生接口)。

    lot_size 必须 1e-9 假 lot:HL 成交可留亚 lot 残量,真实 lot 会让 hft
    按聚合取整误删还有活单的档 → delete_order panic。
    """
    asset = (BacktestAsset()
             .data([npz_fp])
             .linear_asset(1.0)
             .constant_order_latency(lat_ns, 0)
             .l3_fifo_queue_model()
             .no_partial_fill_exchange()
             .trading_value_fee_model(fee, 0.0)
             .tick_size(tick)
             .lot_size(1e-9))
    return HashMapMarketDepthBacktest([asset])

"""示例策略:mid ± 中位点差 双边挂单(比 m3.py 的 join-BBO 深一档位置;内部编号 M1)。

演示件,同 m3.py:不是 alpha。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _quoter import run_quoter  # noqa: E402


def _rnd(px, tick):
    return round(round(px / tick) * tick, 10)


def _tgt(bb, aa, cfg):
    mid = (bb + aa) / 2
    delta = cfg['med_spread_bp'] / 1e4
    tick = cfg['tick']
    return {'B': _rnd(mid * (1 - delta), tick), 'A': _rnd(mid * (1 + delta), tick)}


def strategy(hbt, cfg):
    run_quoter(hbt, cfg, _tgt)

"""示例策略:贴最优价双边挂单(join BBO),价漂了撤旧挂新(内部编号 M3)。

演示件:同一份文件在 l3 和 exact 两个内核下不改一行可跑、结果可比。不是 alpha。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _quoter import run_quoter  # noqa: E402


def strategy(hbt, cfg):
    run_quoter(hbt, cfg, lambda bb, aa, c: {'B': bb, 'A': aa})

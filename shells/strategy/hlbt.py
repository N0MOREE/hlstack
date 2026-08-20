#!/usr/bin/env python3
"""hlbt — 策略壳:一份策略文件,两个内核换着跑。

策略文件的形态照 hftbacktest 社区习惯(while-elapse 轮询,不是回调注册):

    def strategy(hbt, cfg):
        while hbt.elapse(BLOCK_NS) == 0:
            d = hbt.depth(0)
            hbt.submit_buy_order(0, oid, px, qty, GTC, LIMIT, False)
            ...

同一个动词表(elapse / depth / orders / submit_buy_order / submit_sell_order /
cancel / position / state_values)两个内核都给:
  l3    hftbacktest L3FIFO 原生(零翻译,hbt 就是 hftbacktest 句柄)
  exact 本仓库的内核(RL 壳只用它),经 exact_facade 接同一动词表

exact 内核下与 hftbacktest 的语义差异(详见 exact_facade.py):
  · tif / order_type / wait 参数被忽略,只有 GTC LIMIT 一种单
  · cancel 未知或已失效的 oid 为 no-op,返回 0
  · 每边同时只有一张单;同边在途请求期间的新 submit/cancel 被忽略(oid 记 EXPIRED)
  · --minutes 靠 cfg['max_steps'] 注入,由策略循环自行 break

用法:
  python -m shells.strategy.hlbt run --coin SOL --kernel exact \
      --mbo sample/mbo_SOL.npz --strategy shells/strategy/examples/m3.py [--minutes 30]

数据路径:--mbo 缺省为 $HL_REPLAY_ROOT/abc_out/mbo_{COIN}.npz(HL_REPLAY_ROOT 默认 'data');
币种参数读 $HL_FACTS/abc_params.json(HL_FACTS 默认 'sample')。
cfg 注入:tick / px_dec / sz_dec / med_spread_bp / coin / ntl(默认 $50)。
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

R = os.environ.get('HL_REPLAY_ROOT', 'data')
FACTS = os.environ.get('HL_FACTS', 'sample')

BLOCK_NS = 71_400_000          # 块间隔中位数 71.4ms


def _load_strategy(fp):
    spec = importlib.util.spec_from_file_location('user_strategy', fp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, 'strategy'):
        raise SystemExit(f'{fp} must define strategy(hbt, cfg)')
    return mod.strategy


def _build(kernel, mbo, tick, prm=None):
    if not mbo or not os.path.exists(mbo):
        raise SystemExit(f'MBO file not found: {mbo!r} (pass --mbo, e.g. sample/mbo_SOL.npz)')
    if kernel == 'l3':
        from kernels import l3
        return l3.build(mbo, tick)
    if kernel == 'exact':
        from kernels.exact import ExactKernel
        from shells.strategy.exact_facade import wrap  # hbt 动词表 → rider 协议
        return wrap(ExactKernel(mbo, prm, strats=('RL',)))
    raise SystemExit(f'unknown kernel {kernel!r} (choices: l3, exact)')


def cmd_run(a):
    prm = json.load(open(f'{FACTS}/abc_params.json'))[a.coin]
    cfg = dict(prm, coin=a.coin, ntl=a.ntl, block_ns=BLOCK_NS)
    mbo = a.mbo or f'{R}/abc_out/mbo_{a.coin}.npz'
    hbt = _build(a.kernel, mbo, prm['tick'], prm)
    strategy = _load_strategy(a.strategy)
    if a.minutes:
        cfg['max_steps'] = int(a.minutes * 60_000_000_000 / BLOCK_NS)
    strategy(hbt, cfg)
    sv = hbt.state_values(0)
    print(f'#hlbt {a.coin} kernel={a.kernel} strategy={os.path.basename(a.strategy)} '
          f'position={sv.position:.6g} balance={sv.balance:.4f} fee={sv.fee:.4f} '
          f'trades={sv.num_trades} volume=${sv.trading_value:,.0f}')
    hbt.close()


def main():
    ap = argparse.ArgumentParser(prog='hlbt')
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run', help='跑一个策略文件')
    r.add_argument('--coin', required=True)
    r.add_argument('--kernel', default='exact', choices=['l3', 'exact'])
    r.add_argument('--mbo', default=None, help='MBO npz 路径(缺省 $HL_REPLAY_ROOT/abc_out/mbo_{COIN}.npz)')
    r.add_argument('--strategy', required=True, help='定义 strategy(hbt,cfg) 的 py 文件')
    r.add_argument('--minutes', type=float, default=0, help='只跑前 N 分钟(0=全窗)')
    r.add_argument('--ntl', type=float, default=50.0, help='单笔名义额 $')
    r.set_defaults(fn=cmd_run)
    a = ap.parse_args()
    a.fn(a)


if __name__ == '__main__':
    main()

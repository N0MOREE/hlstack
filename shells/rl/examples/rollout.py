#!/usr/bin/env python3
"""策略无关的回测器:把任意策略在 HlEnv 里按块单集整窗滚动,产出与 bench/strat_run.py 同口径的文件。

用法:
    rollout.py COIN MBO.npz OUT.npz --policy 模块:属性 [--b0 N] [--b1 N] [--lat 3] [--seed 0]

--policy 指向三种之一:
    - 可调用对象  act(obs) -> action(MultiDiscrete 两元)
    - 带 predict(obs, deterministic=True) 的对象(stable-baselines3 模型接口)
    - 内置 hold:全程不动(用来验证回测器本身:PnL 必须恒为 0)

单集整窗:episode_blocks 设为窗口长度,中途不截断、不清仓;b0 之前只推进不下单(预热)。
产物 OUT.npz:
    f_RL_C_B / f_RL_C_A   (成交块, 价, 量, 落位块)            —— 与 strat_run 的 f_ 同口径
    pnl_blk / pnl_val     每步块号与含费累计 PnL($)
    pos_blk / pos_val     每步仓位(币)
    meta                  json:窗口、延迟参数、步数、笔数、名义额、期末仓位、期末 PnL
"""
import argparse
import importlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from shells.rl.env import HlEnv  # noqa: E402


def load_policy(spec, env):
    if spec == 'hold':
        zero = np.zeros(2, dtype=np.int64)
        return lambda obs: zero
    mod, _, attr = spec.partition(':')
    obj = getattr(importlib.import_module(mod), attr)
    if isinstance(obj, type):                  # 传的是类:实例化(构造器可选地接 env)
        try:
            obj = obj(env)
        except TypeError:
            obj = obj()
    if hasattr(obj, 'predict'):
        return lambda obs: np.asarray(obj.predict(obs, deterministic=True)[0])
    return obj


def main(a):
    env = HlEnv(a.coin, mbo_fp=a.mbo, action_latency_blocks=a.lat, episode_blocks=10 ** 12)
    act = load_policy(a.policy, env)
    obs, info = env.reset(seed=a.seed)
    hold = np.zeros(2, dtype=np.int64)
    pnl_blk, pnl_val, pos_blk, pos_val = [], [], [], []
    t0 = time.time()
    steps = 0
    while True:
        blk = info['block']
        if a.b1 and blk >= a.b1:
            break
        action = hold if (a.b0 and blk < a.b0) else act(obs)
        obs, r, term, trunc, info = env.step(action)
        steps += 1
        mid = info['mid']
        pnl_blk.append(info['block']); pnl_val.append(info['pnl_ntl'] * env.ntl)
        pos_blk.append(info['block']); pos_val.append(info['pos_ntl'] * env.ntl / mid if mid else 0.0)
        if term or trunc:
            break
    fills = env.fills
    fb = np.array([(f[0], f[2], f[3], f[4]) for f in fills if f[1] == 'B'], dtype=float).reshape(-1, 4)
    fa = np.array([(f[0], f[2], f[3], f[4]) for f in fills if f[1] == 'A'], dtype=float).reshape(-1, 4)
    ntl_usd = float(sum(f[2] * f[3] for f in fills))
    meta = {'coin': a.coin, 'policy': a.policy, 'b0': a.b0, 'b1': a.b1,
            'action_latency_blocks': a.lat, 'steps': steps, 'fills': len(fills),
            'notional_usd': ntl_usd, 'end_pos': pos_val[-1] if pos_val else 0.0,
            'end_pnl_usd': pnl_val[-1] if pnl_val else 0.0, 'wall_s': time.time() - t0}
    np.savez_compressed(a.out, f_RL_C_B=fb, f_RL_C_A=fa,
                        pnl_blk=np.array(pnl_blk), pnl_val=np.array(pnl_val),
                        pos_blk=np.array(pos_blk), pos_val=np.array(pos_val),
                        meta=json.dumps(meta))
    print(f"{a.coin} {a.policy}: {steps:,} 步 / 成交 {len(fills)} 笔 / 名义额 ${ntl_usd:,.0f} / "
          f"期末仓位 {meta['end_pos']:.4f} / 含费 PnL ${meta['end_pnl_usd']:+,.2f} / "
          f"{meta['wall_s']:.1f}s  -> {a.out}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('coin'); ap.add_argument('mbo'); ap.add_argument('out')
    ap.add_argument('--policy', required=True, help='模块:属性,或内置 hold')
    ap.add_argument('--b0', type=int, default=0, help='起始块(之前只预热不下单)')
    ap.add_argument('--b1', type=int, default=0, help='终止块(0=数据尽头)')
    ap.add_argument('--lat', type=int, default=3, help='动作落地延迟(块)')
    ap.add_argument('--seed', type=int, default=0)
    main(ap.parse_args())

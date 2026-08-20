#!/usr/bin/env python3
"""接入示例:任何按 gymnasium 接口工作的 agent 都能直接用 HlEnv。

    python shells/rl/examples/plug_in.py sample/mbo_SOL.npz
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from shells.rl.env import HlEnv  # noqa: E402


class QuoteBoth:
    """最小策略对象:两侧都挂在最优价(动作 2 = 最优价 ∓0 档)。接自己的模型时替换此类。"""

    def __call__(self, obs):
        return np.array([2, 2])


if __name__ == '__main__':
    env = HlEnv('SOL', mbo_fp=sys.argv[1], episode_blocks=500)
    policy = QuoteBoth()
    obs, info = env.reset(seed=0)
    print('obs 维度', obs.shape, '| 动作空间', env.action_space, '| 起始块', info['block'])
    for _ in range(500):
        obs, reward, terminated, truncated, info = env.step(policy(obs))
        if terminated or truncated:
            break
    print('块', info['block'], '| 成交', len(env.fills), '笔 | 含费 PnL', f"{info['pnl_ntl'] * env.ntl:+.2f} $")
    # stable-baselines3 接法(pip install stable-baselines3):
    #   from stable_baselines3 import PPO
    #   model = PPO('MlpPolicy', env).learn(total_timesteps=...); model.save('my_policy')
    #   回测:rollout.py SOL sample/mbo_SOL.npz out.npz --policy my_module:model

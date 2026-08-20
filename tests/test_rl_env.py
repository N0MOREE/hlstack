#!/usr/bin/env python3
"""shells.rl.env.HlEnv 的测试,数据用仓内 sample 切片;未装 gymnasium 时整模块跳过。

覆盖:gymnasium check_env、同 seed 同轨迹、撤单在延迟块数后生效、随机动作下 obs/reward 有限、
addr_obs 开关、单集整窗不变量(成交账本与仓位一致)。
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

pytest.importorskip('gymnasium', reason="requires gymnasium: pip install 'hlstack[rl]'")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_MBO = os.path.join(REPO, 'sample', 'mbo_SOL.npz')
pytestmark = pytest.mark.data

from shells.rl.env import HlEnv  # noqa: E402


def _env(**kw):
    kw.setdefault('episode_blocks', 400)
    kw.setdefault('mbo_fp', SAMPLE_MBO)
    return HlEnv('SOL', **kw)


def test_gymnasium_check_env():
    from gymnasium.utils.env_checker import check_env
    check_env(_env(), skip_render_check=True)


def test_same_seed_same_trajectory():
    rng = np.random.default_rng(7)
    acts = [rng.integers(0, 5, size=2) for _ in range(120)]
    traj = []
    for _ in range(2):
        env = _env()
        obs, _ = env.reset(seed=0)
        rs, os_ = [], [obs]
        for a in acts:
            obs, r, term, trunc, _ = env.step(a)
            rs.append(r); os_.append(obs)
            if term or trunc:
                break
        traj.append((np.array(rs), np.stack(os_)))
        env.close()
    assert np.array_equal(traj[0][0], traj[1][0]), 'same seed and actions: reward sequences differ'
    assert np.array_equal(traj[0][1], traj[1][1]), 'same seed and actions: obs sequences differ'


def test_pull_quote_takes_effect_after_latency():
    env = _env(action_latency_blocks=3)
    env.reset(seed=0)
    env.step([2, 2])                       # 双边挂在最优价
    for _ in range(6):
        env.step([0, 0])
    rd = env._riders()
    assert rd['B'].alive or rd['B'].filled > 0, 'order should be live (or filled) after latency'
    env.step([1, 1])                       # 撤而不挂
    for _ in range(6):
        env.step([0, 0])
    rd = env._riders()
    assert not rd['B'].alive and not rd['A'].alive, 'cancel should take effect after latency'
    env.close()


def test_random_actions_keep_obs_and_reward_finite():
    env = _env(receipt_latency_blocks=5, obs_delay_blocks=1)
    obs, _ = env.reset(seed=1)
    rng = np.random.default_rng(1)
    total_r = 0.0
    for i in range(300):
        obs, r, term, trunc, info = env.step(rng.integers(0, 5, size=2))
        assert np.all(np.isfinite(obs)), f'step {i}: non-finite obs'
        assert np.isfinite(r)
        total_r += r
        if term or trunc:
            break
    # 每步每边最多成交一张 ntl 大小的单,300 步随机挂撤的净仓位远小于 100 x ntl
    assert abs(info['pos_ntl']) < 100
    env.close()


def test_addr_obs_switch():
    """地址特征开关:维度 +4、gymnasium 合规、150 步全有限、占比在 [0,1]。"""
    from gymnasium.utils.env_checker import check_env
    env = _env(addr_obs=True)
    assert env.observation_space.shape[0] == 11 + 2 * env.k + 4
    check_env(env, skip_render_check=True)
    env = _env(addr_obs=True)
    obs, _ = env.reset(seed=3)
    rng = np.random.default_rng(3)
    for _ in range(150):
        obs, r, term, trunc, _ = env.step(rng.integers(0, 5, size=2))
        assert np.isfinite(obs).all()
        top_b, top_a = obs[-4], obs[-2]
        assert 0.0 <= top_b <= 1.0 and 0.0 <= top_a <= 1.0
        if term or trunc:
            break
    # 基线维度不受影响
    assert _env().observation_space.shape[0] == 11 + 2 * _env().k


def test_single_episode_full_window_invariants():
    """episode_blocks 远大于窗口时,400 步内 truncated 恒为 False;
    env.fills 的带符号数量和等于 info 里换算回币的仓位;obs 全有限。"""
    env = _env(episode_blocks=10 ** 9)
    obs, _ = env.reset(seed=5)
    rng = np.random.default_rng(5)
    info = None
    for i in range(400):
        obs, r, term, trunc, info = env.step(rng.integers(0, 5, size=2))
        assert trunc is False, f'step {i}: truncated with episode_blocks=1e9'
        assert np.all(np.isfinite(obs)) and np.isfinite(r)
        if term:
            break
    signed_qty = sum(f[3] if f[1] == 'B' else -f[3] for f in env.fills)
    pos_coins = info['pos_ntl'] * env.ntl / info['mid']
    assert abs(signed_qty - pos_coins) < 1e-9
    env.close()

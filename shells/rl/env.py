#!/usr/bin/env python3
"""HlEnv — gymnasium 标准环境:按非验证节点的信息集与时序建模,成交判定走 exact 内核。

    env = HlEnv('SOL', mbo_fp='sample/mbo_SOL.npz')     # 或 gym.make('HlExact-v0', ...)
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(action)

信息集:每个块边界给出该块 finalize 后一个非验证节点能看到的全部——全市场挂/撤/改单、
官方成交、回执;没有 mempool,没有共识前信息。obs 只由块末状态构成,块内信息不存在。

三个延迟参数(都以块计;块间隔实测中位约 68ms):
  action_latency_blocks   决策→落地。默认 3 块(≈200ms,外部 API 参与者发单到上链的典型区间;
                          有自己节点的实测就改这个数)
  receipt_latency_blocks  成交→可见。默认 0;>0 时 obs 里的仓位/盈亏/成交计数延后 N 块可见,
                          reward 仍按真值记账(延迟的是观测,不是世界)
  obs_delay_blocks        观测接收延迟。默认 0;>0 时整条 obs 延后 N 块

obs   Box(float32,11+2k 维;addr_obs=True 时 +4):仓位/盈亏/时间/盘口 k 档特征/我的挂单位置/
      两侧在途标志(同侧在途请求期间的新动作被忽略——原子替换语义,所以策略要能看见通道占线)。
      addr_obs:两侧最优档的 地址 top1 量占比 + log1p(地址数)。默认关:此前的实验里地址特征
      加进最好的模型后没有改善。
act   MultiDiscrete([k+2, k+2]):每边 0=不动 1=撤而不挂 2..k+2=挂在最优价 ∓(0..k) 档
rew   可插拔(reward_fn(snap)->float);默认 ΔPnL(以 ntl 为单位)− inv_penalty × 仓位²
fills 公开账本 env.fills:[(成交块, 方向, 价, 量, 落位块)],与 bench/strat_run.py 的 f_ 数组同口径
info  {'block', 'pnl_ntl', 'pos_ntl', 'mid'}:块号、含费累计盈亏(ntl 单位)、仓位名义额(ntl 单位)、块末 mid

不建模:自身订单对市场的反作用(回放世界里其他参与者看不见你)。

episode:顺序切窗(reset 从上一集末尾继续;数据走完自动从头重建);terminated = 数据尽头,
truncated = 达到 episode_blocks。整窗单集回测把 episode_blocks 设为窗口长度即可(见 examples/rollout.py)。
reset 语义:集末仓位不平仓、直接清零(不产生成交也不计费);在途请求与簿上挂单无延迟丢弃;
env.fills 与现金/盈亏归零。同 seed + 同动作序列 = 同轨迹。

术语:rider = 内核里代表我方某一侧挂单的句柄对象(ExactKernel.by_grp[(strat, side)][0]),
持有该侧的价、量、已成交量与成交记录。

数据路径:mbo_fp 必填(仓内样例 sample/mbo_SOL.npz);币种参数优先读 MBO 文件 meta.params,
缺省读 $HL_FACTS/abc_params.json(HL_FACTS 默认 'sample')。
"""
import json
import os
import sys

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:                       # pyproject extras: uv pip install -e '.[rl]'
    raise ImportError("gymnasium is required: pip install 'hlstack[rl]'") from e

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from kernels.exact import ExactKernel  # noqa: E402

FACTS = os.environ.get('HL_FACTS', 'sample')

DIST_CLIP_TICKS = 50.0     # obs 里"我的挂单离最优价几档"的上限
EPS = 1e-15                # 判定挂单是否打满的浮点容差
PX_ROUND = 10              # 价格小数位:tick 的整数倍相加后 round 掉浮点尾巴


def default_reward(snap):
    return snap['dpnl_ntl'] - snap['inv_penalty'] * snap['pos_ntl'] ** 2


class HlEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, coin, mbo_fp=None, params=None,
                 action_latency_blocks=3, receipt_latency_blocks=0, obs_delay_blocks=0,
                 reward_fn=None, k_levels=3, ntl=50.0, fee_bp=1.5, inv_penalty=0.0,
                 episode_blocks=20_000, kernel='exact', addr_obs=False):
        if kernel != 'exact':
            raise ValueError('only kernel="exact" is supported')
        if not mbo_fp:
            raise ValueError('mbo_fp is required, e.g. sample/mbo_SOL.npz')
        self.coin = coin
        self.mbo_fp = mbo_fp
        self.prm = params or self._prm_from_meta() \
            or json.load(open(f'{FACTS}/abc_params.json'))[coin]
        self.lat = int(action_latency_blocks)
        self.rcpt = int(receipt_latency_blocks)
        self.obs_delay = int(obs_delay_blocks)
        self.reward_fn = reward_fn or default_reward
        self.k = int(k_levels)
        self.ntl = float(ntl)
        self.fee = fee_bp * 1e-4
        self.inv_penalty = float(inv_penalty)
        self.episode_blocks = int(episode_blocks)
        self.addr_obs = bool(addr_obs)
        self._kern = None

        nobs = 11 + 2 * self.k + (4 if self.addr_obs else 0)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(nobs,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([self.k + 2, self.k + 2])

    def _prm_from_meta(self):
        """币种规格与标定常数:优先 MBO+ meta.params(单文件自含),缺省返回 None 回退 json。"""
        try:
            p = json.loads(str(np.load(self.mbo_fp)['meta'][0])).get('params')
            if p and all(x in p for x in ('tick', 'sz_dec', 'med_spread_bp')):
                return p
        except (OSError, KeyError, ValueError):
            pass
        return None

    # ── 内核生命周期:顺序切窗,数据走完才重建 ──
    def _rebuild(self):
        self._kern = ExactKernel(self.mbo_fp, self.prm, lat=self.lat, strats=('RL',))
        self._mid_hist = []

    def _riders(self):
        return {sd: self._kern.by_grp[('RL', sd)][0] for sd in ('B', 'A')}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self._kern is None or seed is not None:
            self._rebuild()
        k = self._kern
        # 清 rider 与账本(env 级账本;内核只判成交)
        for r in self._riders().values():
            r.alive = False; r.fills.clear(); r.joins.clear(); r.pos = 0.0
        k.pending.clear()
        self._fill_ptr = {'B': 0, 'A': 0}
        self._fills_log = []                   # (blk, side, px, qty) 全部成交,receipt 延迟用
        self.fills = []                        # 公开账本 (blk, side, px, qty, join_blk),与 strat_run f_ 同口径
        self._cash = 0.0; self._pos = 0.0
        self._pnl_prev = 0.0
        self._steps = 0
        self._obs_ring = []
        obs, ended = self._advance_until_valid_mid()
        if ended:                              # 上一集刚好耗尽数据:重建后再来
            self._rebuild()
            return self.reset()
        self._last_obs = obs
        return obs, {'coin': self.coin, 'block': k.cur_blk}

    def _advance_until_valid_mid(self):
        """推进到一个 mid 有效的块,返回 (obs, 数据尽头?)。穿价/空侧块照母本节拍跳过决策。"""
        while True:
            blk = self._kern.advance_block()
            if blk is None:
                return self._last_obs, True
            self._collect_fills(blk)
            m = self._kern.mid()
            if m is not None:
                self._kern.flush_pending(blk)
                self._mid_hist.append(m)
                return self._obs(m), False

    # ── 记账 ──
    def _collect_fills(self, blk):
        for sd, r in self._riders().items():
            new = r.fills[self._fill_ptr[sd]:]
            self._fill_ptr[sd] = len(r.fills)
            for fblk, px, q, jb in new:
                sgn = 1.0 if sd == 'B' else -1.0
                self._pos += sgn * q
                self._cash += -sgn * px * q - self.fee * px * q
                self._fills_log.append((fblk, sd, px, q))
                self.fills.append((fblk, sd, px, q, jb))

    def _visible_pos_pnl(self, mid):
        """receipt 延迟视角:只看见 blk ≤ 当前−rcpt 的成交。仓位/盈亏/笔数必须同一视角,
        否则成交计数会成为绕过回执延迟的信息泄漏通道。"""
        if self.rcpt == 0:
            return self._pos, self._cash + self._pos * mid, len(self._fills_log)
        cut = self._kern.cur_blk - self.rcpt
        pos = cash = 0.0
        nvis = 0
        for fblk, sd, px, q in self._fills_log:
            if fblk > cut:
                continue
            nvis += 1
            sgn = 1.0 if sd == 'B' else -1.0
            pos += sgn * q
            cash += -sgn * px * q - self.fee * px * q
        return pos, cash + pos * mid, nvis

    # ── obs ──
    def _obs(self, mid):
        k = self._kern
        tick = k.tick
        bb, aa = k.bestb(), k.besta()
        vpos, vpnl, nvis = self._visible_pos_pnl(mid)
        h = self._mid_hist
        ret1 = (h[-1] / h[-2] - 1) * 1e4 if len(h) >= 2 else 0.0
        ret5 = (h[-1] / h[-6] - 1) * 1e4 if len(h) >= 6 else 0.0
        rd = self._riders()
        # 打满的单永不再成交:不算"工作中的挂单"(否则会把"不补单"教成正确行为)
        def _working(r):
            return r.alive and r.filled < r.sz - EPS
        bdist = (bb - rd['B'].px) / tick if _working(rd['B']) else -1.0
        adist = (rd['A'].px - aa) / tick if _working(rd['A']) else -1.0
        pend_b = 1.0 if any(p[1] == 'RL' and p[2] == 'B' for p in k.pending) else 0.0
        pend_a = 1.0 if any(p[1] == 'RL' and p[2] == 'A' for p in k.pending) else 0.0
        f = [vpos * mid / self.ntl, vpnl / self.ntl,
             self._steps / max(self.episode_blocks, 1),
             (aa - bb) / tick, ret1, ret5,
             float(nvis) / max(self._steps, 1),
             min(bdist, DIST_CLIP_TICKS), min(adist, DIST_CLIP_TICKS), pend_b, pend_a]
        for i in range(self.k):
            for base, sd, sgn in ((bb, 'B', -1), (aa, 'A', +1)):
                px = round(base + sgn * i * tick, PX_ROUND)
                d = k.lv.get((sd, px))
                f.append(np.log1p((sum(d.values()) if d else 0.0) * mid / self.ntl))
        if self.addr_obs:                      # 地址特征:两侧最优档的地址构成(块末状态,无前视)
            for sd, best in (('B', bb), ('A', aa)):
                d = k.lv.get((sd, best))
                if not d:
                    f += [0.0, 0.0]
                    continue
                by = {}
                for oid, sz in d.items():
                    e = k.orders.get(oid)
                    u = e[3] if e is not None else 0
                    by[u] = by.get(u, 0.0) + sz
                tot = sum(by.values())
                f += [max(by.values()) / tot if tot > 0 else 0.0,
                      float(np.log1p(len(by)))]
        obs = np.asarray(f, dtype=np.float32)
        if self.obs_delay:
            self._obs_ring.append(obs)
            if len(self._obs_ring) > self.obs_delay + 1:
                self._obs_ring.pop(0)
            return self._obs_ring[0]
        return obs

    # ── step ──
    def step(self, action):
        k = self._kern
        bb, aa = k.bestb(), k.besta()
        for j, (sd, base, sgn) in zip(np.asarray(action).ravel(),
                                      (('B', bb, -1), ('A', aa, +1))):
            j = int(j)
            if j == 0:
                continue                       # 不动
            if j == 1:
                k.place('RL', sd, None)        # 撤而不挂
                continue
            tgt = round(base + sgn * (j - 2) * k.tick, PX_ROUND)
            q = round(self.ntl / tgt, k.sz_dec) if k.sz_dec > 0 else max(round(self.ntl / tgt), 1)
            k.place('RL', sd, tgt, qty=q)
        obs, ended = self._advance_until_valid_mid()
        self._last_obs = obs
        self._steps += 1
        mid = self._mid_hist[-1]
        pnl = self._cash + self._pos * mid
        snap = {'dpnl_ntl': (pnl - self._pnl_prev) / self.ntl,
                'pos_ntl': self._pos * mid / self.ntl,
                'pnl_ntl': pnl / self.ntl, 'inv_penalty': self.inv_penalty,
                'fills': self._fills_log[-8:],
                'block': self._kern.cur_blk if self._kern else -1}
        self._pnl_prev = pnl
        reward = float(self.reward_fn(snap))
        terminated = ended                     # 数据尽头 = 自然终止(下次 reset 自动重建)
        if ended:
            self._kern = None
        truncated = (not ended) and self._steps >= self.episode_blocks
        return obs, reward, terminated, truncated, {'block': snap['block'],
                                                    'pnl_ntl': snap['pnl_ntl'],
                                                    'pos_ntl': snap['pos_ntl'],
                                                    'mid': mid}

    def close(self):
        self._kern = None


try:                                           # gym.make('HlExact-v0', coin=...) 可用
    gym.register(id='HlExact-v0', entry_point='shells.rl.env:HlEnv')
except Exception:
    pass

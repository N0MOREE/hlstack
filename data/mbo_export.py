#!/usr/bin/env python3
"""把引擎 L4 diffs 流水转成 hftbacktest 的 L3(MBO)事件数组。

格式约定:diffs 不是自含的 oid 键流。kernels.book.Book 有两条消费端语义,
  转换器必须逐字镜像,否则播种态就错:
  ① _resolve 认领:update/remove 撞上不认识的 oid(modify 换号后的合成单),按
     (档位 → user 匹配 → 量匹配) 认领给该档一张合成号(>2^63)的在簿单;
  ② 首次露面建单:认领不上且 sz>0 的 update,当新单建进来。

状态镜像 Book(orders + lv 档位表,插入序=队列序),发射是覆盖层:
  new(sz>0)          → ADD
  new(sz≤0)          → 只进状态不发射(0 量单会让 hft 删档后 panic;留状态保认领同径)
  update 认领/已知     → FILL + MODIFY(绝对量)| sz≤0 时 FILL + CANCEL
  update 首次露面      → ADD(无 FILL:hft 视角这张单刚出现)
  update 复活(0→正)  → ADD
  remove(remove_fill) → FILL + CANCEL;其余 why → CANCEL;认领不上 → 跳过计数
时间戳:exch_ts = local_ts = 块 ts(ms→ns,ts2blk 查表,缺块取 ≤blk 最近并 +1ms 单调)
        + 块内槽位(文件序);ival = ai 溯源。播种 = 窗口首块前 50ms 的 DEPTH_CLEAR + ADD 序列。

用法: mbo_export.py COIN OUT.npz [--b0 N] [--b1 N] [--diffs F]
      [--ts2blk F] [--seed F] [--tape F] [--fills F]  # 四个输入都可指到切片,缺省走全量路径

v3(MBO+)旁车语义 —— data 数组与 v2 逐字节相同,新信息全在旁车数组:
  blk    int64,与 data 等长:每行的块号;boot 行(播种)= 种子块号
  owner  uint32,与 data 等长:每行主人编号;users 字符串数组做 编号→0x地址 字典
         (0 号保留给"无主"——DEPTH_CLEAR 与缺 user 的行)
  trades 结构化数组:窗口内 fills 带逐行 {blk,tid,taker_oid,maker_oid,px,sz,tside,ai}
         · 口径 = fills 带(不含 remove_swept,与 diffs 口径略有差异,见 ledger_bt 头注 B6)
         · ai 来自吃单带(p5_tape/ 目录下的 taker 流)按 (block, taker_oid) 回填;联不上 → ai=-1
           (触发/清算单,消费端语义 = 块末结算,BTC 笔数占 ~37%/金额 ~11%)
         · tside: B=+1 / 其他=-1(taker 方向)
  ghosts 结构化数组:emitted=False 的状态单事件 {blk,oid,px,side,sz,owner,kind}
         kind 0=add(含 boot 的 0 量种子单)/ 1=del(静默删)/ 3=首见零量时钟标记 / 2=resurrect(0→正量,
         data 同时有 ADD 行);hft 不读它,exact 内核读
  meta   单元素 json 字符串:version/coin/窗口/种子块/全部计数/fallback_1ms 触发数
"""
import gzip, json, sys
import os
import numpy as np

# hftbacktest 行格式内联(与 hftbacktest.types 同值,实测 2.4.4)——
# 产品线不为一个 dtype + 9 个整数背上 Rust 扩展依赖;装了 hft 时下面的自检会逐项核对。
event_dtype = np.dtype([('ev', '<u8'), ('exch_ts', '<i8'), ('local_ts', '<i8'),
                        ('px', '<f8'), ('qty', '<f8'), ('order_id', '<u8'),
                        ('ival', '<i8'), ('fval', '<f8')])
DEPTH_CLEAR_EVENT = 3
ADD_ORDER_EVENT, CANCEL_ORDER_EVENT, MODIFY_ORDER_EVENT, FILL_EVENT = 10, 11, 12, 13
EXCH_EVENT, LOCAL_EVENT = 0x80000000, 0x40000000
BUY_EVENT, SELL_EVENT = 0x20000000, 0x10000000
try:                                   # 装了 hftbacktest(bench extras)就逐项自检,没装照常跑
    from hftbacktest import types as _ht
    assert _ht.event_dtype == event_dtype
    assert (_ht.ADD_ORDER_EVENT, _ht.CANCEL_ORDER_EVENT, _ht.MODIFY_ORDER_EVENT,
            _ht.FILL_EVENT, _ht.DEPTH_CLEAR_EVENT) == (10, 11, 12, 13, 3)
    assert (_ht.EXCH_EVENT, _ht.LOCAL_EVENT, _ht.BUY_EVENT, _ht.SELL_EVENT) == \
           (EXCH_EVENT, LOCAL_EVENT, BUY_EVENT, SELL_EVENT)
except ImportError:
    pass

R = os.environ.get('HL_REPLAY_ROOT', 'data')   # 数据根目录;abc_data/ 块号映射与 fills,seeds/plain/ 种子,p4_out/ 引擎 diffs,p5_tape/ taker 流
EL = EXCH_EVENT | LOCAL_EVENT
NEG = 1 << 63                    # 合成号阈,与 book_replay.NEG 同值
CHUNK = 1 << 20


class Emitter:
    def __init__(self):
        self.chunks, self.buf, self.n = [], np.zeros(CHUNK, dtype=event_dtype), 0
        self.bchunks, self.bbuf = [], np.zeros(CHUNK, dtype=np.int64)     # v3 旁车:块号
        self.ochunks, self.obuf = [], np.zeros(CHUNK, dtype=np.uint32)    # v3 旁车:主人编号

    def put(self, ev, ts, px, qty, oid, ai, blk=0, uid=0):
        b, i = self.buf, self.n
        b[i]['ev'] = ev; b[i]['exch_ts'] = ts; b[i]['local_ts'] = ts
        b[i]['px'] = px; b[i]['qty'] = qty; b[i]['order_id'] = oid
        b[i]['ival'] = ai; b[i]['fval'] = 0.0
        self.bbuf[i] = blk; self.obuf[i] = uid
        self.n += 1
        if self.n == CHUNK:
            self.chunks.append(self.buf.copy()); self.n = 0
            self.bchunks.append(self.bbuf.copy()); self.ochunks.append(self.obuf.copy())

    def done(self):
        self.chunks.append(self.buf[:self.n].copy())
        self.bchunks.append(self.bbuf[:self.n].copy())
        self.ochunks.append(self.obuf[:self.n].copy())
        return (np.concatenate(self.chunks), np.concatenate(self.bchunks),
                np.concatenate(self.ochunks))


def main(coin, out_fp, b0, b1, diffs_fp, ts2blk_fp=None, seed_fp=None,
         tape_fp=None, fills_fp=None):
    z = np.load(ts2blk_fp or f'{R}/abc_data/ts2blk.npz')
    TB, TT = z['blk'], z['ts']

    def lookup_ns(blk):
        """表外块 = 表内锚块 ts + 块距×1ms。锚块自己(块距 0)拿精确 ts。
        没有这个偏移,无事件的表内块会把自己的时间戳拱手让给下一个表外块,
        两块事件共享同一时间基,任何检查点量尺都分不开(HYPE cp6 实证,
        oid 514752135372 的 remove 拿了 cp 块的精确 ts)。缺口最大 15 块,
        表内相邻 ts ≥ 54ms,偏移 ≤15ms 不会越过下一锚块,单调性成立。"""
        i = np.searchsorted(TB, blk, side='right') - 1
        assert i >= 0, f'块 {blk} 早于 ts2blk 表首块 {TB[0]}'
        return int(TT[i]) * 1_000_000 + (blk - int(TB[i])) * 1_000_000

    _last = [0]
    fb = [0]                                # v3:+1ms 兜底触发计数(以前不可查)

    def blk_ts_ns(blk):
        t = lookup_ns(blk)
        if t <= _last[0]:
            t = _last[0] + 1_000_000        # 双保险,理论上不再触发
            fb[0] += 1
        _last[0] = t
        return t

    # ── v3:主人字典(0 号保留给"无主")与幽灵单事件表 ──
    users = ['']
    umap = {'': 0, None: 0}

    def intern(u):
        i = umap.get(u)
        if i is None:
            i = len(users); umap[u] = i; users.append(u)
        return i

    GHOST_DT = np.dtype([('blk', '<i8'), ('oid', '<u8'), ('px', '<f8'),
                         ('side', 'i1'), ('sz', '<f8'), ('owner', '<u4'), ('kind', 'i1')])
    # oid 用 u8:引擎合成号 ≥2^63,data.order_id 本就是 uint64,i8 会在
    # "合成号单被静默删"时 OverflowError;读侧 int() 两种都兼容。
    ghosts = []

    def ghost(blk, oid, e, kind):
        ghosts.append((blk, oid, e[1], 1 if e[0] == 'B' else -1, e[2], intern(e[3]), kind))

    # ── 状态:镜像 book_replay.Book(orders/lv),外加 emitted 标志位 ──
    st = {}                       # oid -> [side, px, sz, user, emitted]
    lv = {}                       # (side,px) -> {oid: sz}  插入序 = 队列序

    def st_add(o, sd, px, sz, u, emitted):
        st[o] = [sd, px, sz, u, emitted]
        lv.setdefault((sd, px), {})[o] = sz

    def st_del(o):
        e = st.pop(o, None)
        if e is None:
            return None
        dd = lv.get((e[0], e[1]))
        if dd is not None:
            dd.pop(o, None)
            if not dd:
                lv.pop((e[0], e[1]), None)
        return e

    def resolve(d):
        """逐字镜像 Book._resolve:user 匹配优先,量匹配兜底,只认领合成号。"""
        dd = lv.get((d.get('side'), d.get('px')))
        if not dd:
            return None
        u, sz = d.get('user'), d.get('sz', 0.0)
        for oo in dd:
            e = st.get(oo)
            if e is not None and e[3] == u and oo > NEG:
                return oo
        for oo, ss in dd.items():
            if abs(ss - sz) < 1e-9 and oo > NEG:
                return oo
        return None

    sn = json.loads(open(seed_fp or f'{R}/seeds/plain/{coin}.json').read())
    sn = sn.get('data', sn)
    for o in sorted(sn['bids'] + sn['asks'], key=lambda x: (x.get('timestamp', 0), x['oid'])):
        st_add(o['oid'], o['side'], o['price'], o['size'], o.get('user_address'),
               o['size'] > 1e-12)
    seedblk = sn['last_block_number']
    if b0 <= seedblk:              # 母本口径(ledger_bt.py:134):种子态含种子块,窗口从其后开始
        print(f'#WARN b0={b0} ≤ 种子块 {seedblk},钳制到 {seedblk + 1}', flush=True)
        b0 = seedblk + 1

    em = Emitter()
    cnt = {'ADD': 0, 'CANCEL': 0, 'FILL': 0, 'MODIFY': 0, 'boot_orders': 0,
           'zero_add_stateonly': 0, 'first_seen_zero_skip': 0, 'first_seen_add': 0, 'resolved_claim': 0,
           'resurrect_add': 0, 'mod_to_zero_as_cancel': 0, 'skip_unresolved': 0,
           'dup_new_readd': 0, 'warm_rows': 0}
    booted = False
    cur_blk = -1
    slot = [0]
    base = [0]

    def take_ts(n=1):
        t = base[0] + slot[0]
        slot[0] += n
        return t

    def boot(first_blk):
        t = lookup_ns(first_blk) - 50_000_000
        em.put(DEPTH_CLEAR_EVENT | EL, t, 0.0, 0.0, 0, -2, seedblk, 0)
        k = 1
        for oid, e in st.items():
            if e[2] > 1e-12:
                em.put(ADD_ORDER_EVENT | EL | (BUY_EVENT if e[0] == 'B' else SELL_EVENT),
                       t + k, e[1], e[2], oid, -2, seedblk, intern(e[3]))
                k += 1
                e[4] = True
            else:
                e[4] = False
                ghost(seedblk, oid, e, 0)
        cnt['boot_orders'] = k - 1

    def apply(d, win):
        typ, o, ai = d['type'], d['oid'], d.get('ai', -1)
        dblk = d['block']
        if typ == 'new':
            sd, px, sz = d['side'], d['px'], d['sz']
            sflag = BUY_EVENT if sd == 'B' else SELL_EVENT
            if o in st:                          # 罕见:同号重挂,先撤旧再挂新
                old = st_del(o)
                if win and old is not None and old[4]:
                    em.put(CANCEL_ORDER_EVENT | EL |
                           (BUY_EVENT if old[0] == 'B' else SELL_EVENT),
                           take_ts(), old[1], 0.0, o, ai, dblk, intern(old[3]))
                cnt['dup_new_readd'] += 1
            emit = sz > 1e-12
            st_add(o, sd, px, sz, d.get('user'), emit)
            if not emit:
                cnt['zero_add_stateonly'] += 1
                if win:
                    ghost(dblk, o, st[o], 0)
            elif win:
                em.put(ADD_ORDER_EVENT | EL | sflag, take_ts(), px, sz, o, ai, dblk, intern(d.get('user')))
                cnt['ADD'] += 1
            return
        e = st.get(o); ro = o
        if e is None:
            r = resolve(d)
            if r is not None:
                ro, e = r, st.get(r)
                cnt['resolved_claim'] += 1
        if typ == 'update':
            if e is None:                        # 首次露面:当新单建(镜像 Book)
                if d.get('sz', 0.0) > 1e-12:
                    st_add(o, d['side'], d['px'], d['sz'], d.get('user'), True)
                    if win:
                        em.put(ADD_ORDER_EVENT | EL |
                               (BUY_EVENT if d['side'] == 'B' else SELL_EVENT),
                               take_ts(), d['px'], d['sz'], o, ai, dblk, intern(d.get('user')))
                    cnt['first_seen_add'] += 1
                elif win:
                    # 首见即零量:镜像 Book 不建状态,但要留时钟标记(kind=3)——
                    # 否则纯由这种行组成的块会从 exact 的块时钟上消失
                    ghosts.append((dblk, o, d['px'], 1 if d['side'] == 'B' else -1,
                                   0.0, intern(d.get('user')), 3))
                    cnt['first_seen_zero_skip'] += 1
                return
            sd, px = e[0], e[1]
            sflag = BUY_EVENT if sd == 'B' else SELL_EVENT
            sz = d['sz']
            if sz <= 1e-12:                      # 量归零 = 这张单没了(镜像 Book)
                emitted = e[4]
                uid = intern(e[3])
                st_del(ro)
                if win and emitted:
                    em.put(FILL_EVENT | EL | sflag, take_ts(), px, 0.0, ro, ai, dblk, uid)
                    em.put(CANCEL_ORDER_EVENT | EL | sflag, take_ts(), px, 0.0, ro, ai, dblk, uid)
                    cnt['FILL'] += 1; cnt['mod_to_zero_as_cancel'] += 1
                elif win:
                    ghosts.append((dblk, ro, px, 1 if sd == 'B' else -1, 0.0, uid, 1))
                return
            e[2] = sz; lv[(sd, px)][ro] = sz
            if win:
                if e[4]:
                    em.put(FILL_EVENT | EL | sflag, take_ts(), px, 0.0, ro, ai, dblk, intern(e[3]))
                    em.put(MODIFY_ORDER_EVENT | EL | sflag, take_ts(), px, sz, ro, ai, dblk, intern(e[3]))
                    cnt['FILL'] += 1; cnt['MODIFY'] += 1
                else:                            # 复活:0 量单被改出正量
                    e[4] = True
                    em.put(ADD_ORDER_EVENT | EL | sflag, take_ts(), px, sz, ro, ai, dblk, intern(e[3]))
                    cnt['resurrect_add'] += 1
                    ghost(dblk, ro, e, 2)
            else:
                e[4] = sz > 1e-12
            return
        # remove
        if e is None:
            if win:
                cnt['skip_unresolved'] += 1
            return
        emitted = e[4]
        sd, px = e[0], e[1]
        uid = intern(e[3])
        st_del(ro)
        if win and emitted:
            sflag = BUY_EVENT if sd == 'B' else SELL_EVENT
            if d.get('why') == 'remove_fill':
                em.put(FILL_EVENT | EL | sflag, take_ts(), px, 0.0, ro, ai, dblk, uid)
                cnt['FILL'] += 1
            em.put(CANCEL_ORDER_EVENT | EL | sflag, take_ts(), px, 0.0, ro, ai, dblk, uid)
            cnt['CANCEL'] += 1
        elif win:
            ghosts.append((dblk, ro, px, 1 if sd == 'B' else -1, 0.0, uid, 1))

    with gzip.open(diffs_fp, 'rb') as fh:
        for ln in fh:
            i = ln.find(b'"block":') + 8
            blk = int(ln[i:ln.find(b',', i)])
            if blk > b1:
                break
            d = json.loads(ln)
            if blk < b0:
                cnt['warm_rows'] += 1
                apply(d, False)
                continue
            if not booted:
                boot(blk); booted = True
            if blk != cur_blk:
                base[0] = blk_ts_ns(blk); slot[0] = 0; cur_blk = blk
            apply(d, True)

    data, blk_arr, owner_arr = em.done()
    assert booted, '窗口内一条 diffs 都没有'
    assert (np.diff(data['exch_ts']) >= 0).all(), 'exch_ts 非降序被破坏'

    # ── v3:trades 表 = 窗口内 fills 带 + 吃单带 ai 回填 ──
    tape_ai = {}                      # (block, taker_oid) -> ai
    with gzip.open(tape_fp or f'{R}/p5_tape/taker_{coin}.ndjson.gz', 'rb') as fh:
        for ln in fh:
            i = ln.find(b'"block":') + 8
            tb = int(ln[i:ln.find(b',', i)])
            if tb < b0:
                continue
            if tb > b1:
                break
            t = json.loads(ln)
            tape_ai.setdefault((tb, t['oid']), t['ai'])
    TRADE_DT = np.dtype([('blk', '<i8'), ('tid', '<i8'), ('taker_oid', '<i8'),
                         ('maker_oid', '<i8'), ('px', '<f8'), ('sz', '<f8'),
                         ('tside', 'i1'), ('ai', '<i4')])
    trows = []
    with gzip.open(fills_fp or f'{R}/abc_data/fills_{coin}.ndjson.gz', 'rb') as fh:
        for ln in fh:
            i = ln.find(b'"block":') + 8
            tb = int(ln[i:ln.find(b',', i)])
            if tb < b0:
                continue
            if tb > b1:
                break
            f = json.loads(ln)
            assert 0 <= f['tid'] < NEG, f'tid 越界 int64: {f["tid"]}'
            trows.append((tb, f['tid'], f['taker_oid'], f['maker_oid'], f['px'], f['sz'],
                          1 if f['tside'] == 'B' else -1,
                          tape_ai.get((tb, f['taker_oid']), -1)))
    trades = np.array(trows, dtype=TRADE_DT)
    ghosts_arr = np.array(ghosts, dtype=GHOST_DT)
    users_arr = np.array(users)
    ntl = trades['px'] * trades['sz']
    haveai = trades['ai'] != -1
    cov_n = float(haveai.mean()) if len(trades) else 0.0
    cov_d = float(ntl[haveai].sum() / ntl.sum()) if len(trades) and ntl.sum() > 0 else 0.0
    prm_fp = os.environ.get('HL_FACTS', 'sample') + '/abc_params.json'
    try:                               # 币种规格与标定常数入 meta:下游单文件自含(读侧缺省回退 json)
        _p = json.load(open(prm_fp))[coin]
        params = {k: _p[k] for k in ('tick', 'px_dec', 'sz_dec', 'med_spread_bp') if k in _p}
    except (OSError, KeyError, ValueError):
        params = None
    meta = dict(version='v3.1', coin=coin, b0=b0, b1=b1, seed_block=int(seedblk),
                params=params,
                counts={k: int(v) for k, v in cnt.items()}, fallback_1ms=int(fb[0]),
                trades_rows=int(len(trades)), tape_join_keys=int(len(tape_ai)),
                ai_cover_n=round(cov_n, 4), ai_cover_ntl=round(cov_d, 4),
                users=int(len(users_arr)), ghosts=int(len(ghosts_arr)))
    np.savez_compressed(out_fp, data=data, blk=blk_arr, owner=owner_arr,
                        users=users_arr, trades=trades, ghosts=ghosts_arr,
                        meta=np.array([json.dumps(meta)]))
    print(f'#DONE {coin} events={len(data):,} 窗口块 {b0}..{b1}')
    print('  ', {k: v for k, v in cnt.items() if v})
    print(f'   v3: trades={len(trades):,} ai覆盖 笔数 {cov_n:.1%} / 金额 {cov_d:.1%}'
          f'  ghosts={len(ghosts_arr)} users={len(users_arr):,} fallback_1ms={fb[0]}')
    print(f'   ts: {data["exch_ts"][0]} .. {data["exch_ts"][-1]}  saved {out_fp}')


if __name__ == '__main__':
    coin = sys.argv[1]; out = sys.argv[2]
    av = sys.argv[3:]

    def opt(name, d=None):
        return av[av.index(name) + 1] if name in av else d
    main(coin, out,
         int(opt('--b0', 1105281018)), int(opt('--b1', 1106599985)),
         opt('--diffs') or f'{R}/p4_out/diffs_{coin}.ndjson.gz',
         opt('--ts2blk'), opt('--seed'), opt('--tape'), opt('--fills'))

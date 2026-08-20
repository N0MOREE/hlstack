#!/usr/bin/env python3
"""微重放台架:不跑 9.31 天,在出错现场前后各取一张 0x 快照,重放几千块就能判语义对不对。

  micro.py run  <COIN> <BLOCK> [--pre 6000] [--post 1500] [--harvest 30000]
                [--flags "--stp-depth 1 ..."] [--tag NAME]
  micro.py suite <套件.json>            按清单批量跑,输出一张汇总表
  micro.py cross <COIN> <run目录>       只扫某次跑的 diffs 有没有穿价块

为什么这么设计
--------------
全程重放一次要 1h45m,改一行语义就要等很久。但语义错是局部的:错在哪个块已由引擎的
块末穿价自检(cross_sites)定位。所以:
  1. 向 0x 要「现场之前」那一刻的整簿快照,当种子(exact,不是近似)
  2. 从种子块往后重放到「现场之后」某块
  3. 再向 0x 要**同一个块号**的整簿快照,逐单对账

关键:第 3 步用 `--to-block <0x 说的 last_block_number>`,我方终态与真值**块号严格相等**,
不留「差几百块所以对不上很正常」的模糊地带。

成本:每个现场 2 credit(前后各一张 l4),两张都落 truth_cache 复用,重跑免费。
"""
import argparse, glob, gzip, heapq, json, os, re, subprocess, time

R = os.environ.get('HL_REPLAY_ROOT', 'data')               # 数据根目录
DATA = f'{R}/p3_data'                                        # p3_data/: 原始 actions/ + fills/ 全量流
BIN = os.environ.get('HL_REPLAY_BIN', 'engine/target/release/hl-replay')
MICRO = f'{R}/micro'                                         # 每次微重放的工作目录
TRUTH = f'{R}/truth_cache'                                   # 0x 快照缓存(重跑免费)
ASSET = {'HYPE': 159, 'BTC': 0, 'ACE': 96}

# 基线机制开关表。追加的实验开关由 --flags 覆盖(last-wins)。
# 注意:必须显式含 --q-order 1,引擎默认关;漏掉它时"关闭 q-order"的消融恒等于 0。
#
# REPO_FLAGS 是仓库 engine/ 识别的全部机制开关。EXTRA_RESEARCH_FLAGS 四个只存在于
# 研究版引擎(二进制路径含 p2_engine);仓库引擎对未知开关以 rc=2 拒收,所以按 BIN 选表,
# 不允许"传了但没生效"或"传了就被拒"的状态。
REPO_FLAGS = ('--filled-rest 1 --synthetic-modify 1 --trust-gone 1 --modify-err-cancel 1 '
              '--ro-proof 1 --ro-dedup 1 --rebind-emit 1 --q-order 1')
EXTRA_RESEARCH_FLAGS = '--ro-unknown-zero 1 --stp-taker 1 --stp-depth 1 --syn-strict 1'
if 'p2_engine' in BIN:                                       # 研究版引擎:全表,顺序保持原样
    BASE_FLAGS = ('--filled-rest 1 --synthetic-modify 1 --trust-gone 1 --modify-err-cancel 1 '
                  '--ro-unknown-zero 1 --stp-taker 1 --stp-depth 1 --ro-proof 1 --ro-dedup 1 '
                  '--syn-strict 1 --rebind-emit 1 --q-order 1')
else:                                                        # 仓库引擎:只传它认识的
    BASE_FLAGS = REPO_FLAGS


# ---------------------------------------------------------------- 块号 ↔ 时间戳
_FILES = None


def _files():
    global _FILES
    if _FILES is None:
        v = []
        for f in glob.glob(f'{DATA}/actions/*.ndjson.gz'):
            m = re.match(r'(\d+)-(\d+)\.ndjson\.gz', os.path.basename(f))
            if m:
                v.append((int(m.group(1)), int(m.group(2)), f))
        v.sort()
        _FILES = v
    return _FILES


def _fills_files():
    v = []
    for f in glob.glob(f'{DATA}/fills/*.ndjson.gz'):
        m = re.match(r'(\d+)-(\d+)\.ndjson\.gz', os.path.basename(f))
        if m:
            v.append((int(m.group(1)), int(m.group(2)), f))
    v.sort()
    return v


def slice_data(dst, lo, hi):
    """只把 [lo,hi] 覆盖到的 actions/fills 文件软链进一个小目录。

    引擎的 parse_pipeline 会读遍 --data 下的**全部**文件(22,533 个 / 24 GB),
    不认 --from-block。所以窗口再小也要扫全盘 —— 实测 3.7 万块的微重放跑了 8 分钟还没完。
    改数据目录而不是改引擎:零风险,且冻结二进制的行为逐字节不变。"""
    for sub, lst in (('actions', _files()), ('fills', _fills_files())):
        d = f'{dst}/{sub}'
        if os.path.isdir(d):
            for f in os.listdir(d):
                os.unlink(f'{d}/{f}')
        os.makedirs(d, exist_ok=True)
        n = 0
        for a, b, f in lst:
            if b >= lo and a <= hi:
                os.symlink(os.path.realpath(f), f'{d}/{os.path.basename(f)}')
                n += 1
        if n == 0:
            raise SystemExit(f'{sub} 在 [{lo},{hi}] 无文件')


def blk2ts(blk):
    """块号 → 毫秒时间戳。找不到那一块就退到文件内最近的一块。"""
    for lo, hi, f in _files():
        if lo <= blk <= hi:
            last = None
            for ln in gzip.open(f, 'rt'):
                i = ln.find('"number":') + 9
                n = int(ln[i:ln.find(',', i)])
                j = ln.find('"timestamp":') + 12
                t = int(ln[j:ln.find('}', j)])
                if n == blk:
                    return t
                if n > blk:
                    return last if last else t
                last = t
            return last
    raise SystemExit(f'块 {blk} 不在 p3_data 覆盖范围 [{_files()[0][0]}, {_files()[-1][1]}]')


# ---------------------------------------------------------------- 0x 真值
def truth_at(coin, ts, quiet=False):
    fn = f'{TRUTH}/{coin}_{ts}.json.gz'
    if os.path.exists(fn):
        return json.load(gzip.open(fn, 'rt'))
    import requests
    key = open(os.path.expanduser('~/.0xarchive_key')).read().strip()
    for att in range(5):
        try:
            if not quiet:
                print(f'  [0x] 拉 {coin} @ {ts} (第 {att + 1} 次,慢档要 20-100 秒)…', flush=True)
            r = requests.get(f'https://api.0xarchive.io/v1/hyperliquid/orderbook/{coin}/l4',
                             params={'timestamp': ts, 'limit': 300000},
                             headers={'X-API-Key': key}, timeout=300)
            r.raise_for_status()
            d = r.json()['data']
            os.makedirs(TRUTH, exist_ok=True)
            json.dump(d, gzip.open(fn, 'wt'))
            return d
        except Exception as ex:
            print(f'  [0x] 重试 {att}: {ex}', flush=True)
            time.sleep(8)
    raise SystemExit('0x 拉不到')


# ---------------------------------------------------------------- 对账
def seed_origin(seed_fp, diff_fp):
    """返回「来自种子、且整个窗口内零事件」的 oid 集合。

    这类单我们既没建也没删 —— 引擎在窗口里对它什么都没做,做的也只能是什么都不做。
    它出现在差异里,只能说明**起始快照与结束快照对它的说法不一致**:
    要么交易所在窗口内静默撤了它(G3p,免费数据里结构性看不见),
    要么它在起始快照里本来就已经死了(第三方真值自身的问题)。
    两种都不是引擎语义的账,所以必须单独拎出来,不许摊进「引擎错了多少」。"""
    sd = json.load(open(seed_fp)) if not seed_fp.endswith('.gz') else json.load(gzip.open(seed_fp, 'rt'))
    sd = sd.get('data', sd)
    seeded = {o['oid'] for o in sd['bids'] + sd['asks']}
    touched = set()
    if os.path.exists(diff_fp):
        # 内存必须以**种子规模**为界,不是 diffs 规模:全程 9.31 天的 BTC diffs 有 4 GB(gz)、
        # 上亿行,把每个 oid 都装进 set 会吃掉几十 GB。只关心种子里那 5–7 万个。
        with gzip.open(diff_fp, 'rb') as f:
            for ln in f:
                i = ln.find(b'"oid":') + 6
                o = int(ln[i:ln.find(b',', i)])
                if o in seeded:
                    touched.add(o)
    return seeded - touched


def reconcile(ours, truth, band_bp=20.0, inert=frozenset()):
    """我方终态 vs 真值,同一块号。返回逐项计数 + 带内明细。

    inert = seed_origin() 的产物:这些 oid 的差异单独计一份,不算在引擎头上。"""
    T = {o['oid']: o for o in truth['bids'] + truth['asks']}
    M = {o['oid']: o for o in ours['orders']}
    miss = set(T) - set(M)          # 真值有、我们漏
    ghost = set(M) - set(T)         # 只有我们有
    bb = max((float(o['price']) for o in truth['bids']), default=0.0)
    ba = min((float(o['price']) for o in truth['asks']), default=0.0)
    mid = (bb + ba) / 2 if bb and ba else 0.0

    def bp(p):
        return abs(float(p) - mid) / mid * 1e4 if mid else 9e9

    inb_miss = [(o, T[o], bp(T[o]['price'])) for o in miss if bp(T[o]['price']) <= band_bp]
    inb_ghost = [(o, M[o], bp(M[o]['px'])) for o in ghost if bp(M[o]['px']) <= band_bp]
    # 尺寸不一致
    szdiff = [o for o in set(T) & set(M)
              if abs(float(T[o]['size']) - float(M[o]['sz'])) > 1e-9]
    inb_sz = [o for o in szdiff if bp(T[o]['price']) <= band_bp]
    union = len(set(T) | set(M))
    ourb = max((o['px'] for o in ours['orders'] if o['side'] == 'B'), default=0.0)
    oura = min((o['px'] for o in ours['orders'] if o['side'] == 'A'), default=9e18)
    # 号错单对:一笔「漏」和一笔「幽」在 (主人, 方向, 价, 量) 上完全相同 —— 同一张单,
    # 只是我们手里的号是算错/合成的(G7)。对簿、盘口、深度、队列的**每一个下游消费者**
    # 它们都是同一张单,账面自相抵消。所以要单独计一类,不能当两笔错。
    pair_m, pair_g = set(), set()
    gkey = {}
    for o, ob, _ in inb_ghost:
        gkey.setdefault((ob.get('user', ''), ob['side'], round(ob['px'], 12),
                         round(ob['sz'], 9)), []).append(o)
    for o, ob, _ in inb_miss:
        k = (ob.get('user_address', ''), ob['side'], round(float(ob['price']), 12),
             round(float(ob['size']), 9))
        if gkey.get(k):
            pair_g.add(gkey[k].pop())
            pair_m.add(o)
    inb_all = [o for o, _, _ in inb_miss] + [o for o, _, _ in inb_ghost] + inb_sz
    inb_seed = [o for o in inb_all if o in inert]
    inb_pair = pair_m | pair_g
    return dict(
        n_truth=len(T), n_ours=len(M), miss=len(miss), ghost=len(ghost), szdiff=len(szdiff),
        inb_miss=len(inb_miss), inb_ghost=len(inb_ghost), inb_sz=len(inb_sz),
        inb_total=len(inb_all), inb_seed=len(inb_seed), inb_pair=len(inb_pair),
        inb_engine=len([o for o in inb_all if o not in inert and o not in inb_pair]),
        seed_inert=len(inert),
        acc=1 - (len(miss) + len(ghost)) / max(union, 1),
        crossed=bool(ourb >= oura), our_bbo=(ourb, oura), truth_bbo=(bb, ba),
        detail_miss=sorted(inb_miss, key=lambda x: x[2])[:200],
        detail_ghost=sorted(inb_ghost, key=lambda x: x[2])[:200])


# ---------------------------------------------------------------- 穿价扫描
def scan_cross(diff_fp, seed_fp=None):
    """流式重建 BBO,报块末穿价/锁定。纯 python,只为独立复核引擎自己的结论。"""
    # 价位存活性按**该价位上还有没有 oid** 判,不按量的浮点和判。
    # 用浮点求和判存活在短窗口上没问题,但在千万块量级上会出错:
    # 残量在 1e-9 门槛之上的漂移累积起来,让一个价位永远清不空,
    # 产生大量假穿价块(同一个价位卡住数百万块)。
    from collections import defaultdict
    bids, asks, side, px = defaultdict(set), defaultdict(set), {}, {}
    if seed_fp:
        sd = json.load(open(seed_fp)) if not seed_fp.endswith('.gz') else json.load(gzip.open(seed_fp, 'rt'))
        sd = sd.get('data', sd)
        seed0 = [(o['oid'], o['side'], float(o['price']), float(o['size']))
                 for o in sd['bids'] + sd['asks']]
    hits, cur, buf = [], -1, []

    # 最优价用**惰性堆**取,不用 max(bids)/min(asks)。
    # 后者是 O(价位数),乘 1,126 万块直接爆炸:HYPE 有几千档,实测 27 分钟没跑完。
    # 堆里可能留着已经空掉的价位,取的时候一路弹掉即可(每个价位最多进出一次,摊还 O(1))。
    bh, ah = [], []

    def _add(s, p, oid):
        if s == 'B':
            if p not in bids:
                heapq.heappush(bh, -p)
            bids[p].add(oid)
        else:
            if p not in asks:
                heapq.heappush(ah, p)
            asks[p].add(oid)

    def _del(s, p, oid):
        bk = bids if s == 'B' else asks
        v = bk.get(p)
        if v is not None:
            v.discard(oid)
            if not v:
                del bk[p]

    def _bestb():
        while bh:
            p = -bh[0]
            if bids.get(p):
                return p
            heapq.heappop(bh)

    def _besta():
        while ah:
            p = ah[0]
            if asks.get(p):
                return p
            heapq.heappop(ah)

    if seed_fp:
        for oid, s, p, _z in seed0:
            side[oid], px[oid] = s, p
            _add(s, p, oid)

    def flush(blk, evs):
        # diff 语义:new = 新增;update(fill_partial) 的 sz 是**剩余量**不是增量;remove = 撤走全部
        for e in evs:
            oid, t = e['oid'], e['type']
            if t == 'new':
                if oid in side:                       # 同号复用:先把老的那张摘干净
                    _del(side[oid], px[oid], oid)
                side[oid], px[oid] = e['side'], e['px']
                _add(e['side'], e['px'], oid)
            elif oid in side:
                if t == 'remove':
                    _del(side[oid], px[oid], oid)
                    side.pop(oid); px.pop(oid)
                elif e['px'] != px[oid]:              # 改量不挪价;真挪价了才动价位索引
                    _del(side[oid], px[oid], oid)
                    px[oid] = e['px']
                    _add(side[oid], e['px'], oid)
        bb = _bestb()
        aa = _besta()
        if bb is not None and aa is not None and aa <= bb:
            hits.append((blk, bb, aa))

    for ln in gzip.open(diff_fp, 'rt'):
        i = ln.find('"block":') + 8
        b = int(ln[i:ln.find(',', i)])
        if b != cur:
            if buf:
                flush(cur, buf)
            buf, cur = [], b
        buf.append(json.loads(ln))
    if buf:
        flush(cur, buf)
    return hits


# ---------------------------------------------------------------- 一个现场
def run_site(coin, blk, pre, post, harvest, flags, tag, quiet=False, keep=False):
    tag = tag or f'{coin}_{blk}'
    d = f'{MICRO}/{tag}'
    os.makedirs(f'{d}/seed', exist_ok=True)

    ts0 = blk2ts(max(blk - pre, _files()[0][0] + 1))
    ts1 = blk2ts(min(blk + post, _files()[-1][1] - 1))
    t0 = truth_at(coin, ts0, quiet)
    t1 = truth_at(coin, ts1, quiet)
    lb0, lb1 = t0['last_block_number'], t1['last_block_number']
    if lb1 <= lb0:
        return dict(tag=tag, err=f'两张真值块号不递增 {lb0} → {lb1}(0x 快照太稀)')
    json.dump({'success': True, 'data': t0}, open(f'{d}/seed/{coin}.json', 'w'))
    open(f'{d}/multi.txt', 'w').write(f'{ASSET[coin]} {coin}\n')

    frm = max(lb0 - harvest, _files()[0][0])
    slice_data(f'{d}/data', frm, lb1)
    cmd = (f'{BIN} --data {d}/data --multi {d}/multi.txt --seed-dir {d}/seed '
           f'--seed-block 0 --from-block {frm} --to-block {lb1} --warmup-blocks 0 '
           f'--out {d}/out --workers 3 {BASE_FLAGS} {flags}')
    t = time.time()
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    wall = time.time() - t
    open(f'{d}/run.log', 'w').write(cmd + '\n\n' + p.stdout + p.stderr)
    snap = f'{d}/out/snapshot_{coin}.json.gz'
    if p.returncode != 0 or not os.path.exists(snap):
        return dict(tag=tag, err=f'引擎失败 rc={p.returncode},见 {d}/run.log')

    ours = json.load(gzip.open(snap, 'rt'))
    if ours['last_block'] != lb1:
        return dict(tag=tag, err=f"块号不等 我方 {ours['last_block']} vs 真值 {lb1}")
    dfp = f'{d}/out/diffs_{coin}.ndjson.gz'
    res = reconcile(ours, t1, inert=seed_origin(f'{d}/seed/{coin}.json', dfp))
    res.update(tag=tag, coin=coin, site=blk, lb0=lb0, lb1=lb1,
               blocks=lb1 - lb0, wall=round(wall, 1), dir=d)
    res['cross'] = scan_cross(dfp, f'{d}/seed/{coin}.json') if os.path.exists(dfp) else []
    res['cross_n'] = len(res['cross'])
    res['cross_at_site'] = [c for c in res['cross'] if abs(c[0] - blk) <= 3]
    if not keep:
        for f in glob.glob(f'{d}/out/missfills_*'):
            os.remove(f)
    return res


def fmt(r):
    if 'err' in r:
        return f"  {r['tag']:22s} FAIL {r['err']}"
    x = 'FAIL 穿价' if r['cross_n'] else 'OK 不穿价'
    return (f"  {r['tag']:22s} {r['blocks']:>6d}块 {r['wall']:>5.1f}s  "
            f"漏 {r['miss']:>4d} 幽 {r['ghost']:>4d} 尺 {r['szdiff']:>3d}  "
            f"带内 {r['inb_total']:>3d}(引擎 {r.get('inb_engine', -1):>2d} / 号错对 "
            f"{r.get('inb_pair', 0):>2d} / 种子惰性 {r.get('inb_seed', -1):>2d})  "
            f"一致 {r['acc'] * 100:5.2f}%  {x}({r['cross_n']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['run', 'suite', 'cross'])
    ap.add_argument('a', nargs='*')
    ap.add_argument('--pre', type=int, default=6000)
    ap.add_argument('--post', type=int, default=1500)
    ap.add_argument('--harvest', type=int, default=30000)
    ap.add_argument('--flags', default='')
    ap.add_argument('--tag', default=None)
    ap.add_argument('--json', default=None)
    ap.add_argument('--keep', action='store_true')
    o = ap.parse_args()

    if o.cmd == 'run':
        coin, blk = o.a[0], int(o.a[1])
        r = run_site(coin, blk, o.pre, o.post, o.harvest, o.flags, o.tag, keep=o.keep)
        print(fmt(r))
        if 'err' not in r:
            print(f"  我方 BBO {r['our_bbo']}   真值 BBO {r['truth_bbo']}")
            for oid, ob, b in r['detail_miss']:
                print(f"    带内漏 {oid} {ob['side']} {ob['price']} × {ob['size']} ({b:.1f}bp)")
            for oid, ob, b in r['detail_ghost']:
                print(f"    带内幽 {oid} {ob['side']} {ob['px']} × {ob['sz']} ({b:.1f}bp)")
            for c in r['cross'][:10]:
                print(f"    穿价块 {c[0]}  买 {c[1]} ≥ 卖 {c[2]}")
        if o.json:
            json.dump(r, open(o.json, 'w'), default=str)
        return

    if o.cmd == 'suite':
        sites = json.load(open(o.a[0]))
        out = []
        for s in sites:
            r = run_site(s['coin'], s['block'], s.get('pre', o.pre), s.get('post', o.post),
                         o.harvest, s.get('flags', o.flags), s.get('tag'), keep=o.keep)
            out.append(r)
            print(fmt(r), flush=True)
        if o.json:
            json.dump(out, open(o.json, 'w'), default=str)
        ok = [r for r in out if 'err' not in r]
        print(f"\n合计 {len(ok)}/{len(out)} 跑通;穿价现场 "
              f"{sum(1 for r in ok if r['cross_n'])} 个;带内差异合计 "
              f"{sum(r['inb_total'] for r in ok)}")
        return

    if o.cmd == 'cross':
        coin, d = o.a[0], o.a[1]
        h = scan_cross(f'{d}/diffs_{coin}.ndjson.gz')
        print(f'{coin}: {len(h)} 个穿价/锁定块')
        for c in h[:50]:
            print(f'  块 {c[0]}  买 {c[1]} ≥ 卖 {c[2]}')


if __name__ == '__main__':
    main()

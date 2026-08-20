#!/usr/bin/env python3
"""样例包自检 a:种子 → 重放切片 diffs → 与真值快照逐 oid 对账。

内核用 kernels.book.Book(与 bench/book_replay.py 同一个被审计部件),
对账口径镜像 bench/micro.py 的 reconcile(±20bp 带内单独计数)。

用法: python sample/check_replay.py [SEED DIFFS TRUTH]   缺省 = sample/ 里的三个文件
判据: 重放终态块号 == 真值块号(不等则非零退出);报 簿上单数/漏单/幽灵单/量不一致
      及各自 ±20bp 带内数。
"""
import gzip
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from kernels.book import Book, _apply_plain  # noqa: E402


def load_snap(fp):
    raw = gzip.open(fp, 'rt').read() if fp.endswith('.gz') else open(fp).read()
    sn = json.loads(raw)
    return sn.get('data', sn)


def main(seed_fp, diffs_fp, truth_fp):
    truth = load_snap(truth_fp)
    b1 = truth['last_block_number']
    bk = Book()
    seedblk, nseed, ntot = bk.seed(seed_fp)
    print(f'种子块 {seedblk}  播入 {nseed}/{ntot} 张;重放到真值块 {b1}'
          f'({b1 - seedblk} 块)')
    cur = -1
    buf = []
    nev = nblk = 0
    for ln in gzip.open(diffs_fp, 'rt'):
        i = ln.find('"block":') + 8
        blk = int(ln[i:ln.find(',', i)])
        if blk <= seedblk:
            continue
        if blk > b1:
            break
        if blk != cur:
            if buf:
                _apply_plain(bk, buf)
                nblk += 1
            buf = []
            cur = blk
        buf.append(json.loads(ln))
        nev += 1
    if buf:
        _apply_plain(bk, buf)
        nblk += 1
    print(f'重放 {nev} 事件 / {nblk} 有事件块;终态块 {cur}')
    if cur != b1:
        print(f'FAIL: final block {cur} != truth block {b1}', file=sys.stderr)
        sys.exit(1)

    # ── 对账(镜像 bench/micro.py reconcile)──
    T = {o['oid']: o for o in truth['bids'] + truth['asks']}
    M = {o: (e[0], e[1], e[2]) for o, e in bk.orders.items() if e[2] > 1e-12}
    miss = set(T) - set(M)
    ghost = set(M) - set(T)
    bb = max(float(o['price']) for o in truth['bids'])
    ba = min(float(o['price']) for o in truth['asks'])
    mid = (bb + ba) / 2

    def bp(p):
        return abs(float(p) - mid) / mid * 1e4

    band = 20.0
    inb_miss = [o for o in miss if bp(T[o]['price']) <= band]
    inb_ghost = [o for o in ghost if bp(M[o][1]) <= band]
    szdiff = [o for o in set(T) & set(M)
              if abs(float(T[o]['size']) - M[o][2]) > 1e-9]
    inb_sz = [o for o in szdiff if bp(T[o]['price']) <= band]
    union = len(set(T) | set(M))
    acc = 1 - (len(miss) + len(ghost)) / max(union, 1)
    ourb = max((v[1] for v in M.values() if v[0] == 'B'), default=0.0)
    oura = min((v[1] for v in M.values() if v[0] == 'A'), default=9e18)
    print(f'真值簿 {len(T)} 张 / 我方簿 {len(M)} 张  oid 一致率 {acc:.4%}')
    print(f'漏单 {len(miss)}(±20bp 带内 {len(inb_miss)})  '
          f'幽灵单 {len(ghost)}(带内 {len(inb_ghost)})  '
          f'量不一致 {len(szdiff)}(带内 {len(inb_sz)})')
    print(f'带内差异合计 {len(inb_miss) + len(inb_ghost) + len(inb_sz)} 笔')
    print(f'我方 BBO {ourb}/{oura}  真值 BBO {bb}/{ba}  '
          f'crossed={ourb >= oura} bbo_match={(ourb, oura) == (bb, ba)}')


if __name__ == '__main__':
    a = sys.argv[1:]
    main(a[0] if a else f'{HERE}/seed_SOL.json',
         a[1] if len(a) > 1 else f'{HERE}/diffs_SOL.ndjson.gz',
         a[2] if len(a) > 2 else f'{HERE}/truth_SOL.json.gz')

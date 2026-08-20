#!/usr/bin/env python3
"""块号→真实毫秒 时间表:从 fills 原始流的块头 (number, timestamp) 逐块建表。

为什么必须查表:块间隔不恒定(实测均值 71.49ms,锚间最大缺口达数百块),线性锚点映射
误差可达数千块,且近半锚点会泄漏未来。l3 臂的 214.2ms 延迟全靠这张表落块。
表外块由消费端(mbo_export/mbo_run)按"≤blk 最近锚 + 块距×1ms"外推,单调性有 +1ms 兜底。

用法: ts2blk_build.py OUT.npz [--fills-dir D] [--lo N]
  默认吃 $HL_REPLAY_ROOT/p3_data/fills 下全部文件;--lo 只建 ≥N 的段(省内存)。
"""
import gzip
import json
import os
import sys

import numpy as np

R = os.environ.get('HL_REPLAY_ROOT', 'data')   # 数据根目录;p3_data/fills/ 放原始 fills 流


def main(out_fp, fills_dir, lo):
    blks, tss = [], []
    for fn in sorted(os.listdir(fills_dir)):
        b0f, b1f = map(int, fn.split('.')[0].split('-'))
        if b1f < lo:
            continue
        with gzip.open(os.path.join(fills_dir, fn), 'rt') as fh:
            for ln in fh:
                h = json.loads(ln)['header']
                blks.append(h['number']); tss.append(h['timestamp'])
    blks = np.array(blks, dtype=np.int64); tss = np.array(tss, dtype=np.int64)
    o = np.argsort(blks)
    blks, tss = blks[o], tss[o]
    assert (np.diff(tss) >= 0).all(), '时间戳非单调'
    np.savez_compressed(out_fp, blk=blks, ts=tss)
    gaps = np.diff(blks)
    print(f'{len(blks):,} 锚块  {blks[0]:,}→{blks[-1]:,}  最大缺口 {gaps.max()} 块,'
          f'缺口>1 比例 {(gaps > 1).mean() * 100:.1f}%  saved {out_fp}')


if __name__ == '__main__':
    av = sys.argv[1:]

    def opt(name, d=None):
        return av[av.index(name) + 1] if name in av else d
    main(av[0], opt('--fills-dir') or f'{R}/p3_data/fills', int(opt('--lo', 0)))

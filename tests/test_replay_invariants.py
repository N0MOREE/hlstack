#!/usr/bin/env python3
"""bench/micro.py 判定函数(scan_cross / reconcile)的纯逻辑测试,不需要本地重放数据。

测的是判据本身,不是引擎输出:引擎输出的正确性靠重放对真值快照对账,
对账依赖的这几个判定函数若自身有错,整条验证链就不成立。
"""
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bench'))
import micro  # noqa: E402


def _write_diffs(tmp_path, evs):
    fp = tmp_path / 'diffs_T.ndjson.gz'
    with gzip.open(fp, 'wt') as w:
        for e in evs:
            w.write(json.dumps(e) + '\n')
    return str(fp)


def _ev(block, oid, typ, side, px, sz, why='x'):
    return dict(block=block, oid=oid, type=typ, side=side, px=px, sz=sz, why=why, user='u')


# ── 穿价扫描 ──
def test_no_cross_when_book_is_sane(tmp_path):
    fp = _write_diffs(tmp_path, [
        _ev(1, 10, 'new', 'B', 99.0, 5.0),
        _ev(1, 11, 'new', 'A', 101.0, 5.0),
    ])
    assert micro.scan_cross(fp) == []


def test_cross_is_detected(tmp_path):
    fp = _write_diffs(tmp_path, [
        _ev(1, 10, 'new', 'B', 99.0, 5.0),
        _ev(1, 11, 'new', 'A', 101.0, 5.0),
        _ev(2, 12, 'new', 'A', 98.0, 5.0),          # 卖 98 < 买 99 → 穿价
    ])
    hits = micro.scan_cross(fp)
    assert len(hits) == 1 and hits[0][0] == 2
    assert hits[0][1] == 99.0 and hits[0][2] == 98.0


def test_lock_counts_as_cross(tmp_path):
    """锁定(买 == 卖)和穿价一样不可能存在于静息簿,必须一起报。"""
    fp = _write_diffs(tmp_path, [
        _ev(1, 10, 'new', 'B', 100.0, 5.0),
        _ev(1, 11, 'new', 'A', 100.0, 5.0),
    ])
    assert len(micro.scan_cross(fp)) == 1


def test_remove_clears_the_level(tmp_path):
    fp = _write_diffs(tmp_path, [
        _ev(1, 10, 'new', 'B', 100.0, 5.0),
        _ev(1, 11, 'new', 'A', 100.0, 5.0),
        _ev(2, 10, 'remove', 'B', 100.0, 5.0),
        _ev(2, 12, 'new', 'B', 99.0, 5.0),
    ])
    hits = micro.scan_cross(fp)
    assert [h[0] for h in hits] == [1]               # 只有第 1 块穿,第 2 块已解开


def test_level_liveness_is_by_order_id_not_by_float_sum(tmp_path):
    """价位存活性按「这个价位上还有没有 oid」判,不按量的浮点和判。

    按浮点和判时,~1e7 块上累积的残量漂移(>1e-9)会让一个价位永远清不空,
    报出数百万个假穿价块。部分成交只改量、不改价,不该动价位索引;只有 remove 才让 oid 离场。"""
    fp = _write_diffs(tmp_path, [
        _ev(1, 10, 'new', 'B', 100.0, 10.0),
        _ev(2, 10, 'update', 'B', 100.0, 4.0),       # 剩 4:量变了,价位仍然活着
        _ev(3, 11, 'new', 'A', 100.0, 1.0),          # 对侧同价 → 这时候确实穿
        _ev(4, 10, 'remove', 'B', 100.0, 4.0),       # 买单真的走了
        _ev(5, 12, 'new', 'A', 101.0, 1.0),
    ])
    hits = micro.scan_cross(fp)
    assert [h[0] for h in hits] == [3]


def test_reprice_moves_the_order_between_levels(tmp_path):
    """update 若带了新价格,必须把 oid 从老价位挪到新价位,否则老价位永远留个影子。"""
    fp = _write_diffs(tmp_path, [
        _ev(1, 10, 'new', 'B', 100.0, 5.0),
        _ev(2, 10, 'update', 'B', 98.0, 5.0),        # 挪到 98
        _ev(3, 11, 'new', 'A', 99.0, 1.0),           # 卖 99 > 买 98 → 不该穿
    ])
    assert micro.scan_cross(fp) == []


def test_same_oid_reused_does_not_leave_a_shadow(tmp_path):
    """引擎的 rebind 会对同一个 oid 再发一次 new。老的那张必须先摘干净。"""
    fp = _write_diffs(tmp_path, [
        _ev(1, 10, 'new', 'B', 100.0, 5.0),
        _ev(2, 10, 'new', 'B', 97.0, 5.0),           # 同号复用,挪到 97
        _ev(3, 11, 'new', 'A', 99.0, 1.0),
    ])
    assert micro.scan_cross(fp) == []


def test_seed_participates_in_the_book(tmp_path):
    seed = tmp_path / 'T.json'
    seed.write_text(json.dumps({'success': True, 'data': {
        'last_block_number': 0,
        'bids': [{'oid': 1, 'side': 'B', 'price': 100.0, 'size': 3.0}],
        'asks': [{'oid': 2, 'side': 'A', 'price': 102.0, 'size': 3.0}]}}))
    fp = _write_diffs(tmp_path, [_ev(1, 3, 'new', 'A', 100.0, 1.0)])
    assert micro.scan_cross(fp, str(seed)) == [(1, 100.0, 100.0)]
    assert micro.scan_cross(fp) == []                # 不给种子就看不见这次穿价


# ── 对账 ──
def _truth(bids, asks):
    return {'bids': [{'oid': o, 'price': p, 'size': s} for o, p, s in bids],
            'asks': [{'oid': o, 'price': p, 'size': s} for o, p, s in asks]}


def _ours(rows):
    return {'orders': [{'oid': o, 'side': sd, 'px': p, 'sz': s} for o, sd, p, s in rows]}


def test_reconcile_perfect_match():
    t = _truth([(1, 99.0, 5.0)], [(2, 101.0, 5.0)])
    o = _ours([(1, 'B', 99.0, 5.0), (2, 'A', 101.0, 5.0)])
    r = micro.reconcile(o, t)
    assert (r['miss'], r['ghost'], r['szdiff'], r['inb_total']) == (0, 0, 0, 0)
    assert r['acc'] == 1.0 and r['crossed'] is False


def test_reconcile_band_is_relative_to_truth_mid():
    """带内/带外按**真值**的中价算 —— 用我方中价会让「我方盘口错了」这件事自己把自己藏起来。"""
    t = _truth([(1, 100.0, 5.0)], [(2, 100.02, 5.0)])   # 中价 100.01
    o = _ours([(1, 'B', 100.0, 5.0), (2, 'A', 100.02, 5.0),
               (3, 'B', 99.99, 1.0),                     # 距中价 2bp → 带内
               (4, 'B', 90.0, 1.0)])                     # 距中价 1001bp → 带外
    r = micro.reconcile(o, t, band_bp=20.0)
    assert r['ghost'] == 2 and r['inb_ghost'] == 1 and r['inb_total'] == 1


def test_reconcile_counts_size_mismatch_separately():
    t = _truth([(1, 100.0, 5.0)], [(2, 100.02, 5.0)])
    o = _ours([(1, 'B', 100.0, 4.0), (2, 'A', 100.02, 5.0)])
    r = micro.reconcile(o, t)
    assert (r['miss'], r['ghost'], r['szdiff'], r['inb_sz']) == (0, 0, 1, 1)


def test_reconcile_flags_our_own_crossing():
    t = _truth([(1, 100.0, 5.0)], [(2, 100.02, 5.0)])
    o = _ours([(1, 'B', 100.0, 5.0), (2, 'A', 99.0, 5.0)])
    assert micro.reconcile(o, t)['crossed'] is True

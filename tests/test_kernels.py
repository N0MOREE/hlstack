#!/usr/bin/env python3
"""kernels.book.Book 的纯逻辑测试,不需要本地重放数据。

覆盖:new/update/remove 往返、unknown update 的认领语义(同档同 user 的合成号单被认领,
不凭空建单)、无候选时的首次露面建单、update 到 0 等同删除。
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from kernels.book import NEG, Book, _apply_plain  # noqa: E402


# ── Book(API = _apply_plain(bk, evs) 落一批事件字典)──

def _ev(t, oid, px, sz, user='0xa', side='B'):
    return {'type': t, 'oid': oid, 'side': side, 'px': px, 'sz': sz, 'user': user}


def test_book_new_update_remove_roundtrip():
    bk = Book(False)
    _apply_plain(bk, [_ev('new', 1, 100.0, 5.0)])
    assert bk.orders[1][2] == 5.0 and bk.ltot(('B', 100.0)) == 5.0
    _apply_plain(bk, [_ev('update', 1, 100.0, 2.0)])
    assert bk.orders[1][2] == 2.0 and bk.ltot(('B', 100.0)) == 2.0
    _apply_plain(bk, [_ev('remove', 1, 100.0, 0.0)])
    assert 1 not in bk.orders and not bk.lv.get(('B', 100.0))


def test_book_unknown_update_claims_synthetic_in_place():
    """认领语义:同档同 user 有合成号(>2^63)单 → unknown update 认领给它,
    单号保持合成号不换、量更新;总单数不变,不凭空造单。"""
    bk = Book(False)
    syn = NEG + 7
    _apply_plain(bk, [_ev('new', syn, 100.0, 5.0)])
    _apply_plain(bk, [_ev('update', 42, 100.0, 3.0)])
    assert syn in bk.orders and bk.orders[syn][2] == 3.0
    assert 42 not in bk.orders and len(bk.orders) == 1


def test_book_unknown_update_without_candidate_creates():
    """没得认领(别的档)→ 首次露面建单(modify 换号后的正常路径)。"""
    bk = Book(False)
    _apply_plain(bk, [_ev('update', 42, 101.0, 3.0)])
    assert bk.orders[42][2] == 3.0


def test_book_update_to_zero_deletes_order():
    bk = Book(False)
    _apply_plain(bk, [_ev('new', 1, 100.0, 5.0), _ev('update', 1, 100.0, 0.0)])
    assert 1 not in bk.orders and not bk.lv.get(('B', 100.0))

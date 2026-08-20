#!/usr/bin/env python3
"""拉一张 0xArchive L4 整簿快照,存成 seeds/plain 兼容 json —— 引擎与 MBO 导出的种子。

需要自己的 0xArchive API key(写在 ~/.0xarchive_key,一行);1 张快照 = 1 credit。
接口与重试口径和 bench/micro.py 的真值拉取同款(慢档实测 20–100 秒)。
没有 key 的替代:引擎冷启动(空簿暖场)——前段簿不完整,窗口起点后移,文档里如实标注。

用法: seed_fetch.py COIN TIMESTAMP_MS OUT.json
  TIMESTAMP_MS = 想要的种子时刻(毫秒);返回体自带 last_block_number(种子块),
  下游(mbo_export / ledger_bt)b0 会自动钳到 种子块+1。
"""
import json
import os
import sys
import time


def main(coin, ts_ms, out_fp):
    import requests
    key = open(os.path.expanduser('~/.0xarchive_key')).read().strip()
    for att in range(5):
        try:
            print(f'[0x] 拉 {coin} @ {ts_ms}(第 {att + 1} 次,慢档要 20–100 秒)…', flush=True)
            r = requests.get(f'https://api.0xarchive.io/v1/hyperliquid/orderbook/{coin}/l4',
                             params={'timestamp': ts_ms, 'limit': 300000},
                             headers={'X-API-Key': key}, timeout=300)
            r.raise_for_status()
            d = r.json()
            body = d.get('data', d)
            assert 'bids' in body and 'last_block_number' in body, f'返回体缺字段: {sorted(body)[:8]}'
            json.dump(d, open(out_fp, 'w'))
            print(f'种子块 {body["last_block_number"]}  买单 {body.get("bid_count")} 张 / '
                  f'卖单 {body.get("ask_count")} 张  saved {out_fp}', flush=True)
            return
        except Exception as ex:
            print(f'[0x] 重试 {att}: {ex}', flush=True)
            time.sleep(8)
    raise SystemExit('0x 拉不到:检查 ~/.0xarchive_key 与网络')


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3])

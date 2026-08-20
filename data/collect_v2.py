#!/usr/bin/env python3
"""从 SQD Portal 拉 Hyperliquid replica-cmds / fills。
v2: 流式落盘(不缓内存)、窗口可调、断点续传、529 退避。"""
import gzip, json, os, time, argparse
import requests

PORTAL = "https://portal.sqd.dev/datasets/{ds}/stream"
SPEC = {
    "actions": dict(
        ds="hyperliquid-replica-cmds", typ="hyperliquidReplicaCmds", key="actions",
        fields={"action": {"actionIndex": True, "user": True, "action": True,
                           "nonce": True, "vaultAddress": True, "status": True,
                           "response": True},
                "block": {"number": True, "timestamp": True}}, window=500),
    "fills": dict(
        ds="hyperliquid-fills", typ="hyperliquidFills", key="fills",
        fields={"fill": {"user": True, "coin": True, "px": True, "sz": True,
                         "side": True, "time": True, "startPosition": True,
                         "dir": True, "closedPnl": True, "oid": True, "crossed": True,
                         "fee": True, "builderFee": True, "tid": True, "cloid": True,
                         "twapId": True},
                "block": {"number": True, "timestamp": True}}, window=20000),
}

def stream_window(sp, frm, to, path, tries=10):
    """流式下载并直接写 gzip;返回 (最后块号, 记录数, 字节数)"""
    body = {"type": sp["typ"], "fromBlock": frm, "toBlock": to,
            "fields": sp["fields"], sp["key"]: [{}]}
    delay = 2.0
    for i in range(tries):
        tmp = path + ".part"
        try:
            with requests.post(PORTAL.format(ds=sp["ds"]), json=body,
                               timeout=(20, 120), stream=True) as r:
                if r.status_code != 200:
                    head = r.raw.read(200) if r.raw else b""
                    if r.status_code in (429, 500, 502, 503, 529) or b"overloaded" in head:
                        print(f"    HTTP {r.status_code},{delay:.0f}s 后重试 ({i+1}/{tries})", flush=True)
                        time.sleep(delay); delay = min(delay * 2, 60); continue
                    raise SystemExit(f"HTTP {r.status_code}: {head[:300]}")
                last, recs, nbytes, buf = frm - 1, 0, 0, b""
                with gzip.open(tmp, "wb", compresslevel=6) as out:
                    for chunk in r.iter_content(1 << 20):
                        if not chunk: continue
                        nbytes += len(chunk); out.write(chunk)
                        buf += chunk
                        *lines, buf = buf.split(b"\n")
                        for L in lines:
                            if not L.strip(): continue
                            try: o = json.loads(L)
                            except Exception: continue
                            n = o.get("header", {}).get("number")
                            if n is not None and n > last: last = n
                            recs += len(o.get(sp["key"], []))
                    if buf.strip():
                        try:
                            o = json.loads(buf)
                            n = o.get("header", {}).get("number")
                            if n is not None and n > last: last = n
                            recs += len(o.get(sp["key"], []))
                        except Exception: pass
            os.replace(tmp, path)
            return last, recs, nbytes
        except requests.exceptions.RequestException as e:
            print(f"    网络异常 {type(e).__name__},{delay:.0f}s 后重试 ({i+1}/{tries})", flush=True)
            if os.path.exists(tmp): os.remove(tmp)
            time.sleep(delay); delay = min(delay * 2, 60)
    raise SystemExit("重试耗尽")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=list(SPEC))
    ap.add_argument("--from", dest="frm", type=int, required=True)
    ap.add_argument("--to", dest="to", type=int, required=True)
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    sp = SPEC[a.kind]
    win = a.window or sp["window"]
    outdir = os.path.join(a.out, a.kind); os.makedirs(outdir, exist_ok=True)
    cur_path = os.path.join(outdir, "cursor.json")
    cur = json.load(open(cur_path)) if os.path.exists(cur_path) else {
        "next": a.frm, "bytes": 0, "records": 0, "files": 0}
    t0 = time.time(); n0 = cur["next"]
    while cur["next"] <= a.to:
        frm = cur["next"]; to = min(frm + win - 1, a.to)
        fn = os.path.join(outdir, f"{frm}-{to}.ndjson.gz")
        ts = time.time()
        last, recs, nb = stream_window(sp, frm, to, fn)
        cur["next"] = max(last + 1, to + 1)
        cur["bytes"] += nb; cur["records"] += recs; cur["files"] += 1
        json.dump(cur, open(cur_path, "w"))
        done = cur["next"] - n0; total = a.to - n0 + 1
        el = time.time() - t0
        print(f"  [{a.kind}] {frm:,}-{to:,}  {recs:>7,} 条  {nb/1e6:6.1f} MB  "
              f"{nb/1e6/max(time.time()-ts,.01):5.1f} MB/s  "
              f"总 {cur['bytes']/1e9:.2f} GB  进度 {done/total*100:5.1f}%  "
              f"ETA {el/max(done,1)*(total-done)/60:5.1f} 分", flush=True)
    print(f"[{a.kind}] 完成: {cur['records']:,} 条, {cur['bytes']/1e9:.2f} GB, "
          f"{cur['files']} 文件, {(time.time()-t0)/60:.1f} 分")

if __name__ == "__main__":
    main()

#!/bin/bash
# 全量重放(5 币 × 9.31 天,12 个语义开关),落盘逐单流水(diffs)。
#
# 开关说明:12 个语义机制开关中,默认关的显式以 --xxx 1 打开(见命令行),默认开的不传。
# 已删除的四个开关 ro-unknown-zero / syn-strict / stp-taker / stp-depth 的默认行为
# 已固化进引擎,命令行不再传;传入会被引擎以 rc=2 拒收。
#
# 落盘流水带 ai(actionIndex):每条变更能追回原始 JSON 里那一条动作。
# ai = -1 表示它由成交流或块末仲裁造成,本来就不对应任何单个动作。
set -e
# 工作目录 = 全量数据根目录(含 data/ replay_coins.txt seeds/plain snapshot_blocks.txt)
cd "${HL_REPLAY_ROOT:?请设置 HL_REPLAY_ROOT=全量数据根目录}"
# 引擎二进制:仓内构建(engine/,cargo build --release)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN="$REPO_ROOT/engine/target/release/hl-replay"
OUT=replay_out
mkdir -p $OUT
"$BIN" \
  --data data --multi replay_coins.txt --seed-dir seeds/plain --seed-block 0 \
  --out $OUT --workers 8 --warmup-blocks 0 --snap-at snapshot_blocks.txt \
  --filled-rest 1 --synthetic-modify 1 --trust-gone 1 --modify-err-cancel 1 \
  --ro-proof 1 --ro-dedup 1 --rebind-emit 1 --q-order 1 \
  >> $OUT/run.log 2>&1
echo "rc=$? 完成于 $(date -Is)" >> $OUT/run.log

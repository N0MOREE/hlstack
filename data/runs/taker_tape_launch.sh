#!/bin/bash
# 吃单带(taker tape):与 replay_launch.sh 完全同配置(5 币 × 9.31 天,12 个语义开关)重跑一遍,
# 只为落 taker_*.ndjson.gz(taker 成交序列,MBO+ 导出时回填 ai 用)。
# 已删除的四个开关 ro-unknown-zero / syn-strict / stp-taker / stp-depth 的默认行为
# 已固化进引擎,命令行不再传;传入会被引擎以 rc=2 拒收。
# --no-diffs:diffs 已验证与 replay_launch.sh 的产物逐位相同(tape 是纯旁路),不重写 12GB。
set -e
# 工作目录 = 全量数据根目录(含 data/ replay_coins.txt seeds/plain)
cd "${HL_REPLAY_ROOT:?请设置 HL_REPLAY_ROOT=全量数据根目录}"
# 引擎二进制:仓内构建(engine/,cargo build --release)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN="$REPO_ROOT/engine/target/release/hl-replay"
OUT=tape_out
mkdir -p $OUT
"$BIN" \
  --data data --multi replay_coins.txt --seed-dir seeds/plain --seed-block 0 \
  --out $OUT --workers 8 --warmup-blocks 0 --no-diffs 1 --taker-tape 1 \
  --filled-rest 1 --synthetic-modify 1 --trust-gone 1 --modify-err-cancel 1 \
  --ro-proof 1 --ro-dedup 1 --rebind-emit 1 --q-order 1 \
  >> $OUT/run.log 2>&1
echo "rc=$? 完成于 $(date -Is)" >> $OUT/run.log

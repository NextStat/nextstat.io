#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FIXTURE="$ROOT_DIR/tests/fixtures/rntuple_bench_large_primitive.root"
ITERS=5
CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT_DIR/target}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fixture)
      FIXTURE="$2"
      shift 2
      ;;
    --iters)
      ITERS="$2"
      shift 2
      ;;
    *)
      echo "unknown arg: $1" >&2
      echo "usage: $0 [--fixture /abs/path/to/file.root] [--iters N]" >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "$FIXTURE" ]]; then
  echo "fixture not found: $FIXTURE" >&2
  exit 2
fi

if ! command -v cargo >/dev/null 2>&1; then
  echo "cargo not found in PATH" >&2
  exit 2
fi

if ! command -v root >/dev/null 2>&1; then
  echo "root not found in PATH" >&2
  exit 2
fi

echo "[rntuple-bench] fixture=$FIXTURE iters=$ITERS"
echo "[rntuple-bench] cargo_target_dir=$CARGO_TARGET_DIR"

NS_OUT="$(
  CARGO_TARGET_DIR="$CARGO_TARGET_DIR" \
  NS_ROOT_RNTUPLE_PERF_CASES="$FIXTURE" \
  NS_ROOT_RNTUPLE_PERF_ITERS="$ITERS" \
  NS_ROOT_RNTUPLE_PERF_MAX_AVG_MS="100000" \
  NS_ROOT_RNTUPLE_PERF_MIN_SUITES_PER_SEC="0" \
  cargo test -p ns-root --test rntuple_perf_gate rntuple_decode_perf_gate_baseline --release -- --ignored --nocapture 2>&1
)"
echo "$NS_OUT"

NS_METRICS="$(echo "$NS_OUT" | sed -n 's/.*avg_ms=\([0-9.]*\).*entries_per_sec=\([0-9.]*\).*/\1 \2/p' | tail -1)"
if [[ -z "$NS_METRICS" ]]; then
  echo "failed to parse ns-root benchmark output" >&2
  exit 1
fi
NS_AVG_MS="$(echo "$NS_METRICS" | awk '{print $1}')"
NS_EPS="$(echo "$NS_METRICS" | awk '{print $2}')"

ROOT_MACRO="$ROOT_DIR/scripts/benchmarks/rntuple_root_read_primitive_bench.C"
ROOT_OUT="$(
  root -l -b -q "${ROOT_MACRO}(\"${FIXTURE}\",${ITERS})" 2>&1
)"
echo "$ROOT_OUT"

ROOT_METRICS="$(echo "$ROOT_OUT" | sed -n 's/.*root_rntuple_bench entries=[0-9]* iters=[0-9]* avg_ms=\([0-9.e+-]*\) entries_per_sec=\([0-9.e+-]*\).*/\1 \2/p' | tail -1)"
if [[ -z "$ROOT_METRICS" ]]; then
  echo "failed to parse ROOT benchmark output" >&2
  exit 1
fi
ROOT_AVG_MS="$(echo "$ROOT_METRICS" | awk '{print $1}')"
ROOT_EPS="$(echo "$ROOT_METRICS" | awk '{print $2}')"

RATIO="$(awk -v ns="$NS_EPS" -v root="$ROOT_EPS" 'BEGIN { if (root == 0) { print "inf"; } else { printf "%.3f", ns / root; } }')"

echo
echo "[rntuple-bench] summary"
echo "ns-root avg_ms=$NS_AVG_MS entries_per_sec=$NS_EPS"
echo "ROOT    avg_ms=$ROOT_AVG_MS entries_per_sec=$ROOT_EPS"
echo "ratio(ns-root/root)=$RATIO"

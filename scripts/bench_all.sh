#!/usr/bin/env bash
# Run the attention benchmark on every built-in adapter.
#
# Usage:
#   ./scripts/bench_all.sh                 # uses GPU 2 by default
#   CUDA_VISIBLE_DEVICES=0 ./scripts/bench_all.sh
#   ./scripts/bench_all.sh --seqlens 2048,4096,8192   # any extra args go to attn-bench
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

: "${CUDA_VISIBLE_DEVICES:=2}"
export CUDA_VISIBLE_DEVICES

echo "[bench_all] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[bench_all] running --all built-in adapters"

exec uv run --project packages/attn-bench attn-bench --all "$@"

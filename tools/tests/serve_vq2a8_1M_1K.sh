#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# EP2 decode graph service; pass the checkpoint and built native artifacts explicitly.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

assets=()
if [[ -n "${GS32_OUTPUT:-}" ]]; then assets+=(--model "$GS32_OUTPUT"); fi
if [[ -n "${GS32_EP_LIBRARY:-}" ]]; then assets+=(--gs32-library "$GS32_EP_LIBRARY"); fi
if [[ -n "${GS32_EP_BINARY:-}" ]]; then assets+=(--gs32-binary "$GS32_EP_BINARY"); fi

# The engine validates that this manual KV budget can accommodate the requested context.
exec python3 -u tools/serve_vq2a8.py "${assets[@]}" \
  --ep-size 2 \
  --gs32-execution full_graph \
  --max-model-len 1049600 \
  --max-num-batched-tokens 128 \
  --kv-cache-mib 16384 \
  --reserve-gib 16 \
  --host 127.0.0.1 \
  --port 8000 \
  "$@"

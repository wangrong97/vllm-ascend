#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Run on the Linux Ascend host. Pass model/native artifacts and device selection explicitly.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER-1}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

# Trailing arguments override defaults through the existing validated launcher.
exec python3 -u "$repo_root/tools/serve_vq2a8.py" \
  --parallel-mode tp-ep --ep-size 2 \
  --gs32-execution full_graph --gs32-fusion none \
  --max-model-len 8192 --max-num-seqs 64 --max-num-batched-tokens 8192 \
  --kv-cache-mib 16384 --reserve-gib 16 \
  --tokenizer-mode deepseek_v4 --host 127.0.0.1 --port 8003 \
  "$@"

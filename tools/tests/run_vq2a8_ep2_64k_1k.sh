#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# 65536 input + 1024 output tokens; requires an existing EP2 full_graph service.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 -u "$script_dir/run_vq2a8_ep2_long_context_latency.py" "$@" --cases 64k

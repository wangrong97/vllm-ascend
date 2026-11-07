# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read the runtime contract without importing the Ascend/Torch package stack."""

import runpy
from pathlib import Path


def source_contract():
    path = Path(__file__).resolve().parents[1] / "vllm_ascend/quantization/methods/w2a8/vq2a8_gs32_contract.py"
    return runpy.run_path(str(path))

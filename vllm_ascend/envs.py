#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# This file is mainly Adapted from vllm-project/vllm/vllm/envs.py
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
from collections.abc import Callable
from typing import Any


def _bool_env(name: str, default: bool) -> bool:
    """Parse a boolean environment variable as ``1`` or ``0``.

    Only ``1`` and ``0`` are accepted (surrounding whitespace is ignored).
    Unset or empty values fall back to ``default``, which keeps
    ``export VAR=`` from crashing. Any other value raises ValueError naming the
    variable, instead of failing inside ``int()`` with an opaque message.
    """
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    value = raw.strip()
    if value == "1":
        return True
    if value == "0":
        return False
    raise ValueError(f"Invalid value for environment variable {name}: {raw!r}. Expected '0' or '1'.")


# The begin-* and end* here are used by the documentation generator
# to extract the used env vars.

# begin-env-vars-definition


env_variables: dict[str, Callable[[], Any]] = {
    # max compile thread number for package building. Usually, it is set to
    # the number of CPU cores. If not set, the default value is None, which
    # means all number of CPU cores will be used.
    "MAX_JOBS": lambda: os.getenv("MAX_JOBS", None),
    # The build type of the package. It can be one of the following values:
    # Release, Debug, RelWithDebugInfo. If not set, the default value is Release.
    "CMAKE_BUILD_TYPE": lambda: os.getenv("CMAKE_BUILD_TYPE"),
    # Whether to compile custom kernels. If not set, the default value is True.
    # If set to False, the custom kernels will not be compiled.
    # This configuration option should only be set to False when running UT
    # scenarios in an environment without an NPU. Do not set it to False in
    # other scenarios.
    "COMPILE_CUSTOM_KERNELS": lambda: _bool_env("COMPILE_CUSTOM_KERNELS", True),
    # The CXX compiler used for compiling the package. If not set, the default
    # value is None, which means the system default CXX compiler will be used.
    "CXX_COMPILER": lambda: os.getenv("CXX_COMPILER", None),
    # The C compiler used for compiling the package. If not set, the default
    # value is None, which means the system default C compiler will be used.
    "C_COMPILER": lambda: os.getenv("C_COMPILER", None),
    # The version of the Ascend chip. It's used for package building.
    # If not set, we will query chip info through `npu-smi`.
    # Please make sure that the version is correct.
    "SOC_VERSION": lambda: os.getenv("SOC_VERSION", None),
    # If set, vllm-ascend will print verbose logs during compilation
    "VERBOSE": lambda: _bool_env("VERBOSE", False),
    # The home path for CANN toolkit. If not set, the default value is
    # /usr/local/Ascend/ascend-toolkit/latest
    "ASCEND_HOME_PATH": lambda: os.getenv("ASCEND_HOME_PATH", None),
    # The path for HCCL library, it's used by pyhccl communicator backend. If
    # not set, the default value is libhccl.so.
    "HCCL_SO_PATH": lambda: os.getenv("HCCL_SO_PATH", None),
    # The version of vllm is installed. This value is used for developers who
    # installed vllm from source locally. In this case, the version of vllm is
    # usually changed. For example, if the version of vllm is "0.9.0", but when
    # it's installed from source, the version of vllm is usually set to "0.9.1".
    # In this case, developers need to set this value to "0.9.0" to make sure
    # that the correct package is installed.
    "VLLM_VERSION": lambda: os.getenv("VLLM_VERSION", None),
    # Whether to anbale dynamic EPLB
    "DYNAMIC_EPLB": lambda: os.getenv("DYNAMIC_EPLB", "false").lower(),
    # Control the aclrtMemcpyBatchAsync compile path for KV cache offloading.
    # "1": force enable, "0": force disable, None: auto-detect from CANN headers.
    "VLLM_ASCEND_ENABLE_BATCH_MEMCPY": lambda: os.getenv("VLLM_ASCEND_ENABLE_BATCH_MEMCPY", None),

    # Kpool indexer MXFP4 伪量化(A4/C4 数值验证): 1 时 q 打分前与 k 池化写缓存前
    # 各过一次 MXFP4(e2m1+per-32 E8M0) 量化->反量化往返, 存储/算子路径不变
    # (bf16 或 fp8 均可叠加)。默认 0 关闭。非敏感变量。
    "VLLM_ASCEND_KPOOL_MXFP4_SIM": lambda: _bool_env("VLLM_ASCEND_KPOOL_MXFP4_SIM", False),

    # Kpool indexer MXFP8 伪量化: 1 时 q/k 过 MXFP8(e4m3 + per-32 E8M0) 量化->反量化往返,
    # 注入点与 MXFP4_SIM 相同; 与 MXFP4_SIM 同时开启时 MXFP4 优先。默认 0。
    "VLLM_ASCEND_KPOOL_MXFP8_SIM": lambda: _bool_env("VLLM_ASCEND_KPOOL_MXFP8_SIM", False),

    # Kpool indexer Hadamard 旋转预处理: 1 时在 indexer 量化(真 C8 / MXFP4 / MXFP8 伪量化)
    # 前对 q/k 施加 128 维归一化 Hadamard 旋转, 抑制离群值。真量化路径在旋转域存取/打分
    # (正交不变性保持点积); 伪量化路径旋转->往返->逆旋转回原域。默认 0。
    "VLLM_ASCEND_KPOOL_HADAMARD_PRE": lambda: _bool_env("VLLM_ASCEND_KPOOL_HADAMARD_PRE", False),

    # Kpool indexer OSCAR 旋转(数据驱动协方差拟合, 见 tools/glm53flash/oscar_indexer_rotation):
    # =1 时 q/k/尾池 改用 per-layer OSCAR 矩阵(替换 Hadamard; 两者同开时 OSCAR 优先)。
    # checkpoint 由 tools/glm53flash/oscar_indexer_rotation/oscar_indexer_rotation.py 拟合。
    "VLLM_ASCEND_KPOOL_OSCAR_ROTATION": lambda: _bool_env("VLLM_ASCEND_KPOOL_OSCAR_ROTATION", False),
    "VLLM_ASCEND_KPOOL_OSCAR_CKPT": lambda: os.getenv("VLLM_ASCEND_KPOOL_OSCAR_CKPT", ""),
}

# end-env-vars-definition


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in env_variables:
        return env_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(env_variables.keys())

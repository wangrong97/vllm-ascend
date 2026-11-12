# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-UT fixtures for the mixed-precision EPLB package.

Importing ``vllm_ascend`` on an unbuilt source tree trips the vLLM plugin
discovery, which does ``from vllm_ascend import _build_info`` while the
package is still partially initialized (build artifact, circular). We
pre-register a stub submodule so the package import completes on CPU CI
runners without a compiled extension. The stub never shadows a real build:
an installed tree resolves first because it appears earlier on sys.path.
"""

import os
import sys
import types

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

_STUB_NAME = "vllm_ascend._build_info"
try:
    __import__(_STUB_NAME)
except Exception:
    stub = types.ModuleType(_STUB_NAME)
    stub.__version__ = "0.0.0+ut"
    sys.modules[_STUB_NAME] = stub

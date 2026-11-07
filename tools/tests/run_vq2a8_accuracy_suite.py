#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Evaluate GPQA Diamond, AIME24 and AIME25 against an existing VQ2A8 service."""

import sys

from run_evalscope_accuracy import main

if __name__ == "__main__":
    raise SystemExit(main(["vq2a8", *sys.argv[1:]]))

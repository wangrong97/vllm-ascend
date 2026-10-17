# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Select complete EP2 local banks for capacity tests without loading an LLM."""

EP_SIZE = 2


def select_bank(config, ep_rank, layers):
    total = config["n_routed_experts"]
    if type(total) is not int or total < 2 * 4 or total % EP_SIZE or ep_rank not in range(EP_SIZE):
        raise ValueError("Require a valid EP2 rank and an even expert count of at least eight.")
    if config["quantization_config"]["format"] != "vq2_mixed_gs32_source":
        raise ValueError("Require a source-v3 GS32 checkpoint.")
    local = total // EP_SIZE
    start = ep_rank * local
    # Prefer a complete non-hash layer; do not silently invent missing experts.
    keys = sorted(layers, key=lambda key: (int(key) < config.get("num_hash_layers", 0), int(key)))
    for key in keys:
        ids = tuple(sorted(layers[key]["gate_up"]))
        if len(ids) == total and set(ids) == set(range(total)):
            positions = tuple(pos for pos, expert in enumerate(ids) if start <= expert < start + local)
            return int(key), positions, tuple(ids[pos] - start for pos in positions), local
    raise ValueError("No complete expert layer is available for the EP2 bank diagnostic.")

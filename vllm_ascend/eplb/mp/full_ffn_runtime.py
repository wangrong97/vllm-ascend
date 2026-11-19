# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Production wiring between MP-EPLB and the full-FFN GS32 bank.

The migrator core is transport-agnostic: it emits :class:`P2POp` bundles
(per-field logical names) and calls a ``CommBackend`` plus, on commit, a
``device_index_writer`` hook. This module provides the full-FFN pieces:

``FullFFNBankSlicer``  — map ops to packed bank byte ranges: the complete
FFN bank stores each projection as one flat uint8 weight blob with
per-slot variable-length offsets, one flat table blob with fixed slots and
one row_scale row per slot, so transfers move byte ranges, not tensors.

``FullFFNStaging``     — staging bundles (``options.n_stage``) that receive
incoming bytes; ``install`` copies a staged bundle into its target slot at
commit time (P1 bytes), before index refresh (P2).

``MPRuntime``          — per-rank bridge owning the controller, the layer
registry and the device index sync after each committed layer.
"""

from __future__ import annotations

import torch

from .migrator import CommBackend, P2POp

# Old bundle field names -> packed bank member (identity perm: no bytes).
_FIELD_TO_MEMBER = {"packed_zn": "weight", "pair_lut": "table", "row_scale_bits": "row_scale", "perm": None,
                    "mxfp4_weight": "weight", "mxfp4_scale": "table"}
PROJECTION_TO_NATIVE = {"w13": "gate", "w2": "down"}


class FullFFNBankSlicer:
    """Resolve migration ops into packed bank slice descriptors."""

    def __init__(self, bank_getter, placement_for_rank):
        """``bank_getter(layer)`` -> packed bank dict; ``placement_for_rank(rank)``
        -> the host :class:`BankSlotTable` mirror used to locate source slots."""
        self._bank_getter = bank_getter
        self._placement_for_rank = placement_for_rank

    def send_slices(self, layer: int, ops: list[P2POp], rank: int):
        """Outgoing byte ranges for ``expert_id`` resident on ``rank``."""
        bank = self._bank_getter(layer)
        table = self._placement_for_rank(rank)
        descriptors = []
        for op in ops:
            member = _FIELD_TO_MEMBER.get(op.field)
            if member is None:
                continue
            found = table.slot_of(op.expert_id, op.projection)
            if found is None:
                raise ValueError(f"send op for absent expert {op.expert_id} on rank {rank}.")
            _qtype, slot = found
            descriptors.append(self._slice(bank, op.projection, member, slot))
        return descriptors

    def recv_shapes(self, layer: int, ops: list[P2POp]):
        """Shape/allocinfo descriptors for staging buffers (one per op)."""
        bank = self._bank_getter(layer)
        descriptors = []
        for op in ops:
            member = _FIELD_TO_MEMBER.get(op.field)
            if member is None:
                continue
            descriptors.append(self._shape(bank, op.projection, member, op.quant_type))
        return descriptors

    def _slice(self, bank, projection, member, slot):
        native = PROJECTION_TO_NATIVE[projection]
        tensor = bank["fields"][f"{native}_{member}"]
        if member == "row_scale":
            return {"tensor": tensor[slot], "rows": 1, "bytes": tensor[slot].numel() * tensor.element_size()}
        native_key = {"w13": "gate_up", "w2": "down"}[projection]
        start, end = bank["weight_offsets"][native_key][slot], bank["weight_offsets"][native_key][slot + 1]
        return {"tensor": tensor[start:end], "rows": 0, "bytes": end - start}

    def _shape(self, bank, projection, member, quant_type):
        geometry = bank["geometry"]
        kind = {"w13": "gate_up", "w2": "down"}[projection]
        rows, cols = geometry["rows"][kind], None
        if member == "row_scale":
            return {"rows": 1, "cols": rows}
        # Weight blobs are variable-length per quant type; table blobs are
        # fixed-size per slot (rows * cols / SCALE_GROUP with cols from the
        # packed gate geometry: cols == gate rows // 2 for down).
        gate_rows = geometry["rows"]["gate_up"]
        table_bytes = rows * (gate_rows // 2) // 32
        if member == "weight":
            return {"rows": 0, "weight_bytes": geometry["slot_bytes"][(kind, quant_type)], "table_bytes": table_bytes}
        return {"rows": 0, "weight_bytes": 0, "table_bytes": table_bytes}


class FullFFNStaging:
    """Per-layer staging bundles sized once geometry is bound."""

    def __init__(self, num_stages: int):
        self.num_stages = max(1, int(num_stages))
        self._buffers: dict[int, list[dict[str, torch.Tensor]]] = {}

    def bind_layer(self, layer: int, shapes, device):
        buffers = []
        for shape in shapes:
            entry = {}
            if shape["rows"]:
                entry["row_scale"] = torch.empty(shape["rows"], shape["cols"], dtype=torch.int64, device=device)
            else:
                entry["weight"] = torch.empty(shape["weight_bytes"], dtype=torch.uint8, device=device)
                entry["table"] = torch.empty(shape["table_bytes"], dtype=torch.uint8, device=device)
            buffers.append(entry)
        self._buffers[layer] = buffers * self.num_stages

    def slot(self, layer: int, staging_index: int):
        pool = self._buffers[layer]
        return pool[staging_index % len(pool)] if pool else None

    def install(self, layer: int, staging_index: int, bank, projection: str, slot: int):
        """P1: copy one staged bundle into its target packed slot."""
        staged = self.slot(layer, staging_index)
        native = PROJECTION_TO_NATIVE[projection]
        offsets = bank["weight_offsets"][{"w13": "gate_up", "w2": "down"}[projection]]
        table = bank["fields"][f"{native}_table"]
        table_bytes = table.numel() // len(bank["expert_ids"])  # fixed-size slots
        for member, tensor in staged.items():
            if member == "row_scale":
                bank["fields"][f"{native}_row_scale"][slot].copy_(tensor)
            elif member == "weight":
                target = bank["fields"][f"{native}_weight"]
                target[offsets[slot] : offsets[slot + 1]].copy_(tensor)
            else:
                table[slot * table_bytes : (slot + 1) * table_bytes].copy_(tensor)


class HCCLCommBackend(CommBackend):
    """Issue bundle-slice transfers over HCCL batch isend/irecv."""

    def __init__(self, group, slicer: FullFFNBankSlicer, staging: FullFFNStaging, rank: int):
        self._group = group
        self._slicer = slicer
        self._staging = staging
        self._rank = rank

    def issue(self, ops, *, layer: int | None = None, recv_ops: list[P2POp] | None = None):
        import torch.distributed as dist

        if layer is None:
            raise ValueError("HCCLCommBackend.issue requires the migrating layer.")
        p2p = []
        for descriptor, op in zip(self._slicer.send_slices(layer, ops, self._rank), ops):
            view = descriptor["tensor"].contiguous()
            p2p.append(dist.P2POp(dist.isend, view, op.peer_rank, self._group))
        for op in recv_ops or []:
            member = _FIELD_TO_MEMBER.get(op.field)
            if member is None:
                continue
            staged = self._staging.slot(layer, op.staging_index)
            target = staged.get("row_scale") if member == "row_scale" else staged.get(member)
            p2p.append(dist.P2POp(dist.irecv, target, op.peer_rank, self._group))
        return dist.batch_isend_irecv(p2p) if p2p else None

    def wait(self, handle) -> None:
        for req in handle or ():
            req.wait()


class MPRuntime:
    """Per-rank bridge: controller <-> GS32ExpertWeights layers."""

    def __init__(self, options, layers, *, ep_size: int, ep_rank: int, quant_types, comm=None,
                 plan_only: bool = False):
        from .controller import MPEplbController

        first = next(iter(layers.values()))
        self.layers = layers
        self.controller = MPEplbController(
            options,
            num_experts=first.num_experts * ep_size,
            ep_size=ep_size,
            ep_rank=ep_rank,
            quant_types=quant_types,
            num_layers=len(layers),
            comm=comm,
            plan_only=plan_only,
        )
        self.controller.executor.device_index_writer = self._sync_device_indexes
        # Local-slot -> global-expert mapping per layer (identical across
        # layers under the linear baseline; rebuilt on commit via the host
        # mirror). Needed because moe_load is indexed by LOCAL slot while the
        # controller's window is keyed by GLOBAL logical expert id.
        self._local_to_global: dict[int, list[int]] = {
            layer: [g for g, _s in sorted(weights.expert_slots.items(), key=lambda kv: kv[1])]
            for layer, weights in layers.items()
        }

    def _sync_device_indexes(self, rank: int, quant_type: str | None = None) -> None:
        """Rebuild device index tensors from the host placement mirror (P2)."""
        if rank != self.controller.ep_rank:
            return
        table = self.controller.placement[rank]
        for weights in self.layers.values():
            weights.full_ffn_expert_ids.fill_(-1)
            weights.full_ffn_logical_slots.fill_(-1)
            weights.resident_mask.fill_(False)
            for qtype in ("vq2", "mxfp4"):
                for slot, expert in enumerate(table.slots[("w13", qtype)]):
                    if expert < 0:
                        continue
                    weights.full_ffn_expert_ids[slot] = expert
                    weights.full_ffn_logical_slots[expert] = slot
                    weights.resident_mask[expert] = True

    def on_step_end(self) -> None:
        """Feed per-layer logical row counters, then advance the slow ring.

        Local ``moe_load`` rows are indexed by LOCAL slot; translate to GLOBAL
        logical expert ids via ``expert_slots`` before recording (the original
        ``enumerate`` version wrongly reported every rank's experts as 0..N).
        """
        window = {}
        for layer, weights in self.layers.items():
            if weights.moe_load is None:
                continue
            rows = weights.moe_load.tolist()
            global_ids = self._local_to_global[layer]
            window[layer] = {global_ids[i]: rows[i] for i in range(len(rows)) if i < len(global_ids)}
            weights.moe_load.zero_()
        for layer, rows in window.items():
            self.controller.record_step({expert: {layer: count} for expert, count in rows.items()})
        self.controller.on_step_end()

    def stacked_local_loads(self) -> "torch.Tensor | None":
        """Stack every layer's local moe_load into one [num_layers, local] tensor.

        Batches the per-layer device buffers so the driver can ship the whole
        window with a single collective instead of one sync per layer.
        Returns None when accounting is disabled (moe_load is None).
        """
        bufs = [w.moe_load for w in self.layers.values() if w.moe_load is not None]
        if not bufs:
            return None
        stacked = torch.stack(bufs)
        for w in self.layers.values():
            if w.moe_load is not None:
                w.moe_load.zero_()
        return stacked

    def feed_global_window(self, global_rows: "list[list[int]]") -> None:
        """Feed one stacked window of GLOBAL per-expert rows into the controller.

        ``global_rows[layer][global_expert] = rows`` — typically produced by
        all-gathering ``stacked_local_loads()`` across the EP group.
        """
        for layer, rows in enumerate(global_rows):
            self.controller.record_step(
                {expert: {layer: count} for expert, count in enumerate(rows) if count > 0}
            )
        self.controller.on_step_begin()
        self.controller.on_step_end()

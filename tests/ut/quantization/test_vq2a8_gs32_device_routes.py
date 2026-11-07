# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-route wrapper contracts, with no emulated NPU execution."""

import ast
import importlib
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.ut.quantization.test_vq2a8_gs32_fused_activation import tagged


def route_kernel_source():
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/triton/activation/vq2a8_gs32_routes.py"
    module = ast.parse(source.read_text(encoding="utf-8"))
    kernel = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "_prefill_metadata_kernel"
    )
    return source, kernel


def route_reference(
    group, group_type, expert_ids, logical_slots, rows, *, allow_padding=False, subset=False, max_count=128
):
    """Independent Python-integer oracle; it never executes device code."""
    counts = list(group) if group_type == 1 else [last - first for first, last in zip([0, *group], group)]
    valid = all(0 <= value <= rows for value in group)
    valid = valid and all(0 <= count <= (rows if max_count is None else max_count) for count in counts)
    valid = valid and (sum(counts) <= rows if allow_padding else sum(counts) == rows)
    for logical, slot in enumerate(logical_slots):
        valid = valid and (
            (slot == -1 and (counts[logical] == 0 or subset))
            or (0 <= slot < len(expert_ids) and expert_ids[slot] == logical)
        )
    for slot, logical in enumerate(expert_ids):
        valid = valid and 0 <= logical < len(counts) and logical_slots[logical] == slot
    if not valid:
        return [0] * len(expert_ids), [0] * rows, [-1] * rows, 0
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)
    physical = [counts[logical] for logical in expert_ids]
    source = [row for logical in expert_ids for row in range(offsets[logical], offsets[logical + 1])]
    live_rows = len(source)
    inverse = [-1] * rows
    for physical_row, logical_row in enumerate(source):
        inverse[logical_row] = physical_row
    source += [-1] * (rows - live_rows)
    return physical, source, inverse, 1


def route_inputs():
    return (
        tagged(torch.arange(3 * 64).reshape(3, 64).bfloat16()),
        tagged(torch.tensor([1, 0, 2, 0], dtype=torch.int64)),
        1,
        tagged(torch.tensor([2, 0, 3], dtype=torch.int64)),
        tagged(torch.tensor([1, -1, 0, 2], dtype=torch.int64)),
    )


@pytest.mark.parametrize("group_type", [0, 1])
@pytest.mark.parametrize(
    "counts",
    [
        [0, 0, 0, 0, 0],
        [2, 0, 3, 0, 1],
        [0, 0, 0, 0, 8],
        [1, 1, 1, 0, 0],
        [-1, 0, 2, 0, 0],
        [9, 0, 0, 0, 0],
        [2**62, 0, 2**62, 0, 0],
    ],
)
def test_identity_register_arithmetic_matches_independent_generic_routes(group_type, counts):
    experts, slots, capacity = [0, 2, 4], [0, -1, 1, -1, 2], 8
    groups = counts if group_type == 1 else [sum(counts[: i + 1]) for i in range(len(counts))]
    # Construct overflowing prefixes as signed int64, as device cumsum would.
    groups = [(value + 2**63) % 2**64 - 2**63 for value in groups]
    expected_counts, gather, inverse, valid = route_reference(
        groups, group_type, experts, slots, capacity, allow_padding=True, max_count=None
    )
    actual = prefill_register_expressions(
        groups, group_type, experts, slots, capacity, allow_padding=True, identity=True
    )
    assert bool(actual["valid"]) == bool(valid)
    assert actual["safe_counts"][: len(experts)].tolist() == expected_counts
    if valid:
        live_rows = sum(expected_counts)
        assert gather == inverse == [*range(live_rows), *([-1] * (capacity - live_rows))]


@pytest.mark.parametrize("experts,slots", [([2, 0], [1, -1, 0]), ([0, 0], [0, -1, -1]), ([0, 2], [1, -1, 0])])
def test_identity_register_arithmetic_rejects_reorder_duplicate_or_noninverse_mapping(experts, slots):
    actual = prefill_register_expressions([2, 0, 3], 1, experts, slots, 8, allow_padding=True, identity=True)
    assert not bool(actual["valid"])
    assert not actual["safe_counts"].count_nonzero()


def test_identity_requires_all_live_experts_while_generic_subset_keeps_routing():
    physical, gather, _, valid = route_reference([2, 3, 1], 1, [2, 0], [1, -1, 0], 8, allow_padding=True, subset=True)
    assert valid and physical == [1, 2] and gather[:3] == [5, 0, 1]
    actual = prefill_register_expressions([2, 3, 1], 1, [0, 2], [0, -1, 1], 8, allow_padding=True, identity=True)
    assert not bool(actual["valid"])


def test_identity_wrapper_forwards_only_compact_metadata(routes, monkeypatch):
    hidden, groups, group_type, _, _ = route_inputs()
    experts = tagged(torch.tensor([0, 2, 3], dtype=torch.int64))
    slots = tagged(torch.tensor([0, -1, 1, 2], dtype=torch.int64))
    retained = []
    result = object()

    def launch(*received, **options):
        assert all(
            actual is expected
            for actual, expected in zip(received, (hidden, groups, group_type, experts, slots), strict=True)
        )
        assert options == {"allow_padding": True, "retained": retained}
        return result

    monkeypatch.setattr(routes, "prepare_identity_route_metadata", launch)
    assert (
        routes.prepare_identity_metadata(
            hidden, groups, group_type, experts, slots, allow_padding=True, retained=retained
        )
        is result
    )
    with pytest.raises(TypeError):
        routes.prepare_identity_metadata(hidden, groups, group_type, experts, slots, subset=True)


@pytest.fixture
def routes(monkeypatch):
    monkeypatch.setattr(torch, "npu", SimpleNamespace(device=lambda device: nullcontext()), raising=False)
    return importlib.import_module("vllm_ascend.ops.fused_moe.vq2a8_gs32_routes")


def prefill_register_expressions(
    group, group_type, experts, slots, rows, *, allow_padding=False, subset=False, identity=False
):
    """Evaluate production integer register arithmetic; this is not NPU execution."""
    source, _ = route_kernel_source()
    module = ast.parse(source.read_text(encoding="utf-8"))
    kernel = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "_prefill_metadata_kernel"
    )
    gather = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "_gather_int64_exact"
    )
    gather.decorator_list = []
    count_sum = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "_sum_int64_counts"
    )
    count_sum.decorator_list = []

    class RegisterOnly(ast.NodeTransformer):
        def visit_Assign(self, node):
            # Inputs stand in for affine GM loads, with their real padded shape.
            if isinstance(node.targets[0], ast.Name) and node.targets[0].id in {
                "raw",
                "previous",
                "slots",
                "expert",
                "previous_expert",
            }:
                return None
            return node

        def visit_Expr(self, node):
            return None  # Device stores are checked through the resulting registers.

    statements = RegisterOnly().visit(ast.Module(body=kernel.body, type_ignores=[])).body
    logical_block = 1 << (len(group) - 1).bit_length()
    expert_block = 1 << (len(experts) - 1).bit_length()

    def gather_float(values, indices, axis):
        assert values.dtype == torch.float32, "Ascend register gather rejects integer source tensors"
        return torch.index_select(values, axis, indices.long())

    namespace = {
        "tl": SimpleNamespace(
            arange=torch.arange,
            full=lambda shape, value, dtype: torch.full(shape, value, dtype=dtype),
            static_range=range,
            gather=gather_float,
            where=torch.where,
            min=lambda value, axis: value.amin(dim=axis),
            sum=lambda value, axis: value.sum(dim=axis),
            cumsum=lambda value, axis: value.cumsum(dim=axis),
            minimum=lambda value, bound: value.clamp_max(bound),
            maximum=lambda value, bound: value.clamp_min(bound),
            int32=torch.int32,
            int64=torch.int64,
            float32=torch.float32,
            constexpr=int,
        ),
        "ROWS": rows,
        "EXPERTS": len(experts),
        "LOGICAL": len(group),
        "LOGICAL_BLOCK": logical_block,
        "EXPERT_BLOCK": expert_block,
        "ROW_BLOCK": 256,
        "GROUP_TYPE": group_type,
        "ALLOW_PADDING": allow_padding,
        "SUBSET": subset,
        "IDENTITY": identity,
        "previous_expert": torch.tensor([-1, *experts[:-1]] + [-1] * (expert_block - len(experts)), dtype=torch.int64),
        "raw": torch.tensor(group + [0] * (logical_block - len(group)), dtype=torch.int64),
        "previous": torch.tensor([0, *group[:-1]] + [0] * (logical_block - len(group)), dtype=torch.int64),
        "slots": torch.tensor(slots + [-1] * (logical_block - len(slots)), dtype=torch.int64),
        "expert": torch.tensor(experts + [0] * (expert_block - len(experts)), dtype=torch.int64),
    }
    constants = [
        node
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id in {"INT64_GATHER_BITS", "INT64_SUM_BITS"}
            for target in node.targets
        )
    ]
    exec(
        compile(ast.Module(body=[*constants, gather, count_sum, *statements], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return namespace


@pytest.mark.parametrize("group_type", [0, 1])
@pytest.mark.parametrize("count", [129, 1025, 32769, 2**24 + 1, 2**32 + 3, 2**48 + 5, 2**62 - 5])
def test_prefill_metadata_preserves_large_int64_counts_and_prefixes(group_type, count):
    counts = [count, 0, count + 2, 7, 0]
    groups = counts if group_type == 1 else [sum(counts[: index + 1]) for index in range(len(counts))]
    namespace = prefill_register_expressions(groups, group_type, [4, 2, 0], [2, -1, 1, -1, 0], sum(counts), subset=True)
    assert bool(namespace["valid"])
    assert namespace["physical_counts"].dtype == torch.int64
    assert namespace["physical_counts"][:3].tolist() == [0, count + 2, count]
    assert namespace["source_starts"][:3].tolist() == [sum(counts), count, 0]
    assert namespace["physical_starts"][:3].tolist() == [0, 0, count + 2]


@pytest.mark.parametrize("group_type", [0, 1])
@pytest.mark.parametrize("bad", [-(2**63), 2**63 - 1, -1, 2**24 + 1, 1026])
def test_prefill_metadata_rejects_extreme_counts_without_overflow(group_type, bad):
    groups = [513, 513, 1025] if group_type == 0 else [513, 0, 512]
    groups[0] = bad
    namespace = prefill_register_expressions(groups, group_type, [2, 0, 1], [1, 2, 0], 1025)
    assert not bool(namespace["valid"])
    assert bool((namespace["logical_counts"] >= 0).all())
    assert bool((namespace["logical_counts"] <= 1025).all())


@pytest.mark.parametrize("counts", [[2**62, 2**62], [2**63 - 1] * 256])
@pytest.mark.parametrize("identity", [False, True])
def test_prefill_metadata_rejects_overflowing_count_sum_without_a_row_cap(counts, identity):
    experts = list(range(len(counts)))
    namespace = prefill_register_expressions(
        counts, 1, experts, experts, 2**63 - 1, allow_padding=True, identity=identity
    )
    assert not bool(namespace["valid"])
    assert not bool(namespace["total_valid"])


@pytest.mark.parametrize("rows", [0, 257, 1025, 32769])
@pytest.mark.parametrize("subset", [False, True])
def test_prefill_metadata_empty_dynamic_bank_is_valid(rows, subset):
    groups = [129, 0, 257] if subset and rows >= 386 else [0, 0, 0]
    namespace = prefill_register_expressions(groups, 1, [1], [-1, 0, -1], rows, allow_padding=True, subset=subset)
    assert bool(namespace["valid"])
    assert namespace["physical_counts"].tolist() == [0]


@pytest.mark.parametrize("rows", [0, 257, 1025, 32769])
@pytest.mark.parametrize("allow_padding,subset", [(False, False), (True, False), (True, True)])
def test_metadata_wrapper_accepts_actual_prefill_capacity_without_host_reads(
    routes, monkeypatch, rows, allow_padding, subset
):
    args = list(route_inputs())
    args[0] = tagged(torch.empty(rows, 32, dtype=torch.bfloat16))
    received = []
    result = tuple(object() for _ in range(5))

    def launch(*values, **options):
        received.append((values, options))
        return result

    monkeypatch.setattr(routes, "prepare_route_metadata", launch)
    monkeypatch.setattr(torch.Tensor, "cpu", lambda *_: pytest.fail("metadata read a dynamic tensor on the host"))
    monkeypatch.setattr(torch.Tensor, "tolist", lambda *_: pytest.fail("metadata read dynamic counts on the host"))
    assert routes.prepare_metadata(*args, allow_padding=allow_padding, subset=subset) is result
    assert received[0][1] == {"allow_padding": allow_padding, "subset": subset}
    assert all(got is expected for got, expected in zip(received[0][0], args, strict=True))


def test_metadata_launcher_has_fixed_row_tiles_and_no_hidden_copy():
    source, _ = route_kernel_source()
    module = ast.parse(source.read_text(encoding="utf-8"))
    launch = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "prepare_metadata")
    calls = [node for node in ast.walk(launch) if isinstance(node, ast.Call)]
    assert not any(
        ast.unparse(call.func) == "torch.empty_like" and ast.unparse(call.args[0]) == "hidden" for call in calls
    )
    row_launch = next(
        call
        for call in calls
        if isinstance(call.func, ast.Subscript) and ast.unparse(call.func.value) == "_prefill_metadata_rows_kernel"
    )
    assert ast.unparse(row_launch.func.slice) == "(triton.cdiv(rows, PREFILL_ROW_BLOCK) + experts - 1,)"
    assert (
        ast.unparse(next(option.value for option in row_launch.keywords if option.arg == "ROW_BLOCK"))
        == "PREFILL_ROW_BLOCK"
    )


def test_metadata_wrapper_preserves_graph_scratch_owner(routes, monkeypatch):
    retained, received = [], []
    expected = object()

    def launch(*args, **kwargs):
        received.append(kwargs)
        return expected

    monkeypatch.setattr(routes, "prepare_route_metadata", launch)
    assert routes.prepare_metadata(*route_inputs(), retained=retained) is expected
    assert received[0]["retained"] is retained


@pytest.mark.parametrize("group_type", [0, 1])
def test_metadata_launcher_keeps_internal_scratch_and_local_compiler_options(group_type):
    source, _ = route_kernel_source()
    module = ast.parse(source.read_text(encoding="utf-8"))
    launch = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "prepare_metadata")
    calls = []

    class Kernel:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: calls.append((args, kwargs))

    namespace = {
        "torch": torch,
        "triton": SimpleNamespace(
            next_power_of_2=lambda n: 1 << (n - 1).bit_length(), cdiv=lambda a, b: (a + b - 1) // b
        ),
        "PREFILL_ROW_BLOCK": 64,
        "_prefill_metadata_kernel": Kernel(),
        "_prefill_metadata_rows_kernel": Kernel(),
    }
    exec(compile(ast.Module(body=[launch], type_ignores=[]), str(source), "exec"), namespace)
    owners = []
    for rows in (12, 129, 512):
        retained = []
        calls.clear()
        result = namespace["prepare_metadata"](
            torch.empty(rows, 32, dtype=torch.bfloat16),
            torch.tensor([rows, rows] if group_type == 0 else [rows, 0]),
            group_type,
            torch.tensor([0, 1]),
            torch.tensor([0, 1]),
            retained=retained,
        )
        assert len(retained) == 3
        assert len(result) == 5
        assert len(calls) == 2
        args, options = calls[0]
        assert all(actual is expected for actual, expected in zip(retained, args[4:7], strict=True))
        assert options["enable_auto_vectorize_v2"] is False
        assert options["GROUP_TYPE"] == group_type
        assert calls[1][0][4] is result[1]  # Gather and inverse remain live outputs.
        assert calls[1][0][5] is result[2]
        assert "enable_auto_vectorize_v2" not in calls[1][1]
        owners.append(retained)
    assert len({id(value) for owner in owners for value in owner}) == 3 * len(owners)


@pytest.mark.parametrize("fault", [None, "inverse", "counts", "flags", "shape", "device"])
def test_restore_wrapper_checks_metadata_and_preserves_inplace_destination(routes, monkeypatch, fault):
    output = tagged(torch.empty(1025, 32, dtype=torch.bfloat16))
    inverse = tagged(torch.full((1025,), -1, dtype=torch.int64))
    counts = tagged(torch.zeros(3, dtype=torch.int64))
    flags = tagged(torch.ones(1, dtype=torch.int32))
    result = tagged(torch.zeros(1025, 32, dtype=torch.bfloat16))
    if fault == "inverse":
        inverse = tagged(torch.zeros(1025, dtype=torch.int32))
    elif fault == "counts":
        counts = tagged(torch.zeros(3, dtype=torch.int32))
    elif fault == "flags":
        flags = tagged(torch.ones(3, dtype=torch.int32))
    elif fault == "shape":
        result = tagged(torch.zeros(1024, 32, dtype=torch.bfloat16))
    elif fault == "device":
        result = tagged(torch.zeros(1025, 32, dtype=torch.bfloat16), index=1)
    observed = []
    expected = object()
    monkeypatch.setattr(routes, "restore_routes", lambda *values: observed.append(values) or expected)
    args = output, inverse, counts, flags, result
    if fault:
        with pytest.raises(ValueError):
            routes.restore(*args)
        assert not observed
    else:
        assert routes.restore(*args) is expected
        assert all(got is wanted for got, wanted in zip(observed[0], args, strict=True))


@pytest.mark.parametrize("fault", [None, "dtype", "shape", "device"])
def test_restore_wrapper_reuses_only_well_formed_device_summary(routes, monkeypatch, fault):
    output = tagged(torch.empty(257, 64, dtype=torch.bfloat16))
    inverse = tagged(torch.full((257,), -1, dtype=torch.int64))
    counts = tagged(torch.zeros(3, dtype=torch.int64))
    flags = tagged(torch.ones(1, dtype=torch.int32))
    result = tagged(torch.zeros_like(output))
    summary = tagged(torch.tensor([0, 1], dtype=torch.int64))
    if fault == "dtype":
        summary = tagged(torch.tensor([0, 1], dtype=torch.int32))
    elif fault == "shape":
        summary = tagged(torch.ones(1, dtype=torch.int64))
    elif fault == "device":
        summary = tagged(torch.tensor([0, 1], dtype=torch.int64), index=1)
    observed = []
    monkeypatch.setattr(routes, "restore_routes", lambda *args, **kwargs: observed.append((args, kwargs)))
    args = output, inverse, counts, flags, result
    if fault:
        with pytest.raises(ValueError, match="metadata"):
            routes.restore(*args, metadata=summary)
        assert not observed
    else:
        routes.restore(*args, metadata=summary)
        assert all(got is wanted for got, wanted in zip(observed[0][0], args, strict=True))
        assert observed[0][1]["metadata"] is summary


def evaluate_row_kernel(name, tensors, options, program_ids, *, source_path=None):
    """Run production control flow on CPU; this does not emulate compilation."""
    source = Path(source_path) if source_path is not None else route_kernel_source()[0]
    module = ast.parse(source.read_text(encoding="utf-8"))
    kernel = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == name)
    kernel.decorator_list = []
    accesses, writes = [], {key: torch.zeros_like(value, dtype=torch.int32) for key, value in tensors.items()}

    class Pointer:
        def __init__(self, name, offsets=0):
            self.name, self.offsets = name, torch.as_tensor(offsets, dtype=torch.int64)

        def __add__(self, offsets):
            return Pointer(self.name, self.offsets + offsets)

        def __sub__(self, offsets):
            return Pointer(self.name, self.offsets - offsets)

    def load(pointer, mask=True, other=0):
        offsets, mask = torch.broadcast_tensors(pointer.offsets, torch.as_tensor(mask))
        storage = tensors[pointer.name].flatten()
        selected = offsets[mask]
        assert bool(((selected >= 0) & (selected < storage.numel())).all())
        values = torch.full(offsets.shape, other, dtype=storage.dtype)
        values[mask] = storage[selected]
        accesses.append(("load", pointer.name, selected.numel()))
        return values

    def store(pointer, value, mask=True):
        offsets, values, mask = torch.broadcast_tensors(pointer.offsets, torch.as_tensor(value), torch.as_tensor(mask))
        storage = tensors[pointer.name].flatten()
        selected = offsets[mask]
        assert bool(((selected >= 0) & (selected < storage.numel())).all())
        storage[selected] = values[mask].to(storage.dtype)
        writes[pointer.name].flatten()[selected] += 1
        accesses.append(("store", pointer.name, selected.numel()))

    namespace = {
        "tl": SimpleNamespace(
            program_id=lambda dimension: torch.tensor(program_ids[dimension], dtype=torch.int32),
            arange=torch.arange,
            full=lambda shape, value, dtype: torch.full(shape, value, dtype=dtype),
            load=load,
            store=store,
            where=torch.where,
            abs=torch.abs,
            min=lambda value, axis: value.amin(dim=axis),
            int32=torch.int32,
            int64=torch.int64,
            float32=torch.float32,
            constexpr=int,
        ),
        "range": lambda *args: (torch.tensor(index, dtype=torch.int64) for index in range(*args)),
    }
    exec(compile(ast.Module(body=[kernel], type_ignores=[]), str(source), "exec"), namespace)
    namespace[name](**{key: Pointer(key) for key in tensors}, **options)
    return accesses, writes


@pytest.mark.parametrize("width", [3, 96, 4096, 4097])
@pytest.mark.parametrize("fault", [None, "live_nan", "live_inf", "inverse_oob", "inverse_negative", "route", "empty"])
def test_restore_register_masks_tails_before_finiteness_and_preserves_unselected_rows(fault, width):
    rows, block = 5, min(4096, 1 << (width - 1).bit_length())
    column_blocks = (width + block - 1) // block
    inverse = torch.tensor([-1, 1, -1, 0, -1], dtype=torch.int64)
    physical_output = torch.full((rows, width), float("nan"))
    physical_output[:2] = torch.arange(1, 2 * width + 1).reshape(2, width)
    if fault in ("live_nan", "live_inf"):
        physical_output[1, -1] = float("nan") if fault == "live_nan" else float("inf")
    elif fault == "inverse_oob":
        inverse[1] = 2**63 - 1
    elif fault == "inverse_negative":
        inverse[1] = -(2**63)
    elif fault == "empty":
        inverse.fill_(-1)
    tensors = {
        "output_ptr": physical_output,
        "inverse_ptr": inverse,
        "metadata_ptr": torch.tensor([0 if fault == "empty" else 2, int(fault != "route")]),
        "result_ptr": torch.full_like(physical_output, 7),
        "block_flags_ptr": torch.empty(rows * column_blocks, dtype=torch.int32),
    }
    for row in range(rows):
        for column_block in range(column_blocks):
            accesses, _ = evaluate_row_kernel(
                "_restore_kernel",
                tensors,
                {"ROWS": rows, "WIDTH": width, "BLOCK": block, "COLUMN_BLOCKS": column_blocks},
                (row, column_block),
            )
            if fault == "route" or inverse[row] < 0 or inverse[row] >= 2:
                assert not any(kind == "load" and name == "output_ptr" for kind, name, _ in accesses)
    assert bool(tensors["block_flags_ptr"].all()) == (fault in (None, "empty"))
    expected = torch.full_like(physical_output, 7)
    if fault not in ("route", "empty"):
        expected[3] = physical_output[0]
        if fault not in ("inverse_oob", "inverse_negative"):
            expected[1] = torch.nan_to_num(physical_output[1], nan=0, posinf=0)
    assert torch.equal(tensors["result_ptr"], expected)


@pytest.mark.parametrize("group_type", [0, 1])
@pytest.mark.parametrize("invalid", [False, True])
@pytest.mark.parametrize(
    "counts,experts",
    [
        ([513, 2, 257, 0, 7], [4, 0, 2]),
        ([0, 1025, 0], [2, 0]),
        ([0, 0, 0], [2, 0, 1]),
        ([255, 256, 257], [2, 0, 1]),
        ([1] * 127, list(reversed(range(127)))),
        ([0] * 128 + [65537], [128, 0, 4]),
    ],
)
def test_prefill_compact_tiles_match_logical_row_oracle(counts, experts, group_type, invalid):
    rows, row_block = sum(counts) + 11, 256
    slots = [-1] * len(counts)
    for slot, expert in enumerate(experts):
        slots[expert] = slot
    group = counts if group_type == 1 else [sum(counts[: index + 1]) for index in range(len(counts))]
    if invalid:
        group = [-(2**63), *group[1:]]
    metadata = prefill_register_expressions(group, group_type, experts, slots, rows, allow_padding=True, subset=True)
    tensors = {
        "counts_ptr": metadata["safe_counts"][: len(experts)].clone(),
        "source_starts_ptr": metadata["source_starts"][: len(experts)].clone(),
        "physical_starts_ptr": metadata["physical_starts"][: len(experts)].clone(),
        "tile_ends_ptr": metadata["tile_ends"][: len(experts)].clone(),
        "gather_ptr": torch.full((rows,), -1, dtype=torch.int64),
        "inverse_ptr": torch.full((rows,), -1, dtype=torch.int64),
    }
    grid = (rows + row_block - 1) // row_block + len(experts) - 1
    assert int(tensors["tile_ends_ptr"][-1]) <= grid
    total_writes = {name: torch.zeros(rows, dtype=torch.int32) for name in ("gather_ptr", "inverse_ptr")}
    for task in range(grid):
        _, writes = evaluate_row_kernel(
            "_prefill_metadata_rows_kernel",
            tensors,
            {
                "ROWS": rows,
                "EXPERTS": len(experts),
                "EXPERT_BLOCK": 1 << (len(experts) - 1).bit_length(),
                "ROW_BLOCK": row_block,
            },
            (task,),
        )
        for name in total_writes:
            total_writes[name] += writes[name]
    expected = route_reference(group, group_type, experts, slots, rows, allow_padding=True, subset=True, max_count=None)
    assert tensors["gather_ptr"].tolist() == (expected[1] if expected[3] else [-1] * rows)
    assert tensors["inverse_ptr"].tolist() == expected[2]
    assert all(bool((writes <= 1).all()) for writes in total_writes.values())
    assert total_writes["gather_ptr"].sum() == sum(expected[0])


def test_prefill_grid_removes_expert_times_capacity_launches():
    rows, experts, block = 32768, 128, 256
    grid = (rows + block - 1) // block + experts - 1
    assert grid == 255
    for counts in ([256] * experts, [rows] + [0] * (experts - 1), [0] * experts):
        metadata = prefill_register_expressions(
            counts, 1, list(range(experts)), list(range(experts)), rows, allow_padding=True
        )
        assert int(metadata["tile_ends"][-1]) <= grid


def test_restore_launcher_reuses_summary_and_bounds_column_work():
    source, _ = route_kernel_source()
    module = ast.parse(source.read_text(encoding="utf-8"))
    launch = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "restore")
    launch.decorator_list = []
    launched = []

    class Kernel:
        def __init__(self, name):
            self.name = name

        def __getitem__(self, grid):
            return lambda *args, **kwargs: launched.append((self.name, grid, args, kwargs))

    namespace = {
        "torch": torch,
        "triton": SimpleNamespace(
            next_power_of_2=lambda value: 1 << (value - 1).bit_length(), cdiv=lambda a, b: (a + b - 1) // b
        ),
        "RESTORE_COLUMN_BLOCK": 4096,
        "_restore_metadata_kernel": Kernel("summary"),
        "_restore_kernel": Kernel("rows"),
    }
    exec(compile(ast.Module(body=[launch], type_ignores=[]), str(source), "exec"), namespace)
    output = torch.empty(3, 4097)
    inverse, counts, flags = (
        torch.zeros(3, dtype=torch.int64),
        torch.zeros(2, dtype=torch.int64),
        torch.ones(1, dtype=torch.int32),
    )
    for reuse in (False, True):
        launched.clear()
        summary = torch.tensor([0, 1], dtype=torch.int64) if reuse else None
        actual = namespace["restore"](output, inverse, counts, flags, output, metadata=summary)
        assert actual.shape == (6,)
        assert [entry[0] for entry in launched] == (["rows"] if reuse else ["summary", "rows"])
        _, grid, args, options = launched[-1]
        assert grid == (3, 2) and options["BLOCK"] == 4096
        if reuse:
            assert args[2] is summary
    for valid in (0, 1):
        actual = namespace["restore"](
            output[:0], inverse[:0], counts, flags, output[:0], metadata=torch.tensor([0, valid])
        )
        assert actual.tolist() == [valid]

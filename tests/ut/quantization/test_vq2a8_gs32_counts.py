# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Count sanitizer arithmetic/registration checks, without NPU execution."""

import ast
import importlib.util
import random
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO = Path(__file__).resolve().parents[3]
KERNEL = REPO / "vllm_ascend/ops/triton/activation/vq2a8_gs32_counts.py"
WRAPPER = REPO / "vllm_ascend/ops/fused_moe/vq2a8_gs32_counts.py"
OPERATOR = "vq2a8_gs32_stage::_sanitize_device_counts"
SCHEMA = f"{OPERATOR}(Tensor counts, int rows, bool allow_padding) -> (Tensor, Tensor)"
INT64_MAX = (1 << 63) - 1


def production_register_arithmetic(counts, rows, allow_padding):
    """Evaluate the actual kernel's integer expressions on CPU registers only."""
    module = ast.parse(KERNEL.read_text(encoding="utf-8"))
    kernel = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_sanitize_device_counts_kernel"
    )
    statements = [
        node
        for node in kernel.body
        if not isinstance(node, ast.Expr)
        and not (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "raw")
    ]
    block = 1 << (len(counts) - 1).bit_length()
    namespace = {
        "tl": SimpleNamespace(
            arange=torch.arange,
            where=torch.where,
            int32=torch.int32,
            sum=lambda x, axis: x.sum(dim=axis),
            min=lambda x, axis: x.amin(dim=axis),
            minimum=lambda x, value: x.clamp_max(value),
            maximum=lambda x, value: x.clamp_min(value),
        ),
        "BLOCK": block,
        "GROUPS": len(counts),
        "ROWS": rows,
        "ALLOW_PADDING": allow_padding,
        "INT64_SUM_BITS": 32,
        "raw": torch.tensor([*counts, *([0] * (block - len(counts)))], dtype=torch.int64),
    }
    exec(compile(ast.Module(body=statements, type_ignores=[]), str(KERNEL), "exec"), namespace)
    return namespace["safe_counts"][: len(counts)].tolist(), int(namespace["valid"])


def integer_oracle(counts, rows, allow_padding):
    # Python integers do not overflow; this is independent of split-word math.
    total = sum(counts)
    valid = all(0 <= count <= rows for count in counts)
    valid = valid and (total <= rows if allow_padding else total == rows)
    return (counts if valid else [0] * len(counts)), int(valid)


@pytest.mark.parametrize("groups", (1, 2, 3, 7, 255, 256))
@pytest.mark.parametrize("rows", (1, 8, 128, 129, 2**24 + 1, 2**32 + 3, 2**62, INT64_MAX))
@pytest.mark.parametrize("allow_padding", (False, True))
def test_count_sanitizer_matches_big_integer_oracle(groups, rows, allow_padding):
    cases = [[0] * groups, [rows] + [0] * (groups - 1), [rows] * groups]
    for first in (-1, -(1 << 63), INT64_MAX):
        cases.append([first] + [0] * (groups - 1))
    if groups > 1:
        cases.extend(([rows - 1, 1] + [0] * (groups - 2), [rows, 1] + [0] * (groups - 2)))
    for counts in cases:
        assert production_register_arithmetic(counts, rows, allow_padding) == integer_oracle(
            counts, rows, allow_padding
        )


def test_count_sanitizer_random_extremes_do_not_wrap_into_valid_routes():
    generator = random.Random(95032)
    for _ in range(200):
        rows = generator.choice((129, 2**32 + 7, 2**62 + 1, INT64_MAX))
        counts = [
            generator.choice((0, 1, -1, -(1 << 63), INT64_MAX, rows, generator.randrange(rows)))
            for _ in range(generator.randint(1, 256))
        ]
        allow_padding = bool(generator.randrange(2))
        assert production_register_arithmetic(counts, rows, allow_padding) == integer_oracle(
            counts, rows, allow_padding
        )


def load_wrapper(monkeypatch, *, missing=False, wrong=False, registered=False):
    registrations = []

    def find_schema(name, overload):
        assert name == OPERATOR and overload == ""
        if missing:
            raise RuntimeError("missing schema")
        return SimpleNamespace(schema=lambda: SCHEMA.replace("int rows", "float rows") if wrong else SCHEMA)

    monkeypatch.setattr(torch._C, "_dispatch_find_schema_or_throw", find_schema)
    monkeypatch.setattr(torch._C, "_dispatch_has_kernel_for_dispatch_key", lambda *args: registered)
    monkeypatch.setattr(
        torch.library, "impl", lambda name, backend: lambda fn: registrations.append((name, backend, fn))
    )
    spec = importlib.util.spec_from_file_location("_gs32_counts_wrapper_test", WRAPPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, registrations


@pytest.mark.parametrize("registered", (False, True))
def test_registration_is_host_only_and_idempotent(monkeypatch, registered):
    module, registrations = load_wrapper(monkeypatch, registered=registered)
    assert len(registrations) == (0 if registered else 1)
    if registrations:
        assert registrations[0] == (OPERATOR, "PrivateUse1", module._sanitize_device_counts)


@pytest.mark.parametrize("fault", ("missing", "wrong"))
def test_registration_rejects_stale_bridge(monkeypatch, fault):
    with pytest.raises(RuntimeError, match="schema"):
        load_wrapper(monkeypatch, **{fault: True})


class NpuMetadataTensor(torch.Tensor):
    @property
    def device(self):
        return SimpleNamespace(type="npu", index=0)


class UnindexedNpuMetadataTensor(torch.Tensor):
    @property
    def device(self):
        return SimpleNamespace(type="npu", index=None)


def test_wrapper_rejects_implicit_device(monkeypatch):
    wrapper, _ = load_wrapper(monkeypatch)
    counts = torch.ones(1, dtype=torch.int64).as_subclass(UnindexedNpuMetadataTensor)
    with pytest.raises(ValueError, match="explicit NPU"):
        wrapper._sanitize_device_counts(counts, 8, True)


def test_wrapper_keeps_counts_on_device_and_launches_once(monkeypatch):
    wrapper, _ = load_wrapper(monkeypatch)
    counts = torch.tensor([1, 0, 2], dtype=torch.int64).as_subclass(NpuMetadataTensor)
    calls = []
    outputs = (object(), object())
    kernel = SimpleNamespace(sanitize_device_counts=lambda *args: calls.append(args) or outputs)
    monkeypatch.setitem(sys.modules, "vllm_ascend.ops.triton.activation.vq2a8_gs32_counts", kernel)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(device=lambda _: nullcontext()), raising=False)
    monkeypatch.setattr(torch.Tensor, "cpu", lambda *_: pytest.fail("count host transfer"))
    monkeypatch.setattr(torch.Tensor, "tolist", lambda *_: pytest.fail("count host transfer"))
    monkeypatch.setattr(torch.Tensor, "item", lambda *_: pytest.fail("count host read"))
    assert wrapper._sanitize_device_counts(counts, 8, True) is outputs
    assert len(calls) == 1 and calls[0][0] is counts and calls[0][1:] == (8, True)


@pytest.mark.parametrize("rows,padding", ((0, True), (-1, True), (1 << 63, True), (True, True), (1, 1)))
def test_wrapper_rejects_invalid_host_capacity_and_padding(monkeypatch, rows, padding):
    wrapper, _ = load_wrapper(monkeypatch)
    counts = torch.ones(1, dtype=torch.int64).as_subclass(NpuMetadataTensor)
    with pytest.raises(ValueError, match="positive int64"):
        wrapper._sanitize_device_counts(counts, rows, padding)


@pytest.mark.parametrize(
    "shape,dtype", (((0,), torch.int64), ((257,), torch.int64), ((1, 1), torch.int64), ((1,), torch.int32))
)
def test_wrapper_rejects_invalid_counts_metadata(monkeypatch, shape, dtype):
    wrapper, _ = load_wrapper(monkeypatch)
    counts = torch.zeros(shape, dtype=dtype).as_subclass(NpuMetadataTensor)
    with pytest.raises(ValueError, match="int64 counts"):
        wrapper._sanitize_device_counts(counts, 8, True)


def test_launcher_has_one_kernel_and_fresh_outputs():
    module = ast.parse(KERNEL.read_text(encoding="utf-8"))
    launch = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "sanitize_device_counts"
    )
    calls = [node for node in ast.walk(launch) if isinstance(node, ast.Call)]
    launches = [node for node in calls if isinstance(node.func, ast.Subscript)]
    assert len(launches) == 1 and ast.unparse(launches[0].func) == "_sanitize_device_counts_kernel[1,]"
    allocations = [ast.unparse(node.func) for node in calls if ast.unparse(node.func).startswith("torch.")]
    assert allocations == ["torch.empty_like", "torch.empty"]

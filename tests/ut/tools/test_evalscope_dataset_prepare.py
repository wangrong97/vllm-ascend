# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

PREPARER = Path(__file__).resolve().parents[3] / "tools/tests/prepare_evalscope_datasets.py"


@pytest.fixture
def preparer(monkeypatch):
    monkeypatch.syspath_prepend(str(PREPARER.parent))
    spec = importlib.util.spec_from_file_location("evalscope_dataset_prepare_test", PREPARER)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def native_adapters(monkeypatch, tmp_path):
    calls = []
    configs = []
    default_root = tmp_path / "native cache"
    adapters = {}
    loads = []
    allow_download = [True]

    class NativeDatasetDict:
        # EvalScope 1.12.0 DatasetDict exposes keys but no __iter__. Preserve
        # that interface so a plain dict cannot hide invalid iteration.
        def __init__(self, subsets):
            self.subsets = subsets

        def __getitem__(self, key):
            return self.subsets[key]

        def keys(self):
            return self.subsets.keys()

        def __len__(self):
            return len(self.subsets)

    sources = {
        "aime25": ("evalscope/aime25", "test"),
        "aime24": ("evalscope/aime24", "test"),
        "gpqa_diamond": ("AI-ModelScope/gpqa_diamond", "train"),
    }

    def cache_hash(*args):
        return hashlib.sha256(json.dumps(args).encode()).hexdigest()

    def safe_filename(name):
        return name.replace("/", "_")

    def load(adapter):
        loads.append(adapter.dataset_id)
        digest = cache_hash(adapter.dataset_id, adapter.eval_split, "default", None, "modelscope", {})
        cache = Path(adapter.dataset_dir) / "datasets" / f"{safe_filename(adapter.dataset_id)}-{digest}"
        if not cache.exists():
            assert allow_download[0], "offline loader must never download"
            cache.mkdir(parents=True)
            (cache / "state.json").write_text('{"_data_files":[{"filename":"data-00000.arrow"}]}')
            (cache / "dataset_info.json").write_text('{"features":{}}')
            (cache / "data-00000.arrow").write_bytes(b"native raw records, not postprocessed Sample")
        return NativeDatasetDict(adapter.subsets)

    for name, count in (("aime25", 30), ("aime24", 30), ("gpqa_diamond", 198)):
        samples = [
            SimpleNamespace(
                input="Question",
                target="A" if name == "gpqa_diamond" else "42",
                choices=["one", "two", "three", "four"],
            )
            for _ in range(count)
        ]
        adapter = SimpleNamespace(
            dataset_id=sources[name][0],
            eval_split=sources[name][1],
            subsets={"default": samples},
            dataset_revision=None,
            dataset_hub="modelscope",
            subset_list=["default"],
            few_shot_num=0,
            force_redownload=False,
        )
        adapter.load_dataset = lambda adapter=adapter: load(adapter)
        adapters[name] = adapter

    class TaskConfig:
        def __init__(self, **kwargs):
            configs.append(kwargs)
            self.dataset_dir = kwargs.get("dataset_dir", str(default_root))

    def get_benchmark(name, config):
        calls.append((name, config.dataset_dir))
        adapters[name].dataset_dir = config.dataset_dir
        return adapters[name]

    def unexpected(*args, **kwargs):
        pytest.fail("preparing datasets must not run inference")

    modules = {
        name: ModuleType(name)
        for name in (
            "evalscope",
            "evalscope.api",
            "evalscope.api.registry",
            "evalscope.config",
            "evalscope.api.dataset",
            "evalscope.api.dataset.loader",
            "evalscope.utils",
            "evalscope.utils.io_utils",
        )
    }
    modules["evalscope"].__path__ = []
    modules["evalscope.api"].__path__ = []
    modules["evalscope"].run_task = unexpected
    modules["evalscope.api.registry"].get_benchmark = get_benchmark
    modules["evalscope.config"].TaskConfig = TaskConfig
    modules["evalscope.api.dataset.loader"]._dataset_cache_hash = cache_hash
    modules["evalscope.utils.io_utils"].safe_filename = safe_filename
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return SimpleNamespace(
        adapters=adapters,
        calls=calls,
        configs=configs,
        default_root=default_root,
        sources=sources,
        loads=loads,
        allow_download=allow_download,
    )


def test_defaults_prepare_all_three_datasets_with_native_cache(preparer):
    args = preparer.parse_args([])
    assert args.datasets == ["aime25", "aime24", "gpqa_diamond"]
    assert args.dataset_dir is None


def test_optional_dataset_directory_is_resolved_without_creating_it(preparer, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = preparer.parse_args(["--dataset-dir", "cache with spaces", "--datasets", "aime25"])
    assert args.dataset_dir == tmp_path / "cache with spaces"
    assert args.datasets == ["aime25"]
    assert not args.dataset_dir.exists()


@pytest.mark.parametrize("datasets", [["unknown"], ["aime25", "aime25"]])
def test_invalid_dataset_selection_rejected(preparer, datasets):
    with pytest.raises(SystemExit) as exc:
        preparer.parse_args(["--datasets", *datasets])
    assert exc.value.code == 2


@pytest.mark.parametrize("custom_cache", [False, True])
def test_loads_complete_native_datasets_without_inference(preparer, native_adapters, tmp_path, custom_cache):
    names = ["aime25", "aime24", "gpqa_diamond"]
    root = tmp_path / "explicit cache" if custom_cache else None
    summary = preparer.preload_datasets(names, root)
    actual_root = root or native_adapters.default_root
    assert summary == {
        "dataset_dir": str(actual_root),
        "datasets": {
            name: {
                "dataset_id": native_adapters.sources[name][0],
                "split": native_adapters.sources[name][1],
                "records": count,
            }
            for name, count in (("aime25", 30), ("aime24", 30), ("gpqa_diamond", 198))
        },
    }
    assert native_adapters.calls == [(name, str(actual_root)) for name in names]
    expected = {"datasets": names, "limit": None, "repeats": 1, "seed": 0}
    if custom_cache:
        expected["dataset_dir"] = str(root)
    assert native_adapters.configs == [expected]


@pytest.mark.parametrize("subsets", [{}, {"test": []}, {"default": [], "extra": []}])
def test_unexpected_subset_layout_fails(preparer, native_adapters, subsets):
    native_adapters.adapters["aime25"].subsets = subsets
    with pytest.raises(ValueError, match="expected the default subset"):
        preparer.preload_datasets(["aime25"])


@pytest.mark.parametrize("count", [0, 29, 31])
def test_incomplete_or_oversized_dataset_fails(preparer, native_adapters, count):
    samples = [SimpleNamespace(input="Question", target="42") for _ in range(count)]
    native_adapters.adapters["aime25"].subsets["default"] = samples
    with pytest.raises(ValueError, match="expected 30 questions"):
        preparer.preload_datasets(["aime25"])


@pytest.mark.parametrize(
    "field,value", [("input", ""), ("input", []), ("target", ""), ("target", "  "), ("target", None)]
)
def test_missing_question_or_answer_fails(preparer, native_adapters, field, value):
    sample = native_adapters.adapters["aime25"].subsets["default"][0]
    setattr(sample, field, value)
    with pytest.raises(ValueError, match="sample 0 has an empty input or answer"):
        preparer.preload_datasets(["aime25"])


def test_string_zero_is_a_valid_math_answer(preparer, native_adapters):
    native_adapters.adapters["aime25"].subsets["default"][0].target = "0"
    assert preparer.preload_datasets(["aime25"])["datasets"]["aime25"]["records"] == 30


def test_missing_or_wrong_dependency_fails_before_dataset_import(preparer, monkeypatch, capsys):
    def unavailable():
        raise ValueError("EvalScope 1.12.0 is not installed")

    def unexpected(*args):
        pytest.fail("dataset import/download must not run before version check passes")

    monkeypatch.setattr(preparer, "check_evalscope_version", unavailable)
    monkeypatch.setattr(preparer, "preload_datasets", unexpected)
    assert preparer.main([]) == 1
    assert "EVALSCOPE_DATASETS=FAIL" in capsys.readouterr().err


def test_broken_optional_dependency_fails_cleanly(preparer, monkeypatch, capsys):
    monkeypatch.setattr(preparer, "check_evalscope_version", lambda: "1.12.0")
    monkeypatch.setitem(sys.modules, "evalscope.api.registry", None)
    assert preparer.main(["--datasets", "aime25"]) == 1
    assert "EVALSCOPE_DATASETS=FAIL ModuleNotFoundError:" in capsys.readouterr().err


def test_ready_output_reports_actual_cache_and_dataset_summary(preparer, tmp_path, monkeypatch, capsys):
    observed = []
    summary = {
        "dataset_dir": str(tmp_path),
        "datasets": {"aime25": {"dataset_id": "native/aime25", "split": "test", "records": 30}},
    }
    monkeypatch.setattr(preparer, "check_evalscope_version", lambda: "1.12.0")

    def preload(names, dataset_dir, output_dir):
        observed.append((names, dataset_dir, output_dir))
        return summary

    monkeypatch.setattr(preparer, "preload_datasets", preload)
    assert preparer.main(["--datasets", "aime25", "--dataset-dir", str(tmp_path)]) == 0
    assert observed == [(["aime25"], tmp_path, None)]
    output = capsys.readouterr().out
    line = next(line for line in output.splitlines() if line.startswith("EVALSCOPE_DATASETS=READY"))
    assert json.loads(line.removeprefix("EVALSCOPE_DATASETS=READY").strip()) == summary


def test_download_failure_is_reported_without_ready_marker(preparer, monkeypatch, capsys):
    monkeypatch.setattr(preparer, "check_evalscope_version", lambda: "1.12.0")

    def unavailable(*args):
        raise ConnectionError("dataset source unavailable")

    monkeypatch.setattr(preparer, "preload_datasets", unavailable)
    assert preparer.main([]) == 1
    output = capsys.readouterr()
    assert "EVALSCOPE_DATASETS=READY" not in output.out
    assert "EVALSCOPE_DATASETS=FAIL" in output.err
    assert "dataset source unavailable" in output.err


def test_gpqa_alias_and_duplicate_alias_rejected(preparer):
    assert preparer.parse_args(["--datasets", "GPQA", "AIME24"]).datasets == ["gpqa_diamond", "aime24"]
    with pytest.raises(SystemExit) as exc:
        preparer.parse_args(["--datasets", "gpqa", "gpqa_diamond"])
    assert exc.value.code == 2


def test_export_requires_new_destination(preparer, tmp_path):
    with pytest.raises(SystemExit) as exc:
        preparer.parse_args(["--output-dir", str(tmp_path)])
    assert exc.value.code == 2


@pytest.mark.parametrize("names", [["aime25"], ["gpqa_diamond", "aime24", "aime25"]])
def test_portable_export_uses_native_cache_without_inference_or_download(preparer, native_adapters, tmp_path, names):
    output = tmp_path / "portable package"
    summary = preparer.preload_datasets(names, output_dir=output)
    assert summary["offline_bundle"] == str(output)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["evalscope_version"] == "1.12.0"
    assert set(manifest["datasets"]) == set(names)
    native_adapters.allow_download[0] = False
    native_adapters.loads.clear()
    result = preparer.validate_offline_bundle(output, names)
    assert result["datasets"] == manifest["datasets"]
    assert native_adapters.loads == [native_adapters.sources[name][0] for name in names]
    for entry in result["datasets"].values():
        assert set(entry["files"]) == {"state.json", "dataset_info.json", "data-00000.arrow"}
        assert (output / entry["cache_path"] / "data-00000.arrow").read_bytes().startswith(b"native raw records")


def _bundle(preparer, native_adapters, tmp_path):
    output = tmp_path / "offline"
    preparer.preload_datasets(["aime25", "gpqa_diamond"], output_dir=output)
    manifest = json.loads((output / "manifest.json").read_text())
    native_adapters.loads.clear()
    native_adapters.allow_download[0] = False
    return output, manifest


@pytest.mark.parametrize("damage", ["missing_file", "modified_file", "extra_file", "missing_shard", "external_shard"])
def test_damaged_bundle_fails_before_loading_any_adapter(preparer, native_adapters, tmp_path, damage):
    output, manifest = _bundle(preparer, native_adapters, tmp_path)
    cache = output / manifest["datasets"]["gpqa_diamond"]["cache_path"]
    if damage == "missing_file":
        (cache / "data-00000.arrow").unlink()
    elif damage == "modified_file":
        (cache / "data-00000.arrow").write_bytes(b"corrupt")
    elif damage == "extra_file":
        (cache / "extra.arrow").write_bytes(b"extra")
    else:
        filename = "missing.arrow" if damage == "missing_shard" else "../external.arrow"
        (cache / "state.json").write_text(json.dumps({"_data_files": [{"filename": filename}]}))
    with pytest.raises(ValueError):
        preparer.validate_offline_bundle(output, ["aime25", "gpqa_diamond"])
    assert native_adapters.loads == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("dataset_id", "other/repo"),
        ("split", "test"),
        ("subset", "main"),
        ("records", 197),
        ("cache_path", "../outside"),
        ("cache_path", "/outside"),
        ("files", {}),
    ],
)
def test_invalid_dataset_manifest_rejected(preparer, native_adapters, tmp_path, field, value):
    output, manifest = _bundle(preparer, native_adapters, tmp_path)
    manifest["datasets"]["gpqa_diamond"][field] = value
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        preparer.validate_offline_bundle(output, ["gpqa_diamond"])
    assert native_adapters.loads == []


@pytest.mark.parametrize(
    "replacement",
    [None, [], {}, {"schema_version": 2}, {"schema_version": 1, "evalscope_version": "1.11.0", "datasets": {}}],
)
def test_invalid_bundle_manifest_rejected(preparer, native_adapters, tmp_path, replacement):
    output, _ = _bundle(preparer, native_adapters, tmp_path)
    (output / "manifest.json").write_text(json.dumps(replacement))
    with pytest.raises(ValueError):
        preparer.validate_offline_bundle(output, ["aime25"])
    assert native_adapters.loads == []


def test_missing_selected_dataset_fails_without_network(preparer, native_adapters, tmp_path):
    output, _ = _bundle(preparer, native_adapters, tmp_path)
    with pytest.raises(ValueError, match="missing aime24"):
        preparer.validate_offline_bundle(output, ["aime24"])
    assert native_adapters.loads == []


def test_native_schema_is_checked_after_hash_validation(preparer, native_adapters, tmp_path):
    output, _ = _bundle(preparer, native_adapters, tmp_path)
    native_adapters.adapters["gpqa_diamond"].subsets["default"][0].target = "E"
    with pytest.raises(ValueError, match="four choices and an A-D answer"):
        preparer.validate_offline_bundle(output, ["gpqa_diamond"])


def test_failed_export_does_not_publish_partial_bundle(preparer, native_adapters, tmp_path, monkeypatch):
    output = tmp_path / "not ready"

    def broken_copy(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(preparer.shutil, "copytree", broken_copy)
    with pytest.raises(OSError, match="disk full"):
        preparer.preload_datasets(["aime25"], output_dir=output)
    assert not output.exists()
    assert not list(tmp_path.glob(".not ready-*"))


def test_existing_bundle_is_not_overwritten(preparer, native_adapters, tmp_path):
    output, _ = _bundle(preparer, native_adapters, tmp_path)
    original = (output / "manifest.json").read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        preparer.preload_datasets(["aime25"], output_dir=output)
    assert (output / "manifest.json").read_bytes() == original


def test_export_inside_source_cache_is_rejected_before_copy(preparer, native_adapters):
    preparer.preload_datasets(["aime25"])
    source = preparer._native_cache_path(native_adapters.default_root, "aime25", native_adapters.adapters["aime25"])
    output = source / "nested package"
    with pytest.raises(ValueError, match="outside the source"):
        preparer.preload_datasets(["aime25"], output_dir=output)
    assert not output.exists()


def test_bash_help_works_without_evalscope_from_other_directory(tmp_path):
    result = subprocess.run(
        ["bash", str(PREPARER.with_suffix(".sh")), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "--dataset-dir" in result.stdout
    assert "--output-dir" in result.stdout

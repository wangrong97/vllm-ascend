# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Prepare complete EvalScope datasets and export a portable native cache bundle."""

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from types import MappingProxyType

from run_evalscope_accuracy import EVALSCOPE_VERSION, check_evalscope_version, dataset_name
from summarize_evalscope_accuracy import DATASET_SIZES

BUNDLE_SCHEMA = 1
DATASET_SOURCES = MappingProxyType(
    {
        "aime24": ("evalscope/aime24", "test"),
        "aime25": ("evalscope/aime25", "test"),
        "gpqa_diamond": ("AI-ModelScope/gpqa_diamond", "train"),
    }
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", type=dataset_name, default=["aime25", "aime24", "gpqa_diamond"])
    parser.add_argument("--dataset-dir", type=Path, help="Cache root; use the same option when running evaluation")
    parser.add_argument("--output-dir", type=Path, help="Export selected datasets into a new portable offline bundle")
    args = parser.parse_args(argv)
    if len(set(args.datasets)) != len(args.datasets):
        parser.error("--datasets must not contain duplicates")
    if args.dataset_dir is not None:
        args.dataset_dir = args.dataset_dir.expanduser().resolve()
    if args.output_dir is not None:
        args.output_dir = args.output_dir.expanduser().resolve()
        if args.output_dir.exists():
            parser.error("--output-dir already exists; choose a new directory to preserve the existing bundle")
    return args


def _adapters(names: list[str], dataset_dir: Path | None):
    # Delay the optional EvalScope imports until after --help and the version check.
    # Constructing adapters and loading data does not construct or call a model.
    from evalscope.api.registry import get_benchmark
    from evalscope.config import TaskConfig

    options = {"dataset_dir": str(dataset_dir)} if dataset_dir is not None else {}
    config = TaskConfig(datasets=names, limit=None, repeats=1, seed=0, **options)
    cache_root = Path(config.dataset_dir).expanduser().resolve()
    return cache_root, [(name, get_benchmark(name, config)) for name in names]


def _native_cache_path(root: Path, name: str, adapter) -> Path:
    # Use the pinned loader's cache key rather than maintaining a second hash
    # algorithm. This is the raw Arrow cache, before record_to_sample/templates.
    from evalscope.api.dataset.loader import _dataset_cache_hash
    from evalscope.utils.io_utils import safe_filename

    dataset_id, split = DATASET_SOURCES[name]
    if (adapter.dataset_id, adapter.eval_split) != (dataset_id, split):
        raise ValueError(f"{name}: unexpected native dataset ID or split")
    if list(adapter.subset_list) != ["default"] or adapter.few_shot_num != 0:
        raise ValueError(f"{name}: offline bundles require the native default subset and zero-shot adapter")
    if adapter.force_redownload or Path(dataset_id).exists():
        raise ValueError(f"{name}: redownload/local source shadowing would bypass the portable cache")
    cache_hash = _dataset_cache_hash(dataset_id, split, "default", adapter.dataset_revision, adapter.dataset_hub, {})
    return root / "datasets" / f"{safe_filename(dataset_id)}-{cache_hash}"


def _validate_samples(name: str, subsets) -> int:
    if set(subsets.keys()) != {"default"}:
        raise ValueError(f"{name}: expected the default subset, found {list(subsets.keys())}")
    samples = subsets["default"]
    expected = DATASET_SIZES[name]
    if len(samples) != expected:
        raise ValueError(f"{name}: expected {expected} questions, found {len(samples)}")
    for sample_index, sample in enumerate(samples):
        if not sample.input or not sample.target or not str(sample.target).strip():
            raise ValueError(f"{name}: sample {sample_index} has an empty input or answer")
        if name == "gpqa_diamond" and (
            not isinstance(sample.choices, list)
            or len(sample.choices) != 4
            or not all(isinstance(choice, str) and choice.strip() for choice in sample.choices)
            or sample.target not in ("A", "B", "C", "D")
        ):
            raise ValueError(f"{name}: sample {sample_index} must have four choices and an A-D answer")
    return len(samples)


def _file_inventory(directory: Path) -> dict:
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError(f"Missing/nonportable native dataset cache: {directory}")
    inventory = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Native dataset cache must not contain symlinks: {path}")
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            inventory[path.relative_to(directory).as_posix()] = {
                "sha256": digest.hexdigest(),
                "size_bytes": path.stat().st_size,
            }
    if not {"state.json", "dataset_info.json"}.issubset(inventory) or not any(
        name.endswith(".arrow") for name in inventory
    ):
        raise ValueError(f"Incomplete native Arrow dataset cache: {directory}")
    try:
        state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
        shards = state.get("_data_files") if isinstance(state, dict) else None
        if not isinstance(shards, list) or not shards:
            raise ValueError("missing Arrow shard list")
        for shard in shards:
            name = shard.get("filename") if isinstance(shard, dict) else None
            if (
                not isinstance(name, str)
                or name not in inventory
                or not name.endswith(".arrow")
                or "\\" in name
                or Path(name).is_absolute()
                or any(part in (".", "..") for part in name.split("/"))
            ):
                raise ValueError("nonportable/missing Arrow shard reference")
    except (OSError, ValueError) as exc:
        raise ValueError(f"Invalid native Arrow cache state: {directory}: {exc}") from exc
    return inventory


def _export_bundle(output_dir: Path, sources: dict, summary: dict) -> dict:
    if any(output_dir.resolve().is_relative_to(source.resolve()) for source in sources.values()):
        raise ValueError("Offline bundle destination must be outside the source dataset caches")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        raise ValueError(f"Offline bundle already exists: {output_dir}")
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}-", dir=output_dir.parent) as temporary:
        staging = Path(temporary) / "bundle"
        staging.mkdir()
        manifest = {"schema_version": BUNDLE_SCHEMA, "evalscope_version": EVALSCOPE_VERSION, "datasets": {}}
        for name, source in sources.items():
            relative = Path("datasets") / source.name
            target = staging / relative
            # Validate the source before copying; no links to external caches.
            _file_inventory(source)
            shutil.copytree(source, target)
            manifest["datasets"][name] = {
                **summary["datasets"][name],
                "subset": "default",
                "cache_path": relative.as_posix(),
                "files": _file_inventory(target),
            }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        staging.rename(output_dir)
    return manifest


def validate_offline_bundle(bundle_dir: Path, names: list[str]) -> dict:
    """Check every selected cache before native adapters load it; never download.

    The caller passes this directory unchanged as EvalScope's --dataset-dir.
    RemoteDataLoader's cache-hit branch uses datasets.load_from_disk directly.
    """
    root = Path(bundle_dir).expanduser().resolve()
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read offline bundle manifest: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != BUNDLE_SCHEMA:
        raise ValueError("Unsupported offline bundle schema")
    if manifest.get("evalscope_version") != EVALSCOPE_VERSION or not isinstance(manifest.get("datasets"), dict):
        raise ValueError("Offline bundle must contain EvalScope 1.12.0 datasets")
    if not names or len(set(names)) != len(names) or any(name not in DATASET_SIZES for name in names):
        raise ValueError("Choose unique supported datasets when validating an offline bundle")
    _, adapters = _adapters(names, root)
    selected = {}
    for name, adapter in adapters:
        entry = manifest["datasets"].get(name)
        if not isinstance(entry, dict):
            raise ValueError(f"Offline bundle is missing {name}")
        cache = _native_cache_path(root, name, adapter)
        dataset_id, split = DATASET_SOURCES[name]
        expected = {"dataset_id": dataset_id, "split": split, "subset": "default", "records": DATASET_SIZES[name]}
        if any(entry.get(key) != value for key, value in expected.items()):
            raise ValueError(f"{name}: offline bundle dataset metadata mismatch")
        if (
            entry.get("cache_path") != cache.relative_to(root).as_posix()
            or cache.resolve().parent != (root / "datasets").resolve()
        ):
            raise ValueError(f"{name}: offline cache path does not match the pinned native loader")
        if (root / "datasets").is_symlink():
            raise ValueError("Offline datasets directory must not be a symlink")
        actual_files = _file_inventory(cache)
        if entry.get("files") != actual_files:
            raise ValueError(f"{name}: offline cache file list, size or SHA-256 mismatch")
        selected[name] = entry
    # All selected paths/bytes must pass before loading any adapter. Each loader
    # is guaranteed its exact cache hit, including subset/split/source/revision.
    for name, adapter in adapters:
        _validate_samples(name, adapter.load_dataset())
    return {
        "bundle_dir": str(root),
        "schema_version": BUNDLE_SCHEMA,
        "evalscope_version": EVALSCOPE_VERSION,
        "datasets": selected,
    }


def preload_datasets(names: list[str], dataset_dir: Path | None = None, output_dir: Path | None = None) -> dict:
    cache_root, adapters = _adapters(names, dataset_dir)
    print(f"Dataset cache: {cache_root / 'datasets'}", flush=True)
    summary = {"dataset_dir": str(cache_root), "datasets": {}}
    sources = {}
    for index, (name, adapter) in enumerate(adapters, start=1):
        print(f"[{index}/{len(names)}] Loading {name}: {adapter.dataset_id} ({adapter.eval_split})", flush=True)
        subsets = adapter.load_dataset()
        count = _validate_samples(name, subsets)
        summary["datasets"][name] = {
            "dataset_id": adapter.dataset_id,
            "split": adapter.eval_split,
            "records": count,
        }
        if output_dir is not None:
            sources[name] = _native_cache_path(cache_root, name, adapter)
        print(f"[{index}/{len(names)}] READY {name}: {count} questions", flush=True)
    if output_dir is not None:
        _export_bundle(output_dir, sources, summary)
        summary["offline_bundle"] = str(output_dir)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        check_evalscope_version()
        summary = preload_datasets(args.datasets, args.dataset_dir, args.output_dir)
    except Exception as exc:
        print(f"EVALSCOPE_DATASETS=FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("EVALSCOPE_DATASETS=READY " + json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

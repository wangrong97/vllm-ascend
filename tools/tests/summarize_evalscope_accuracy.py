# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Audit predictions and read official accuracy reports from EvalScope 1.12.0.

The on-disk schema is ModelResult.model_dump() from the pinned upstream source:
https://github.com/modelscope/evalscope/blob/v1.12.0/evalscope/api/evaluator/cache.py
Report v2 persists primary_metric_identity and metrics (not a top-level score):
https://github.com/modelscope/evalscope/blob/v1.12.0/evalscope/report/report.py
This checks completion and report consistency; EvalScope alone performs grading.
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from types import MappingProxyType

DATASET_SIZES = MappingProxyType({"aime24": 30, "aime25": 30, "gpqa_diamond": 198})
STOP_REASONS = frozenset({"stop", "max_tokens", "model_length", "tool_calls", "content_filter", "unknown"})


def _require(condition: bool, location: str, message: str) -> None:
    if not condition:
        raise ValueError(f"{location}: invalid EvalScope 1.12.0 prediction: {message}")


def _read_prediction(record: object, location: str) -> tuple[int, str, bool, int | None, bool]:
    _require(isinstance(record, dict), location, "expected ModelResult object")
    index = record.get("index")
    _require(type(index) is int and index >= 0, location, "missing/nonnegative integer index")
    _require(isinstance(record.get("model"), str), location, "missing model")
    _require(isinstance(record.get("messages"), list), location, "missing messages")
    output = record.get("model_output")
    _require(isinstance(output, dict), location, "missing model_output")
    choices = output.get("choices")
    _require(isinstance(choices, list) and len(choices) == 1, location, "expected exactly one choice")
    choice = choices[0]
    _require(isinstance(choice, dict), location, "choice must be an object")
    stop_reason = choice.get("stop_reason")
    _require(isinstance(stop_reason, str) and stop_reason in STOP_REASONS, location, "unknown stop_reason schema")
    message = choice.get("message")
    _require(isinstance(message, dict) and message.get("role") == "assistant", location, "missing assistant message")
    content = message.get("content")
    _require(isinstance(content, (str, list)), location, "missing assistant content")
    if isinstance(content, str):
        final_text = content
    else:
        parts = []
        for part in content:
            _require(isinstance(part, dict), location, "content part must be an object")
            kind = part.get("type")
            _require(kind in ("text", "reasoning"), location, "unsupported content part type")
            field = "text" if kind == "text" else "reasoning"
            _require(isinstance(part.get(field), str), location, f"missing content part {field}")
            if kind == "text":
                parts.append(part["text"])
        final_text = "\n".join(parts)
    _require("usage" in output, location, "missing usage field (use null for unknown usage)")
    usage = output["usage"]
    output_tokens = None
    if usage is not None:
        _require(isinstance(usage, dict), location, "usage must be an object or null")
        for field in ("input_tokens", "output_tokens", "total_tokens"):
            value = usage.get(field)
            _require(type(value) is int and value >= 0, location, f"invalid usage.{field}")
        output_tokens = usage["output_tokens"]
    _require("error" in output, location, "missing error field")
    error = output["error"]
    _require(error is None or isinstance(error, str), location, "error must be a string or null")
    return index, stop_reason, not final_text.strip(), output_tokens, bool(error)


def _read_report(path: Path, dataset: str, model: str, records: int, expected: int) -> dict:
    """Read the native default-subset accuracy:mean report without scoring it again.

    BenchmarkMeta defaults to mean for all three supported benchmarks. Repeats
    duplicate/reindex samples; Mean.num counts attempts, unlike pass_at_k.num
    which counts question groups. Never silently substitute one aggregation.
    https://github.com/modelscope/evalscope/blob/v1.12.0/evalscope/metrics/aggregators/aggregators.py
    """

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(f"{path}: invalid EvalScope 1.12.0 report: {message}")

    def count(value: object, field: str) -> int:
        require(type(value) is int and value >= 0, f"{field} must be a nonnegative integer")
        return value

    def score(value: object, field: str) -> float:
        require(
            type(value) in (int, float) and 0 <= value <= 1 and math.isfinite(value),
            f"{field} must be a finite accuracy ratio in [0, 1]",
        )
        return float(value)

    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"Cannot read official EvalScope report {path}: {exc}") from exc
    require(isinstance(report, dict), "expected Report object")
    require(type(report.get("schema_version")) is int and report["schema_version"] == 2, "expected schema_version=2")
    require(report.get("dataset_name") == dataset, "dataset_name does not match prediction dataset")
    require(report.get("model_name") == model, "model_name does not match prediction directory")
    identity = {"name": "accuracy", "aggregation": "mean", "dimensions": {}}
    require(report.get("primary_metric_identity") == identity, "expected primary accuracy:mean with no dimensions")
    require(report.get("primary_metric_unavailable_reason") is None, "primary metric is unavailable")
    metrics = report.get("metrics")
    require(isinstance(metrics, list) and all(isinstance(item, dict) for item in metrics), "missing metrics list")
    primary = [item for item in metrics if item.get("identity") == identity]
    require(len(primary) == 1, "primary identity must match exactly one metric")
    metric = primary[0]
    semantics = metric.get("semantics")
    require(
        isinstance(semantics, dict)
        and semantics.get("kind") == "quality"
        and semantics.get("semantic_id") == "quality.accuracy.ratio",
        "primary metric is not accuracy quality",
    )
    result_score = score(metric.get("score"), "metric.score")
    result_num = count(metric.get("num"), "metric.num")
    require(count(report.get("num"), "report.num") == result_num, "report and metric sample counts disagree")
    categories = metric.get("categories")
    require(
        isinstance(categories, list) and len(categories) == 1 and isinstance(categories[0], dict),
        "expected one category for the default subset",
    )
    category = categories[0]
    subsets = category.get("subsets")
    require(
        isinstance(subsets, list) and len(subsets) == 1 and isinstance(subsets[0], dict),
        "expected exactly one default subset",
    )
    subset = subsets[0]
    require(subset.get("name") == "default" and subset.get("is_aggregate") is False, "expected the real default subset")
    for name, node in (("category", category), ("subset", subset)):
        require(count(node.get("num"), f"{name}.num") == result_num, f"{name} and metric sample counts disagree")
        require(
            math.isclose(score(node.get("score"), f"{name}.score"), result_score, rel_tol=0, abs_tol=1e-8),
            f"{name} and metric scores disagree",
        )
    for name, node in (("metric", metric), ("category", category)):
        require(
            math.isclose(score(node.get("macro_score"), f"{name}.macro_score"), result_score, rel_tol=0, abs_tol=1e-8),
            f"{name} macro_score disagrees for the single subset",
        )
    execution = report.get("execution_summary")
    require(isinstance(execution, dict), "missing native execution_summary")
    execution_subsets = execution.get("subsets")
    require(
        isinstance(execution_subsets, dict)
        and set(execution_subsets) == {"default"}
        and isinstance(execution_subsets["default"], dict),
        "expected default execution subset",
    )
    for field in ("requested", "succeeded", "errored"):
        value = count(execution.get(field), f"execution_summary.{field}")
        require(
            count(execution_subsets["default"].get(field), f"execution subset.{field}") == value,
            f"execution {field} disagrees with subset",
        )
    require(type(execution.get("incomplete")) is bool, "missing execution incomplete flag")
    require(execution["succeeded"] <= execution["requested"], "more successful than requested samples")
    require(execution["errored"] <= execution["requested"], "more errored than requested samples")
    require(
        execution["incomplete"] == (execution["succeeded"] < execution["requested"]),
        "inconsistent execution incomplete flag",
    )
    mismatch = not (result_num == records == expected == execution["requested"] == execution["succeeded"])
    return {
        "report_file": str(path),
        "score": result_score,
        "metric": identity["name"],
        "aggregation": identity["aggregation"],
        "metric_identity": identity,
        "num": result_num,
        "execution_summary": execution,
        "report_count_mismatch": mismatch,
        "report_invalid": bool(mismatch or execution["incomplete"] or execution["errored"]),
    }


def summarize_run(work_dir: Path, datasets: list[str], expected_counts: dict[str, int]) -> dict:
    """Summarize the default subset of the requested single-model benchmarks.

    Missing/malformed predictions or official reports raise ValueError. Artifacts with context,
    filtering, explicit model errors, unexpected completion reasons, duplicate
    samples, or count mismatches have invalid=True. Truncation and an empty final
    answer alone do not invalidate a run.
    """
    if not datasets or len(set(datasets)) != len(datasets):
        raise ValueError("Specify at least one dataset, without duplicates")
    summaries = {}
    model_dirs = set()
    for dataset in datasets:
        if dataset not in DATASET_SIZES:
            raise ValueError(f"Unsupported dataset: {dataset}")
        expected = expected_counts.get(dataset)
        if type(expected) is not int or expected <= 0:
            raise ValueError(f"Expected a positive record count for {dataset}")
        paths = sorted((work_dir / "predictions").glob(f"*/{dataset}_default.jsonl"))
        if len(paths) != 1:
            raise ValueError(f"Expected one prediction file for {dataset} in {work_dir}; found {len(paths)}")
        path = paths[0]
        model_dirs.add(path.parent)
        indices = set()
        stops = Counter()
        records = empty_final = errors = duplicates = known = token_total = 0
        try:
            with path.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    location = f"{path}:{line_number}"
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"{location}: malformed prediction JSON") from exc
                    index, stop, empty, tokens, error = _read_prediction(record, location)
                    records += 1
                    duplicates += index in indices
                    indices.add(index)
                    stops[stop] += 1
                    empty_final += empty
                    errors += error
                    if tokens is not None:
                        known += 1
                        token_total += tokens
        except (OSError, UnicodeError) as exc:
            raise ValueError(f"Cannot read prediction file {path}: {exc}") from exc
        if not records:
            raise ValueError(f"Empty prediction file: {path}")
        report = _read_report(
            work_dir / "reports" / path.parent.name / f"{dataset}.json", dataset, path.parent.name, records, expected
        )
        mismatch = records != expected
        # DatasetDict/DataLoader reindex after repeating: every attempt has a
        # unique ID in [0, expected), even when the original question repeats.
        index_mismatch = indices != set(range(expected))
        summaries[dataset] = {
            **report,
            "prediction_file": str(path),
            "records": records,
            "expected_records": expected,
            "count_mismatch": mismatch,
            "duplicate_indices": duplicates,
            "index_mismatch": index_mismatch,
            "stop_reasons": dict(sorted(stops.items())),
            "max_tokens": stops["max_tokens"],
            "empty_final_content": empty_final,
            "model_length": stops["model_length"],
            "content_filter": stops["content_filter"],
            "unknown": stops["unknown"],
            "tool_calls": stops["tool_calls"],
            "errors": errors,
            "output_tokens": {"known_records": known, "unknown_records": records - known, "total": token_total},
            "invalid": bool(
                mismatch
                or index_mismatch
                or report["report_invalid"]
                or duplicates
                or errors
                or stops["model_length"]
                or stops["content_filter"]
                or stops["unknown"]
                or stops["tool_calls"]
            ),
        }
    if len(model_dirs) != 1:
        raise ValueError("Requested datasets belong to different model prediction directories")
    return {"invalid": any(item["invalid"] for item in summaries.values()), "datasets": summaries}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work_dir", type=Path)
    parser.add_argument("--datasets", nargs="+", choices=tuple(DATASET_SIZES), default=["aime25"])
    parser.add_argument("--limit", type=int, help="Number of questions requested per dataset")
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if args.repeats <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error("--limit and --repeats must be positive")
    expected = {
        dataset: min(DATASET_SIZES[dataset], args.limit or DATASET_SIZES[dataset]) * args.repeats
        for dataset in args.datasets
    }
    try:
        summary = summarize_run(args.work_dir, args.datasets, expected)
    except ValueError as exc:
        parser.exit(1, f"Accuracy audit failed: {exc}\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return int(summary["invalid"])


if __name__ == "__main__":
    raise SystemExit(main())

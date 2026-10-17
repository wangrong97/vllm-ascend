# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "tools/tests/summarize_evalscope_accuracy.py"
summarize_run = runpy.run_path(str(SCRIPT))["summarize_run"]


def _prediction(index=0, stop="stop", content="\\boxed{70}", tokens=1024, error=None):
    # EvalScope v1.12.0 CacheManager.save_prediction_cache serializes ModelResult,
    # not an OpenAI response: top-level index/model_output and choice.stop_reason.
    # https://github.com/modelscope/evalscope/blob/v1.12.0/evalscope/api/evaluator/cache.py
    return {
        "index": index,
        "model": "baseline",
        "model_output": {
            "id": "cmpl-test",
            "model": "baseline",
            "choices": [
                {
                    "message": {"role": "assistant", "content": content, "source": "generate"},
                    "stop_reason": stop,
                    "logprobs": None,
                }
            ],
            "usage": None
            if tokens is None
            else {"input_tokens": 50, "output_tokens": tokens, "total_tokens": 50 + tokens},
            "time": 1.0,
            "metadata": None,
            "error": error,
        },
        "messages": [{"role": "user", "content": "Solve the problem."}],
        "agent_trace": None,
        "metadata": {},
    }


def _write(tmp_path, records, dataset="aime25", model="baseline"):
    path = tmp_path / "predictions" / model / f"{dataset}_default.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    _write_report(tmp_path, _report(len(records), dataset=dataset, model=model), dataset, model)
    return path


def _report(num, *, dataset="aime25", model="baseline", score=0.6667):
    # Native v1.12.0 Report.to_dict() serializes schema v2; score is a Python
    # property, NOT a JSON root field. Primary identity selects the metric.
    # https://github.com/modelscope/evalscope/blob/v1.12.0/evalscope/report/report.py
    identity = {"name": "accuracy", "aggregation": "mean", "dimensions": {}}
    return {
        "schema_version": 2,
        "name": f"{model}@{dataset}",
        "dataset_name": dataset,
        "model_name": model,
        "num": num,
        "primary_metric_identity": identity,
        "primary_metric_unavailable_reason": None,
        "metrics": [
            {
                "identity": identity.copy(),
                "legacy_name": None,
                "num": num,
                "score": score,
                "macro_score": score,
                "semantics": {
                    "semantic_id": "quality.accuracy.ratio",
                    "metric_name": "Accuracy",
                    "kind": "quality",
                    "direction": "higher_is_better",
                    "raw_unit": "ratio",
                    "value_range": {"min": 0, "max": 1},
                    "display_kind": "percent",
                    "display_multiplier": 100,
                    "display_unit": "%",
                    "display_precision": 2,
                },
                "categories": [
                    {
                        "name": ["default"],
                        "num": num,
                        "score": score,
                        "macro_score": score,
                        "subsets": [{"name": "default", "num": num, "score": score, "is_aggregate": False}],
                    }
                ],
            }
        ],
        "execution_summary": {
            "requested": num,
            "succeeded": num,
            "errored": 0,
            "incomplete": False,
            "subsets": {"default": {"requested": num, "succeeded": num, "errored": 0}},
        },
    }


def _write_report(tmp_path, report, dataset="aime25", model="baseline"):
    path = tmp_path / "reports" / model / f"{dataset}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_pinned_cache_schema_counts_usage_truncation_and_reasoning_separately(tmp_path):
    _write(
        tmp_path,
        [
            _prediction(),
            _prediction(1, "max_tokens", [{"type": "reasoning", "reasoning": "Still thinking"}], 8192),
            _prediction(
                2,
                content=[{"type": "reasoning", "reasoning": "Calculation"}, {"type": "text", "text": "\\boxed{70}"}],
                tokens=None,
            ),
            _prediction(3, content="   ", tokens=0),
        ],
    )
    result = summarize_run(tmp_path, ["aime25"], {"aime25": 4})
    assert result["invalid"] is False
    counts = result["datasets"]["aime25"]
    assert counts["records"] == 4
    assert counts["stop_reasons"] == {"max_tokens": 1, "stop": 3}
    assert counts["max_tokens"] == 1
    assert counts["empty_final_content"] == 2
    assert counts["output_tokens"] == {"known_records": 3, "unknown_records": 1, "total": 9216}
    assert counts["score"] == 0.6667
    assert counts["metric"] == "accuracy" and counts["aggregation"] == "mean"
    assert counts["num"] == 4 and counts["report_count_mismatch"] is False
    assert json.loads(json.dumps(result)) == result


@pytest.mark.parametrize("stop,error", [("model_length", None), ("content_filter", None), ("stop", "request failed")])
def test_context_filter_or_explicit_errors_invalidate_run(tmp_path, stop, error):
    _write(tmp_path, [_prediction(stop=stop, error=error)])
    result = summarize_run(tmp_path, ["aime25"], {"aime25": 1})
    assert result["invalid"] is True
    counts = result["datasets"]["aime25"]
    assert counts["errors"] == int(error is not None)
    assert counts["model_length"] == int(stop == "model_length")
    assert counts["content_filter"] == int(stop == "content_filter")


@pytest.mark.parametrize("stop", ["unknown", "tool_calls"])
def test_unexpected_plaintext_completion_cannot_look_valid(tmp_path, stop):
    _write(tmp_path, [_prediction(stop=stop)])
    result = summarize_run(tmp_path, ["aime25"], {"aime25": 1})
    assert result["invalid"] is True
    assert result["datasets"]["aime25"][stop] == 1


def test_incomplete_or_duplicate_records_cannot_look_complete(tmp_path):
    _write(tmp_path, [_prediction(), _prediction()])
    counts = summarize_run(tmp_path, ["aime25"], {"aime25": 3})["datasets"]["aime25"]
    assert counts["invalid"] is True
    assert counts["count_mismatch"] is True
    assert counts["duplicate_indices"] == 1
    assert summarize_run(tmp_path, ["aime25"], {"aime25": 2})["invalid"] is True


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row.pop("model_output"),
        lambda row: row.update(index=True),
        lambda row: row["model_output"].update(choices=[]),
        lambda row: row["model_output"].pop("usage"),
        lambda row: row["model_output"].pop("error"),
        lambda row: row["model_output"]["usage"].update(output_tokens=-1),
        lambda row: row["model_output"]["usage"].update(output_tokens=True),
        lambda row: row["model_output"]["choices"][0].update(stop_reason="length"),
        lambda row: row["model_output"]["choices"][0]["message"].update(content=None),
        lambda row: row["model_output"]["choices"][0]["message"].update(content=[{"type": "text"}]),
    ],
)
def test_malformed_pinned_schema_fails_loudly(tmp_path, mutation):
    record = copy.deepcopy(_prediction())
    mutation(record)
    _write(tmp_path, [record])
    with pytest.raises(ValueError, match="invalid EvalScope 1.12.0 prediction"):
        summarize_run(tmp_path, ["aime25"], {"aime25": 1})


@pytest.mark.parametrize("content", ["", "{bad json}\n", "\n"])
def test_empty_or_malformed_artifact_is_not_zero_success(tmp_path, content):
    path = _write(tmp_path, [])
    path.write_text(content)
    with pytest.raises(ValueError):
        summarize_run(tmp_path, ["aime25"], {"aime25": 1})


def test_missing_or_ambiguous_model_artifacts_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="found 0"):
        summarize_run(tmp_path, ["aime25"], {"aime25": 1})
    _write(tmp_path, [_prediction()])
    _write(tmp_path, [_prediction()], model="vq2a8")
    with pytest.raises(ValueError, match="found 2"):
        summarize_run(tmp_path, ["aime25"], {"aime25": 1})


def test_cannot_combine_datasets_from_different_models(tmp_path):
    _write(tmp_path, [_prediction()])
    _write(tmp_path, [_prediction()], dataset="aime24", model="vq2a8")
    with pytest.raises(ValueError, match="different model"):
        summarize_run(tmp_path, ["aime25", "aime24"], {"aime25": 1, "aime24": 1})


@pytest.mark.parametrize("stop,expected_code", [("stop", 0), ("model_length", 1)])
def test_cli_reports_counts_and_failure_status(tmp_path, stop, expected_code):
    _write(tmp_path, [_prediction(stop=stop)])
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path), "--limit", "1"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == expected_code
    assert json.loads(result.stdout)["datasets"]["aime25"]["expected_records"] == 1


def test_repeats_use_unique_attempt_indices_and_mean_attempt_count(tmp_path):
    # Official repeats=2, limit=2 expands to ids 0,1,2,3; each pair shares
    # group_id. Mean counts all attempts, pass@k would count two groups.
    _write(tmp_path, [_prediction(index) for index in (3, 1, 0, 2)])
    report = _report(4, score=0.25)
    diagnostic = copy.deepcopy(report["metrics"][0])
    diagnostic["identity"] = {"name": "output_tokens", "aggregation": "mean", "dimensions": {}}
    diagnostic["score"] = 1024
    diagnostic["semantics"]["kind"] = "diagnostic"
    report["metrics"].insert(0, diagnostic)
    _write_report(tmp_path, report)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path), "--limit", "2", "--repeats", "2"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)["datasets"]["aime25"]
    assert summary["score"] == 0.25 and summary["num"] == 4
    assert summary["records"] == summary["expected_records"] == 4
    assert summary["duplicate_indices"] == 0 and summary["index_mismatch"] is False


@pytest.mark.parametrize("indices", [(0, 2), (1, 2)])
def test_missing_or_out_of_range_indices_invalidate_even_matching_counts(tmp_path, indices):
    _write(tmp_path, [_prediction(index) for index in indices])
    summary = summarize_run(tmp_path, ["aime25"], {"aime25": 2})
    assert summary["invalid"] is True
    assert summary["datasets"]["aime25"]["index_mismatch"] is True


@pytest.mark.parametrize("dataset", ["aime24", "aime25", "gpqa_diamond"])
def test_official_scores_are_read_verbatim_without_regrading_predictions(tmp_path, dataset):
    _write(tmp_path, [_prediction(content="There is no parsable answer.")], dataset)
    _write_report(tmp_path, _report(1, dataset=dataset, score=0.0), dataset)
    summary = summarize_run(tmp_path, [dataset], {dataset: 1})
    assert summary["invalid"] is False
    assert summary["datasets"][dataset]["score"] == 0.0


@pytest.mark.parametrize("content", [None, "{invalid", "null", "[]"])
def test_missing_or_malformed_official_report_never_succeeds(tmp_path, content):
    _write(tmp_path, [_prediction()])
    path = tmp_path / "reports/baseline/aime25.json"
    if content is None:
        path.unlink()
    else:
        path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="report|Report"):
        summarize_run(tmp_path, ["aime25"], {"aime25": 1})


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r.update(schema_version=1),
        lambda r: r.update(dataset_name="aime24"),
        lambda r: r.update(model_name="other"),
        lambda r: r.update(primary_metric_identity=None),
        lambda r: r["primary_metric_identity"].update(aggregation="pass_at_k", dimensions={"k": 2}),
        lambda r: r.update(primary_metric_unavailable_reason="metric missing"),
        lambda r: r.update(metrics=[]),
        lambda r: r["metrics"].append(copy.deepcopy(r["metrics"][0])),
        lambda r: r["metrics"][0]["semantics"].update(kind="diagnostic"),
        lambda r: r["metrics"][0].update(score=float("nan")),
        lambda r: r["metrics"][0].update(score=float("inf")),
        lambda r: r["metrics"][0].update(score=66.67),
        lambda r: r["metrics"][0].update(score=True),
        lambda r: r["metrics"][0].update(num=True),
        lambda r: r.update(num=2),
        lambda r: r["metrics"][0]["categories"][0].update(num=2),
        lambda r: r["metrics"][0]["categories"][0].update(score=0.5),
        lambda r: r["metrics"][0]["categories"][0]["subsets"][0].update(is_aggregate=True),
        lambda r: r["metrics"][0]["categories"][0]["subsets"][0].update(num=2),
        lambda r: r["metrics"][0]["categories"][0]["subsets"][0].update(score=0.5),
        lambda r: r["metrics"][0].update(macro_score=0.5),
        lambda r: r.update(execution_summary=None),
        lambda r: r["execution_summary"].update(incomplete=True),
        lambda r: r["execution_summary"]["subsets"]["default"].update(succeeded=0),
    ],
)
def test_inconsistent_or_unsupported_official_report_is_rejected(tmp_path, mutation):
    _write(tmp_path, [_prediction()])
    report = _report(1)
    mutation(report)
    _write_report(tmp_path, report)
    with pytest.raises(ValueError, match="invalid EvalScope 1.12.0 report"):
        summarize_run(tmp_path, ["aime25"], {"aime25": 1})


def test_report_with_fewer_graded_attempts_cannot_look_complete(tmp_path):
    _write(tmp_path, [_prediction(0), _prediction(1)])
    report = _report(1)
    report["execution_summary"] = {
        "requested": 2,
        "succeeded": 1,
        "errored": 1,
        "incomplete": True,
        "subsets": {"default": {"requested": 2, "succeeded": 1, "errored": 1}},
    }
    _write_report(tmp_path, report)
    summary = summarize_run(tmp_path, ["aime25"], {"aime25": 2})
    assert summary["invalid"] is True
    assert summary["datasets"]["aime25"]["report_count_mismatch"] is True
    assert summary["datasets"]["aime25"]["num"] == 1

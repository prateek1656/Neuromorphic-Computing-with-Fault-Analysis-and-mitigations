"""Post-experiment result validation: given a finished run's already-parsed
metrics.json/config.json (no file I/O here - see experiments/validate_results.py
for that), is this something safe to report, or does it show signs of
corruption or a broken pipeline?

Pure functions over dicts, deliberately decoupled from disk so tests can feed
synthetic data directly. Every check here targets a real property of the
actual schema experiments/run.py::run_experiment() produces (verified by
reading real generated results/*/metrics.json files, not assumed) - not a
hypothetical field.

Structural checks (category A) gate everything else: a metrics.json with
missing keys or mismatched list lengths can't be meaningfully checked for
numeric sanity (an IndexError, not a signal, would result) - if any
structural check fails, validate() stops there and returns only those
results.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from itertools import pairwise

EXPECTED_METRICS_KEYS = (
    "epoch",
    "batch",
    "loss",
    "accuracy",
    "health",
    "faults",
    "soft_mitigations",
    "remappings",
    "layer_resets",
    "detection_cost",
)

# Real bookkeeping counters (system.py: total_faults_injected, stats[...],
# HealthMonitor.total_detection_cost) - only ever accumulate, never reset
# mid-run. A decrease means real corruption, not a modeling choice.
_CUMULATIVE_COUNTER_KEYS = (
    "faults",
    "soft_mitigations",
    "remappings",
    "layer_resets",
    "detection_cost",
)

_RANDOM_CHANCE_ACCURACY = 10.0  # CIFAR-10/MNIST: 10 classes


class Level(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass
class CheckResult:
    name: str
    level: Level
    message: str


def overall_verdict(results: list[CheckResult]) -> Level:
    if any(r.level == Level.FAIL for r in results):
        return Level.FAIL
    if any(r.level == Level.WARN for r in results):
        return Level.WARN
    return Level.PASS


def validate(metrics: dict, config: dict, results_name: str | None = None) -> list[CheckResult]:
    results = _check_structural(metrics, config, results_name)
    if any(r.level == Level.FAIL for r in results):
        return results  # can't meaningfully check numeric properties on a broken schema

    results += _check_numeric_sanity(metrics)
    results += _check_config_consistency(metrics, config)
    results += _check_qualitative(metrics)
    results += _check_outliers(metrics)
    return results


def _check_structural(metrics: dict, config: dict, results_name: str | None) -> list[CheckResult]:
    results = []

    missing_keys = set(EXPECTED_METRICS_KEYS) - set(metrics)
    if missing_keys:
        results.append(
            CheckResult(
                "structural.missing_keys",
                Level.FAIL,
                f"metrics.json is missing expected keys: {sorted(missing_keys)}",
            )
        )
        return results  # nothing else here is safe to check

    lengths = {key: len(metrics[key]) for key in EXPECTED_METRICS_KEYS}
    if len(set(lengths.values())) > 1:
        results.append(
            CheckResult(
                "structural.mismatched_lengths",
                Level.FAIL,
                f"metrics.json lists have mismatched lengths (partially-written/corrupted file?): {lengths}",
            )
        )
        return results

    results.append(
        CheckResult("structural.schema", Level.PASS, "all expected keys present, lengths match")
    )

    epochs = metrics["epoch"]
    batches = metrics["batch"]
    if any(e < 0 for e in epochs) or any(b < 0 for b in batches):
        results.append(
            CheckResult(
                "structural.negative_counters", Level.FAIL, "epoch/batch contain negative values"
            )
        )
    elif any(b2 < b1 for b1, b2 in pairwise(batches)):
        results.append(
            CheckResult("structural.batch_not_monotonic", Level.FAIL, "batch is not non-decreasing")
        )
    else:
        results.append(CheckResult("structural.epoch_batch", Level.PASS, "epoch/batch are sane"))

    if results_name is not None and config.get("name") != results_name:
        results.append(
            CheckResult(
                "structural.name_mismatch",
                Level.FAIL,
                f"config.json name={config.get('name')!r} does not match results directory {results_name!r} "
                "(copy/paste mixup between two runs?)",
            )
        )
    else:
        results.append(
            CheckResult(
                "structural.name_match", Level.PASS, "config name matches results directory"
            )
        )

    return results


def _all_finite(values) -> bool:
    return all(math.isfinite(v) for v in values)


def _check_numeric_sanity(metrics: dict) -> list[CheckResult]:
    results = []

    for key in ("loss", "accuracy", "health", "detection_cost"):
        if not _all_finite(metrics[key]):
            results.append(
                CheckResult(
                    f"numeric.{key}_finite", Level.FAIL, f"{key} contains NaN/Inf - real corruption"
                )
            )
        else:
            results.append(
                CheckResult(f"numeric.{key}_finite", Level.PASS, f"{key} is finite throughout")
            )

    for key in ("accuracy", "health"):
        if _all_finite(metrics[key]) and not all(0.0 <= v <= 100.0 for v in metrics[key]):
            results.append(
                CheckResult(
                    f"numeric.{key}_bounds", Level.FAIL, f"{key} has a value outside [0, 100]"
                )
            )
        elif _all_finite(metrics[key]):
            results.append(
                CheckResult(f"numeric.{key}_bounds", Level.PASS, f"{key} stays within [0, 100]")
            )

    if _all_finite(metrics["loss"]) and any(v < 0 for v in metrics["loss"]):
        results.append(
            CheckResult(
                "numeric.loss_non_negative",
                Level.FAIL,
                "loss contains a negative value - impossible for cross-entropy loss",
            )
        )
    elif _all_finite(metrics["loss"]):
        results.append(
            CheckResult("numeric.loss_non_negative", Level.PASS, "loss stays non-negative")
        )

    for key in _CUMULATIVE_COUNTER_KEYS:
        values = metrics[key]
        if any(v < 0 for v in values):
            results.append(
                CheckResult(
                    f"numeric.{key}_non_negative", Level.FAIL, f"{key} contains a negative value"
                )
            )
            continue
        if any(v2 < v1 for v1, v2 in pairwise(values)):
            results.append(
                CheckResult(
                    f"numeric.{key}_monotonic",
                    Level.FAIL,
                    f"{key} decreases at some point - this counter only ever accumulates in system.py",
                )
            )
        else:
            results.append(
                CheckResult(f"numeric.{key}_monotonic", Level.PASS, f"{key} is non-decreasing")
            )

    return results


def _check_config_consistency(metrics: dict, config: dict) -> list[CheckResult]:
    results = []
    mitigation = config.get("mitigation", {})
    fault = config.get("fault", {})
    retraining = config.get("retraining", {})

    flag_to_metric = {
        "enable_soft_mitigation": "soft_mitigations",
        "enable_remapping": "remappings",
        "enable_layer_reset": "layer_resets",
    }
    for flag, metric_key in flag_to_metric.items():
        if mitigation.get(flag) is False and any(v != 0 for v in metrics[metric_key]):
            results.append(
                CheckResult(
                    f"consistency.{flag}",
                    Level.FAIL,
                    f"mitigation.{flag} is False but {metric_key} is non-zero - dispatch.py's "
                    "no-unconditional-fallthrough guarantee did not hold for this run",
                )
            )
        else:
            results.append(
                CheckResult(
                    f"consistency.{flag}", Level.PASS, f"{metric_key} consistent with {flag}"
                )
            )

    if (
        fault.get("density") == 0
        and fault.get("failure_model") == "density"
        and any(v != 0 for v in metrics["faults"])
    ):
        results.append(
            CheckResult(
                "consistency.zero_density_baseline",
                Level.FAIL,
                "fault.density=0 but faults is non-zero - a 'no-fault baseline' run was not actually clean",
            )
        )
    else:
        results.append(
            CheckResult(
                "consistency.zero_density_baseline", Level.PASS, "no-fault baseline stays clean"
            )
        )

    if retraining.get("enable_fault_aware") is True:
        faults = metrics["faults"]
        nonzero = [v for v in faults if v != 0]
        if nonzero and len(set(nonzero)) > 1:
            results.append(
                CheckResult(
                    "consistency.fault_aware_retraining_single_injection",
                    Level.FAIL,
                    f"retraining.enable_fault_aware=True but faults changed across multiple non-zero "
                    f"values ({sorted(set(nonzero))}) - implies periodic re-injection happened",
                )
            )
        else:
            results.append(
                CheckResult(
                    "consistency.fault_aware_retraining_single_injection",
                    Level.PASS,
                    "faults count stayed fixed after the one-time injection",
                )
            )

    return results


def _check_qualitative(metrics: dict) -> list[CheckResult]:
    results = []
    accuracy = metrics["accuracy"]
    loss = metrics["loss"]

    if accuracy[-1] <= _RANDOM_CHANCE_ACCURACY + 5.0:
        results.append(
            CheckResult(
                "qualitative.final_accuracy",
                Level.WARN,
                f"final accuracy ({accuracy[-1]:.2f}%) does not clearly exceed random chance "
                f"({_RANDOM_CHANCE_ACCURACY}%) - no clear evidence of learning",
            )
        )
    else:
        results.append(
            CheckResult(
                "qualitative.final_accuracy", Level.PASS, "final accuracy clears random chance"
            )
        )

    if loss[-1] >= loss[0]:
        results.append(
            CheckResult(
                "qualitative.loss_trend",
                Level.WARN,
                f"loss did not decrease over the run (first={loss[0]:.4f}, last={loss[-1]:.4f})",
            )
        )
    else:
        results.append(
            CheckResult("qualitative.loss_trend", Level.PASS, "loss decreased over the run")
        )

    for key in ("accuracy", "loss"):
        values = metrics[key]
        if len(values) > 1 and len(set(values)) == 1:
            results.append(
                CheckResult(
                    f"qualitative.{key}_flat",
                    Level.WARN,
                    f"{key} is bit-identical across every logged point - possible frozen/broken training loop",
                )
            )
        else:
            results.append(
                CheckResult(f"qualitative.{key}_flat", Level.PASS, f"{key} varies across the run")
            )

    return results


def _check_outliers(metrics: dict) -> list[CheckResult]:
    results = []
    accuracy = metrics["accuracy"]
    loss = metrics["loss"]

    accuracy_jumps = [abs(b - a) for a, b in pairwise(accuracy)]
    if any(jump > 50.0 for jump in accuracy_jumps):
        results.append(
            CheckResult(
                "outlier.accuracy_jump",
                Level.WARN,
                f"accuracy jumped by more than 50 points in one step (max jump={max(accuracy_jumps):.2f})",
            )
        )
    else:
        results.append(
            CheckResult(
                "outlier.accuracy_jump", Level.PASS, "no implausible single-step accuracy jump"
            )
        )

    loss_ratio_jumps = [
        (b / a if a > 0 else float("inf")) for a, b in pairwise(loss) if a > 0 or b > 0
    ]
    if any(ratio > 10.0 or (ratio > 0 and ratio < 0.1) for ratio in loss_ratio_jumps):
        results.append(
            CheckResult(
                "outlier.loss_jump",
                Level.WARN,
                "loss jumped by more than 10x (or dropped to less than 1/10th) in one step",
            )
        )
    else:
        results.append(
            CheckResult("outlier.loss_jump", Level.PASS, "no implausible single-step loss jump")
        )

    return results

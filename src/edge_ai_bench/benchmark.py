"""Reference benchmark runner used until the ONNX adapter is installed."""

from dataclasses import asdict
import json
import platform
import statistics
import time
from typing import Any
from pathlib import Path

from .config import ExperimentConfig
from .synthetic import SyntheticBatch


def _reference_inference(signals: list[list[list[float]]], output_labels: int) -> list[list[float]]:
    outputs = []
    for sample in signals:
        mean = sum(value for channel in sample for value in channel) / (len(sample) * len(sample[0]))
        outputs.append([1.0 / (1.0 + pow(2.718281828, -(mean + label * 0.01))) for label in range(output_labels)])
    return outputs


def run(config: ExperimentConfig, batch: SyntheticBatch) -> dict[str, Any]:
    """Run a deterministic reference benchmark and return a JSON-safe record."""
    for _ in range(config.warmup):
        _reference_inference(batch.signals, config.output_labels)

    durations = []
    for _ in range(config.iterations):
        started = time.perf_counter()
        predictions = _reference_inference(batch.signals, config.output_labels)
        durations.append(time.perf_counter() - started)

    ordered = sorted(durations)
    p50 = ordered[len(ordered) // 2]
    p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
    result = {
        "schema_version": 1,
        "experiment": {**asdict(config), "output_path": str(config.output_path)},
        "dataset": {"name": "synthetic-fixture", "samples": len(batch.signals)},
        "metrics": {"latency_seconds_p50": p50, "latency_seconds_p95": p95, "throughput_samples_per_second": len(batch.signals) / statistics.mean(durations)},
        "artifact": None,
        "validation": None,
        "environment": {"python": platform.python_version(), "system": platform.platform(), "processor": platform.processor()},
        "status": "synthetic_smoke_only",
        "prediction_shape": [len(predictions), len(predictions[0]) if predictions else config.output_labels],
    }
    return result


def write_result(result: dict[str, Any], path: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

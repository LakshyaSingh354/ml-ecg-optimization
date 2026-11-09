"""Compiler and quantization experiments for the checkpoint deployment artifact."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np

from .artifacts import compare_outputs, inspect_artifact
from .checkpoint_baseline import (
    CheckpointBaselineConfig,
    _benchmark_onnx,
    _current_rss_bytes,
    _load_arrays,
    _metrics,
    _select,
    _standardize,
)
from .onnx_runtime import OnnxRuntimeConfig, OnnxRuntimeSession
from .ptbxl import load_metadata, load_statement_map
from .real_baseline import PTBXL_URL


@dataclass(frozen=True)
class OptimizationConfig:
    input_path: Path = Path("artifacts/ecg_hybrid_net_exp12_resnet_medium-fp32.onnx")
    optimized_path: Path = Path("artifacts/ecg_hybrid_net_exp12_resnet_medium-ort-optimized.onnx")
    dynamic_path: Path = Path("artifacts/ecg_hybrid_net_exp12_resnet_medium-int8-dynamic.onnx")
    static_path: Path = Path("artifacts/ecg_hybrid_net_exp12_resnet_medium-int8-static.onnx")
    optimized_dynamic_path: Path = Path("artifacts/ecg_hybrid_net_exp12_resnet_medium-ort-optimized-int8-dynamic.onnx")
    optimized_static_path: Path = Path("artifacts/ecg_hybrid_net_exp12_resnet_medium-ort-optimized-int8-static.onnx")
    result_path: Path = Path("results/ecg-hybrid-net-optimizations.json")
    data_dir: Path = Path("data/ptb-xl")
    train_limit: int = 1000
    test_limit: int = 200
    calibration_limit: int = 100
    warmup: int = 3
    iterations: int = 10
    download_workers: int = 16
    threads: int = 2
    batch_sizes: tuple[int, ...] = (1, 4, 8, 16)
    repeats: int = 3

    def __post_init__(self) -> None:
        if min(self.train_limit, self.test_limit, self.calibration_limit, self.iterations, self.repeats) <= 0:
            raise ValueError("dataset limits and iterations must be positive")
        if self.warmup < 0 or self.threads <= 0:
            raise ValueError("warmup must be non-negative and threads must be positive")


class _CalibrationReader:
    def __init__(self, signals: np.ndarray) -> None:
        self._batches = iter(
            {"ecg": np.ascontiguousarray(signals[start:start + 1]).transpose(0, 2, 1)}
            for start in range(len(signals))
        )

    def get_next(self) -> dict[str, np.ndarray] | None:
        return next(self._batches, None)

    def rewind(self) -> None:
        raise RuntimeError("calibration reader is single-use")


def _load_data(config: OptimizationConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], str]:
    statement_map = load_statement_map(config.data_dir / "scp_statements.csv")
    records = load_metadata(config.data_dir / "ptbxl_database.csv", statement_map)
    train_records = _select(records, 1, 8, config.train_limit)
    test_records = _select(records, 10, 10, config.test_limit)
    calibration_records = train_records[:config.calibration_limit]
    loader_config = CheckpointBaselineConfig(
        data_dir=config.data_dir,
        download_workers=config.download_workers,
    )
    train, _ = _load_arrays(train_records, loader_config)
    calibration, _ = _load_arrays(calibration_records, loader_config)
    test, labels = _load_arrays(test_records, loader_config)
    train, calibration, test, _ = _standardize(train, calibration, test)
    calibration_ids = [record.ecg_id for record in calibration_records]
    calibration_hash = hashlib.sha256(
        json.dumps(calibration_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return calibration, test, labels, calibration_ids, calibration_hash


def _compile_graph(input_path: Path, output_path: Path) -> None:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    options.optimized_model_filepath = str(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    session = ort.InferenceSession(str(input_path), sess_options=options, providers=["CPUExecutionProvider"])
    del session


def _quantize(input_path: Path, dynamic_path: Path, static_path: Path, calibration: np.ndarray) -> None:
    from onnxruntime.quantization import (
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_dynamic,
        quantize_static,
    )

    dynamic_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        str(input_path),
        str(dynamic_path),
        weight_type=QuantType.QInt8,
        per_channel=True,
        extra_options={"DefaultTensorType": "float32"},
    )
    quantize_static(
        str(input_path),
        str(static_path),
        _CalibrationReader(calibration),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=CalibrationMethod.MinMax,
        extra_options={"DefaultTensorType": "float32"},
    )


def _evaluate_artifact(
    name: str,
    path: Path,
    reference_logits: np.ndarray,
    test: np.ndarray,
    labels: np.ndarray,
    config: OptimizationConfig,
    calibration_hash: str,
    runtime_optimization: str = "disabled",
) -> dict[str, Any]:
    artifact = inspect_artifact(path)
    session_started = time.perf_counter()
    session = OnnxRuntimeSession(
        path,
        OnnxRuntimeConfig(threads=config.threads, optimization=runtime_optimization),
    )
    initialization_seconds = time.perf_counter() - session_started
    logits = session.run(np.ascontiguousarray(test).transpose(0, 2, 1))[0]
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    comparison = compare_outputs(reference_logits, logits, atol=1e-1, rtol=1e-1)
    benchmark_runs = [
        _benchmark_onnx(session, test, config.warmup, config.iterations, config.batch_sizes)
        for _ in range(config.repeats)
    ]
    batches = {}
    for batch_size in config.batch_sizes:
        measurements = [run[str(batch_size)] for run in benchmark_runs]
        batches[str(batch_size)] = {
            "batch_size": batch_size,
            "repeats": config.repeats,
            "latency_seconds_p50": float(np.median([item["latency_seconds_p50"] for item in measurements])),
            "latency_seconds_p95": float(np.median([item["latency_seconds_p95"] for item in measurements])),
            "throughput_samples_per_second": float(np.median([item["throughput_samples_per_second"] for item in measurements])),
            "runs": measurements,
        }
    return {
        "name": name,
        "artifact": artifact.as_dict(),
        "metrics": _metrics(probabilities, labels),
        "calibration_hash": calibration_hash,
        "output_validation": {
            "shape": comparison.shape,
            "max_absolute_error": comparison.max_absolute_error,
            "max_relative_error": comparison.max_relative_error,
            "allclose_at_1e-1": comparison.allclose,
        },
        "benchmark": {
            "provider": session.provider,
            "threads": session.threads,
            "optimization": session.optimization,
            "initialization_seconds": initialization_seconds,
            "process_current_rss_bytes": _current_rss_bytes(),
            "batches": batches,
        },
    }


def run_optimizations(config: OptimizationConfig) -> dict[str, Any]:
    """Evaluate four isolated compiler and quantization experiment conditions."""
    calibration, test, labels, calibration_ids, calibration_hash = _load_data(config)
    reference_session = OnnxRuntimeSession(
        config.input_path,
        OnnxRuntimeConfig(threads=config.threads, optimization="disabled"),
    )
    reference_logits = reference_session.run(np.ascontiguousarray(test).transpose(0, 2, 1))[0]
    _compile_graph(config.input_path, config.optimized_path)
    _quantize(config.input_path, config.dynamic_path, config.static_path, calibration)
    _compile_graph(config.dynamic_path, config.optimized_dynamic_path)
    _compile_graph(config.static_path, config.optimized_static_path)
    experiments = [
        {
            "name": "baseline",
            "description": "Original FP32 graph with ORT graph optimization disabled",
            "variants": [("fp32", config.input_path)],
        },
        {
            "name": "compiler_only",
            "description": "Serialized ORT-optimized FP32 graph with runtime optimization disabled",
            "variants": [("fp32_compiled", config.optimized_path)],
        },
        {
            "name": "quantization_only",
            "description": "INT8 graphs quantized from the original FP32 graph",
            "variants": [
                ("int8_dynamic", config.dynamic_path),
                ("int8_static", config.static_path),
            ],
        },
        {
            "name": "compiler_plus_quantization",
            "description": "INT8 graphs quantized from the serialized ORT-optimized graph",
            "variants": [
                ("int8_dynamic_compiled", config.optimized_dynamic_path),
                ("int8_static_compiled", config.optimized_static_path),
            ],
        },
    ]
    evaluated_experiments = []
    for experiment in experiments:
        evaluated_experiments.append({
            **experiment,
            "variants": [
                _evaluate_artifact(
                    variant_name,
                    variant_path,
                    reference_logits,
                    test,
                    labels,
                    config,
                    calibration_hash,
                )
                for variant_name, variant_path in experiment["variants"]
            ],
        })
    variants = [variant for experiment in evaluated_experiments for variant in experiment["variants"]]
    result = {
        "schema_version": 1,
        "experiment": {
            "name": "ecg-hybrid-net-exp12-four-condition-optimization",
            "dataset": "PTB-XL",
            "source": PTBXL_URL,
            "train_samples": config.train_limit,
            "test_samples": len(test),
            "calibration_samples": len(calibration),
            "calibration_split": "train folds 1-8, first deterministic records",
            "calibration_record_ids": calibration_ids,
            "calibration_record_ids_sha256": calibration_hash,
            "threads": config.threads,
            "batch_sizes": list(config.batch_sizes),
            "runtime_optimization": "disabled",
            "compiler_optimization": "ORT_ENABLE_EXTENDED",
            "combined_order": "quantize original graph, then serialize optimized INT8 graph",
            "conditions": [experiment["name"] for experiment in experiments],
        },
        "environment": {
            "python": platform.python_version(),
            "system": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
        "experiments": evaluated_experiments,
        "variants": variants,
        "pareto_frontier": _pareto_frontier(variants),
        "status": "compiler_quantization_experiment",
    }
    config.result_path.parent.mkdir(parents=True, exist_ok=True)
    config.result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _pareto_frontier(variants: list[dict[str, Any]]) -> list[str]:
    """Return non-dominated variants for size, batch-1 latency, and macro F1."""
    points = {
        variant["name"]: (
            variant["artifact"]["size_bytes"],
            variant["benchmark"]["batches"]["1"]["latency_seconds_p50"],
            -variant["metrics"]["macro_f1"],
        )
        for variant in variants
    }
    frontier = []
    for name, point in points.items():
        dominated = any(
            other != name
            and all(points[other][index] <= point[index] for index in range(3))
            and any(points[other][index] < point[index] for index in range(3))
            for other in points
        )
        if not dominated:
            frontier.append(name)
    return frontier

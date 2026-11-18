"""Compiler and quantization experiments for the checkpoint deployment artifact."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import gc
from typing import Any

import numpy as np

from .artifacts import compare_outputs, inspect_artifact
from .checkpoint_baseline import (
    CheckpointBaselineConfig,
    _current_rss_bytes,
    _load_arrays,
    _metrics,
    _select,
)
from .isolated_benchmark import _system_metadata
from .calibration import compare_calibration_manifests
from .metrics import bootstrap_macro_intervals
from .onnx_runtime import ORT_DEFAULT_OPTIMIZATION, OnnxRuntimeConfig, OnnxRuntimeSession
from .pareto import nondominated_names
from .profiler import demonstrate_static_int8_mechanism, profile_model
from .ptbxl import load_metadata, load_statement_map
from .real_baseline import PTBXL_URL
from .reporting import static_int8_interaction


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
    thread_counts: tuple[int, ...] = (1, 2, 4)
    batch_sizes: tuple[int, ...] = (1, 4, 8, 16)
    repeats: int = 3
    evaluation_batch_size: int = 64
    isolated_repeats: int = 3
    quantization_per_channel: bool = True
    calibration_method: str = "minmax"
    bootstrap_resamples: int = 1000
    bootstrap_seed: int = 0
    profile_nodes: bool = True
    profile_iterations: int = 5
    reference_manifest: Path | None = None

    def __post_init__(self) -> None:
        if min(self.train_limit, self.test_limit, self.calibration_limit, self.iterations, self.repeats) <= 0:
            raise ValueError("dataset limits and iterations must be positive")
        if self.warmup < 0 or self.threads <= 0:
            raise ValueError("warmup must be non-negative and threads must be positive")
        if self.evaluation_batch_size <= 0:
            raise ValueError("evaluation_batch_size must be positive")
        if self.isolated_repeats <= 0:
            raise ValueError("isolated_repeats must be positive")
        if not self.thread_counts or any(count <= 0 for count in self.thread_counts):
            raise ValueError("thread_counts must be positive")
        if self.bootstrap_resamples <= 0:
            raise ValueError("bootstrap_resamples must be positive")
        if self.profile_iterations <= 0:
            raise ValueError("profile_iterations must be positive")
        if self.calibration_method not in {"minmax", "percentile", "entropy"}:
            raise ValueError("calibration_method must be minmax, percentile, or entropy")


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


def _load_data(config: OptimizationConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, list[int], str]:
    statement_map = load_statement_map(config.data_dir / "scp_statements.csv")
    records = load_metadata(config.data_dir / "ptbxl_database.csv", statement_map)
    train_records = _select(records, 1, 8, config.train_limit)
    test_records = _select(records, 10, 10, config.test_limit)
    calibration_records = train_records[:config.calibration_limit]
    loader_config = CheckpointBaselineConfig(
        data_dir=config.data_dir,
        download_workers=config.download_workers,
    )
    train_sum = 0.0
    train_sum_squares = 0.0
    train_count = 0
    chunk_size = 256
    for start in range(0, len(train_records), chunk_size):
        chunk, _ = _load_arrays(train_records[start:start + chunk_size], loader_config)
        values = chunk.astype(np.float64, copy=False)
        train_sum += float(values.sum())
        train_sum_squares += float(np.square(values).sum())
        train_count += values.size
        del chunk
        gc.collect()
    train_mean = train_sum / train_count
    train_variance = max(0.0, train_sum_squares / train_count - train_mean * train_mean)
    train_scale = np.sqrt(train_variance) if train_variance > 0 else 1.0
    calibration, _ = _load_arrays(calibration_records, loader_config)
    test, labels = _load_arrays(test_records, loader_config)
    calibration = ((calibration - train_mean) / train_scale).astype(np.float32)
    test = ((test - train_mean) / train_scale).astype(np.float32)
    calibration_ids = [record.ecg_id for record in calibration_records]
    calibration_hash = hashlib.sha256(
        json.dumps(calibration_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return calibration, test, labels, len(train_records), calibration_ids, calibration_hash


def _compile_graph(input_path: Path, output_path: Path) -> None:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    options.optimized_model_filepath = str(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    session = ort.InferenceSession(str(input_path), sess_options=options, providers=["CPUExecutionProvider"])
    del session


def _quantize(
    input_path: Path,
    dynamic_path: Path,
    static_path: Path,
    calibration: np.ndarray,
    *,
    per_channel: bool,
    calibration_method: str,
) -> None:
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
        per_channel=per_channel,
        extra_options={"DefaultTensorType": "float32"},
    )
    quantize_static(
        str(input_path),
        str(static_path),
        _CalibrationReader(calibration),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
        calibrate_method={
            "minmax": CalibrationMethod.MinMax,
            "percentile": CalibrationMethod.Percentile,
            "entropy": CalibrationMethod.Entropy,
        }[calibration_method],
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
    signals_path: Path,
    runtime_optimization: str = "disabled",
) -> dict[str, Any]:
    artifact = inspect_artifact(path)
    session_started = time.perf_counter()
    session = OnnxRuntimeSession(
        path,
        OnnxRuntimeConfig(threads=config.threads, optimization=runtime_optimization),
    )
    initialization_seconds = time.perf_counter() - session_started
    logits = _predict_onnx_in_chunks(session, test, config.evaluation_batch_size)
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    comparison = compare_outputs(reference_logits, logits, atol=1e-1, rtol=1e-1)
    primary_runs = _isolated_benchmarks(path, signals_path, config, config.threads, runtime_optimization)
    thread_sweep = []
    for thread_count in config.thread_counts:
        if thread_count == config.threads:
            runs = primary_runs
        else:
            runs = _isolated_benchmarks(path, signals_path, config, thread_count, runtime_optimization)
        thread_sweep.append(_thread_sweep_point(thread_count, runs, config))
    metrics = _metrics(probabilities, labels)
    intervals = bootstrap_macro_intervals(
        probabilities,
        labels,
        resamples=config.bootstrap_resamples,
        seed=config.bootstrap_seed,
    )
    metrics["bootstrap"] = intervals
    metrics["macro_auroc_ci95"] = intervals["macro_auroc_ci95"]
    metrics["macro_f1_ci95"] = intervals["macro_f1_ci95"]
    return {
        "name": name,
        "artifact": artifact.as_dict(),
        "metrics": metrics,
        "calibration_hash": calibration_hash,
        "output_validation": {
            "shape": comparison.shape,
            "max_absolute_error": comparison.max_absolute_error,
            "max_relative_error": comparison.max_relative_error,
            "allclose_at_1e-1": comparison.allclose,
        },
        "benchmark": _benchmark_record(
            session,
            initialization_seconds,
            primary_runs,
            config,
            runtime_optimization,
            thread_sweep,
        ),
    }


def _aggregate_batches(benchmark_runs: list[dict[str, Any]], config: OptimizationConfig) -> dict[str, Any]:
    batches = {}
    for batch_size in config.batch_sizes:
        measurements = [run["batches"][str(batch_size)] for run in benchmark_runs]
        subprocess_p50 = np.asarray(
            [item["latency_seconds_p50"] for item in measurements],
            dtype=np.float64,
        )
        subprocess_p50_mean = float(np.mean(subprocess_p50))
        subprocess_p50_std = float(np.std(subprocess_p50, ddof=1)) if len(subprocess_p50) > 1 else 0.0
        confidence_half_width = _confidence_half_width(subprocess_p50_std, len(subprocess_p50))
        batches[str(batch_size)] = {
            "batch_size": batch_size,
            "repeats": config.repeats,
            "latency_seconds_p50": float(np.median([item["latency_seconds_p50"] for item in measurements])),
            "latency_seconds_p95": float(np.median([item["latency_seconds_p95"] for item in measurements])),
            "throughput_samples_per_second": float(np.median([item["throughput_samples_per_second"] for item in measurements])),
            "subprocess_p50_mean_seconds": subprocess_p50_mean,
            "subprocess_p50_std_seconds": subprocess_p50_std,
            "subprocess_p50_ci95_seconds": [
                subprocess_p50_mean - confidence_half_width,
                subprocess_p50_mean + confidence_half_width,
            ],
            "runs": measurements,
        }
    return batches


def _thread_sweep_point(
    thread_count: int,
    benchmark_runs: list[dict[str, Any]],
    config: OptimizationConfig,
) -> dict[str, Any]:
    return {
        "threads": thread_count,
        "thread_axis": "intra_op_num_threads",
        "initialization_seconds_median": float(np.median([
            run["initialization_seconds"] for run in benchmark_runs
        ])),
        "current_rss_bytes_median": int(np.median([run["current_rss_bytes"] for run in benchmark_runs])),
        "peak_rss_bytes_median": int(np.median([run["peak_rss_bytes"] for run in benchmark_runs])),
        "peak_rss_bytes_max": max(run["peak_rss_bytes"] for run in benchmark_runs),
        "batches": _aggregate_batches(benchmark_runs, config),
    }


def _benchmark_record(
    session: OnnxRuntimeSession,
    initialization_seconds: float,
    benchmark_runs: list[dict[str, Any]],
    config: OptimizationConfig,
    runtime_optimization: str,
    thread_sweep: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "provider": session.provider,
        "threads": session.threads,
        "thread_axis": "intra_op_num_threads",
        "optimization": session.optimization,
        "ort_default_optimization": runtime_optimization == ORT_DEFAULT_OPTIMIZATION,
        "initialization_seconds": initialization_seconds,
        "initialization_seconds_isolated_median": float(np.median([
            run["initialization_seconds"] for run in benchmark_runs
        ])),
        "process_current_rss_bytes": _current_rss_bytes(),
        "measurement_mode": "fresh_subprocess_per_repeat",
        "isolated_repeats": config.isolated_repeats,
        "isolated_runs": benchmark_runs,
        "system": benchmark_runs[0]["system"],
        "isolated_memory": {
            "current_rss_bytes_median": int(np.median([run["current_rss_bytes"] for run in benchmark_runs])),
            "peak_rss_bytes_median": int(np.median([run["peak_rss_bytes"] for run in benchmark_runs])),
            "peak_rss_bytes_max": max(run["peak_rss_bytes"] for run in benchmark_runs),
        },
        "batches": _aggregate_batches(benchmark_runs, config),
        "thread_sweep": thread_sweep,
    }


_T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
}


def _confidence_half_width(sample_std: float, sample_count: int) -> float:
    if sample_count <= 1:
        return 0.0
    critical = _T_CRITICAL_95.get(sample_count - 1, 1.96)
    return critical * sample_std / np.sqrt(sample_count)


def _isolated_benchmark_command(
    artifact: Path,
    signals_path: Path,
    threads: int,
    optimization: str,
    config: OptimizationConfig,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "edge_ai_bench.isolated_benchmark",
        "--artifact", str(artifact),
        "--signals", str(signals_path),
        "--threads", str(threads),
        "--optimization", optimization,
        "--warmup", str(config.warmup),
        "--iterations", str(config.iterations),
        "--batch-sizes", *(str(size) for size in config.batch_sizes),
    ]


def _isolated_benchmarks(
    artifact: Path,
    signals_path: Path,
    config: OptimizationConfig,
    threads: int,
    optimization: str,
) -> list[dict[str, Any]]:
    command = _isolated_benchmark_command(artifact, signals_path, threads, optimization, config)
    return [
        json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
        for _ in range(config.isolated_repeats)
    ]


def _predict_onnx_in_chunks(session: OnnxRuntimeSession, signals: np.ndarray, batch_size: int) -> np.ndarray:
    outputs = []
    for start in range(0, len(signals), batch_size):
        batch = np.ascontiguousarray(signals[start:start + batch_size]).transpose(0, 2, 1)
        outputs.append(session.run(batch)[0])
        del batch
    return np.concatenate(outputs, axis=0)


def deployment_conditions(config: OptimizationConfig) -> list[dict[str, Any]]:
    """Four disabled-runtime conditions plus ORT's default optimization level.

    The default arms reload the original FP32 graph, both INT8 graphs, and the
    compiled static INT8 graph at ORT_ENABLE_ALL. Static INT8 is included so the
    compiled-versus-uncompiled gap can be remeasured at the runtime default.
    """
    disabled = "disabled"
    return [
        {
            "name": "baseline",
            "description": "Original FP32 graph with ORT graph optimization disabled",
            "runtime_optimization": disabled,
            "variants": [("fp32", config.input_path)],
        },
        {
            "name": "compiler_only",
            "description": "Serialized ORT-optimized FP32 graph with runtime optimization disabled",
            "runtime_optimization": disabled,
            "variants": [("fp32_compiled", config.optimized_path)],
        },
        {
            "name": "quantization_only",
            "description": "INT8 graphs quantized from the original FP32 graph",
            "runtime_optimization": disabled,
            "variants": [
                ("int8_dynamic", config.dynamic_path),
                ("int8_static", config.static_path),
            ],
        },
        {
            "name": "compiler_plus_quantization",
            "description": "INT8 graphs quantized from the original graph, then serialized with ORT_ENABLE_EXTENDED",
            "runtime_optimization": disabled,
            "variants": [
                ("int8_dynamic_compiled", config.optimized_dynamic_path),
                ("int8_static_compiled", config.optimized_static_path),
            ],
        },
        {
            "name": "default_runtime_optimization",
            "description": "FP32 and INT8 graphs loaded at ORT's default graph optimization level (ORT_ENABLE_ALL)",
            "runtime_optimization": ORT_DEFAULT_OPTIMIZATION,
            "variants": [
                ("fp32_default", config.input_path),
                ("int8_dynamic_default", config.dynamic_path),
                ("int8_static_default", config.static_path),
                ("int8_static_compiled_default", config.optimized_static_path),
            ],
        },
    ]


def _profile_static_interaction(
    config: OptimizationConfig,
    test: np.ndarray,
) -> dict[str, Any]:
    sample = np.ascontiguousarray(test[:1]).transpose(0, 2, 1)
    profiles = {
        "int8_static_disabled": profile_model(
            config.static_path,
            sample,
            threads=config.threads,
            optimization="disabled",
            iterations=config.profile_iterations,
        ),
        "int8_static_compiled_disabled": profile_model(
            config.optimized_static_path,
            sample,
            threads=config.threads,
            optimization="disabled",
            iterations=config.profile_iterations,
        ),
        "int8_static_default": profile_model(
            config.static_path,
            sample,
            threads=config.threads,
            optimization=ORT_DEFAULT_OPTIMIZATION,
            iterations=config.profile_iterations,
        ),
    }
    return {
        "profiles": profiles,
        "mechanism": demonstrate_static_int8_mechanism(profiles),
    }


def _environment() -> dict[str, Any]:
    system = _system_metadata()
    cpu = system.get("lscpu") or {}
    return {
        "python": platform.python_version(),
        "system": platform.platform(),
        "cpu_count": os.cpu_count(),
        "cpu_model": cpu.get("Model name"),
        "onnxruntime_version": system.get("onnxruntime_version"),
        "runtime": system,
    }


def run_optimizations(config: OptimizationConfig) -> dict[str, Any]:
    """Evaluate compiler, quantization, and ORT-default optimization conditions."""
    calibration, test, labels, selected_train_records, calibration_ids, calibration_hash = _load_data(config)
    reference_session = OnnxRuntimeSession(
        config.input_path,
        OnnxRuntimeConfig(threads=config.threads, optimization="disabled"),
    )
    reference_logits = _predict_onnx_in_chunks(
        reference_session,
        test,
        config.evaluation_batch_size,
    )
    _compile_graph(config.input_path, config.optimized_path)
    _quantize(
        config.input_path,
        config.dynamic_path,
        config.static_path,
        calibration,
        per_channel=config.quantization_per_channel,
        calibration_method=config.calibration_method,
    )
    _compile_graph(config.dynamic_path, config.optimized_dynamic_path)
    _compile_graph(config.static_path, config.optimized_static_path)
    signals_directory = tempfile.TemporaryDirectory(prefix="edge-ai-benchmark-")
    signals_path = Path(signals_directory.name) / "test-signals.npy"
    np.save(signals_path, test)
    experiments = deployment_conditions(config)
    evaluated_experiments = []
    for experiment in experiments:
        runtime_optimization = experiment["runtime_optimization"]
        evaluated_experiments.append({
            "name": experiment["name"],
            "description": experiment["description"],
            "runtime_optimization": runtime_optimization,
            "variants": [
                _evaluate_artifact(
                    variant_name,
                    variant_path,
                    reference_logits,
                    test,
                    labels,
                    config,
                    calibration_hash,
                    signals_path,
                    runtime_optimization,
                )
                for variant_name, variant_path in experiment["variants"]
            ],
        })
    variants = [variant for experiment in evaluated_experiments for variant in experiment["variants"]]
    node_profiles = _profile_static_interaction(config, test) if config.profile_nodes else None
    result = {
        "schema_version": 2,
        "experiment": {
            "name": "ecg-hybrid-net-exp12-optimization",
            "dataset": "PTB-XL",
            "source": PTBXL_URL,
            "train_samples": config.train_limit,
            "test_samples": len(test),
            "selected_train_records": selected_train_records,
            "selected_test_records": len(test),
            "calibration_samples": len(calibration),
            "calibration_split": "train folds 1-8, first deterministic records",
            "calibration_record_ids": calibration_ids,
            "calibration_record_ids_sha256": calibration_hash,
            "threads": config.threads,
            "thread_counts": list(config.thread_counts),
            "thread_axis": "intra_op_num_threads",
            "batch_sizes": list(config.batch_sizes),
            "runtime_optimization": "disabled",
            "ort_default_optimization": "ORT_ENABLE_ALL",
            "compiler_optimization": "ORT_ENABLE_EXTENDED",
            "combined_order": "quantize original graph, then serialize optimized INT8 graph",
            "evaluation_batch_size": config.evaluation_batch_size,
            "isolated_repeats": config.isolated_repeats,
            "quantization_per_channel": config.quantization_per_channel,
            "calibration_method": config.calibration_method,
            "bootstrap_resamples": config.bootstrap_resamples,
            "bootstrap_seed": config.bootstrap_seed,
            "conditions": [experiment["name"] for experiment in experiments],
        },
        "environment": _environment(),
        "experiments": evaluated_experiments,
        "variants": variants,
        "pareto_frontier": _pareto_frontier(variants),
        "pareto_frontiers": _pareto_frontiers(variants),
        "static_int8_interaction": static_int8_interaction(variants),
        "node_profiles": node_profiles,
        "status": "compiler_quantization_experiment",
    }
    if config.reference_manifest is not None:
        reference = json.loads(config.reference_manifest.read_text(encoding="utf-8"))
        result["calibration_comparison"] = compare_calibration_manifests(reference, result)
    config.result_path.parent.mkdir(parents=True, exist_ok=True)
    config.result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    signals_directory.cleanup()
    return result


def _pareto_frontier(variants: list[dict[str, Any]]) -> list[str]:
    """Return non-dominated variants for size, batch-1 latency, and macro F1."""
    return _pareto_frontier_for(
        variants,
        lambda variant: (
            variant["artifact"]["size_bytes"],
            variant["benchmark"]["batches"]["1"]["latency_seconds_p50"],
            -variant["metrics"]["macro_f1"],
        ),
    )


def _pareto_frontiers(variants: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Return frontiers for distinct deployment priorities."""
    batch_one = lambda variant: variant["benchmark"]["batches"]["1"]
    return {
        "size_latency_f1": _pareto_frontier(variants),
        "size_throughput_f1": _pareto_frontier_for(
            variants,
            lambda variant: (
                variant["artifact"]["size_bytes"],
                -batch_one(variant)["throughput_samples_per_second"],
                -variant["metrics"]["macro_f1"],
            ),
        ),
        "size_p95_latency_f1": _pareto_frontier_for(
            variants,
            lambda variant: (
                variant["artifact"]["size_bytes"],
                batch_one(variant)["latency_seconds_p95"],
                -variant["metrics"]["macro_f1"],
            ),
        ),
        "size_latency_auroc": _pareto_frontier_for(
            variants,
            lambda variant: (
                variant["artifact"]["size_bytes"],
                batch_one(variant)["latency_seconds_p50"],
                -variant["metrics"]["macro_auroc"],
            ),
        ),
        "size_latency_f1_memory": _pareto_frontier_for(
            variants,
            lambda variant: (
                variant["artifact"]["size_bytes"],
                batch_one(variant)["latency_seconds_p50"],
                -variant["metrics"]["macro_f1"],
                variant["benchmark"]["isolated_memory"]["current_rss_bytes_median"],
            ),
        ),
        "size_vs_batch1_latency": _pareto_frontier_for(
            variants,
            lambda variant: (
                variant["artifact"]["size_bytes"],
                batch_one(variant)["latency_seconds_p50"],
            ),
        ),
        "batch1_latency_vs_f1": _pareto_frontier_for(
            variants,
            lambda variant: (
                batch_one(variant)["latency_seconds_p50"],
                -variant["metrics"]["macro_f1"],
            ),
        ),
        "init_rss_latency": _pareto_frontier_for(
            variants,
            lambda variant: (
                variant["benchmark"]["initialization_seconds_isolated_median"],
                variant["benchmark"]["isolated_memory"]["current_rss_bytes_median"],
                batch_one(variant)["latency_seconds_p50"],
            ),
        ),
    }


def _pareto_frontier_for(
    variants: list[dict[str, Any]],
    point_factory: Any,
) -> list[str]:
    return nondominated_names({variant["name"]: point_factory(variant) for variant in variants})

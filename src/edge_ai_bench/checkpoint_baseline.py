"""Evaluate and export the downloaded PTB-XL ResNet-Transformer checkpoint."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import platform
import resource
import time
from typing import Any

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from .artifacts import compare_outputs, inspect_artifact
from .checkpoint_model import load_checkpoint, predict_logits
from .ptbxl import PTBXLRecord, load_metadata, load_statement_map, load_waveform, multilabel_vector
from .real_baseline import PTBXL_URL, _ensure_waveform


CHECKPOINT_CLASSES = ("CD", "HYP", "MI", "NORM", "STTC")


@dataclass(frozen=True)
class CheckpointBaselineConfig:
    checkpoint_path: Path = Path("artifacts/ecg_hybrid_net_exp12_resnet_medium.pth")
    onnx_path: Path = Path("artifacts/ecg_hybrid_net_exp12_resnet_medium-fp32.onnx")
    result_path: Path = Path("results/ecg-hybrid-net-baseline.json")
    data_dir: Path = Path("data/ptb-xl")
    train_limit: int = 1000
    validation_limit: int = 200
    test_limit: int = 200
    batch_size: int = 32
    warmup: int = 5
    iterations: int = 30
    download_workers: int = 16
    onnx_threads: int = 1
    onnx_optimization: str = "all"
    benchmark_batch_sizes: tuple[int, ...] = (1, 4, 8, 16)

    def __post_init__(self) -> None:
        if self.onnx_threads <= 0:
            raise ValueError("onnx_threads must be positive")
        if self.onnx_optimization not in {"disabled", "basic", "extended", "all"}:
            raise ValueError("onnx_optimization must be disabled, basic, extended, or all")
        if not self.benchmark_batch_sizes or any(size <= 0 for size in self.benchmark_batch_sizes):
            raise ValueError("benchmark_batch_sizes must contain positive batch sizes")


@dataclass(frozen=True)
class CheckpointSweepConfig:
    artifact_path: Path = Path("artifacts/ecg_hybrid_net_exp12_resnet_medium-fp32.onnx")
    result_path: Path = Path("results/ecg-hybrid-net-fp32-sweep.json")
    data_dir: Path = Path("data/ptb-xl")
    train_limit: int = 1000
    test_limit: int = 200
    warmup: int = 3
    iterations: int = 10
    download_workers: int = 16
    thread_counts: tuple[int, ...] = (1, 2, 4, 8)
    optimizations: tuple[str, ...] = ("disabled", "basic", "extended", "all")
    batch_sizes: tuple[int, ...] = (1, 4, 8, 16)

    def __post_init__(self) -> None:
        if min(self.train_limit, self.test_limit, self.iterations) <= 0:
            raise ValueError("train_limit, test_limit, and iterations must be positive")
        if self.warmup < 0 or not self.thread_counts or any(value <= 0 for value in self.thread_counts):
            raise ValueError("warmup and thread_counts must be valid")
        allowed = {"disabled", "basic", "extended", "all"}
        if not self.optimizations or any(value not in allowed for value in self.optimizations):
            raise ValueError("optimizations must use supported graph optimization levels")
        if not self.batch_sizes or any(value <= 0 for value in self.batch_sizes):
            raise ValueError("batch_sizes must contain positive values")


def _select(records: list[PTBXLRecord], first_fold: int, last_fold: int, limit: int) -> list[PTBXLRecord]:
    return [record for record in records if first_fold <= record.fold <= last_fold and record.diagnostic_superclasses][:limit]


def _load_arrays(records: list[PTBXLRecord], config: CheckpointBaselineConfig) -> tuple[np.ndarray, np.ndarray]:
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=config.download_workers) as executor:
        list(executor.map(lambda record: _ensure_waveform(record, config.data_dir), records))
    signals = []
    labels = []
    for record in records:
        signals.append(load_waveform(record, config.data_dir, resolution="lr"))
        labels.append(multilabel_vector(record.diagnostic_superclasses, CHECKPOINT_CLASSES))
    return np.asarray(signals, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def _standardize(train: np.ndarray, validation: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    scaler.fit(train.reshape(-1, 1))
    return tuple(scaler.transform(values.reshape(-1, 1)).reshape(values.shape).astype(np.float32) for values in (train, validation, test)) + (scaler,)


def _predict(model: Any, signals: np.ndarray, batch_size: int) -> np.ndarray:
    outputs = []
    for start in range(0, len(signals), batch_size):
        batch = signals[start:start + batch_size]
        outputs.append(predict_logits(model, _torch_tensor(batch)).detach().cpu().numpy())
    return 1.0 / (1.0 + np.exp(-np.concatenate(outputs)))


def _torch_tensor(values: np.ndarray) -> Any:
    import torch

    return torch.from_numpy(values).transpose(1, 2)


def _metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    auc = roc_auc_score(labels, probabilities, average="macro")
    predictions = (probabilities >= 0.5).astype(np.int64)
    class_auc = roc_auc_score(labels, probabilities, average=None)
    class_f1 = f1_score(labels, predictions, average=None, zero_division=0)
    return {
        "macro_auroc": float(auc),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "per_class": {
            class_name: {
                "auroc": float(class_auc[index]),
                "f1": float(class_f1[index]),
                "positive_count": int(labels[:, index].sum()),
            }
            for index, class_name in enumerate(CHECKPOINT_CLASSES)
        },
        "samples": int(len(labels)),
        "positive_counts": labels.sum(axis=0).tolist(),
    }


def _benchmark(model: Any, signal: np.ndarray, warmup: int, iterations: int) -> dict[str, float]:
    import torch

    tensor = _torch_tensor(signal[:1])
    for _ in range(warmup):
        predict_logits(model, tensor)
    durations = []
    for _ in range(iterations):
        started = time.perf_counter()
        predict_logits(model, tensor)
        durations.append(time.perf_counter() - started)
    ordered = sorted(durations)
    return {
        "batch_size": 1,
        "latency_seconds_p50": ordered[len(ordered) // 2],
        "latency_seconds_p95": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        "throughput_samples_per_second": float(1.0 / np.mean(durations)),
        "torch_threads": torch.get_num_threads(),
    }


def _peak_rss_bytes() -> int:
    """Return the process peak resident set size using the Linux/macOS unit contract."""
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak_rss * (1024 if platform.system() != "Darwin" else 1))


def _current_rss_bytes() -> int:
    """Return current process RSS where the host exposes Linux proc status."""
    status_path = Path("/proc/self/status")
    if status_path.is_file():
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    return _peak_rss_bytes()


def _benchmark_onnx(
    session: Any,
    signals: np.ndarray,
    warmup: int,
    iterations: int,
    batch_sizes: tuple[int, ...],
) -> dict[str, Any]:
    """Benchmark ONNX Runtime with fixed batches and return deployment metrics."""
    batch_sizes = tuple(size for size in batch_sizes if size <= len(signals))
    if not batch_sizes:
        raise ValueError("at least one benchmark batch size must fit the evaluation set")
    results: dict[str, Any] = {}
    for batch_size in batch_sizes:
        batch = np.ascontiguousarray(signals[:batch_size]).transpose(0, 2, 1)
        for _ in range(warmup):
            session.run(batch)
        durations = []
        for _ in range(iterations):
            started = time.perf_counter()
            session.run(batch)
            durations.append(time.perf_counter() - started)
        ordered = sorted(durations)
        results[str(batch_size)] = {
            "batch_size": batch_size,
            "latency_seconds_p50": ordered[len(ordered) // 2],
            "latency_seconds_p95": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
            "throughput_samples_per_second": float(batch_size / np.mean(durations)),
        }
    return results


def run_checkpoint_sweep(config: CheckpointSweepConfig) -> dict[str, Any]:
    """Measure fixed FP32 ONNX Runtime configurations on the same test signals."""
    statement_map = load_statement_map(config.data_dir / "scp_statements.csv")
    records = load_metadata(config.data_dir / "ptbxl_database.csv", statement_map)
    train_records = _select(records, 1, 8, config.train_limit)
    test_records = _select(records, 10, 10, config.test_limit)
    train, _ = _load_arrays(train_records, CheckpointBaselineConfig(
        data_dir=config.data_dir,
        download_workers=config.download_workers,
    ))
    test, _ = _load_arrays(test_records, CheckpointBaselineConfig(
        data_dir=config.data_dir,
        download_workers=config.download_workers,
    ))
    _, _, test, _ = _standardize(train, train, test)
    artifact = inspect_artifact(config.artifact_path)
    sweep_results = []
    for optimization in config.optimizations:
        for threads in config.thread_counts:
            from .onnx_runtime import OnnxRuntimeConfig, OnnxRuntimeSession

            started = time.perf_counter()
            session = OnnxRuntimeSession(
                config.artifact_path,
                OnnxRuntimeConfig(threads=threads, optimization=optimization),
            )
            initialization_seconds = time.perf_counter() - started
            sample_output = session.run(
                np.ascontiguousarray(test[:1]).transpose(0, 2, 1)
            )[0]
            sweep_results.append({
                "threads": threads,
                "optimization": optimization,
                "initialization_seconds": initialization_seconds,
                "batches": _benchmark_onnx(
                    session, test, config.warmup, config.iterations, config.batch_sizes
                ),
                "sample_output_shape": list(sample_output.shape),
                "sample_output_finite": bool(np.isfinite(sample_output).all()),
                "process_current_rss_bytes": _current_rss_bytes(),
                "process_peak_rss_bytes": _peak_rss_bytes(),
            })
    result = {
        "schema_version": 1,
        "experiment": {
            "name": "ecg-hybrid-net-exp12-fp32-onnxruntime-sweep",
            "artifact": artifact.as_dict(),
            "dataset": "PTB-XL",
            "train_samples": len(train_records),
            "test_samples": len(test_records),
            "warmup": config.warmup,
            "iterations": config.iterations,
            "batch_sizes": list(config.batch_sizes),
        },
        "environment": {
            "python": platform.python_version(),
            "system": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
        "results": sweep_results,
        "status": "fp32_runtime_sweep",
    }
    config.result_path.parent.mkdir(parents=True, exist_ok=True)
    config.result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def run_checkpoint_baseline(config: CheckpointBaselineConfig) -> dict[str, Any]:
    """Evaluate the downloaded checkpoint and export its FP32 ONNX artifact."""
    statement_path = config.data_dir / "scp_statements.csv"
    database_path = config.data_dir / "ptbxl_database.csv"
    statement_map = load_statement_map(statement_path)
    records = load_metadata(database_path, statement_map)
    train_records = _select(records, 1, 8, config.train_limit)
    validation_records = _select(records, 9, 9, config.validation_limit)
    test_records = _select(records, 10, 10, config.test_limit)
    train, train_labels = _load_arrays(train_records, config)
    validation, validation_labels = _load_arrays(validation_records, config)
    test, test_labels = _load_arrays(test_records, config)
    train, validation, test, scaler = _standardize(train, validation, test)

    model = load_checkpoint(config.checkpoint_path)
    train_probabilities = _predict(model, train, config.batch_size)
    validation_probabilities = _predict(model, validation, config.batch_size)
    test_probabilities = _predict(model, test, config.batch_size)
    import torch
    sample_tensor = _torch_tensor(test[:1])
    with torch.no_grad():
        sample_logits = model(sample_tensor)
    config.onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        sample_tensor,
        config.onnx_path,
        input_names=["ecg"],
        output_names=["logits"],
        dynamic_axes={"ecg": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    artifact = inspect_artifact(config.onnx_path)
    from .onnx_runtime import OnnxRuntimeConfig, OnnxRuntimeSession

    session_init_rss_before = _current_rss_bytes()
    session_started = time.perf_counter()
    session = OnnxRuntimeSession(
        config.onnx_path,
        OnnxRuntimeConfig(
            threads=config.onnx_threads,
            optimization=config.onnx_optimization,
        ),
    )
    initialization_seconds = time.perf_counter() - session_started
    session_init_rss_after = _current_rss_bytes()
    onnx_logits = session.run(sample_tensor.numpy())[0]
    validation_output = compare_outputs(sample_logits.cpu().numpy(), onnx_logits, atol=1e-4, rtol=1e-4)
    onnx_benchmark = _benchmark_onnx(
        session,
        test,
        config.warmup,
        config.iterations,
        config.benchmark_batch_sizes,
    )
    result = {
        "schema_version": 3,
        "experiment": {"name": "ecg-hybrid-net-exp12", "checkpoint": str(config.checkpoint_path), "preprocessing": "global-training-scaler", "resolution": "lr"},
        "dataset": {"name": "PTB-XL", "source": PTBXL_URL, "classes": CHECKPOINT_CLASSES, "folds": {"train": "1-8", "validation": "9", "test": "10"}},
        "splits": {"train": len(train_records), "validation": len(validation_records), "test": len(test_records)},
        "metrics": {"train": _metrics(train_probabilities, train_labels), "validation": _metrics(validation_probabilities, validation_labels), "test": _metrics(test_probabilities, test_labels)},
        "artifact": artifact.as_dict(),
        "validation": {"shape": validation_output.shape, "max_absolute_error": validation_output.max_absolute_error, "max_relative_error": validation_output.max_relative_error, "allclose": validation_output.allclose},
        "benchmark": {
            "pytorch_reference": _benchmark(model, test, config.warmup, config.iterations),
            "onnxruntime": {
                "provider": session.provider,
                "threads": session.threads,
                "optimization": session.optimization,
                "batch_sizes": list(config.benchmark_batch_sizes),
                "initialization_seconds": initialization_seconds,
                "session_init_rss_delta_bytes": max(0, session_init_rss_after - session_init_rss_before),
                "process_current_rss_bytes": _current_rss_bytes(),
                "process_peak_rss_bytes": _peak_rss_bytes(),
                "process_peak_rss_bytes": _peak_rss_bytes(),
                "batches": onnx_benchmark,
            },
        },
        "environment": {
            "python": platform.python_version(),
            "system": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "onnxruntime": session.provider,
        },
        "status": "real_checkpoint_baseline",
    }
    config.result_path.parent.mkdir(parents=True, exist_ok=True)
    config.result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result

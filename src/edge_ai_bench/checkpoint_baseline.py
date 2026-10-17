"""Evaluate and export the downloaded PTB-XL ResNet-Transformer checkpoint."""

from dataclasses import dataclass
import json
from pathlib import Path
import platform
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
    return {
        "macro_auroc": float(auc),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
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
        "torch_threads": torch.get_num_threads(),
    }


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
    torch.onnx.export(model, sample_tensor, config.onnx_path, input_names=["ecg"], output_names=["logits"], opset_version=17, dynamo=False)
    artifact = inspect_artifact(config.onnx_path)
    import onnxruntime as ort
    session = ort.InferenceSession(str(config.onnx_path), providers=["CPUExecutionProvider"])
    onnx_logits = session.run(None, {"ecg": sample_tensor.numpy()})[0]
    validation_output = compare_outputs(sample_logits.cpu().numpy(), onnx_logits, atol=1e-4, rtol=1e-4)
    result = {
        "schema_version": 3,
        "experiment": {"name": "ecg-hybrid-net-exp12", "checkpoint": str(config.checkpoint_path), "preprocessing": "global-training-scaler", "resolution": "lr"},
        "dataset": {"name": "PTB-XL", "source": PTBXL_URL, "classes": CHECKPOINT_CLASSES, "folds": {"train": "1-8", "validation": "9", "test": "10"}},
        "splits": {"train": len(train_records), "validation": len(validation_records), "test": len(test_records)},
        "metrics": {"train": _metrics(train_probabilities, train_labels), "validation": _metrics(validation_probabilities, validation_labels), "test": _metrics(test_probabilities, test_labels)},
        "artifact": artifact.as_dict(),
        "validation": {"shape": validation_output.shape, "max_absolute_error": validation_output.max_absolute_error, "max_relative_error": validation_output.max_relative_error, "allclose": validation_output.allclose},
        "benchmark": _benchmark(model, test, config.warmup, config.iterations),
        "environment": {"python": platform.python_version(), "system": platform.platform(), "processor": platform.processor()},
        "status": "real_checkpoint_baseline",
    }
    config.result_path.parent.mkdir(parents=True, exist_ok=True)
    config.result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result

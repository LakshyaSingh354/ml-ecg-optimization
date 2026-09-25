"""Small real-data PTB-XL baseline for establishing the experiment contract."""

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import platform
import shutil
import time
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from .artifacts import compare_outputs, inspect_artifact
from .ptbxl import PTBXLRecord, load_metadata, load_statement_map, load_waveform, multilabel_vector, normalize_signal


PTBXL_URL = "https://physionet.org/files/ptb-xl/1.0.3"
PTBXL_WFDB_DIRECTORY = "ptb-xl/1.0.3"
DEFAULT_CLASSES = ("NORM", "MI", "STTC", "CD", "HYP")


@dataclass(frozen=True)
class RealBaselineConfig:
    data_dir: Path = Path("data/ptb-xl")
    checkpoint_path: Path = Path("artifacts/ptbxl-logistic.joblib")
    artifact_path: Path = Path("artifacts/ptbxl-logistic-fp32.onnx")
    result_path: Path = Path("results/ptbxl-baseline.json")
    train_limit: int = 1000
    validation_limit: int = 200
    test_limit: int = 200
    feature_points: int = 32
    seed: int = 7
    warmup: int = 5
    iterations: int = 30
    download_workers: int = 16

    def __post_init__(self) -> None:
        if min(self.train_limit, self.validation_limit, self.test_limit) <= 0:
            raise ValueError("split limits must be positive")
        if self.download_workers <= 0:
            raise ValueError("download_workers must be positive")


def _download(url: str, destination: Path, *, expected_size: int | None = None) -> None:
    from urllib.request import urlopen

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and (expected_size is None or destination.stat().st_size == expected_size):
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urlopen(url, timeout=30) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
    temporary.replace(destination)
    if expected_size is not None and destination.stat().st_size != expected_size:
        raise IOError(f"unexpected download size for {destination}: {destination.stat().st_size} bytes")


def _ensure_waveform(record: PTBXLRecord, root: Path) -> None:
    for relative_path in (record.filename_lr, f"{record.filename_lr}.dat", f"{record.filename_lr}.hea"):
        destination = root / relative_path
        if destination.suffix not in {".dat", ".hea"}:
            continue
        _download(f"{PTBXL_URL}/{relative_path}", destination, expected_size=24000 if destination.suffix == ".dat" else None)


def prepare_metadata(config: RealBaselineConfig) -> dict[str, dict[str, Any]]:
    """Download only the two PTB-XL metadata files needed for indexing."""
    database_path = config.data_dir / "ptbxl_database.csv"
    statement_path = config.data_dir / "scp_statements.csv"
    _download(f"{PTBXL_URL}/ptbxl_database.csv", database_path)
    _download(f"{PTBXL_URL}/scp_statements.csv", statement_path)
    statement_map = load_statement_map(statement_path)
    return statement_map, load_metadata(database_path, statement_map)


def _select_records(records: list[PTBXLRecord], fold_start: int, fold_end: int, limit: int) -> list[PTBXLRecord]:
    selected = [record for record in records if fold_start <= record.fold <= fold_end and record.diagnostic_superclasses]
    return selected[:limit]


def _features(signal: Any, points: int) -> np.ndarray:
    values = normalize_signal(signal, axis=0)
    values = np.asarray(values, dtype=np.float32)
    indices = np.linspace(0, values.shape[0] - 1, points).astype(int)
    sampled = values[indices].T
    summary = np.concatenate([sampled.reshape(-1), values.mean(axis=0), values.std(axis=0)])
    return summary.astype(np.float32)


def _load_split(records: list[PTBXLRecord], root: Path, points: int, classes: tuple[str, ...], download_workers: int) -> tuple[np.ndarray, np.ndarray]:
    with ThreadPoolExecutor(max_workers=download_workers) as executor:
        list(executor.map(lambda record: _ensure_waveform(record, root), records))
    features: list[np.ndarray] = []
    labels: list[list[int]] = []
    for record in records:
        signal = load_waveform(record, root, resolution="lr")
        features.append(_features(signal, points))
        labels.append(multilabel_vector(record.diagnostic_superclasses, classes))
    return np.stack(features), np.asarray(labels, dtype=np.int64)


def _metrics(model: Any, features: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    probabilities = model.predict_proba(features)
    if isinstance(probabilities, list):
        probabilities = np.column_stack([value[:, 1] for value in probabilities])
    predictions = (probabilities >= 0.5).astype(np.int64)
    auc_values = []
    for index in range(labels.shape[1]):
        if len(np.unique(labels[:, index])) == 2:
            auc_values.append(roc_auc_score(labels[:, index], probabilities[:, index]))
    return {
        "macro_auroc": float(np.mean(auc_values)) if auc_values else None,
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "samples": int(labels.shape[0]),
        "positive_counts": labels.sum(axis=0).tolist(),
    }


def _onnx_probabilities(outputs: list[Any], class_count: int) -> np.ndarray:
    """Normalize skl2onnx classifier probability outputs to ``(samples, classes)``."""
    probability_output = outputs[-1]
    if isinstance(probability_output, list) and probability_output and isinstance(probability_output[0], dict):
        return np.asarray(
            [[float(probability_output[row].get(column, 0.0)) for column in range(class_count)] for row in range(len(probability_output))],
            dtype=np.float32,
        )
    values = np.asarray(probability_output, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != class_count:
        raise ValueError(f"unexpected ONNX probability output shape: {values.shape}")
    return values


def _benchmark(session: Any, features: np.ndarray, warmup: int, iterations: int) -> dict[str, float]:
    for _ in range(warmup):
        session.run(None, {session.get_inputs()[0].name: features})
    durations = []
    for _ in range(iterations):
        started = time.perf_counter()
        session.run(None, {session.get_inputs()[0].name: features})
        durations.append(time.perf_counter() - started)
    ordered = sorted(durations)
    return {
        "latency_seconds_p50": ordered[len(ordered) // 2],
        "latency_seconds_p95": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        "throughput_samples_per_second": float(features.shape[0] / np.mean(durations)),
    }


def run_baseline(config: RealBaselineConfig) -> dict[str, Any]:
    """Train, export, validate, and benchmark a bounded real PTB-XL baseline."""
    statement_map, records = prepare_metadata(config)
    train_records = _select_records(records, 1, 8, config.train_limit)
    validation_records = _select_records(records, 9, 9, config.validation_limit)
    test_records = _select_records(records, 10, 10, config.test_limit)
    if not train_records or not validation_records or not test_records:
        raise RuntimeError("PTB-XL split selection returned an empty split")

    train_features, train_labels = _load_split(train_records, config.data_dir, config.feature_points, DEFAULT_CLASSES, config.download_workers)
    validation_features, validation_labels = _load_split(validation_records, config.data_dir, config.feature_points, DEFAULT_CLASSES, config.download_workers)
    test_features, test_labels = _load_split(test_records, config.data_dir, config.feature_points, DEFAULT_CLASSES, config.download_workers)
    model = make_pipeline(
        StandardScaler(),
        OneVsRestClassifier(LogisticRegression(max_iter=300, random_state=config.seed)),
    )
    model.fit(train_features, train_labels)

    import joblib

    config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.checkpoint_path)
    checkpoint = inspect_artifact(config.checkpoint_path)

    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    import onnxruntime as ort
    import onnx

    onnx_model = convert_sklearn(model, initial_types=[("features", FloatTensorType([None, train_features.shape[1]]))], target_opset=13)
    config.artifact_path.parent.mkdir(parents=True, exist_ok=True)
    config.artifact_path.write_bytes(onnx_model.SerializeToString())
    artifact = inspect_artifact(config.artifact_path)
    onnx.checker.check_model(onnx_model)
    session = ort.InferenceSession(str(config.artifact_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    onnx_outputs = session.run(None, {input_name: test_features.astype(np.float32)})
    model_outputs = model.predict_proba(test_features)
    if isinstance(model_outputs, list):
        model_outputs = np.column_stack([value[:, 1] for value in model_outputs])
    validation = compare_outputs(model_outputs, _onnx_probabilities(onnx_outputs, len(DEFAULT_CLASSES)), atol=1e-4, rtol=1e-4)
    result = {
        "schema_version": 2,
        "experiment": {"name": "ptbxl-real-baseline", "seed": config.seed, "resolution": "lr", "feature_points": config.feature_points},
        "dataset": {"name": "PTB-XL", "source": PTBXL_URL, "classes": DEFAULT_CLASSES, "folds": {"train": "1-8", "validation": "9", "test": "10"}},
        "splits": {"train": len(train_records), "validation": len(validation_records), "test": len(test_records)},
        "metrics": {"validation": _metrics(model, validation_features, validation_labels), "test": _metrics(model, test_features, test_labels)},
        "artifact": artifact.as_dict(),
        "checkpoint": checkpoint.as_dict(),
        "validation": {"shape": validation.shape, "max_absolute_error": validation.max_absolute_error, "max_relative_error": validation.max_relative_error, "allclose": validation.allclose},
        "benchmark": _benchmark(session, test_features.astype(np.float32), config.warmup, config.iterations),
        "environment": {"python": platform.python_version(), "system": platform.platform(), "processor": platform.processor()},
        "status": "real_baseline",
    }
    config.result_path.parent.mkdir(parents=True, exist_ok=True)
    config.result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result

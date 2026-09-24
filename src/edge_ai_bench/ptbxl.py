"""PTB-XL metadata parsing and signal preprocessing primitives."""

from ast import literal_eval
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PTBXLRecord:
    """Minimal record needed to build a patient-wise PTB-XL example."""

    ecg_id: int
    patient_id: int
    fold: int
    diagnostic_superclasses: tuple[str, ...]
    filename_lr: str
    filename_hr: str


def _diagnostic_superclasses(scp_codes: str, statement_map: dict[str, Any]) -> tuple[str, ...]:
    codes = literal_eval(scp_codes)
    labels = {
        statement_map[code]["diagnostic_class"]
        for code, score in codes.items()
        if float(score) >= 50.0 and code in statement_map and statement_map[code].get("diagnostic", False)
    }
    return tuple(sorted(labels))


def load_metadata(path: Path, statement_map: dict[str, Any]) -> list[PTBXLRecord]:
    """Load PTB-XL metadata without opening waveform files or patient data."""
    records: list[PTBXLRecord] = []
    with path.open(newline="", encoding="utf-8") as metadata_file:
        for row in csv.DictReader(metadata_file):
            records.append(
                PTBXLRecord(
                    ecg_id=int(row["ecg_id"]),
                    patient_id=int(row["patient_id"]),
                    fold=int(row["strat_fold"]),
                    diagnostic_superclasses=_diagnostic_superclasses(row["scp_codes"], statement_map),
                    filename_lr=row["filename_lr"],
                    filename_hr=row["filename_hr"],
                )
            )
    return records


def normalize_signal(signal: Any, *, axis: int = -1) -> Any:
    """Per-lead z-normalize a signal array without changing its shape."""
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("Install the benchmark extra to preprocess ECG arrays") from error

    values = np.asarray(signal, dtype=np.float32)
    mean = np.mean(values, axis=axis, keepdims=True)
    standard_deviation = np.std(values, axis=axis, keepdims=True)
    return (values - mean) / np.maximum(standard_deviation, np.finfo(np.float32).eps)


def multilabel_vector(labels: tuple[str, ...], classes: tuple[str, ...]) -> list[int]:
    """Encode diagnostic superclass labels in a stable class order."""
    label_set = set(labels)
    return [int(name in label_set) for name in classes]

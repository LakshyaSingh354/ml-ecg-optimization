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


def _integer_field(value: str) -> int:
    parsed = float(value)
    if not parsed.is_integer():
        raise ValueError(f"expected an integer-valued PTB-XL field, got {value!r}")
    return int(parsed)


def _diagnostic_superclasses(scp_codes: str, statement_map: dict[str, Any]) -> tuple[str, ...]:
    codes = literal_eval(scp_codes)
    labels = {
        statement_map[code]["diagnostic_class"]
        for code, score in codes.items()
        if float(score) >= 50.0
        and code in statement_map
        and statement_map[code].get("diagnostic", False)
        and statement_map[code].get("diagnostic_class")
    }
    return tuple(sorted(labels))


def load_metadata(path: Path, statement_map: dict[str, Any]) -> list[PTBXLRecord]:
    """Load PTB-XL metadata without opening waveform files or patient data."""
    records: list[PTBXLRecord] = []
    with path.open(newline="", encoding="utf-8") as metadata_file:
        for row in csv.DictReader(metadata_file):
            required_fields = ("ecg_id", "patient_id", "strat_fold", "scp_codes", "filename_lr", "filename_hr")
            if any(not row.get(field) for field in required_fields):
                continue
            records.append(
                PTBXLRecord(
                    ecg_id=_integer_field(row["ecg_id"]),
                    patient_id=_integer_field(row["patient_id"]),
                    fold=_integer_field(row["strat_fold"]),
                    diagnostic_superclasses=_diagnostic_superclasses(row["scp_codes"], statement_map),
                    filename_lr=row["filename_lr"],
                    filename_hr=row["filename_hr"],
                )
            )
    return records


def load_statement_map(path: Path) -> dict[str, dict[str, Any]]:
    """Load the official PTB-XL SCP statement map from CSV."""
    statements: dict[str, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as statement_file:
        for row in csv.DictReader(statement_file):
            code = row.get("scp_code") or row.get("") or row.get(None)
            if isinstance(code, list):
                code = code[0]
            if not code:
                continue
            statements[code] = {
                "diagnostic": row.get("diagnostic", "") in {"1", "1.0"},
                "diagnostic_class": row.get("diagnostic_class") or None,
                "form": row.get("form", "") in {"1", "1.0"},
                "rhythm": row.get("rhythm", "") in {"1", "1.0"},
            }
    return statements


def load_waveform(record: PTBXLRecord, root: Path, *, resolution: str = "hr", remote_directory: str | None = None) -> Any:
    """Load one PTB-XL waveform through WFDB, without copying it into the repo."""
    if resolution not in {"lr", "hr"}:
        raise ValueError("resolution must be 'lr' or 'hr'")
    try:
        import wfdb
    except ImportError as error:
        raise RuntimeError("Install the benchmark extra to load PTB-XL waveforms") from error

    relative_path = record.filename_lr if resolution == "lr" else record.filename_hr
    record_path = root / relative_path
    if record_path.with_suffix(".hea").exists():
        signal, _ = wfdb.rdsamp(str(record_path))
    elif remote_directory is not None:
        signal, _ = wfdb.rdsamp(relative_path, pn_dir=remote_directory)
    else:
        raise FileNotFoundError(f"PTB-XL waveform is not available locally: {record_path}")
    return signal


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

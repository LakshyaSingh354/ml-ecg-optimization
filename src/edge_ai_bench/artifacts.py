"""Model artifact metadata and numerical validation helpers."""

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ArtifactMetadata:
    """Stable identity and size information for a model artifact."""

    path: str
    size_bytes: int
    sha256: str

    @property
    def size_kib(self) -> float:
        return self.size_bytes / 1024.0

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "size_kib": self.size_kib}


def inspect_artifact(path: Path, chunk_size: int = 1024 * 1024) -> ArtifactMetadata:
    """Return file size and SHA-256 without loading the complete artifact."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not path.is_file():
        raise FileNotFoundError(f"model artifact does not exist: {path}")

    digest = hashlib.sha256()
    size_bytes = 0
    with path.open("rb") as artifact:
        while chunk := artifact.read(chunk_size):
            digest.update(chunk)
            size_bytes += len(chunk)
    return ArtifactMetadata(path=str(path), size_bytes=size_bytes, sha256=digest.hexdigest())


@dataclass(frozen=True)
class OutputComparison:
    """Numerical agreement summary for corresponding model outputs."""

    shape: tuple[int, ...]
    max_absolute_error: float
    max_relative_error: float
    allclose: bool


def validate_outputs(reference_outputs: list[Any], candidate_outputs: list[Any], *, atol: float = 1e-5, rtol: float = 1e-5) -> list[OutputComparison]:
    """Compare every corresponding output tensor and reject count mismatches."""
    if len(reference_outputs) != len(candidate_outputs):
        raise ValueError("reference and candidate output counts must match")
    return [
        compare_outputs(reference, candidate, atol=atol, rtol=rtol)
        for reference, candidate in zip(reference_outputs, candidate_outputs)
    ]


def compare_outputs(reference: Any, candidate: Any, *, atol: float = 1e-5, rtol: float = 1e-5) -> OutputComparison:
    """Compare array-like outputs using the same tolerance contract as NumPy."""
    if atol < 0 or rtol < 0:
        raise ValueError("atol and rtol must be non-negative")
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("Install the benchmark extra to compare numerical outputs") from error

    reference_array = np.asarray(reference)
    candidate_array = np.asarray(candidate)
    if reference_array.shape != candidate_array.shape:
        return OutputComparison(
            shape=tuple(candidate_array.shape),
            max_absolute_error=float("inf"),
            max_relative_error=float("inf"),
            allclose=False,
        )
    difference = np.abs(reference_array - candidate_array)
    denominator = np.maximum(np.abs(reference_array), np.finfo(float).eps)
    return OutputComparison(
        shape=tuple(candidate_array.shape),
        max_absolute_error=float(np.max(difference)) if difference.size else 0.0,
        max_relative_error=float(np.max(difference / denominator)) if difference.size else 0.0,
        allclose=bool(np.allclose(reference_array, candidate_array, atol=atol, rtol=rtol)),
    )

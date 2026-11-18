"""ONNX Runtime session construction and inference."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import ArtifactMetadata, inspect_artifact


# SessionOptions defaults to ORT_ENABLE_ALL. "all" is that default, not a custom level.
ORT_DEFAULT_OPTIMIZATION = "all"


@dataclass(frozen=True)
class OnnxRuntimeConfig:
    """Explicit runtime settings that affect benchmark comparability."""

    provider: str = "CPUExecutionProvider"
    threads: int = 1
    optimization: str = ORT_DEFAULT_OPTIMIZATION
    profile: bool = False
    profile_prefix: str | None = None

    def __post_init__(self) -> None:
        if self.threads <= 0:
            raise ValueError("threads must be positive")
        if self.optimization not in {"disabled", "basic", "extended", "all"}:
            raise ValueError("optimization must be disabled, basic, extended, or all")


class OnnxRuntimeSession:
    """Small adapter that keeps ONNX Runtime optional at package import time."""

    def __init__(self, model_path: Path, config: OnnxRuntimeConfig) -> None:
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError("Install the benchmark extra to use ONNX Runtime") from error

        options = ort.SessionOptions()
        options.intra_op_num_threads = config.threads
        options.graph_optimization_level = {
            "disabled": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
            "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        }[config.optimization]
        if config.profile:
            options.enable_profiling = True
            if config.profile_prefix is not None:
                options.profile_file_prefix = config.profile_prefix
        available = ort.get_available_providers()
        if config.provider not in available:
            raise ValueError(f"provider {config.provider!r} is unavailable; available providers: {available}")
        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=[config.provider],
        )
        self.provider = config.provider
        self.threads = config.threads
        self.optimization = config.optimization
        self._profiling = config.profile
        self.artifact: ArtifactMetadata = inspect_artifact(model_path)

    @property
    def input_name(self) -> str:
        return self._session.get_inputs()[0].name

    @property
    def output_names(self) -> list[str]:
        return [output.name for output in self._session.get_outputs()]

    def run(self, inputs: Any) -> list[Any]:
        """Run the model using the first declared ONNX input."""
        return self._session.run(None, {self.input_name: inputs})

    def end_profiling(self) -> str:
        """Finish an ORT profile and return the Chrome-trace path."""
        if not self._profiling:
            raise RuntimeError("profiling was not enabled for this session")
        return self._session.end_profiling()

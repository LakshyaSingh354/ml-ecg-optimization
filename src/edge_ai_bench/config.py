"""Experiment configuration loading."""

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seed: int
    samples: int
    warmup: int
    iterations: int
    input_shape: tuple[int, int]
    output_labels: int
    model_name: str
    optimization: str
    provider: str
    threads: int
    output_path: Path


def load_config(path: Path) -> ExperimentConfig:
    """Load and validate a TOML experiment manifest."""
    with path.open("rb") as config_file:
        raw = tomllib.load(config_file)

    experiment = raw["experiment"]
    model = raw["model"]
    runtime = raw["runtime"]
    output = raw["output"]
    input_shape = tuple(model["input_shape"])
    if len(input_shape) != 2 or any(not isinstance(value, int) or value <= 0 for value in input_shape):
        raise ValueError("model.input_shape must contain two positive integers")
    for field in ("samples", "warmup", "iterations"):
        if experiment[field] < 0:
            raise ValueError(f"experiment.{field} must be non-negative")
    if experiment["iterations"] == 0:
        raise ValueError("experiment.iterations must be positive")
    if runtime["threads"] <= 0:
        raise ValueError("runtime.threads must be positive")

    return ExperimentConfig(
        name=experiment["name"],
        seed=experiment["seed"],
        samples=experiment["samples"],
        warmup=experiment["warmup"],
        iterations=experiment["iterations"],
        input_shape=input_shape,
        output_labels=model["output_labels"],
        model_name=model["name"],
        optimization=model["optimization"],
        provider=runtime["provider"],
        threads=runtime["threads"],
        output_path=Path(output["path"]),
    )

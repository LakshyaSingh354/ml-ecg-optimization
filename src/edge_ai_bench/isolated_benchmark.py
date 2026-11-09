"""Fresh-process benchmark worker used for reliable memory measurements."""

import argparse
import json
from pathlib import Path
import platform
import resource
import time
from typing import Any

import numpy as np

from .onnx_runtime import OnnxRuntimeConfig, OnnxRuntimeSession


def _current_rss_bytes() -> int:
    status_path = Path("/proc/self/status")
    if status_path.is_file():
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def _peak_rss_bytes() -> int:
    multiplier = 1 if platform.system() == "Darwin" else 1024
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * multiplier)


def benchmark(
    artifact: Path,
    signals_path: Path,
    threads: int,
    optimization: str,
    warmup: int,
    iterations: int,
    batch_sizes: tuple[int, ...],
) -> dict[str, Any]:
    signals = np.load(signals_path, mmap_mode="r")
    started = time.perf_counter()
    session = OnnxRuntimeSession(
        artifact,
        OnnxRuntimeConfig(threads=threads, optimization=optimization),
    )
    initialization_seconds = time.perf_counter() - started
    batches: dict[str, Any] = {}
    for batch_size in batch_sizes:
        batch = np.ascontiguousarray(signals[:batch_size]).transpose(0, 2, 1)
        for _ in range(warmup):
            session.run(batch)
        durations = []
        for _ in range(iterations):
            tick = time.perf_counter()
            session.run(batch)
            durations.append(time.perf_counter() - tick)
        ordered = sorted(durations)
        batches[str(batch_size)] = {
            "batch_size": batch_size,
            "latency_seconds_p50": ordered[len(ordered) // 2],
            "latency_seconds_p95": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
            "throughput_samples_per_second": float(batch_size / np.mean(durations)),
        }
    return {
        "initialization_seconds": initialization_seconds,
        "current_rss_bytes": _current_rss_bytes(),
        "peak_rss_bytes": _peak_rss_bytes(),
        "batches": batches,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--signals", type=Path, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--optimization", choices=("disabled", "basic", "extended", "all"), required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", required=True)
    args = parser.parse_args()
    print(json.dumps(benchmark(
        args.artifact,
        args.signals,
        args.threads,
        args.optimization,
        args.warmup,
        args.iterations,
        tuple(args.batch_sizes),
    )))


if __name__ == "__main__":
    main()

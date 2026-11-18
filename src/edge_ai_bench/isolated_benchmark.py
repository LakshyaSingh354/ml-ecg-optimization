"""Fresh-process benchmark worker used for reliable memory measurements."""

import argparse
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import time
from typing import Any

import numpy as np

from .onnx_runtime import OnnxRuntimeConfig, OnnxRuntimeSession


def _system_metadata() -> dict[str, Any]:
    lscpu = _lscpu_metadata()
    metadata: dict[str, Any] = {
        "python_version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "os": platform.platform(),
        "kernel": platform.release(),
        "onnxruntime_version": _onnxruntime_version(),
        "cpu_count_logical": os.cpu_count(),
        "process_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "lscpu": lscpu,
    }
    metadata["cpu_frequency"] = _frequency_metadata()
    return metadata


def _lscpu_metadata() -> dict[str, Any] | None:
    try:
        output = subprocess.run(
            ["lscpu"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    values = {}
    for line in output.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    wanted = (
        "Architecture",
        "Model name",
        "CPU family",
        "Model",
        "Thread(s) per core",
        "Core(s) per socket",
        "Socket(s)",
        "CPU max MHz",
        "CPU min MHz",
    )
    return {key: values[key] for key in wanted if key in values}


def _onnxruntime_version() -> str | None:
    try:
        import onnxruntime
    except ImportError:
        return None
    return onnxruntime.__version__


def _frequency_metadata() -> dict[str, Any]:
    paths = sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_cur_freq"))
    values = []
    governors = []
    for path in paths:
        try:
            values.append(int(path.read_text().strip()))
        except (OSError, ValueError):
            pass
        governor_path = path.with_name("scaling_governor")
        try:
            governors.append(governor_path.read_text().strip())
        except OSError:
            pass
    return {
        "source": "sysfs_scaling_cur_freq_khz",
        "sample_count": len(values),
        "min_khz": min(values) if values else None,
        "max_khz": max(values) if values else None,
        "mean_khz": float(np.mean(values)) if values else None,
        "governors": sorted(set(governors)),
    }


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
            "latency_seconds": durations,
        }
    return {
        "system": _system_metadata(),
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

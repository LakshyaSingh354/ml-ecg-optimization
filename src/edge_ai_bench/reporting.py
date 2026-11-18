"""Tables derived from an optimization manifest."""

from statistics import median
from typing import Any

from .pareto import frontier_from_rows


def runtime_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    """Name the CPU and ONNX Runtime version recorded with the timed runs."""
    variant = manifest["variants"][0]
    benchmark = variant["benchmark"]
    system = benchmark.get("system")
    if system is None:
        system = benchmark["isolated_runs"][0]["system"]
    cpu = system.get("lscpu") or {}
    environment = manifest.get("environment") or {}
    return {
        "cpu_model": cpu.get("Model name") or environment.get("cpu_model"),
        "onnxruntime_version": system.get("onnxruntime_version") or environment.get("onnxruntime_version"),
        "python_version": system.get("python_version") or environment.get("python"),
        "logical_cpus": system.get("cpu_count_logical") or environment.get("cpu_count"),
        "threads": manifest["experiment"].get("threads"),
        "thread_axis": manifest["experiment"].get("thread_axis", "intra_op_num_threads"),
    }


def isolated_initialization_seconds(variant: dict[str, Any]) -> float:
    runs = variant["benchmark"]["isolated_runs"]
    return float(median(run["initialization_seconds"] for run in runs))


def deployment_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per variant with initialization, RSS, and batch-1..16 latency."""
    rows = []
    for variant in manifest["variants"]:
        benchmark = variant["benchmark"]
        memory = benchmark["isolated_memory"]
        batches = benchmark["batches"]
        row = {
            "name": variant["name"],
            "initialization_seconds": isolated_initialization_seconds(variant),
            "current_rss_bytes": memory["current_rss_bytes_median"],
            "peak_rss_bytes": memory["peak_rss_bytes_median"],
            "macro_auroc": variant["metrics"]["macro_auroc"],
            "macro_f1": variant["metrics"]["macro_f1"],
            "macro_auroc_ci95": variant["metrics"].get("macro_auroc_ci95"),
            "macro_f1_ci95": variant["metrics"].get("macro_f1_ci95"),
            "size_bytes": variant["artifact"]["size_bytes"],
        }
        for batch_size, batch in batches.items():
            row[f"latency_p50_batch_{batch_size}"] = batch["latency_seconds_p50"]
            row[f"throughput_batch_{batch_size}"] = batch["throughput_samples_per_second"]
        rows.append(row)
    return rows


def batch_scaling_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for variant in manifest["variants"]:
        for batch_size, batch in variant["benchmark"]["batches"].items():
            rows.append({
                "name": variant["name"],
                "batch_size": int(batch_size),
                "latency_seconds_p50": batch["latency_seconds_p50"],
                "latency_seconds_p95": batch["latency_seconds_p95"],
                "throughput_samples_per_second": batch["throughput_samples_per_second"],
            })
    return rows


def displayed_frontiers(manifest: dict[str, Any]) -> dict[str, list[str]]:
    """Frontiers stored in the manifest, plus 2-D projections used by the figures."""
    stored = dict(manifest.get("pareto_frontiers") or {})
    rows = deployment_rows(manifest)
    stored["size_vs_batch1_latency"] = frontier_from_rows(
        rows,
        lambda row: (row["size_bytes"], row["latency_p50_batch_1"]),
    )
    stored["batch1_latency_vs_f1"] = frontier_from_rows(
        rows,
        lambda row: (row["latency_p50_batch_1"], -row["macro_f1"]),
    )
    stored["size_vs_f1"] = frontier_from_rows(
        rows,
        lambda row: (row["size_bytes"], -row["macro_f1"]),
    )
    stored["init_vs_batch1_latency"] = frontier_from_rows(
        rows,
        lambda row: (row["initialization_seconds"], row["latency_p50_batch_1"]),
    )
    return stored


def node_mechanism_summary(manifest: dict[str, Any]) -> dict[str, Any] | None:
    """Extract ORT profiler group shares when a schema v2 manifest recorded node profiles."""
    profiles = manifest.get("node_profiles")
    if not isinstance(profiles, dict):
        return None
    mechanism = profiles.get("mechanism")
    if not isinstance(mechanism, dict) or mechanism.get("status") != "measured":
        return None
    groups = mechanism.get("groups") or {}
    totals = mechanism.get("total_node_duration_us") or {}
    rows = []
    for label in ("int8_static_disabled", "int8_static_compiled_disabled", "int8_static_default"):
        share = (groups.get(label) or {}).get("share") or {}
        rows.append({
            "profile": label,
            "total_us": float(totals.get(label, 0.0)),
            "qdq_share": float(share.get("qdq", 0.0)),
            "quantized_kernel_share": float(share.get("quantized_kernel", 0.0)),
            "other_share": float(share.get("other", 0.0)),
        })
    return {
        "status": "measured",
        "rows": rows,
        "operator_deltas": mechanism.get("disabled_static_minus_compiled") or [],
    }


def static_int8_interaction(variants: list[dict[str, Any]]) -> dict[str, Any]:
    """Test whether the static-versus-compiled gap keeps its sign at ORT's default level."""
    by_name = {variant["name"]: variant for variant in variants}
    required = (
        "int8_static",
        "int8_static_compiled",
        "int8_static_default",
        "int8_static_compiled_default",
    )
    missing = [name for name in required if name not in by_name]
    if missing:
        return {"status": "incomplete", "missing": missing}

    def latency(name: str) -> float:
        return float(by_name[name]["benchmark"]["batches"]["1"]["latency_seconds_p50"])

    disabled_static = latency("int8_static")
    disabled_compiled = latency("int8_static_compiled")
    default_static = latency("int8_static_default")
    default_compiled = latency("int8_static_compiled_default")
    disabled_gap = disabled_static - disabled_compiled
    default_gap = default_static - default_compiled
    return {
        "status": "measured",
        "ort_default_optimization": "ORT_ENABLE_ALL",
        "disabled_runtime": {
            "int8_static_p50_seconds": disabled_static,
            "int8_static_compiled_p50_seconds": disabled_compiled,
            "static_minus_compiled_seconds": disabled_gap,
            "static_over_compiled": disabled_static / disabled_compiled,
        },
        "default_runtime": {
            "int8_static_p50_seconds": default_static,
            "int8_static_compiled_p50_seconds": default_compiled,
            "static_minus_compiled_seconds": default_gap,
            "static_over_compiled": default_static / default_compiled,
        },
        "interaction_survives": (disabled_gap > 0 and default_gap > 0) or (disabled_gap < 0 and default_gap < 0),
        "definition": (
            "The interaction survives when the sign of the batch-1 median latency gap "
            "between static INT8 and compiled static INT8 is the same with runtime "
            "optimization disabled and with ORT's default level, ORT_ENABLE_ALL."
        ),
    }


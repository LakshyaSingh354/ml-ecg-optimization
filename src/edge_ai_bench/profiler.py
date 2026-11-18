"""Per-node timings from ONNX Runtime's built-in profiler."""

import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from .onnx_runtime import OnnxRuntimeConfig, OnnxRuntimeSession


def _op_group(op_name: str) -> str:
    if op_name in {"QuantizeLinear", "DequantizeLinear"}:
        return "qdq"
    if op_name.startswith("QLinear") or op_name == "QGemm":
        return "quantized_kernel"
    return "other"


def summarize_node_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate complete Node events from an ORT Chrome trace."""
    totals: dict[str, float] = {}
    calls: dict[str, int] = {}
    for event in events:
        if event.get("cat") != "Node" or event.get("ph") not in (None, "X"):
            continue
        arguments = event.get("args") or {}
        op_name = str(arguments.get("op_name") or event.get("name") or "unknown")
        totals[op_name] = totals.get(op_name, 0.0) + float(event.get("dur") or 0.0)
        calls[op_name] = calls.get(op_name, 0) + 1
    total = float(sum(totals.values()))
    operators = [
        {
            "op_name": op_name,
            "calls": calls[op_name],
            "total_us": duration,
            "mean_us": duration / calls[op_name],
            "share": duration / total if total else 0.0,
            "group": _op_group(op_name),
        }
        for op_name, duration in sorted(totals.items(), key=lambda item: item[1], reverse=True)
    ]
    return {
        "total_node_duration_us": total,
        "operators": operators,
        "groups": _group_shares(operators, total),
    }


def _group_shares(operators: list[dict[str, Any]], total: float) -> dict[str, Any]:
    totals = {"qdq": 0.0, "quantized_kernel": 0.0, "other": 0.0}
    for row in operators:
        totals[row["group"]] += float(row["total_us"])
    return {
        "total_us": totals,
        "share": {name: (duration / total if total else 0.0) for name, duration in totals.items()},
    }


def profile_model(
    model_path: Path,
    batch: np.ndarray,
    *,
    threads: int,
    optimization: str,
    iterations: int = 5,
) -> dict[str, Any]:
    """Run a few inferences with ORT profiling enabled and summarize node time."""
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    with tempfile.TemporaryDirectory(prefix="edge-ai-ort-profile-") as directory:
        session = OnnxRuntimeSession(
            model_path,
            OnnxRuntimeConfig(
                threads=threads,
                optimization=optimization,
                profile=True,
                profile_prefix=str(Path(directory) / "ort"),
            ),
        )
        for _ in range(iterations):
            session.run(batch)
        trace_path = Path(session.end_profiling())
        events = json.loads(trace_path.read_text(encoding="utf-8"))
    summary = summarize_node_events(events)
    summary.update({
        "artifact": str(model_path),
        "threads": threads,
        "optimization": optimization,
        "iterations": iterations,
        "batch_shape": list(np.asarray(batch).shape),
    })
    return summary


def operator_deltas(left: dict[str, Any], right: dict[str, Any], *, limit: int = 12) -> list[dict[str, Any]]:
    """Compare total node time. Positive delta means `left` spent more time."""
    left_ops = {row["op_name"]: row for row in left["operators"]}
    right_ops = {row["op_name"]: row for row in right["operators"]}
    rows = []
    for op_name in set(left_ops) | set(right_ops):
        left_us = float(left_ops.get(op_name, {}).get("total_us", 0.0))
        right_us = float(right_ops.get(op_name, {}).get("total_us", 0.0))
        rows.append({
            "op_name": op_name,
            "group": _op_group(op_name),
            "left_us": left_us,
            "right_us": right_us,
            "delta_us": left_us - right_us,
        })
    rows.sort(key=lambda row: abs(row["delta_us"]), reverse=True)
    return rows[:limit]


def demonstrate_static_int8_mechanism(profiles: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Attribute the static-INT8 interaction to measured node time, not only graph diffs."""
    required = ("int8_static_disabled", "int8_static_compiled_disabled", "int8_static_default")
    missing = [name for name in required if name not in profiles]
    if missing:
        return {"status": "incomplete", "missing": missing}
    static = profiles["int8_static_disabled"]
    compiled = profiles["int8_static_compiled_disabled"]
    default = profiles["int8_static_default"]
    return {
        "status": "measured",
        "source": "onnxruntime_enable_profiling",
        "duration_unit": "microseconds",
        "groups": {
            "int8_static_disabled": static["groups"],
            "int8_static_compiled_disabled": compiled["groups"],
            "int8_static_default": default["groups"],
        },
        "total_node_duration_us": {
            "int8_static_disabled": static["total_node_duration_us"],
            "int8_static_compiled_disabled": compiled["total_node_duration_us"],
            "int8_static_default": default["total_node_duration_us"],
        },
        "disabled_static_minus_compiled": operator_deltas(static, compiled),
        "default_static_minus_disabled_static": operator_deltas(default, static),
    }

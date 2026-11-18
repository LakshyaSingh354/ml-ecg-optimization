"""Operator-level comparison for quantized ONNX Runtime artifacts."""

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


_QUANTIZED_PREFIXES = ("QLinear", "QGemm")
_COMPUTE_OPS = {
    "Add",
    "Conv",
    "Div",
    "Gemm",
    "GlobalAveragePool",
    "LayerNormalization",
    "MatMul",
    "Mul",
    "Softmax",
    "Sqrt",
}


def _artifact_identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _is_quantized_operator(op_type: str) -> bool:
    return op_type.startswith(_QUANTIZED_PREFIXES)


def _node_kind(op_type: str) -> str:
    if op_type in {"QuantizeLinear", "DequantizeLinear"}:
        return "qdq"
    if _is_quantized_operator(op_type):
        return "quantized"
    return "fp32"


def _boundary_count(nodes: list[Any]) -> int:
    producer_kind: dict[str, str] = {}
    boundaries = 0
    for node in nodes:
        kind = _node_kind(node.op_type)
        for input_name in node.input:
            previous = producer_kind.get(input_name)
            if previous is not None and previous != kind:
                boundaries += 1
        for output_name in node.output:
            producer_kind[output_name] = kind
    return boundaries


def _graph_summary(model: Any) -> dict[str, Any]:
    nodes = list(model.graph.node)
    producers = {output: node for node in nodes for output in node.output}
    op_counts = Counter(node.op_type for node in nodes)
    domain_counts = Counter(node.domain or "" for node in nodes)
    quantized = [node for node in nodes if _is_quantized_operator(node.op_type)]
    qdq = [node for node in nodes if node.op_type in {"QuantizeLinear", "DequantizeLinear"}]
    ort_domain = [node for node in nodes if (node.domain or "") == "com.microsoft"]
    fp32_fallback = [
        node for node in nodes
        if node.op_type in _COMPUTE_OPS and not _is_quantized_operator(node.op_type)
    ]
    qdq_backed_compute = [
        node for node in nodes
        if node.op_type in _COMPUTE_OPS
        and any(
            producers.get(input_name) is not None
            and producers[input_name].op_type == "DequantizeLinear"
            for input_name in node.input
        )
    ]
    return {
        "node_count": len(nodes),
        "initializer_count": len(model.graph.initializer),
        "operator_count": len(op_counts),
        "operator_counts": dict(sorted(op_counts.items())),
        "domain_counts": dict(sorted(domain_counts.items())),
        "quantize_linear_count": op_counts["QuantizeLinear"],
        "dequantize_linear_count": op_counts["DequantizeLinear"],
        "qdq_node_count": len(qdq),
        "quantized_operator_count": len(quantized),
        "quantized_operator_counts": dict(sorted(Counter(node.op_type for node in quantized).items())),
        "qdq_backed_compute_operator_count": len(qdq_backed_compute),
        "qdq_backed_compute_operator_counts": dict(
            sorted(Counter(node.op_type for node in qdq_backed_compute).items())
        ),
        "ort_domain_node_count": len(ort_domain),
        "ort_domain_operator_counts": dict(sorted(Counter(node.op_type for node in ort_domain).items())),
        "fp32_fallback_operator_count": len(fp32_fallback),
        "fp32_fallback_operator_counts": dict(sorted(Counter(node.op_type for node in fp32_fallback).items())),
        "quantization_boundary_count": _boundary_count(nodes),
        "input_count": len(model.graph.input),
        "output_count": len(model.graph.output),
    }


def analyze_operator_interaction(
    static_path: Path,
    compiled_static_path: Path,
    result_path: Path,
) -> dict[str, Any]:
    """Persist observable operator and graph-structure differences."""
    try:
        import onnx
    except ImportError as error:
        raise RuntimeError("Install the benchmark extra to analyze ONNX graphs") from error

    static_model = onnx.load(str(static_path), load_external_data=False)
    compiled_model = onnx.load(str(compiled_static_path), load_external_data=False)
    static_summary = _graph_summary(static_model)
    compiled_summary = _graph_summary(compiled_model)
    static_ops = set(static_summary["operator_counts"])
    compiled_ops = set(compiled_summary["operator_counts"])
    result = {
        "schema_version": 1,
        "analysis": "static_int8_operator_interaction",
        "artifacts": {
            "static_int8": _artifact_identity(static_path),
            "compiled_static_int8": _artifact_identity(compiled_static_path),
        },
        "definitions": {
            "quantized_operator": "Operator type starts with QLinear or equals QGemm.",
            "ort_domain_operator": "Node has domain com.microsoft; this is an observable ORT fused/custom-domain signal, not a universal fusion proof.",
            "fp32_fallback_operator": "Non-quantized node in the declared compute-op whitelist.",
            "quantization_boundary": "Graph edge whose producer and consumer categories differ among fp32, qdq, and quantized.",
        },
        "static_int8": static_summary,
        "compiled_static_int8": compiled_summary,
        "differences": {
            "node_count_delta": compiled_summary["node_count"] - static_summary["node_count"],
            "qdq_node_count_delta": compiled_summary["qdq_node_count"] - static_summary["qdq_node_count"],
            "quantized_operator_count_delta": compiled_summary["quantized_operator_count"] - static_summary["quantized_operator_count"],
            "fp32_fallback_operator_count_delta": (
                compiled_summary["fp32_fallback_operator_count"]
                - static_summary["fp32_fallback_operator_count"]
            ),
            "quantization_boundary_count_delta": (
                compiled_summary["quantization_boundary_count"]
                - static_summary["quantization_boundary_count"]
            ),
            "operator_types_removed": sorted(static_ops - compiled_ops),
            "operator_types_introduced": sorted(compiled_ops - static_ops),
        },
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result

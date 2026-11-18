from types import SimpleNamespace

from edge_ai_bench.operator_analysis import _boundary_count, _is_quantized_operator
from edge_ai_bench.optimizations import _confidence_half_width


def _node(op_type: str, inputs: list[str], outputs: list[str]):
    return SimpleNamespace(op_type=op_type, input=inputs, output=outputs)


def test_quantized_operator_detection():
    assert _is_quantized_operator("QLinearConv")
    assert _is_quantized_operator("QGemm")
    assert not _is_quantized_operator("Conv")


def test_boundary_count_tracks_category_transitions():
    nodes = [
        _node("QuantizeLinear", ["input"], ["q"]),
        _node("QLinearConv", ["q"], ["q_conv"]),
        _node("DequantizeLinear", ["q_conv"], ["fp"]),
        _node("Relu", ["fp"], ["output"]),
    ]
    assert _boundary_count(nodes) == 3


def test_confidence_interval_uses_t_critical_value():
    assert _confidence_half_width(1.0, 5) > 1.0
    assert _confidence_half_width(1.0, 1) == 0.0

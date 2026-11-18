from edge_ai_bench.profiler import demonstrate_static_int8_mechanism, summarize_node_events


def test_summarize_node_events_groups_operators():
    events = [
        {"cat": "Node", "ph": "X", "dur": 100.0, "args": {"op_name": "QuantizeLinear"}},
        {"cat": "Node", "ph": "X", "dur": 300.0, "args": {"op_name": "QLinearConv"}},
        {"cat": "Other", "ph": "X", "dur": 1000.0, "args": {"op_name": "Ignored"}},
    ]
    summary = summarize_node_events(events)
    assert summary["total_node_duration_us"] == 400.0
    assert summary["groups"]["share"]["qdq"] == 0.25


def test_demonstrate_static_int8_mechanism_reports_deltas():
    profiles = {
        "int8_static_disabled": {
            "operators": [{"op_name": "QLinearConv", "total_us": 400.0}],
            "groups": {"share": {"qdq": 0.1, "quantized_kernel": 0.8, "other": 0.1}},
            "total_node_duration_us": 400.0,
        },
        "int8_static_compiled_disabled": {
            "operators": [{"op_name": "QLinearConv", "total_us": 250.0}],
            "groups": {"share": {"qdq": 0.05, "quantized_kernel": 0.9, "other": 0.05}},
            "total_node_duration_us": 250.0,
        },
        "int8_static_default": {
            "operators": [{"op_name": "QLinearConv", "total_us": 420.0}],
            "groups": {"share": {"qdq": 0.08, "quantized_kernel": 0.85, "other": 0.07}},
            "total_node_duration_us": 420.0,
        },
    }
    result = demonstrate_static_int8_mechanism(profiles)
    assert result["status"] == "measured"
    assert result["disabled_static_minus_compiled"][0]["delta_us"] > 0
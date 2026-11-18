from edge_ai_bench.reporting import node_mechanism_summary


def test_node_mechanism_summary_extracts_group_shares():
    manifest = {
        "node_profiles": {
            "mechanism": {
                "status": "measured",
                "groups": {
                    "int8_static_disabled": {
                        "share": {"qdq": 0.5, "quantized_kernel": 0.2, "other": 0.3},
                    },
                    "int8_static_compiled_disabled": {
                        "share": {"qdq": 0.1, "quantized_kernel": 0.7, "other": 0.2},
                    },
                    "int8_static_default": {
                        "share": {"qdq": 0.3, "quantized_kernel": 0.4, "other": 0.3},
                    },
                },
                "total_node_duration_us": {
                    "int8_static_disabled": 1000.0,
                    "int8_static_compiled_disabled": 400.0,
                    "int8_static_default": 600.0,
                },
                "disabled_static_minus_compiled": [
                    {"op_name": "DequantizeLinear", "delta_us": 120.0},
                ],
            }
        }
    }
    summary = node_mechanism_summary(manifest)
    assert summary is not None
    assert len(summary["rows"]) == 3
    assert summary["rows"][0]["qdq_share"] == 0.5
    assert summary["operator_deltas"][0]["op_name"] == "DequantizeLinear"
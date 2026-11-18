from edge_ai_bench.reporting import displayed_frontiers, static_int8_interaction


def _variant(name: str, *, latency: float, size: int, f1: float) -> dict:
    return {
        "name": name,
        "artifact": {"size_bytes": size},
        "metrics": {"macro_f1": f1, "macro_auroc": 0.9},
        "benchmark": {
            "batches": {
                "1": {
                    "latency_seconds_p50": latency,
                    "latency_seconds_p95": latency * 1.2,
                    "throughput_samples_per_second": 1.0 / latency,
                }
            },
            "isolated_runs": [{"initialization_seconds": 0.1}],
            "isolated_memory": {
                "current_rss_bytes_median": 1000,
                "peak_rss_bytes_median": 1100,
            },
            "initialization_seconds_isolated_median": 0.1,
        },
    }


def test_static_int8_interaction_detects_survival():
    variants = [
        _variant("int8_static", latency=0.20, size=1000, f1=0.7),
        _variant("int8_static_compiled", latency=0.15, size=900, f1=0.7),
        _variant("int8_static_default", latency=0.18, size=1000, f1=0.7),
        _variant("int8_static_compiled_default", latency=0.14, size=900, f1=0.7),
    ]
    result = static_int8_interaction(variants)
    assert result["status"] == "measured"
    assert result["interaction_survives"] is True


def test_displayed_frontiers_adds_two_dimensional_projections():
    manifest = {
        "variants": [
            _variant("fast_large", latency=0.2, size=1000, f1=0.6),
            _variant("slow_small", latency=0.4, size=500, f1=0.8),
        ],
        "pareto_frontiers": {},
    }
    frontiers = displayed_frontiers(manifest)
    assert "size_vs_batch1_latency" in frontiers
    assert "batch1_latency_vs_f1" in frontiers
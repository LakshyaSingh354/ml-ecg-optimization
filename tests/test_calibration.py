import pytest

from edge_ai_bench.calibration import compare_calibration_manifests


def _manifest(method: str, auroc: float) -> dict:
    return {
        "experiment": {
            "calibration_method": method,
            "calibration_samples": 100,
            "calibration_record_ids_sha256": "same",
            "selected_test_records": 200,
            "quantization_per_channel": False,
        },
        "variants": [
            {
                "name": "int8_static",
                "metrics": {
                    "macro_auroc": auroc,
                    "macro_f1": 0.7,
                    "macro_auroc_ci95": [auroc - 0.01, auroc + 0.01],
                    "macro_f1_ci95": [0.69, 0.71],
                },
            }
        ],
    }


def test_compare_calibration_manifests_reports_deltas():
    result = compare_calibration_manifests(
        _manifest("minmax", 0.91),
        _manifest("percentile", 0.905),
    )
    assert result["reference_method"] == "minmax"
    assert result["candidate_method"] == "percentile"
    assert result["variants"][0]["delta_auroc"] == pytest.approx(-0.005)
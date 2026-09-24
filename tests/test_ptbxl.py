import csv

import pytest

from edge_ai_bench.ptbxl import load_metadata, multilabel_vector, normalize_signal


def test_ptbxl_metadata_loading_and_label_encoding(tmp_path):
    metadata_path = tmp_path / "ptbxl_database.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as metadata_file:
        writer = csv.DictWriter(
            metadata_file,
            fieldnames=["ecg_id", "patient_id", "strat_fold", "scp_codes", "filename_lr", "filename_hr"],
        )
        writer.writeheader()
        writer.writerow({
            "ecg_id": "1",
            "patient_id": "10",
            "strat_fold": "1",
            "scp_codes": "{'NORM': 100.0, 'PVC': 20.0}",
            "filename_lr": "records100/00000/00001_lr",
            "filename_hr": "records100/00000/00001_hr",
        })
    statement_map = {
        "NORM": {"diagnostic": True, "diagnostic_class": "NORM"},
        "PVC": {"diagnostic": True, "diagnostic_class": "CD"},
    }

    records = load_metadata(metadata_path, statement_map)

    assert records[0].patient_id == 10
    assert records[0].diagnostic_superclasses == ("NORM",)
    assert multilabel_vector(records[0].diagnostic_superclasses, ("NORM", "CD")) == [1, 0]


def test_signal_normalization_preserves_shape_and_standardizes_leads():
    np = pytest.importorskip("numpy")
    signal = np.array([[1.0, 2.0, 3.0], [10.0, 10.0, 10.0]], dtype=np.float32)

    normalized = normalize_signal(signal)

    assert normalized.shape == signal.shape
    np.testing.assert_allclose(normalized[0].mean(), 0.0, atol=1e-6)
    np.testing.assert_allclose(normalized[0].std(), 1.0, atol=1e-6)
    np.testing.assert_allclose(normalized[1], 0.0, atol=1e-6)
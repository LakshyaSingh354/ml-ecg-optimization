import csv

import pytest

from edge_ai_bench.ptbxl import load_metadata, load_statement_map, load_waveform, multilabel_vector, normalize_signal


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


def test_statement_map_and_waveform_resolution_contract(tmp_path, monkeypatch):
    statement_path = tmp_path / "scp_statements.csv"
    statement_path.write_text(
        "scp_code,diagnostic,diagnostic_class,form,rhythm\n"
        "NORM,1,NORM,0,0\n"
        "PVC,1,CD,1,0\n",
        encoding="utf-8",
    )
    statements = load_statement_map(statement_path)
    record = load_metadata_fixture(tmp_path, statements)
    calls = []

    class FakeWFDB:
        @staticmethod
        def rdsamp(path):
            calls.append(path)
            return [[0.0]], {"fs": 100}

    monkeypatch.setitem(__import__("sys").modules, "wfdb", FakeWFDB)
    (tmp_path / f"{record.filename_lr}.hea").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / f"{record.filename_lr}.hea").touch()
    (tmp_path / f"{record.filename_hr}.hea").touch()
    load_waveform(record, tmp_path, resolution="lr")
    load_waveform(record, tmp_path, resolution="hr")

    assert calls == [str(tmp_path / record.filename_lr), str(tmp_path / record.filename_hr)]
    with pytest.raises(ValueError, match="resolution"):
        load_waveform(record, tmp_path, resolution="medium")

    (tmp_path / f"{record.filename_lr}.hea").unlink()
    with pytest.raises(FileNotFoundError, match="not available locally"):
        load_waveform(record, tmp_path, resolution="lr")


def load_metadata_fixture(tmp_path, statement_map):
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
            "scp_codes": "{'NORM': 100.0}",
            "filename_lr": "records100/00000/00001_lr",
            "filename_hr": "records100/00000/00001_hr",
        })
    return load_metadata(metadata_path, statement_map)[0]
import pytest

from edge_ai_bench.real_baseline import RealBaselineConfig, _features, _onnx_probabilities, _select_records
from edge_ai_bench.ptbxl import PTBXLRecord


def test_real_baseline_record_selection_respects_folds_and_limit():
    records = [
        PTBXLRecord(index, index, fold, ("NORM",), "lr", "hr")
        for index, fold in enumerate([1, 8, 9, 10, 10])
    ]

    selected = _select_records(records, 1, 8, 1)

    assert [record.fold for record in selected] == [1]
    assert [record.fold for record in _select_records(records, 10, 10, 10)] == [10, 10]


def test_real_baseline_features_are_fixed_size():
    np = pytest.importorskip("numpy")
    signal = np.arange(100 * 12, dtype=np.float32).reshape(100, 12)

    features = _features(signal, points=8)

    assert features.shape == (12 * 8 + 12 + 12,)
    assert features.dtype == np.float32


def test_onnx_probability_maps_become_a_matrix():
    np = pytest.importorskip("numpy")

    probabilities = _onnx_probabilities([[None], [{0: 0.25, 1: 0.75}, {0: 0.9, 1: 0.1}]], 2)

    np.testing.assert_allclose(probabilities, [[0.25, 0.75], [0.9, 0.1]])


def test_real_baseline_defaults_are_representative():
    config = RealBaselineConfig()

    assert (config.train_limit, config.validation_limit, config.test_limit) == (1000, 200, 200)
    assert config.checkpoint_path.suffix == ".joblib"
    assert config.artifact_path.suffix == ".onnx"


def test_real_baseline_rejects_invalid_download_workers():
    with pytest.raises(ValueError, match="download_workers"):
        RealBaselineConfig(download_workers=0)

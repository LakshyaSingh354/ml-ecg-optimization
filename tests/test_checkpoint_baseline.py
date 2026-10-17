import numpy as np

from edge_ai_bench.checkpoint_baseline import CHECKPOINT_CLASSES, CheckpointBaselineConfig, _standardize


def test_checkpoint_baseline_uses_upstream_class_order():
    assert CHECKPOINT_CLASSES == ("CD", "HYP", "MI", "NORM", "STTC")


def test_checkpoint_baseline_standardizes_using_training_data_only():
    train = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    validation = train + 100
    test = train - 100

    normalized_train, normalized_validation, normalized_test, scaler = _standardize(train, validation, test)

    assert normalized_train.shape == train.shape
    assert normalized_validation.shape == validation.shape
    assert normalized_test.shape == test.shape
    assert scaler.mean_.shape == (1,)
    assert not np.allclose(normalized_validation.mean(), 0.0)

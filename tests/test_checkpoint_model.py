from pathlib import Path

import pytest

from edge_ai_bench.checkpoint_model import load_checkpoint, predict_logits


CHECKPOINT = Path(__file__).parents[1] / "artifacts/ecg_hybrid_net_exp12_resnet_medium.pth"


@pytest.mark.integration
def test_downloaded_checkpoint_loads_strictly_and_runs():
    torch = pytest.importorskip("torch")
    if not CHECKPOINT.exists():
        pytest.skip("downloaded checkpoint is not present")

    model = load_checkpoint(CHECKPOINT)
    logits = predict_logits(model, torch.zeros((1, 12, 1000), dtype=torch.float32))

    assert tuple(logits.shape) == (1, 5)
    assert torch.isfinite(logits).all()

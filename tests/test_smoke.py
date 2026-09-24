import json
from pathlib import Path

from edge_ai_bench.benchmark import run
from edge_ai_bench.config import load_config
from edge_ai_bench.synthetic import make_batch


ROOT = Path(__file__).parents[1]


def test_config_and_benchmark_are_deterministic():
    config = load_config(ROOT / "configs/synthetic.toml")
    first = run(config, make_batch(config.samples, config.input_shape[0], config.input_shape[1], config.output_labels, config.seed))
    second = run(config, make_batch(config.samples, config.input_shape[0], config.input_shape[1], config.output_labels, config.seed))

    assert first["dataset"] == second["dataset"]
    assert first["prediction_shape"] == [config.samples, config.output_labels]
    assert first["status"] == "synthetic_smoke_only"
    assert first["metrics"]["throughput_samples_per_second"] > 0
    assert first["artifact"] is None
    assert first["validation"] is None


def test_result_record_is_json_serializable():
    config = load_config(ROOT / "configs/synthetic.toml")
    result = run(config, make_batch(config.samples, 12, 128, config.output_labels, config.seed))
    json.dumps(result)

"""Command-line interface for benchmark experiments."""

import argparse
import json
from pathlib import Path

from .benchmark import run, write_result
from .config import load_config
from .real_baseline import RealBaselineConfig, run_baseline
from .synthetic import make_batch


def main() -> None:
    parser = argparse.ArgumentParser(prog="edge-ai-bench")
    subparsers = parser.add_subparsers(dest="command", required=True)
    benchmark_parser = subparsers.add_parser("benchmark", help="run the synthetic smoke benchmark")
    benchmark_parser.add_argument("--config", type=Path, required=True)
    baseline_parser = subparsers.add_parser("real-baseline", help="run a bounded PTB-XL baseline")
    baseline_parser.add_argument("--train-limit", type=int, default=1000)
    baseline_parser.add_argument("--validation-limit", type=int, default=200)
    baseline_parser.add_argument("--test-limit", type=int, default=200)
    args = parser.parse_args()

    if args.command == "benchmark":
        config = load_config(args.config)
        batch = make_batch(config.samples, config.input_shape[0], config.input_shape[1], config.output_labels, config.seed)
        result = run(config, batch)
        write_result(result, str(config.output_path))
        print(f"wrote {config.output_path}")
    elif args.command == "real-baseline":
        result = run_baseline(RealBaselineConfig(
            train_limit=args.train_limit,
            validation_limit=args.validation_limit,
            test_limit=args.test_limit,
        ))
        print(json.dumps({
            "status": result["status"],
            "splits": result["splits"],
            "metrics": result["metrics"],
            "artifact": result["artifact"],
            "checkpoint": result["checkpoint"],
            "benchmark": result["benchmark"],
        }, indent=2))


if __name__ == "__main__":
    main()

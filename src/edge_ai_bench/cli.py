"""Command-line interface for benchmark experiments."""

import argparse
import json
from pathlib import Path

from .benchmark import run, write_result
from .checkpoint_baseline import CheckpointBaselineConfig, CheckpointSweepConfig, run_checkpoint_baseline, run_checkpoint_sweep
from .config import load_config
from .real_baseline import RealBaselineConfig, run_baseline
from .optimizations import OptimizationConfig, run_optimizations
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
    checkpoint_parser = subparsers.add_parser("checkpoint-baseline", help="evaluate the downloaded ResNet-Transformer checkpoint")
    checkpoint_parser.add_argument("--train-limit", type=int, default=1000)
    checkpoint_parser.add_argument("--validation-limit", type=int, default=200)
    checkpoint_parser.add_argument("--test-limit", type=int, default=200)
    checkpoint_parser.add_argument("--onnx-threads", type=int, default=1)
    checkpoint_parser.add_argument(
        "--onnx-optimization",
        choices=("disabled", "basic", "extended", "all"),
        default="all",
    )
    sweep_parser = subparsers.add_parser("checkpoint-sweep", help="sweep FP32 ONNX Runtime settings")
    sweep_parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8])
    sweep_parser.add_argument(
        "--optimizations",
        nargs="+",
        choices=("disabled", "basic", "extended", "all"),
        default=["disabled", "basic", "extended", "all"],
    )
    optimization_parser = subparsers.add_parser(
        "optimize-checkpoint",
        help="compile and quantize the checkpoint deployment artifact",
    )
    optimization_parser.add_argument("--threads", type=int, default=2)
    optimization_parser.add_argument("--train-limit", type=int, default=1000)
    optimization_parser.add_argument("--test-limit", type=int, default=200)
    optimization_parser.add_argument("--calibration-limit", type=int, default=100)
    optimization_parser.add_argument("--result-path", type=Path, default=Path("results/ecg-hybrid-net-optimizations.json"))
    optimization_parser.add_argument("--isolated-repeats", type=int, default=3)
    optimization_parser.add_argument(
        "--per-tensor",
        action="store_true",
        help="use per-tensor rather than per-channel weight quantization",
    )
    optimization_parser.add_argument(
        "--calibration-method",
        choices=("minmax", "percentile"),
        default="minmax",
    )
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
    elif args.command == "checkpoint-sweep":
        result = run_checkpoint_sweep(CheckpointSweepConfig(
            thread_counts=tuple(args.threads),
            optimizations=tuple(args.optimizations),
        ))
        print(json.dumps({
            "status": result["status"],
            "artifact": result["experiment"]["artifact"],
            "results": result["results"],
        }, indent=2))
    elif args.command == "optimize-checkpoint":
        result = run_optimizations(OptimizationConfig(
            threads=args.threads,
            train_limit=args.train_limit,
            test_limit=args.test_limit,
            calibration_limit=args.calibration_limit,
            isolated_repeats=args.isolated_repeats,
            quantization_per_channel=not args.per_tensor,
            calibration_method=args.calibration_method,
            result_path=args.result_path,
        ))
        print(json.dumps({
            "status": result["status"],
            "variants": result["variants"],
        }, indent=2))
    elif args.command == "checkpoint-baseline":
        result = run_checkpoint_baseline(CheckpointBaselineConfig(
            train_limit=args.train_limit,
            validation_limit=args.validation_limit,
            test_limit=args.test_limit,
        ))
        print(json.dumps({
            "status": result["status"],
            "splits": result["splits"],
            "metrics": result["metrics"],
            "artifact": result["artifact"],
            "validation": result["validation"],
            "benchmark": result["benchmark"],
        }, indent=2))


if __name__ == "__main__":
    main()

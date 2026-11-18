"""Command-line interface for benchmark experiments."""

import argparse
import json
from pathlib import Path

from .benchmark import run, write_result
from .checkpoint_baseline import CheckpointBaselineConfig, CheckpointSweepConfig, run_checkpoint_baseline, run_checkpoint_sweep
from .config import load_config
from .real_baseline import RealBaselineConfig, run_baseline
from .calibration import compare_calibration_manifests
from .optimizations import OptimizationConfig, run_optimizations
from .operator_analysis import analyze_operator_interaction
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
        choices=("minmax", "percentile", "entropy"),
        default="minmax",
    )
    optimization_parser.add_argument(
        "--reference-manifest",
        type=Path,
        default=None,
        help="minmax (or other) manifest for calibration-method comparison",
    )
    optimization_parser.add_argument("--thread-counts", type=int, nargs="+", default=[1, 2, 4])
    optimization_parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    optimization_parser.add_argument("--no-profile", action="store_true")
    optimization_parser.add_argument(
        "--paper-protocol",
        action="store_true",
        help=(
            "match the frozen paper split (full PTB-XL folds 1-8/10 test selection, "
            "100-record MinMax calibration, per-tensor weights, two primary threads, "
            "five isolated repeats, and thread sweep 1/2/4)"
        ),
    )
    compare_parser = subparsers.add_parser(
        "compare-calibration",
        help="compare percentile/entropy manifests against a reference manifest",
    )
    compare_parser.add_argument("--reference", type=Path, required=True)
    compare_parser.add_argument("--candidate", type=Path, required=True)
    compare_parser.add_argument("--result-path", type=Path, default=Path("results/calibration-comparison.json"))
    analysis_parser = subparsers.add_parser(
        "analyze-operators",
        help="compare Static INT8 and compiled Static INT8 graph operators",
    )
    analysis_parser.add_argument(
        "--static-path",
        type=Path,
        default=Path("artifacts/ecg_hybrid_net_exp12_resnet_medium-int8-static.onnx"),
    )
    analysis_parser.add_argument(
        "--compiled-static-path",
        type=Path,
        default=Path("artifacts/ecg_hybrid_net_exp12_resnet_medium-ort-optimized-int8-static.onnx"),
    )
    analysis_parser.add_argument(
        "--result-path",
        type=Path,
        default=Path("results/ecg-hybrid-net-static-int8-operator-analysis.json"),
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
        if args.paper_protocol:
            args.train_limit = 20_000
            args.test_limit = 3_000
            args.calibration_limit = 100
            args.isolated_repeats = 5
            args.per_tensor = True
            args.threads = 2
            args.thread_counts = [1, 2, 4]
            args.calibration_method = "minmax"
            default_result = Path("results/ecg-hybrid-net-optimizations.json")
            if args.result_path == default_result:
                args.result_path = Path(
                    "results/ecg-hybrid-net-final-repeat5-pertensor-minmax-statistical-v2.json"
                )
        result = run_optimizations(OptimizationConfig(
            threads=args.threads,
            thread_counts=tuple(args.thread_counts),
            train_limit=args.train_limit,
            test_limit=args.test_limit,
            calibration_limit=args.calibration_limit,
            isolated_repeats=args.isolated_repeats,
            quantization_per_channel=not args.per_tensor,
            calibration_method=args.calibration_method,
            bootstrap_resamples=args.bootstrap_resamples,
            profile_nodes=not args.no_profile,
            reference_manifest=args.reference_manifest,
            result_path=args.result_path,
        ))
        summary = {
            "status": result["status"],
            "schema_version": result.get("schema_version"),
            "static_int8_interaction": result.get("static_int8_interaction"),
            "pareto_frontier": result.get("pareto_frontier"),
            "environment": result.get("environment"),
        }
        if "calibration_comparison" in result:
            summary["calibration_comparison"] = result["calibration_comparison"]
        print(json.dumps(summary, indent=2))
    elif args.command == "compare-calibration":
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
        candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
        comparison = compare_calibration_manifests(reference, candidate)
        args.result_path.parent.mkdir(parents=True, exist_ok=True)
        args.result_path.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(comparison, indent=2))
    elif args.command == "analyze-operators":
        result = analyze_operator_interaction(
            args.static_path,
            args.compiled_static_path,
            args.result_path,
        )
        print(json.dumps(result, indent=2))
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

"""Command-line interface for benchmark experiments."""

import argparse
from pathlib import Path

from .benchmark import run, write_result
from .config import load_config
from .synthetic import make_batch


def main() -> None:
    parser = argparse.ArgumentParser(prog="edge-ai-bench")
    subparsers = parser.add_subparsers(dest="command", required=True)
    benchmark_parser = subparsers.add_parser("benchmark", help="run the synthetic smoke benchmark")
    benchmark_parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "benchmark":
        config = load_config(args.config)
        batch = make_batch(config.samples, config.input_shape[0], config.input_shape[1], config.output_labels, config.seed)
        result = run(config, batch)
        write_result(result, str(config.output_path))
        print(f"wrote {config.output_path}")


if __name__ == "__main__":
    main()

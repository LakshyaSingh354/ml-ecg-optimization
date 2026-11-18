"""Point manifest-lock.json at a new result manifest and record its SHA-256."""

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="authoritative JSON manifest under results/")
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path(__file__).with_name("manifest-lock.json"),
        help="manifest lock file to update",
    )
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    experiment = payload["experiment"]
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    lock["manifest"] = str(Path("..") / manifest.relative_to(manifest.parents[1])).replace("\\", "/")
    lock["sha256"] = digest
    lock["protocol"] = {
        "dataset": experiment.get("dataset", "PTB-XL"),
        "selected_test_records": experiment.get("selected_test_records"),
        "threads": experiment.get("threads"),
        "calibration_samples": experiment.get("calibration_samples"),
        "calibration_method": experiment.get("calibration_method"),
        "weight_quantization": "per-channel" if experiment.get("quantization_per_channel") else "per-tensor",
        "isolated_repeats": experiment.get("isolated_repeats"),
        "timed_iterations_per_subprocess": 10,
        "batch_sizes": experiment.get("batch_sizes"),
    }
    lock["variants"] = [variant["name"] for variant in payload["variants"]]
    args.lock.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(f"updated {args.lock}")
    print(f"sha256={digest}")


if __name__ == "__main__":
    main()
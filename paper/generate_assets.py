"""Generate workshop-paper tables and figures from the authoritative manifest."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.ticker import MultipleLocator


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/ecg-hybrid-net-final-repeat5-pertensor-minmax.json"
LOCK = ROOT / "paper/manifest-lock.json"
OUTPUT = ROOT / "paper"


def load_rows() -> list[dict]:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    digest = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    if digest != lock["sha256"]:
        raise RuntimeError(
            f"authoritative manifest hash mismatch: expected {lock['sha256']}, got {digest}"
        )
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = []
    for variant in data["variants"]:
        batch = variant["benchmark"]["batches"]["1"]
        rows.append({
            "variant": variant["name"],
            "auroc": variant["metrics"]["macro_auroc"],
            "f1": variant["metrics"]["macro_f1"],
            "size_kib": variant["artifact"]["size_kib"],
            "latency_ms": batch["latency_seconds_p50"] * 1000,
            "p95_ms": batch["latency_seconds_p95"] * 1000,
            "throughput": batch["throughput_samples_per_second"],
            "rss_mib": variant["benchmark"]["isolated_memory"]["current_rss_bytes_median"] / 2**20,
            "peak_rss_mib": variant["benchmark"]["isolated_memory"]["peak_rss_bytes_max"] / 2**20,
        })
    return rows


def write_tables(rows: list[dict]) -> None:
    tables = OUTPUT / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (tables / "final_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    latex = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Variant & AUROC & F1 & Size (KiB) & p50 (ms) & p95 (ms) & Throughput \\",
        r"\midrule",
    ]
    for row in rows:
        latex.append(
            f"{row['variant'].replace('_', r'\\_')} & "
            f"{row['auroc']:.4f} & {row['f1']:.4f} & {row['size_kib']:.1f} & "
            f"{row['latency_ms']:.2f} & {row['p95_ms']:.2f} & {row['throughput']:.2f} \\\\"
        )
    latex.extend([r"\bottomrule", r"\end{tabular}", ""])
    (tables / "final_results.tex").write_text("\n".join(latex), encoding="utf-8")


def plot_accuracy_latency(rows: list[dict]) -> None:
    fig, axis = plt.subplots(figsize=(7.2, 4.5))
    for row in rows:
        axis.scatter(row["latency_ms"], row["f1"], s=row["size_kib"]/ 64, alpha=0.85, label=row["variant"])
    offsets = {
        "fp32": (-32, -14),
        "fp32_compiled": (-10, 10),
        "int8_dynamic": (8, -15),
        "int8_static": (8, -15),
        "int8_dynamic_compiled": (8, 10),
        "int8_static_compiled": (8, 8),
    }
    for row in rows:
        axis.annotate(
            row["variant"],
            (row["latency_ms"], row["f1"]),
            xytext=offsets[row["variant"]],
            textcoords="offset points",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )
    axis.set_xlabel("Batch-1 median latency (ms)")
    axis.set_ylabel("Macro F1")
    axis.set_title("Accuracy–latency trade-off")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT / "figures/accuracy_latency.png", dpi=520)
    plt.close(fig)


def plot_size_throughput(rows: list[dict]) -> None:
    fig, axis = plt.subplots(figsize=(7.2, 4.5))
    for row in rows:
        axis.scatter(row["size_kib"], row["throughput"], s=55, alpha=0.85, label=row["variant"])
    offsets = {
        "fp32": (8, 8),
        "fp32_compiled": (8, -16),
        "int8_dynamic": (8, 8),
        "int8_static": (8, -16),
        "int8_dynamic_compiled": (8, -16),
        "int8_static_compiled": (8, 8),
    }
    for row in rows:
        axis.annotate(
            row["variant"],
            (row["size_kib"], row["throughput"]),
            xytext=offsets[row["variant"]],
            textcoords="offset points",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )
    axis.set_xlabel("Serialized artifact size (KiB)")
    axis.set_ylabel("Throughput (samples/s)")
    axis.set_title("Model size–throughput trade-off")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT / "figures/size_throughput.png", dpi=220)
    plt.close(fig)


def plot_pipeline() -> None:
    fig, axis = plt.subplots(figsize=(11, 3.2))
    axis.set_xlim(0, 11)
    axis.set_ylim(0, 3)
    axis.axis("off")
    boxes = [
        (0.3, 1.1, "FP32 ONNX\nbaseline"),
        (3.0, 1.1, "ORT graph\nserialization"),
        (5.7, 1.1, "INT8\nquantization"),
        (8.4, 1.1, "Quantize then\ncompile"),
    ]
    for x, y, label in boxes:
        axis.add_patch(FancyBboxPatch((x, y), 2.0, 0.8, boxstyle="round,pad=0.04", facecolor="#e8f1fb", edgecolor="#24527a"))
        axis.text(x + 1.0, y + 0.4, label, ha="center", va="center", fontsize=10)
    for x1, x2 in [(2.3, 3.0), (5.0, 5.7), (7.7, 8.4)]:
        axis.add_patch(FancyArrowPatch((x1, 1.5), (x2, 1.5), arrowstyle="->", mutation_scale=15, linewidth=1.5))
    axis.text(1.3, 0.55, "Condition 1", ha="center", fontsize=9)
    axis.text(4.0, 0.55, "Condition 2", ha="center", fontsize=9)
    axis.text(6.7, 0.55, "Condition 3", ha="center", fontsize=9)
    axis.text(9.4, 0.55, "Condition 4", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT / "figures/optimization_pipeline.png", dpi=220)
    plt.close(fig)


def main() -> None:
    rows = load_rows()
    (OUTPUT / "figures").mkdir(parents=True, exist_ok=True)
    write_tables(rows)
    plot_accuracy_latency(rows)
    plot_size_throughput(rows)
    plot_pipeline()


if __name__ == "__main__":
    main()

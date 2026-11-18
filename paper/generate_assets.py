"""Generate workshop-paper tables and figures from the authoritative manifest."""

import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from edge_ai_bench.reporting import (  # noqa: E402
    batch_scaling_rows,
    deployment_rows,
    displayed_frontiers,
    node_mechanism_summary,
    runtime_identity,
    static_int8_interaction,
)

MANIFEST = ROOT / "results/ecg-hybrid-net-final-repeat5-pertensor-minmax-statistical.json"
LOCK = ROOT / "paper/manifest-lock.json"
OUTPUT = ROOT / "paper"


def load_manifest() -> dict:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    digest = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    if digest != lock["sha256"]:
        raise RuntimeError(
            f"authoritative manifest hash mismatch: expected {lock['sha256']}, got {digest}"
        )
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def load_rows(manifest: dict) -> list[dict]:
    rows = []
    for row in deployment_rows(manifest):
        batch_one = row.get("latency_p50_batch_1")
        if batch_one is None:
            continue
        metrics = next(
            variant["metrics"]
            for variant in manifest["variants"]
            if variant["name"] == row["name"]
        )
        bootstrap = metrics.get("bootstrap") or {}
        auroc_ci = row.get("macro_auroc_ci95") or bootstrap.get("macro_auroc_ci95")
        f1_ci = row.get("macro_f1_ci95") or bootstrap.get("macro_f1_ci95")
        benchmark = next(
            variant["benchmark"]
            for variant in manifest["variants"]
            if variant["name"] == row["name"]
        )
        batch = benchmark["batches"]["1"]
        rows.append({
            "variant": row["name"],
            "auroc": row["macro_auroc"],
            "f1": row["macro_f1"],
            "auroc_ci95_low": auroc_ci[0] if auroc_ci else None,
            "auroc_ci95_high": auroc_ci[1] if auroc_ci else None,
            "f1_ci95_low": f1_ci[0] if f1_ci else None,
            "f1_ci95_high": f1_ci[1] if f1_ci else None,
            "size_kib": row["size_bytes"] / 1024,
            "latency_ms": batch_one * 1000,
            "p95_ms": benchmark["batches"]["1"]["latency_seconds_p95"] * 1000,
            "throughput": row.get("throughput_batch_1", batch["throughput_samples_per_second"]),
            "init_ms": row["initialization_seconds"] * 1000,
            "rss_mib": row["current_rss_bytes"] / 2**20,
            "peak_rss_mib": row["peak_rss_bytes"] / 2**20,
            "subprocess_p50_std_ms": batch.get("subprocess_p50_std_seconds", 0.0) * 1000,
            "subprocess_p50_ci95_low_ms": batch["subprocess_p50_ci95_seconds"][0] * 1000
            if batch.get("subprocess_p50_ci95_seconds")
            else None,
            "subprocess_p50_ci95_high_ms": batch["subprocess_p50_ci95_seconds"][1] * 1000
            if batch.get("subprocess_p50_ci95_seconds")
            else None,
        })
    return rows


def _format_ci(low: float | None, high: float | None) -> str:
    if low is None or high is None:
        return ""
    return f"[{low:.4f}, {high:.4f}]"


def write_tables(rows: list[dict], manifest: dict, frontiers: dict[str, list[str]]) -> None:
    tables = OUTPUT / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    fields = [
        "variant",
        "auroc",
        "auroc_ci95",
        "f1",
        "f1_ci95",
        "size_kib",
        "init_ms",
        "rss_mib",
        "peak_rss_mib",
        "latency_ms",
        "p95_ms",
        "throughput",
    ]
    enriched = []
    for row in rows:
        enriched.append({
            **row,
            "auroc_ci95": _format_ci(row["auroc_ci95_low"], row["auroc_ci95_high"]),
            "f1_ci95": _format_ci(row["f1_ci95_low"], row["f1_ci95_high"]),
            "on_pareto_size_latency_f1": row["variant"] in frontiers.get("size_latency_f1", []),
        })
    with (tables / "final_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields + ["on_pareto_size_latency_f1"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(enriched)

    latex = [
        r"\begin{tabular}{lrrrrrrrrr}",
        r"\toprule",
        r"Variant & AUROC & F1 & Size (KiB) & Init (ms) & RSS (MiB) & p50 (ms) & p95 (ms) & Throughput \\",
        r"\midrule",
    ]
    has_ci = any(row.get("auroc_ci95_low") is not None for row in rows)
    for row in rows:
        auroc = f"{row['auroc']:.4f}"
        f1 = f"{row['f1']:.4f}"
        if has_ci and row.get("auroc_ci95_low") is not None:
            auroc = (
                f"{auroc} {{\\scriptsize "
                f"({row['auroc_ci95_low']:.3f}, {row['auroc_ci95_high']:.3f})}}"
            )
        if has_ci and row.get("f1_ci95_low") is not None:
            f1 = (
                f"{f1} {{\\scriptsize "
                f"({row['f1_ci95_low']:.3f}, {row['f1_ci95_high']:.3f})}}"
            )
        latex.append(
            f"{row['variant'].replace('_', r'\_')} & "
            f"{auroc} & {f1} & {row['size_kib']:.1f} & "
            f"{row['init_ms']:.1f} & {row['rss_mib']:.1f} & "
            f"{row['latency_ms']:.2f} & {row['p95_ms']:.2f} & {row['throughput']:.2f} \\\\"
        )
    latex.extend([r"\bottomrule", r"\end{tabular}", ""])
    (tables / "final_results.tex").write_text("\n".join(latex), encoding="utf-8")

    identity = runtime_identity(manifest)
    (tables / "runtime_identity.json").write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    (tables / "pareto_frontiers.json").write_text(json.dumps(frontiers, indent=2) + "\n", encoding="utf-8")

    with (tables / "batch_scaling.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "name",
                "batch_size",
                "latency_seconds_p50",
                "latency_seconds_p95",
                "throughput_samples_per_second",
            ],
        )
        writer.writeheader()
        writer.writerows(batch_scaling_rows(manifest))

    sweep_rows = []
    for variant in manifest["variants"]:
        sweep = variant["benchmark"].get("thread_sweep")
        if not sweep:
            continue
        for point in sweep:
            batch_one = point["batches"]["1"]
            sweep_rows.append({
                "variant": variant["name"],
                "threads": point["threads"],
                "init_ms": point["initialization_seconds_median"] * 1000,
                "rss_mib": point["current_rss_bytes_median"] / 2**20,
                "latency_ms": batch_one["latency_seconds_p50"] * 1000,
                "throughput": batch_one["throughput_samples_per_second"],
            })
    if sweep_rows:
        with (tables / "thread_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(sweep_rows[0]))
            writer.writeheader()
            writer.writerows(sweep_rows)

    subtitle = (
        f"CPU: {identity.get('cpu_model') or 'unknown'}; "
        f"ONNX Runtime {identity.get('onnxruntime_version') or 'unknown'}; "
        f"primary threads={identity.get('threads')}"
    )
    (tables / "runtime_identity.txt").write_text(subtitle + "\n", encoding="utf-8")


def write_supplementary_tables(manifest: dict[str, Any]) -> None:
    tables = OUTPUT / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    interaction = manifest.get("static_int8_interaction")
    if not isinstance(interaction, dict) or interaction.get("status") != "measured":
        interaction = static_int8_interaction(manifest["variants"])
    (tables / "static_int8_interaction.json").write_text(
        json.dumps(interaction, indent=2) + "\n",
        encoding="utf-8",
    )

    mechanism = node_mechanism_summary(manifest)
    if mechanism is not None:
        (tables / "node_mechanism.json").write_text(json.dumps(mechanism, indent=2) + "\n", encoding="utf-8")
        latex = [
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            r"Profile & Total ($\mu$s) & Q/DQ share & QLinear share & Other share \\",
            r"\midrule",
        ]
        for row in mechanism["rows"]:
            latex.append(
                f"{row['profile'].replace('_', r'\_')} & "
                f"{row['total_us']:.0f} & {row['qdq_share'] * 100:.1f}\\% & "
                f"{row['quantized_kernel_share'] * 100:.1f}\\% & {row['other_share'] * 100:.1f}\\% \\\\"
            )
        latex.extend([r"\bottomrule", r"\end{tabular}", ""])
        (tables / "node_mechanism.tex").write_text("\n".join(latex), encoding="utf-8")

        deltas = mechanism.get("operator_deltas") or []
        if deltas:
            with (tables / "node_operator_deltas.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["op_name", "group", "left_us", "right_us", "delta_us"],
                )
                writer.writeheader()
                writer.writerows(deltas)

    comparison_path = ROOT / "results/calibration-comparison-minmax-percentile.json"
    if comparison_path.is_file():
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        (tables / "calibration_comparison.json").write_text(
            json.dumps(comparison, indent=2) + "\n",
            encoding="utf-8",
        )
        focused = [
            row for row in comparison.get("variants", [])
            if row["variant"].startswith("int8_")
        ]
        if focused:
            latex = [
                r"\begin{tabular}{lrrr}",
                r"\toprule",
                r"Variant & MinMax AUROC & Percentile AUROC & $\Delta$ AUROC \\",
                r"\midrule",
            ]
            for row in focused:
                latex.append(
                    f"{row['variant'].replace('_', r'\_')} & "
                    f"{row['reference_auroc']:.4f} & {row['candidate_auroc']:.4f} & "
                    f"{row['delta_auroc']:+.5f} \\\\"
                )
            latex.extend([r"\bottomrule", r"\end{tabular}", ""])
            (tables / "calibration_comparison.tex").write_text("\n".join(latex), encoding="utf-8")


def _annotate_offsets() -> dict[str, tuple[int, int]]:
    return {
        "fp32": (-32, -14),
        "fp32_compiled": (-10, 10),
        "fp32_default": (-10, -18),
        "int8_dynamic": (8, -15),
        "int8_static": (8, -15),
        "int8_dynamic_default": (8, 10),
        "int8_static_default": (8, 8),
        "int8_dynamic_compiled": (8, 10),
        "int8_static_compiled": (8, 8),
        "int8_static_compiled_default": (8, -12),
    }


def plot_accuracy_latency(rows: list[dict], frontiers: dict[str, list[str]]) -> None:
    frontier = set(frontiers.get("batch1_latency_vs_f1", []))
    fig, axis = plt.subplots(figsize=(7.6, 4.8))
    for row in rows:
        on_frontier = row["variant"] in frontier
        axis.scatter(
            row["latency_ms"],
            row["f1"],
            s=row["size_kib"] / 64,
            alpha=0.95 if on_frontier else 0.55,
            edgecolors="#b00020" if on_frontier else "none",
            linewidths=1.5 if on_frontier else 0.0,
        )
    offsets = _annotate_offsets()
    for row in rows:
        axis.annotate(
            row["variant"],
            (row["latency_ms"], row["f1"]),
            xytext=offsets.get(row["variant"], (8, 8)),
            textcoords="offset points",
            fontsize=7 if len(rows) > 8 else 8,
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )
    axis.set_xlabel("Batch-1 median latency (ms)")
    axis.set_ylabel("Macro F1")
    axis.set_title("Accuracy–latency trade-off (red edge = Pareto frontier)")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT / "figures/accuracy_latency.png", dpi=520)
    plt.close(fig)


def plot_size_throughput(rows: list[dict], frontiers: dict[str, list[str]]) -> None:
    frontier = set(frontiers.get("size_throughput_f1", frontiers.get("size_latency_f1", [])))
    fig, axis = plt.subplots(figsize=(7.6, 4.8))
    for row in rows:
        on_frontier = row["variant"] in frontier
        axis.scatter(
            row["size_kib"],
            row["throughput"],
            s=55,
            alpha=0.95 if on_frontier else 0.55,
            edgecolors="#b00020" if on_frontier else "none",
            linewidths=1.5 if on_frontier else 0.0,
        )
    offsets = _annotate_offsets()
    for row in rows:
        axis.annotate(
            row["variant"],
            (row["size_kib"], row["throughput"]),
            xytext=offsets.get(row["variant"], (8, 8)),
            textcoords="offset points",
            fontsize=7 if len(rows) > 8 else 8,
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )
    axis.set_xlabel("Serialized artifact size (KiB)")
    axis.set_ylabel("Throughput (samples/s)")
    axis.set_title("Model size–throughput trade-off")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT / "figures/size_throughput.png", dpi=220)
    plt.close(fig)


def plot_pareto_size_latency(rows: list[dict], frontier_names: list[str]) -> None:
    by_name = {row["variant"]: row for row in rows}
    frontier_rows = [by_name[name] for name in frontier_names if name in by_name]
    fig, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.scatter(
        [row["size_kib"] for row in rows],
        [row["latency_ms"] for row in rows],
        s=40,
        alpha=0.35,
        color="#888888",
        label="dominated",
    )
    axis.scatter(
        [row["size_kib"] for row in frontier_rows],
        [row["latency_ms"] for row in frontier_rows],
        s=70,
        color="#b00020",
        label="Pareto frontier",
    )
    ordered = sorted(frontier_rows, key=lambda row: row["size_kib"])
    if len(ordered) >= 2:
        axis.plot(
            [row["size_kib"] for row in ordered],
            [row["latency_ms"] for row in ordered],
            color="#b00020",
            linewidth=1.2,
            alpha=0.7,
        )
    for row in frontier_rows:
        axis.annotate(row["variant"], (row["size_kib"], row["latency_ms"]), fontsize=8)
    axis.set_xlabel("Artifact size (KiB)")
    axis.set_ylabel("Batch-1 median latency (ms)")
    axis.set_title("Size–latency Pareto frontier")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "figures/pareto_size_latency.png", dpi=220)
    plt.close(fig)


def plot_batch_scaling(manifest: dict) -> None:
    scaling = batch_scaling_rows(manifest)
    if not scaling:
        return
    by_variant: dict[str, list[dict]] = {}
    for row in scaling:
        by_variant.setdefault(row["name"], []).append(row)
    fig, axis = plt.subplots(figsize=(7.4, 4.6))
    for name, points in sorted(by_variant.items()):
        ordered = sorted(points, key=lambda item: item["batch_size"])
        axis.plot(
            [item["batch_size"] for item in ordered],
            [item["latency_seconds_p50"] * 1000 for item in ordered],
            marker="o",
            linewidth=1.2,
            label=name,
        )
    axis.set_xlabel("Batch size")
    axis.set_ylabel("Median latency (ms)")
    axis.set_title("Batch scaling (p50 latency)")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(OUTPUT / "figures/batch_scaling.png", dpi=220)
    plt.close(fig)


def plot_thread_sweep(manifest: dict) -> None:
    series: dict[str, list[tuple[int, float]]] = {}
    for variant in manifest["variants"]:
        sweep = variant["benchmark"].get("thread_sweep")
        if not sweep:
            continue
        points = []
        for point in sorted(sweep, key=lambda item: item["threads"]):
            points.append((
                point["threads"],
                point["batches"]["1"]["latency_seconds_p50"] * 1000,
            ))
        series[variant["name"]] = points
    if not series:
        return
    fig, axis = plt.subplots(figsize=(7.0, 4.4))
    for name, points in sorted(series.items()):
        axis.plot([item[0] for item in points], [item[1] for item in points], marker="o", label=name)
    axis.set_xlabel("intra_op_num_threads")
    axis.set_ylabel("Batch-1 median latency (ms)")
    axis.set_title("Thread-count sweep")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(OUTPUT / "figures/thread_sweep.png", dpi=220)
    plt.close(fig)


def plot_pipeline(condition_count: int) -> None:
    fig, axis = plt.subplots(figsize=(11, 7.2))
    axis.set_xlim(0, 11)
    axis.set_ylim(0, 8)
    axis.axis("off")
    checkpoint = (4.15, 6.85, 2.7, 0.7)
    axis.add_patch(FancyBboxPatch(
        checkpoint[:2],
        checkpoint[2],
        checkpoint[3],
        boxstyle="round,pad=0.04",
        facecolor="#dcecf8",
        edgecolor="#24527a",
        linewidth=1.5,
    ))
    axis.text(5.5, 7.2, "Common FP32 ONNX checkpoint", ha="center", va="center", fontsize=11, weight="bold")

    variants = [
        (1.0, 5.0, "FP32 baseline", "original graph"),
        (5.0, 5.0, "FP32 compiled", "ORT extended"),
        (1.0, 3.5, "Dynamic INT8", "quantize"),
        (5.0, 3.5, "Static INT8", "calibrate + QDQ"),
        (1.0, 2.0, "Compiled dynamic INT8", "quantize → ORT"),
        (5.0, 2.0, "Compiled static INT8", "calibrate + QDQ → ORT"),
    ]
    for x, y, label, detail in variants:
        axis.add_patch(FancyBboxPatch(
            (x, y),
            4.0,
            0.78,
            boxstyle="round,pad=0.04",
            facecolor="#f4f4f4" if "compiled" not in label.lower() else "#e8f3e8",
            edgecolor="#4b4b4b",
            linewidth=1.1,
        ))
        axis.text(x + 2.0, y + 0.48, label, ha="center", va="center", fontsize=9.5, weight="bold")
        axis.text(x + 2.0, y + 0.2, detail, ha="center", va="center", fontsize=8)
        axis.add_patch(FancyArrowPatch(
            (5.5, checkpoint[1]),
            (x + 2.0, y + 0.8),
            arrowstyle="->",
            mutation_scale=12,
            linewidth=1.0,
            color="#6a6a6a",
            connectionstyle="arc3,rad=0.05" if x < 3 else "arc3,rad=-0.05",
        ))

    evaluation = (1.0, 0.45, 9.0, 0.85)
    axis.add_patch(FancyBboxPatch(
        evaluation[:2],
        evaluation[2],
        evaluation[3],
        boxstyle="round,pad=0.04",
        facecolor="#fff3d6",
        edgecolor="#9a6b00",
        linewidth=1.5,
    ))
    axis.text(
        5.5,
        0.93,
        "Shared evaluation: AUROC/F1 (+bootstrap CIs) • init • RSS • batch scaling • "
        "thread sweep • Pareto frontiers • ORT node profiler",
        ha="center",
        va="center",
        fontsize=9.0,
        weight="bold",
    )
    for x, y, _, _ in variants:
        axis.add_patch(FancyArrowPatch(
            (x + 2.0, y),
            (5.5, evaluation[1] + evaluation[3]),
            arrowstyle="->",
            mutation_scale=10,
            linewidth=0.8,
            color="#8a8a8a",
            connectionstyle="arc3,rad=0.03" if x < 3 else "arc3,rad=-0.03",
        ))
    if condition_count >= 5:
        axis.text(
            5.5,
            1.55,
            "Condition 5 reloads key artifacts at ORT's default graph optimization level (ORT_ENABLE_ALL)",
            ha="center",
            va="center",
            fontsize=9,
            style="italic",
        )
    else:
        axis.text(
            5.5,
            1.55,
            "Six deployment artifacts evaluated under one fixed protocol",
            ha="center",
            va="center",
            fontsize=9,
            style="italic",
        )
    axis.set_title(
        f"{condition_count}-condition deployment study with shared evaluation protocol",
        fontsize=13,
        pad=12,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT / "figures/optimization_pipeline.png", dpi=260)
    plt.close(fig)


def main() -> None:
    manifest = load_manifest()
    rows = load_rows(manifest)
    frontiers = displayed_frontiers(manifest)
    (OUTPUT / "figures").mkdir(parents=True, exist_ok=True)
    write_tables(rows, manifest, frontiers)
    write_supplementary_tables(manifest)
    plot_accuracy_latency(rows, frontiers)
    plot_size_throughput(rows, frontiers)
    plot_pareto_size_latency(rows, frontiers.get("size_vs_batch1_latency", manifest.get("pareto_frontier", [])))
    plot_batch_scaling(manifest)
    plot_thread_sweep(manifest)
    plot_pipeline(len(manifest.get("experiment", {}).get("conditions", [])) or 4)


if __name__ == "__main__":
    main()
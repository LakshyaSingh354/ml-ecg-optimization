"""Comparisons between static-quantization calibration methods."""

from typing import Any


def _variant_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {variant["name"]: variant for variant in manifest["variants"]}


def _interval(metrics: dict[str, Any], key: str) -> list[float] | None:
    value = metrics.get(key)
    if value is None and isinstance(metrics.get("bootstrap"), dict):
        value = metrics["bootstrap"].get(key)
    if value is None:
        return None
    return [float(value[0]), float(value[1])]


def compare_calibration_manifests(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Compare a candidate calibration method against a reference manifest, variant by variant."""
    reference_experiment = reference["experiment"]
    candidate_experiment = candidate["experiment"]
    reference_variants = _variant_map(reference)
    candidate_variants = _variant_map(candidate)
    rows = []
    for name in reference_variants:
        if name not in candidate_variants:
            continue
        reference_metrics = reference_variants[name]["metrics"]
        candidate_metrics = candidate_variants[name]["metrics"]
        rows.append({
            "variant": name,
            "reference_auroc": float(reference_metrics["macro_auroc"]),
            "candidate_auroc": float(candidate_metrics["macro_auroc"]),
            "delta_auroc": float(candidate_metrics["macro_auroc"] - reference_metrics["macro_auroc"]),
            "reference_auroc_ci95": _interval(reference_metrics, "macro_auroc_ci95"),
            "candidate_auroc_ci95": _interval(candidate_metrics, "macro_auroc_ci95"),
            "reference_f1": float(reference_metrics["macro_f1"]),
            "candidate_f1": float(candidate_metrics["macro_f1"]),
            "delta_f1": float(candidate_metrics["macro_f1"] - reference_metrics["macro_f1"]),
            "reference_f1_ci95": _interval(reference_metrics, "macro_f1_ci95"),
            "candidate_f1_ci95": _interval(candidate_metrics, "macro_f1_ci95"),
        })
    return {
        "reference_method": reference_experiment.get("calibration_method"),
        "candidate_method": candidate_experiment.get("calibration_method"),
        "reference_calibration_samples": reference_experiment.get("calibration_samples"),
        "candidate_calibration_samples": candidate_experiment.get("calibration_samples"),
        "same_calibration_records": (
            reference_experiment.get("calibration_record_ids_sha256")
            == candidate_experiment.get("calibration_record_ids_sha256")
        ),
        "calibration_record_ids_sha256": reference_experiment.get("calibration_record_ids_sha256"),
        "same_test_records": (
            reference_experiment.get("selected_test_records")
            == candidate_experiment.get("selected_test_records")
        ),
        "selected_test_records": reference_experiment.get("selected_test_records"),
        "weight_quantization_matches": (
            reference_experiment.get("quantization_per_channel")
            == candidate_experiment.get("quantization_per_channel")
        ),
        "variants": rows,
    }

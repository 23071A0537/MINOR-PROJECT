from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _format_driver_list(mean_shap: dict[str, float], *, positive: bool, top_k: int = 3) -> list[dict[str, Any]]:
    ordered = sorted(mean_shap.items(), key=lambda kv: kv[1], reverse=positive)
    picked: list[dict[str, Any]] = []

    for feature, value in ordered:
        if positive and value <= 0:
            continue
        if (not positive) and value >= 0:
            continue
        picked.append({
            "feature": feature,
            "mean_shap": round(float(value), 6),
        })
        if len(picked) >= top_k:
            break

    return picked


def _confidence_note(pred_count: int, mean_conf: float | None) -> str:
    if pred_count == 0:
        return (
            "No direct prediction-confidence estimate is available for this class "
            "because it was never the top predicted label in the provided run."
        )

    if mean_conf is None:
        return "Prediction confidence is unavailable for this class due to missing confidence values."

    if mean_conf >= 0.85:
        return "High confidence regime for this class in observed predictions (mean confidence >= 0.85)."
    if mean_conf >= 0.60:
        return "Moderate confidence regime for this class in observed predictions (0.60 <= mean confidence < 0.85)."
    return "Low confidence regime for this class in observed predictions (mean confidence < 0.60)."


def _class_interpretation(
    class_name: str,
    pred_count: int,
    shap_count: int,
    top_pos: list[dict[str, Any]],
    top_neg: list[dict[str, Any]],
) -> str:
    if shap_count == 0:
        if pred_count == 0:
            return (
                f"No class-conditional SHAP interpretation could be computed for {class_name} because "
                "the class did not occur as the predicted label in explained samples."
            )
        return (
            f"{class_name} has predicted samples, but SHAP vectors for this class are not available in the "
            "current explanation subset."
        )

    pos_str = ", ".join([f"{x['feature']} ({x['mean_shap']:+.4f})" for x in top_pos]) or "none"
    neg_str = ", ".join([f"{x['feature']} ({x['mean_shap']:+.4f})" for x in top_neg]) or "none"

    return (
        f"Classified as {class_name} when positive SHAP contributions such as {pos_str} outweigh opposing "
        f"signals such as {neg_str} in the latent feature space."
    )


def generate_classwise_json(input_path: Path, output_path: Path) -> dict[str, Any]:
    with input_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    results = payload.get("results", [])
    all_predictions: list[dict[str, Any]] = []
    all_explanations: list[dict[str, Any]] = []

    for result in results:
        source_file = result.get("file")
        for pred in result.get("predictions", []):
            item = dict(pred)
            item["file"] = source_file
            all_predictions.append(item)
        for exp in result.get("shap_explanations", []):
            item = dict(exp)
            item["file"] = source_file
            all_explanations.append(item)

    class_order: list[str] = []
    if all_predictions:
        first_probs = all_predictions[0].get("probabilities", {})
        if isinstance(first_probs, dict):
            class_order = list(first_probs.keys())

    if not class_order:
        observed = set()
        for pred in all_predictions:
            probs = pred.get("probabilities", {})
            if isinstance(probs, dict):
                observed.update(probs.keys())
        class_order = sorted(observed)

    probs_by_class: dict[str, list[float]] = defaultdict(list)
    pred_conf_by_class: dict[str, list[float]] = defaultdict(list)
    shap_rows_by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for pred in all_predictions:
        pred_label = pred.get("pred_label")
        conf = pred.get("confidence")
        probs = pred.get("probabilities", {})

        if isinstance(pred_label, str) and isinstance(conf, (int, float)):
            pred_conf_by_class[pred_label].append(float(conf))

        if isinstance(probs, dict):
            for cls, val in probs.items():
                if isinstance(val, (int, float)):
                    probs_by_class[cls].append(float(val))

    for exp in all_explanations:
        pred_label = exp.get("pred_label")
        if isinstance(pred_label, str):
            shap_rows_by_class[pred_label].append(exp)

    class_entries: list[dict[str, Any]] = []

    for cls in class_order:
        class_probs = probs_by_class.get(cls, [])
        class_conf = pred_conf_by_class.get(cls, [])
        class_shap_rows = shap_rows_by_class.get(cls, [])

        feature_values: dict[str, list[float]] = defaultdict(list)
        for row in class_shap_rows:
            shap_values = row.get("shap_values", {})
            if not isinstance(shap_values, dict):
                continue
            for feature, value in shap_values.items():
                if isinstance(value, (int, float)):
                    feature_values[feature].append(float(value))

        mean_shap = {feature: _safe_mean(vals) for feature, vals in feature_values.items()}
        mean_shap = {feature: val for feature, val in mean_shap.items() if val is not None}

        top_positive = _format_driver_list(mean_shap, positive=True, top_k=3)
        top_negative = _format_driver_list(mean_shap, positive=False, top_k=3)

        mean_pred_conf = _safe_mean(class_conf)
        mean_prob = _safe_mean(class_probs)

        limitations: list[str] = []
        if len(class_shap_rows) == 0:
            limitations.append(
                "No SHAP vectors were available for this class in the explained sample subset."
            )
        if len(class_shap_rows) < len(class_conf):
            limitations.append(
                "Only a subset of predicted samples includes SHAP vectors because live inference limits explained rows."
            )
        if len(class_conf) < 3:
            limitations.append(
                "Interpretation stability is limited by low support (fewer than 3 predicted samples)."
            )

        class_entries.append(
            {
                "class_name": cls,
                "support": {
                    "predicted_count": len(class_conf),
                    "probability_observations": len(class_probs),
                    "shap_samples_available": len(class_shap_rows),
                },
                "confidence": {
                    "mean_predicted_confidence": None if mean_pred_conf is None else round(mean_pred_conf, 6),
                    "mean_probability_across_all_samples": None if mean_prob is None else round(mean_prob, 6),
                },
                "top_positive_drivers": top_positive,
                "top_negative_drivers": top_negative,
                "technical_interpretation": _class_interpretation(
                    cls,
                    len(class_conf),
                    len(class_shap_rows),
                    top_positive,
                    top_negative,
                ),
                "confidence_note": _confidence_note(len(class_conf), mean_pred_conf),
                "limitation_notes": limitations,
            }
        )

    global_limitations = [
        "The source file provides SHAP values aligned to the predicted class per explained sample, not a full class-by-class SHAP tensor for every sample.",
        "Class-wise SHAP interpretation for classes with zero predicted support is necessarily limited.",
        "Results are specific to the current live inference run and should be re-generated for new data distributions.",
    ]

    output = {
        "run_metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_file": str(input_path),
            "source_created_at": payload.get("created_at"),
            "weights": payload.get("weights", {}),
            "total_predictions": len(all_predictions),
            "total_shap_explanations": len(all_explanations),
            "class_order": class_order,
        },
        "class_wise_explanations": class_entries,
        "confidence_and_limits": {
            "global_limitations": global_limitations,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate class-wise SHAP interpretation JSON with confidence and limitation notes."
    )
    parser.add_argument(
        "--input",
        default="artifacts/inference/live_predictions.json",
        help="Path to live predictions JSON with SHAP explanations.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/explainability/class_wise_shap_interpretation.json",
        help="Path for generated class-wise interpretation JSON.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_path}")

    output = generate_classwise_json(input_path, output_path)
    print(
        "Generated class-wise SHAP JSON:",
        output_path,
        "| classes=", len(output.get("class_wise_explanations", [])),
    )


if __name__ == "__main__":
    main()

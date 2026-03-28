from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


def _as_dataframe(
    data: pd.DataFrame | np.ndarray,
    feature_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()

    array = np.asarray(data, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"Expected 2D background data for LIME, got shape {array.shape}")

    if feature_names is None:
        feature_names = [f"z{i}" for i in range(array.shape[1])]

    return pd.DataFrame(array, columns=list(feature_names))


def _as_1d_row(sample_row: pd.Series | np.ndarray | Sequence[float]) -> np.ndarray:
    if isinstance(sample_row, pd.Series):
        row = sample_row.to_numpy(dtype=float)
    else:
        row = np.asarray(sample_row, dtype=float)

    if row.ndim == 1:
        return row
    if row.ndim == 2 and 1 in row.shape:
        return row.reshape(-1)
    raise ValueError(f"Expected 1D sample row for LIME, got shape {row.shape}")


def _weight_sign_label(weight: float) -> str:
    if weight > 0:
        return "supports_predicted_class"
    if weight < 0:
        return "opposes_predicted_class"
    return "neutral"


def _because_statement(pred_label: str, contributions: Sequence[Dict[str, Any]], top_n: int = 3) -> str:
    positives = [c for c in contributions if c["weight"] > 0][:top_n]
    negatives = [c for c in contributions if c["weight"] < 0][:top_n]

    if positives:
        pos_text = ", ".join(
            f"{c['feature']} ({c['weight']:+.4f})" for c in positives
        )
    else:
        pos_text = "no strong positive local drivers"

    if negatives:
        neg_text = ", ".join(
            f"{c['feature']} ({c['weight']:+.4f})" for c in negatives
        )
    else:
        neg_text = "minimal opposing local drivers"

    return (
        f"Classified as {pred_label} because local LIME drivers {pos_text} "
        f"outweighed opposing signals {neg_text}."
    )


def build_lime_explainer(
    background_data: pd.DataFrame | np.ndarray,
    class_names: Sequence[str],
    feature_names: Optional[Sequence[str]] = None,
    random_state: int = 42,
    discretize_continuous: bool = True,
):
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError as exc:
        raise ImportError(
            "LIME dependency is missing. Install it with 'pip install lime'."
        ) from exc

    background_df = _as_dataframe(background_data, feature_names)
    feature_list = list(feature_names) if feature_names is not None else list(background_df.columns)

    return LimeTabularExplainer(
        training_data=background_df.to_numpy(dtype=float),
        feature_names=feature_list,
        class_names=[str(c) for c in class_names],
        mode="classification",
        discretize_continuous=bool(discretize_continuous),
        random_state=int(random_state),
    )


def explain_lime_instance(
    explainer,
    predict_proba_fn: Callable[[np.ndarray], np.ndarray],
    sample_row: pd.Series | np.ndarray | Sequence[float],
    pred_class: int,
    class_names: Sequence[str],
    feature_names: Sequence[str],
    num_features: int = 10,
    num_samples: int = 2000,
) -> Dict[str, Any]:
    row = _as_1d_row(sample_row)
    feature_list = list(feature_names)
    if len(feature_list) != row.shape[0]:
        raise ValueError(
            f"Feature count mismatch for LIME row: row has {row.shape[0]}, "
            f"feature_names has {len(feature_list)}"
        )

    class_idx = int(pred_class)
    capped_num_features = max(1, min(int(num_features), len(feature_list)))

    explanation = explainer.explain_instance(
        data_row=row,
        predict_fn=predict_proba_fn,
        labels=[class_idx],
        num_features=capped_num_features,
        num_samples=int(num_samples),
    )

    local_pairs = explanation.local_exp.get(class_idx, [])
    local_pairs = sorted(local_pairs, key=lambda item: abs(item[1]), reverse=True)

    contributions: List[Dict[str, Any]] = []
    for feat_idx, weight in local_pairs:
        feature_index = int(feat_idx)
        feature_name = feature_list[feature_index]
        weight_val = float(weight)
        contributions.append(
            {
                "feature": feature_name,
                "feature_index": feature_index,
                "feature_value": float(row[feature_index]),
                "weight": weight_val,
                "direction": _weight_sign_label(weight_val),
            }
        )

    rule_contributions = [
        {"rule": str(rule), "weight": float(weight)}
        for rule, weight in explanation.as_list(label=class_idx)
    ]

    if 0 <= class_idx < len(class_names):
        pred_label = str(class_names[class_idx])
    else:
        pred_label = f"Class_{class_idx}"

    intercept_value: Optional[float] = None
    if isinstance(explanation.intercept, (list, tuple, np.ndarray)):
        if class_idx < len(explanation.intercept):
            intercept_value = float(explanation.intercept[class_idx])

    return {
        "predicted_class_index": class_idx,
        "predicted_label": pred_label,
        "intercept": intercept_value,
        "num_features_used": int(len(contributions)),
        "feature_contributions": contributions,
        "rule_contributions": rule_contributions,
        "because": _because_statement(pred_label, contributions),
    }


def compute_shap_lime_consistency(
    lime_contributions: Sequence[Dict[str, Any]],
    shap_values: Optional[Dict[str, float]],
    top_k: int = 5,
) -> Dict[str, Any]:
    top_k = max(1, int(top_k))
    shap_values = shap_values or {}

    if not shap_values:
        return {
            "status": "shap_missing",
            "summary": "SHAP values are unavailable for this row, so consistency cannot be computed.",
            "top_k": top_k,
            "jaccard_overlap": 0.0,
            "sign_agreement_ratio": 0.0,
            "overlap_features": [],
            "sign_divergence_features": [],
        }

    lime_rank = sorted(
        [
            (str(item.get("feature", "")), float(item.get("weight", 0.0)))
            for item in lime_contributions
            if item.get("feature")
        ],
        key=lambda pair: abs(pair[1]),
        reverse=True,
    )[:top_k]

    shap_rank = sorted(
        [(str(feature), float(value)) for feature, value in shap_values.items()],
        key=lambda pair: abs(pair[1]),
        reverse=True,
    )[:top_k]

    lime_dict = {feature: weight for feature, weight in lime_rank}
    shap_dict = {feature: value for feature, value in shap_rank}

    lime_set = set(lime_dict)
    shap_set = set(shap_dict)
    overlap = sorted(lime_set & shap_set)
    union = lime_set | shap_set

    jaccard_overlap = float(len(overlap) / len(union)) if union else 0.0

    sign_matches = 0
    sign_divergence_features: List[Dict[str, Any]] = []
    for feature in overlap:
        lime_weight = float(lime_dict[feature])
        shap_value = float(shap_dict[feature])
        if np.sign(lime_weight) == np.sign(shap_value) or lime_weight == 0.0 or shap_value == 0.0:
            sign_matches += 1
        else:
            sign_divergence_features.append(
                {
                    "feature": feature,
                    "lime_weight": lime_weight,
                    "shap_value": shap_value,
                }
            )

    sign_agreement_ratio = float(sign_matches / len(overlap)) if overlap else 0.0

    if not overlap:
        status = "divergence"
        summary = "No overlap between top SHAP and top LIME drivers for this row."
    elif sign_divergence_features:
        status = "partial_agreement"
        summary = "Top SHAP and LIME drivers overlap, but at least one shared feature has opposite sign."
    elif jaccard_overlap >= 0.4 and sign_agreement_ratio >= 0.6:
        status = "agreement"
        summary = "Top SHAP and LIME drivers are largely aligned for this row."
    else:
        status = "partial_agreement"
        summary = "SHAP and LIME are partially aligned but overlap is modest."

    return {
        "status": status,
        "summary": summary,
        "top_k": top_k,
        "jaccard_overlap": round(jaccard_overlap, 4),
        "sign_agreement_ratio": round(sign_agreement_ratio, 4),
        "overlap_features": overlap,
        "sign_divergence_features": sign_divergence_features,
        "top_lime": [
            {"feature": feature, "weight": weight} for feature, weight in lime_rank
        ],
        "top_shap": [
            {"feature": feature, "value": value} for feature, value in shap_rank
        ],
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
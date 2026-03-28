import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


CLASS_NAMES = ["NORMALL", "DoSD", "PROBE", "EXPLOIT", "MALWARE"]


def load_vqc_proba(vqc_path: Path) -> pd.DataFrame:
    vqc = pd.read_parquet(vqc_path)
    missing = [c for c in CLASS_NAMES if c not in vqc.columns]
    if missing:
        raise ValueError(f"VQC proba missing columns: {missing}")
    return vqc[CLASS_NAMES]


def load_rf_proba(rf_path: Path) -> pd.DataFrame:
    rf = pd.read_parquet(rf_path)
    renamed = {f"p_{name}": name for name in CLASS_NAMES}
    missing = [c for c in renamed.keys() if c not in rf.columns]
    if missing:
        raise ValueError(f"RF proba missing columns: {missing}")
    rf = rf.rename(columns=renamed)
    return rf[CLASS_NAMES]


def load_xgb_proba(model_path: Path, z_test_path: Path) -> pd.DataFrame:
    model = joblib.load(model_path)
    z_test = pd.read_parquet(z_test_path)
    proba = model.predict_proba(z_test)
    class_order = list(model.classes_)
    if class_order != [0, 1, 2, 3, 4]:
        raise ValueError(f"Unexpected XGBoost class order: {class_order}")
    return pd.DataFrame(proba, columns=CLASS_NAMES)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build hybrid ensemble and save metrics.")
    parser.add_argument(
        "--output",
        default="hybrid layer output.json",
        help="Output JSON file name (default: hybrid layer output.json)",
    )
    parser.add_argument("--w_vqc", type=float, default=0.1)
    parser.add_argument("--w_xgb", type=float, default=0.55)
    parser.add_argument("--w_rf", type=float, default=0.35)
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[3]
    vqc_path = base / "vqc_ensemble_v6" / "winner_test_proba.parquet"
    rf_path = base / "random_forest_output" / "rf_test_proba.parquet"
    xgb_model_path = base / "xgboost_output" / "xgboost_model.pkl"
    z_test_path = base / "VAE" / "vae_a_output_16" / "vae_a_z_test.parquet"
    y_test_path = base / "PreProcessing" / "stage_2_with_zero_v2" / "stage2_y_test.parquet"

    vqc = load_vqc_proba(vqc_path)
    rf = load_rf_proba(rf_path)
    xgb = load_xgb_proba(xgb_model_path, z_test_path)
    y_test = pd.read_parquet(y_test_path)["y"].to_numpy()

    if not (len(vqc) == len(rf) == len(xgb) == len(y_test)):
        raise ValueError("Length mismatch across VQC/RF/XGB/y_test")

    weights = np.array([args.w_vqc, args.w_xgb, args.w_rf], dtype=float)
    if not np.isclose(weights.sum(), 1.0):
        weights = weights / weights.sum()

    hybrid_proba = (
        weights[0] * vqc.to_numpy()
        + weights[1] * xgb.to_numpy()
        + weights[2] * rf.to_numpy()
    )
    y_pred = hybrid_proba.argmax(axis=1)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "f1_micro": f1_score(y_test, y_pred, average="micro"),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
    }

    output = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "dataset": "PreProcessing/stage_2_with_zero_v2",
        "weights": {
            "vqc": float(weights[0]),
            "xgboost": float(weights[1]),
            "random_forest": float(weights[2]),
        },
        "inputs": {
            "vqc_test_proba": str(vqc_path),
            "rf_test_proba": str(rf_path),
            "xgb_model": str(xgb_model_path),
            "vae_z_test": str(z_test_path),
            "y_test": str(y_test_path),
        },
        "metrics": metrics,
    }

    output_path = base / args.output
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Saved hybrid metrics to {output_path}")


if __name__ == "__main__":
    main()

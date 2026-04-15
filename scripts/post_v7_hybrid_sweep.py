import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support


CLASS_NAMES = ["NORMALL", "DoSD", "PROBE", "EXPLOIT", "MALWARE"]
MALWARE_IDX = 4
PROBE_IDX = 2


def load_y_test(y_path: Path) -> np.ndarray:
    y_df = pd.read_parquet(y_path)
    if isinstance(y_df, pd.Series):
        y = y_df.to_numpy()
    elif "y" in y_df.columns:
        y = y_df["y"].to_numpy()
    else:
        y = y_df.iloc[:, 0].to_numpy()
    return y.astype(int).reshape(-1)


def load_v7_vqc_proba(v7_dir: Path) -> np.ndarray:
    vqc_a_path = v7_dir / "vqc_a" / "test_proba.npy"
    vqc_b_path = v7_dir / "vqc_b" / "test_proba.npy"

    if not vqc_a_path.exists() or not vqc_b_path.exists():
        missing = []
        if not vqc_a_path.exists():
            missing.append(str(vqc_a_path))
        if not vqc_b_path.exists():
            missing.append(str(vqc_b_path))
        raise FileNotFoundError("Missing VQC outputs: " + ", ".join(missing))

    vqc_a = np.load(vqc_a_path)
    vqc_b = np.load(vqc_b_path)
    if vqc_a.shape != vqc_b.shape:
        raise ValueError(f"VQC-A/VQC-B shape mismatch: {vqc_a.shape} vs {vqc_b.shape}")
    if vqc_a.shape[1] != 5:
        raise ValueError(f"Unexpected VQC proba shape: {vqc_a.shape}")
    return 0.5 * (vqc_a + vqc_b)


def load_rf_proba(rf_path: Path) -> np.ndarray:
    rf = pd.read_parquet(rf_path)
    renamed = {f"p_{name}": name for name in CLASS_NAMES}
    for col in renamed:
        if col not in rf.columns:
            raise ValueError(f"RF proba missing column: {col}")
    rf = rf.rename(columns=renamed)
    return rf[CLASS_NAMES].to_numpy()


def load_xgb_proba(model_path: Path, z_test_path: Path) -> np.ndarray:
    model = joblib.load(model_path)
    z_test = pd.read_parquet(z_test_path)
    proba = model.predict_proba(z_test)

    cls = list(model.classes_)
    if set(cls) != {0, 1, 2, 3, 4}:
        raise ValueError(f"Unexpected XGBoost class labels: {cls}")
    if cls != [0, 1, 2, 3, 4]:
        reorder = [cls.index(i) for i in [0, 1, 2, 3, 4]]
        proba = proba[:, reorder]

    if proba.shape[1] != 5:
        raise ValueError(f"Unexpected XGBoost proba shape: {proba.shape}")
    return proba


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    precision, recall, f1_pc, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3, 4],
        zero_division=0,
    )
    return {
        "f1_macro": macro,
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1_per_class": f1_pc.tolist(),
    }


def row_from_metrics(weights: dict, metrics: dict, floors: dict) -> dict:
    row = {
        "weights": weights,
        "f1_macro": metrics["f1_macro"],
        "malware_precision": float(metrics["precision"][MALWARE_IDX]),
        "malware_f1": float(metrics["f1_per_class"][MALWARE_IDX]),
        "probe_precision": float(metrics["precision"][PROBE_IDX]),
        "probe_f1": float(metrics["f1_per_class"][PROBE_IDX]),
    }
    row["is_feasible"] = (
        row["f1_macro"] >= floors["hybrid_macro_f1_min"]
        and row["malware_precision"] >= floors["malware_precision_min"]
        and row["probe_precision"] >= floors["probe_precision_min"]
    )
    return row


def pick_best(rows: list[dict], feasible_only: bool) -> dict | None:
    candidates = [r for r in rows if r["is_feasible"]] if feasible_only else rows
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda r: (r["malware_f1"], r["f1_macro"], r["malware_precision"]),
        reverse=True,
    )[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-train v7 hybrid constrained sweep")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--v7-dir", default="VQC/vqc_v7_phase1_trained")
    parser.add_argument("--out-report", default="artifacts/reports/v7_hybrid_weight_sweep_constraints.json")
    parser.add_argument("--macro-floor", type=float, default=0.82)
    parser.add_argument("--malware-precision-floor", type=float, default=0.70)
    parser.add_argument("--probe-precision-floor", type=float, default=0.80)
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    floors = {
        "hybrid_macro_f1_min": args.macro_floor,
        "malware_precision_min": args.malware_precision_floor,
        "probe_precision_min": args.probe_precision_floor,
    }

    y_test = load_y_test(root / "PreProcessing" / "stage_2_with_zero_v2" / "stage2_y_test.parquet")
    vqc = load_v7_vqc_proba(root / args.v7_dir)
    xgb = load_xgb_proba(
        root / "xgboost_output" / "xgboost_model.pkl",
        root / "VAE" / "vae_a_output_16" / "vae_a_z_test.parquet",
    )
    rf = load_rf_proba(root / "random_forest_output" / "rf_test_proba.parquet")

    if not (len(vqc) == len(xgb) == len(rf) == len(y_test)):
        raise ValueError(
            f"Length mismatch: vqc={len(vqc)} xgb={len(xgb)} rf={len(rf)} y={len(y_test)}"
        )

    vqc_df = pd.DataFrame(vqc, columns=CLASS_NAMES)
    vqc_df.to_parquet(root / args.v7_dir / "vqc_ensemble_test_proba.parquet", index=False)

    vqc_metrics = evaluate(y_test, np.argmax(vqc, axis=1))

    xgb_share_grid = [0.50, 0.611111, 0.70]
    vqc_weight_grid = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    rows = []
    for w_vqc in vqc_weight_grid:
        rem = 1.0 - w_vqc
        for xgb_share in xgb_share_grid:
            w_xgb = round(rem * xgb_share, 6)
            w_rf = round(rem * (1.0 - xgb_share), 6)
            proba = w_vqc * vqc + w_xgb * xgb + w_rf * rf
            y_pred = np.argmax(proba, axis=1)
            m = evaluate(y_test, y_pred)
            row = row_from_metrics(
                weights={"vqc": float(w_vqc), "xgb": float(w_xgb), "rf": float(w_rf)},
                metrics=m,
                floors=floors,
            )
            rows.append(row)

    focus_rows = [r for r in rows if 0.40 <= r["weights"]["vqc"] <= 0.50]

    best_feasible = pick_best(rows, feasible_only=True)
    best_overall = pick_best(rows, feasible_only=False)
    best_focus_feasible = pick_best(focus_rows, feasible_only=True)
    best_focus_overall = pick_best(focus_rows, feasible_only=False)

    report = {
        "constraints": floors,
        "inputs": {
            "v7_vqc_dir": str(root / args.v7_dir),
            "xgb_model": str(root / "xgboost_output" / "xgboost_model.pkl"),
            "rf_test_proba": str(root / "random_forest_output" / "rf_test_proba.parquet"),
            "y_test": str(root / "PreProcessing" / "stage_2_with_zero_v2" / "stage2_y_test.parquet"),
        },
        "v7_vqc_only": {
            "f1_macro": vqc_metrics["f1_macro"],
            "malware_precision": float(vqc_metrics["precision"][MALWARE_IDX]),
            "malware_f1": float(vqc_metrics["f1_per_class"][MALWARE_IDX]),
            "probe_precision": float(vqc_metrics["precision"][PROBE_IDX]),
            "probe_f1": float(vqc_metrics["f1_per_class"][PROBE_IDX]),
        },
        "grid": {
            "vqc_weight_grid": vqc_weight_grid,
            "xgb_share_grid_of_remaining_weight": xgb_share_grid,
            "total_configs": len(rows),
        },
        "results": rows,
        "best_feasible": best_feasible,
        "best_overall_by_malware_f1": best_overall,
        "focus_range_vqc_0_40_to_0_50": {
            "best_feasible": best_focus_feasible,
            "best_overall": best_focus_overall,
        },
    }

    out_path = root / args.out_report
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Saved: {out_path}")
    if best_feasible is None:
        print("No feasible configuration met all constraints.")
    else:
        print(
            "Best feasible:",
            best_feasible["weights"],
            f"macro={best_feasible['f1_macro']:.4f}",
            f"malware_P={best_feasible['malware_precision']:.4f}",
            f"probe_P={best_feasible['probe_precision']:.4f}",
        )


if __name__ == "__main__":
    main()

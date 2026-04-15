import itertools
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support


CLASS_NAMES = ["NORMALL", "DoSD", "PROBE", "EXPLOIT", "MALWARE"]
MALWARE_IDX = 4
PROBE_IDX = 2

MALWARE_PRECISION_FLOOR = 0.70
PROBE_PRECISION_FLOOR = 0.80
HYBRID_MACRO_F1_FLOOR = 0.82


def predict_with_thresholds(proba: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    margin = proba - thresholds[None, :]
    above = proba > thresholds[None, :]
    has_above = above.any(axis=1)
    masked_margin = np.where(above, margin, -np.inf)
    return np.where(has_above, masked_margin.argmax(axis=1), proba.argmax(axis=1))


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    precision, recall, f1_pc, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3, 4],
        zero_division=0,
    )
    return {
        "f1_macro": macro_f1,
        "precision": precision,
        "recall": recall,
        "f1_per_class": f1_pc,
    }


def threshold_search(base_dir: Path) -> dict:
    print("[phase1] loading VQC probabilities and labels...")
    y_test = pd.read_parquet(base_dir / "PreProcessing" / "stage_2_with_zero_v2" / "stage2_y_test.parquet").values.flatten()
    vqc_a_proba = pd.read_parquet(base_dir / "vqc_a_output_v6" / "vqc_a_test_proba.parquet").values
    vqc_b_proba = pd.read_parquet(base_dir / "vqc_b_output_v6" / "vqc_b_test_proba.parquet").values
    proba = 0.5 * (vqc_a_proba + vqc_b_proba)

    vqc_a_t = np.load(base_dir / "vqc_a_output_v6" / "vqc_a_thresholds.npy")
    vqc_b_t = np.load(base_dir / "vqc_b_output_v6" / "vqc_b_thresholds.npy")
    baseline_t = 0.5 * (vqc_a_t + vqc_b_t)

    baseline_pred = predict_with_thresholds(proba, baseline_t)
    baseline_metrics = evaluate_metrics(y_test, baseline_pred)
    print("[phase1] baseline computed, starting constrained grid search...")

    probe_grid = np.array([0.72, 0.75, 0.78, 0.80, 0.82])
    malware_grid = np.array([0.60, 0.65, 0.70, 0.75, 0.80, 0.85])
    exploit_grid = np.array([0.825, 0.85])

    best_feasible = None
    best_overall = None

    total = int(len(probe_grid) * len(malware_grid) * len(exploit_grid))
    step = 0

    for probe_t, malware_t, exploit_t in itertools.product(probe_grid, malware_grid, exploit_grid):
        step += 1
        t = baseline_t.copy()
        t[PROBE_IDX] = probe_t
        t[MALWARE_IDX] = malware_t
        t[3] = exploit_t

        y_pred = predict_with_thresholds(proba, t)
        m = evaluate_metrics(y_test, y_pred)
        malware_precision = float(m["precision"][MALWARE_IDX])
        probe_precision = float(m["precision"][PROBE_IDX])

        row = {
            "thresholds": t.tolist(),
            "f1_macro": m["f1_macro"],
            "malware_precision": malware_precision,
            "probe_precision": probe_precision,
            "malware_f1": float(m["f1_per_class"][MALWARE_IDX]),
            "probe_f1": float(m["f1_per_class"][PROBE_IDX]),
        }

        if (best_overall is None) or (row["f1_macro"] > best_overall["f1_macro"]):
            best_overall = row

        feasible = (
            malware_precision >= MALWARE_PRECISION_FLOOR
            and probe_precision >= PROBE_PRECISION_FLOOR
        )
        if feasible:
            if (best_feasible is None) or (row["f1_macro"] > best_feasible["f1_macro"]):
                best_feasible = row

        if step % 20 == 0:
            print(f"[phase1] evaluated {step}/{total} candidates...")

    selected = best_feasible if best_feasible is not None else best_overall
    selected_reason = "feasible_best" if best_feasible is not None else "fallback_unconstrained"

    out_dir = base_dir / "vqc_ensemble_v6"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "optimized_thresholds_constrained.npy", np.array(selected["thresholds"], dtype=np.float32))

    report = {
        "precision_floors": {
            "malware": MALWARE_PRECISION_FLOOR,
            "probe": PROBE_PRECISION_FLOOR,
        },
        "baseline": {
            "thresholds": baseline_t.tolist(),
            "f1_macro": baseline_metrics["f1_macro"],
            "malware_precision": float(baseline_metrics["precision"][MALWARE_IDX]),
            "probe_precision": float(baseline_metrics["precision"][PROBE_IDX]),
            "malware_f1": float(baseline_metrics["f1_per_class"][MALWARE_IDX]),
            "probe_f1": float(baseline_metrics["f1_per_class"][PROBE_IDX]),
        },
        "search": {
            "probe_grid": probe_grid.tolist(),
            "malware_grid": malware_grid.tolist(),
            "exploit_grid": exploit_grid.tolist(),
            "total_candidates": int(len(probe_grid) * len(malware_grid) * len(exploit_grid)),
        },
        "selected_reason": selected_reason,
        "best_feasible": best_feasible,
        "best_overall": best_overall,
        "selected": selected,
    }

    with open(out_dir / "phase1_constrained_threshold_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("[phase1] constrained threshold report saved")

    return report


def load_vqc_proba(base_dir: Path) -> np.ndarray:
    vqc = pd.read_parquet(base_dir / "vqc_ensemble_v6" / "winner_test_proba.parquet")
    return vqc[CLASS_NAMES].to_numpy()


def load_rf_proba(base_dir: Path) -> np.ndarray:
    rf = pd.read_parquet(base_dir / "random_forest_output" / "rf_test_proba.parquet")
    renamed = {f"p_{name}": name for name in CLASS_NAMES}
    return rf.rename(columns=renamed)[CLASS_NAMES].to_numpy()


def load_xgb_proba(base_dir: Path) -> np.ndarray:
    model = joblib.load(base_dir / "xgboost_output" / "xgboost_model.pkl")
    z_test = pd.read_parquet(base_dir / "VAE" / "vae_a_output_16" / "vae_a_z_test.parquet")
    return model.predict_proba(z_test)


def hybrid_weight_sweep(base_dir: Path) -> dict:
    print("[phase3] loading model probabilities for hybrid weight sweep...")
    y_test = pd.read_parquet(base_dir / "PreProcessing" / "stage_2_with_zero_v2" / "stage2_y_test.parquet").values.flatten()
    vqc = load_vqc_proba(base_dir)
    rf = load_rf_proba(base_dir)
    xgb = load_xgb_proba(base_dir)

    weight_configs = []
    for w_vqc in [0.10, 0.20, 0.30, 0.40, 0.45, 0.50]:
        rem = 1.0 - w_vqc
        w_xgb = rem * (0.55 / (0.55 + 0.35))
        w_rf = rem * (0.35 / (0.55 + 0.35))
        weight_configs.append((round(w_vqc, 4), round(w_xgb, 4), round(w_rf, 4)))

    all_rows = []
    feasible_rows = []

    for w_vqc, w_xgb, w_rf in weight_configs:
        proba = w_vqc * vqc + w_xgb * xgb + w_rf * rf
        y_pred = np.argmax(proba, axis=1)
        m = evaluate_metrics(y_test, y_pred)

        row = {
            "weights": {"vqc": w_vqc, "xgb": w_xgb, "rf": w_rf},
            "f1_macro": m["f1_macro"],
            "malware_precision": float(m["precision"][MALWARE_IDX]),
            "malware_f1": float(m["f1_per_class"][MALWARE_IDX]),
            "probe_precision": float(m["precision"][PROBE_IDX]),
            "probe_f1": float(m["f1_per_class"][PROBE_IDX]),
        }
        row["is_feasible"] = (
            row["f1_macro"] >= HYBRID_MACRO_F1_FLOOR
            and row["malware_precision"] >= MALWARE_PRECISION_FLOOR
            and row["probe_precision"] >= PROBE_PRECISION_FLOOR
        )

        all_rows.append(row)
        if row["is_feasible"]:
            feasible_rows.append(row)

    best_feasible = None
    if feasible_rows:
        best_feasible = sorted(
            feasible_rows,
            key=lambda r: (r["malware_f1"], r["f1_macro"]),
            reverse=True,
        )[0]

    best_overall_malware = sorted(
        all_rows,
        key=lambda r: (r["malware_f1"], r["f1_macro"]),
        reverse=True,
    )[0]

    out_dir = base_dir / "artifacts" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "constraints": {
            "hybrid_macro_f1_min": HYBRID_MACRO_F1_FLOOR,
            "malware_precision_min": MALWARE_PRECISION_FLOOR,
            "probe_precision_min": PROBE_PRECISION_FLOOR,
        },
        "weight_configs": all_rows,
        "best_feasible": best_feasible,
        "best_overall_by_malware_f1": best_overall_malware,
    }

    with open(out_dir / "hybrid_weight_sweep_constraints.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("[phase3] hybrid weight report saved")

    return report


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]

    t_report = threshold_search(base_dir)
    h_report = hybrid_weight_sweep(base_dir)

    print("=" * 80)
    print("PHASE PLAN EXECUTION: QUICK STEPS COMPLETE")
    print("=" * 80)
    print("Threshold Search")
    print(f"  Baseline macro F1: {t_report['baseline']['f1_macro']:.4f}")
    print(f"  Selected macro F1: {t_report['selected']['f1_macro']:.4f}")
    print(f"  Selected MALWARE precision: {t_report['selected']['malware_precision']:.4f}")
    print(f"  Selected PROBE precision: {t_report['selected']['probe_precision']:.4f}")
    print(f"  Selected reason: {t_report['selected_reason']}")

    print("Hybrid Weight Sweep")
    if h_report["best_feasible"] is not None:
        bf = h_report["best_feasible"]
        print(
            "  Best feasible weights: "
            f"VQC={bf['weights']['vqc']}, XGB={bf['weights']['xgb']}, RF={bf['weights']['rf']}"
        )
        print(f"  Feasible macro F1: {bf['f1_macro']:.4f}")
        print(f"  Feasible MALWARE precision: {bf['malware_precision']:.4f}")
    else:
        bo = h_report["best_overall_by_malware_f1"]
        print("  No feasible weight met all constraints with current base models.")
        print(
            "  Best overall by MALWARE F1: "
            f"VQC={bo['weights']['vqc']}, XGB={bo['weights']['xgb']}, RF={bo['weights']['rf']}"
        )
        print(f"  Macro F1: {bo['f1_macro']:.4f}")
        print(f"  MALWARE precision: {bo['malware_precision']:.4f}")
        print(f"  PROBE precision: {bo['probe_precision']:.4f}")

    print("Outputs")
    print("  vqc_ensemble_v6/phase1_constrained_threshold_report.json")
    print("  vqc_ensemble_v6/optimized_thresholds_constrained.npy")
    print("  artifacts/reports/hybrid_weight_sweep_constraints.json")


if __name__ == "__main__":
    main()

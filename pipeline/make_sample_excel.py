import argparse
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow.dataset as ds


def load_rows_random(parquet_path: Path, indices: np.ndarray) -> pd.DataFrame:
    dataset = ds.dataset(parquet_path, format="parquet")
    table = dataset.take(indices)
    return table.to_pandas()


def stratified_indices(y_values: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    indices = []
    for cls in sorted(set(y_values.tolist())):
        cls_idx = np.where(y_values == cls)[0]
        if len(cls_idx) < per_class:
            raise ValueError(f"Not enough rows for class {cls}: {len(cls_idx)} < {per_class}")
        pick = rng.choice(cls_idx, size=per_class, replace=False)
        indices.append(pick)
    return np.concatenate(indices)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample combined dataset and export Excel.")
    parser.add_argument("--config", default="pipeline/config.json")
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    project_root = config_path.parent.parent

    pipeline_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(pipeline_dir))
    import live_inference as li
    from pipeline_utils import load_json, resolve_path, ensure_exists

    config = load_json(config_path)
    live_cfg = config["live_inference"]

    stage2_dir = project_root / "PreProcessing" / "stage_2_with_zero_v2"
    x_test_path = stage2_dir / "stage2_X_test.parquet"
    y_test_path = stage2_dir / "stage2_y_test.parquet"
    if not x_test_path.exists() or not y_test_path.exists():
        raise FileNotFoundError("Stage 2 test set not found. Expected stage2_X_test.parquet and stage2_y_test.parquet.")

    dataset = ds.dataset(x_test_path, format="parquet")
    total = dataset.count_rows()
    if args.sample_size > total:
        raise ValueError(f"Sample size {args.sample_size} exceeds total rows {total}.")

    y_all = ds.dataset(y_test_path, format="parquet").to_table(columns=["y"]).to_pandas()["y"].to_numpy()
    per_class = args.sample_size // 5
    if per_class * 5 != args.sample_size:
        raise ValueError("Sample size must be divisible by 5 for equal stratification.")
    indices = stratified_indices(y_all, per_class, args.seed)

    df = load_rows_random(x_test_path, indices)
    y_df = load_rows_random(y_test_path, indices)
    gt_labels = y_df["y"].astype(int).tolist()

    artifacts = live_cfg["artifacts"]
    stage2_path = resolve_path(project_root, artifacts["stage2_artefacts"])
    vae_ckpt_path = resolve_path(project_root, artifacts["vae_checkpoint"])
    vae_cfg_path = resolve_path(project_root, artifacts["vae_config"])
    xgb_model_path = resolve_path(project_root, artifacts["xgb_model"])
    rf_model_path = resolve_path(project_root, artifacts["rf_model"])
    vqc_model_path = resolve_path(project_root, artifacts["vqc_model"])

    required = [stage2_path, vae_ckpt_path, vae_cfg_path, xgb_model_path, rf_model_path]
    ensure_exists(required, "inference artefacts")

    artefacts = li.load_stage2_artefacts(stage2_path)
    feature_names = artefacts["feature_names"]
    sentinel_value = float(artefacts["sentinel_value"])
    scaler_min = np.array(artefacts["scaler_col_min"], dtype=float)
    scaler_max = np.array(artefacts["scaler_col_max"], dtype=float)

    x_scaled = li.preprocess_dataframe(df, feature_names, sentinel_value, scaler_min, scaler_max)
    z_angles = li.encode_with_vae(x_scaled, vae_ckpt_path, vae_cfg_path)

    xgb_model = li.load_model(xgb_model_path)
    rf_model = li.load_model(rf_model_path)
    vqc_model = li.load_model(vqc_model_path) if vqc_model_path.exists() else None

    xgb_proba = li.predict_proba(xgb_model, z_angles)
    rf_proba = li.predict_proba(rf_model, z_angles)

    weights = live_cfg["weights"]
    w_vqc = float(weights["vqc"])
    w_xgb = float(weights["xgb"])
    w_rf = float(weights["rf"])

    if vqc_model is None:
        vqc_proba = np.zeros_like(xgb_proba)
        w_vqc = 0.0
    else:
        vqc_proba = li.predict_proba(vqc_model, z_angles)

    weight_sum = w_vqc + w_xgb + w_rf
    if weight_sum <= 0:
        raise ValueError("Invalid weights: sum must be > 0")
    w_v = w_vqc / weight_sum
    w_x = w_xgb / weight_sum
    w_r = w_rf / weight_sum

    hybrid_proba = w_v * vqc_proba + w_x * xgb_proba + w_r * rf_proba
    preds = hybrid_proba.argmax(axis=1)
    pred_labels = [li.CLASS_NAMES[i] for i in preds]
    conf = hybrid_proba.max(axis=1)

    out_df = df.copy()
    gt_names = [li.CLASS_NAMES[i] for i in gt_labels]
    out_df["ground_truth"] = gt_names
    out_df["predicted_label"] = pred_labels
    out_df["confidence"] = conf.astype(float)

    accuracy = float((np.array(gt_names) == np.array(pred_labels)).mean())

    summary = pd.DataFrame(
        {
            "sample_size": [args.sample_size],
            "accuracy": [accuracy],
            "seed": [args.seed],
            "created_at": [datetime.utcnow().isoformat() + "Z"],
            "vqc_used": [vqc_model is not None],
        }
    )

    output_path = project_root / "pipeline" / "sample_predictions.xlsx"
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        out_df.to_excel(writer, index=False, sheet_name="samples")
        summary.to_excel(writer, index=False, sheet_name="summary")

    print(f"Saved sample Excel to {output_path}")
    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()

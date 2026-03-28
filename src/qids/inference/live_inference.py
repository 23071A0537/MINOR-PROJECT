import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

INFERENCE_DIR = Path(__file__).resolve().parent
EXPLAINABILITY_DIR = INFERENCE_DIR.parent / "explainability"
for _path in (INFERENCE_DIR, EXPLAINABILITY_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from lime_shared_helper import (
    build_lime_explainer,
    compute_shap_lime_consistency,
    explain_lime_instance,
    write_json,
)
from pipeline_utils import load_json, resolve_path, ensure_exists


CLASS_NAMES = ["NORMALL", "DoSD", "PROBE", "EXPLOIT", "MALWARE"]
LABEL_COLUMNS = [
    "label",
    "attack_cat",
    "Attack",
    "Label",
    "class",
    "master_label",
    "original_label",
]


def to_quantum_angles(mu: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(mu) * torch.pi


class SentinelAwareVAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 8, hidden_dims: list[int] | None = None) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        enc = []
        in_d = input_dim
        for h in hidden_dims:
            enc += [nn.Linear(in_d, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(0.05)]
            in_d = h
        self.encoder = nn.Sequential(*enc)
        self.fc_mu = nn.Linear(in_d, latent_dim)
        self.fc_log_var = nn.Linear(in_d, latent_dim)

        dec = []
        in_d = latent_dim
        for h in reversed(hidden_dims):
            dec += [nn.Linear(in_d, h), nn.BatchNorm1d(h), nn.GELU()]
            in_d = h
        dec += [nn.Linear(in_d, input_dim), nn.Sigmoid()]
        self.decoder = nn.Sequential(*dec)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        lv = torch.clamp(self.fc_log_var(h), -4, 4)
        return mu, lv

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, lv = self.encode(x)
        return self.decoder(mu), mu, lv


def load_stage2_artefacts(path: Path) -> Dict[str, Any]:
    artefacts = load_json(path)
    required = ["feature_names", "sentinel_value", "scaler_col_min", "scaler_col_max"]
    missing = [k for k in required if k not in artefacts]
    if missing:
        raise ValueError(f"Missing keys in stage2_preprocessing_artefacts.json: {missing}")
    return artefacts


def apply_ohe(df: pd.DataFrame, raw_col: str, prefix: str, feature_names: list[str]) -> pd.DataFrame:
    ohe_cols = [c for c in feature_names if c.startswith(f"{prefix}__")]
    if raw_col not in df.columns:
        return df
    values = df[raw_col].fillna("").astype(str).str.lower()
    mapped = values.map(
        lambda v: f"{prefix}__{v}" if f"{prefix}__{v}" in ohe_cols else f"{prefix}____OHE_ABSENT__"
    )
    dummies = pd.get_dummies(mapped)
    for col in ohe_cols:
        if col in dummies.columns:
            df[col] = dummies[col].astype(float)
        else:
            df[col] = 0.0
    if raw_col not in feature_names:
        df = df.drop(columns=[raw_col])
    return df


def frequency_encode(series: pd.Series) -> pd.Series:
    freqs = series.value_counts(normalize=True)
    return series.map(freqs).fillna(0.0).astype(float)


def preprocess_dataframe(
    df: pd.DataFrame,
    feature_names: list[str],
    sentinel_value: float,
    scaler_min: np.ndarray,
    scaler_max: np.ndarray,
) -> pd.DataFrame:
    df = df.copy()

    for col in LABEL_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=[col])

    df = apply_ohe(df, "protocol_type", "protocol_type", feature_names)
    df = apply_ohe(df, "state", "state", feature_names)

    for col in df.columns:
        if col in feature_names:
            if df[col].dtype == object:
                df[col] = frequency_encode(df[col].astype(str))
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in feature_names:
        if col not in df.columns:
            df[col] = sentinel_value

    df = df[feature_names]

    x = df.to_numpy(dtype=float)
    mask = x == sentinel_value
    denom = scaler_max - scaler_min
    denom = np.where(denom == 0, 1.0, denom)
    x_scaled = (x - scaler_min) / denom
    x_scaled = np.clip(x_scaled, 0.0, 1.0)
    x_scaled[mask] = sentinel_value

    return pd.DataFrame(x_scaled, columns=feature_names)


def encode_with_vae(x_df: pd.DataFrame, ckpt_path: Path, config_path: Path) -> np.ndarray:
    config = load_json(config_path)
    input_dim = int(config["input_dim"])
    hidden_dims = config.get("hidden_dims", [512, 256, 128])

    model = SentinelAwareVAE(input_dim, latent_dim=8, hidden_dims=hidden_dims)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    x_tensor = torch.tensor(x_df.to_numpy(dtype=np.float32))
    batch_size = 4096
    all_angles = []
    with torch.no_grad():
        for i in range(0, len(x_tensor), batch_size):
            batch = x_tensor[i : i + batch_size]
            mu, _ = model.encode(batch)
            angles = to_quantum_angles(mu).cpu().numpy()
            all_angles.append(angles)

    return np.vstack(all_angles)


def load_model(path: Path) -> Any:
    return joblib.load(path)


def predict_proba(model: Any, x: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(x)
    return np.asarray(proba, dtype=float)


def build_output_records(
    proba: np.ndarray, detail: str
) -> Dict[str, Any]:
    preds = proba.argmax(axis=1)
    labels = [CLASS_NAMES[i] for i in preds]
    conf = proba.max(axis=1)

    if detail == "summary":
        counts = {name: int((np.array(labels) == name).sum()) for name in CLASS_NAMES}
        return {"counts": counts}

    records = []
    for idx, (label, score, row_proba) in enumerate(zip(labels, conf, proba)):
        records.append(
            {
                "row": int(idx),
                "pred_label": label,
                "confidence": float(score),
                "probabilities": {CLASS_NAMES[i]: float(row_proba[i]) for i in range(len(CLASS_NAMES))},
            }
        )
    return {"predictions": records}


def make_latent_df(z_angles: np.ndarray, background_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if background_df is not None:
        cols = list(background_df.columns)
        if z_angles.shape[1] != len(cols):
            raise ValueError(
                f"Latent feature mismatch: got {z_angles.shape[1]} cols, "
                f"background has {len(cols)} cols"
            )
    else:
        cols = [f"z{i}" for i in range(z_angles.shape[1])]
    return pd.DataFrame(z_angles, columns=cols)


def build_shap_explainer(
    background_df: pd.DataFrame,
    predict_proba_fn,
):
    # Local import keeps SHAP optional unless enabled in config.
    import shap

    print("Building SHAP explainer... this can take a bit.")
    return shap.Explainer(predict_proba_fn, background_df)


def explain_samples(
    explainer,
    latent_df: pd.DataFrame,
    proba: np.ndarray,
    max_samples: int,
) -> List[Dict[str, Any]]:
    explanations: List[Dict[str, Any]] = []
    total = min(max_samples, len(latent_df))

    for idx in range(total):
        pred_class = int(np.argmax(proba[idx]))
        sv = explainer(latent_df.iloc[[idx]])
        values = sv.values[0, :, pred_class]

        base_values = sv.base_values
        if isinstance(base_values, np.ndarray) and base_values.ndim >= 2:
            base_value = float(base_values[0, pred_class])
        elif isinstance(base_values, np.ndarray) and base_values.ndim == 1:
            base_value = float(base_values[pred_class])
        else:
            base_value = float(base_values)

        explanations.append(
            {
                "row": int(idx),
                "pred_label": CLASS_NAMES[pred_class],
                "confidence": float(proba[idx, pred_class]),
                "base_value": base_value,
                "shap_values": dict(zip(latent_df.columns, values.tolist())),
            }
        )

    return explanations


def main() -> None:
    parser = argparse.ArgumentParser(description="Run live inference on new CSV data.")
    parser.add_argument(
        "--config",
        default="configs/pipeline/config.json",
        help="Path to config JSON (default: configs/pipeline/config.json)",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_json(config_path)
    project_root = resolve_path(config_path.parent, config["project_root"])

    live_cfg = config["live_inference"]
    artifacts = live_cfg["artifacts"]

    stage2_path = resolve_path(project_root, artifacts["stage2_artefacts"])
    vae_ckpt_path = resolve_path(project_root, artifacts["vae_checkpoint"])
    vae_cfg_path = resolve_path(project_root, artifacts["vae_config"])
    xgb_model_path = resolve_path(project_root, artifacts["xgb_model"])
    rf_model_path = resolve_path(project_root, artifacts["rf_model"])
    vqc_model_value = str(artifacts.get("vqc_model", "")).strip()
    vqc_model_path = resolve_path(project_root, vqc_model_value) if vqc_model_value else None

    require_xgb = live_cfg.get("require_xgb", True)
    require_vqc = bool(live_cfg.get("require_vqc", True))
    required = [stage2_path, vae_ckpt_path, vae_cfg_path, rf_model_path]
    if require_xgb:
        required.append(xgb_model_path)
    ensure_exists(required, "live inference artefacts")

    if require_vqc:
        if vqc_model_path is None:
            raise ValueError("live_inference.artifacts.vqc_model must be set when require_vqc=true.")
        ensure_exists([vqc_model_path], "VQC model")

    artefacts = load_stage2_artefacts(stage2_path)
    feature_names = artefacts["feature_names"]
    sentinel_value = float(artefacts["sentinel_value"])
    scaler_min = np.array(artefacts["scaler_col_min"], dtype=float)
    scaler_max = np.array(artefacts["scaler_col_max"], dtype=float)

    input_glob = live_cfg["input_glob"]
    input_paths = sorted(project_root.glob(input_glob))
    if not input_paths:
        raise FileNotFoundError(f"No input CSV files found for: {input_glob}")

    xgb_model = load_model(xgb_model_path) if xgb_model_path.exists() else None
    rf_model = load_model(rf_model_path)
    if hasattr(rf_model, "verbose"):
        rf_model.verbose = 0

    if xgb_model is not None and hasattr(xgb_model, "set_params"):
        try:
            xgb_model.set_params(verbosity=0)
        except Exception:
            pass

    vqc_model = None
    if vqc_model_path is not None and vqc_model_path.exists():
        vqc_model = load_model(vqc_model_path)

    weights = live_cfg["weights"]
    w_vqc = float(weights["vqc"])
    w_xgb = float(weights["xgb"])
    w_rf = float(weights["rf"])

    if xgb_model is None:
        if require_xgb:
            raise FileNotFoundError("XGBoost model not found. Set require_xgb=false to continue.")
        w_xgb = 0.0

    def _load_background_frame(background_value: str, label: str) -> pd.DataFrame:
        background_path = resolve_path(project_root, background_value)
        ensure_exists([background_path], label)
        suffix = background_path.suffix.lower()
        if suffix == ".parquet":
            return pd.read_parquet(background_path)
        if suffix == ".csv":
            return pd.read_csv(background_path)
        raise ValueError(f"Unsupported background file type for {label}: {background_path}")

    shap_cfg = live_cfg.get("shap", {})
    shap_enabled = bool(shap_cfg.get("enabled", False))
    shap_explainer = None
    shap_background = None
    shap_max_samples = int(shap_cfg.get("max_samples", 1))

    if shap_enabled:
        shap_background = _load_background_frame(shap_cfg["background"], "SHAP background")

    lime_cfg = live_cfg.get("lime", {})
    lime_enabled = bool(lime_cfg.get("enabled", True))
    lime_explainer = None
    lime_background = None
    lime_random_state = int(lime_cfg.get("random_state", 42))
    lime_num_features = int(lime_cfg.get("num_features", 10))
    lime_num_samples = int(lime_cfg.get("num_samples", 2000))
    default_lime_max = min(shap_max_samples, 50) if shap_enabled else 50
    lime_max_samples = int(lime_cfg.get("max_samples", default_lime_max))
    lime_consistency_top_k = int(lime_cfg.get("consistency_top_k", 5))
    lime_output_json = resolve_path(
        project_root,
        lime_cfg.get("output_json", "artifacts/inference/lime_local_explanations.json"),
    )

    lime_background_value = lime_cfg.get("background")
    if lime_background_value:
        lime_background = _load_background_frame(lime_background_value, "LIME background")

    lime_results: List[Dict[str, Any]] = []
    results = []
    for path in input_paths:
        df = pd.read_csv(path)
        x_scaled = preprocess_dataframe(df, feature_names, sentinel_value, scaler_min, scaler_max)
        z_angles = encode_with_vae(x_scaled, vae_ckpt_path, vae_cfg_path)

        rf_proba = predict_proba(rf_model, z_angles)
        xgb_proba = predict_proba(xgb_model, z_angles) if xgb_model is not None else np.zeros_like(rf_proba)

        if vqc_model is None:
            if require_vqc:
                raise FileNotFoundError("VQC model not found or path is not configured while require_vqc=true.")
            vqc_proba = np.zeros_like(xgb_proba)
            w_vqc = 0.0
        else:
            vqc_proba = predict_proba(vqc_model, z_angles)

        weight_sum = w_vqc + w_xgb + w_rf
        if weight_sum <= 0:
            raise ValueError("Invalid weights: sum must be > 0")
        w_v = w_vqc / weight_sum
        w_x = w_xgb / weight_sum
        w_r = w_rf / weight_sum

        hybrid_proba = w_v * vqc_proba + w_x * xgb_proba + w_r * rf_proba
        detail = live_cfg.get("output_detail", "full")
        payload = build_output_records(hybrid_proba, detail)

        latent_df = None
        if shap_enabled or lime_enabled:
            reference_background = shap_background if shap_background is not None else lime_background
            latent_df = make_latent_df(z_angles, reference_background)

        def _predict_proba(x):
            x = np.asarray(x, dtype=float)
            rf_p = predict_proba(rf_model, x)
            xgb_p = predict_proba(xgb_model, x) if xgb_model is not None else np.zeros_like(rf_p)
            vqc_p = predict_proba(vqc_model, x) if vqc_model is not None else np.zeros_like(rf_p)
            return w_v * vqc_p + w_x * xgb_p + w_r * rf_p

        if shap_enabled:
            if shap_explainer is None:
                shap_explainer = build_shap_explainer(shap_background, _predict_proba)

            payload["shap_explanations"] = explain_samples(
                shap_explainer,
                latent_df,
                hybrid_proba,
                shap_max_samples,
            )

        if lime_enabled:
            if latent_df is None:
                latent_df = make_latent_df(z_angles, shap_background)

            if lime_explainer is None:
                lime_training_df = (
                    lime_background
                    if lime_background is not None
                    else shap_background
                    if shap_background is not None
                    else latent_df
                )
                lime_explainer = build_lime_explainer(
                    lime_training_df,
                    class_names=CLASS_NAMES,
                    feature_names=list(latent_df.columns),
                    random_state=lime_random_state,
                )

            shap_by_row = {
                int(item["row"]): item.get("shap_values", {})
                for item in payload.get("shap_explanations", [])
            }

            rows_to_explain = min(lime_max_samples, len(latent_df))
            file_lime_records: List[Dict[str, Any]] = []
            for idx in range(rows_to_explain):
                pred_class = int(np.argmax(hybrid_proba[idx]))
                lime_local = explain_lime_instance(
                    explainer=lime_explainer,
                    predict_proba_fn=_predict_proba,
                    sample_row=latent_df.iloc[idx].to_numpy(dtype=float),
                    pred_class=pred_class,
                    class_names=CLASS_NAMES,
                    feature_names=list(latent_df.columns),
                    num_features=lime_num_features,
                    num_samples=lime_num_samples,
                )
                consistency = compute_shap_lime_consistency(
                    lime_contributions=lime_local.get("feature_contributions", []),
                    shap_values=shap_by_row.get(idx),
                    top_k=lime_consistency_top_k,
                )

                file_lime_records.append(
                    {
                        "row": int(idx),
                        "pred_label": CLASS_NAMES[pred_class],
                        "confidence": float(hybrid_proba[idx, pred_class]),
                        "probabilities": {
                            CLASS_NAMES[i]: float(hybrid_proba[idx, i]) for i in range(len(CLASS_NAMES))
                        },
                        "lime_local_explanation": lime_local,
                        "shap_lime_consistency": consistency,
                    }
                )

            lime_results.append(
                {
                    "file": str(path),
                    "rows": int(len(df)),
                    "rows_explained": int(rows_to_explain),
                    "explanations": file_lime_records,
                }
            )
            payload["lime_explanations_count"] = int(rows_to_explain)
        payload.update({"file": str(path), "rows": int(len(df))})
        results.append(payload)

    output_json = resolve_path(project_root, live_cfg["output_json"])
    output = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "input_glob": input_glob,
        "weights": {"vqc": w_v, "xgb": w_x, "rf": w_r},
        "results": results,
    }
    write_json(output_json, output)

    if lime_enabled:
        lime_output = {
            "created_at": datetime.utcnow().isoformat() + "Z",
            "input_glob": input_glob,
            "weights": {"vqc": w_v, "xgb": w_x, "rf": w_r},
            "lime_config": {
                "max_samples": lime_max_samples,
                "num_features": lime_num_features,
                "num_samples": lime_num_samples,
                "random_state": lime_random_state,
                "consistency_top_k": lime_consistency_top_k,
            },
            "results": lime_results,
        }
        write_json(lime_output_json, lime_output)
        print(f"Saved LIME local explanations to {lime_output_json}")

    print(f"Saved predictions to {output_json}")


if __name__ == "__main__":
    main()

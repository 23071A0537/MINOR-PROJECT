from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shap


LABEL_COLUMNS = [
    "label",
    "attack_cat",
    "Attack",
    "Label",
    "class",
    "master_label",
    "original_label",
]

CONNECTION_RATE_KEYWORDS = (
    "rate",
    "packets/s",
    "pkts",
    "count",
    "srv",
    "ct_",
)

PAYLOAD_SIZE_KEYWORDS = (
    "byte",
    "bytes",
    "packet length",
    "segment size",
    "size",
    "payload",
    "sbytes",
    "dbytes",
    "src_bytes",
    "dst_bytes",
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(base: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def to_quantum_angles(mu: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(mu) * torch.pi


class SentinelAwareVAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 8, hidden_dims: list[int] | None = None) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        enc_layers: list[nn.Module] = []
        in_d = input_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(in_d, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(0.05)]
            in_d = h
        self.encoder = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(in_d, latent_dim)
        self.fc_log_var = nn.Linear(in_d, latent_dim)

        dec_layers: list[nn.Module] = []
        in_d = latent_dim
        for h in reversed(hidden_dims):
            dec_layers += [nn.Linear(in_d, h), nn.BatchNorm1d(h), nn.GELU()]
            in_d = h
        dec_layers += [nn.Linear(in_d, input_dim), nn.Sigmoid()]
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        lv = torch.clamp(self.fc_log_var(h), -4, 4)
        return mu, lv


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

    model = SentinelAwareVAE(input_dim=input_dim, latent_dim=8, hidden_dims=hidden_dims)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    x_tensor = torch.tensor(x_df.to_numpy(dtype=np.float32))
    batch_size = 4096
    all_angles: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(x_tensor), batch_size):
            batch = x_tensor[i : i + batch_size]
            mu, _ = model.encode(batch)
            angles = to_quantum_angles(mu).cpu().numpy()
            all_angles.append(angles)

    return np.vstack(all_angles)


def sort_latent_cols(columns: list[str]) -> list[str]:
    def _key(col: str) -> tuple[int, str]:
        m = re.search(r"(\d+)$", col)
        if m:
            return (int(m.group(1)), col)
        return (10**9, col)

    return sorted(columns, key=_key)


def group_columns(columns: list[str], keywords: tuple[str, ...]) -> list[str]:
    picks: list[str] = []
    for col in columns:
        lowered = col.lower()
        if any(k in lowered for k in keywords):
            picks.append(col)
    return picks


def corr_abs(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return 0.0
    a_sel = a[mask]
    b_sel = b[mask]
    if np.std(a_sel) == 0 or np.std(b_sel) == 0:
        return 0.0
    return float(abs(np.corrcoef(a_sel, b_sel)[0, 1]))


def best_alignment_scores(
    raw_df: pd.DataFrame,
    latent_df: pd.DataFrame,
) -> dict[str, dict[str, float | str]]:
    conn_cols = group_columns(list(raw_df.columns), CONNECTION_RATE_KEYWORDS)
    payload_cols = group_columns(list(raw_df.columns), PAYLOAD_SIZE_KEYWORDS)

    aligned: dict[str, dict[str, float | str]] = {}
    for z_col in latent_df.columns:
        z_vals = pd.to_numeric(latent_df[z_col], errors="coerce").to_numpy(dtype=float)

        conn_best_score = 0.0
        conn_best_col = ""
        for col in conn_cols:
            score = corr_abs(z_vals, pd.to_numeric(raw_df[col], errors="coerce").to_numpy(dtype=float))
            if score > conn_best_score:
                conn_best_score = score
                conn_best_col = col

        payload_best_score = 0.0
        payload_best_col = ""
        for col in payload_cols:
            score = corr_abs(z_vals, pd.to_numeric(raw_df[col], errors="coerce").to_numpy(dtype=float))
            if score > payload_best_score:
                payload_best_score = score
                payload_best_col = col

        aligned[z_col] = {
            "connection_rate_score": round(conn_best_score, 6),
            "connection_rate_feature": conn_best_col,
            "payload_size_score": round(payload_best_score, 6),
            "payload_size_feature": payload_best_col,
        }

    return aligned


def load_shap_samples(live_predictions_path: Path) -> tuple[np.ndarray, list[int], list[str]]:
    payload = load_json(live_predictions_path)
    results = payload.get("results", [])
    if not results:
        raise ValueError("No results found in live predictions JSON.")

    shap_rows: list[dict[str, Any]] = []
    for result in results:
        shap_rows.extend(result.get("shap_explanations", []))

    if not shap_rows:
        raise ValueError("No shap_explanations found in live predictions JSON.")

    first = shap_rows[0].get("shap_values", {})
    if not isinstance(first, dict) or not first:
        raise ValueError("Invalid shap_values format in live predictions JSON.")

    latent_cols = sort_latent_cols(list(first.keys()))

    row_indices: list[int] = []
    shap_matrix = np.zeros((len(shap_rows), len(latent_cols)), dtype=float)
    for i, item in enumerate(shap_rows):
        row_indices.append(int(item.get("row", i)))
        sv = item.get("shap_values", {})
        for j, col in enumerate(latent_cols):
            shap_matrix[i, j] = float(sv.get(col, 0.0))

    return shap_matrix, row_indices, latent_cols


def choose_semantic_top_dims(
    top_candidates: list[str],
    alignment: dict[str, dict[str, float | str]],
) -> tuple[str, str]:
    if not top_candidates:
        raise ValueError("No top candidates available for semantic selection.")

    top_connection = max(top_candidates, key=lambda c: float(alignment[c]["connection_rate_score"]))

    payload_candidates = [c for c in top_candidates if c != top_connection]
    if payload_candidates:
        top_payload = max(payload_candidates, key=lambda c: float(alignment[c]["payload_size_score"]))
    else:
        top_payload = top_connection

    return top_connection, top_payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate SHAP summary plot over 8D VAE latent space from live inference artifacts."
    )
    parser.add_argument(
        "--config",
        default="configs/pipeline/config.json",
        help="Path to pipeline config JSON.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/plots/shap_summary_vae_latent_space.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--stats-output",
        default="artifacts/plots/shap_summary_vae_latent_space_stats.json",
        help="Output JSON with mean |SHAP| and latent alignment notes.",
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
    live_predictions_path = resolve_path(project_root, live_cfg["output_json"])

    input_glob = live_cfg["input_glob"]
    input_paths = sorted(project_root.glob(input_glob))
    if not input_paths:
        raise FileNotFoundError(f"No input files found for pattern: {input_glob}")
    input_path = input_paths[0]

    stage2 = load_json(stage2_path)
    feature_names = stage2["feature_names"]
    sentinel_value = float(stage2["sentinel_value"])
    scaler_min = np.array(stage2["scaler_col_min"], dtype=float)
    scaler_max = np.array(stage2["scaler_col_max"], dtype=float)

    raw_df = pd.read_csv(input_path)
    x_scaled = preprocess_dataframe(raw_df, feature_names, sentinel_value, scaler_min, scaler_max)
    z_angles = encode_with_vae(x_scaled, vae_ckpt_path, vae_cfg_path)

    shap_matrix, row_indices, shap_cols = load_shap_samples(live_predictions_path)
    if z_angles.shape[1] != len(shap_cols):
        raise ValueError(
            f"Latent dimension mismatch: encoded {z_angles.shape[1]} columns, SHAP has {len(shap_cols)}."
        )

    latent_df = pd.DataFrame(z_angles, columns=shap_cols)
    max_row = int(max(row_indices))
    if max_row >= len(latent_df):
        raise IndexError(
            f"SHAP row index {max_row} is out of range for encoded latent dataframe with {len(latent_df)} rows."
        )

    latent_plot_df = latent_df.iloc[row_indices].reset_index(drop=True)
    raw_plot_df = raw_df.iloc[row_indices].reset_index(drop=True)

    mean_abs = np.abs(shap_matrix).mean(axis=0)
    shap_ranking = sorted(zip(shap_cols, mean_abs.tolist()), key=lambda kv: kv[1], reverse=True)
    top_two = [name for name, _ in shap_ranking[:2]]

    alignment = best_alignment_scores(raw_plot_df, latent_plot_df)
    top_connection, top_payload = choose_semantic_top_dims(top_two, alignment)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values=shap_matrix,
        features=latent_plot_df,
        feature_names=shap_cols,
        max_display=len(shap_cols),
        plot_type="dot",
        color_bar=True,
        show=False,
    )

    ax = plt.gca()
    ax.set_title("SHAP Summary Plot on 8D VAE Latent Space", fontsize=14, pad=12)
    ax.set_xlabel("SHAP value (impact on model output)")

    subtitle = (
        f"Top mean |SHAP| dimensions: {top_two[0]} and {top_two[1]} | "
        f"Connection-rate aligned: {top_connection} | Payload-size aligned: {top_payload}"
    )
    plt.figtext(0.5, 0.01, subtitle, ha="center", fontsize=9)

    output_path = resolve_path(project_root, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()

    stats_payload = {
        "output_plot": str(output_path),
        "source_input": str(input_path),
        "source_live_predictions": str(live_predictions_path),
        "num_samples": int(len(latent_plot_df)),
        "latent_columns": shap_cols,
        "mean_abs_shap": {name: round(val, 8) for name, val in shap_ranking},
        "top_two_mean_abs_shap": top_two,
        "alignment_notes": {
            "connection_rate_best_dimension": top_connection,
            "payload_size_best_dimension": top_payload,
            "selection_pool": top_two,
            "per_dimension_alignment": alignment,
        },
    }

    stats_path = resolve_path(project_root, args.stats_output)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats_payload, f, indent=2)

    print(f"Saved plot to: {output_path}")
    print(f"Saved stats to: {stats_path}")


if __name__ == "__main__":
    main()

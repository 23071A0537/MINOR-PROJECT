from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CLASS_NAMES = {
    0: "NORMALL",
    1: "DoSD",
    2: "PROBE",
    3: "EXPLOIT",
    4: "MALWARE",
}


def inverse_scale(x_scaled_df: pd.DataFrame, scaler_min: np.ndarray, scaler_max: np.ndarray, sentinel_value: float) -> pd.DataFrame:
    x = x_scaled_df.to_numpy(dtype=float)
    mask = x == sentinel_value

    denom = scaler_max - scaler_min
    denom = np.where(denom == 0, 1.0, denom)

    x_raw = x * denom + scaler_min
    x_raw[mask] = sentinel_value

    return pd.DataFrame(x_raw, columns=x_scaled_df.columns)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a balanced multi-class CSV for live inference.")
    parser.add_argument(
        "--x-path",
        default="PreProcessing/stage_2_with_zero_v2/stage2_X_test.parquet",
        help="Path to stage2_X_test parquet.",
    )
    parser.add_argument(
        "--y-path",
        default="PreProcessing/stage_2_with_zero_v2/stage2_y_test.parquet",
        help="Path to stage2_y_test parquet.",
    )
    parser.add_argument(
        "--artefacts-path",
        default="PreProcessing/stage_2_with_zero_v2/stage2_preprocessing_artefacts.json",
        help="Path to stage2_preprocessing_artefacts.json.",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=100,
        help="Number of rows to sample per class.",
    )
    parser.add_argument(
        "--total-samples",
        type=int,
        default=None,
        help="Total number of random rows to sample across all classes. If set, overrides --samples-per-class.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/inference/all_classes_input.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed. Leave unset for different random samples each run.",
    )
    args = parser.parse_args()

    x_path = Path(args.x_path)
    y_path = Path(args.y_path)
    artefacts_path = Path(args.artefacts_path)
    output_path = Path(args.output)

    x_df = pd.read_parquet(x_path)
    y_df = pd.read_parquet(y_path)

    y_col = y_df.columns[0]
    y_series = y_df[y_col].astype(int)

    with artefacts_path.open("r", encoding="utf-8") as f:
        artefacts = json.load(f)

    scaler_min = np.array(artefacts["scaler_col_min"], dtype=float)
    scaler_max = np.array(artefacts["scaler_col_max"], dtype=float)
    sentinel_value = float(artefacts["sentinel_value"])

    sampled_indices = []
    rng = np.random.default_rng(args.seed)

    if args.total_samples is not None:
        if args.total_samples <= 0:
            raise ValueError("--total-samples must be > 0")
        if args.total_samples > len(x_df):
            raise ValueError(
                f"--total-samples ({args.total_samples}) exceeds dataset rows ({len(x_df)})."
            )
        sampled_indices = rng.choice(len(x_df), size=args.total_samples, replace=False).tolist()
    else:
        for cls in sorted(CLASS_NAMES.keys()):
            idx = np.flatnonzero(y_series.to_numpy() == cls)
            if len(idx) == 0:
                continue
            k = min(args.samples_per_class, len(idx))
            chosen = rng.choice(idx, size=k, replace=False)
            sampled_indices.extend(chosen.tolist())

    sampled_indices = np.array(sampled_indices, dtype=int)
    sampled_indices.sort()

    x_sample_scaled = x_df.iloc[sampled_indices].reset_index(drop=True)
    y_sample = y_series.iloc[sampled_indices].reset_index(drop=True)

    x_sample_raw = inverse_scale(x_sample_scaled, scaler_min, scaler_max, sentinel_value)
    x_sample_raw["master_label"] = y_sample.map(CLASS_NAMES).astype(str)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    x_sample_raw.to_csv(output_path, index=False)

    counts = y_sample.value_counts().sort_index()
    print(f"Saved: {output_path.resolve()}")
    if args.seed is None:
        print("Sampling mode: random seed not fixed (fresh sample each run).")
    else:
        print(f"Sampling seed: {args.seed}")
    print("Class distribution in generated CSV:")
    for cls, count in counts.items():
        print(f"  {CLASS_NAMES[int(cls)]}: {int(count)}")


if __name__ == "__main__":
    main()

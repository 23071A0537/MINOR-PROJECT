import argparse
import sys
from pathlib import Path

from pipeline_utils import ensure_exists, load_json, resolve_path, run_cmd
from plotting import plot_confusion_matrix, plot_f1_scores


def run_notebook(notebook_path: Path, project_root: Path) -> None:
    cmd = [
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--inplace",
        str(notebook_path),
    ]
    run_cmd(cmd, project_root)


def refresh_live_input(project_root: Path, live_cfg: dict) -> None:
    random_cfg = live_cfg.get("random_input", {})
    if not bool(random_cfg.get("enabled", False)):
        return

    input_glob = str(live_cfg.get("input_glob", "")).strip()
    if not input_glob:
        raise ValueError("live_inference.input_glob must be set when random_input is enabled.")

    if any(ch in input_glob for ch in "*?[]"):
        raise ValueError(
            "live_inference.random_input.enabled=true requires live_inference.input_glob to be a concrete file path (no wildcard)."
        )

    sample_size = int(random_cfg.get("sample_size", 50))
    if sample_size <= 0:
        raise ValueError("live_inference.random_input.sample_size must be > 0")

    x_path = str(
        random_cfg.get(
            "x_path",
            "PreProcessing/stage_2_with_zero_v2/stage2_X_test.parquet",
        )
    )
    y_path = str(
        random_cfg.get(
            "y_path",
            "PreProcessing/stage_2_with_zero_v2/stage2_y_test.parquet",
        )
    )
    artefacts_path = str(
        random_cfg.get(
            "artefacts_path",
            "PreProcessing/stage_2_with_zero_v2/stage2_preprocessing_artefacts.json",
        )
    )

    cmd = [
        sys.executable,
        str(resolve_path(project_root, "src/qids/preprocessing/make_all_classes_input.py")),
        "--x-path",
        x_path,
        "--y-path",
        y_path,
        "--artefacts-path",
        artefacts_path,
        "--total-samples",
        str(sample_size),
        "--output",
        input_glob,
    ]

    if random_cfg.get("seed") is not None:
        cmd.extend(["--seed", str(random_cfg["seed"])])

    run_cmd(cmd, project_root)
    print(f"Random live input regenerated: {sample_size} rows -> {resolve_path(project_root, input_glob)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the end-to-end pipeline.")
    parser.add_argument(
        "--config",
        default="configs/pipeline/config.json",
        help="Path to config JSON (default: configs/pipeline/config.json)",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_json(config_path)

    project_root = resolve_path(config_path.parent, config["project_root"])
    run_mode = config["run_mode"]
    notebooks = config["notebooks"]
    artifacts = config["artifacts"]
    stage5 = config["stage5"]
    plots = config["plots"]
    live_cfg = config.get("live_inference")

    if run_mode.get("live_inference") == "run":
        if live_cfg is None:
            raise ValueError("live_inference configuration is required when run_mode.live_inference is 'run'.")

        refresh_live_input(project_root, live_cfg)

        cmd = [
            sys.executable,
            str(resolve_path(project_root, "src/qids/inference/live_inference.py")),
            "--config",
            str(config_path),
        ]
        run_cmd(cmd, project_root)
        print("Live inference completed successfully.")
        return

    # Stage 1
    if run_mode["stage1"] == "notebook":
        run_notebook(resolve_path(project_root, notebooks["stage1"]), project_root)

    # Stage 2
    if run_mode["stage2"] == "notebook":
        run_notebook(resolve_path(project_root, notebooks["stage2"]), project_root)

    # Stage 3
    if run_mode["stage3"] == "notebook":
        run_notebook(resolve_path(project_root, notebooks["stage3"]), project_root)

    # Stage 4
    if run_mode["stage4"] == "notebook":
        run_notebook(resolve_path(project_root, notebooks["stage4"]), project_root)

    # Validate required artifacts before Stage 5
    required = [
        resolve_path(project_root, artifacts["stage2_y_test"]),
        resolve_path(project_root, artifacts["vae_z_test"]),
        resolve_path(project_root, artifacts["vqc_winner_test_proba"]),
        resolve_path(project_root, artifacts["rf_test_proba"]),
        resolve_path(project_root, artifacts["xgb_model"]),
    ]
    ensure_exists(required, "stage artifacts")

    # Stage 5 - Hybrid assembly
    weights = stage5["weights"]
    output_json = resolve_path(project_root, stage5["output_json"])
    cmd = [
        sys.executable,
        str(resolve_path(project_root, "src/qids/inference/hybrid_assembly.py")),
        "--output",
        str(output_json),
        "--w_vqc",
        str(weights["vqc"]),
        "--w_xgb",
        str(weights["xgb"]),
        "--w_rf",
        str(weights["rf"]),
    ]
    run_cmd(cmd, project_root)

    if plots.get("enabled", False):
        plots_dir = resolve_path(project_root, plots["output_dir"])
        plots_dir.mkdir(parents=True, exist_ok=True)
        report = load_json(output_json)
        plot_confusion_matrix(
            report["metrics"]["confusion_matrix"],
            plots_dir / "confusion_matrix.png",
        )
        plot_f1_scores(
            report["metrics"]["classification_report"],
            plots_dir / "per_class_f1.png",
        )

    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()

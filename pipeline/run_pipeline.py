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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the end-to-end pipeline.")
    parser.add_argument(
        "--config",
        default="pipeline/config.json",
        help="Path to config JSON (default: pipeline/config.json)",
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
        cmd = [
            sys.executable,
            str(resolve_path(project_root, "pipeline/live_inference.py")),
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
        str(resolve_path(project_root, "hybrid_assembly.py")),
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

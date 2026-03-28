# Pipeline

Runtime scripts have been moved to `src/qids/inference` with root wrappers in `scripts/`.

## What it does

- Optional notebook execution for stages 1-4
- Validates expected artifacts before stage 5
- Runs hybrid assembly with configured weights
- Generates plots from the final metrics

## Quick start

1. Edit `configs/pipeline/config.json` to set `run_mode` and paths.
2. Run:

```bash
python scripts/run_pipeline.py --config configs/pipeline/config.json
```

## Notes

- Notebook execution requires `jupyter` to be installed and on PATH.
- Stage 5 uses `src/qids/inference/hybrid_assembly.py` and expects the v6 VQC winner proba by default.

## Explainability Webpage

Use the Streamlit dashboard to present SHAP + LIME explanation artifacts.

1. Ensure these artifacts exist:
   - `artifacts/inference/lime_local_explanations.json`
   - `artifacts/explainability/class_wise_shap_interpretation.json` (optional but recommended)
2. Launch:

```bash
python scripts/explainability_dashboard.py
```

The sidebar lets you point to different artifact paths if needed.

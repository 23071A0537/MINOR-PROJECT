# Pipeline

This folder contains a reproducible, configurable pipeline runner for stages 1-5.

## What it does

- Optional notebook execution for stages 1-4
- Validates expected artifacts before stage 5
- Runs hybrid assembly with configured weights
- Generates plots from the final metrics

## Quick start

1. Edit `pipeline/config.json` to set `run_mode` and paths.
2. Run:

```bash
python pipeline/run_pipeline.py --config pipeline/config.json
```

## Notes

- Notebook execution requires `jupyter` to be installed and on PATH.
- Stage 5 uses `hybrid_assembly.py` and expects the v6 VQC winner proba by default.

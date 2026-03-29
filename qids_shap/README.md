# QIDS — SHAP Explainability Layer

This folder is now a lightweight compatibility folder (README + requirements).
Active explainability source code has been moved to `src/qids/explainability`.

## 📁 File Structure

```
src/qids/explainability/
├── shap_pipeline.py          ← Core pipeline class (import this in your app)
├── save_artefacts_colab.py   ← Run once in Colab to export model files
├── app.py                    ← Streamlit dashboard
├── explainability_dashboard.py
└── lime_shared_helper.py

qids_shap/
├── README.md
└── requirements.txt

qids_models/
    ├── xgboost_model.pkl
    ├── rf_model.pkl
    ├── vae_encoder.pkl
    └── shap_background.parquet
```

## 🚀 Quick Start

### Step 1 — Save artefacts from Colab

Paste `src/qids/explainability/save_artefacts_colab.py` cells into your Colab notebook.
Download `qids_models.zip`, extract into this folder.

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3A — Use in your pipeline (Python)

```python
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / "src"))

from qids.explainability.shap_pipeline import QIDSSHAPPipeline
import pandas as pd

pipeline = QIDSSHAPPipeline.load("qids_models/")
raw_df   = pd.read_csv("network_traffic.csv")   # your raw features
result   = pipeline.explain(raw_df, sample_idx=0)

print(result["explanation_text"])
print(result["global_importance_df"])
# result["chart_bar_b64"] → base64 PNG for bar chart
# result["chart_waterfall_b64"] → base64 PNG for waterfall
```

### Step 3B — Launch Streamlit dashboard

```bash
python scripts/explainability_dashboard.py
```

## 🔑 result dict keys

| Key                    | Type      | Description                   |
| ---------------------- | --------- | ----------------------------- |
| `predicted_label`      | str       | e.g. "DoS Attack"             |
| `confidence`           | float     | % confidence                  |
| `probabilities`        | dict      | per-class probabilities       |
| `shap_values_dict`     | dict      | SHAP value per latent feature |
| `global_importance_df` | DataFrame | mean \|SHAP\| table           |
| `explanation_text`     | str       | human-readable summary        |
| `chart_bar_b64`        | str       | base64 PNG bar chart          |
| `chart_beeswarm_b64`   | str       | base64 PNG beeswarm           |
| `chart_waterfall_b64`  | str       | base64 PNG waterfall          |

## ⚠️ Keras VAE Note

If your VAE encoder is a Keras model (not sklearn), in `src/qids/explainability/shap_pipeline.py`
replace the `_encode` method's `.transform()` call with `.predict()` and
load the model with `tf.keras.models.load_model()` instead of `joblib.load()`.

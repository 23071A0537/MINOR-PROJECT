"""
save_artefacts_colab.py
────────────────────────
Run this ONCE in your Colab notebook AFTER training to save everything
your production pipeline needs.

Paste and run each section in its own cell.
"""

# ══════════════════════════════════════════════════════════════════════════════
#  CELL A — Save models + background (run after your training cells)
# ══════════════════════════════════════════════════════════════════════════════
import joblib, os
import pandas as pd

os.makedirs("qids_models", exist_ok=True)

# ── Models ────────────────────────────────────────────────────────────────────
joblib.dump(xgb_model,    "qids_models/xgboost_model.pkl")
joblib.dump(rf_model,     "qids_models/rf_model.pkl")
joblib.dump(vae_encoder,  "qids_models/vae_encoder.pkl")   # your VAE encoder object
#   If your VAE encoder is a custom Keras/PyTorch model, see the note below ↓

# ── SHAP background (latent space) ───────────────────────────────────────────
#   X_sample is the 500-row latent DataFrame you used in the SHAP notebook
X_sample.to_parquet("qids_models/shap_background.parquet", index=False)

print("✅ Saved:")
for f in os.listdir("qids_models"):
    size = os.path.getsize(f"qids_models/{f}") / 1024
    print(f"   {f:40s}  {size:.1f} KB")


# ══════════════════════════════════════════════════════════════════════════════
#  CELL B — If your VAE encoder is a Keras model (not sklearn)
# ══════════════════════════════════════════════════════════════════════════════
#
#   Option 1: Save as SavedModel (recommended)
#   vae_encoder_model.save("qids_models/vae_encoder_keras")
#
#   Option 2: Save weights only
#   vae_encoder_model.save_weights("qids_models/vae_encoder_weights.h5")
#
#   Then in shap_pipeline.py replace self.vae_encoder.transform(raw_df) with:
#   z = self.vae_encoder.predict(raw_df.values)


# ══════════════════════════════════════════════════════════════════════════════
#  CELL C — Zip and download everything
# ══════════════════════════════════════════════════════════════════════════════
import zipfile
from google.colab import files

with zipfile.ZipFile("qids_models.zip", "w") as zf:
    for f in os.listdir("qids_models"):
        zf.write(f"qids_models/{f}", f)
        print(f"  Added: {f}")

print("\n✅ Downloading qids_models.zip …")
files.download("qids_models.zip")

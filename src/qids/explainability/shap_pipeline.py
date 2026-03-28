"""
shap_pipeline.py
────────────────
QIDS (Quantum Intrusion Detection System) — SHAP Explainability Layer
Pipeline: Raw Features → VAE Encoder → Latent z0–z7 → Ensemble → SHAP

Usage:
    from shap_pipeline import QIDSSHAPPipeline
    pipeline = QIDSSHAPPipeline.load("models/")
    result   = pipeline.explain(raw_features_df)
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import io, base64, os, warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_DIR))

from lime_shared_helper import (
    build_lime_explainer,
    compute_shap_lime_consistency,
    explain_lime_instance,
)


# ─────────────────────────────────────────────────────────────────────────────
#  CLASS LABELS  (edit to match your dataset's actual class names)
# ─────────────────────────────────────────────────────────────────────────────
CLASS_LABELS = {
    0: "Normal",
    1: "DoS Attack",
    2: "Probe Attack",
    3: "R2L Attack",
    4: "U2R Attack",
}

SEVERITY = {          # colour coding for UI
    "Normal":      "#00e5a0",
    "DoS Attack":  "#ff4c60",
    "Probe Attack":"#ffa500",
    "R2L Attack":  "#ff4c60",
    "U2R Attack":  "#ff1a1a",
}


# ─────────────────────────────────────────────────────────────────────────────
#  PIPELINE CLASS
# ─────────────────────────────────────────────────────────────────────────────
class QIDSSHAPPipeline:
    """
    One object that wraps everything needed for inference + explanation.

    Attributes
    ----------
    vae_encoder       : fitted VAE encoder (sklearn-like transform interface)
    xgb_model         : fitted XGBoostClassifier
    rf_model          : fitted RandomForestClassifier
    background_latent : pd.DataFrame  — 500-sample SHAP background (z0–z7)
    explainer         : shap.Explainer  — built once, reused every call
    lime_explainer    : LimeTabularExplainer — built once, reused every call
    """

    def __init__(self, vae_encoder, xgb_model, rf_model, background_latent: pd.DataFrame):
        self.vae_encoder       = vae_encoder
        self.xgb_model         = xgb_model
        self.rf_model          = rf_model
        self.background_latent = background_latent
        self._n_classes        = len(xgb_model.classes_)
        self._latent_cols      = background_latent.columns.tolist()
        self._class_names      = [CLASS_LABELS.get(i, f"Class_{i}") for i in range(self._n_classes)]

        print("⚙️  Building SHAP Explainer (one-time, ~30s)…")
        self.explainer = shap.Explainer(self._ensemble_predict_proba,
                                        self.background_latent)
        print("✅ SHAP Explainer ready.")

        print("⚙️  Building LIME Explainer (one-time, ~10s)…")
        self.lime_explainer = build_lime_explainer(
            self.background_latent,
            class_names=self._class_names,
            feature_names=self._latent_cols,
            random_state=42,
        )
        print("✅ LIME Explainer ready.")

    # ── Save / Load ──────────────────────────────────────────────────────────
    @classmethod
    def load(cls, model_dir: str) -> "QIDSSHAPPipeline":
        """Load all artefacts from a directory."""
        vae  = joblib.load(os.path.join(model_dir, "vae_encoder.pkl"))
        xgb  = joblib.load(os.path.join(model_dir, "xgboost_model.pkl"))
        rf   = joblib.load(os.path.join(model_dir, "rf_model.pkl"))
        bg   = pd.read_parquet(os.path.join(model_dir, "shap_background.parquet"))
        return cls(vae, xgb, rf, bg)

    def save(self, model_dir: str):
        """Persist all artefacts (call this once from Colab)."""
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.vae_encoder,       os.path.join(model_dir, "vae_encoder.pkl"))
        joblib.dump(self.xgb_model,         os.path.join(model_dir, "xgboost_model.pkl"))
        joblib.dump(self.rf_model,          os.path.join(model_dir, "rf_model.pkl"))
        self.background_latent.to_parquet(  os.path.join(model_dir, "shap_background.parquet"), index=False)
        print(f"✅ All artefacts saved to '{model_dir}/'")

    # ── Internal helpers ─────────────────────────────────────────────────────
    def _encode(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Raw features → latent z0–z7 via VAE encoder."""
        z = self.vae_encoder.transform(raw_df)          # shape (n, 8)
        return pd.DataFrame(z, columns=self._latent_cols)

    def _ensemble_predict_proba(self, X):
        return 0.6 * self.xgb_model.predict_proba(X) + \
               0.4 * self.rf_model.predict_proba(X)

    # ── Main explain entry-point ──────────────────────────────────────────────
    def explain(self, raw_df: pd.DataFrame, sample_idx: int = 0) -> dict:
        """
        Parameters
        ----------
        raw_df     : pd.DataFrame — one or more rows of raw network features
        sample_idx : which row to explain (for waterfall / text)

        Returns
        -------
        dict with keys:
            predicted_class, confidence, probabilities,
            shap_values_dict, explanation_text,
            global_importance_df, lime_local_explanation,
            shap_lime_consistency,
            chart_beeswarm_b64, chart_bar_b64, chart_waterfall_b64
        """
        # 1. Encode
        latent_df = self._encode(raw_df)

        # 2. Predict
        probs      = self._ensemble_predict_proba(latent_df)
        pred_class = int(np.argmax(probs[sample_idx]))
        confidence = float(probs[sample_idx, pred_class])
        label      = CLASS_LABELS.get(pred_class, f"Class_{pred_class}")

        # 3. SHAP
        sv = self.explainer(latent_df)          # shape (n_samples, n_features, n_classes)

        # 4. Numerical summary
        global_imp = self._global_importance(sv)
        shap_values_dict = dict(zip(self._latent_cols,
                                    sv.values[sample_idx, :, pred_class].tolist()))

        # 4b. LIME local explanation + SHAP-LIME consistency
        lime_local = explain_lime_instance(
            explainer=self.lime_explainer,
            predict_proba_fn=self._ensemble_predict_proba,
            sample_row=latent_df.iloc[sample_idx].to_numpy(dtype=float),
            pred_class=pred_class,
            class_names=self._class_names,
            feature_names=self._latent_cols,
            num_features=min(10, len(self._latent_cols)),
            num_samples=2000,
        )
        shap_lime_consistency = compute_shap_lime_consistency(
            lime_contributions=lime_local.get("feature_contributions", []),
            shap_values=shap_values_dict,
            top_k=5,
        )

        # 5. Charts
        bar_b64       = self._chart_bar(sv, pred_class)
        beeswarm_b64  = self._chart_beeswarm(sv, pred_class, latent_df)
        waterfall_b64 = self._chart_waterfall(sv, sample_idx, pred_class, latent_df)

        # 6. Text
        text = self._text_explanation(sv, sample_idx, pred_class,
                                       label, confidence, latent_df)

        return {
            # ── Prediction ──
            "predicted_class":   pred_class,
            "predicted_label":   label,
            "confidence":        round(confidence * 100, 2),
            "probabilities":     {CLASS_LABELS.get(i, f"Class_{i}"): round(float(p), 4)
                                  for i, p in enumerate(probs[sample_idx])},
            "severity_color":    SEVERITY.get(label, "#888888"),

            # ── Numerical ──
            "shap_values_dict":  shap_values_dict,
            "global_importance_df": global_imp,
            "lime_local_explanation": lime_local,
            "shap_lime_consistency": shap_lime_consistency,

            # ── Text ──
            "explanation_text":  text,

            # ── Charts (base64 PNG — drop into <img src="data:image/png;base64,..."> ) ──
            "chart_bar_b64":       bar_b64,
            "chart_beeswarm_b64":  beeswarm_b64,
            "chart_waterfall_b64": waterfall_b64,
        }

    # ── Numerical summary ────────────────────────────────────────────────────
    def _global_importance(self, sv) -> pd.DataFrame:
        """Mean |SHAP| per feature per class, sorted by overall importance."""
        rows = []
        for cls in range(self._n_classes):
            mean_abs = np.abs(sv.values[:, :, cls]).mean(axis=0)
            for feat, val in zip(self._latent_cols, mean_abs):
                rows.append({"Feature": feat,
                             "Class": CLASS_LABELS.get(cls, f"Class_{cls}"),
                             "Mean_Abs_SHAP": round(float(val), 6)})
        df = pd.DataFrame(rows)
        pivot = df.pivot(index="Feature", columns="Class", values="Mean_Abs_SHAP")
        pivot["Overall"] = pivot.mean(axis=1)
        return pivot.sort_values("Overall", ascending=False).round(6)

    # ── Chart helpers — all return base64 PNG strings ────────────────────────
    @staticmethod
    def _fig_to_b64(fig) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        return b64

    def _chart_bar(self, sv, pred_class: int) -> str:
        label = CLASS_LABELS.get(pred_class, f"Class_{pred_class}")
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor("#0d1117")
        shap.summary_plot(sv[:, :, pred_class], self.background_latent,
                          plot_type="bar", show=False, color="#00e5a0")
        ax = plt.gca()
        ax.set_facecolor("#0d1117")
        ax.set_title(f"Global Feature Importance — {label}",
                     color="white", fontsize=12, pad=10)
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        return self._fig_to_b64(fig)

    def _chart_beeswarm(self, sv, pred_class: int, latent_df: pd.DataFrame) -> str:
        label = CLASS_LABELS.get(pred_class, f"Class_{pred_class}")
        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor("#0d1117")
        shap.summary_plot(sv[:, :, pred_class], latent_df,
                          plot_type="dot", show=False, color_bar=True)
        ax = plt.gca()
        ax.set_facecolor("#0d1117")
        ax.set_title(f"SHAP Summary (Beeswarm) — {label}",
                     color="white", fontsize=12, pad=10)
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        return self._fig_to_b64(fig)

    def _chart_waterfall(self, sv, idx: int, pred_class: int,
                          latent_df: pd.DataFrame) -> str:
        sv_single = shap.Explanation(
            values       = sv.values[idx, :, pred_class],
            base_values  = sv.base_values[idx, pred_class],
            data         = sv.data[idx],
            feature_names= self._latent_cols
        )
        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor("#0d1117")
        shap.plots.waterfall(sv_single, show=False)
        ax = plt.gca()
        ax.set_facecolor("#0d1117")
        label = CLASS_LABELS.get(pred_class, f"Class_{pred_class}")
        ax.set_title(f"Waterfall — Sample #{idx} → {label}",
                     color="white", fontsize=12, pad=10)
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        return self._fig_to_b64(fig)

    # ── Text explanation ─────────────────────────────────────────────────────
    def _text_explanation(self, sv, idx: int, pred_class: int,
                           label: str, confidence: float,
                           latent_df: pd.DataFrame) -> str:
        shap_vals = sv.values[idx, :, pred_class]
        top_idx   = np.argsort(np.abs(shap_vals))[::-1][:3]
        lines = [
            f"🔴 INTRUSION DETECTED: {label}" if label != "Normal"
            else f"🟢 Traffic classified as: {label}",
            f"   Confidence : {confidence*100:.1f}%",
            f"   Model      : XGBoost (60%) + Random Forest (40%) via VAE-A latent space",
            "",
            "📌 Top latent features driving this classification:",
        ]
        for rank, i in enumerate(top_idx, 1):
            direction = "↑ pushed toward this class" if shap_vals[i] > 0 \
                        else "↓ pushed away from other classes"
            lines.append(
                f"   {rank}. {self._latent_cols[i]:>4}  "
                f"(encoded value: {latent_df.iloc[idx, i]:+.4f})  "
                f"SHAP: {shap_vals[i]:+.4f}  {direction}"
            )
        lines += [
            "",
            f"💡 The VAE encoder compressed your raw network features into 8 latent",
            f"   dimensions (z0–z7). SHAP then identified which of those compressed",
            f"   signals most influenced the {label} classification.",
        ]
        return "\n".join(lines)

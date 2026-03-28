"""
app.py — QIDS SHAP Dashboard (Streamlit)
────────────────────────────────────────
Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from shap_pipeline import QIDSSHAPPipeline, CLASS_LABELS, SEVERITY

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QIDS — SHAP Explainability",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark theme CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
  body, .stApp { background: #0d1117; color: #e6edf3; font-family: 'JetBrains Mono', monospace; }
  .block-container { padding: 2rem 2.5rem; }
  .metric-card {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 10px; padding: 1.2rem 1.5rem; margin-bottom: 1rem;
  }
  .metric-card h3 { margin: 0 0 .3rem 0; font-size: .8rem; color: #8b949e; letter-spacing:.08em; text-transform:uppercase; }
  .metric-card p  { margin: 0; font-size: 1.6rem; font-weight: 700; }
  .threat-banner {
    border-radius: 10px; padding: 1rem 1.5rem; margin-bottom: 1.5rem;
    font-size: 1.1rem; font-weight: 700; letter-spacing: .04em;
  }
  .explanation-box {
    background: #161b22; border: 1px solid #30363d; border-radius: 8px;
    padding: 1rem 1.4rem; font-family: monospace; font-size: .85rem;
    white-space: pre-wrap; line-height: 1.7;
  }
  table { width: 100%; border-collapse: collapse; }
  th { background: #161b22; color: #8b949e; font-size:.75rem; padding:.5rem .8rem; text-align:left; }
  td { padding: .4rem .8rem; border-bottom: 1px solid #21262d; font-size:.85rem; }
  tr:hover td { background: #1c2128; }
</style>
""", unsafe_allow_html=True)


# ── Load pipeline (cached so it only builds once) ────────────────────────────
@st.cache_resource
def load_pipeline():
    return QIDSSHAPPipeline.load("qids_models/")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ QIDS · SHAP Panel")
    st.markdown("---")
    st.markdown("**Model:** XGBoost 60% + RF 40%")
    st.markdown("**Encoder:** VAE-A (latent dim = 8)")
    st.markdown("**Explainer:** SHAP KernelExplainer")
    st.markdown("---")
    st.markdown("### 📂 Input Mode")
    input_mode = st.radio("", ["Upload CSV", "Manual Entry"], label_visibility="collapsed")
    st.markdown("---")
    sample_idx = st.number_input("Sample index to explain", min_value=0, value=0, step=1)
    run_btn    = st.button("▶ Run SHAP Explanation", use_container_width=True)


# ── Main header ───────────────────────────────────────────────────────────────
st.markdown("# 🔍 Quantum Intrusion Detection — SHAP Explainability")
st.markdown("Understand **why** the ensemble model classified network traffic as it did.")
st.markdown("---")

raw_df = None

# ── Input: Upload CSV ─────────────────────────────────────────────────────────
if input_mode == "Upload CSV":
    uploaded = st.file_uploader("Upload raw network feature CSV", type=["csv", "parquet"])
    if uploaded:
        raw_df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") \
                 else pd.read_parquet(uploaded)
        st.success(f"✅ Loaded {raw_df.shape[0]} samples × {raw_df.shape[1]} features")
        with st.expander("Preview data"):
            st.dataframe(raw_df.head(5))

# ── Input: Manual Entry ───────────────────────────────────────────────────────
else:
    st.markdown("#### Enter raw network feature values")
    st.caption("These are your original network traffic features — before VAE encoding.")
    # Placeholder: 10 common NSL-KDD features (replace with your actual feature names)
    feature_names = [
        "duration","protocol_type","service","flag",
        "src_bytes","dst_bytes","land","wrong_fragment",
        "urgent","hot"
    ]
    cols = st.columns(5)
    values = {}
    for i, feat in enumerate(feature_names):
        values[feat] = cols[i % 5].number_input(feat, value=0.0, format="%.4f")
    raw_df = pd.DataFrame([values])


# ── Run ───────────────────────────────────────────────────────────────────────
if run_btn and raw_df is not None:
    pipeline = load_pipeline()
    idx = min(int(sample_idx), len(raw_df) - 1)

    with st.spinner("🔄 Encoding → Ensemble → SHAP…"):
        result = pipeline.explain(raw_df, sample_idx=idx)

    # ── Threat banner ─────────────────────────────────────────────────────────
    color = result["severity_color"]
    label = result["predicted_label"]
    conf  = result["confidence"]
    icon  = "🔴" if label != "Normal" else "🟢"
    st.markdown(
        f'<div class="threat-banner" style="background:{color}22;border:1.5px solid {color};color:{color}">'
        f'{icon} &nbsp; {label.upper()} &nbsp;·&nbsp; {conf:.1f}% confidence</div>',
        unsafe_allow_html=True
    )

    # ── Metric row ────────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    for col, title, val in [
        (m1, "PREDICTED CLASS", result["predicted_label"]),
        (m2, "CONFIDENCE",      f"{result['confidence']}%"),
        (m3, "SAMPLE INDEX",    f"#{idx}"),
        (m4, "FEATURES",        f"z0–z7 (VAE-A latent)"),
    ]:
        col.markdown(
            f'<div class="metric-card"><h3>{title}</h3><p>{val}</p></div>',
            unsafe_allow_html=True
        )

    # ── Probability breakdown ─────────────────────────────────────────────────
    st.markdown("### 📊 Class Probabilities")
    prob_df = pd.DataFrame(
        result["probabilities"].items(), columns=["Class", "Probability"]
    ).sort_values("Probability", ascending=False)
    prob_df["Probability"] = prob_df["Probability"].map(lambda x: f"{x*100:.2f}%")
    st.dataframe(prob_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Charts ────────────────────────────────────────────────────────────────
    st.markdown("### 📈 SHAP Visualisations")
    tab1, tab2, tab3 = st.tabs(["📊 Bar Chart", "🌸 Beeswarm", "💧 Waterfall"])

    with tab1:
        st.markdown("**Global Feature Importance** — average impact of each latent dimension across all samples.")
        st.image(f"data:image/png;base64,{result['chart_bar_b64']}", use_column_width=True)

    with tab2:
        st.markdown("**Summary Beeswarm** — each dot = one sample. Red = high feature value, blue = low.")
        st.image(f"data:image/png;base64,{result['chart_beeswarm_b64']}", use_column_width=True)

    with tab3:
        st.markdown(f"**Waterfall** — how each latent feature pushed sample #{idx} from baseline to final prediction.")
        st.image(f"data:image/png;base64,{result['chart_waterfall_b64']}", use_column_width=True)

    st.markdown("---")

    # ── Global importance table ───────────────────────────────────────────────
    st.markdown("### 🔢 Numerical Importance Table (mean |SHAP|)")
    st.dataframe(result["global_importance_df"].style.background_gradient(
        cmap="YlOrRd", subset=["Overall"]), use_container_width=True)

    # ── Text explanation ──────────────────────────────────────────────────────
    st.markdown("### 🗣️ Plain-English Explanation")
    st.markdown(
        f'<div class="explanation-box">{result["explanation_text"]}</div>',
        unsafe_allow_html=True
    )

    # ── Per-feature SHAP values ───────────────────────────────────────────────
    with st.expander("🔬 Raw SHAP values for this sample"):
        sv_df = pd.DataFrame(
            result["shap_values_dict"].items(), columns=["Latent Feature", "SHAP Value"]
        ).sort_values("SHAP Value", key=abs, ascending=False)
        sv_df["Direction"] = sv_df["SHAP Value"].apply(
            lambda v: "↑ Increases prediction" if v > 0 else "↓ Decreases prediction"
        )
        st.dataframe(sv_df, use_container_width=True, hide_index=True)

elif run_btn and raw_df is None:
    st.warning("⚠️ Please provide input data first.")
else:
    st.info("👈 Upload a CSV or enter features in the sidebar, then click **Run SHAP Explanation**.")

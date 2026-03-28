from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LIME_PATH = PROJECT_ROOT / "artifacts" / "inference" / "lime_local_explanations.json"
DEFAULT_CLASSWISE_PATH = PROJECT_ROOT / "artifacts" / "explainability" / "class_wise_shap_interpretation.json"


st.set_page_config(
    page_title="QIDS Explainability Dashboard",
    page_icon="🛰",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
  --bg-1: #06131f;
  --bg-2: #0c2533;
  --panel: rgba(255, 255, 255, 0.08);
  --panel-border: rgba(255, 255, 255, 0.18);
  --text-main: #ecf4f9;
  --text-muted: #a7bdcb;
  --good: #26d07c;
  --warn: #ffb347;
  --bad: #ff5f56;
  --accent: #30c5ff;
}

.stApp {
  background:
    radial-gradient(60rem 40rem at 8% -10%, rgba(48, 197, 255, 0.24), transparent 60%),
    radial-gradient(50rem 35rem at 100% 0%, rgba(255, 179, 71, 0.20), transparent 60%),
    linear-gradient(160deg, var(--bg-1), var(--bg-2));
  color: var(--text-main);
  font-family: 'Space Grotesk', sans-serif;
}

.main .block-container {
  padding-top: 1.1rem;
  padding-bottom: 2.0rem;
}

h1, h2, h3 {
  font-family: 'Space Grotesk', sans-serif;
  letter-spacing: 0.01em;
}

p, li, .stMarkdown, .stCaption {
  color: var(--text-main);
}

.metric-card {
  border: 1px solid var(--panel-border);
  background: var(--panel);
  backdrop-filter: blur(10px);
  border-radius: 16px;
  padding: 0.9rem 1rem;
  box-shadow: 0 10px 25px rgba(0,0,0,0.18);
  animation: fadeInUp 420ms ease;
}

.metric-title {
  color: var(--text-muted);
  font-size: 0.74rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.metric-value {
  margin-top: 0.2rem;
  font-weight: 700;
  font-size: 1.5rem;
}

.metric-sub {
  margin-top: 0.15rem;
  color: var(--text-muted);
  font-size: 0.82rem;
}

.status-chip {
  display: inline-block;
  border-radius: 999px;
  padding: 0.28rem 0.72rem;
  font-size: 0.80rem;
  font-weight: 600;
  border: 1px solid rgba(255, 255, 255, 0.2);
}

.card {
  border: 1px solid var(--panel-border);
  background: var(--panel);
  border-radius: 16px;
  padding: 0.9rem 1rem;
  box-shadow: 0 10px 25px rgba(0,0,0,0.18);
  animation: fadeInUp 420ms ease;
}

.kv {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.88rem;
}

.lime-row {
  margin-bottom: 0.44rem;
}

.lime-label {
  display: flex;
  justify-content: space-between;
  gap: 0.9rem;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.83rem;
  color: var(--text-main);
  margin-bottom: 0.18rem;
}

.track {
  width: 100%;
  height: 9px;
  border-radius: 999px;
  background: rgba(255,255,255,0.14);
  overflow: hidden;
}

.bar {
  height: 100%;
  border-radius: 999px;
}

@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(8px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

[data-testid="stDataFrame"] div[role="table"] {
  border-radius: 12px;
}

</style>
""",
    unsafe_allow_html=True,
)


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _flatten_lime_payload(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for entry in payload.get("results", []):
        source_file = entry.get("file", "unknown")
        for explanation in entry.get("explanations", []):
            row_obj = {
                "source_file": source_file,
                "row": int(explanation.get("row", -1)),
                "pred_label": explanation.get("pred_label", "N/A"),
                "confidence": float(explanation.get("confidence", 0.0)),
                "probabilities": explanation.get("probabilities", {}),
                "lime_local_explanation": explanation.get("lime_local_explanation", {}),
                "shap_lime_consistency": explanation.get("shap_lime_consistency", {}),
            }
            rows.append(row_obj)
    return rows


def _status_color(status: str) -> str:
    status = status.lower()
    if status == "agreement":
        return "#26d07c"
    if status == "partial_agreement":
        return "#ffb347"
    if status == "divergence":
        return "#ff5f56"
    return "#9db3bf"


INPUT_SIGNAL_KEYWORDS = (
    "packet",
    "pkt",
    "byte",
    "size",
    "duration",
    "rate",
    "count",
    "ttl",
    "window",
    "fragment",
    "payload",
    "src",
    "dst",
)


@st.cache_data(show_spinner=False)
def _load_source_dataframe(path_text: str) -> pd.DataFrame | None:
    path = Path(path_text)
    if not path.exists():
        return None

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    return None


def _pick_candidate_input_features(source_df: pd.DataFrame, limit: int = 12) -> List[str]:
    numeric_cols = [col for col in source_df.columns if pd.api.types.is_numeric_dtype(source_df[col])]
    if not numeric_cols:
        return []

    selected: List[str] = []
    for col in numeric_cols:
        lowered = col.lower()
        if any(keyword in lowered for keyword in INPUT_SIGNAL_KEYWORDS):
            selected.append(col)

    if len(selected) >= limit:
        return selected[:limit]

    variances = source_df[numeric_cols].var(numeric_only=True).sort_values(ascending=False)
    for col in variances.index.tolist():
        if col not in selected:
            selected.append(col)
        if len(selected) >= limit:
            break
    return selected


def _percentile_rank(series: pd.Series, value: float) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float((numeric <= value).mean() * 100.0)


def _level_from_percentile(percentile: float) -> str:
    if percentile >= 95:
        return "very high"
    if percentile >= 75:
        return "high"
    if percentile <= 5:
        return "very low"
    if percentile <= 25:
        return "low"
    return "normal"


def _signal_type(feature_name: str) -> str:
    lowered = feature_name.lower()
    if "byte" in lowered or "packet" in lowered or "pkt" in lowered or "size" in lowered:
        return "traffic volume"
    if "duration" in lowered:
        return "flow duration"
    if "rate" in lowered:
        return "traffic rate"
    if "count" in lowered:
        return "connection count"
    if "ttl" in lowered:
        return "header timing"
    return "network signal"


def _build_input_based_explanation(
    source_file: str,
    row_index: int,
    pred_label: str,
    confidence: float,
) -> Dict[str, Any]:
    source_df = _load_source_dataframe(source_file)
    if source_df is None:
        return {
            "summary": "Input-source file could not be loaded for this row, so input-based reasoning is unavailable.",
            "signals": [],
        }

    if row_index < 0 or row_index >= len(source_df):
        return {
            "summary": "Row index is outside the source file range, so input-based reasoning is unavailable.",
            "signals": [],
        }

    row = source_df.iloc[row_index]
    candidate_features = _pick_candidate_input_features(source_df)
    signals: List[Dict[str, Any]] = []

    for feature in candidate_features:
        raw_value = row.get(feature)
        if pd.isna(raw_value):
            continue

        try:
            value = float(raw_value)
        except Exception:
            continue

        percentile = _percentile_rank(source_df[feature], value)
        if percentile is None:
            continue

        signals.append(
            {
                "feature": feature,
                "value": value,
                "percentile": round(percentile, 2),
                "level": _level_from_percentile(percentile),
                "signal_type": _signal_type(feature),
                "deviation_score": round(abs(percentile - 50.0), 2),
            }
        )

    if not signals:
        return {
            "summary": (
                f"Predicted {pred_label} at {confidence * 100:.2f}% confidence, but no numeric input features "
                "were available to build a readable input-based explanation."
            ),
            "signals": [],
        }

    signals.sort(key=lambda item: item["deviation_score"], reverse=True)
    top_signals = signals[:3]

    highlights = "; ".join(
        f"{item['feature']} is {item['level']} (value {item['value']:.4g}, p{item['percentile']})"
        for item in top_signals
    )

    summary = (
        f"Predicted {pred_label} with {confidence * 100:.2f}% confidence because key input signals show: "
        f"{highlights}."
    )

    return {
        "summary": summary,
        "signals": signals[:8],
    }


def _render_metric_card(title: str, value: str, subtitle: str) -> None:
    st.markdown(
        (
            "<div class='metric-card'>"
            f"<div class='metric-title'>{title}</div>"
            f"<div class='metric-value'>{value}</div>"
            f"<div class='metric-sub'>{subtitle}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_probability_table(probabilities: Dict[str, Any]) -> None:
    rows = [
        {"Class": str(label), "Probability": float(prob)}
        for label, prob in probabilities.items()
    ]
    if not rows:
        st.info("No probability vector available for this sample.")
        return

    prob_df = pd.DataFrame(rows).sort_values("Probability", ascending=False)
    prob_df["Probability %"] = (prob_df["Probability"] * 100).map(lambda x: f"{x:.2f}%")
    st.dataframe(
        prob_df[["Class", "Probability %"]],
        use_container_width=True,
        hide_index=True,
    )


def _render_lime_bars(contributions: List[Dict[str, Any]]) -> None:
    if not contributions:
        st.info("No LIME feature contributions found.")
        return

    max_abs = max(abs(float(item.get("weight", 0.0))) for item in contributions) or 1.0
    html_parts = ["<div class='card'>"]

    for item in contributions:
        feature = str(item.get("feature", "?"))
        weight = float(item.get("weight", 0.0))
        width = max(4.0, (abs(weight) / max_abs) * 100.0)
        color = "#26d07c" if weight >= 0 else "#ff5f56"

        html_parts.append(
            "<div class='lime-row'>"
            f"<div class='lime-label'><span>{feature}</span><span>{weight:+.4f}</span></div>"
            "<div class='track'>"
            f"<div class='bar' style='width:{width:.2f}%; background:{color};'></div>"
            "</div>"
            "</div>"
        )

    html_parts.append("</div>")
    st.markdown("".join(html_parts), unsafe_allow_html=True)


def _render_consistency(consistency: Dict[str, Any]) -> None:
    status = str(consistency.get("status", "unknown"))
    color = _status_color(status)
    summary = str(consistency.get("summary", "No summary available."))
    overlap = consistency.get("overlap_features", [])
    sign_divergence = consistency.get("sign_divergence_features", [])

    st.markdown(
        (
            "<div class='card'>"
            f"<span class='status-chip' style='background:{color}22; color:{color}; border-color:{color}66;'>"
            f"{status}</span>"
            f"<p style='margin-top:0.65rem;'>{summary}</p>"
            f"<p class='kv'>jaccard_overlap: {float(consistency.get('jaccard_overlap', 0.0)):.4f}</p>"
            f"<p class='kv'>sign_agreement_ratio: {float(consistency.get('sign_agreement_ratio', 0.0)):.4f}</p>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    if overlap:
        st.write("Overlap features")
        st.write(", ".join(str(x) for x in overlap))

    if sign_divergence:
        st.write("Sign divergence details")
        div_df = pd.DataFrame(sign_divergence)
        st.dataframe(div_df, use_container_width=True, hide_index=True)


def _render_classwise_panel(classwise_payload: Dict[str, Any]) -> None:
    classes = classwise_payload.get("class_wise_explanations", [])
    if not classes:
        st.info("No class-wise SHAP interpretation entries found.")
        return

    class_names = [entry.get("class_name", "unknown") for entry in classes]
    selected_name = st.selectbox("Class-wise SHAP view", class_names)
    selected = next((item for item in classes if item.get("class_name") == selected_name), classes[0])

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        _render_metric_card(
            "Predicted Count",
            str(selected.get("support", {}).get("predicted_count", 0)),
            "predictions for this class",
        )
    with col_b:
        mean_conf = selected.get("confidence", {}).get("mean_predicted_confidence")
        mean_conf_str = "N/A" if mean_conf is None else f"{float(mean_conf) * 100:.2f}%"
        _render_metric_card("Mean Pred Confidence", mean_conf_str, "from predictions")
    with col_c:
        shap_rows = selected.get("support", {}).get("shap_samples_available", 0)
        _render_metric_card("SHAP Samples", str(shap_rows), "rows with SHAP vectors")

    st.markdown("#### Technical interpretation")
    st.markdown(f"<div class='card'>{selected.get('technical_interpretation', 'N/A')}</div>", unsafe_allow_html=True)

    pos = pd.DataFrame(selected.get("top_positive_drivers", []))
    neg = pd.DataFrame(selected.get("top_negative_drivers", []))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("Positive drivers")
        if pos.empty:
            st.info("No positive SHAP drivers listed.")
        else:
            st.dataframe(pos, use_container_width=True, hide_index=True)

    with c2:
        st.markdown("Negative drivers")
        if neg.empty:
            st.info("No negative SHAP drivers listed.")
        else:
            st.dataframe(neg, use_container_width=True, hide_index=True)


with st.sidebar:
    st.title("Explainability Viewer")
    st.caption("SHAP + LIME local explanations")
    st.markdown("---")

    lime_path_text = st.text_input("LIME artifact path", str(DEFAULT_LIME_PATH))
    classwise_path_text = st.text_input("Class-wise SHAP path", str(DEFAULT_CLASSWISE_PATH))

    st.markdown("---")
    st.caption("Tip: regenerate artifacts via live inference, then refresh this page.")


st.markdown("# Quantum IDS Explanation Webpage")
st.caption("Presentation dashboard for local LIME, SHAP signals, and consistency triage.")
st.markdown("---")

lime_path = Path(lime_path_text)
if not lime_path.exists():
    st.error(f"LIME artifact not found: {lime_path}")
    st.stop()

try:
    lime_payload = _load_json(lime_path)
except Exception as exc:
    st.error(f"Failed to parse LIME artifact: {exc}")
    st.stop()

rows = _flatten_lime_payload(lime_payload)
if not rows:
    st.warning("No explanation rows found in the selected LIME artifact.")
    st.stop()

status_counts: Dict[str, int] = {}
for row in rows:
    status = str(row.get("shap_lime_consistency", {}).get("status", "unknown"))
    status_counts[status] = status_counts.get(status, 0) + 1

metric_cols = st.columns(4)
with metric_cols[0]:
    _render_metric_card("Explained Rows", str(len(rows)), "rows with local explanation")
with metric_cols[1]:
    agreement = status_counts.get("agreement", 0)
    _render_metric_card("Agreement", str(agreement), "SHAP and LIME aligned")
with metric_cols[2]:
    partial = status_counts.get("partial_agreement", 0)
    _render_metric_card("Partial", str(partial), "mixed consistency")
with metric_cols[3]:
    divergence = status_counts.get("divergence", 0)
    _render_metric_card("Divergence", str(divergence), "triage candidates")

st.markdown("---")

source_files = sorted({str(item["source_file"]) for item in rows})
classes = sorted({str(item["pred_label"]) for item in rows})
statuses = sorted({str(item.get("shap_lime_consistency", {}).get("status", "unknown")) for item in rows})

f1, f2, f3 = st.columns(3)
with f1:
    selected_file = st.selectbox("Source file", options=["All"] + source_files)
with f2:
    selected_class = st.selectbox("Predicted class", options=["All"] + classes)
with f3:
    selected_status = st.selectbox("Consistency status", options=["All"] + statuses)

filtered_rows = []
for item in rows:
    if selected_file != "All" and str(item["source_file"]) != selected_file:
        continue
    if selected_class != "All" and str(item["pred_label"]) != selected_class:
        continue
    status = str(item.get("shap_lime_consistency", {}).get("status", "unknown"))
    if selected_status != "All" and status != selected_status:
        continue
    filtered_rows.append(item)

if not filtered_rows:
    st.warning("No rows matched the selected filters.")
    st.stop()

row_labels = [
    f"row {item['row']} | {item['pred_label']} | {item['source_file']}"
    for item in filtered_rows
]
selected_row_label = st.selectbox("Select explanation row", row_labels)
selected_index = row_labels.index(selected_row_label)
selected_row = filtered_rows[selected_index]
input_based = _build_input_based_explanation(
    source_file=str(selected_row.get("source_file", "")),
    row_index=int(selected_row.get("row", -1)),
    pred_label=str(selected_row.get("pred_label", "N/A")),
    confidence=float(selected_row.get("confidence", 0.0)),
)

st.markdown("---")

left, right = st.columns([1.2, 1.0])
with left:
    st.markdown("### Prediction snapshot")
    st.markdown(
        (
            "<div class='card'>"
            f"<p class='kv'>row: {selected_row['row']}</p>"
            f"<p class='kv'>predicted_label: {selected_row['pred_label']}</p>"
            f"<p class='kv'>confidence: {float(selected_row['confidence']) * 100:.2f}%</p>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    st.markdown("### Class probabilities")
    _render_probability_table(selected_row.get("probabilities", {}))

with right:
    st.markdown("### SHAP-LIME consistency")
    _render_consistency(selected_row.get("shap_lime_consistency", {}))

st.markdown("---")

st.markdown("### Input-based explanation")
st.markdown(
    f"<div class='card'>{input_based.get('summary', 'No input-based summary available.')}</div>",
    unsafe_allow_html=True,
)

input_signals = input_based.get("signals", [])
if input_signals:
    input_df = pd.DataFrame(input_signals)
    input_df = input_df[["feature", "signal_type", "value", "percentile", "level", "deviation_score"]]
    input_df = input_df.rename(
        columns={
            "feature": "Feature",
            "signal_type": "Signal Type",
            "value": "Value",
            "percentile": "Percentile",
            "level": "Level",
            "deviation_score": "Deviation",
        }
    )
    st.dataframe(input_df, use_container_width=True, hide_index=True)

st.markdown("---")

lime_section = selected_row.get("lime_local_explanation", {})
st.markdown("### LIME local explanation")
st.markdown(f"<div class='card'>{lime_section.get('because', 'No textual explanation found.')}</div>", unsafe_allow_html=True)

contribs = lime_section.get("feature_contributions", [])
st.markdown("#### Feature contribution bars")
_render_lime_bars(contribs)

if contribs:
    contrib_df = pd.DataFrame(contribs)
    contrib_df = contrib_df.sort_values("weight", key=lambda s: s.abs(), ascending=False)
    st.markdown("#### Contribution table")
    st.dataframe(contrib_df, use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("## Class-wise SHAP summary")

classwise_path = Path(classwise_path_text)
if classwise_path.exists():
    try:
        classwise_payload = _load_json(classwise_path)
        _render_classwise_panel(classwise_payload)
    except Exception as exc:
        st.warning(f"Class-wise SHAP file is present but could not be parsed: {exc}")
else:
    st.info("Class-wise SHAP file not found. Generate it to unlock this section.")

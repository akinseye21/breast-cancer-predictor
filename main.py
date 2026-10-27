"""
Breast Cancer Prediction App (Streamlit)
Run with:  streamlit run app.py
Needs in the same folder: breast_cancer_model.joblib (from the notebook) and Breast_Cancer_Diagnostics.csv
"""
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import streamlit as st

MODEL_PATH = "breast_cancer_model.joblib"
DATA_PATH = "Breast_Cancer_Diagnostics.csv"

# ----------------------------------------------------------------------------
# Logic (no UI code here, so it is easy to test)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["features"], bundle["model_name"]


@st.cache_data
def load_reference(features):
    """Reference statistics from the original dataset (used for slider ranges and charts)."""
    df = pd.read_csv(DATA_PATH)
    y = df["diagnosis"].map({"B": 0, "M": 1})
    X = df[list(features)]
    return {
        "min": X.min(), "max": X.max(), "median": X.median(),
        "benign_median": X[y == 0].median(), "malignant_median": X[y == 1].median(),
        "n_rows": len(df), "pct_malignant": float(y.mean() * 100),
    }


def predict_one(model, features, values: dict, threshold: float):
    """Predict for a single patient given {feature: value}."""
    row = pd.DataFrame([values])[list(features)]
    proba = float(model.predict_proba(row)[0, 1])
    return proba, proba >= threshold


def predict_batch(model, features, df: pd.DataFrame, threshold: float):
    """Predict for many patients. Returns a copy of df with prediction columns added."""
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing[:5])}{' ...' if len(missing) > 5 else ''}")
    proba = model.predict_proba(df[list(features)])[:, 1]
    out = df.copy()
    out["P(malignant)"] = proba.round(4)
    out["Prediction"] = np.where(proba >= threshold, "Malignant", "Benign")
    return out


def risk_band(p: float) -> str:
    if p < 0.10: return "Very low"
    if p < 0.40: return "Low"
    if p < 0.60: return "Uncertain / borderline"
    if p < 0.90: return "High"
    return "Very high"


def nice(name: str) -> str:
    return name.replace("_", " ").title()


# ---- UI ----------------------------------------------------------------------
st.set_page_config(page_title="Breast Cancer Predictor", page_icon="🩺", layout="wide")

try:
    model, FEATURES, MODEL_NAME = load_model()
    REF = load_reference(tuple(FEATURES))
except FileNotFoundError as e:
    st.error(f"Missing file: {e.filename}. Put it in the same folder as app.py and rerun.")
    st.stop()

# Initialise each input with the dataset median the first time the app loads
for f in FEATURES:
    st.session_state.setdefault(f"in_{f}", float(REF["median"][f]))


def set_preset(values):
    for f in FEATURES:
        st.session_state[f"in_{f}"] = float(values[f])


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Controls")
    mode = st.radio("Input mode", ["Single patient", "Batch (CSV upload)"])
    threshold = st.slider(
        "Decision threshold", 0.05, 0.95, 0.50, 0.05,
        help="Predict 'Malignant' when P(malignant) ≥ threshold. Lowering it catches more cancers "
             "(higher recall) at the cost of more false alarms.")
    st.caption("A lower threshold is safer for screening: it misses fewer cancers.")
    st.divider()
    st.subheader("Model")
    st.write(f"**{MODEL_NAME}**")
    st.caption(f"Trained on {REF['n_rows']} patients ({REF['pct_malignant']:.0f}% malignant), "
               "30 features from fine-needle aspirate images.")

# ---------------- Header ----------------
st.title("🩺 Breast Cancer Prediction")
st.caption("Predicts whether a tumour is **benign** or **malignant** from cell-nucleus measurements (WDBC dataset).")
st.warning("**Educational tool only.** This is not a medical device and must not be used for real diagnosis. "
           "Always consult a qualified clinician.", icon="⚠️")

# =============================================================================
# SINGLE PATIENT
# =============================================================================
if mode == "Single patient":
    st.subheader("1. Enter the measurements")
    b1, b2, b3, _ = st.columns([1.3, 1.3, 1, 3])
    b1.button("Example: typical benign", on_click=set_preset, args=(REF["benign_median"],), use_container_width=True)
    b2.button("Example: typical malignant", on_click=set_preset, args=(REF["malignant_median"],), use_container_width=True)
    b3.button("Reset", on_click=set_preset, args=(REF["median"],), use_container_width=True)

    groups = {"Mean values": "_mean", "Standard error": "_se", "Worst values": "_worst"}
    tabs = st.tabs(list(groups))
    for tab, suffix in zip(tabs, groups.values()):
        with tab:
            cols = st.columns(2)
            feats = [f for f in FEATURES if f.endswith(suffix)]
            for i, f in enumerate(feats):
                lo, hi = float(REF["min"][f]), float(REF["max"][f])
                span = hi - lo
                decimals = int(max(2, 3 - np.floor(np.log10(span))))
                cols[i % 2].slider(nice(f), lo, hi, step=span / 200, format=f"%.{decimals}f", key=f"in_{f}")

    values = {f: st.session_state[f"in_{f}"] for f in FEATURES}
    proba, is_malignant = predict_one(model, FEATURES, values, threshold)

    st.divider()
    st.subheader("2. Result")
    left, right = st.columns([1, 1.4])

    with left:
        if is_malignant:
            st.error("### Prediction: MALIGNANT")
        else:
            st.success("### Prediction: BENIGN")
        m1, m2 = st.columns(2)
        m1.metric("P(malignant)", f"{proba:.1%}")
        m2.metric("Risk band", risk_band(proba))
        st.progress(proba, text=f"Malignancy probability (threshold = {threshold:.0%})")
        if 0.35 <= proba <= 0.65:
            st.info("The probability is close to the middle, so the model is unsure. "
                    "Treat this result with extra caution.")

    with right:
        st.markdown("**Where this patient sits vs. typical cases**")
        key_feats = ["radius_worst", "texture_worst", "concavity_worst",
                     "concave_points_worst", "symmetry_worst", "area_worst"]
        fig, ax = plt.subplots(figsize=(6.5, 3.4))
        for i, f in enumerate(key_feats):
            rng = REF["max"][f] - REF["min"][f]
            scale = lambda v: (v - REF["min"][f]) / rng
            b, m, p = scale(REF["benign_median"][f]), scale(REF["malignant_median"][f]), scale(values[f])
            ax.plot([b, m], [i, i], color="#cccccc", lw=3, zorder=1)
            ax.scatter(b, i, color="#2a9d8f", s=60, zorder=2, label="Typical benign" if i == 0 else None)
            ax.scatter(m, i, color="#e76f51", s=60, zorder=2, label="Typical malignant" if i == 0 else None)
            ax.scatter(p, i, color="black", marker="D", s=70, zorder=3, label="This patient" if i == 0 else None)
        ax.set_yticks(range(len(key_feats)))
        ax.set_yticklabels([nice(f) for f in key_feats])
        ax.invert_yaxis(); ax.set_xlim(-0.02, 1.02)
        ax.set_xlabel("Position within dataset range (0 = lowest, 1 = highest)")
        ax.grid(axis="x", alpha=0.3)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        ax.legend(loc="lower right", fontsize=8, frameon=False)
        st.pyplot(fig, clear_figure=True)

# =============================================================================
# BATCH
# =============================================================================
else:
    st.subheader("Upload a CSV of patients")
    st.caption("The file must contain the 30 feature columns (same names as the training data). "
               "Extra columns such as `id` or `diagnosis` are kept and ignored by the model.")
    up = st.file_uploader("CSV file", type="csv")
    if up is None:
        st.info("Upload a file to see predictions. Tip: you can try it with the original dataset.")
    else:
        try:
            result = predict_batch(model, FEATURES, pd.read_csv(up), threshold)
        except ValueError as e:
            st.error(str(e))
        else:
            n_m = int((result["Prediction"] == "Malignant").sum())
            c1, c2, c3 = st.columns(3)
            c1.metric("Patients", len(result))
            c2.metric("Predicted malignant", n_m)
            c3.metric("Predicted benign", len(result) - n_m)

            fig, ax = plt.subplots(figsize=(6, 2.8))
            ax.hist(result["P(malignant)"], bins=20, color="#e76f51", alpha=0.85)
            ax.axvline(threshold, color="black", ls="--", label=f"Threshold {threshold:.0%}")
            ax.set_xlabel("P(malignant)"); ax.set_ylabel("Patients"); ax.legend(frameon=False)
            for s in ("top", "right"): ax.spines[s].set_visible(False)
            st.pyplot(fig, clear_figure=True)

            st.dataframe(result, use_container_width=True, height=300)
            st.download_button("⬇️ Download predictions (CSV)", result.to_csv(index=False).encode(),
                               "predictions.csv", "text/csv")

st.divider()
st.markdown(
    """
    <a href="https://www.linkedin.com/in/abayomi-akinseye-77b114119/" target="_blank" style="text-decoration:none;">
        <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg"
             width="20" style="vertical-align:middle;"> LinkedIn
    </a>
    &nbsp;&nbsp;&nbsp;
    <a href="https://github.com/akinseye21" target="_blank" style="text-decoration:none;">
        <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg"
             width="20" style="vertical-align:middle;"> GitHub
    </a>
    &nbsp;&nbsp;&nbsp;
        <a href="https://web.facebook.com/abayomiakinseyeemmanuel/" target="_blank" style="text-decoration:none;">
            <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/facebook/facebook-original.svg"
                 width="20" style="vertical-align:middle;"> Facebook
        </a>
    """,
    unsafe_allow_html=True,
)

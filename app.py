import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Expresso Churn Predictor", layout="wide")

st.title("📞 Expresso Churn Predictor")
st.markdown("Enteer customer features below and click **Predict** to estimate churn probability.")

# Load model + metadata
MODEL_PATH = Path("artifacts/model.joblib")
META_PATH = Path("artifacts/metadata.json")

@st.cache_resource
def load_model():
    assert MODEL_PATH.exists(), "Model file not found. Train it by running: python train_model.py"
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_meta():
    if META_PATH.exists():
        with open(META_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

pipe = load_model()
meta = load_meta()
feature_cols = meta.get("features_after_drop", [])
num_cols = set(meta.get("numeric_columns", []))
cat_cols = set(meta.get("categorical_columns", []))

left_cols, right_cols = st.columns(2)
inputs = {}

# Optional defaults for common categoricals (you can edit)
defaults = {
    "TENURE": "2 - 3 years",
    "REGION": "DAKAR",
    "TOP_PACK": "UNKNOWN"
}

for i, col in enumerate(feature_cols):
    with (left_cols if i % 2 == 0 else right_cols):
        if col in cat_cols:
            val = st.text_input(f"{col} (categorical)", value=defaults.get(col, ""))
            inputs[col] = val if val != "" else None
        else:
            val = st.number_input(f"{col} (numeric)", value=0.0, step=1.0, format="%.4f")
            inputs[col] = float(val)

if st.button("Predict"):
    X_input = pd.DataFrame([inputs], columns=feature_cols)
    # Pipeline handles imputation + OHE + model
    if hasattr(pipe, "predict_proba"):
        proba = float(pipe.predict_proba(X_input)[:, 1][0])
        st.metric("Churn probability", f"{proba:.3f}")
        st.progress(proba)
    pred = int(pipe.predict(X_input)[0])
    st.success(f"Predicted class: **{pred}** (1=churn, 0=stay)")
else:
    st.info("Fill the form and click **Predict**.")

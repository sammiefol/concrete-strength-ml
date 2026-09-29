from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------- Page setup ----------
st.set_page_config(page_title="Concrete Strength Predictor", page_icon="🧱")

st.title("Concrete Strength Predictor")
st.write(
    "Enter a concrete mix design and the age at testing to estimate its "
    "compressive strength. The model was trained on 1,005 laboratory tests "
    "from the UCI Concrete Compressive Strength dataset."
)

# ---------- Load the trained model ----------
# The model lives in models/, one folder up from this file (app/).
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "concrete_gb_model.joblib"


@st.cache_resource  # load the model once, not on every click
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

# ---------- Inputs ----------
# Ranges match the training data, so the model is never asked about
# mixes far outside what it has seen.
st.subheader("Mix design (kg per m³ of concrete)")

col1, col2 = st.columns(2)
with col1:
    cement = st.number_input("Cement", min_value=100.0, max_value=540.0, value=350.0, step=5.0)
    slag = st.number_input("Blast furnace slag", min_value=0.0, max_value=360.0, value=0.0, step=5.0)
    fly_ash = st.number_input("Fly ash", min_value=0.0, max_value=200.0, value=0.0, step=5.0)
    water = st.number_input("Water", min_value=120.0, max_value=250.0, value=175.0, step=1.0)
with col2:
    superplasticizer = st.number_input("Superplasticizer", min_value=0.0, max_value=32.0, value=0.0, step=0.5)
    coarse_agg = st.number_input("Coarse aggregate", min_value=800.0, max_value=1150.0, value=1050.0, step=5.0)
    fine_agg = st.number_input("Fine aggregate", min_value=590.0, max_value=995.0, value=750.0, step=5.0)
    age = st.number_input("Age at testing (days)", min_value=1, max_value=365, value=28, step=1)

# ---------- Engineering features ----------
wc_ratio = water / cement
st.write(f"Water–cement ratio: **{wc_ratio:.2f}**")

# ---------- Prediction ----------
if st.button("Predict strength", type="primary"):
    # Same 9 columns, in the same order, as the model was trained on
    mix = pd.DataFrame([{
        "cement": cement,
        "slag": slag,
        "fly_ash": fly_ash,
        "water": water,
        "superplasticizer": superplasticizer,
        "coarse_agg": coarse_agg,
        "fine_agg": fine_agg,
        "wc_ratio": wc_ratio,
        "log_age": np.log(age),
    }])

    strength = model.predict(mix)[0]

    st.metric("Predicted compressive strength", f"{strength:.1f} MPa")
    st.caption(
        "On unseen test mixes, predictions were off by about 5 MPa on average "
        "(RMSE 5.20 MPa). Use this as an early estimate, not a replacement "
        "for laboratory testing."
    )

st.divider()
st.caption(
    "Built by Samuel Folorunso. "
    "[Source code on GitHub](https://github.com/sammiefol/concrete-strength-ml)"
)

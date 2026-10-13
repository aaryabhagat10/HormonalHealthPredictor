
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Load trained model
MODEL_PATH = Path(__file__).parent / "pcos_knn_model.pkl"
model = joblib.load(MODEL_PATH)

st.set_page_config(
    page_title="Hormonal Health Predictor",
    page_icon="🌸",
    layout="centered"
)

st.title("🌸 AI-Based Hormonal Health Pattern Predictor")
st.write(
    "Explore patterns associated with PCOS using "
    "basic health information and machine learning."
)

st.info(
    "This educational prototype estimates patterns in a dataset. "
    "It cannot diagnose PCOS or replace medical advice."
)

st.subheader("Enter your information")

with st.form("pcos_form"):

    age = st.number_input(
        "Age (years)", min_value=18, max_value=60, value=25
    )

    weight = st.number_input(
        "Weight (kg)", min_value=25.0, max_value=200.0,
        value=54.0, step=0.5
    )

    bmi = st.number_input(
        "BMI", min_value=10.0, max_value=60.0,
        value=21.1, step=0.1
    )

    cycle = st.selectbox(
        "Menstrual cycle pattern",
        ["Regular", "Irregular"]
    )

    cycle_length = st.number_input(
        "Cycle length (days)",
        min_value=10, max_value=100, value=28
    )

    weight_gain = st.selectbox(
        "Recent unexplained weight gain?",
        ["No", "Yes"]
    )

    hair_growth = st.selectbox(
        "Excess facial or body hair growth?",
        ["No", "Yes"]
    )

    skin_darkening = st.selectbox(
        "Skin darkening?",
        ["No", "Yes"]
    )

    hair_loss = st.selectbox(
        "Hair loss?",
        ["No", "Yes"]
    )

    pimples = st.selectbox(
        "Pimples or acne?",
        ["No", "Yes"]
    )

    submitted = st.form_submit_button(
        "Analyze Pattern",
        use_container_width=True
    )

if submitted:

    # Match the dataset's encoding
    cycle_value = 2 if cycle == "Regular" else 4

    yes_no = lambda value: 1 if value == "Yes" else 0

    # Preserve the exact training feature names and order
    input_data = pd.DataFrame([{
        "Age (yrs)": age,
        "Weight (Kg)": weight,
        "BMI": bmi,
        "Cycle(R/I)": cycle_value,
        "Cycle length(days)": cycle_length,
        "Weight gain(Y/N)": yes_no(weight_gain),
        "hair growth(Y/N)": yes_no(hair_growth),
        "Skin darkening (Y/N)": yes_no(skin_darkening),
        "Hair loss(Y/N)": yes_no(hair_loss),
        "Pimples(Y/N)": yes_no(pimples)
    }])

    prediction = model.predict(input_data)[0]

    st.subheader("Model output")

    if prediction == 1:
        st.warning(
            "The model identified a pattern associated with "
            "PCOS-positive records in its training dataset."
        )
    else:
        st.success(
            "The model identified a pattern associated with "
            "PCOS-negative records in its training dataset."
        )

    st.caption(
        "This result is not a diagnosis and does not rule out PCOS. "
        "Symptoms overlap with other conditions. Consult a qualified "
        "healthcare professional for appropriate assessment."
    )

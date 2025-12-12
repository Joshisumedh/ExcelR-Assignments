import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Fixed feature order from your dataset
FEATURE_NAMES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

# ----- Load model & scaler -----
@st.cache_resource
def load_resources(model_path="log_reg_model.pkl",
                   scaler_path="scaler.pkl"):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_resources()

st.set_page_config(page_title="Diabetes Predictor", layout="centered")

st.title("Diabetes Risk Predictor (Logistic Regression)")
st.write(
    "Enter patient information below to predict the probability of diabetes (Outcome = 1)."
)

# ----- Sidebar Notes -----
st.sidebar.header("Notes")
st.sidebar.markdown(
    "- Model: Logistic Regression trained on Pima dataset\n"
    "- Zero values were imputed during training\n"
    "- Update inputs and click **Predict**"
)

# Default median-based values
DEFAULTS = {
    "Pregnancies": 1,
    "Glucose": 117,
    "BloodPressure": 70,
    "SkinThickness": 20,
    "Insulin": 79,
    "BMI": 32.0,
    "DiabetesPedigreeFunction": 0.3725,
    "Age": 29,
}

# ----- Collect inputs -----
inputs = {}
st.markdown("### Input patient features")
cols = st.columns(4)

for i, feat in enumerate(FEATURE_NAMES):
    col = cols[i % 4]
    default = DEFAULTS.get(feat, 0)

    if feat in ["Pregnancies", "Age"]:
        inputs[feat] = col.number_input(feat, min_value=0, max_value=120, value=int(default))
    elif feat in ["Glucose", "BloodPressure", "SkinThickness", "Insulin"]:
        inputs[feat] = col.number_input(feat, min_value=0.0, step=1.0, value=float(default))
    elif feat == "BMI":
        inputs[feat] = col.number_input(feat, min_value=0.0, step=0.1, value=float(default))
    else:
        inputs[feat] = col.number_input(feat, min_value=0.0, step=0.001, value=float(default))

st.write("")

# ----- Predict -----
if st.button("Predict"):
    x = np.array([inputs[f] for f in FEATURE_NAMES]).reshape(1, -1)
    x_scaled = scaler.transform(x)

    pred_prob = model.predict_proba(x_scaled)[0, 1]
    pred_class = int(model.predict(x_scaled)[0])

    st.subheader("Prediction")
    st.write(f"**Predicted class:** {'Diabetes (1)' if pred_class == 1 else 'No Diabetes (0)'}")
    st.write(f"**Predicted probability (Outcome = 1):** {pred_prob:.3f}")

    threshold = 0.5
    st.caption(f"Threshold = {threshold}. Adjust interpretation based on clinical need.")

    if pred_prob >= threshold:
        st.success("Model predicts likely diabetes.")
    else:
        st.info("Model predicts unlikely diabetes.")

    # Coefficients / Odds ratios
    st.markdown("---")
    st.subheader("Feature influence (Odds Ratios)")
    coefs = model.coef_[0]
    odds_ratios = np.exp(coefs)

    df_coef = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "coefficient": coefs,
        "odds_ratio": odds_ratios
    }).sort_values(by="odds_ratio", ascending=False)

    st.dataframe(df_coef.style.format({"coefficient": "{:.4f}", "odds_ratio": "{:.3f}"}), height=280)

# ----- Footer -----
st.markdown("---")
st.write("Model info:")
st.write("- Algorithm: Logistic Regression")
st.write("- Educational use only — not for clinical decisions.")

"""A small Streamlit form for trying the saved Ames house-price model."""

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_FOLDER = Path(__file__).resolve().parent
MODEL_FILE = PROJECT_FOLDER / "models" / "ames_price_pipeline.joblib"
METRICS_FILE = PROJECT_FOLDER / "models" / "model_metrics.json"

st.set_page_config(page_title="Ames House Price Estimator", page_icon="🏠")
st.title("Ames House Price Estimator")
st.write("Enter a few details about a home to get a practice estimate of its sale price.")
st.caption("This is a learning project. The estimate is not a professional appraisal.")


@st.cache_resource
def load_model():
    # Streamlit reruns this file when a control changes, so cache the saved model.
    return joblib.load(MODEL_FILE)


if not MODEL_FILE.exists():
    st.error("Model file not found. Run the final section of the notebook first.")
    st.stop()

model = load_model()
metrics = json.loads(METRICS_FILE.read_text(encoding="utf-8"))

with st.form("house_details"):
    st.subheader("House details")

    left, right = st.columns(2)
    overall_quality = left.slider("Overall quality (1–10)", 1, 10, 6)
    living_area = right.number_input("Living area above ground (sq ft)", 300, 6000, 1500, 50)

    left, right = st.columns(2)
    lot_area = left.number_input("Lot area (sq ft)", 1000, 50000, 8000, 250)
    year_built = right.number_input("Year built", 1870, 2010, 1990)

    left, right = st.columns(2)
    basement_area = left.number_input("Basement area (sq ft)", 0, 4000, 900, 50)
    garage_cars = right.number_input("Garage capacity (cars)", 0, 5, 2)

    left, right = st.columns(2)
    bedrooms = left.number_input("Bedrooms above ground", 0, 10, 3)
    full_bathrooms = right.number_input("Full bathrooms", 0, 5, 2)

    left, right = st.columns(2)
    fireplaces = left.number_input("Fireplaces", 0, 5, 0)
    neighborhood = right.selectbox(
        "Neighborhood",
        [
            "Blmngtn", "Blueste", "BrDale", "BrkSide", "ClearCr", "CollgCr",
            "Crawfor", "Edwards", "Gilbert", "IDOTRR", "MeadowV", "Mitchel",
            "NAmes", "NoRidge", "NPkVill", "NridgHt", "NWAmes", "OldTown",
            "SWISU", "Sawyer", "SawyerW", "Somerst", "StoneBr", "Timber", "Veenker",
        ],
    )

    quality_choices = ["Ex", "Gd", "TA", "Fa", "Po"]
    left, right = st.columns(2)
    kitchen_quality = left.selectbox("Kitchen quality", quality_choices, index=2)
    exterior_quality = right.selectbox("Exterior quality", quality_choices, index=2)

    left, right = st.columns(2)
    basement_quality = left.selectbox("Basement quality", quality_choices, index=2)
    central_air = right.selectbox("Central air", ["Y", "N"])

    estimate_button = st.form_submit_button("Estimate price")

if estimate_button:
    # These names must match the columns the notebook used to train the model.
    house = pd.DataFrame([{
        "OverallQual": overall_quality,
        "GrLivArea": living_area,
        "GarageCars": garage_cars,
        "TotalBsmtSF": basement_area,
        "YearBuilt": year_built,
        "FullBath": full_bathrooms,
        "BedroomAbvGr": bedrooms,
        "LotArea": lot_area,
        "Fireplaces": fireplaces,
        "KitchenQual": kitchen_quality,
        "Neighborhood": neighborhood,
        "ExterQual": exterior_quality,
        "BsmtQual": basement_quality,
        "CentralAir": central_air,
    }])
    estimated_price = model.predict(house)[0]
    st.success(f"Estimated sale price: **${estimated_price:,.0f}**")

st.header("How the model did")
st.write(
    "The notebook held back 20% of the sales while comparing models. "
    "These scores come from those homes, which the models did not train on."
)
first, second, third = st.columns(3)
first.metric("Selected model", metrics["best_model"])
second.metric("Typical error (MAE)", f"${metrics['holdout_metrics']['MAE']:,.0f}")
third.metric("R²", f"{metrics['holdout_metrics']['R2']:.3f}")

comparison_file = PROJECT_FOLDER / "data" / "processed" / "model_comparison.csv"
if comparison_file.exists():
    comparison = pd.read_csv(comparison_file).sort_values("RMSE")
    st.subheader("Models compared (lower RMSE is better)")
    st.dataframe(comparison, hide_index=True)
    st.bar_chart(comparison.set_index("Model")["RMSE"])

importance_file = PROJECT_FOLDER / "data" / "processed" / "feature_importance.csv"
if importance_file.exists():
    importance = pd.read_csv(importance_file).head(10).sort_values("Importance")
    st.subheader("Features the selected model relied on")
    st.bar_chart(importance.set_index("Feature")["Importance"])
    st.caption("Importance shows model reliance; it does not prove that a feature causes a price change.")

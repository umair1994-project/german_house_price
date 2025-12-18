# ----------------------------
# 1. Imports
# ----------------------------
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
from datetime import datetime


st.set_page_config(
    page_title="German Housing Price Predictor",
    layout="wide"
)

# ----------------------------
# 3. Load Model
# ----------------------------
MODEL_PATH = "models/german_price_model_optimized.pkl"
model = joblib.load(MODEL_PATH)

CURRENT_YEAR = datetime.now().year


st.title("German Housing Price Predictor")
st.write(
    "Predict the **purchase price** of German real estate using a trained "
    "machine learning model (Gradient Boosting)."
)

st.sidebar.header("Property Details")

def user_input_features():

    obj_regio1 = st.sidebar.selectbox(
        "Region",
        [
            "Bayern",
            "Berlin",
            "Brandenburg",
            "Bremen",
            "Hamburg",
            "Hessen",
            "Mecklenburg_Vorpommern",
            "Niedersachsen",
            "Nordrhein_Westfalen",
            "Rheinland_Pfalz",
            "Saarland",
            "Sachsen",
            "Sachsen_Anhalt",
            "Schleswig_Holstein",
         ]
    )

    obj_heatingType = st.sidebar.selectbox(
        "Heating Type",
        [
            "central_heating",
            "combined_heat_and_power_plant",
            "district_heating",
            "electric_heating",
            "floor_heating",
            "gas_heating",
            "heat_pump",
            "night_storage_heater",
            "no_information",
            "oil_heating",
            "self_contained_central_heating",
            "solar_heating",
            "stove_heating",
            "wood_pellet_heating",
            "(blank)"
        ]
    )

    obj_condition = st.sidebar.selectbox(
        "Property Condition",
        [
            "first_time_use",
            "first_time_use_after_refurbishment",
            "fully_renovated",
            "mint_condition",
            "modernized",
            "need_of_renovation",
            "negotiable",
            "no_information",
            "refurbished",
            "ripe_for_demolition",
            "well_kept"
        ]
    )

    obj_immotype = st.sidebar.selectbox(
        "Property Type",
        [
            "apartment",
            "house",
            "other",
            "no_information"
        ]
    )

    obj_barrierFree = st.sidebar.selectbox(
        "Barrier Free",
        ["yes", "no", "missing"]
    )

    obj_livingSpace = st.sidebar.number_input(
        "Living Space (sqm)", 10, 1000, 75
    )

    obj_noRooms = st.sidebar.number_input(
        "Number of Rooms", 1, 20, 3
    )

    obj_yearConstructed = st.sidebar.number_input(
        "Year Constructed", 1800, CURRENT_YEAR, 1995
    )

    building_age = CURRENT_YEAR - obj_yearConstructed

    return pd.DataFrame([{
        "obj_regio1": obj_regio1,
        "obj_heatingType": obj_heatingType,
        "obj_condition": obj_condition,
        "obj_immotype": obj_immotype,
        "obj_barrierFree": obj_barrierFree,
        "obj_livingSpace": obj_livingSpace,
        "obj_noRooms": obj_noRooms,
        "obj_yearConstructed": obj_yearConstructed,
        "building_age": building_age
    }])

input_df = user_input_features()

st.subheader("Predicted Purchase Price")

prediction = model.predict(input_df)[0]

st.metric(
    label="Estimated Price",
    value=f"€ {prediction:,.0f}"
)



EDA_IMAGES = {
    
    "Price per Square Meter": "output/price_per_sqm.png",
    "Top 10 Feature Importances": "output/top_10_feature_importances.png"
}

for title, path in EDA_IMAGES.items():
    st.write(f"### {title}")
    try:
        img = Image.open(path)
        st.image(img)
    except:
        st.warning(f"Image not found: {path}")





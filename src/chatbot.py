# src/chatbot.py
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import os

# --- Configuration ---
MODEL_DIR = "models/"
BEST_MODEL_FILE = os.path.join(MODEL_DIR, "best_model.joblib")
SEGMENTED_DATA_PATH = "data/segmented_data.csv"

# Default statistical values
DEFAULT_FEATURES = {
    "Bedroom": 4.0,
    "Bathroom": 3.0,
    "Floors": 2.5,
    "Parking": 1.0,
    "House Age": 5.0,
    "Amenities Count": 10.0,
    "Build Area (Aana)": 6.0,
    "Price per Bedroom": 1_000_000.0,
    "City": "Kathmandu",
    "Face": "East",
}

# --- Load ML Artifacts ---
@st.cache_resource
def load_artifacts():
    """Load fitted pipeline and cluster map."""
    try:
        pipeline = joblib.load(BEST_MODEL_FILE)  # Contains fitted preprocessor + model
        df_segmented = pd.read_csv(SEGMENTED_DATA_PATH)

        cluster_map = (
            df_segmented
            .groupby("Cluster")[["Bedroom", "Build Area (Aana)"]]
            .mean()
            .reset_index()
        )

        return pipeline, cluster_map

    except Exception as e:
        return None, str(e)


# --- Feature Preparation ---
def prepare_input_df(bhk, aana, city, face, bathrooms, cluster_map):
    features = DEFAULT_FEATURES.copy()
    features.update({
        "Bedroom": float(bhk),
        "Bathroom": float(bathrooms),
        "Build Area (Aana)": float(aana),
        "City": city,
        "Face": face,
    })

    # Infer cluster safely
    if cluster_map is not None:
        user_point = np.array([features["Bedroom"], features["Build Area (Aana)"]])
        distances = []
        for _, row in cluster_map.iterrows():
            cluster_point = np.array([row["Bedroom"], row["Build Area (Aana)"]])
            distances.append((np.linalg.norm(user_point - cluster_point), row["Cluster"]))
        features["Cluster"] = min(distances, key=lambda x: x[0])[1]
    else:
        features["Cluster"] = 0

    feature_order = [
        "Bedroom", "Bathroom", "Floors", "Parking",
        "House Age", "Amenities Count",
        "Build Area (Aana)", "Price per Bedroom",
        "City", "Face", "Cluster"
    ]
    return pd.DataFrame([features])[feature_order]


# --- Prediction ---
def predict_price(input_df, pipeline):
    """Predict price using the fitted pipeline."""
    try:
        prediction = pipeline.predict(input_df)[0]  # pipeline handles data preprocessing
        return prediction, None
    except Exception as e:
        return None, str(e)


# --- Format Price ---
def format_price(price):
    """Format price in Lakhs or Crores for display."""
    if price >= 10_000_000:
        return f"Rs. {price / 10_000_000:.2f} Crore"
    elif price >= 100_000:
        return f"Rs. {price / 100_000:.2f} Lakh"
    return f"Rs. {price:,.0f}"

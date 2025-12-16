# src/chatbot.py
import re
import joblib
import pandas as pd
import numpy as np
import streamlit as st # <-- CRITICAL FIX: ADD THIS IMPORT
import os

# --- Configuration ---
MODEL_DIR = 'models/'
BEST_MODEL_FILE = MODEL_DIR + 'best_model.joblib'
PREPROCESSOR_FILE = MODEL_DIR + 'preprocessor.joblib'
SEGMENTED_DATA_PATH = 'data/segmented_data.csv'

# Default values based on median/mode of the training data 
DEFAULT_FEATURES = {
    'Bedroom': 4.0, 'Bathroom': 3.0, 'Floors': 2.5, 'Parking': 1.0, 
    'House Age': 5.0, 'Amenities Count': 10.0, 'Build Area (Aana)': 6.0,
    'Price per Bedroom': 1000000.0, 
    'City': 'Kathmandu', 'Face': 'East' 
}

# --- Load Artifacts ---

@st.cache_resource
def load_artifacts():
    """Loads the trained model, preprocessor, and segmentation data."""
    if not os.path.exists(BEST_MODEL_FILE):
        return f"Error: Best model file not found at {BEST_MODEL_FILE}. Run src/models.py first.", None, None

    try:
        model = joblib.load(BEST_MODEL_FILE)
        preprocessor = joblib.load(PREPROCESSOR_FILE)
        df_segmented = pd.read_csv(SEGMENTED_DATA_PATH)
        
        # Calculate cluster map for new predictions
        cluster_map = df_segmented.groupby('Cluster')[['Bedroom', 'Build Area (Aana)']].mean().reset_index()
        return model, preprocessor, cluster_map
    except FileNotFoundError as e:
        return f"Error loading model artifacts: {e}", None, None

# --- NLP Feature Extraction ---

def extract_features_from_query(query: str, cluster_map: pd.DataFrame) -> pd.DataFrame:
    """
    Parses a natural language query using regex and returns a DataFrame 
    ready for prediction.
    """
    query = query.lower()
    features = DEFAULT_FEATURES.copy()

    # 1. City (Location)
    cities = ['kathmandu', 'lalitpur', 'bhaktapur', 'pokhara', 'chitwan']
    for city in cities:
        if city in query:
            features['City'] = city.title()
            break
            
    # 2. Bedrooms (BHK)
    bhk_match = re.search(r'(\d+)\s*bhk|\s*(\d+)\s*bed(room)?s?|(\d+)\s*b/r', query)
    if bhk_match:
        features['Bedroom'] = float(bhk_match.group(1) or bhk_match.group(2) or bhk_match.group(4))

    # 3. Bathrooms
    bath_match = re.search(r'(\d+)\s*bath(room)?s?', query)
    if bath_match:
        features['Bathroom'] = float(bath_match.group(1))

    # 4. Land/Built Area (Aana)
    area_match = re.search(r'(\d+(?:\.\d+)?)\s*aana', query)
    if area_match:
        features['Build Area (Aana)'] = float(area_match.group(1))

    # 5. Floors
    floor_match = re.search(r'(\d+)\s*floor', query)
    if floor_match:
        features['Floors'] = float(floor_match.group(1))

    # 6. Face (Direction)
    directions = ['north', 'south', 'east', 'west']
    for direction in directions:
        if direction in query:
            features['Face'] = direction.title()
            break
            
    # 7. Recalculate 'Price per Bedroom' (Crucial Feature)
    # Use a default/median price for calculation, although the unscaled price value 
    # itself is less important than the Bedroom/Area features for clustering.
    # The actual feature is calculated by the preprocessor based on the median price.
    
    # 8. Infer the Market Cluster (Segmentation)
    user_data = np.array([features['Bedroom'], features['Build Area (Aana)']])
    
    # Simple Euclidean distance to find the closest cluster profile
    distances = []
    for index, row in cluster_map.iterrows():
        cluster_data = np.array([row['Bedroom'], row['Build Area (Aana)']])
        dist = np.linalg.norm(user_data - cluster_data)
        distances.append((dist, row['Cluster']))
    
    features['Cluster'] = min(distances, key=lambda x: x[0])[1]

    # Reorder the features to match the training data order (excluding Price)
    feature_order = [
        'Bedroom', 'Bathroom', 'Floors', 'Parking', 'House Age', 
        'Amenities Count', 'Build Area (Aana)', 'Price per Bedroom',
        'City', 'Face', 'Cluster'
    ]
    
    input_df = pd.DataFrame([features])[feature_order]
    
    return input_df

# --- Prediction Function ---

def predict_price(query: str):
    """
    Runs the full prediction pipeline: loads artifacts, parses query, 
    and predicts price.
    """
    model, preprocessor, cluster_map = load_artifacts()

    if isinstance(model, str):
        return model, None, None

    # 1. Extract features from text
    input_df = extract_features_from_query(query, cluster_map)
    
    # 2. Predict Price (using the full pipeline)
    prediction = model.predict(input_df)[0]

    return prediction, input_df, cluster_map
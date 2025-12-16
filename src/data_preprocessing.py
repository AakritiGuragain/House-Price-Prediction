# src/data_preprocessing.py
import pandas as pd
import numpy as np
import re
from datetime import datetime

# --- Configuration ---
DATA_PATH = '/Users/aakritiguragain/Desktop/House Price Prediction/data/2020-4-27.csv'

def clean_and_engineer_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs data cleaning, missing value imputation, and feature engineering.

    Args:
        df: The raw DataFrame loaded from the CSV file.

    Returns:
        The cleaned and feature-engineered DataFrame.
    """
    print(f"Starting preprocessing for dataset with {len(df)} records...")
    
    # --- 1. Basic Cleaning and Type Conversion ---
    
    # Replace common placeholder strings with NaN
    df.replace(['N/A', 'NA', 'na', 'N/a'], np.nan, inplace=True)

    # Convert essential columns to numeric, coercing errors to NaN
    numeric_cols = ['Price', 'Bedroom', 'Bathroom', 'Floors', 'Parking', 'Year']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # --- 2. Missing Value Imputation (Strategy: Median) ---
    
    # Impute Bedroom, Bathroom, Parking (These are important numerical features)
    df['Bedroom'].fillna(df['Bedroom'].median(), inplace=True)
    df['Bathroom'].fillna(df['Bathroom'].median(), inplace=True)
    df['Parking'].fillna(df['Parking'].median(), inplace=True)
    
    # Impute 'Floors' and 'Year' (as planned)
    median_floors = df['Floors'].median()
    df['Floors'].fillna(median_floors, inplace=True)
    
    median_year = df['Year'].median()
    df['Year'].fillna(median_year, inplace=True)
    
    print("Missing values in key numerical columns imputed with median.")
    
    # --- 3. Feature Engineering ---

    # 3.1 Feature: House Age
    current_year = datetime.now().year
    df['House Age'] = current_year - df['Year']
    # Ensure Age is not negative (in case of future or erroneous years)
    df['House Age'] = np.maximum(df['House Age'], 0)
    print("'House Age' feature created.")

    # 3.2 Feature: Amenities Count
    df['Amenities'].fillna("[]", inplace=True)
    def count_amenities(amenities_str):
        try:
            # Safely count the number of quoted strings in the list representation
            return len(re.findall(r"'([^']*)'", amenities_str))
        except:
            return 0

    df['Amenities Count'] = df['Amenities'].apply(count_amenities)
    print("'Amenities Count' feature created.")

    # 3.3 Feature: Build Area (Numeric Extraction & Imputation)
    def extract_area_in_aana(area_str):
        if pd.isna(area_str) or not isinstance(area_str, str):
            return np.nan
        # Pattern: Ropani-Aana-Paisa-Daam (e.g., 1-0-0-0 Aana)
        match = re.search(r'(\d+)-(\d+)-(\d+)-(\d+)\s*Aana', area_str)
        if match:
            # Conversion: 1 Ropani = 16 Aana
            r = int(match.group(1)) # Ropani
            a = int(match.group(2)) # Aana
            total_aana = (r * 16) + a
            return total_aana
        
        # Fallback for simpler 'X Aana' or 'X Sq. Feet' cases (focusing on Aana for consistency)
        # Assuming most listings use the full R-A-P-D format for land size.
        return np.nan

    df['Build Area (Aana)'] = df['Build Area'].apply(extract_area_in_aana)

    # Impute missing 'Build Area (Aana)' with its median
    median_build_area = df['Build Area (Aana)'].median()
    df['Build Area (Aana)'].fillna(median_build_area, inplace=True)
    print("'Build Area (Aana)' feature created and imputed.")
    
    # 3.4 Feature: Price per Bedroom
    # Use the cleaned 'Price' and imputed 'Bedroom'
    # Adjust bedroom count to 1 if 0 to prevent division by zero, assuming a minimum living space
    df['Adjusted Bedroom'] = df['Bedroom'].apply(lambda x: x if x >= 1 else 1)
    df['Price per Bedroom'] = df['Price'] / df['Adjusted Bedroom']
    df.drop(columns=['Adjusted Bedroom'], inplace=True)
    print("'Price per Bedroom' feature created.")

    # --- 4. Final Cleanup: Drop Original and Redundant Features ---
    
    # Identify categorical/text columns that need encoding later
    df_final = df[[
        'Title', 'Address', 'City', 'Face', 'Posted', 'Price', # Keep Price (Target) and key Categoricals
        'Bedroom', 'Bathroom', 'Floors', 'Parking',          # Keep cleaned originals
        'House Age', 'Amenities Count', 'Build Area (Aana)', # Keep new engineered features
        'Price per Bedroom'                                  # Keep new engineered feature
    ]].copy()
    
    print(f"\nPreprocessing complete. Final dataset shape: {df_final.shape}")
    return df_final

if __name__ == '__main__':
    try:
        # Load the dataset (adjust path if needed when running directly)
        raw_df = pd.read_csv(DATA_PATH)
        
        # Run the cleaning and engineering process
        processed_df = clean_and_engineer_data(raw_df)
        
        # Display the results for verification
        print("\n--- Processed Data Head (Verification) ---")
        print(processed_df.head())
        print("\n--- New Feature Statistics ---")
        print(processed_df[['House Age', 'Amenities Count', 'Build Area (Aana)', 'Price per Bedroom']].describe())

    except FileNotFoundError:
        print(f"Error: The file {DATA_PATH} was not found.")
        print("Please ensure you have placed '2020-4-27.csv' inside the 'data/' folder.")
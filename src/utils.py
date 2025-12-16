# src/utils.py
import pandas as pd
import numpy as np
import re
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os

# --- Configuration ---
RAW_DATA_PATH = 'data/2020-4-27.csv'
PROCESSED_DATA_PATH = 'data/processed_data.csv'

@st.cache_data
def clean_and_engineer_data():
    """Performs data cleaning, imputation, and feature engineering."""
    try:
        df = pd.read_csv(RAW_DATA_PATH)
    except FileNotFoundError:
        return pd.DataFrame() 

    df.replace(['N/A', 'NA', 'na', 'N/a', 'nan'], np.nan, inplace=True)
    numeric_cols = ['Price', 'Bedroom', 'Bathroom', 'Floors', 'Parking', 'Year']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    for col in ['Bedroom', 'Bathroom', 'Parking', 'Floors', 'Year']:
        df[col].fillna(df[col].median(), inplace=True)
    
    current_year = datetime.now().year
    df['House Age'] = np.maximum(current_year - df['Year'], 0)

    df['Amenities'].fillna("[]", inplace=True)
    def count_amenities(amenities_str):
        try:
            return len(re.findall(r"'([^']*)'", amenities_str))
        except:
            return 0
    df['Amenities Count'] = df['Amenities'].apply(count_amenities)

    def extract_area_in_aana(area_str):
        if pd.isna(area_str) or not isinstance(area_str, str): return np.nan
        match = re.search(r'(\d+)-(\d+)-(\d+)-(\d+)\s*Aana', area_str)
        if match:
            r = int(match.group(1)) # Ropani (1 Ropani = 16 Aana)
            a = int(match.group(2)) # Aana
            return (r * 16) + a
        return np.nan

    df['Build Area (Aana)'] = df['Build Area'].apply(extract_area_in_aana)
    df['Build Area (Aana)'].fillna(df['Build Area (Aana)'].median(), inplace=True)
    
    df['Adjusted Bedroom'] = df['Bedroom'].apply(lambda x: x if x >= 1 else 1)
    df['Price per Bedroom'] = df['Price'] / df['Adjusted Bedroom']
    df.drop(columns=['Adjusted Bedroom'], inplace=True, errors='ignore')

    df_final = df[[
        'Title', 'Address', 'City', 'Face', 'Posted', 'Price', 
        'Bedroom', 'Bathroom', 'Floors', 'Parking',          
        'House Age', 'Amenities Count', 'Build Area (Aana)', 
        'Price per Bedroom'                                  
    ]].copy()
    
    df_final.to_csv(PROCESSED_DATA_PATH, index=False)
    
    return df_final

def generate_eda_plots(df: pd.DataFrame):
    """Generates and displays key EDA plots using Matplotlib/Seaborn in Streamlit."""
    st.subheader(" Exploratory Data Analysis (EDA) Visualizations")
    sns.set_style("whitegrid")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Price vs. Bedrooms", "City Price Distribution", "Build Area vs. Price", "Correlation Heatmap"])

    with tab1:
        st.markdown("#### Relationship: Price Distribution by Number of Bedrooms")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x='Bedroom', y='Price', data=df[df['Bedroom'] <= 10], ax=ax)
        ax.set_title('Price Distribution by Number of Bedrooms')
        ax.set_xlabel('Number of Bedrooms')
        ax.set_ylabel('Price (Log Scale)')
        ax.set_yscale('log')
        st.pyplot(fig)

    with tab2:
        st.markdown("#### Categorical Analysis: City-wise Price Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        city_order = df.groupby('City')['Price'].median().sort_values(ascending=False).index
        sns.boxplot(x='City', y='Price', data=df, order=city_order, ax=ax)
        ax.set_title('City-wise Price Distribution')
        ax.set_xlabel('City')
        ax.set_ylabel('Price (Log Scale)')
        plt.xticks(rotation=45, ha='right')
        ax.set_yscale('log')
        plt.tight_layout()
        st.pyplot(fig)

    with tab3:
        st.markdown("#### Relationship: Price vs. Built-up Area (Aana)")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(x='Build Area (Aana)', y='Price', data=df[df['Build Area (Aana)'] < 50], ax=ax)
        ax.set_title('Price vs. Built-up Area (Aana)')
        ax.set_xlabel('Built-up Area (Aana)')
        ax.set_ylabel('Price')
        ax.ticklabel_format(style='plain', axis='y')
        st.pyplot(fig)

    with tab4:
        st.markdown("#### Correlation Heatmap (Justifying Model Inputs)")
        numeric_cols = ['Price', 'Bedroom', 'Bathroom', 'Floors', 'Parking', 
                        'House Age', 'Amenities Count', 'Build Area (Aana)', 'Price per Bedroom']
        corr_matrix = df[numeric_cols].corr()

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, linecolor='black', ax=ax)
        ax.set_title('Correlation Heatmap')
        st.pyplot(fig)
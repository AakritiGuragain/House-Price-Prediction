# main_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Import logic modules
from src.utils import clean_and_engineer_data, generate_eda_plots, PROCESSED_DATA_PATH
from src.clustering import find_optimal_k, run_kmeans_clustering, PROCESSED_DATA_PATH as CLUSTER_IN_PATH
from src.chatbot import load_artifacts, prepare_input_df, predict_price, format_price

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="House Price Prediction System")

# --- Sidebar Navigation ---
st.sidebar.title("Project Navigation")
page = st.sidebar.radio("Go to:", [
    "1. Introduction & Setup",
    "2. Data Preprocessing & EDA",
    "3. K-Means Clustering (Segmentation)",
    "4. ML Prediction Models & Chatbot"
])

# --- PHASE 1: Introduction & Setup ---
if page == "1. Introduction & Setup":
    st.title("House Price Prediction System (Final Year Project)")
    st.markdown("""
    This application demonstrates a complete machine learning pipeline, from raw data cleaning 
    to deployment of an **Intelligent Price Prediction Chatbot**.
    """)
    
    if not os.path.exists('data/2020-4-27.csv'):
        st.error("Error: Raw data file 'data/2020-4-27.csv' not found.")
        st.info("Please ensure your data file is in the `data/` directory.")
        
    st.subheader("Data Files and Artifacts Status")
    
    processed_status = "Found" if os.path.exists(PROCESSED_DATA_PATH) else "❌ Not Found (Run Step 2)"
    st.markdown(f"- Processed Data (`data/processed_data.csv`): {processed_status}")
    
    segmented_status = "Found" if os.path.exists('data/segmented_data.csv') else "❌ Not Found (Run Step 3)"
    st.markdown(f"- Segmented Data (`data/segmented_data.csv`): {segmented_status}")
    
    model_status = "Found" if os.path.exists('models/best_model.joblib') else "❌ Not Found (Run Step 4)"
    st.markdown(f"- Trained Model (`models/best_model.joblib`): {model_status}")


# --- PHASE 2: Data Preprocessing & EDA (Step 3 & 4) ---
elif page == "2. Data Preprocessing & EDA":
    st.title("Step 3 & 4: Data Preprocessing and EDA")
    
    st.subheader("Data Cleaning and Feature Engineering")
    if st.button("Run Preprocessing"):
        with st.spinner("Processing data..."):
            df_processed = clean_and_engineer_data()
        
        if not df_processed.empty:
            st.success(f"Preprocessing complete. Final dataset shape: {df_processed.shape}")
            st.dataframe(df_processed.head())
        else:
            st.error("Preprocessing failed. Check file paths.")
            
    try:
        df_processed = pd.read_csv(PROCESSED_DATA_PATH)
        st.markdown("---")
        generate_eda_plots(df_processed)
    except FileNotFoundError:
        st.info("Processed data not available. Please run the preprocessing step above.")


# --- PHASE 3: K-Means Clustering (Step 5) ---
elif page == "3. K-Means Clustering (Segmentation)":
    st.title("Step 5: K-Means Clustering (Market Segmentation)")
    
    try:
        df_processed = pd.read_csv(CLUSTER_IN_PATH)
    except FileNotFoundError:
        st.error("Cannot load processed data. Please complete Step 2 first.")
        st.stop()

    wcss, X_scaled, scaler = find_optimal_k(df_processed)
    max_k = 10
    
    st.subheader("3.1 Elbow Method for Optimal K")
    fig_elbow, ax_elbow = plt.subplots(figsize=(8, 4))
    ax_elbow.plot(range(1, max_k + 1), wcss, marker='o', linestyle='--')
    ax_elbow.set_title('Elbow Method for Optimal K (Market Segments)')
    ax_elbow.set_xlabel('Number of Clusters (K)')
    ax_elbow.set_ylabel('WCSS (Inertia)')
    st.pyplot(fig_elbow)
    st.info("The optimal 'K' is chosen where the drop in WCSS starts to flatten (the 'elbow').")

    st.subheader("3.2 Apply K-Means and Profile Clusters")
    optimal_k = st.slider("Select Optimal K based on the graph:", 2, 5, 3)
    
    if st.button(f"Run K-Means with K={optimal_k}"):
        with st.spinner(f"Clustering data into {optimal_k} segments..."):
            df_segmented, cluster_profile = run_kmeans_clustering(df_processed, optimal_k)
        
        st.success("Clustering complete. Segmented data saved.")
        st.markdown(f"#### Cluster Profile (K={optimal_k})")
        
        profile_display = cluster_profile.rename(columns={'Build Area (Aana)': 'Area (Aana)', 'Price per Bedroom': 'Price/Bdr'})
        st.dataframe(profile_display)
        
        fig_box, ax_box = plt.subplots(figsize=(8, 4))
        sns.boxplot(x='Cluster', y='Price', data=df_segmented, ax=ax_box)
        ax_box.set_title(f'Price Distribution by {optimal_k} Market Segments')
        ax_box.set_xlabel('Cluster Label')
        ax_box.set_ylabel('Price (Log Scale)')
        ax_box.set_yscale('log')
        st.pyplot(fig_box)


# --- PHASE 4: ML Prediction Models & Chatbot (Step 6-9) ---
elif page == "4. ML Prediction Models & Chatbot":
    st.header("Step 6-9: ML Prediction Models & Chatbot System")
    
    # --- 4.1 Model Performance ---
    st.subheader("4.1 Model Performance Evaluation (R² and RMSE)")
    try:
        results_df = pd.read_csv('models/model_performance_results.csv')
        results_df['RMSE'] = results_df['RMSE'].apply(lambda x: f'{x:,.2f}')
        
        st.dataframe(results_df.style.highlight_max(subset=['R2 Score'], axis=0, color='lightgreen'), use_container_width=True)
        st.markdown(f"**Best Model:** The **{results_df.iloc[0]['Model']}** achieved the highest $R^2$ of **{results_df.iloc[0]['R2 Score']:.4f}**, confirming its suitability for deployment.")
    except FileNotFoundError:
        st.error("Model performance file not found. Please run `python src/models.py` first.")

    # --- Load Pipeline and Artifacts ---
    pipeline, cluster_map_or_error = load_artifacts()
    if pipeline is None:
        st.error(f"Failed to load pipeline: {cluster_map_or_error}")
        st.stop()

    # --- 4.2 Prediction Chatbot ---
    st.subheader("4.2 Intelligent Price Prediction Chatbot")
    st.markdown("""
    **System Instructions:** Enter your house requirements using the form below. 
    The system extracts features, infers the market segment (Cluster), and uses the trained Random Forest model to provide a forecast.
    """)

    # --- Simple Input Fields ---
    bhk = st.number_input("Number of Bedrooms (BHK)", 1, 15, 4)
    bathrooms = st.number_input("Number of Bathrooms", 1, 10, 3)
    aana = st.number_input("Land Area (Aana)", 1.0, 100.0, 6.0, step=0.5)
    city = st.selectbox("City", ["Kathmandu", "Lalitpur", "Bhaktapur", "Pokhara", "Chitwan", "Other"])
    face = st.selectbox("House Face", ["East", "North", "West", "South", "North-East", "Other"])

    if st.button("Predict Price"):
        # Prepare input dataframe
        input_df = prepare_input_df(bhk, aana, city, face, bathrooms, cluster_map_or_error)

        # Make prediction using fitted pipeline
        prediction, error = predict_price(input_df, pipeline)

        if error:
            st.error(f"Prediction Error: {error}")
        else:
            st.success("Prediction Complete!")
            prediction_lakhs = prediction / 100_000
            st.markdown(f"## **Predicted Price:** Rs. {prediction:,.0f} (Approx. **{prediction_lakhs:,.2f} Lakhs**)")

            # Display features and cluster info
            col1, col2 = st.columns(2)

            with col1:
                st.info("Features Extracted and Cluster Inferred:")
                display_features = input_df[['City', 'Bedroom', 'Bathroom', 'Build Area (Aana)', 'Floors', 'Cluster']].T
                display_features.columns = ['Input']
                st.dataframe(display_features)

            with col2:
                st.warning("Cluster Profile Details:")
                inferred_cluster_id = input_df['Cluster'].iloc[0]
                inferred_profile = cluster_map_or_error[cluster_map_or_error['Cluster'] == inferred_cluster_id].T
                inferred_profile.columns = [f'Cluster {inferred_cluster_id} Profile']
                st.dataframe(inferred_profile.drop(index='Cluster', errors='ignore'))
                st.markdown(f"**Conclusion:** The request falls into **Cluster {inferred_cluster_id}**, guiding the model toward the expected market price range.")

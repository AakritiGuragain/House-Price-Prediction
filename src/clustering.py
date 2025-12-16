# src/clustering.py (Complete Code)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import streamlit as st

# --- Configuration ---
PROCESSED_DATA_PATH = 'data/processed_data.csv'
SEGMENTED_DATA_PATH = 'data/segmented_data.csv'

#selected features for clustering
CLUSTERING_FEATURES = [
    'Bedroom', 'Bathroom', 'Floors', 'Parking',
    'House Age', 'Amenities Count', 'Build Area (Aana)',
    'Price per Bedroom' 
]
#This function finds best K using elbow method
def find_optimal_k(df: pd.DataFrame, max_k=10):
    """Performs the Elbow Method to find the optimal number of clusters (K)."""
    # This is the function the main_app is trying to import
    X = df[CLUSTERING_FEATURES].copy() #Only keep columns needed for clustering
    scaler = StandardScaler() #All features are put on same scale
    X_scaled = scaler.fit_transform(X)

    wcss = [] #WCSS = how bad clustering is (we want to minimize this)
    for k in range(1, max_k + 1):
        # We must use n_init=10 explicitly to silence a future warning in sklearn
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10) 
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
    
    return wcss, X_scaled, scaler #draws elbow plot
#elbow plot ma jun ma gayera suddern bend hunxa that is our k


@st.cache_data
def run_kmeans_clustering(df: pd.DataFrame, optimal_k: int):
    """Applies K-Means clustering and returns the segmented DataFrame and cluster profile."""
    X = df[CLUSTERING_FEATURES].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X) #scale features
    
    kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled) #Each house gets a cluster number
    
    # Calculate the mean features for each cluster (the "profile")
    cluster_profile = df.groupby('Cluster')[['Price'] + CLUSTERING_FEATURES].mean()
    
    df.to_csv(SEGMENTED_DATA_PATH, index=False)
    
    cluster_profile['Price (in Lakhs)'] = cluster_profile['Price'] / 100000 
    
    return df, cluster_profile.drop(columns=['Price']).round(2)
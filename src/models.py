# src/models.py
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- Models ---
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

# --- Configuration ---
SEGMENTED_DATA_PATH = 'data/segmented_data.csv'
MODEL_DIR = 'models/' 
RESULTS_FILE = MODEL_DIR + 'model_performance_results.csv'
BEST_MODEL_FILE = MODEL_DIR + 'best_model.joblib'


def evaluate_model(y_true, y_pred, model_name):
    """Calculates and returns key regression metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'Model': model_name,
        'RMSE': rmse,
        'MAE': mae,
        'R2 Score': r2
    }


def run_ml_pipeline():
    """Loads data, prepares features, trains models, and evaluates performance."""
    print("--- Starting ML Model Training and Evaluation ---")
    
    # 1. Load Data
    try:
        df = pd.read_csv(SEGMENTED_DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Segmented data not found at {SEGMENTED_DATA_PATH}. Please run src/clustering.py first.")
        return

    df.drop(columns=['Title', 'Address', 'Posted'], inplace=True, errors='ignore')
    
    # 2. Define Target and Features
    X = df.drop('Price', axis=1)
    y = df['Price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Dataset split: Train={len(X_train)}, Test={len(X_test)}")

    # 3. Define Preprocessing Steps
    numerical_features = [
        'Bedroom', 'Bathroom', 'Floors', 'Parking', 'House Age', 
        'Amenities Count', 'Build Area (Aana)', 'Price per Bedroom'
    ]
    categorical_features = ['City', 'Face', 'Cluster']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )
    
    # 4. Define Models
    models = {
        "Random Forest Regressor": RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1, max_depth=15),
        "KNN Regressor": KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
    }
    
    results = []
    best_r2 = -np.inf
    best_model = None
    best_model_name = ""

    # 5. Train and Evaluate Models
    print("\n5. Training and Evaluating Models...")
    for name, model in models.items():
        print(f"\nTraining: {name}")
        
        # Create pipeline with fitted preprocessor + model
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        metrics = evaluate_model(y_test, y_pred, name)
        results.append(metrics)
        print(f"  RMSE: {metrics['RMSE']:.2f}, R2: {metrics['R2 Score']:.4f}")
        
        if metrics['R2 Score'] > best_r2:
            best_r2 = metrics['R2 Score']
            best_model = pipeline
            best_model_name = name

    # 6. Save Results and Best Model
    results_df = pd.DataFrame(results).sort_values(by='R2 Score', ascending=False)
    results_df.to_csv(RESULTS_FILE, index=False)
    print(f"\nAll model results saved to {RESULTS_FILE}")
    
    joblib.dump(best_model, BEST_MODEL_FILE)
    print(f"Best Model ({best_model_name}) saved to {BEST_MODEL_FILE}")
    
    # 7. Feature Importance (for Random Forest)
    if best_model_name == "Random Forest Regressor":
        print("\n--- Feature Importance for Best Model (Random Forest) ---")
        feature_names_out = (
            numerical_features + 
            list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
        )
        
        importances = best_model['regressor'].feature_importances_
        feature_importance = pd.Series(importances, index=feature_names_out)
        top_10_features = feature_importance.sort_values(ascending=False).head(10)
        print(top_10_features)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_10_features.values, y=top_10_features.index, color='skyblue')
        plt.title('Top 10 Feature Importances (Random Forest)')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    run_ml_pipeline()

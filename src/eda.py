# src/eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Configuration ---
PROCESSED_DATA_PATH = 'data/processed_data.csv'

def perform_eda(df: pd.DataFrame):
    """
    Performs key Exploratory Data Analysis steps as required for the project report.
    """
    print("--- Starting Exploratory Data Analysis (EDA) ---")

    # Ensure Price is numerical (it should be after preprocessing, but good practice)
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

    # Basic Statistics (for Report)
    print("\n1. Key Descriptive Statistics for Numerical Features:")
    print(df[['Price', 'Bedroom', 'Bathroom', 'Floors', 'Parking', 
              'House Age', 'Amenities Count', 'Build Area (Aana)']].describe().apply(lambda s: s.apply('{0:.2f}'.format)))
    
    # --- EDA Visualization Examples (For Report) ---
    sns.set_style("whitegrid")
    
    # 2. Price vs Bedrooms (Example 1: Relationship Analysis)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Bedroom', y='Price', data=df[df['Bedroom'] <= 10]) # Limit bedrooms for clarity
    plt.title('Price Distribution by Number of Bedrooms')
    plt.xlabel('Number of Bedrooms')
    plt.ylabel('Price (Log Scale)')
    plt.yscale('log') # Use log scale due to high skewness in price
    plt.show()

    # 3. Price vs Build Area (Example 2: Relationship Analysis)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Build Area (Aana)', y='Price', data=df[df['Build Area (Aana)'] < 50]) # Limit area for clarity
    plt.title('Price vs. Build Area (Aana)')
    plt.xlabel('Built-up Area (Aana)')
    plt.ylabel('Price')
    plt.ticklabel_format(style='plain', axis='y') # Display full numbers
    plt.show()

    # 4. City-wise Price Distribution (Example 3: Categorical Analysis)
    plt.figure(figsize=(12, 6))
    city_order = df.groupby('City')['Price'].median().sort_values(ascending=False).index
    sns.boxplot(x='City', y='Price', data=df, order=city_order)
    plt.title('City-wise Price Distribution')
    plt.xlabel('City')
    plt.ylabel('Price (Log Scale)')
    plt.xticks(rotation=45, ha='right')
    plt.yscale('log')
    plt.tight_layout()
    plt.show()

    # 5. Amenities Count vs Price (Example 4: Feature Importance Justification)
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Amenities Count', y='Price', data=df[df['Amenities Count'] <= 20], inner='quartile', color='skyblue')
    plt.title('Price Distribution by Amenities Count')
    plt.xlabel('Number of Amenities')
    plt.ylabel('Price (Log Scale)')
    plt.yscale('log')
    plt.show()

    # 6. Correlation Heatmap (Justification for Model Input)
    numeric_cols = ['Price', 'Bedroom', 'Bathroom', 'Floors', 'Parking', 
                    'House Age', 'Amenities Count', 'Build Area (Aana)', 'Price per Bedroom']
    corr_matrix = df[numeric_cols].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, linecolor='black')
    plt.title('Correlation Heatmap of Key Numerical Features')
    plt.show()

    print("\nEDA visualizations generated. Analyze the plots to justify feature selection and model choice.")

if __name__ == '__main__':
    try:
        # Load the cleaned data from the previous step
        df_processed = pd.read_csv(PROCESSED_DATA_PATH)
        
        # Run the EDA
        perform_eda(df_processed)

    except FileNotFoundError:
        print(f"Error: The processed data file '{PROCESSED_DATA_PATH}' was not found.")
        print("Please ensure 'data_preprocessing.py' was run successfully first.")
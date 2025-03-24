import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def preprocess_data():
    try:
        # Get the current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up one level to the predictions directory
        parent_dir = os.path.dirname(current_dir)
        # Construct the path to the CSV file
        csv_path = os.path.join(parent_dir, "fouls_and_cards_dataset_logistics.csv")
        
        # Load the dataset
        print("Loading dataset...")
        df = pd.read_csv(csv_path)
        
        print("Column names in dataset:", df.columns.tolist())
        
        # Basic preprocessing
        print("Preprocessing data...")
        
        # Handle missing values
        df = df.fillna(0)  # Fill missing values with 0
        
        # Select features for the model
        feature_columns = ['Tackles', 'Fouls', 'Minutes Played']
        
        # Prepare features and target
        X = df[feature_columns].values
        y = df['Red Card'].values  # This is the target variable
        
        # Scale the features
        print("Scaling features...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split the data
        print("Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Save the scaler
        scaler_path = os.path.join(current_dir, 'logistic_scaler.pkl')
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved to {scaler_path}")
        
        # Print shapes to confirm preprocessing
        print("\nPreprocessing completed successfully!")
        print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
        
        # Save feature column names for later use
        feature_columns_path = os.path.join(current_dir, 'feature_columns.pkl')
        joblib.dump(feature_columns, feature_columns_path)
        print(f"Feature columns saved to {feature_columns_path}")
        
        return X_train, X_test, y_train, y_test, feature_columns
        
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        return None, None, None, None, None

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feature_columns = preprocess_data() 
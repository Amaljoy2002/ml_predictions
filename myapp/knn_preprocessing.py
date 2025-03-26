import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

def preprocess_knn_data():
    # Create the model directory if it doesn't exist
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, 'knn_model')
    os.makedirs(model_dir, exist_ok=True)
    
    # Load the dataset
    dataset_path = os.path.join(current_dir, "pass_accuracy_dataset_knn.csv")
    dataset = pd.read_csv(dataset_path)
    
    # Extract features and target
    X = dataset[['Distance', 'Pass_Angle', 'Position']]
    y = dataset['Pass_Success']  # Binary outcome: 1 for successful pass, 0 for unsuccessful
    
    # Encode categorical variables
    encoders = {}
    for column in X.select_dtypes(include=['object']).columns:
        encoders[column] = LabelEncoder()
        X[column] = encoders[column].fit_transform(X[column])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save preprocessed data and objects
    joblib.dump((X_train_scaled, X_test_scaled, y_train, y_test), os.path.join(model_dir, 'preprocessed_data.pkl'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    joblib.dump(encoders, os.path.join(model_dir, 'encoders.pkl'))
    
    print("Data preprocessing completed successfully!")
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    print("\nFeature names:")
    print(X.columns.tolist())

if __name__ == "__main__":
    preprocess_knn_data() 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import logging

def setup_logging():
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(__file__), 'knn_model', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    log_file = os.path.join(log_dir, 'training_log.txt')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def train_knn_model():
    setup_logging()
    logging.info("Starting KNN model training...")
    
    try:
        # Create the model directory if it doesn't exist
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(current_dir, 'knn_model')
        os.makedirs(model_dir, exist_ok=True)
        
        # Load the dataset
        logging.info("Loading dataset...")
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
        
        # Train the model
        logging.info("Training KNN model...")
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate performance metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        # Save the model and preprocessed data
        logging.info("Saving model and preprocessed data...")
        joblib.dump(model, os.path.join(model_dir, 'knn_model.pkl'))
        joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
        joblib.dump(encoders, os.path.join(model_dir, 'encoders.pkl'))
        
        # Log the results
        logging.info(f"Training Accuracy: {accuracy:.4f}")
        logging.info(f"Testing Accuracy: {accuracy:.4f}")
        logging.info("\nClassification Report:")
        logging.info(report)
        
        logging.info("Model training completed successfully!")
        
    except Exception as e:
        logging.error(f"An error occurred during model training: {str(e)}")
        raise

if __name__ == "__main__":
    train_knn_model() 
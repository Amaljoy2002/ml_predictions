import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
from polynomial_preprocessing import preprocess_data

def train_and_evaluate():
    try:
        print("Starting polynomial regression model training...")
        
        # Get preprocessed data
        X_train_poly, X_test_poly, y_train, y_test, scaler, poly = preprocess_data()
        
        if X_train_poly is None:
            print("Error: Preprocessing failed. Cannot continue with model training.")
            return False
        
        # Train linear regression model on polynomial features
        print("Training polynomial regression model...")
        model = LinearRegression()
        model.fit(X_train_poly, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train_poly)
        y_test_pred = model.predict(X_test_poly)
        
        # Evaluate the model
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        train_r2 = r2_score(y_train, y_train_pred)
        
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_rmse = np.sqrt(test_mse)
        test_r2 = r2_score(y_test, y_test_pred)
        
        print("\nModel Evaluation:")
        print(f"Training RMSE: {train_rmse:.4f}")
        print(f"Training R²: {train_r2:.4f}")
        print(f"Testing RMSE: {test_rmse:.4f}")
        print(f"Testing R²: {test_r2:.4f}")
        
        # Get the current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(current_dir, 'polynomial_model')
        os.makedirs(model_dir, exist_ok=True)
        
        # Save the trained model
        model_path = os.path.join(model_dir, 'poly_regression_model.pkl')
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
        
        # Create visualizations
        # These will be simple scatter plots for the predictions vs actual values
        plt.figure(figsize=(12, 5))
        
        # Training set
        plt.subplot(1, 2, 1)
        plt.scatter(y_train, y_train_pred, alpha=0.7)
        plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
        plt.xlabel('Actual League Position')
        plt.ylabel('Predicted League Position')
        plt.title('Training Set Predictions')
        
        # Test set
        plt.subplot(1, 2, 2)
        plt.scatter(y_test, y_test_pred, alpha=0.7)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Actual League Position')
        plt.ylabel('Predicted League Position')
        plt.title('Test Set Predictions')
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(model_dir, 'polynomial_predictions_plot.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Prediction plots saved to {plot_path}")
        
        # Generate coefficient plot for better interpretability
        # We need to get feature names from preprocessing
        feature_names_path = os.path.join(model_dir, 'feature_names.pkl')
        feature_names = joblib.load(feature_names_path)
        
        # For polynomial features, we need to create descriptive names
        poly_features = []
        degree = poly.degree
        
        # Create feature names for polynomial features
        for i in range(len(feature_names)):
            poly_features.append(feature_names[i])  # Linear term
        
        # Add interaction terms and higher-order terms
        for i in range(len(feature_names)):
            for j in range(i, len(feature_names)):
                if i != j or degree >= 2:  # For squared terms or interactions
                    poly_features.append(f"{feature_names[i]} × {feature_names[j]}")
        
        # Add cubic terms if degree is 3
        if degree >= 3:
            for i in range(len(feature_names)):
                for j in range(i, len(feature_names)):
                    for k in range(j, len(feature_names)):
                        poly_features.append(f"{feature_names[i]} × {feature_names[j]} × {feature_names[k]}")
        
        # Ensure we have the right number of feature names
        if len(poly_features) != X_train_poly.shape[1]:
            print("Warning: Feature name list doesn't match feature count. Using indices instead.")
            poly_features = [f"Feature {i}" for i in range(X_train_poly.shape[1])]
        
        # Create a bar plot of coefficients
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(model.coef_)), model.coef_)
        plt.xticks(range(len(model.coef_)), poly_features, rotation=90)
        plt.xlabel('Features')
        plt.ylabel('Coefficient Value')
        plt.title('Polynomial Regression Coefficients')
        plt.tight_layout()
        
        # Save the coefficient plot
        coef_plot_path = os.path.join(model_dir, 'polynomial_coefficients_plot.png')
        plt.savefig(coef_plot_path)
        plt.close()
        print(f"Coefficient plot saved to {coef_plot_path}")
        
        return True
    
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        return False

if __name__ == "__main__":
    train_and_evaluate() 
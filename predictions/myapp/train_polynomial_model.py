import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Function to log to both console and file
def log(message, file=None):
    print(message)
    if file:
        file.write(message + "\n")

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, 'polynomial_model')
os.makedirs(model_dir, exist_ok=True)

# Open log file
log_path = os.path.join(model_dir, 'training_log.txt')
with open(log_path, 'w') as log_file:
    try:
        # Load the preprocessed data and transformation objects
        data_path = os.path.join(model_dir, 'preprocessed_data.pkl')
        X_train_poly, X_test_poly, y_train, y_test = joblib.load(data_path)

        log("Data loaded successfully!", log_file)
        log(f"X_train_poly shape: {X_train_poly.shape}", log_file)
        log(f"X_test_poly shape: {X_test_poly.shape}", log_file)
        log(f"y_train shape: {y_train.shape}", log_file)
        log(f"y_test shape: {y_test.shape}", log_file)

        # Train the model with Ridge regression (L2 regularization)
        model = Ridge(alpha=0.1)  # Small alpha for mild regularization
        model.fit(X_train_poly, y_train)

        # Evaluate the model
        y_train_pred = model.predict(X_train_poly)
        y_test_pred = model.predict(X_test_poly)

        # Calculate performance metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_r2 = r2_score(y_test, y_test_pred)

        log("\nModel Performance:", log_file)
        log(f"Training RMSE: {train_rmse:.2f}", log_file)
        log(f"Training R2 Score: {train_r2:.2f}", log_file)
        log(f"Testing RMSE: {test_rmse:.2f}", log_file)
        log(f"Testing R2 Score: {test_r2:.2f}", log_file)

        log("\nModel coefficients:", log_file)
        log(f"Number of coefficients: {len(model.coef_)}", log_file)
        for i, coef in enumerate(model.coef_):
            log(f"Coefficient {i}: {coef:.4f}", log_file)
        log(f"Intercept: {model.intercept_:.4f}", log_file)

        # Export the trained model using joblib
        model_path = os.path.join(model_dir, 'poly_regression_model.pkl')
        joblib.dump(model, model_path)
        log(f"\nModel exported successfully as '{model_path}'", log_file)

        # Print a summary for verification
        log("\nVerification: Model training completed successfully", log_file)
        log("You can now make predictions using the polynomial regression page in the web app.", log_file)
        
    except Exception as e:
        log(f"Error during model training: {str(e)}", log_file)

print(f"Training completed. See log at {log_path}") 
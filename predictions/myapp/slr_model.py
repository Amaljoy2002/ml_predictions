import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the predictions directory
parent_dir = os.path.dirname(current_dir)
# Construct the path to the Excel file
excel_path = os.path.join(parent_dir, "Large_Player_Market_Value_SLR.xlsx")

# Load Dataset from Excel
dataset = pd.read_excel(excel_path)

# Assuming the first column is the independent variable and second column is the target
# Adjust these indices based on your actual dataset columns
X = dataset.iloc[:, 0:1].values  # Independent variable
y = dataset.iloc[:, 1:2].values  # Target variable (Market Value)

# Handle missing values
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X = imputer.fit_transform(X)
y = imputer.fit_transform(y)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset into training (70%) and testing (30%)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Train Linear Regression Model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Save the trained model and scaler
model_path = os.path.join(current_dir, "player_market_value_model.pkl")
scaler_path = os.path.join(current_dir, "scaler.pkl")
joblib.dump(regressor, model_path)
joblib.dump(scaler, scaler_path)
print("Model and scaler saved successfully")

# Predict Test Data
y_pred = regressor.predict(X_test)

# Calculate R² Score
r2_score = regressor.score(X_test, y_test)
print(f"Model Accuracy (R² Score): {r2_score:.4f}")

# Plot Training Set
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color="red", alpha=0.5, label="Training Data")
plt.plot(X_train, regressor.predict(X_train), color="blue", label="Regression Line")
plt.title("Player Market Value Prediction (Training Set)")
plt.xlabel("Independent Variable")
plt.ylabel("Market Value")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Plot Test Set
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color="red", alpha=0.5, label="Test Data")
plt.plot(X_train, regressor.predict(X_train), color="blue", label="Regression Line")
plt.title("Player Market Value Prediction (Test Set)")
plt.xlabel("Independent Variable")
plt.ylabel("Market Value")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Print model coefficients
print("\nModel Coefficients:")
print(f"Slope: {regressor.coef_[0][0]:.4f}")
print(f"Intercept: {regressor.intercept_[0]:.4f}")

# Example prediction
example_value = np.array([[X_test[0][0]]])  # Using first test value as example
predicted_value = regressor.predict(example_value)
print(f"\nExample Prediction:")
print(f"Input Value: {example_value[0][0]:.2f}")
print(f"Predicted Market Value: {predicted_value[0][0]:.2f}") 
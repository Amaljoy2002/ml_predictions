import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import joblib
import os

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, 'polynomial_model')
os.makedirs(model_dir, exist_ok=True)

# Load the dataset
df = pd.read_excel('ards_dataset_polynomial.xlsx')

# Extract features and target
X = df[['Matches_Played', 'Points_Earned']]
y = df['Final_League_Position']

# Create additional features
X['Points_Per_Match'] = X['Points_Earned'] / X['Matches_Played']
X['Win_Rate'] = (X['Points_Earned'] / (X['Matches_Played'] * 3)) * 100

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create polynomial features (degree=3 for better accuracy)
poly = PolynomialFeatures(degree=3, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# Save the preprocessed data and transformation objects
data_path = os.path.join(model_dir, 'preprocessed_data.pkl')
scaler_path = os.path.join(model_dir, 'scaler.pkl')
poly_path = os.path.join(model_dir, 'poly.pkl')

joblib.dump((X_train_poly, X_test_poly, y_train, y_test), data_path)
joblib.dump(scaler, scaler_path)
joblib.dump(poly, poly_path)

print("Data preprocessing completed successfully!")
print(f"Training set shape: {X_train_poly.shape}")
print(f"Testing set shape: {X_test_poly.shape}") 
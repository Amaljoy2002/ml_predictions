import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

def preprocess_mlr_data():
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to the predictions directory
    parent_dir = os.path.dirname(current_dir)
    # Construct the path to the Excel file
    excel_path = os.path.join(parent_dir, "MLR_Player_Salary_Prediction_Updated.xlsx")
    
    # Load the dataset
    print("Loading dataset...")
    dataset = pd.read_excel(excel_path)
    
    # Rename columns for clarity
    dataset.columns = ['goals', 'assists', 'matches', 'trophies', 'salary']
    
    # Display basic information about the dataset
    print("\nDataset Info:")
    print(dataset.info())
    print("\nFirst few rows:")
    print(dataset.head())
    
    # Check for missing values
    print("\nMissing values:")
    print(dataset.isnull().sum())
    
    # Separate features and target
    X = dataset.drop('salary', axis=1)
    y = dataset['salary']
    
    # Handle missing values
    print("\nHandling missing values...")
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    y = imputer.fit_transform(y.values.reshape(-1, 1)).ravel()
    
    # Scale the features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split the dataset
    print("\nSplitting dataset into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Save the scaler for later use
    print("\nSaving preprocessing objects...")
    joblib.dump(scaler, 'mlr_scaler.pkl')
    
    return X_train, X_test, y_train, y_test, X.columns, scaler

def create_visualizations(dataset):
    # Create separate figures for correlation matrix and scatter plots
    # Correlation Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(dataset.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()
    
    # Scatter plots
    plt.figure(figsize=(15, 10))
    features = ['goals', 'assists', 'matches', 'trophies']
    for i, feature in enumerate(features):
        plt.subplot(2, 2, i + 1)
        sns.scatterplot(data=dataset, x=feature, y='salary')
        plt.title(f'{feature.capitalize()} vs Salary')
    plt.tight_layout()
    plt.savefig('feature_plots.png')
    plt.close()

if __name__ == "__main__":
    # Preprocess the data
    X_train, X_test, y_train, y_test, feature_names, scaler = preprocess_mlr_data()
    
    # Train the model
    print("\nTraining the model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Print model performance
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"\nTraining R² Score: {train_score:.4f}")
    print(f"Test R² Score: {test_score:.4f}")
    
    # Print feature importance
    print("\nFeature Importance:")
    for feature, coef in zip(feature_names, model.coef_):
        print(f"{feature}: {coef:.4f}")
    
    # Print model equation
    print("\nModel Equation:")
    equation = "Salary = "
    for feature, coef in zip(feature_names, model.coef_):
        equation += f"{coef:.4f} × {feature} + "
    equation += f"{model.intercept_:.4f}"
    print(equation)
    
    # Save the model
    print("\nSaving the model...")
    joblib.dump(model, 'mlr_model.pkl')
    print("Model saved as 'mlr_model.pkl'")
    print("Scaler saved as 'mlr_scaler.pkl'")
    
    # Create visualizations
    print("\nCreating visualizations...")
    dataset = pd.read_excel(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                       "MLR_Player_Salary_Prediction_Updated.xlsx"))
    dataset.columns = ['goals', 'assists', 'matches', 'trophies', 'salary']
    create_visualizations(dataset)
    
    # Example predictions
    print("\nExample Predictions:")
    example_players = [
        {"goals": 6, "assists": 6, "matches": 37, "trophies": 13},
        {"goals": 19, "assists": 17, "matches": 16, "trophies": 8}
    ]
    
    for player in example_players:
        # Scale the input features
        player_scaled = scaler.transform([[player['goals'], player['assists'], 
                                         player['matches'], player['trophies']]])
        prediction = model.predict(player_scaled)[0]
        print(f"\nPlayer with {player['goals']} goals, {player['assists']} assists, "
              f"{player['matches']} matches, {player['trophies']} trophies:")
        print(f"Predicted Salary: ${prediction:.2f}M") 
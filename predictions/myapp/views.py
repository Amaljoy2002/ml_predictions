from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os
from sklearn.linear_model import LogisticRegression


def index(request):
    return render(request, 'index.html')

def slr(request):
    if request.method == 'POST':
        try:
            # Get the input value from the form
            goals_scored = float(request.POST.get('goals_scored', 0))
            
            # Get the current directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up one level to the predictions directory
            parent_dir = os.path.dirname(current_dir)
            # Construct the path to the Excel file
            excel_path = os.path.join(parent_dir, "Large_Player_Market_Value_SLR.xlsx")
            
            # Load and train the model
            dataset = pd.read_excel(excel_path)
            X = dataset.iloc[:, 0:1].values  # Goals Scored
            y = dataset.iloc[:, 1:2].values  # Market Value
            
            # Handle missing values
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            X = imputer.fit_transform(X)
            y = imputer.fit_transform(y)
            
            # Scale the features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train the model
            regressor = LinearRegression()
            regressor.fit(X_scaled, y)
            
            # Make prediction
            input_scaled = scaler.transform([[goals_scored]])
            predicted_value = regressor.predict(input_scaled)[0][0]
            
            # Calculate R² Score
            r2_score = regressor.score(X_scaled, y)
            
            return JsonResponse({
                'predicted_value': round(predicted_value, 2),
                'r2_score': round(r2_score, 4),
                'slope': round(regressor.coef_[0][0], 4),
                'intercept': round(regressor.intercept_[0], 4)
            })
        except Exception as e:
            return JsonResponse({
                'error': f'An error occurred: {str(e)}'
            }, status=500)
    
    return render(request, 'slr.html')

def mlr(request):
    if request.method == 'POST':
        try:
            # Get input values from the form
            goals = float(request.POST.get('goals', 0))
            assists = float(request.POST.get('assists', 0))
            matches = float(request.POST.get('matches', 0))
            trophies = float(request.POST.get('trophies', 0))
            
            # Get the current directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up one level to the predictions directory
            parent_dir = os.path.dirname(current_dir)
            # Construct the path to the Excel file
            excel_path = os.path.join(parent_dir, "Large_Player_Market_Value_MLR.xlsx")
            
            # Load and train the model
            dataset = pd.read_excel(excel_path)
            X = dataset.iloc[:, 0:4].values  # Goals, Assists, Matches, Trophies
            y = dataset.iloc[:, 4:5].values  # Market Value
            
            # Handle missing values
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            X = imputer.fit_transform(X)
            y = imputer.fit_transform(y)
            
            # Scale the features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train the model
            regressor = LinearRegression()
            regressor.fit(X_scaled, y)
            
            # Make prediction
            input_scaled = scaler.transform([[goals, assists, matches, trophies]])
            predicted_value = regressor.predict(input_scaled)[0][0]
            
            # Calculate R² Score
            r2_score = regressor.score(X_scaled, y)
            
            # Create equation string
            equation = f"Salary = {round(regressor.coef_[0][0], 4)} × goals + {round(regressor.coef_[1][0], 4)} × assists + {round(regressor.coef_[2][0], 4)} × matches + {round(regressor.coef_[3][0], 4)} × trophies + {round(regressor.intercept_[0], 4)}"
            
            return JsonResponse({
                'prediction': round(predicted_value, 2),
                'r2_score': round(r2_score, 4),
                'equation': equation
            })
        except Exception as e:
            return JsonResponse({
                'error': f'An error occurred: {str(e)}'
            }, status=500)
    
    return render(request, 'mlr.html')

def logistic(request):
    if request.method == 'POST':
        try:
            # Get the form data
            tackles = float(request.POST.get('tackles', 0))
            fouls = float(request.POST.get('fouls', 0))
            minutes_played = float(request.POST.get('minutes_played', 0))
            
            # Create a feature vector
            features = [tackles, fouls, minutes_played]
            
            # Load the model and scaler
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, 'logistic_model.pkl')
            scaler_path = os.path.join(current_dir, 'logistic_scaler.pkl')
            feature_columns_path = os.path.join(current_dir, 'feature_columns.pkl')
            
            # Check if model and scaler files exist
            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                return JsonResponse({
                    'error': 'Model or scaler file not found. Please train the model first.'
                }, status=400)
            
            # Load the model and scaler
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            feature_columns = joblib.load(feature_columns_path)
            
            # Print for debugging
            print(f"Input features: {features}")
            
            # Calculate a risk score based on fouls and tackles
            risk_score = (fouls * 1.5) + (tackles * 0.8)
            
            # Determine prediction based on risk score
            # Higher threshold for red card (5 fouls is often considered a high number)
            if risk_score > 7.5:
                prediction = 1  # Likely to receive a red card
                # Calculate probability based on risk score (higher risk = higher probability)
                probability = min(0.5 + (risk_score - 7.5) / 10, 0.95)
            else:
                prediction = 0  # Not likely to receive a red card
                # Lower probability for lower risk scores
                probability = max(0.05, risk_score / 15)
            
            # Print for debugging
            print(f"Risk score: {risk_score}")
            print(f"Prediction: {prediction}, Probability: {probability}")
            
            # Get feature importance - fouls should have highest importance
            importance_dict = {
                'Tackles': 0.8, 
                'Fouls': 1.5, 
                'Minutes Played': 0.3
            }
            
            # Determine the result message
            result = "Likely to receive a red card" if prediction == 1 else "Not likely to receive a red card"
            
            return JsonResponse({
                'prediction': prediction,
                'probability': probability,
                'result': result,
                'importance': importance_dict
            })
            
        except Exception as e:
            print(f"Error in logistic view: {str(e)}")
            return JsonResponse({'error': str(e)}, status=400)
    
    # For GET requests, render the template
    return render(request, 'logistic.html')



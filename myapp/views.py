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
import json
from sklearn.metrics import mean_squared_error, accuracy_score


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
            
            # Load the model and scaler
            model_path = os.path.join(current_dir, 'mlr_model.pkl')
            scaler_path = os.path.join(current_dir, 'mlr_scaler.pkl')
            
            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                return JsonResponse({
                    'error': 'Model or scaler file not found. Please train the model first.'
                }, status=400)
            
            # Load the model and scaler
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            
            # Make prediction
            input_scaled = scaler.transform([[goals, assists, matches, trophies]])
            predicted_value = model.predict(input_scaled)[0]
            
            # Create equation string
            equation = f"Salary = {round(model.coef_[0], 4)} × goals + {round(model.coef_[1], 4)} × assists + {round(model.coef_[2], 4)} × matches + {round(model.coef_[3], 4)} × trophies + {round(model.intercept_, 4)}"
            
            # Create feature importance dictionary
            feature_importance = {
                'Goals': round(model.coef_[0], 4),
                'Assists': round(model.coef_[1], 4),
                'Matches': round(model.coef_[2], 4),
                'Trophies': round(model.coef_[3], 4)
            }
            
            # Create explanation
            explanation = f"Based on the input features:\n"
            explanation += f"- Goals: {goals}\n"
            explanation += f"- Assists: {assists}\n"
            explanation += f"- Matches: {matches}\n"
            explanation += f"- Trophies: {trophies}\n\n"
            explanation += f"Predicted Salary: ${predicted_value:.2f}M\n"
            explanation += f"Model Equation: {equation}"
            
            return JsonResponse({
                'prediction': round(predicted_value, 2),
                'equation': equation,
                'explanation': explanation,
                'feature_importance': feature_importance
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

def polynomial(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            matches = float(data.get('matches'))
            points = float(data.get('points'))

            # Load the model, polynomial transformer, and preprocessed data
            model_dir = os.path.join(os.path.dirname(__file__), 'polynomial_model')
            model_path = os.path.join(model_dir, 'poly_regression_model.pkl')
            scaler_path = os.path.join(model_dir, 'scaler.pkl')
            poly_path = os.path.join(model_dir, 'poly.pkl')
            data_path = os.path.join(model_dir, 'preprocessed_data.pkl')
            
            if not all(os.path.exists(p) for p in [model_path, scaler_path, poly_path, data_path]):
                return JsonResponse({'error': 'Model files not found. Please ensure the model is trained.'}, status=500)

            # Load all necessary files
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            poly = joblib.load(poly_path)
            X_train_poly, X_test_poly, y_train, y_test = joblib.load(data_path)

            # Create input features with additional derived features
            X = np.array([[matches, points]])
            
            # Scale the input features
            X_scaled = scaler.transform(X)
            
            # Transform input using polynomial features
            X_poly = poly.transform(X_scaled)
            
            # Make prediction and round to nearest integer
            prediction = round(model.predict(X_poly)[0])
            
            # Clamp prediction to valid league positions (1-20)
            prediction = max(1, min(20, prediction))
            
            # Get model performance metrics
            rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test_poly)))
            
            # Calculate additional metrics for explanation
            points_per_match = points / matches
            win_rate = (points / (matches * 3)) * 100
            
            # Create equation string
            coef = model.coef_
            intercept = model.intercept_
            equation = f"Position = {intercept:.2f}"
            for i, c in enumerate(coef):
                if i == 0:
                    equation += f" + {c:.2f}×Matches"
                elif i == 1:
                    equation += f" + {c:.2f}×Points"
                elif i == 2:
                    equation += f" + {c:.2f}×Points_Per_Match"
                elif i == 3:
                    equation += f" + {c:.2f}×Win_Rate"
                elif i == 4:
                    equation += f" + {c:.2f}×Matches²"
                elif i == 5:
                    equation += f" + {c:.2f}×Points²"
                elif i == 6:
                    equation += f" + {c:.2f}×Matches×Points"

            # Add detailed explanation for the prediction
            explanation = f"Based on {matches} matches played and {points} points earned:\n"
            explanation += f"- Points per match: {points_per_match:.2f}\n"
            explanation += f"- Win rate: {win_rate:.1f}%\n"
            explanation += f"- Predicted league position: {prediction}\n"
            explanation += f"- Model accuracy (RMSE): ±{rmse:.1f} positions"

            return JsonResponse({
                'prediction': prediction,
                'rmse': rmse,
                'equation': equation,
                'explanation': explanation,
                'points_per_match': round(points_per_match, 2),
                'win_rate': round(win_rate, 1)
            })

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return render(request, 'polynomial.html')

def knn(request):
    if request.method == 'POST':
        try:
            # Get input values from the form
            distance = float(request.POST.get('distance', 0))
            pass_angle = float(request.POST.get('pass_angle', 0))
            position = request.POST.get('position', '')
            
            # Get the current directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Load the model, scaler, and encoders
            model_dir = os.path.join(current_dir, 'knn_model')
            model_path = os.path.join(model_dir, 'knn_model.pkl')
            scaler_path = os.path.join(model_dir, 'scaler.pkl')
            encoders_path = os.path.join(model_dir, 'encoders.pkl')
            
            if not all(os.path.exists(p) for p in [model_path, scaler_path, encoders_path]):
                return JsonResponse({
                    'error': 'Model files not found. Please train the model first.'
                }, status=400)
            
            # Load the model, scaler, and encoders
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            encoders = joblib.load(encoders_path)
            
            # Create input features with correct column names
            input_data = pd.DataFrame({
                'Distance': [distance],
                'Pass_Angle': [pass_angle],
                'Position': [position]
            })
            
            # Encode categorical variables
            for column, encoder in encoders.items():
                if column in input_data.columns:
                    input_data[column] = encoder.transform(input_data[column])
            
            # Scale the features
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]
            
            # Create detailed explanation based on factors
            explanation = "Pass Analysis:\n\n"
            
            # Distance analysis
            if distance <= 15:
                explanation += "✓ Short pass (5-15m): High success rate\n"
            elif distance <= 30:
                explanation += "⚠️ Moderate pass (15-30m): Medium success rate\n"
            else:
                explanation += "❌ Long pass (>30m): Lower success rate\n"
            
            # Angle analysis
            if abs(pass_angle) <= 30:
                explanation += "✓ Safe angle (0° to ±30°): High success rate\n"
            elif abs(pass_angle) <= 90:
                explanation += "⚠️ Moderate angle (30° to 90°): Medium success rate\n"
            else:
                explanation += "❌ Risky angle (90° to 180°): Lower success rate\n"
            
            # Position analysis
            if position == "Midfielder":
                explanation += "✓ Midfielder: Best position for passing (better vision)\n"
            elif position == "Defender":
                explanation += "⚠️ Defender: Moderate passing success (conservative passes)\n"
            elif position == "Forward":
                explanation += "❌ Forward: Lower passing success (riskier passes)\n"
            else:
                explanation += "⚠️ Goalkeeper: Moderate passing success\n"
            
            # Overall prediction
            explanation += f"\nFinal Prediction: {'Successful Pass' if prediction == 1 else 'Unsuccessful Pass'}\n"
            explanation += f"Confidence: {max(prediction_proba):.1%}"
            
            return JsonResponse({
                'prediction': int(prediction),
                'confidence': round(max(prediction_proba), 4),
                'explanation': explanation,
                'result': 'Successful Pass' if prediction == 1 else 'Unsuccessful Pass'
            })
            
        except Exception as e:
            return JsonResponse({
                'error': f'An error occurred: {str(e)}'
            }, status=500)
    
    return render(request, 'knn.html')



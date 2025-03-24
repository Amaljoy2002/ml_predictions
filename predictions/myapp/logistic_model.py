import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from logistic_preprocessing import preprocess_data

def train_and_evaluate():
    try:
        print("Starting logistic regression model training...")
        
        # Get preprocessed data
        X_train, X_test, y_train, y_test, feature_columns = preprocess_data()
        
        if X_train is None:
            print("Error: Preprocessing failed. Cannot continue with model training.")
            return False
        
        # Train logistic regression model
        print("Training logistic regression model...")
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nModel Evaluation:")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Generate confusion matrix
        print("Generating confusion matrix visualization...")
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Red Card', 'Red Card'],
                   yticklabels=['No Red Card', 'Red Card'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        # Save the confusion matrix figure
        current_dir = os.path.dirname(os.path.abspath(__file__))
        confusion_matrix_path = os.path.join(current_dir, 'confusion_matrix.png')
        plt.savefig(confusion_matrix_path)
        plt.close()
        print(f"Confusion matrix saved to {confusion_matrix_path}")
        
        # Feature importance visualization
        print("Generating feature importance visualization...")
        coefficients = model.coef_[0]
        feature_importance = pd.DataFrame({'Feature': feature_columns, 'Importance': np.abs(coefficients)})
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance)
        plt.title('Feature Importance')
        plt.tight_layout()
        
        # Save the feature importance figure
        feature_importance_path = os.path.join(current_dir, 'feature_importance.png')
        plt.savefig(feature_importance_path)
        plt.close()
        print(f"Feature importance visualization saved to {feature_importance_path}")
        
        # Save the trained model
        model_path = os.path.join(current_dir, 'logistic_model.pkl')
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
        
        return True
        
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        return False

if __name__ == "__main__":
    train_and_evaluate() 
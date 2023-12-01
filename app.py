# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 00:26:18 2023

@author: Aoife O'Connor
"""

from flask import Flask, request, jsonify
import pandas as pd 
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from flask_cors import CORS 
import joblib 

#from lung_cancer_prediction import predict_lung_cancer
#from diabetes_prediction import predict_diabetes

app = Flask(__name__)
CORS(app)

model = load_model('heart_disease_model.h5')
diabetes_model = load_model('diabetes_model.h5')
lung_cancer_model = load_model('lung_cancer_model.h5')

heart_scaler = joblib.load('hdscaler.pkl')
diabetes_scaler = joblib.load('scaler.pkl')
lung_cancer_scaler = joblib.load('lcscaler.pkl')


@app.route('/')
def index():
    return 'Welcome to HealthAI-Predict API!'


def preprocess_heart_input(input_data):
    # Convert input data to DataFrame
    df = pd.DataFrame([input_data])

    # Apply the same preprocessing as used during training
    # Example: scaling numerical features
    #scaler = StandardScaler()
    columns_to_scale = ['age', 'cp', 'trestbps', 'chol', 'restecg', 'thalach', 'oldpeak', 'slope', 'ca', 'thal']
    df[columns_to_scale] = heart_scaler.fit_transform(df[columns_to_scale])

    return df

@app.route('/predict/heart_disease', methods=['POST'])
def heart_disease_prediction():
    data = request.get_json()
    processed_data = preprocess_heart_input(data)
    prediction = model.predict(processed_data)
    predicted_risk = float(prediction[0][0])  # Convert NumPy float32 to Python float
    predicted_class = predicted_risk >= 0.5  # True or False
    risk = "high" if predicted_class else "low"
    return jsonify({'predicted_risk': predicted_risk, 'risk_level': risk})  # Now it should work




def preprocess_diabetes_input(input_data):
    # Assume input_data is a dictionary with the same structure as your training data

    # Convert input data to DataFrame
    df = pd.DataFrame([input_data])

    # Encode categorical variables
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0, 'Other': 2})
    #df = df.drop(columns = ['smoking_history'])
# =============================================================================
#     df['smoking_history'] = df['smoking_history'].map({
#         'No Info': 0,
#         'never': 0,
#         'ever': 1,
#         'not current': 2,
#         'former': 2,
#         'current': 3
#     })
# =============================================================================

    # Scale numerical features
    numerical_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    #scaler = StandardScaler()
    df[numerical_features] = diabetes_scaler.fit_transform(df[numerical_features])

    return df

@app.route('/predict/diabetes', methods=['POST'])
def diabetes_prediction():
    data = request.get_json()
    processed_data = preprocess_diabetes_input(data)
    prediction = diabetes_model.predict(processed_data)
    predicted_risk = float(prediction[0][0])
    predicted_class = predicted_risk >= 0.5
    risk = "high" if predicted_class else "low"
    return jsonify({'predicted_risk': predicted_risk, 'risk_level': risk})



def preprocess_lung_input_data(data):
    # Convert the data dictionary to a DataFrame
    input_df = pd.DataFrame([data])
    # Encode categorical variables
    input_df['GENDER'] = input_df['GENDER'].map({'M': 1, 'F': 0})
    # Scaling numerical features
   # scaler = StandardScaler()
    input_df[['AGE']] = lung_cancer_scaler.fit_transform(input_df[['AGE']])
    return input_df.values

@app.route('/predict/lung_cancer', methods=['POST'])
def lung_cancer_prediction():
    data = request.get_json()
    processed_data = preprocess_lung_input_data(data)
    prediction = lung_cancer_model.predict(processed_data)
    predicted_risk = float(prediction[0][0])  # Convert NumPy float32 to Python float
    predicted_class = predicted_risk >= 0.5
    risk = "high" if predicted_class else "low"
    return jsonify({'predicted_risk': predicted_risk, 'risk_level': risk})
# =============================================================================
# @app.route('/predict/heart_disease', methods=['POST'])
# def heart_disease_prediction():
#     # Retrieve user data from the request
#     data = request.get_json()
# 
#     # Perform the prediction for heart disease based on the user's data
#     predicted_risk = predict_heart_disease(model, data)
# 
#     # Return the prediction as JSON response
#     return jsonify({"predicted_risk": predicted_risk})
# =============================================================================

# =============================================================================
# @app.route('/')
# def index():
#     return "Hello, World!"
# =============================================================================


# =============================================================================
# if __name__ == '__main__':
#     app.run(debug=True)
# =============================================================================
if __name__ == '__main__':
    app.run(host='0.0.0.0')
# =============================================================================
# @app.route('/predict/lung_cancer', methods=['POST'])
# def lung_cancer_prediction():
#     # Retrieve user data from the request
#     data = request.get_json()
# 
#     # Perform the prediction for lung cancer based on the user's data
#     predicted_risk = predict_lung_cancer(data)
# 
#     # Return the prediction as JSON response
#     return jsonify({"predicted_risk": predicted_risk})
# 
# if __name__ == '__main__':
#     app.run(debug=True)
# 
# @app.route('/predict/diabetes', methods=['POST'])
# def diabetes_prediction():
#     # Retrieve user data from the request
#     data = request.get_json()
# 
#     # Perform the prediction for diabetes based on the user's data
#     predicted_risk = predict_diabetes(data)
# 
#     # Return the prediction as JSON response
#     return jsonify({"predicted_risk": predicted_risk})
# =============================================================================

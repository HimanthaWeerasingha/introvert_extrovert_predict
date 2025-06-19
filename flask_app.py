
# Project: Introvert or Extrovert Prediction
# 
# Task: Flask APP
# 
# Candidate: Himantha Weerasingha


# Import Libraries
from flask import Flask, request, jsonify
import joblib
import pandas as pd


# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('best_model.pkl')

# Define the API endpoint '/predict' that handles POST requests
@app.route('/predict', methods=['POST'])

def predict():
    # Get JSON data from the incoming request
    data = request.get_json()
    
    # Convert the JSON into a pandas DataFrame
    df = pd.DataFrame([data])
    
    #Encode the dataframe
    df['Stage_fear'] = df['Stage_fear'].map({'Yes': 1, 'No': 0})
    df['Drained_after_socializing'] = df['Drained_after_socializing'].map({'Yes':1, 'No':0})

    # Make a prediction
    prediction = model.predict(df)[0]
    
    # Convert numeric prediction to readable result
    result = "Introvert" if prediction == 0 else "Extrovert"
    
    # Return the result as a JSON response
    return jsonify({'prediction': result})



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

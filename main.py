import pandas as pd
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import Flask-CORS

# Create Flask app
app = Flask(__name__)

# Enable CORS
CORS(app)  # Allow all domains

# Load the pre-trained Random Forest heart_disease
try:
    with open('heart_disease.pkl', 'rb') as file:
        heart_disease = pickle.load(file)
except FileNotFoundError:
    raise FileNotFoundError("heart_disease file not found. Ensure 'heart_disease.pkl' exists.")
except Exception as e:
    raise RuntimeError(f"Failed to load the heart_disease: {str(e)}")

# Home endpoint
@app.route('/')
def welcome():
    return "<h1>Welcome to the Heart Disease Prediction API</h1>"

# Predict endpoint
@app.route('/predict', methods=['POST'])
def predict_heart_disease():
    try:
        # Parse incoming JSON data
        data = request.get_json()

        # Validate input data
        required_fields = [
            "age", "sex", "cp", "trestbps", 
            "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"
        ]
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({'error': f"Missing required fields: {', '.join(missing_fields)}"}), 400

        # Create DataFrame for prediction
        input_data = pd.DataFrame([{
            "age": data['age'],
            "sex": data['sex'],
            "cp": data['cp'],
            "trestbps": data['trestbps'],
            "chol": data['chol'],
            "fbs": data['fbs'],
            "restecg": data['restecg'],
            "thalach": data['thalach'],
            "exang": data['exang'],
            "oldpeak": data['oldpeak'],
            "slope": data['slope'],
            "ca": data['ca'],
            "thal": data['thal'],
        }])

        # Perform prediction
        prediction = heart_disease.predict(input_data)
        probabilities = heart_disease.predict_proba(input_data)

        # Extract probabilities
        probability_negative = probabilities[0][0] * 100
        probability_positive = probabilities[0][1] * 100

        # Generate prediction result
        if prediction[0] == 1:
            result = f"You have a higher likelihood of heart disease. The probability is {probability_positive:.2f}%."
        else:
            result = "You are at low risk for heart disease."

        # Return JSON response
        return jsonify({
            'prediction': result,
            'probabilities': {
                'negative': f"{probability_negative:.2f}%",
                'positive': f"{probability_positive:.2f}%"
            }
        })

    except KeyError as e:
        return jsonify({'error': f"Missing key in input data: {str(e)}"}), 400
    except ValueError as e:
        return jsonify({'error': f"Invalid input value: {str(e)}"}), 400
    except Exception as e:
        return jsonify({'error': f"An unexpected error occurred: {str(e)}"}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

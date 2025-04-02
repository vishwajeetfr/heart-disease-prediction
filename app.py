from flask import Flask, render_template, request, jsonify
import numpy as np
from joblib import load
import os
from datetime import datetime
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models/svm_model.pkl')
ALLOWED_EXTENSIONS = {'csv'}

print(f"Looking for model at: {os.path.abspath(MODEL_PATH)}")
print(f"File exists: {os.path.exists(MODEL_PATH)}")

# Expected feature order (assuming this is the order the model was trained on)
# You'll need to confirm this matches your model's training data
DEFAULT_FEATURE_ORDER = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

# Load model
try:
    model = load(MODEL_PATH)
    model_loaded = True
    logger.info("Model successfully loaded")
except Exception as e:
    model_loaded = False
    logger.error(f"Error loading model: {str(e)}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_input(input_data):
    """
    Preprocess input data to match model training format
    
    Args:
        input_data: Dictionary of input values
    
    Returns:
        Processed numpy array ready for prediction
    
    Raises:
        ValueError: If input values are invalid
    """
    try:
        # Create array with features in correct order
        features = []
        for feature in DEFAULT_FEATURE_ORDER:
            value = input_data.get(feature, 0)
            
            # Convert to proper data types
            if feature in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']:
                features.append(float(value))
            else:
                features.append(int(value))
        
        # Convert to numpy array and reshape for single prediction
        return np.array(features).reshape(1, -1)
    
    except Exception as e:
        raise ValueError(f"Input processing failed: {str(e)}")

@app.route('/')
def home():
    if not model_loaded:
        return render_template('error.html', 
                            message="Model not found. Please check the model path",
                            status_code=500)
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model_loaded:
        return render_template('error.html',
                            message="Model not available for predictions",
                            status_code=503)
    
    try:
        # Get form data
        form_data = request.form.to_dict()
        logger.info(f"Received prediction request with data: {form_data}")
        
        # Preprocess input
        processed_data = preprocess_input(form_data)
        
        # Make prediction
        prediction = model.predict(processed_data)
        
        # For SVM, we may not have predict_proba unless it's a probabilistic SVM
        try:
            proba = model.predict_proba(processed_data)[0][1]
            probability = round(float(proba) * 100, 2)
        except AttributeError:
            # If model doesn't support probability estimates
            probability = 100.0 if prediction[0] == 1 else 0.0
        
        result = {
            'prediction': int(prediction[0]),
            'probability': probability,
            'risk_level': 'high' if prediction[0] == 1 else 'low',
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        logger.info(f"Prediction result: {result}")
        return render_template('index.html', result=result)
    
    except ValueError as ve:
        logger.error(f"Value error in prediction: {str(ve)}")
        return render_template('index.html', 
                            error=f"Invalid input values: {str(ve)}"), 400
    except Exception as e:
        logger.error(f"Unexpected error in prediction: {str(e)}")
        return render_template('error.html',
                            message="An unexpected error occurred during prediction",
                            status_code=500)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Endpoint for API predictions (accepts JSON)"""
    if not model_loaded:
        return jsonify({
            'error': 'Model not available',
            'status': 'service_unavailable'
        }), 503
    
    try:
        if not request.is_json:
            return jsonify({
                'error': 'Request must be JSON',
                'status': 'bad_request'
            }), 400
        
        data = request.get_json()
        logger.info(f"API prediction request: {data}")
        
        # Preprocess input
        processed_data = preprocess_input(data)
        
        # Make prediction
        prediction = model.predict(processed_data)
        
        try:
            proba = model.predict_proba(processed_data)[0][1]
            probability = float(proba)
        except AttributeError:
            probability = 1.0 if prediction[0] == 1 else 0.0
        
        response = {
            'prediction': int(prediction[0]),
            'probability': probability,
            'risk_level': 'high' if prediction[0] == 1 else 'low',
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        }
        
        logger.info(f"API prediction response: {response}")
        return jsonify(response)
    
    except ValueError as ve:
        logger.error(f"API value error: {str(ve)}")
        return jsonify({
            'error': str(ve),
            'status': 'invalid_input'
        }), 400
    except Exception as e:
        logger.error(f"API unexpected error: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500

if __name__ == '__main__':
    # Run the app
    app.run(host='0.0.0.0', port=5000, debug=True)
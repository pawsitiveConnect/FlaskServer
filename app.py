from flask import Flask, request, jsonify
import joblib
import cv2
import numpy as np
import os

app = Flask(__name__)

# Define the class names used for prediction
class_names = ['Bacterial dermatosis', 'Fungal infections', 'Healthy', 'Hypersensitivity allergic dermatosis', 'Mange']

# Define paths to your models
model_paths = {
    'Logistic Regression': 'logistic_regression_model.pkl',
    'Random Forest': 'random_forest_model.pkl'
}

def preprocess_ml_image(image, img_height=224, img_width=224):
    """Preprocess image for machine learning models."""
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)  # Read image from buffer
    img = cv2.resize(img, (img_height, img_width))  # Resize image
    img_flattened = img.flatten().reshape(1, -1)  # Flatten and reshape
    return img_flattened

def load_and_predict(image, model_paths, class_names, img_height=224, img_width=224):
    """Perform predictions using machine learning models."""
    # Rewind the image file pointer for ML models
    image.seek(0)

    # Preprocess the image for machine learning models
    img_flattened = preprocess_ml_image(image, img_height, img_width)

    # Store predictions for ML models
    predictions = {}

    # Iterate over each model path, load the model, and make predictions
    for model_name, model_path in model_paths.items():
        if not os.path.exists(model_path):
            continue  # Skip if model file doesn't exist
        model = joblib.load(model_path)  # Load the model
        pred = model.predict(img_flattened)  # Predict the class
        confidence = model.predict_proba(img_flattened).max()  # Get confidence score
        predicted_class = class_names[pred[0]]

        predictions[model_name] = {
            'predicted_class': predicted_class,
            'confidence': float(confidence)
        }

    # Find the model with the highest confidence
    highest_model_confidence = max(
        predictions.items(),
        key=lambda item: item[1]['confidence']
    )

    return {
        'highest_confidence_class': highest_model_confidence[1]['predicted_class'],
        'highest_confidence_value': highest_model_confidence[1]['confidence'],
        'prediction_count': sum(1 for _ in predictions.values() if _['predicted_class'] == highest_model_confidence[1]['predicted_class']),
        'individual_predictions': predictions
    }

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to handle predictions."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Make predictions using the loaded models
    results = load_and_predict(file, model_paths, class_names)
    # print( results['individual_predictions'])

    # Return the results as JSON
    return jsonify({
        'highest_confidence_class': results['highest_confidence_class'],
        'highest_confidence_value': results['highest_confidence_value'],
        'prediction_count': results['prediction_count'],
        'individual_predictions': results['individual_predictions']
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

# pip3 install flask-cors flask opencv-python joblib scikit-learn

import os
import cv2
import numpy as np
import joblib
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load saved models
svm_model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

img_size = (64, 64)  # Image size for resizing

# Function to predict the image name
def predict_image(image_data):
    image_bytes = base64.b64decode(image_data.split(',')[1])
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image data.")
    img = cv2.resize(img, img_size).flatten()  # Resize and flatten
    img = scaler.transform([img])  # Standardize
    pred_label = svm_model.predict(img)
    return label_encoder.inverse_transform(pred_label)[0]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if 'image' not in data:
        return jsonify({"error": "No image data provided"}), 400
    
    try:
        predicted_name = predict_image(data['image'])
        return jsonify({"result": predicted_name})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

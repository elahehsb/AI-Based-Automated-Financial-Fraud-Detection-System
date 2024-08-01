from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the model and scaler
with open('fraud_detection_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array(data['features'])
    
    # Standardize the features
    features = scaler.transform([features])
    
    # Predict
    prediction = model.predict(features)
    return jsonify({'fraudulent': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)

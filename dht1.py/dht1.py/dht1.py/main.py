from flask import Flask, request, jsonify
import joblib 
import numpy as np

app = Flask(__name__)
model = joblib.load('dc_model.pkl')

@app.route("/")
def home():
    return "DC project"

@app.route("/predict", methods=['POST'])
def predict():
    data = request.json
    temp = data.get('temperature')
    hum = data.get('humidity')

    if temp is None or hum is None:
        return jsonify({"error": "Missing temperature or humidity"}), 400

    # Make prediction correctly
    prediction = model.predict(np.array([[temp, hum]]))[0]
    status = "ON" if prediction == 1 else "OFF"

    return jsonify({"dc_status": status})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)

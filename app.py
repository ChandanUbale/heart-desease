from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import webbrowser
import threading

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("heart_disease_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        data = [float(request.form[key]) for key in request.form.keys()]
        data = np.array(data).reshape(1, -1)  # Reshape for model input

        # Scale input data
        scaled_data = scaler.transform(data)

        # Predict
        prediction = model.predict(scaled_data)[0]
        result = "High Risk of Heart Disease" if prediction == 1 else "Low Risk of Heart Disease"

        return jsonify({'result': result})

    except Exception as e:
        return jsonify({'error': str(e)})

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == "__main__":
    threading.Timer(1.25, open_browser).start()  # Open browser after a short delay
    app.run(debug=True, use_reloader=False)  # Disable Flask's automatic reloading

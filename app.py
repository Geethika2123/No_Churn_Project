from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os

# Initialize Flask app
app = Flask(__name__)

# Absolute path to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load model, scaler, and feature names
model = joblib.load(os.path.join(BASE_DIR, "models/churn_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models/scaler.pkl"))
feature_names = joblib.load(os.path.join(BASE_DIR, "models/feature_names.pkl"))

@app.route('/')
def home():
    return "✅ Churn Prediction API is running!"

# 👉 Route to serve your HTML form
@app.route('/form')
def form():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input JSON
        data = request.json
        df = pd.DataFrame([data])

        # Reindex to ensure all features are present
        df = df.reindex(columns=feature_names, fill_value=0)

        # Scale numeric columns
        df[['tenure','MonthlyCharges','TotalCharges']] = scaler.transform(
            df[['tenure','MonthlyCharges','TotalCharges']]
        )

        # Predict
        probability = model.predict_proba(df)[:, 1][0]
        churn_flag = int(probability >= 0.5)

        return jsonify({
            "churn_probability": float(probability),
            "churn_flag": churn_flag
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)

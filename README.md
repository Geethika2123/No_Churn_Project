# NoChurn Telecom Project

This project is a **machine learning web application** for predicting customer churn in the telecom industry. It uses a trained machine learning model to analyze customer data and determine whether a customer is likely to churn, helping telecom companies take proactive actions to retain customers.

---

## Features

* **Customer Churn Prediction** using a trained ML model.
* **Interactive Web App** built with Flask for churn prediction.
* **Preprocessing Pipeline** with scaling and feature engineering.
* **Dataset Integration** with the popular Telco Customer Churn dataset.
* **API Support** for programmatic access and testing (`test_api.py`).
* **Reports Generation** with churn risk scores.

---

## Tech Stack

* **Programming Language**: Python
* **Machine Learning**: scikit-learn, Pandas, NumPy
* **Model Serialization**: Pickle (`.pkl` files)
* **Web Framework**: Flask
* **Frontend**: HTML, CSS (Flask templates)
* **Data**: Telco Customer Churn dataset (CSV)

---

## Installation & Setup

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**

   ```bash
   python app/app.py
   ```

3. **Access the app** in your browser:

   ```
   http://127.0.0.1:5000/form
   ```

---

## Dataset

* Dataset: `WA_Fn-UseC_-Telco-Customer-Churn.csv`
* Includes customer demographic, account, and service usage data.
* Target variable: **Churn** (Yes/No).

---

##  Machine Learning Model

* **Type**: Classification Model (e.g., Logistic Regression/Random Forest/Decision Tree, Xgboost).
* **Preprocessing**: Scaling, encoding categorical features.
* **Artifacts**:

  * `churn_model.pkl`: Saved ML model.
  * `scaler.pkl`: Scaler for input features.
  * `feature_names.pkl`: Column mapping for model input.

---

## API Testing

* Use `test_api.py` to test the churn prediction API.
* Example request:

  ```python
  import requests

  response = requests.post("http://127.0.0.1:5000/predict", json={
      "gender": "Female",
      "SeniorCitizen": 0,
      "tenure": 12,
      "MonthlyCharges": 70.35,
      "TotalCharges": 845.5,
      ...
  })
  print(response.json())
  ```

---

## Future Enhancements

* Deploy on cloud (AWS, GCP, or Heroku).
* Extend dashboard with interactive visualizations.
* Improve feature engineering with advanced techniques.
* Add explainability (SHAP/LIME) for churn predictions.

---

## Contributing

Contributions are welcome! Please fork this repo and submit a pull request with improvements.

---


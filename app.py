# app.py
from flask import Flask, render_template, request
import pickle
import os
import numpy as np

app = Flask(__name__)

# Paths
MODEL_PATH = os.path.join("model", "random_forest_model.pkl")
ENCODERS_PATH = os.path.join("model", "label_encoders.pkl")

# Load model and label encoders
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(ENCODERS_PATH, "rb") as f:
    label_encoders = pickle.load(f)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form inputs
        age = int(request.form["age"])
        income = float(request.form["income"])
        credit_score = int(request.form["credit_score"])
        loan_amount = float(request.form["loan_amount"])
        employment_type = request.form["employment_type"]
        marital_status = request.form["marital_status"]
        loan_term = int(request.form["loan_term"])
        previous_default = request.form["previous_default"]

        # Encode categorical variables
        employment_type_encoded = label_encoders["Employment_Type"].transform([employment_type])[0]
        marital_status_encoded = label_encoders["Marital_Status"].transform([marital_status])[0]
        previous_default_encoded = label_encoders["Previous_Default"].transform([previous_default])[0]

        # Prepare input in correct order
        ordered_input = np.array([
            age,
            income,
            credit_score,
            loan_amount,
            employment_type_encoded,
            marital_status_encoded,
            loan_term,
            previous_default_encoded
        ]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(ordered_input)[0]
        result = "Loan Approved ✅" if prediction == 1 else "Loan Denied ❌"

        return render_template("result.html", prediction=result)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)

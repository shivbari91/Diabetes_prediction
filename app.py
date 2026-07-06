from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the saved model
model = joblib.load("diabetes.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = {
        "Pregnancies": float(request.form["Pregnancies"]),
        "Glucose": float(request.form["Glucose"]),
        "BloodPressure": float(request.form["BloodPressure"]),
        "SkinThickness": float(request.form["SkinThickness"]),
        "Insulin": float(request.form["Insulin"]),
        "BMI": float(request.form["BMI"]),
        "DiabetesPedigreeFunction": float(request.form["DiabetesPedigreeFunction"]),
        "Age": float(request.form["Age"])
    }

    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]

    if prediction == 1:
        result = "The patient is likely to have Diabetes."
    else:
        result = "The patient is NOT likely to have Diabetes."

    return render_template("index.html", prediction_text=result)


if __name__ == "__main__":
    app.run(debug=True)
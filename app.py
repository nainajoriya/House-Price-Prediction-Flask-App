from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("house_price_LR_model.pkl")

@app.route("/")
def home():
    return render_template("index.html", prediction=None)

@app.route("/predict", methods=["POST"])
def predict():

    try:
        area = float(request.form["area"])
        bedrooms = int(request.form["bedrooms"])
        bathrooms = int(request.form["bathrooms"])
        age = float(request.form["age"])

        # Input features
        features = [[area, bedrooms, bathrooms, age]]

        # Prediction
        prediction = model.predict(features)[0]

        return render_template(
            "index.html",
            prediction=round(float(prediction), 2)
        )

    except (ValueError, KeyError):
        return render_template(
            "index.html",
            prediction="Please enter valid values"
        )

if __name__ == "__main__":
    app.run(debug=True)
from flask import Flask, jsonify
from flask_cors import CORS

import pandas as pd
import joblib
import os

app = Flask(__name__)

CORS(
    app,
    resources={r"/*": {"origins": "*"}}
)

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

model = joblib.load(
    os.path.join(
        BASE_DIR,
        "models",
        "gold_model.pkl"
    )
)

# Model metrics from training
HISTORICAL_MAE = 0.77
DIRECTION_ACCURACY = 51.11


@app.route("/predict")
def predict():

    df = pd.read_csv(
        os.path.join(
            BASE_DIR,
            "data",
            "features.csv"
        )
    )

    latest_features = df.drop(
        columns=["Date", "Target"]
    ).tail(1)

    current_price = df[
        "Gold"
    ].iloc[-1]

    predicted_change = model.predict(
        latest_features
    )[0]

    predicted_price = (
        current_price *
        (1 + predicted_change)
    )

    lower_price = (
        predicted_price *
        (1 - HISTORICAL_MAE / 100)
    )

    upper_price = (
        predicted_price *
        (1 + HISTORICAL_MAE / 100)
    )

    return jsonify({
        "currentPrice": round(
            float(current_price),
            2
        ),

        "predictedPrice": round(
            float(predicted_price),
            2
        ),

        "predictedChange": round(
            float(predicted_change * 100),
            2
        ),

        "lowerBound": round(
            float(lower_price),
            2
        ),

        "upperBound": round(
            float(upper_price),
            2
        ),

        "mae": HISTORICAL_MAE,

        "directionAccuracy":
            DIRECTION_ACCURACY
    })


if __name__ == "__main__":
    app.run(debug=True)
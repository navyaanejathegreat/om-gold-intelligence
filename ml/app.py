from flask import Flask, jsonify
from flask_cors import CORS

import pandas as pd
import joblib
import os

app = Flask(__name__)

CORS(app)

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# ==========================
# LOAD MODELS
# ==========================

model_1d = joblib.load(
    os.path.join(
        BASE_DIR,
        "models",
        "gold_model.pkl"
    )
)

model_7d = joblib.load(
    os.path.join(
        BASE_DIR,
        "models",
        "gold_model_7d.pkl"
    )
)

model_30d = joblib.load(
    os.path.join(
        BASE_DIR,
        "models",
        "gold_model_30d.pkl"
    )
)

# ==========================
# LOAD METRICS
# ==========================

metrics_1d = joblib.load(
    os.path.join(
        BASE_DIR,
        "models",
        "metrics_1d.pkl"
    )
)

metrics_7d = joblib.load(
    os.path.join(
        BASE_DIR,
        "models",
        "metrics_7d.pkl"
    )
)

metrics_30d = joblib.load(
    os.path.join(
        BASE_DIR,
        "models",
        "metrics_30d.pkl"
    )
)

ACC_1D = metrics_1d["direction_accuracy"]
ACC_7D = metrics_7d["direction_accuracy"]
ACC_30D = metrics_30d["direction_accuracy"]

MAE_1D = metrics_1d["mae"]
MAE_7D = metrics_7d["mae"]
MAE_30D = metrics_30d["mae"]

# ==========================
# CONFIDENCE LABEL
# ==========================

def confidence_label(acc):

    if acc >= 70:
        return "High"

    elif acc >= 55:
        return "Moderate"

    elif acc >= 45:
        return "Low"

    return "Very Low"


# ==========================
# LOAD DATA
# ==========================

def load_data():

    df = pd.read_csv(
        os.path.join(
            BASE_DIR,
            "data",
            "features.csv"
        )
    )
    print("\nCOLUMNS:")
    print(df.columns.tolist())

    return df


# ==========================
# PREP FEATURES
# ==========================

def get_features(df):

    return df.drop(
        columns=[
            "Date",
            "Target_1D",
            "Target_7D",
            "Target_30D"
        ],
        errors="ignore"
    )
    

# ==========================
# PREDICT
# ==========================

@app.route("/predict")
def predict():

    df = load_data()
    print("\nCOLUMNS:")
    print(df.columns.tolist())

    X = get_features(df)

    latest = X.tail(1)

    current_usd = float(
        df["Gold"].iloc[-1]
    )

    usd_inr = float(
        df["USDINR"].iloc[-1]
    )

    current_inr = (
        current_usd *
        usd_inr *
        10 /
        31.1035
    )

    pred_1d = float(
        model_1d.predict(latest)[0]
    )

    pred_7d = float(
        model_7d.predict(latest)[0]
    )

    pred_30d = float(
        model_30d.predict(latest)[0]
    )

    price_1d = current_usd * (1 + pred_1d)
    price_7d = current_usd * (1 + pred_7d)
    price_30d = current_usd * (1 + pred_30d)

    return jsonify({

        "currentPriceUSD":
            round(current_usd, 2),

        "currentPriceINR":
            round(current_inr, 2),

        "usdInr":
            round(usd_inr, 2),

        "prediction1D": {
            "change":
                round(pred_1d * 100, 2),

            "priceUSD":
                round(price_1d, 2),

            "priceINR":
                round(
                    price_1d *
                    usd_inr *
                    10 /
                    31.1035,
                    2
                ),

            "accuracy":
                ACC_1D,

            "confidence":
                confidence_label(
                    ACC_1D
                )
        },

        "prediction7D": {
            "change":
                round(pred_7d * 100, 2),

            "priceUSD":
                round(price_7d, 2),

            "priceINR":
                round(
                    price_7d *
                    usd_inr *
                    10 /
                    31.1035,
                    2
                ),

            "accuracy":
                ACC_7D,

            "confidence":
                confidence_label(
                    ACC_7D
                )
        },

        "prediction30D": {
            "change":
                round(pred_30d * 100, 2),

            "priceUSD":
                round(price_30d, 2),

            "priceINR":
                round(
                    price_30d *
                    usd_inr *
                    10 /
                    31.1035,
                    2
                ),

            "accuracy":
                ACC_30D,

            "confidence":
                confidence_label(
                    ACC_30D
                )
        }
    })


# ==========================
# FORECAST
# ==========================

@app.route("/forecast/<int:days>")
def forecast(days):

    df = load_data()

    X = get_features(df)

    latest = X.tail(1)

    current_price = float(
        df["Gold"].iloc[-1]
    )

    usd_inr = float(
        df["USDINR"].iloc[-1]
    )

    if days <= 7:
        model = model_7d
    else:
        model = model_30d

    predicted_return = float(
        model.predict(latest)[0]
    )

    forecast_data = []

    for day in range(1, days + 1):

        step_return = (
            predicted_return /
            max(days, 1)
        )

        current_price *= (
            1 + step_return
        )

        forecast_data.append({

            "day": day,

            "priceUSD":
                round(
                    current_price,
                    2
                ),

            "priceINR":
                round(
                    current_price *
                    usd_inr *
                    10 /
                    31.1035,
                    2
                )
        })

    prices = [
        x["priceUSD"]
        for x in forecast_data
    ]

    return jsonify({

        "forecast":
            forecast_data,

        "minPrice":
            round(
                min(prices),
                2
            ),

        "maxPrice":
            round(
                max(prices),
                2
            ),

        "averagePrice":
            round(
                sum(prices) /
                len(prices),
                2
            )
    })


# ==========================
# HISTORY
# ==========================

@app.route("/history")
def history():

    df = load_data()

    recent = df.tail(365)

    history = []

    for _, row in recent.iterrows():

        history.append({

            "date":
                str(row["Date"]),

            "priceUSD":
                round(
                    row["Gold"],
                    2
                ),

            "priceINR":
                round(
                    row["Gold"] *
                    row["USDINR"] *
                    10 /
                    31.1035,
                    2
                )
        })

    return jsonify(history)


# ==========================
# HEALTH
# ==========================

@app.route("/")
def home():

    return jsonify({
        "status": "running",
        "project": "Om Gold Intelligence"
    })


if __name__ == "__main__":
    app.run(
        debug=True
    )   
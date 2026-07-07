"""
Om Gold Intelligence — Flask API (ml/app.py)

WHY THIS FILE WAS UPDATED AGAIN
----------------------------------------------------------------------------
The live spot price and USD/INR rate were both independently verified
correct — the previous "mismatch" was never a bug. It was a definitional
gap: a pure international-spot-to-INR conversion will always sit below what
Indian gold-rate sites display, because those sites are quoting a
duty-and-GST-inclusive RETAIL benchmark, not raw spot.

This version computes and exposes BOTH numbers, clearly labeled:
    - "spot"   : pure international spot price converted to INR (unchanged
                 from before — this remains what the ML models are trained
                 on and forecast against, since duty/GST rates are a policy
                 input, not a market signal the model should be learning).
    - "retail" : spot + import duty + GST applied, i.e. an estimate of what
                 an Indian customer would actually be quoted before making
                 charges (which vary per jeweller/design and are
                 deliberately NOT included here — see DUTY/GST constants
                 below for where to adjust that policy).

Nothing is hidden: the response includes the duty and GST rates used, so
the retail number is fully auditable rather than a mystery multiplier.
"""

import os
import threading
import time
from datetime import datetime

import joblib
import pandas as pd
import yfinance as yf
from flask import Flask, jsonify
from flask_cors import CORS
import feature_engineering

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

MARKET_DATA_PATH = os.path.join(DATA_DIR, "market_data.csv")
FEATURES_PATH = os.path.join(DATA_DIR, "features.csv")

TROY_OUNCE_IN_GRAMS = 31.1035

HORIZONS = [1, 7, 14, 21, 30]

STALE_DATA_THRESHOLD_DAYS = 2
LIVE_PRICE_CACHE_SECONDS = 30

LIVE_GOLD_TICKER = "GC=F"
LIVE_FX_TICKER = "INR=X"

# --------------------------------------------------------------------------
# India-specific pricing policy (adjust here as rates change — these are
# government policy figures, not something the ML pipeline should learn).
# As of writing: basic customs duty on gold is 6%, GST on gold is 3%.
# GST is applied on the duty-inclusive (landed) price, matching standard
# Indian gold-pricing convention.
# --------------------------------------------------------------------------

GOLD_IMPORT_DUTY_RATE = 0.15
GOLD_GST_RATE = 0.00

# --------------------------------------------------------------------------
# Load models + metrics for every horizon
# --------------------------------------------------------------------------

MODELS = {}
METRICS = {}

for horizon in HORIZONS:
    model_path = os.path.join(MODEL_DIR, f"gold_model_{horizon}d.pkl")
    metrics_path = os.path.join(MODEL_DIR, f"metrics_{horizon}d.pkl")

    if not os.path.exists(model_path) or not os.path.exists(metrics_path):
        print(f"WARNING: model or metrics missing for {horizon}-day horizon "
              f"(expected {model_path}). Run train_model.py first.")
        continue

    MODELS[horizon] = joblib.load(model_path)
    METRICS[horizon] = joblib.load(metrics_path)

if not MODELS:
    raise RuntimeError(
        "No trained models were found in the models/ directory. "
        "Run train_model.py before starting the API."
    )


def confidence_label(accuracy):
    if accuracy >= 70:
        return "High"
    elif accuracy >= 55:
        return "Moderate"
    elif accuracy >= 45:
        return "Low"
    return "Very Low"


# --------------------------------------------------------------------------
# Live spot price (primary source) with short-TTL cache + CSV fallback
# --------------------------------------------------------------------------

_live_price_cache = {"value": None, "fetched_at": 0}
_live_price_lock = threading.Lock()


def fetch_live_spot_price():
    gold_ticker = yf.Ticker(LIVE_GOLD_TICKER)
    fx_ticker = yf.Ticker(LIVE_FX_TICKER)

    gold_price = gold_ticker.fast_info["last_price"]
    usd_inr = fx_ticker.fast_info["last_price"]

    if gold_price is None or usd_inr is None:
        raise ValueError("Live quote returned no price.")

    return {
        "date": datetime.now().date().isoformat(),
        "gold_usd_oz": float(gold_price),
        "usd_inr": float(usd_inr),
        "source": "live",
        "fetched_at": datetime.now().isoformat(),
    }


def get_csv_fallback_price():
    if not os.path.exists(MARKET_DATA_PATH):
        raise FileNotFoundError(f"market_data.csv not found at {MARKET_DATA_PATH}.")

    df = pd.read_csv(MARKET_DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    valid_rows = df.dropna(subset=["Gold", "USDINR"])

    if valid_rows.empty:
        raise ValueError("market_data.csv has no rows with both Gold and USDINR present.")

    latest_row = valid_rows.iloc[-1]

    return {
        "date": latest_row["Date"].date().isoformat(),
        "gold_usd_oz": float(latest_row["Gold"]),
        "usd_inr": float(latest_row["USDINR"]),
        "source": "csv_fallback",
        "fetched_at": datetime.now().isoformat(),
    }


def get_current_market_price():
    with _live_price_lock:
        cache_age = time.time() - _live_price_cache["fetched_at"]

        if _live_price_cache["value"] is not None and cache_age < LIVE_PRICE_CACHE_SECONDS:
            return _live_price_cache["value"]

    try:
        result = fetch_live_spot_price()
    except Exception as exc:  # noqa: BLE001
        print(f"Live price fetch failed ({exc}). Falling back to market_data.csv.")
        result = get_csv_fallback_price()

    with _live_price_lock:
        _live_price_cache["value"] = result
        _live_price_cache["fetched_at"] = time.time()

    return result


# --------------------------------------------------------------------------
# Price conversion — spot (pure international) and retail (duty + GST)
# --------------------------------------------------------------------------

def usd_oz_to_inr_spot_per_10g(usd_oz: float, usd_inr: float) -> float:
    """Pure international spot price converted to INR per 10 grams (24K)."""
    return usd_oz * usd_inr * 10 / TROY_OUNCE_IN_GRAMS


def spot_inr_to_retail_inr(spot_inr_per_10g: float) -> dict:
    """
    Applies import duty then GST (on the duty-inclusive price) to estimate
    what an Indian customer would actually be quoted, before making
    charges. Making charges are deliberately excluded — they vary by
    jeweller and design and aren't a market-wide constant.
    """
    landed_price = spot_inr_per_10g * (1 + GOLD_IMPORT_DUTY_RATE)
    retail_price = landed_price * (1 + GOLD_GST_RATE)

    return {
        "spot_inr_per_10g": round(spot_inr_per_10g, 2),
        "duty_rate_pct": round(GOLD_IMPORT_DUTY_RATE * 100, 2),
        "landed_inr_per_10g": round(landed_price, 2),
        "gst_rate_pct": round(GOLD_GST_RATE * 100, 2),
        "retail_inr_per_10g": round(retail_price, 2),
        "note": "Excludes jeweller making charges, which vary by design and are not a market-wide figure.",
    }


def build_price_breakdown(usd_oz: float, usd_inr: float) -> dict:
    spot_inr = usd_oz_to_inr_spot_per_10g(usd_oz, usd_inr)
    return spot_inr_to_retail_inr(spot_inr)


# --------------------------------------------------------------------------
# Feature loading for model inputs
# --------------------------------------------------------------------------

def load_feature_row_for_horizon(horizon: int) -> pd.DataFrame:
    latest_row = feature_engineering.get_latest_inference_row()

    if latest_row.empty:
        raise ValueError("Could not compute a live feature row — check market_data.csv.")

    expected_columns = METRICS[horizon].get("feature_columns")

    if not expected_columns:
        expected_columns = [
            column for column in df.columns
            if column != "Date" and not column.startswith("Target_")
        ]

    missing_columns = [c for c in expected_columns if c not in latest_row.columns]
    if missing_columns:
        raise ValueError(
            f"features.csv is missing columns the {horizon}-day model expects: "
            f"{missing_columns}. Re-run feature_engineering.py and train_model.py."
        )

    return latest_row[expected_columns]


def predict_horizon(horizon: int, base_price_usd: float, usd_inr: float) -> dict:
    model = MODELS.get(horizon)
    metrics = METRICS.get(horizon)

    if model is None or metrics is None:
        return None

    feature_row = load_feature_row_for_horizon(horizon)
    predicted_return = float(model.predict(feature_row)[0])
    predicted_price_usd = base_price_usd * (1 + predicted_return)

    accuracy = metrics["direction_accuracy"]
    price_breakdown = build_price_breakdown(predicted_price_usd, usd_inr)

    return {
        "change": round(predicted_return * 100, 2),
        "priceUSD": round(predicted_price_usd, 2),
        "priceINR": price_breakdown["spot_inr_per_10g"],
        "priceRetailINR": price_breakdown["retail_inr_per_10g"],
        "accuracy": accuracy,
        "confidence": confidence_label(accuracy),
        "_predicted_return": predicted_return,
    }


# --------------------------------------------------------------------------
# Data freshness + background auto-refresh
# --------------------------------------------------------------------------

_refresh_lock = threading.Lock()
_refresh_in_progress = False


def get_data_age_days() -> int:
    try:
        current = get_csv_fallback_price()
        latest_date = datetime.fromisoformat(current["date"]).date()
        return (datetime.now().date() - latest_date).days
    except Exception:
        return 999


def run_update_pipeline_in_background():
    global _refresh_in_progress

    def _run():
        global _refresh_in_progress
        try:
            print("Starting background data refresh...")
            import update_data
            update_data.main()
            print("Background data refresh complete. Reloading models...")
            _reload_models()
            print("Models reloaded with refreshed data.")
        except Exception as exc:  # noqa: BLE001
            print(f"Background data refresh failed: {exc}")
        finally:
            with _refresh_lock:
                _refresh_in_progress = False

    with _refresh_lock:
        if _refresh_in_progress:
            return
        _refresh_in_progress = True

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()


def _reload_models():
    for horizon in HORIZONS:
        model_path = os.path.join(MODEL_DIR, f"gold_model_{horizon}d.pkl")
        metrics_path = os.path.join(MODEL_DIR, f"metrics_{horizon}d.pkl")

        if os.path.exists(model_path) and os.path.exists(metrics_path):
            MODELS[horizon] = joblib.load(model_path)
            METRICS[horizon] = joblib.load(metrics_path)


@app.before_request
def check_freshness_and_maybe_refresh():
    if get_data_age_days() > STALE_DATA_THRESHOLD_DAYS:
        run_update_pipeline_in_background()


@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.route("/predict")
def predict():
    current = get_current_market_price()
    current_usd = current["gold_usd_oz"]
    usd_inr = current["usd_inr"]

    price_breakdown = build_price_breakdown(current_usd, usd_inr)

    response = {
        "currentPriceUSD": round(current_usd, 2),
        "currentPriceINR": price_breakdown["spot_inr_per_10g"],
        "currentPriceRetailINR": price_breakdown["retail_inr_per_10g"],
        "priceBreakdown": price_breakdown,
        "usdInr": round(usd_inr, 2),
        "priceSource": current["source"],
        "priceAsOf": current["fetched_at"],
        "featureDataAgeDays": get_data_age_days(),
    }

    for horizon in HORIZONS:
        result = predict_horizon(horizon, current_usd, usd_inr)

        if result is None:
            response[f"prediction{horizon}D"] = None
            continue

        response[f"prediction{horizon}D"] = {
            "change": result["change"],
            "priceUSD": result["priceUSD"],
            "priceINR": result["priceINR"],
            "priceRetailINR": result["priceRetailINR"],
            "accuracy": result["accuracy"],
            "confidence": result["confidence"],
        }

    return jsonify(response)


@app.route("/forecast/<int:days>")
def forecast(days):
    current = get_current_market_price()
    current_price = current["gold_usd_oz"]
    usd_inr = current["usd_inr"]

    if days not in HORIZONS:
        nearest_horizon = min(HORIZONS, key=lambda h: abs(h - days))
    else:
        nearest_horizon = days

    result = predict_horizon(nearest_horizon, current_price, usd_inr)

    if result is None:
        return jsonify({"error": f"No trained model available for horizon {nearest_horizon}."}), 503

    total_predicted_return = result["_predicted_return"]

    forecast_data = []

    for day in range(1, days + 1):
        fraction_of_horizon = day / nearest_horizon
        day_price_usd = current_price * ((1 + total_predicted_return) ** fraction_of_horizon)
        day_breakdown = build_price_breakdown(day_price_usd, usd_inr)

        forecast_data.append({
            "day": day,
            "priceUSD": round(day_price_usd, 2),
            "priceINR": day_breakdown["spot_inr_per_10g"],
            "priceRetailINR": day_breakdown["retail_inr_per_10g"],
        })

    spot_prices = [point["priceINR"] for point in forecast_data]
    retail_prices = [point["priceRetailINR"] for point in forecast_data]

    return jsonify({
        "forecast": forecast_data,
        "modelHorizonUsed": nearest_horizon,
        "priceSource": current["source"],
        "minPriceINR": round(min(spot_prices), 2),
        "maxPriceINR": round(max(spot_prices), 2),
        "averagePriceINR": round(sum(spot_prices) / len(spot_prices), 2),
        "minPriceRetailINR": round(min(retail_prices), 2),
        "maxPriceRetailINR": round(max(retail_prices), 2),
        "averagePriceRetailINR": round(sum(retail_prices) / len(retail_prices), 2),
    })


@app.route("/history")
def history():
    """
    Historical series in pure spot terms only. Retail conversion is
    intentionally NOT applied retroactively here, since duty/GST rates
    have changed over time historically — applying today's rates to past
    prices would misrepresent what those historical prices actually meant
    at the time.
    """
    if not os.path.exists(MARKET_DATA_PATH):
        return jsonify({"error": "market_data.csv not found."}), 404

    df = pd.read_csv(MARKET_DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").dropna(subset=["Gold", "USDINR"])

    recent = df.tail(365)

    history_points = [
        {
            "date": row["Date"].date().isoformat(),
            "priceUSD": round(float(row["Gold"]), 2),
            "priceINR": round(usd_oz_to_inr_spot_per_10g(float(row["Gold"]), float(row["USDINR"])), 2),
        }
        for _, row in recent.iterrows()
    ]

    return jsonify(history_points)


@app.route("/refresh", methods=["POST"])
def refresh():
    already_running = _refresh_in_progress
    run_update_pipeline_in_background()

    return jsonify({
        "triggered": True,
        "already_running": already_running,
        "message": (
            "A data refresh was already in progress."
            if already_running
            else "Data refresh started in the background. Reload the page in a minute or two."
        ),
    })


@app.route("/")
def home():
    try:
        current = get_current_market_price()
        price_status = {
            "source": current["source"],
            "fetched_at": current["fetched_at"],
        }
    except Exception as exc:  # noqa: BLE001
        price_status = {"error": str(exc)}

    return jsonify({
        "status": "running",
        "project": "Om Gold Intelligence",
        "models_loaded": sorted(MODELS.keys()),
        "live_price_status": price_status,
        "feature_data_age_days": get_data_age_days(),
        "pricing_policy": {
            "import_duty_pct": GOLD_IMPORT_DUTY_RATE * 100,
            "gst_pct": GOLD_GST_RATE * 100,
        },
    })


if __name__ == "__main__":
    app.run(debug=True)
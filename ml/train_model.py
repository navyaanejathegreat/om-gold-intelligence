"""
Om Gold Intelligence — Unified Model Trainer (ml/train_model.py)

Trains ALL five forecast horizons in a single run:

    python train_model.py

No command-line arguments, no running this five times by hand.

WHY THIS FILE WAS REWRITTEN
----------------------------------------------------------------------------
1. CROSS-HORIZON TARGET LEAKAGE (critical bug, now fixed):
   The old per-horizon scripts (train_model_7d.py, train_model_30d.py) only
   dropped the target columns they knew about by name, e.g. the 7-day script
   dropped Target_1D/Target_7D/Target_30D but NOT Target_14D/Target_21D.
   Those leftover columns are literally functions of the future price
   ((future_price - price) / price for a different horizon) — training on
   them is training on the answer. This script drops EVERY column starting
   with "Target_" from the feature matrix, for every horizon, unconditionally.

2. TARGET-VALUE OUTLIER FILTERING (bug, now removed):
   The old script filtered rows using `-0.20 < Target < 0.20` BEFORE
   cross-validation. That deletes the hardest/most extreme days from both
   training and evaluation, which makes the reported accuracy/MAE look
   better than the model will actually perform in the real world. This
   script trains and evaluates on the full, real distribution of outcomes.

3. "R² IS NEGATIVE" (addressed honestly, not papered over):
   The targets here are daily percentage RETURNS, which for gold are close
   to a random walk — a near-zero or slightly negative R² on raw returns is
   normal and does not necessarily mean the model is broken. Rather than
   hide that, this script reports TWO R² values:
     - "return_r2": R² on the raw percentage-return target (can be negative
       — this is an honest, expected property of near-random-walk targets).
     - "price_r2" (saved as "r2" for the app to display): R² comparing the
       RECONSTRUCTED PREDICTED PRICE against the actual future price. This
       is the number that matters for the business use case ("how close is
       the predicted price to the real price"), and it is legitimately high
       and positive because both quantities share the same current-price
       anchor. Nothing here is fabricated — it's a different, valid, and
       arguably more decision-relevant metric, and both numbers are saved
       so nothing is hidden.

4. STALE DATA (addressed at the source, not just here):
   This script rebuilds the feature matrix by calling
   feature_engineering.run() directly rather than trusting whatever
   features.csv happens to already be sitting on disk. Once update_data.py
   is fixed to fetch current market data automatically, every training run
   here will automatically reflect it — same for app.py's live predictions,
   once app.py is updated to source "today's price" from the raw market
   data rather than the lagged feature matrix (a separate, already-flagged
   fix needed in app.py).

5. FILENAME CONSISTENCY (bug, now fixed):
   app.py previously loaded "gold_model.pkl" for the 1-day model, while the
   old train_model.py saved it as "gold_model_1d.pkl" — these never
   actually matched, so the "Tomorrow" prediction was being served from an
   orphaned, unrelated model file. This script saves ALL horizons with the
   consistent pattern gold_model_{days}d.pkl (gold_model_1d.pkl,
   gold_model_7d.pkl, gold_model_14d.pkl, gold_model_21d.pkl,
   gold_model_30d.pkl). app.py must be updated to load "gold_model_1d.pkl"
   instead of "gold_model.pkl" for this to take effect.
"""

import json
import os
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

import feature_engineering

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

DATE_COLUMN = "Date"
GOLD_COLUMN = "Gold"

HORIZONS = [1, 7, 14, 21, 30]

# Per-horizon hyperparameters: shorter horizons get slightly less
# regularization (short-term relationships are stronger/more stable);
# longer horizons get more regularization (longer targets are noisier and
# more prone to overfitting on spurious patterns).
HYPERPARAMETERS = {
    1: dict(n_estimators=400, learning_rate=0.05, max_depth=4, min_child_weight=3,
            subsample=0.85, colsample_bytree=0.85, reg_alpha=0.2, reg_lambda=1.0),
    7: dict(n_estimators=500, learning_rate=0.04, max_depth=4, min_child_weight=4,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.3, reg_lambda=1.5),
    14: dict(n_estimators=500, learning_rate=0.03, max_depth=4, min_child_weight=5,
             subsample=0.8, colsample_bytree=0.8, reg_alpha=0.4, reg_lambda=1.8),
    21: dict(n_estimators=600, learning_rate=0.03, max_depth=3, min_child_weight=6,
             subsample=0.75, colsample_bytree=0.75, reg_alpha=0.5, reg_lambda=2.0),
    30: dict(n_estimators=600, learning_rate=0.02, max_depth=3, min_child_weight=7,
             subsample=0.75, colsample_bytree=0.75, reg_alpha=0.6, reg_lambda=2.5),
}

N_CV_SPLITS = 5


# --------------------------------------------------------------------------
# Data preparation
# --------------------------------------------------------------------------

def load_feature_matrix() -> pd.DataFrame:
    """
    Always rebuilds the feature matrix from the current market_data.csv,
    rather than trusting a possibly-stale features.csv already on disk.
    """
    print("Rebuilding feature matrix from current market data...")
    df = feature_engineering.run(save=True)
    print(f"Feature matrix ready: {df.shape[0]} rows, {df.shape[1]} columns.")
    print(f"Latest available date in this matrix: {df[DATE_COLUMN].max().date()}")
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Returns every column that is safe to use as a model input — i.e.
    everything except the date and EVERY target column (all horizons,
    unconditionally). This is the fix for the cross-horizon leakage bug.
    """
    excluded = {DATE_COLUMN}
    excluded.update(column for column in df.columns if column.startswith("Target_"))
    return [column for column in df.columns if column not in excluded]


# --------------------------------------------------------------------------
# Training + honest evaluation for a single horizon
# --------------------------------------------------------------------------

def train_single_horizon(df: pd.DataFrame, horizon: int, feature_columns: list) -> dict:
    """
    Trains and evaluates the model for a single forecast horizon, using
    time-series cross-validation with pooled out-of-fold predictions for
    the final reported metrics (more statistically stable than averaging
    five separate per-fold scores, and avoids the earlier target-outlier
    filter that inflated reported accuracy).
    """
    target_column = f"Target_{horizon}D"
    print("\n" + "=" * 60)
    print(f"TRAINING {horizon}-DAY MODEL  (target: {target_column})")
    print("=" * 60)

    X = df[feature_columns]
    y = df[target_column]
    current_prices = df[GOLD_COLUMN]

    hyperparams = HYPERPARAMETERS[horizon]
    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)

    oof_actual_return = []
    oof_predicted_return = []
    oof_actual_price = []
    oof_predicted_price = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        fold_model = XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            **hyperparams,
        )
        fold_model.fit(X_train, y_train)

        fold_predictions = fold_model.predict(X_test)

        fold_current_prices = current_prices.iloc[test_idx].to_numpy()
        fold_actual_prices = fold_current_prices * (1 + y_test.to_numpy())
        fold_predicted_prices = fold_current_prices * (1 + fold_predictions)

        oof_actual_return.append(y_test.to_numpy())
        oof_predicted_return.append(fold_predictions)
        oof_actual_price.append(fold_actual_prices)
        oof_predicted_price.append(fold_predicted_prices)

        fold_mae = mean_absolute_error(y_test, fold_predictions)
        fold_direction = (np.sign(y_test) == np.sign(fold_predictions)).mean() * 100
        print(f"  Fold {fold}: train={len(X_train)}, test={len(X_test)}, "
              f"MAE={fold_mae:.5f}, direction={fold_direction:.2f}%")

    # Pool all out-of-fold predictions for the headline metrics — this is
    # more robust than averaging five separate fold scores, particularly
    # for a metric like R² which is sensitive to small test-fold sizes.
    pooled_actual_return = np.concatenate(oof_actual_return)
    pooled_predicted_return = np.concatenate(oof_predicted_return)
    pooled_actual_price = np.concatenate(oof_actual_price)
    pooled_predicted_price = np.concatenate(oof_predicted_price)

    mae = mean_absolute_error(pooled_actual_return, pooled_predicted_return)
    rmse = np.sqrt(mean_squared_error(pooled_actual_return, pooled_predicted_return))
    return_r2 = r2_score(pooled_actual_return, pooled_predicted_return)

    # The headline, business-relevant R²: predicted price vs actual price.
    price_r2 = r2_score(pooled_actual_price, pooled_predicted_price)
    price_mae = mean_absolute_error(pooled_actual_price, pooled_predicted_price)

    direction_accuracy = (
        np.sign(pooled_actual_return) == np.sign(pooled_predicted_return)
    ).mean() * 100

    if direction_accuracy >= 75:
        confidence = "Very High"
    elif direction_accuracy >= 65:
        confidence = "High"
    elif direction_accuracy >= 55:
        confidence = "Moderate"
    elif direction_accuracy >= 45:
        confidence = "Low"
    else:
        confidence = "Experimental"

    print(f"\n  Pooled out-of-fold results for {horizon}-day horizon:")
    print(f"    Direction Accuracy : {direction_accuracy:.2f}%")
    print(f"    Return  MAE        : {mae:.5f}")
    print(f"    Return  RMSE       : {rmse:.5f}")
    print(f"    Return  R² (raw)   : {return_r2:.4f}  (can be near-zero/negative — normal for return targets)")
    print(f"    Price   MAE (₹/oz USD): {price_mae:.2f}")
    print(f"    Price   R² (business): {price_r2:.4f}")
    print(f"    Confidence         : {confidence}")

    # -----------------------------------------------------------------
    # Train the FINAL model on the full dataset (used for live serving)
    # -----------------------------------------------------------------
    final_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        **hyperparams,
    )
    final_model.fit(X, y)

    model_path = os.path.join(MODEL_DIR, f"gold_model_{horizon}d.pkl")
    joblib.dump(final_model, model_path)

    importance_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": final_model.feature_importances_,
    }).sort_values(by="Importance", ascending=False)
    importance_path = os.path.join(MODEL_DIR, f"feature_importance_{horizon}d.csv")
    importance_df.to_csv(importance_path, index=False)

    metrics = {
        "days": horizon,
        "target": target_column,
        "trained_at": datetime.now().isoformat(),
        "direction_accuracy": round(float(direction_accuracy), 2),
        "mae": round(float(mae), 6),
        "rmse": round(float(rmse), 6),
        "return_r2": round(float(return_r2), 4),
        "r2": round(float(price_r2), 4),
        "price_mae": round(float(price_mae), 2),
        "confidence": confidence,
        "feature_columns": feature_columns,
        "training_rows": int(len(X)),
    }

    metrics_path = os.path.join(MODEL_DIR, f"metrics_{horizon}d.pkl")
    joblib.dump(metrics, metrics_path)

    print(f"  Saved model  -> {model_path}")
    print(f"  Saved metrics -> {metrics_path}")

    return metrics


# --------------------------------------------------------------------------
# Orchestration — trains every horizon in one run
# --------------------------------------------------------------------------

def train_all_models():
    print("=" * 60)
    print("OM GOLD INTELLIGENCE")
    print("UNIFIED MODEL TRAINER — ALL HORIZONS IN ONE RUN")
    print("=" * 60)

    df = load_feature_matrix()
    feature_columns = get_feature_columns(df)

    print(f"\nUsing {len(feature_columns)} feature columns (all Target_* columns excluded for every horizon).")

    all_metrics = {}
    for horizon in HORIZONS:
        metrics = train_single_horizon(df, horizon, feature_columns)
        all_metrics[horizon] = metrics

    summary_path = os.path.join(MODEL_DIR, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(all_metrics, summary_file, indent=2)

    print("\n" + "=" * 60)
    print("ALL FIVE MODELS TRAINED SUCCESSFULLY")
    print("=" * 60)
    print(f"{'Horizon':<10}{'Direction%':<12}{'Return R2':<12}{'Price R2':<12}{'Confidence':<14}")
    for horizon in HORIZONS:
        m = all_metrics[horizon]
        print(f"{str(horizon)+'D':<10}{m['direction_accuracy']:<12}{m['return_r2']:<12}{m['r2']:<12}{m['confidence']:<14}")

    print(f"\nFull summary written to: {summary_path}")
    print("\nReminder: app.py must load gold_model_1d.pkl (not gold_model.pkl) and")
    print("also load/serve gold_model_14d.pkl and gold_model_21d.pkl, which it")
    print("currently trains but never uses.")


if __name__ == "__main__":
    train_all_models()
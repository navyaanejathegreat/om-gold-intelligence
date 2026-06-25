import pandas as pd
import os
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)

from xgboost import XGBRegressor

print("Loading features...")

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

data_path = os.path.join(
    BASE_DIR,
    "data",
    "features.csv"
)

df = pd.read_csv(data_path)
df["Date"] = pd.to_datetime(df["Date"])

df = df[
    df["Date"] >= "2022  -01-01"
]



# ==================================
# REMOVE EXTREME OUTLIERS
# ==================================

print("\nOriginal Shape:")
print(df.shape)

print(
    df[
        ["Gold", "Target_7D"]
    ].head()
)

print("\nLargest 7D Targets:")
print(
    df[
        ["Date", "Gold", "Target_7D"]
    ]
    .sort_values(
        "Target_7D",
        ascending=False
    )
    .head(10)
)

# ==================================
# FEATURES
# ==================================

X = df.drop(
    columns=[
        "Date",
        "Target_1D",
        "Target_7D",
        "Target_30D"
    ]
)

# ==================================
# TARGET
# ==================================

y = df["Target_30D"]

print("\nTarget Statistics:")
print(y.describe())

print("\nUnique Targets:")
print(y.nunique())

print("\nTarget Std:")
print(y.std())

print("\nAny NaNs?")
print(X.isna().sum().sum())

print("\nShape:")
print(X.shape)

# ==================================
# TRAIN TEST SPLIT
# ==================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)

print("\nTrain Shape:")
print(X_train.shape)

print("\nTest Shape:")
print(X_test.shape)

# ==================================
# MODEL
# ==================================

print("\nTraining 7 Day Model...")

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective="reg:squarederror"
)

model.fit(
    X_train,
    y_train
)

# ==================================
# PREDICTIONS
# ==================================

predictions = model.predict(
    X_test
)

comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions
})

print("\nPrediction Sample:")
print(
    comparison.head(20)
)

# ==================================
# METRICS
# ==================================

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)

direction_actual = np.sign(
    y_test
)

direction_pred = np.sign(
    predictions
)

direction_accuracy = (
    (
        direction_actual ==
        direction_pred
    )
    .mean()
)
mape = mean_absolute_percentage_error(
    y_test,
    predictions
)


print("\n===================")
print("7 DAY RESULTS")
print("===================")

print(
    f"\nDirection Accuracy: "
    f"{direction_accuracy * 100:.2f}%"
)
confidence_score = round(
    direction_accuracy * 100,
    2
)

if confidence_score >= 80:
    confidence = "Exceptional"

elif confidence_score >= 70:
    confidence = "Strong"

elif confidence_score >= 60:
    confidence = "Reliable"

elif confidence_score >= 55:
    confidence = "Moderate"

elif confidence_score >= 50:
    confidence = "Weak"

elif confidence_score >= 45:
    confidence = "Poor"

else:
    confidence = "Experimental"

print("\nConfidence:")
print(confidence)

print(
    f"Confidence Score: "
    f"{confidence_score}"
)
print(
    f"\nMAE: {mae}"
)

print(
    f"\nRMSE: {rmse}"
)

print(
    f"\nR2 Score: {r2}"
)
print(
    f"\nMAPE: {mape * 100:.2f}%"
)

# ==================================
# FEATURE IMPORTANCE
# ==================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance":
    model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Features:")
print(
    importance.head(10)
)

# ==================================
# NEXT 7 DAY FORECAST
# ==================================

latest_features = X.tail(1)

predicted_return = model.predict(
    latest_features
)[0]

current_price = (
    latest_features["Gold"]
    .iloc[0]
)

predicted_price = (
    current_price *
    (1 + predicted_return)
)

print("\n===================")
print("7 DAY FORECAST")
print("===================")

print(
    f"Current Price: "
    f"{current_price:.2f}"
)

print(
    f"Predicted 7D Change: "
    f"{predicted_return * 100:.2f}%"
)

print(
    f"Predicted 7D Price: "
    f"{predicted_price:.2f}"
)

# ==================================
# SAVE MODEL
# ==================================

model_dir = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(
    model_dir,
    exist_ok=True
)

model_path = os.path.join(
    model_dir,
    "gold_model_30d.pkl"
)

joblib.dump(
    model,
    model_path
)

print("\n7 Day Model Saved!")
metrics = {
    "mae": float(mae),
    "rmse": float(rmse),
    "r2": float(r2),
    "direction_accuracy": float(
        confidence_score
    ),
    "confidence": confidence
}

joblib.dump(
    metrics,
    os.path.join(
        model_dir,
        "metrics_30d.pkl"
    )
)
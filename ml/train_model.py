import pandas as pd
import os
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
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

# ==========================
# DATE FILTER
# ==========================

df["Date"] = pd.to_datetime(
    df["Date"]
)

df = df[
    df["Date"] >= "2022-01-01"
]

print("\nAfter Date Filter:")
print(df.shape)

# ==========================
# OUTLIER REMOVAL
# ==========================

print("\nOriginal Shape:")
print(df.shape)

df = df[
    (df["Target_1D"] > -0.20) &
    (df["Target_1D"] < 0.20)
]

print("\nShape After Outlier Removal:")
print(df.shape)

print("\nLargest Targets:")

print(
    df[
        ["Date", "Gold", "Target_1D"]
    ]
    .sort_values(
        "Target_1D",
        ascending=False
    )
    .head(10)
)

print("\nDataset Info:")
print(df.shape)

print(
    df[
        ["Gold", "Target_1D"]
    ].head(10)
)

print("\nTarget Statistics:")
print(
    df["Target_1D"].describe()
)

# ==========================
# FEATURES
# ==========================

X = df.drop(
    columns=[
        "Date",
        "Target_1D",
        "Target_7D",
        "Target_30D"
    ]
)

y = df["Target_1D"]

print("\nAny NaNs?")
print(
    X.isna().sum().sum()
)

print("\nShape:")
print(X.shape)

print("\n====================")
print("TARGET DEBUG")
print("====================")

print("\nUnique Targets:")
print(y.nunique())

print("\nTarget Std:")
print(y.std())

print("\nTarget Range:")
print(
    y.min(),
    y.max()
)

# ==========================
# TRAIN TEST SPLIT
# ==========================

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

print("\nTrain Target:")
print(y_train.describe())

print("\nTest Target:")
print(y_test.describe())

# ==========================
# MODEL
# ==========================

print("\nTraining Model...")

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

# ==========================
# PREDICTIONS
# ==========================

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

print("\nLast 20 Predictions:")
print(
    comparison.tail(20)
)

# ==========================
# METRICS
# ==========================

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

# Ignore tiny movements (<0.1%)

mask = np.abs(y_test) > 0.001

direction_actual = np.sign(
    y_test[mask]
)

direction_pred = np.sign(
    predictions[mask]
)

direction_accuracy = (
    direction_actual ==
    direction_pred
).mean()

print("\nDirection Accuracy:")
print(
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

print("\nMean Absolute Error:")
print(mae)

print("\nRMSE:")
print(rmse)

print("\nR2 Score:")
print(r2)

# ==========================
# FEATURE IMPORTANCE
# ==========================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance":
    model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 15 Features:")
print(
    importance.head(15)
)

# ==========================
# TOMORROW PREDICTION
# ==========================

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

lower_price = (
    current_price *
    (1 + predicted_return - mae)
)

upper_price = (
    current_price *
    (1 + predicted_return + mae)
)

print("\n======================")
print("TOMORROW PREDICTION")
print("======================")

print(
    f"Current Gold Price: {current_price:.2f}"
)

print(
    f"Predicted Change: {predicted_return * 100:.2f}%"
)

print(
    f"Predicted Tomorrow Price: {predicted_price:.2f}"
)

print(
    f"Confidence Range: "
    f"{lower_price:.2f} - {upper_price:.2f}"
)

print("\nPrediction Statistics:")
print(
    pd.Series(predictions).describe()
)

print("\nActual Statistics:")
print(
    y_test.describe()
)
print("\nPositive Days:")
print((y_test > 0).mean() * 100)

print("\nNegative Days:")
print((y_test < 0).mean() * 100)
comparison["Actual_Dir"] = np.sign(comparison["Actual"])
comparison["Pred_Dir"] = np.sign(comparison["Predicted"])

print(
    comparison[
        comparison["Actual_Dir"] !=
        comparison["Pred_Dir"]
    ]
)
# ==========================
# SAVE MODEL
# ==========================

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
    "gold_model.pkl"
)

joblib.dump(
    model,
    model_path
)

print("\nModel Saved!")
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
        "metrics_1d.pkl"
    )
)
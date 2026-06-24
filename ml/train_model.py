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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(
    BASE_DIR,
    "data",
    "features.csv"
)

df = pd.read_csv(data_path)

# ==================================
# REMOVE BAD TARGET VALUES
# ==================================

print("\nOriginal Shape:")
print(df.shape)

df = df[
    (df["Target"] > -0.20) &
    (df["Target"] < 0.20)
]

print("\nShape After Outlier Removal:")
print(df.shape)

print("\nLargest Targets:")
print(
    df[["Date", "Gold", "Target"]]
    .sort_values("Target", ascending=False)
    .head(10)
)

print("\nDataset Info:")
print(df.shape)

print(df[["Gold", "Target"]].head(10))

print("\nTarget Statistics:")
print(df["Target"].describe())

# ==================================
# FEATURES & TARGET
# ==================================

X = df.drop(
    columns=["Date", "Target"]
)

y = df["Target"]

print("\nTraining Features:")
print(X.columns)

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

print("\nTrain Target:")
print(y_train.describe())

print("\nTest Target:")
print(y_test.describe())

# ==================================
# MODEL
# ==================================

print("\nTraining Model...")

model = XGBRegressor(
    n_estimators=1500,
    learning_rate=0.01,
    max_depth=8,
    min_child_weight=3,
    subsample=0.9,
    colsample_bytree=0.9,
    gamma=0.1,
    random_state=42,
    objective="reg:squarederror"
)


model.fit(
    X_train,
    y_train
)

predictions = model.predict(X_test)

comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions
})

print("\nPrediction Sample:")
print(comparison.head(20))

print("\nLast 20 Predictions:")
print(comparison.tail(20))

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
direction_actual = np.sign(y_test)
direction_pred = np.sign(predictions)

direction_accuracy = (
    (direction_actual == direction_pred)
    .mean()
)

print("\nDirection Accuracy:")
print(f"{direction_accuracy * 100:.2f}%")

print("\nMean Absolute Error:")
print(mae)

print("\nRMSE:")
print(rmse)

print("\nR2 Score:")
print(r2)

# ==================================
# FEATURE IMPORTANCE
# ==================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance)

# ==================================
# TOMORROW PREDICTION
# ==================================

latest_features = X.tail(1)

predicted_return = model.predict(
    latest_features
)[0]
current_price = latest_features[
    "Gold"
].iloc[0]

predicted_price = (
    current_price *
    (1 + predicted_return)
)
historical_error = mae

confidence_lower = (
    predicted_return - historical_error
)

confidence_upper = (
    predicted_return + historical_error
)
lower_price = current_price * (
    1 + confidence_lower
)

upper_price = current_price * (
    1 + confidence_upper
)
print("\nPrediction Range:")
print(
    f"{lower_price:.2f} - {upper_price:.2f}"
)
current_price = latest_features[
    "Gold"
].iloc[0]

predicted_price = current_price * (
    1 + predicted_return
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
print("\nPrediction Statistics:")
print(pd.Series(predictions).describe())
print("\nActual Statistics:")
print(y_test.describe())

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
    "gold_model.pkl"
)

joblib.dump(
    model,
    model_path
)

print("\nModel Saved!")
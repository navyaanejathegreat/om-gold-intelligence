import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(
    os.path.join(
        BASE_DIR,
        "models",
        "gold_model.pkl"
    )
)

df = pd.read_csv(
    os.path.join(
        BASE_DIR,
        "data",
        "features.csv"
    )
)

latest = df.drop(
    columns=["Date", "Target"]
).tail(1)

predicted_return = model.predict(latest)[0]

current_price = latest["Gold"].iloc[0]

predicted_price = current_price * (
    1 + predicted_return
)

print("\nCurrent Price:", current_price)
print("Predicted Change:", predicted_return * 100)
print("Tomorrow Price:", predicted_price)
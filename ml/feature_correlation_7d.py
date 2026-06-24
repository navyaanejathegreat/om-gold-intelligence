import pandas as pd

df = pd.read_csv("data/features.csv")

corr = (
    df.corr(numeric_only=True)
    ["Target_7D"]
    .sort_values(
        ascending=False
    )
)

print(corr.head(30))
print(corr.tail(30))
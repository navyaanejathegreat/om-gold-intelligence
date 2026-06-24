import pandas as pd
import ta

print("Loading market data...")

df = pd.read_csv("data/market_data.csv")
df["Date"] = pd.to_datetime(df["Date"])

# Fill missing values
df = df.ffill()
df = df.bfill()

print("\nAfter filling:")
print(df.isna().sum())

print(df.shape)

print("\nMissing values in raw data:")
print(df.isna().sum())

print("\nFirst 20 rows:")
print(df.head(20))
print("\nOriginal Shape:")
print(df.shape)

print("\nMissing Values BEFORE indicators:")
print(df.isna().sum())

# Convert Date column
df["Date"] = pd.to_datetime(df["Date"])

print("\nCreating indicators...")

# RSI
df["RSI"] = ta.momentum.RSIIndicator(
    close=df["Gold"],
    window=14
).rsi()

# EMA 20
df["EMA_20"] = ta.trend.EMAIndicator(
    close=df["Gold"],
    window=20
).ema_indicator()

# EMA 50
df["EMA_50"] = ta.trend.EMAIndicator(
    close=df["Gold"],
    window=50
).ema_indicator()

# MACD
macd = ta.trend.MACD(
    close=df["Gold"]
)

df["MACD"] = macd.macd()

# Lag Features
df["Gold_1"] = df["Gold"].shift(1)
df["Gold_2"] = df["Gold"].shift(2)
df["Gold_3"] = df["Gold"].shift(3)
df["Gold_5"] = df["Gold"].shift(5)
df["Gold_7"] = df["Gold"].shift(7)
df["Gold_14"] = df["Gold"].shift(14)
df["Gold_30"] = df["Gold"].shift(30)

# Moving Averages
df["Gold_MA_20"] = df["Gold"].rolling(20).mean()
df["Gold_MA_50"] = df["Gold"].rolling(50).mean()
df["Gold_MA_100"] = df["Gold"].rolling(100).mean()
df["Gold_MA_200"] = df["Gold"].rolling(200).mean()

# Returns
df["Return_1"] = df["Gold"].pct_change(1)
df["Return_3"] = df["Gold"].pct_change(3)
df["Return_7"] = df["Gold"].pct_change(7)
df["Return_14"] = df["Gold"].pct_change(14)

# Volatility
df["Volatility_7"] = (
    df["Gold"].pct_change().rolling(7).std()
)

df["Volatility_14"] = (
    df["Gold"].pct_change().rolling(14).std()
)

# Target
print("Creating target...")

df["Target"] = (
    df["Gold"].shift(-1) - df["Gold"]
) / df["Gold"]
df["Gold_vs_MA20"] = (
    df["Gold"] - df["Gold_MA_20"]
) / df["Gold_MA_20"]

df["Gold_vs_MA50"] = (
    df["Gold"] - df["Gold_MA_50"]
) / df["Gold_MA_50"]

df["EMA_Diff"] = (
    df["EMA_20"] - df["EMA_50"]
) / df["EMA_50"]

df["Momentum_5"] = (
    df["Gold"] - df["Gold_5"]
) / df["Gold_5"]

df["Momentum_14"] = (
    df["Gold"] - df["Gold_14"]
) / df["Gold_14"]

print("\nShape BEFORE dropna:")
print(df.shape)

print("\nMissing Values AFTER indicators:")
print(df.isna().sum())

print("\nRemoving empty rows...")

df.dropna(inplace=True)

print("\nShape AFTER dropna:")
print(df.shape)

if len(df) == 0:
    print("\nERROR: All rows were removed.")
    print("One or more columns contain only NaN values.")
else:
    df.to_csv(
        "data/features.csv",
        index=False
    )

    print("\nFeatures Created Successfully!")
    print(df.head())
import pandas as pd
import ta

print("Loading market data...")

df = pd.read_csv("data/market_data.csv")

df["Date"] = pd.to_datetime(
    df["Date"]
)

# Fill missing values
df = df.ffill()
df = df.bfill()

print("\nAfter filling:")
print(df.isna().sum())

print("\nDataset Shape:")
print(df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nCreating indicators...")

# ==================================
# Technical Indicators
# ==================================

df["RSI"] = ta.momentum.RSIIndicator(
    close=df["Gold"],
    window=14
).rsi()
df["RSI_7"] = ta.momentum.RSIIndicator(
    close=df["Gold"],
    window=7
).rsi()

df["RSI_21"] = ta.momentum.RSIIndicator(
    close=df["Gold"],
    window=21
).rsi()
df["EMA_20"] = ta.trend.EMAIndicator(
    close=df["Gold"],
    window=20
).ema_indicator()

df["EMA_50"] = ta.trend.EMAIndicator(
    close=df["Gold"],
    window=50
).ema_indicator()

macd = ta.trend.MACD(
    close=df["Gold"]
)

df["MACD"] = macd.macd()

# ==================================
# Lag Features
# ==================================

df["Gold_1"] = df["Gold"].shift(1)
df["Gold_2"] = df["Gold"].shift(2)
df["Gold_3"] = df["Gold"].shift(3)
df["Gold_5"] = df["Gold"].shift(5)
df["Gold_7"] = df["Gold"].shift(7)
df["Gold_14"] = df["Gold"].shift(14)
df["Gold_30"] = df["Gold"].shift(30)

# ==================================
# Moving Averages
# ==================================

df["Gold_MA_20"] = (
    df["Gold"]
    .rolling(20)
    .mean()
)

df["Gold_MA_50"] = (
    df["Gold"]
    .rolling(50)
    .mean()
)

df["Gold_MA_100"] = (
    df["Gold"]
    .rolling(100)
    .mean()
)

df["Gold_MA_200"] = (
    df["Gold"]
    .rolling(200)
    .mean()
)

# ==================================
# Returns
# ==================================

df["Return_1"] = df["Gold"].pct_change(1)
df["Return_3"] = df["Gold"].pct_change(3)
df["Return_7"] = df["Gold"].pct_change(7)
df["Return_14"] = df["Gold"].pct_change(14)

# ==================================
# Volatility
# ==================================

df["Volatility_7"] = (
    df["Gold"]
    .pct_change()
    .rolling(7)
    .std()
)

df["Volatility_14"] = (
    df["Gold"]
    .pct_change()
    .rolling(14)
    .std()
)

# ==================================
# Trend Features
# ==================================

df["Gold_vs_MA20"] = (
    df["Gold"] -
    df["Gold_MA_20"]
) / df["Gold_MA_20"]

df["Gold_vs_MA50"] = (
    df["Gold"] -
    df["Gold_MA_50"]
) / df["Gold_MA_50"]
df["Above_MA50"] = (
    df["Gold"] >
    df["Gold_MA_50"]
).astype(int)

df["Above_MA200"] = (
    df["Gold"] >
    df["Gold_MA_200"]
).astype(int)

df["EMA_Diff"] = (
    df["EMA_20"] -
    df["EMA_50"]
) / df["EMA_50"]
# Additional Moving Averages

df["Gold_MA_10"] = (
    df["Gold"].rolling(10).mean()
)

df["Gold_MA_30"] = (
    df["Gold"].rolling(30).mean()
)

df["Gold_MA_90"] = (
    df["Gold"].rolling(90).mean()
)

# ==================================
# Momentum Features
# ==================================

df["Momentum_3"] = (
    df["Gold"] -
    df["Gold"].shift(3)
)

df["Momentum_5"] = (
    df["Gold"] -
    df["Gold"].shift(5)
)

df["Momentum_7"] = (
    df["Gold"] -
    df["Gold"].shift(7)
)

df["Momentum_14"] = (
    df["Gold"] -
    df["Gold"].shift(14)
)

df["Momentum_30"] = (
    df["Gold"] -
    df["Gold"].shift(30)
)

# ==================================
# Market Change Features
# ==================================

df["Silver_Change"] = (
    df["Silver"]
    .pct_change()
)

df["Oil_Change"] = (
    df["Oil"]
    .pct_change()
)

df["VIX_Change"] = (
    df["VIX"]
    .pct_change()
)


if "DXY" in df.columns:
    df["DXY_Change"] = (
        df["DXY"]
        .pct_change()
    )

if "SP500" in df.columns:
    df["SP500_Change"] = (
        df["SP500"]
        .pct_change()
    )

if "TNX" in df.columns:
    df["TNX_Change"] = (
        df["TNX"]
        .pct_change()
    )

# ==================================
# Target
# ==================================

print("Creating target...")

print("Creating targets...")

# 1 Day Return
df["Target_1D"] = (
    df["Gold"].shift(-1) -
    df["Gold"]
) / df["Gold"]

# 7 Day Return
df["Target_7D"] = (
    df["Gold"].shift(-7) -
    df["Gold"]
) / df["Gold"]

# 30 Day Return
df["Target_30D"] = (
    df["Gold"].shift(-30) -
    df["Gold"]
) / df["Gold"]

print("\nShape BEFORE dropna:")
print(df.shape)

print("\nMissing Values:")
print(df.isna().sum())

print("\nRemoving empty rows...")

df.dropna(inplace=True)

print("\nShape AFTER dropna:")
print(df.shape)

df.to_csv(
    "data/features.csv",
    index=False
)

print("\nFeatures Created Successfully!")

print("\nColumns:")
print(df.columns)

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())
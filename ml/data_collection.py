import yfinance as yf
import pandas as pd

print("Downloading market data...")

tickers = {
    "Gold": "GOLDBEES.NS",
    "Silver": "SI=F",
    "Oil": "CL=F",
    "Nifty": "^NSEI",
    "VIX": "^INDIAVIX"
}

dataframes = []

for name, ticker in tickers.items():

    print(f"Downloading {name}...")

    df = yf.download(
        ticker,
        start="2010-01-01",
        progress=False
    )

    df = df[["Close"]]
    if df.empty:
        print(f"Failed: {ticker}")
        continue


    df.columns = [name]

    dataframes.append(df)

merged_df = pd.concat(dataframes, axis=1)

merged_df.to_csv("data/market_data.csv")

print("\nData Saved Successfully!")
print("\nFirst 5 rows:")

print(merged_df.head())

print("\nDataset Shape:")
print(merged_df.shape)

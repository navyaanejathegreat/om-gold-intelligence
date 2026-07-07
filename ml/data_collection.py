"""
Om Gold Intelligence — Market Data Collection (ml/data_collection.py)

Downloads Gold, Silver, Oil, SP500, VIX, TNX, DXY, and USDINR and saves them
to data/market_data.csv.

    python data_collection.py            # incremental (only fetches what's missing)
    python data_collection.py --full     # forces a full 2010-present re-download

WHY THIS FILE WAS REWRITTEN
----------------------------------------------------------------------------
The previous version re-downloaded the ENTIRE history (2010-01-01 to today)
for all eight tickers on every single run, then overwrote market_data.csv
from scratch. That's the main reason a full pipeline run (triggered by
update_data.py, whether manually or from app.py's auto-refresh) took
minutes instead of seconds — almost all of that time was re-fetching years
of data you already had.

This version:
    1. Reads the existing market_data.csv (if present) and only requests
       the date range that's actually missing — typically just the last
       few days.
    2. Re-requests the last couple of already-stored days too (not just
       "day after last stored date"), so if a value was previously
       provisional/incomplete for a date whose market hadn't fully closed
       yet, it gets a chance to be corrected rather than staying wrong
       forever.
    3. Merges the newly fetched rows into the existing dataset by date,
       with the newly fetched value winning on any overlapping date —
       and never produces duplicate date rows.
    4. Falls back to the original full 2010-present backfill automatically
       if no market_data.csv exists yet, or if --full is passed explicitly.
    5. Never crashes the whole run because one ticker failed — it reports
       exactly which ticker(s) failed and continues with the others,
       since a temporary VIX or DXY outage shouldn't block getting a
       current Gold price.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MARKET_DATA_PATH = DATA_DIR / "market_data.csv"

DATE_COLUMN = "Date"

# Ticker symbol -> output column name (unchanged from the original file, so
# this stays compatible with all existing stored history).
TICKERS = {
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Oil": "CL=F",
    "SP500": "^GSPC",
    "VIX": "^VIX",
    "TNX": "^TNX",
    "DXY": "DX-Y.NYB",
    "USDINR": "INR=X",
}

FULL_BACKFILL_START_DATE = "2010-01-01"

# When doing an incremental update, re-fetch this many of the most recently
# stored days too, in case they were previously provisional/incomplete.
REFETCH_BUFFER_DAYS = 3

DATA_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Existing dataset helpers
# --------------------------------------------------------------------------

def load_existing_dataset() -> pd.DataFrame:
    """Loads market_data.csv from disk, or an empty frame with expected columns."""
    expected_columns = [DATE_COLUMN] + list(TICKERS.keys())

    if not MARKET_DATA_PATH.exists():
        return pd.DataFrame(columns=expected_columns)

    df = pd.read_csv(MARKET_DATA_PATH)
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])

    for column in expected_columns:
        if column not in df.columns:
            df[column] = pd.NA

    df = df[expected_columns].sort_values(DATE_COLUMN).reset_index(drop=True)
    return df


def determine_fetch_start_date(existing_df: pd.DataFrame, force_full: bool):
    """
    Returns the date to start fetching from. Full backfill if the dataset
    is empty or --full was passed; otherwise a few days before the last
    stored date, to allow correcting recently-provisional values.
    """
    if force_full or existing_df.empty:
        return FULL_BACKFILL_START_DATE

    last_stored_date = existing_df[DATE_COLUMN].max().date()
    refetch_from = last_stored_date - timedelta(days=REFETCH_BUFFER_DAYS)

    return refetch_from.isoformat()


# --------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------

def fetch_ticker_series(name: str, ticker: str, start_date: str) -> pd.Series:
    """
    Downloads the daily closing price series for a single ticker.
    Returns an empty Series (never raises) if the fetch fails, so one bad
    ticker doesn't block the rest of the run.
    """
    try:
        print(f"Fetching {name} ({ticker}) from {start_date}...")

        raw = yf.download(
            ticker,
            start=start_date,
            progress=False,
            auto_adjust=True,
        )

        if raw.empty or "Close" not in raw:
            print(f"  No data returned for {name} ({ticker}).")
            return pd.Series(dtype="float64", name=name)

        close_series = raw["Close"]

        if isinstance(close_series, pd.DataFrame):
            close_series = close_series.iloc[:, 0]

        close_series = close_series.rename(name)
        close_series.index = pd.to_datetime(close_series.index).tz_localize(None)
        close_series.index.name = DATE_COLUMN

        print(f"  Retrieved {len(close_series)} row(s) for {name}.")
        return close_series

    except Exception as exc:  # noqa: BLE001 - one ticker failing must not stop the others
        print(f"  Failed to fetch {name} ({ticker}): {exc}")
        return pd.Series(dtype="float64", name=name)


def fetch_all_tickers(start_date: str) -> pd.DataFrame:
    """Fetches every configured ticker from start_date to today and merges them."""
    merged_df = pd.DataFrame()

    for name, ticker in TICKERS.items():
        series = fetch_ticker_series(name, ticker, start_date)

        if series.empty:
            continue

        if merged_df.empty:
            merged_df = series.to_frame()
        else:
            merged_df = merged_df.join(series, how="outer")

    if merged_df.empty:
        return pd.DataFrame(columns=[DATE_COLUMN] + list(TICKERS.keys()))

    merged_df = merged_df.reset_index()
    return merged_df


# --------------------------------------------------------------------------
# Merge (duplicate-safe) + save
# --------------------------------------------------------------------------

def merge_and_deduplicate(existing_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combines existing and newly fetched data with no duplicate dates. When
    a date exists in both, the newly fetched row wins — this is what lets
    a previously provisional/incomplete day get corrected on a later run.
    """
    if new_df.empty:
        return existing_df

    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    combined_df = combined_df.sort_values(DATE_COLUMN)
    combined_df = combined_df.drop_duplicates(subset=[DATE_COLUMN], keep="last")
    combined_df = combined_df.reset_index(drop=True)

    return combined_df


def save_dataset(df: pd.DataFrame) -> None:
    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)
    df.to_csv(MARKET_DATA_PATH, index=False)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def run(force_full: bool = False) -> pd.DataFrame:
    print("=" * 60)
    print("OM GOLD INTELLIGENCE — MARKET DATA COLLECTION")
    print("=" * 60)

    existing_df = load_existing_dataset()
    rows_before = len(existing_df)

    start_date = determine_fetch_start_date(existing_df, force_full)

    if force_full or existing_df.empty:
        print(f"\nRunning a FULL backfill from {start_date} to today.")
    else:
        print(
            f"\nRunning an INCREMENTAL update from {start_date} to today "
            f"(re-checking the last {REFETCH_BUFFER_DAYS} stored day(s) too)."
        )

    new_df = fetch_all_tickers(start_date)

    if new_df.empty:
        print("\nNo data could be retrieved. Keeping existing dataset unchanged.")
        combined_df = existing_df
    else:
        combined_df = merge_and_deduplicate(existing_df, new_df)

    rows_after = len(combined_df)

    if combined_df.empty:
        raise RuntimeError(
            "market_data.csv would be empty after this run. Aborting without "
            "overwriting any existing file."
        )

    save_dataset(combined_df)

    print("\n" + "=" * 60)
    print("Data Saved Successfully!")
    print("=" * 60)
    print(f"Rows before: {rows_before}")
    print(f"Rows after : {rows_after}")
    print(f"Net new rows: {rows_after - rows_before}")
    print(f"Latest date : {combined_df[DATE_COLUMN].max().date()}")
    print(f"\nSaved to: {MARKET_DATA_PATH}")

    print("\nFirst 5 rows:")
    print(combined_df.head())
    print("\nLast 5 rows:")
    print(combined_df.tail())

    return combined_df


if __name__ == "__main__":
    full_flag = "--full" in sys.argv
    run(force_full=full_flag)
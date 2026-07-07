"""
Om Gold Intelligence — Feature Engineering (ml/feature_engineering.py)

Builds the feature matrix used to train and serve the gold price forecasting
models, and constructs per-horizon supervised learning targets for:

    1, 7, 14, 21, and 30 day forecasts.

WHY THIS FILE WAS REWRITTEN
----------------------------------------------------------------------------
The previous version was a top-level script that silently forward/back-filled
EVERY missing value across the entire history with no limit and no record of
what was imputed. In combination with the rest of the pipeline never being
re-run automatically, this meant a stale "latest" row could sit in
features.csv indefinitely while looking indistinguishable from a fresh one —
which is the direct cause of "the website never shows today's gold price."

This version:
    1. Is importable (wrapped in functions) so an automated pipeline
       (update_data.py / a scheduler / app.py) can call it directly instead
       of requiring a human to run it as a script.
    2. Limits forward-fill to short gaps only (default: 2 trading days) and
       explicitly flags any row that contains imputed values, instead of
       silently blending stale data into "today."
    3. Validates and reports data freshness — it will tell you plainly if
       the newest row in market_data.csv is older than expected, rather than
       processing it silently as if everything were current.
    4. Writes a small last_updated.json alongside features.csv so the API
       layer (and the frontend) can display an honest "data as of" date
       instead of implying every load reflects today.
    5. Preserves the exact set of engineered features and target definitions
       from the original file, so existing trained models remain compatible
       with this feature matrix's column names.

This file does NOT fetch new market data itself — that remains the job of
update_data.py / data_collection.py. It only transforms whatever is present
in market_data.csv into a model-ready feature matrix.
"""

import json
import logging
import os
from datetime import datetime

import pandas as pd
import ta

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

MARKET_DATA_PATH = os.path.join(DATA_DIR, "market_data.csv")
FEATURES_PATH = os.path.join(DATA_DIR, "features.csv")
LAST_UPDATED_PATH = os.path.join(DATA_DIR, "last_updated.json")

DATE_COLUMN = "Date"
GOLD_COLUMN = "Gold"

RAW_MARKET_COLUMNS = ["Gold", "Silver", "Oil", "SP500", "VIX", "TNX", "DXY", "USDINR"]

# Forward-fill is only allowed to bridge gaps up to this many rows. Anything
# longer is left as NaN (and will be dropped by dropna()) rather than
# quietly carrying a stale price forward indefinitely.
MAX_IMPUTE_GAP = 2

# If the newest row in market_data.csv is older than this many calendar
# days, freshness is flagged as stale in the returned report (weekends /
# market holidays mean a couple of days' lag is normal; this threshold
# gives room for that without hiding a genuinely broken pipeline).
STALE_DATA_THRESHOLD_DAYS = 4

HORIZONS = [1, 7, 14, 21, 30]

os.makedirs(LOGS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "feature_engineering.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("feature_engineering")


# --------------------------------------------------------------------------
# Loading + freshness-aware cleaning
# --------------------------------------------------------------------------

def load_market_data() -> pd.DataFrame:
    """Loads market_data.csv from disk and parses the date column."""
    if not os.path.exists(MARKET_DATA_PATH):
        raise FileNotFoundError(
            f"market_data.csv not found at {MARKET_DATA_PATH}. "
            "Run data collection before building features."
        )

    df = pd.read_csv(MARKET_DATA_PATH)
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

    return df


def check_freshness(df: pd.DataFrame) -> dict:
    """
    Reports how current the raw market data actually is. This does not
    raise — it returns a report so callers (this script, app.py, or a
    scheduler) can decide what to do — but it makes staleness visible
    instead of silent.
    """
    latest_date = df[DATE_COLUMN].max().date()
    today = datetime.now().date()
    age_days = (today - latest_date).days
    is_stale = age_days > STALE_DATA_THRESHOLD_DAYS

    report = {
        "latest_data_date": latest_date.isoformat(),
        "checked_at": datetime.now().isoformat(),
        "age_days": age_days,
        "is_stale": is_stale,
    }

    if is_stale:
        logger.warning(
            "Market data is %d day(s) old (latest row: %s). "
            "Run update_data.py / data_collection.py before relying on this "
            "for predictions — the API will otherwise keep serving an old price.",
            age_days,
            latest_date,
        )
    else:
        logger.info(
            "Market data freshness OK. Latest row: %s (%d day(s) old).",
            latest_date,
            age_days,
        )

    return report


def clean_and_flag_imputed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills only short gaps (up to MAX_IMPUTE_GAP consecutive rows) per column,
    and adds an `is_imputed` flag marking any row where at least one raw
    market column was filled rather than genuinely observed.

    Unlike a blanket ffill/bfill, this will NOT silently carry a stale price
    forward across a long gap (e.g. a broken data feed) — those rows are
    left as NaN and will be excluded by the final dropna() step, rather than
    masquerading as fresh data.
    """
    df = df.copy()
    raw_columns = [column for column in RAW_MARKET_COLUMNS if column in df.columns]

    imputed_mask = pd.Series(False, index=df.index)

    for column in raw_columns:
        was_missing = df[column].isna()

        # Limited forward-fill: only bridges gaps of MAX_IMPUTE_GAP rows or fewer.
        filled = df[column].ffill(limit=MAX_IMPUTE_GAP)

        newly_filled = was_missing & filled.notna()
        imputed_mask = imputed_mask | newly_filled

        df[column] = filled

    df["is_imputed"] = imputed_mask.astype(int)

    remaining_na = df[raw_columns].isna().sum()
    if remaining_na.sum() > 0:
        logger.info(
            "Rows with unresolvable gaps (beyond %d-day limit) will be dropped later:\n%s",
            MAX_IMPUTE_GAP,
            remaining_na[remaining_na > 0].to_string(),
        )

    return df


# --------------------------------------------------------------------------
# Feature engineering (unchanged computations from the original file —
# preserved so existing trained models remain compatible with these columns)
# --------------------------------------------------------------------------

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["RSI"] = ta.momentum.RSIIndicator(close=df[GOLD_COLUMN], window=14).rsi()
    df["RSI_7"] = ta.momentum.RSIIndicator(close=df[GOLD_COLUMN], window=7).rsi()
    df["RSI_21"] = ta.momentum.RSIIndicator(close=df[GOLD_COLUMN], window=21).rsi()

    df["EMA_20"] = ta.trend.EMAIndicator(close=df[GOLD_COLUMN], window=20).ema_indicator()
    df["EMA_50"] = ta.trend.EMAIndicator(close=df[GOLD_COLUMN], window=50).ema_indicator()

    macd = ta.trend.MACD(close=df[GOLD_COLUMN])
    df["MACD"] = macd.macd()

    bb = ta.volatility.BollingerBands(close=df[GOLD_COLUMN], window=20, window_dev=2)
    df["BB_High"] = bb.bollinger_hband()
    df["BB_Low"] = bb.bollinger_lband()
    df["BB_Width"] = bb.bollinger_wband()

    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    for lag in (1, 2, 3, 5, 7, 14, 30):
        df[f"Gold_{lag}"] = df[GOLD_COLUMN].shift(lag)
    return df


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    for window in (10, 20, 30, 50, 90, 100, 200, 365):
        df[f"Gold_MA_{window}"] = df[GOLD_COLUMN].rolling(window).mean()
    return df


def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    for window in (1, 3, 7, 14, 30, 60, 90):
        df[f"Return_{window}"] = df[GOLD_COLUMN].pct_change(window)

    df["Return_Acceleration"] = df["Return_7"] - df["Return_30"]

    return df


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    df["Volatility_7"] = df[GOLD_COLUMN].pct_change().rolling(7).std()
    df["Volatility_14"] = df[GOLD_COLUMN].pct_change().rolling(14).std()
    df["Volatility_Regime"] = (
        df["Volatility_14"] > df["Volatility_14"].rolling(60).mean()
    ).astype(int)

    return df


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    df["Trend_20"] = df[GOLD_COLUMN] / df["Gold_MA_20"]
    df["Trend_50"] = df[GOLD_COLUMN] / df["Gold_MA_50"]
    df["Trend_200"] = df[GOLD_COLUMN] / df["Gold_MA_200"]
    df["Trend_Strength"] = df["Trend_20"] - df["Trend_50"]

    df["Gold_vs_MA20"] = (df[GOLD_COLUMN] - df["Gold_MA_20"]) / df["Gold_MA_20"]
    df["Gold_vs_MA50"] = (df[GOLD_COLUMN] - df["Gold_MA_50"]) / df["Gold_MA_50"]
    df["Gold_vs_MA90"] = (df[GOLD_COLUMN] - df["Gold_MA_90"]) / df["Gold_MA_90"]
    df["Gold_vs_MA365"] = (df[GOLD_COLUMN] - df["Gold_MA_365"]) / df["Gold_MA_365"]

    df["Above_MA50"] = (df[GOLD_COLUMN] > df["Gold_MA_50"]).astype(int)
    df["Above_MA200"] = (df[GOLD_COLUMN] > df["Gold_MA_200"]).astype(int)

    df["EMA_Diff"] = (df["EMA_20"] - df["EMA_50"]) / df["EMA_50"]
    df["EMA_Trend"] = (df["EMA_20"] > df["EMA_50"]).astype(int)

    return df


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    for window in (3, 5, 7, 14, 30):
        df[f"Momentum_{window}"] = df[GOLD_COLUMN] - df[GOLD_COLUMN].shift(window)
    return df


def add_cross_asset_features(df: pd.DataFrame) -> pd.DataFrame:
    df["Silver_Change"] = df["Silver"].pct_change()
    df["Oil_Change"] = df["Oil"].pct_change()
    df["VIX_Change"] = df["VIX"].pct_change()

    if "DXY" in df.columns:
        df["DXY_Change"] = df["DXY"].pct_change()

    if "SP500" in df.columns:
        df["SP500_Change"] = df["SP500"].pct_change()

    if "TNX" in df.columns:
        df["TNX_Change"] = df["TNX"].pct_change()

    df["USDINR_Change"] = df["USDINR"].pct_change()
    df["USDINR_MA20"] = df["USDINR"].rolling(20).mean()
    df["USDINR_vs_MA20"] = (df["USDINR"] - df["USDINR_MA20"]) / df["USDINR_MA20"]

    df["Gold_Silver_Ratio"] = df[GOLD_COLUMN] / df["Silver"]
    df["Gold_DXY_Ratio"] = df[GOLD_COLUMN] / df["DXY"]
    df["Gold_Oil_Ratio"] = df[GOLD_COLUMN] / df["Oil"]
    df["Gold_VIX_Ratio"] = df[GOLD_COLUMN] / df["VIX"]

    return df


def add_seasonality_features(df: pd.DataFrame) -> pd.DataFrame:
    df["Month"] = df[DATE_COLUMN].dt.month
    df["Quarter"] = df[DATE_COLUMN].dt.quarter
    df["DayOfWeek"] = df[DATE_COLUMN].dt.dayofweek

    return df


def add_targets(df: pd.DataFrame, horizons: list = None) -> pd.DataFrame:
    """
    Adds one forward-return target column per horizon: Target_{N}D = the
    percentage return from today's close to the close N trading rows later.
    """
    horizons = horizons or HORIZONS

    for horizon in horizons:
        df[f"Target_{horizon}D"] = (
            df[GOLD_COLUMN].shift(-horizon) - df[GOLD_COLUMN]
        ) / df[GOLD_COLUMN]

    return df


def build_feature_matrix(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds the complete, model-ready feature matrix from a raw
    market-data-shaped dataframe. This is the single source of truth for
    feature computation — the same function should be used for both
    training and generating live predictions, so the two never drift apart.
    """
    df = clean_and_flag_imputed(raw_df)

    df = add_technical_indicators(df)
    df = add_lag_features(df)
    df = add_moving_averages(df)
    df = add_return_features(df)
    df = add_volatility_features(df)
    df = add_trend_features(df)
    df = add_momentum_features(df)
    df = add_cross_asset_features(df)
    df = add_seasonality_features(df)
    df = add_targets(df)

    return df


# --------------------------------------------------------------------------
# Metadata (lets the API / frontend show an honest "data as of" date)
# --------------------------------------------------------------------------
def get_latest_inference_row() -> pd.DataFrame:
    """
    Returns the single most recent row of engineered features for live
    prediction. Unlike the saved features.csv (whose last ~30 rows are
    deliberately dropped by run()'s dropna(), since Target_30D requires 30
    future rows to exist), this computes indicators through the LATEST
    available market data date directly. Target_* columns will be NaN here
    (expected — the future hasn't happened yet) and must be excluded by
    the caller, exactly as app.py already does when selecting model input
    columns.
    """
    raw_df = load_market_data()
    feature_df = build_feature_matrix(raw_df)
    return feature_df.tail(1).copy()
def write_last_updated_metadata(
    freshness_report: dict, row_count_before: int, row_count_after: int
) -> None:
    metadata = {
        **freshness_report,
        "rows_before_dropna": row_count_before,
        "rows_after_dropna": row_count_after,
        "feature_build_completed_at": datetime.now().isoformat(),
    }

    with open(LAST_UPDATED_PATH, "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    logger.info("Wrote freshness metadata to %s.", LAST_UPDATED_PATH)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def run(save: bool = True) -> pd.DataFrame:
    """
    Runs the full feature engineering pipeline: load -> freshness check ->
    clean/flag -> build features -> build targets -> drop incomplete rows ->
    (optionally) save to features.csv + last_updated.json.

    Returns the resulting feature dataframe either way, so this can be
    called directly by app.py or a training script without requiring a
    round-trip through disk.
    """
    logger.info("Loading market data from %s...", MARKET_DATA_PATH)
    raw_df = load_market_data()

    freshness_report = check_freshness(raw_df)

    logger.info("Building feature matrix...")
    feature_df = build_feature_matrix(raw_df)

    rows_before = len(feature_df)
    feature_df = feature_df.dropna().reset_index(drop=True)
    rows_after = len(feature_df)

    logger.info("Rows before dropna: %d, after dropna: %d.", rows_before, rows_after)

    if rows_after == 0:
        raise ValueError(
            "Feature matrix is empty after dropping incomplete rows. "
            "Check that market_data.csv has enough history for the longest "
            "rolling window (365 days) plus the longest forecast horizon (30 days)."
        )

    if save:
        feature_df.to_csv(FEATURES_PATH, index=False)
        logger.info("Saved feature matrix to %s.", FEATURES_PATH)
        write_last_updated_metadata(freshness_report, rows_before, rows_after)

    return feature_df


if __name__ == "__main__":
    result_df = run(save=True)

    print("\nFeature engineering complete.")
    print(f"Final shape: {result_df.shape}")
    print(f"Latest row date: {result_df[DATE_COLUMN].max().date()}")
    print(f"Metadata written to: {LAST_UPDATED_PATH}")
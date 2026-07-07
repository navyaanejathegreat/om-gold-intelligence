"""
Om Gold Intelligence — CFTC Commitments of Traders Data (ml/cot_data.py)

Fetches weekly Commitments of Traders (COT) positioning data for COMEX Gold
futures directly from the CFTC's official public Socrata API (no API key
required) and derives features that are genuinely predictive signals in
published gold-forecasting research — as opposed to the purely
price-derived technical indicators already in feature_engineering.py, which
by construction cannot contain information not already reflected in price.

WHY THIS DATA MATTERS
----------------------------------------------------------------------------
The COT report shows how large speculators ("non-commercial" traders — hedge
funds, CTAs, money managers) are positioned in gold futures each week. This
is a genuinely different information source from price history: it reflects
who is currently long/short and by how much, which price-derived technical
indicators cannot capture. Persistent net-long positioning by speculators,
or sharp week-over-week shifts in that positioning, are among the most
commonly cited leading indicators in gold market analysis.

DATA SOURCE
----------------------------------------------------------------------------
    Endpoint: https://publicreporting.cftc.gov/resource/6dca-aqww.json
              (CFTC Legacy Commitments of Traders, Futures Only — official
              Socrata Open Data API, confirmed live, no API key needed)
    Contract: COMEX Gold, cftc_contract_market_code = "088691"
              (100 troy oz contract — matches the GC=F futures contract
              already used elsewhere in this pipeline)
    Frequency: weekly, as of each Tuesday, published the following Friday.

IMPORTANT LIMITATION, STATED UP FRONT: COT data is WEEKLY, not daily. This
module forward-fills each week's values to daily frequency (a trading day
carries the most recently PUBLISHED report's values, never a future one)
so it can be merged into the daily feature matrix without leaking future
information. This means COT-derived features change only once a week even
though the model runs daily — that is expected and correct, not a bug.
"""

import logging
import os
from datetime import datetime, timedelta

import pandas as pd
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

COT_CACHE_PATH = os.path.join(DATA_DIR, "cot_gold.csv")

COT_API_URL = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
GOLD_CONTRACT_CODE = "088691"  # COMEX Gold, Legacy Futures Only, 100 troy oz

DATE_COLUMN = "report_date_as_yyyy_mm_dd"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "cot_data.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("cot_data")


# --------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------

def fetch_gold_cot_data(start_date: str = "2010-01-01") -> pd.DataFrame:
    """
    Fetches all available weekly COT reports for COMEX Gold from start_date
    to today, using proper query-parameter encoding (requests handles this
    correctly, unlike a hand-built URL string).

    Returns an empty DataFrame (never raises) if the request fails, so a
    temporary CFTC API outage doesn't break the whole feature pipeline —
    callers should fall back to the cached CSV in that case.
    """
    params = {
        "$where": f"cftc_contract_market_code='{GOLD_CONTRACT_CODE}' "
                  f"AND report_date_as_yyyy_mm_dd >= '{start_date}T00:00:00.000'",
        "$order": f"{DATE_COLUMN} ASC",
        "$limit": 5000,
    }

    try:
        response = requests.get(COT_API_URL, params=params, timeout=30)
        response.raise_for_status()
        records = response.json()
    except Exception as exc:  # noqa: BLE001 - network/API failures are expected occasionally
        logger.error("Failed to fetch CFTC COT data: %s", exc)
        return pd.DataFrame()

    if not records:
        logger.warning("CFTC COT API returned no records for the requested range.")
        return pd.DataFrame()

    df = pd.DataFrame.from_records(records)
    logger.info("Fetched %d COT report row(s) for Gold (contract %s).", len(df), GOLD_CONTRACT_CODE)

    return df


# --------------------------------------------------------------------------
# Feature derivation
# --------------------------------------------------------------------------

def build_cot_features(raw_cot_df: pd.DataFrame) -> pd.DataFrame:
    """
    Derives positioning features from the raw COT report fields. All
    numeric fields arrive from the API as strings and must be cast
    explicitly.
    """
    if raw_cot_df.empty:
        return pd.DataFrame(columns=["date"])

    df = raw_cot_df.copy()

    numeric_columns = [
        "open_interest_all",
        "noncomm_positions_long_all",
        "noncomm_positions_short_all",
        "comm_positions_long_all",
        "comm_positions_short_all",
        "traders_noncomm_long_all",
        "traders_noncomm_short_all",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df["date"] = pd.to_datetime(df[DATE_COLUMN]).dt.tz_localize(None)
    df = df.sort_values("date").reset_index(drop=True)

    # Net non-commercial (speculative) position — the standard "smart
    # money" positioning signal used in gold COT analysis.
    df["cot_noncomm_net"] = df["noncomm_positions_long_all"] - df["noncomm_positions_short_all"]

    # Normalized by open interest, so the signal is comparable across
    # different market-size regimes over a multi-year history.
    df["cot_noncomm_net_pct_oi"] = df["cot_noncomm_net"] / df["open_interest_all"]

    # Net commercial (hedger) position — commercials are typically the
    # other side of speculative trades; their positioning is watched as a
    # contrarian signal (heavy commercial shorting often coincides with
    # speculative excess on the long side).
    df["cot_comm_net"] = df["comm_positions_long_all"] - df["comm_positions_short_all"]
    df["cot_comm_net_pct_oi"] = df["cot_comm_net"] / df["open_interest_all"]

    # Week-over-week change in speculative positioning — captures shifts
    # in sentiment, not just the current level.
    df["cot_noncomm_net_change_1w"] = df["cot_noncomm_net"].diff(1)

    # 4-week rolling average — smooths single-week noise while still
    # reflecting the current positioning regime.
    df["cot_noncomm_net_ma_4w"] = df["cot_noncomm_net"].rolling(window=4, min_periods=1).mean()

    # Trader participation breadth — a large number of speculative traders
    # holding positions suggests broader conviction than a few large ones.
    df["cot_noncomm_trader_count"] = df["traders_noncomm_long_all"] + df["traders_noncomm_short_all"]

    feature_columns = [
        "date",
        "cot_noncomm_net",
        "cot_noncomm_net_pct_oi",
        "cot_comm_net",
        "cot_comm_net_pct_oi",
        "cot_noncomm_net_change_1w",
        "cot_noncomm_net_ma_4w",
        "cot_noncomm_trader_count",
    ]

    return df[feature_columns]


# --------------------------------------------------------------------------
# Caching (COT updates only weekly — no need to hit the API every run)
# --------------------------------------------------------------------------

def load_or_fetch_cot_features(force_refresh: bool = False) -> pd.DataFrame:
    """
    Returns COT features, using a local cache when it's still fresh
    (updated within the last 7 days) to avoid hitting the CFTC API on
    every single feature-engineering run — COT data only changes weekly
    regardless of how often this is called.
    """
    if not force_refresh and os.path.exists(COT_CACHE_PATH):
        cached_df = pd.read_csv(COT_CACHE_PATH, parse_dates=["date"])

        if not cached_df.empty:
            cache_age_days = (datetime.now().date() - cached_df["date"].max().date()).days

            if cache_age_days < 7:
                logger.info("Using cached COT data (%d day(s) old).", cache_age_days)
                return cached_df

    logger.info("Fetching fresh COT data from CFTC...")
    raw_df = fetch_gold_cot_data()

    if raw_df.empty:
        if os.path.exists(COT_CACHE_PATH):
            logger.warning("CFTC fetch failed; falling back to existing cache.")
            return pd.read_csv(COT_CACHE_PATH, parse_dates=["date"])
        logger.warning("CFTC fetch failed and no cache exists. Returning empty COT features.")
        return pd.DataFrame(columns=["date"])

    feature_df = build_cot_features(raw_df)
    feature_df.to_csv(COT_CACHE_PATH, index=False)
    logger.info("Cached %d COT feature row(s) to %s.", len(feature_df), COT_CACHE_PATH)

    return feature_df


# --------------------------------------------------------------------------
# Merge into a daily price series (forward-fill, never look ahead)
# --------------------------------------------------------------------------

def merge_cot_into_daily(daily_df: pd.DataFrame, date_column: str = "Date") -> pd.DataFrame:
    """
    Merges weekly COT features into a daily price dataframe via an as-of
    (backward) merge: each trading day receives the most recently
    PUBLISHED COT report as of that day — never a future one. This is the
    critical correctness guarantee that prevents look-ahead leakage from
    weekly data into a daily model.
    """
    cot_df = load_or_fetch_cot_features()

    if cot_df.empty:
        logger.warning("No COT data available; proceeding without COT features.")
        return daily_df

    daily_df = daily_df.copy()
    daily_df[date_column] = pd.to_datetime(daily_df[date_column])
    cot_df = cot_df.sort_values("date")

    # CFTC publishes Tuesday's data the following Friday — add a 3-day
    # publication lag so a trading day only "sees" a COT report that had
    # actually been released by that date, not merely measured by then.
    cot_df = cot_df.copy()
    cot_df["available_from"] = cot_df["date"] + pd.Timedelta(days=3)

    merged_df = pd.merge_asof(
        daily_df.sort_values(date_column),
        cot_df.drop(columns=["date"]).sort_values("available_from"),
        left_on=date_column,
        right_on="available_from",
        direction="backward",
    )

    merged_df = merged_df.drop(columns=["available_from"])

    return merged_df


if __name__ == "__main__":
    cot_features = load_or_fetch_cot_features(force_refresh=True)

    if cot_features.empty:
        print("No COT data could be retrieved.")
    else:
        print(f"Retrieved {len(cot_features)} weekly COT report(s) for Gold.")
        print(f"Date range: {cot_features['date'].min().date()} to {cot_features['date'].max().date()}")
        print("\nMost recent 3 reports:")
        print(cot_features.tail(3).to_string(index=False))
"""
Om Gold Intelligence — Update Pipeline (ml/update_data.py)

Runs the full data + model refresh pipeline:

    data_collection.py     -> pulls latest market data
    feature_engineering.py -> rebuilds the model-ready feature matrix
    train_model.py          -> retrains all five horizons in one run

    python update_data.py            # normal run (skips if already fresh)
    python update_data.py --force    # always runs the full pipeline

WHY THIS FILE WAS REWRITTEN
----------------------------------------------------------------------------
1. `subprocess.run(["python", ...])` — on many machines (especially macOS/
   Linux where only `python3` is on PATH), a bare "python" call fails
   outright, silently breaking every background refresh triggered from
   app.py. This now uses `sys.executable`, i.e. the exact same Python
   interpreter already running this script — guaranteed to exist and to be
   the correct one (same virtualenv, same installed packages).

2. The old SCRIPTS list included train_model_7d.py and train_model_30d.py.
   Those are now obsolete — train_model.py trains all five horizons
   (1/7/14/21/30 day) in a single run. Keeping the old per-horizon scripts
   in this pipeline would have wasted time re-training with the old,
   leakage-prone logic and then immediately overwriting those same model
   files again with train_model.py's output. train_direction_model.py is
   also dropped from this pipeline since app.py never loads or serves
   its output — remove this comment and add it back to SCRIPTS if that
   changes.

3. NO FRESHNESS SHORT-CIRCUIT: the old script always ran the entire
   pipeline (including a full data re-download and a full five-model
   retrain) every single time it was invoked — including every time
   app.py's background-refresh trigger fired. This version checks whether
   the data is already fresh first, and skips straight to a no-op if so
   (pass --force to override this and always run everything).

4. NO OVERLAP PROTECTION: if this script were triggered twice concurrently
   (e.g. a manual run while app.py's background thread is also mid-refresh),
   both processes would race to write the same CSV/model files. A simple
   lock file now prevents a second run from starting while one is already
   in progress.

5. SILENT/OPAQUE FAILURES: the old script only printed subprocess output
   and raised a bare RuntimeError. This version records a structured
   status report (success/failure, which step failed, how long each step
   took) in last_updated.json, so app.py — or you, looking at the file
   directly — can see exactly what happened on the last run.
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MARKET_DATA_PATH = DATA_DIR / "market_data.csv"
LAST_UPDATED_PATH = BASE_DIR / "last_updated.json"
LOCK_FILE_PATH = BASE_DIR / ".update_pipeline.lock"

# Pipeline steps, in order. train_model.py now trains all five horizons in
# a single run — the old per-horizon scripts (train_model_7d.py,
# train_model_30d.py) are intentionally excluded; see module docstring.
SCRIPTS = [
    "data_collection.py",
    "feature_engineering.py",
    "train_model.py",
]

# If the newest row in market_data.csv is this many days old or newer, a
# normal (non --force) run will skip the pipeline entirely rather than
# re-downloading and re-training for no reason.
FRESHNESS_THRESHOLD_DAYS = 1


# --------------------------------------------------------------------------
# Freshness check
# --------------------------------------------------------------------------

def get_current_data_age_days():
    """
    Returns how many days old the latest row of market_data.csv is, or
    None if the file doesn't exist yet / has no usable rows (in which case
    the pipeline should always run).
    """
    if not MARKET_DATA_PATH.exists():
        return None

    try:
        df = pd.read_csv(MARKET_DATA_PATH)
        df["Date"] = pd.to_datetime(df["Date"])
        valid_dates = df.dropna(subset=["Gold"])["Date"]

        if valid_dates.empty:
            return None

        latest_date = valid_dates.max().date()
        return (datetime.now().date() - latest_date).days
    except Exception as exc:  # noqa: BLE001 - a corrupt/unreadable file should trigger a full run
        print(f"Could not determine current data age ({exc}). Will run the full pipeline.")
        return None


# --------------------------------------------------------------------------
# Lock file — prevents two overlapping pipeline runs
# --------------------------------------------------------------------------

def acquire_lock() -> bool:
    if LOCK_FILE_PATH.exists():
        try:
            lock_age_seconds = time.time() - LOCK_FILE_PATH.stat().st_mtime
        except OSError:
            lock_age_seconds = 0

        # A lock older than 30 minutes almost certainly means a previous
        # run crashed without cleaning up, rather than a genuinely
        # still-running pipeline — treat it as stale and proceed.
        if lock_age_seconds < 1800:
            return False

        print("Found a stale lock file (>30 min old). Proceeding anyway.")

    LOCK_FILE_PATH.write_text(str(os.getpid()))
    return True


def release_lock():
    try:
        LOCK_FILE_PATH.unlink(missing_ok=True)
    except OSError:
        pass


# --------------------------------------------------------------------------
# Running each pipeline step
# --------------------------------------------------------------------------

def run_script(script_name: str) -> dict:
    import subprocess

    script_path = BASE_DIR / script_name

    if not script_path.exists():
        raise FileNotFoundError(f"Pipeline step not found: {script_path}")

    print("\n" + "=" * 60)
    print(f"Running {script_name}...")
    print("=" * 60)

    started_at = time.time()

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        cwd=str(BASE_DIR),
    )

    duration_seconds = round(time.time() - started_at, 1)

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"{script_name} failed after {duration_seconds}s (exit code {result.returncode})."
        )

    print(f"{script_name} completed successfully in {duration_seconds}s.")

    return {"script": script_name, "duration_seconds": duration_seconds, "status": "success"}


# --------------------------------------------------------------------------
# Status reporting
# --------------------------------------------------------------------------

def write_status(status: str, steps_run=None, error_message=None, skipped=False, data_age_days=None):
    report = {
        "last_updated": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
        "last_updated_iso": datetime.now().isoformat(),
        "status": status,
        "skipped": skipped,
        "data_age_days_before_run": data_age_days,
        "steps_run": steps_run or [],
    }

    if error_message:
        report["error"] = error_message

    with open(LAST_UPDATED_PATH, "w", encoding="utf-8") as status_file:
        json.dump(report, status_file, indent=4)

    print(f"\nStatus written to {LAST_UPDATED_PATH}")


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def main(force: bool = False):
    print("\nStarting Gold Intelligence Update Pipeline...\n")

    if not acquire_lock():
        print("Another update is already in progress. Exiting without changes.")
        return

    try:
        data_age_days = get_current_data_age_days()

        if not force and data_age_days is not None and data_age_days <= FRESHNESS_THRESHOLD_DAYS:
            print(
                f"Market data is already fresh ({data_age_days} day(s) old, "
                f"threshold is {FRESHNESS_THRESHOLD_DAYS}). Skipping pipeline. "
                "Pass --force to run anyway."
            )
            write_status(status="skipped_already_fresh", skipped=True, data_age_days=data_age_days)
            return

        if data_age_days is None:
            print("No usable existing data found. Running the full pipeline.")
        else:
            print(f"Market data is {data_age_days} day(s) old. Running the full pipeline.")

        steps_run = []

        for script in SCRIPTS:
            try:
                step_result = run_script(script)
                steps_run.append(step_result)
            except Exception as exc:
                steps_run.append({"script": script, "status": "failed", "error": str(exc)})
                write_status(
                    status="failed",
                    steps_run=steps_run,
                    error_message=str(exc),
                    data_age_days=data_age_days,
                )
                raise

        write_status(status="success", steps_run=steps_run, data_age_days=data_age_days)

        print("\n" + "=" * 60)
        print("Gold Intelligence is fully updated!")
        print("=" * 60)

    finally:
        release_lock()


if __name__ == "__main__":
    force_flag = "--force" in sys.argv
    main(force=force_flag)
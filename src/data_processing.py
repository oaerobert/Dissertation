from __future__ import annotations

import os
import numpy as np
import pandas as pd


# 1) List your CSVs explicitly (Windows paths).
CSV_PATHS = [
    r"C:\Users\okeae\Downloads\GUI_ENERGY_PRICES_201501010000-201601010000.csv",
    r"C:\Users\okeae\Downloads\GUI_ENERGY_PRICES_201601010000-201701010000.csv",
    r"C:\Users\okeae\Downloads\GUI_ENERGY_PRICES_201701010000-201801010000.csv",
    r"C:\Users\okeae\Downloads\GUI_ENERGY_PRICES_201801010000-201901010000.csv",
    r"C:\Users\okeae\Downloads\GUI_ENERGY_PRICES_201901010000-202001010000.csv",
]

# 2) Output paths (relative to where you run the script).
OUT_DIR = os.path.join("data", "processed")
OUT_CSV = os.path.join(OUT_DIR, "uk_hourly_master.csv")
OUT_PARQUET = os.path.join(OUT_DIR, "uk_hourly_master.parquet")


def parse_mtu_start_utc(mtu_value: str) -> pd.Timestamp:
    """
    Convert the 'MTU (UTC)' string like:
      '01/01/2015 00:00:00 - 01/01/2015 01:00:00'
    into the START timestamp as a timezone-aware UTC datetime.
    """
    # Split on the dash to separate start and end times.
    start_str = str(mtu_value).split("-")[0].strip()

    # Parse with day-first format (your file uses DD/MM/YYYY).
    # utc=True makes it timezone-aware and set to UTC.
    return pd.to_datetime(start_str, dayfirst=True, utc=True, errors="coerce")


def read_one_csv(path: str) -> pd.DataFrame:
    """
    Read one yearly CSV and keep only the columns we need.
    """
    df = pd.read_csv(path)

    # Keep only relevant columns.
    df = df[["MTU (UTC)", "Area", "Day-ahead Price (GBP/MWh)"]].copy()

    # Parse timestamp (start of MTU interval) into UTC datetime.
    df["timestamp_utc"] = df["MTU (UTC)"].map(parse_mtu_start_utc)

    # Convert price to numeric (coerce bad strings to NaN).
    df["day_ahead_price"] = pd.to_numeric(df["Day-ahead Price (GBP/MWh)"], errors="coerce")

    # Keep only clean rows.
    df = df.dropna(subset=["timestamp_utc", "day_ahead_price"])

    # Rename / keep only final raw columns we want.
    df = df[["timestamp_utc", "Area", "day_ahead_price"]].rename(columns={"Area": "area"})

    # Unit is implied by the column name; store it explicitly.
    df["unit"] = "GBP/MWh"

    return df


def enforce_continuous_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (area, unit), reindex to a continuous hourly timeline.
    Missing hours become explicit rows and get flagged in is_missing_hour.
    """
    out = []

    for (area, unit), g in df.groupby(["area", "unit"], sort=False):
        g = g.sort_values("timestamp_utc").set_index("timestamp_utc")

        # Create a full hourly index from first to last timestamp.
        full_idx = pd.date_range(
            start=g.index.min().floor("h"),
            end=g.index.max().ceil("h"),
            freq="h",
            tz="UTC",
        )

        # Reindex creates rows for missing timestamps.
        r = g.reindex(full_idx)
        r.index.name = "timestamp_utc"

        # Restore identifiers after reindexing.
        r["area"] = area
        r["unit"] = unit

        # Flag which rows were missing in the original data.
        r["is_missing_hour"] = r["day_ahead_price"].isna()

        out.append(r.reset_index())

    df2 = pd.concat(out, ignore_index=True)
    df2 = df2.sort_values(["area", "unit", "timestamp_utc"], kind="mergesort").reset_index(drop=True)
    return df2


def add_core_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add log price/returns, rolling vol, and seasonality columns.
    """
    df = df.sort_values(["area", "unit", "timestamp_utc"], kind="mergesort").copy()

    # Seasonality
    df["hour"] = df["timestamp_utc"].dt.hour.astype("int16")
    df["dayofweek"] = df["timestamp_utc"].dt.dayofweek.astype("int16")  # Monday=0
    df["month"] = df["timestamp_utc"].dt.month.astype("int16")

    # Log price: only defined if price > 0
    df["log_price"] = np.where(df["day_ahead_price"] > 0, np.log(df["day_ahead_price"]), np.nan)

    # Log return = log_price - previous hour log_price (per area/unit)
    df["log_return"] = df.groupby(["area", "unit"], sort=False)["log_price"].diff(1)

    # Rolling realised vol = std of returns over windows
    df["rv_24h"] = (
        df.groupby(["area", "unit"], sort=False)["log_return"]
        .rolling(24, min_periods=24)
        .std()
        .reset_index(level=[0, 1], drop=True)
    )

    df["rv_168h"] = (
        df.groupby(["area", "unit"], sort=False)["log_return"]
        .rolling(168, min_periods=168)
        .std()
        .reset_index(level=[0, 1], drop=True)
    )

    return df


def main() -> None:
    # Read and stack all years.
    frames = [read_one_csv(p) for p in CSV_PATHS]
    df = pd.concat(frames, ignore_index=True)

    # Sort strictly by keys.
    df = df.sort_values(["area", "unit", "timestamp_utc"], kind="mergesort")

    # Remove duplicates: same (area, unit, timestamp) → keep last.
    df = df.drop_duplicates(subset=["area", "unit", "timestamp_utc"], keep="last")

    # Enforce continuous hourly timeline and flag missing hours.
    df = enforce_continuous_hourly(df)

    # Add core engineered columns.
    df = add_core_features(df)

    # Final column order (keep it clean).
    df = df[
        [
            "timestamp_utc",
            "area",
            "unit",
            "day_ahead_price",
            "is_missing_hour",
            "log_price",
            "log_return",
            "rv_24h",
            "rv_168h",
            "hour",
            "dayofweek",
            "month",
        ]
    ].sort_values(["area", "unit", "timestamp_utc"], kind="mergesort").reset_index(drop=True)

    # Create output directory and write files.
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    df.to_parquet(OUT_PARQUET, index=False)

    # Basic sanity checks
    if not pd.api.types.is_datetime64tz_dtype(df["timestamp_utc"]):
        raise RuntimeError("timestamp_utc is not timezone-aware UTC datetime.")

    # Confirm strict hourly continuity per group (timestamps only)
    for (area, unit), g in df.groupby(["area", "unit"], sort=False):
        diffs = g["timestamp_utc"].diff().dropna()
        if not (diffs == pd.Timedelta(hours=1)).all():
            raise RuntimeError(f"Non-hourly gaps remain in {area}/{unit} timeline.")

    print(f"Wrote CSV: {OUT_CSV}")
    print(f"Wrote Parquet: {OUT_PARQUET}")
    print(f"Rows: {len(df):,} | Missing-hour rows: {int(df['is_missing_hour'].sum()):,}")


if __name__ == "__main__":
    main()

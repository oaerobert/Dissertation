from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_arch


# --------- USER SETTINGS ---------
INPUT_CSV = os.path.join("data", "processed", "processed_data.csv")

FIG_DIR = os.path.join("results", "figures")
TAB_DIR = os.path.join("results", "tables")

# Spike definition choice:
# Option A: return spike if |return| > k * rolling_7d_std (7d = 168 hours)
SPIKE_K = 3.0
ROLL_STD_HOURS = 168

# If your file includes multiple areas, set one; otherwise leave as None.
AREA_FILTER = None  # e.g. "BZN|GB"


def ensure_dirs() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TAB_DIR, exist_ok=True)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_CSV)

    # Parse timestamp to timezone-aware UTC; if it's already ISO with +00:00, pandas will keep it.
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")

    # Keep only rows with valid timestamp
    df = df.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc")

    # Optional: select one area if multiple exist
    if AREA_FILTER is not None and "area" in df.columns:
        df = df[df["area"].astype(str) == AREA_FILTER].copy()

    # Make sure price column exists (some people rename it)
    if "day_ahead_price" not in df.columns:
        raise ValueError("Expected column 'day_ahead_price' not found in processed_data.csv")

    # If returns aren't present for some reason, compute log_return from price
    if "log_return" not in df.columns:
        price = pd.to_numeric(df["day_ahead_price"], errors="coerce")
        log_price = np.where(price > 0, np.log(price), np.nan)
        df["log_return"] = pd.Series(log_price).diff()

    # Force numeric
    df["day_ahead_price"] = pd.to_numeric(df["day_ahead_price"], errors="coerce")
    df["log_return"] = pd.to_numeric(df["log_return"], errors="coerce")

    return df.reset_index(drop=True)


def define_spikes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Spike if |return| > rolling_7d_std * k
    """
    df = df.sort_values("timestamp_utc").copy()

    # Rolling 7-day std of returns (hourly data: 168 hours)
    df["ret_std_7d"] = df["log_return"].rolling(ROLL_STD_HOURS, min_periods=ROLL_STD_HOURS).std()

    # Spike indicator
    df["is_spike"] = (df["log_return"].abs() > (SPIKE_K * df["ret_std_7d"]))

    return df


def save_summary_stats(df: pd.DataFrame) -> None:
    """
    Table: mean, sd, skewness, kurtosis for price and returns.
    """
    price = df["day_ahead_price"].dropna()
    ret = df["log_return"].dropna()

    summary = pd.DataFrame(
        {
            "series": ["price", "returns"],
            "mean": [price.mean(), ret.mean()],
            "sd": [price.std(ddof=1), ret.std(ddof=1)],
            "skewness": [price.skew(), ret.skew()],
            "kurtosis": [price.kurtosis(), ret.kurtosis()],
            "n": [price.shape[0], ret.shape[0]],
        }
    )

    out_path = os.path.join(TAB_DIR, "summary_stats.csv")
    summary.to_csv(out_path, index=False)


def save_spike_frequency(df: pd.DataFrame) -> None:
    """
    Table: spike count and percentage of sample.
    """
    valid = df["is_spike"].dropna()
    spike_count = int(valid.sum())
    n = int(valid.shape[0])
    pct = (spike_count / n) * 100 if n > 0 else np.nan

    out = pd.DataFrame(
        {
            "definition": [f"|return| > {SPIKE_K} * rolling_{ROLL_STD_HOURS}h_std"],
            "spike_count": [spike_count],
            "sample_size": [n],
            "spike_pct": [pct],
        }
    )

    out_path = os.path.join(TAB_DIR, "spike_frequency.csv")
    out.to_csv(out_path, index=False)


def plot_price_series(df: pd.DataFrame) -> None:
    plt.figure()
    plt.plot(df["timestamp_utc"], df["day_ahead_price"])
    plt.xlabel("Time (UTC)")
    plt.ylabel("Day-ahead price")
    plt.title("Price time series")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "price_time_series.png"), dpi=200)
    plt.close()


def plot_return_series(df: pd.DataFrame) -> None:
    plt.figure()
    plt.plot(df["timestamp_utc"], df["log_return"])
    plt.xlabel("Time (UTC)")
    plt.ylabel("Log return")
    plt.title("Return time series")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "return_time_series.png"), dpi=200)
    plt.close()


def plot_return_histogram(df: pd.DataFrame) -> None:
    r = df["log_return"].dropna()
    plt.figure()
    plt.hist(r, bins=100)
    plt.xlabel("Log return")
    plt.ylabel("Frequency")
    plt.title("Histogram of returns")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "return_histogram.png"), dpi=200)
    plt.close()


def plot_rolling_vol(df: pd.DataFrame) -> None:
    """
    Rolling volatility plot (24h and 7d).
    If rv_24h/rv_168h columns exist from your processing step, use them.
    Otherwise compute from returns.
    """
    d = df.sort_values("timestamp_utc").copy()

    if "rv_24h" not in d.columns:
        d["rv_24h"] = d["log_return"].rolling(24, min_periods=24).std()
    if "rv_168h" not in d.columns:
        d["rv_168h"] = d["log_return"].rolling(168, min_periods=168).std()

    plt.figure()
    plt.plot(d["timestamp_utc"], d["rv_24h"], label="24h")
    plt.plot(d["timestamp_utc"], d["rv_168h"], label="7d (168h)")
    plt.xlabel("Time (UTC)")
    plt.ylabel("Realised volatility (std of returns)")
    plt.title("Rolling volatility (24h and 7d)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "rolling_volatility.png"), dpi=200)
    plt.close()


def acf_manual(x: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Simple ACF (no confidence bands), so you don't need extra packages.
    """
    x = x[np.isfinite(x)]
    x = x - x.mean()
    denom = np.dot(x, x)
    acf_vals = np.empty(max_lag + 1)
    acf_vals[0] = 1.0
    for lag in range(1, max_lag + 1):
        acf_vals[lag] = np.dot(x[:-lag], x[lag:]) / denom
    return acf_vals


def plot_acf_returns_and_squared(df: pd.DataFrame, max_lag: int = 100) -> None:
    r = df["log_return"].to_numpy(dtype=float)
    r2 = (df["log_return"] ** 2).to_numpy(dtype=float)

    acf_r = acf_manual(r, max_lag)
    acf_r2 = acf_manual(r2, max_lag)
    lags = np.arange(0, max_lag + 1)

    plt.figure()
    plt.stem(lags, acf_r, basefmt=" ")
    plt.xlabel("Lag")
    plt.ylabel("ACF")
    plt.title("ACF of returns")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "acf_returns.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.stem(lags, acf_r2, basefmt=" ")
    plt.xlabel("Lag")
    plt.ylabel("ACF")
    plt.title("ACF of squared returns")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "acf_squared_returns.png"), dpi=200)
    plt.close()


def run_gatekeeper_tests(df: pd.DataFrame) -> None:
    """
    Saves ADF and ARCH LM test results to a small CSV.
    """
    r = df["log_return"].dropna()

    # ADF stationarity test
    adf_stat, adf_p, _, _, adf_crit, _ = adfuller(r, autolag="AIC")

    # ARCH LM test (heteroskedasticity)
    # Choose lags; 24 is a common “1 day” choice for hourly data
    arch_lm_stat, arch_lm_p, _, _ = het_arch(r, nlags=24)

    out = pd.DataFrame(
        {
            "test": ["ADF (returns)", "ARCH LM (returns, nlags=24)"],
            "statistic": [adf_stat, arch_lm_stat],
            "p_value": [adf_p, arch_lm_p],
            "crit_1pct": [adf_crit.get("1%"), np.nan],
            "crit_5pct": [adf_crit.get("5%"), np.nan],
            "crit_10pct": [adf_crit.get("10%"), np.nan],
        }
    )

    out_path = os.path.join(TAB_DIR, "gatekeeper_tests.csv")
    out.to_csv(out_path, index=False)


def main() -> None:
    ensure_dirs()

    df = load_data()
    df = define_spikes(df)

    # ---- Figures ----
    plot_price_series(df)
    plot_return_series(df)
    plot_return_histogram(df)
    plot_rolling_vol(df)
    plot_acf_returns_and_squared(df, max_lag=100)

    # ---- Tables ----
    save_summary_stats(df)
    save_spike_frequency(df)
    run_gatekeeper_tests(df)

    # Optional: also save the analysis-ready dataset with spike flag
    df.to_csv(os.path.join(TAB_DIR, "data_with_spikes.csv"), index=False)

    print(f"Saved figures to: {FIG_DIR}")
    print(f"Saved tables to: {TAB_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()

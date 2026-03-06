import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
from arch import arch_model


# =========================================================
# CONFIG
# =========================================================
DATA_PATH = Path(r"C:\Users\okeae\PycharmProjects\Dissertation Coding\.venv\data\processed\processed_data.csv")
OUTPUT_DIR = Path(r"C:\Users\okeae\PycharmProjects\Dissertation Coding\.venv\data\results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TIME_COL = "timestamp_utc"
RETURN_COL = "log_return"
VOL_TARGET_COL = "rv_24h"

TEST_START_DATE = "2019-01-01"
MIN_TRAIN_SIZE = 24 * 365
REFIT_EVERY = 24 * 7

RF_RMSE = 0.009145874
RF_MAE = 0.005036


# =========================================================
# LOAD + CLEAN
# =========================================================
df = pd.read_csv(DATA_PATH)

required_cols = [TIME_COL, RETURN_COL, VOL_TARGET_COL]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df[TIME_COL] = pd.to_datetime(df[TIME_COL], utc=True, errors="coerce")
df = df.sort_values(TIME_COL).reset_index(drop=True)

df = df[[TIME_COL, RETURN_COL, VOL_TARGET_COL]].copy()
df[RETURN_COL] = pd.to_numeric(df[RETURN_COL], errors="coerce")
df[VOL_TARGET_COL] = pd.to_numeric(df[VOL_TARGET_COL], errors="coerce")
df = df.dropna(subset=[TIME_COL, RETURN_COL, VOL_TARGET_COL]).reset_index(drop=True)

if len(df) < MIN_TRAIN_SIZE + 100:
    raise ValueError("Not enough data after cleaning.")

test_start_ts = pd.Timestamp(TEST_START_DATE, tz="UTC")
test_indices = df.index[df[TIME_COL] >= test_start_ts]

if len(test_indices) == 0:
    raise ValueError(f"No rows found on or after {TEST_START_DATE}.")

test_start_idx = int(test_indices[0])

if test_start_idx < MIN_TRAIN_SIZE:
    test_start_idx = MIN_TRAIN_SIZE

print(f"Total observations: {len(df):,}")
print(f"Train size before test start: {test_start_idx:,}")
print(f"Test observations: {len(df) - test_start_idx:,}")
print(f"Evaluation period: {df.iloc[test_start_idx][TIME_COL]} to {df.iloc[-1][TIME_COL]}")


# =========================================================
# MODEL FITTERS
# =========================================================
def fit_vol_model(train_returns: pd.Series, model_name: str):
    train_returns = train_returns.dropna()

    if model_name == "GARCH":
        model = arch_model(
            train_returns,
            mean="AR",
            lags=1,
            vol="GARCH",
            p=1,
            q=1,
            dist="normal",
            rescale=False
        )

    elif model_name == "EGARCH":
        model = arch_model(
            train_returns,
            mean="AR",
            lags=1,
            vol="EGARCH",
            p=1,
            q=1,
            dist="normal",
            rescale=False
        )

    elif model_name == "GJR":
        model = arch_model(
            train_returns,
            mean="AR",
            lags=1,
            vol="GARCH",
            p=1,
            o=1,
            q=1,
            dist="normal",
            rescale=False
        )

    else:
        raise ValueError("model_name must be one of: GARCH, EGARCH, GJR")

    res = model.fit(disp="off", show_warning=False)
    return res


def one_step_vol_forecast(fitted_res):
    fc = fitted_res.forecast(horizon=1, reindex=False)
    var_fc = float(fc.variance.iloc[-1, 0])
    vol_fc = np.sqrt(max(var_fc, 0.0))
    return vol_fc


# =========================================================
# ROLLING OUT-OF-SAMPLE FORECASTS
# =========================================================
models = ["GARCH", "EGARCH", "GJR"]
cached_models = {m: None for m in models}
forecast_rows = []

for i in range(test_start_idx, len(df)):
    row = {
        TIME_COL: df.iloc[i][TIME_COL],
        "realized_vol": float(df.iloc[i][VOL_TARGET_COL])
    }

    train_returns = df.iloc[:i][RETURN_COL]
    refit_now = ((i - test_start_idx) % REFIT_EVERY == 0)

    for m in models:
        try:
            if refit_now or cached_models[m] is None:
                fitted = fit_vol_model(train_returns, m)
                cached_models[m] = fitted
            else:
                fitted = cached_models[m]

            row[f"{m}_forecast"] = one_step_vol_forecast(fitted)

        except Exception as e:
            print(f"[WARN] {m} failed at index {i}: {e}")
            row[f"{m}_forecast"] = np.nan

    forecast_rows.append(row)

forecast_df = pd.DataFrame(forecast_rows)
forecast_df = forecast_df.dropna().reset_index(drop=True)

if forecast_df.empty:
    raise ValueError("No valid forecasts generated.")

forecast_file = OUTPUT_DIR / "econometric_volatility_forecasts.csv"
forecast_df.to_csv(forecast_file, index=False)
print(f"Saved econometric forecast path to: {forecast_file}")


# =========================================================
# FORECAST ACCURACY COMPARISON
# =========================================================
results = []

for m in models:
    pred_col = f"{m}_forecast"
    valid = forecast_df.dropna(subset=["realized_vol", pred_col]).copy()

    if len(valid) == 0:
        continue

    mse = mean_squared_error(valid["realized_vol"], valid[pred_col])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(valid["realized_vol"], valid[pred_col])

    results.append({
        "task": "volatility",
        "model": m,
        "rmse": rmse,
        "mae": mae
    })

results.append({
    "task": "volatility",
    "model": "RandomForest",
    "rmse": RF_RMSE,
    "mae": RF_MAE
})

comparison_df = pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)

comparison_file = OUTPUT_DIR / "forecast_accuracy_comparison.csv"
comparison_df.to_csv(comparison_file, index=False)

print("\nForecast Accuracy Comparison")
print(comparison_df.to_string(index=False))
print(f"\nSaved comparison table to: {comparison_file}")

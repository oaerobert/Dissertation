import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


# ==========================================================
# 1) LOAD PROCESSED DATA (your path)
# ==========================================================

FILE_PATH = r"C:\Users\okeae\PycharmProjects\Dissertation Coding\.venv\data\processed\processed_data.csv"
df = pd.read_csv(FILE_PATH)

# Assumes these exist from your processing pipeline:
# timestamp_utc, log_return, rv_24h (or you can compute), hour, dayofweek, month, is_spike (optional)
df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
df = df.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc").reset_index(drop=True)

df["log_return"] = pd.to_numeric(df["log_return"], errors="coerce")
df = df.dropna(subset=["log_return"]).reset_index(drop=True)

# If you don't already have realised volatility columns, create them.
if "rv_24h" not in df.columns:
    df["rv_24h"] = df["log_return"].rolling(24, min_periods=24).std()

if "rv_168h" not in df.columns:
    df["rv_168h"] = df["log_return"].rolling(168, min_periods=168).std()


# ==========================================================
# 2) SPIKE LABEL (if not already present)
#    Uses your earlier definition: |r| > k * rolling_7d_std
# ==========================================================

if "is_spike" not in df.columns:
    K = 3.0
    df["ret_std_7d"] = df["log_return"].rolling(168, min_periods=168).std()
    df["is_spike"] = (df["log_return"].abs() > (K * df["ret_std_7d"]))


# ==========================================================
# 3) FEATURE ENGINEERING (lags + seasonality)
# ==========================================================

# Lagged returns (1..24) and weekly lag (168)
for lag in range(1, 25):
    df[f"r_lag_{lag}"] = df["log_return"].shift(lag)

df["r_lag_168"] = df["log_return"].shift(168)

# Lagged realised volatility features (useful predictors)
df["rv24_lag_1"] = df["rv_24h"].shift(1)
df["rv168_lag_1"] = df["rv_168h"].shift(1)

# Seasonality features (if missing, create)
if "hour" not in df.columns:
    df["hour"] = df["timestamp_utc"].dt.hour
if "dayofweek" not in df.columns:
    df["dayofweek"] = df["timestamp_utc"].dt.dayofweek
if "month" not in df.columns:
    df["month"] = df["timestamp_utc"].dt.month

# Target 1: next-hour realised volatility proxy (predict rv_24h at t+1)
df["y_vol_next"] = df["rv_24h"].shift(-1)

# Target 2: spike at next hour (predict is_spike at t+1)
df["y_spike_next"] = df["is_spike"].shift(-1)

# Final feature list
FEATURES = (
    [f"r_lag_{lag}" for lag in range(1, 25)]
    + ["r_lag_168", "rv24_lag_1", "rv168_lag_1", "hour", "dayofweek", "month"]
)

# Drop rows with any missing in features/targets (from lags and rolling windows)
model_df = df.dropna(subset=FEATURES + ["y_vol_next", "y_spike_next"]).reset_index(drop=True)

X = model_df[FEATURES]
y_vol = model_df["y_vol_next"]
y_spike = model_df["y_spike_next"].astype(int)


# ==========================================================
# 4) ROLLING-WINDOW OUT-OF-SAMPLE VALIDATION
# ==========================================================

# Settings (hourly data):
TRAIN_HOURS = 24 * 365 * 2     # 2 years training
TEST_HOURS = 24 * 30           # 1 month test blocks
STEP_HOURS = TEST_HOURS        # roll forward by 1 month

vol_preds = []
vol_true = []

spike_preds = []
spike_probs = []
spike_true = []

n = len(model_df)
start = TRAIN_HOURS
end = start + TEST_HOURS

while end <= n:
    train_slice = slice(0, start)         # expanding window (safe + common)
    test_slice = slice(start, end)

    X_train, X_test = X.iloc[train_slice], X.iloc[test_slice]
    yv_train, yv_test = y_vol.iloc[train_slice], y_vol.iloc[test_slice]
    ys_train, ys_test = y_spike.iloc[train_slice], y_spike.iloc[test_slice]

    # --- Volatility regressor ---
    rf_reg = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    rf_reg.fit(X_train, yv_train)
    yv_hat = rf_reg.predict(X_test)

    vol_preds.extend(yv_hat.tolist())
    vol_true.extend(yv_test.tolist())

    # --- Spike classifier ---
    rf_clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    rf_clf.fit(X_train, ys_train)
    ys_prob = rf_clf.predict_proba(X_test)[:, 1]
    ys_hat = (ys_prob >= 0.5).astype(int)

    spike_probs.extend(ys_prob.tolist())
    spike_preds.extend(ys_hat.tolist())
    spike_true.extend(ys_test.tolist())

    # Roll forward
    start += STEP_HOURS
    end += STEP_HOURS


# ==========================================================
# 5) EVALUATION METRICS
# ==========================================================

vol_rmse = mean_squared_error(vol_true, vol_preds, squared=False)
vol_mae = mean_absolute_error(vol_true, vol_preds)

spike_precision = precision_score(spike_true, spike_preds, zero_division=0)
spike_recall = recall_score(spike_true, spike_preds, zero_division=0)
spike_f1 = f1_score(spike_true, spike_preds, zero_division=0)

# AUC needs both classes present; handle edge case
try:
    spike_auc = roc_auc_score(spike_true, spike_probs)
except ValueError:
    spike_auc = np.nan

print("=== VOLATILITY FORECAST (RF) ===")
print("RMSE:", vol_rmse)
print("MAE :", vol_mae)

print("\n=== SPIKE PREDICTION (RF) ===")
print("Precision:", spike_precision)
print("Recall   :", spike_recall)
print("F1       :", spike_f1)
print("AUC      :", spike_auc)


# ==========================================================
# 6) SAVE RESULTS TO CSV (for dissertation tables)
# ==========================================================

out_metrics = pd.DataFrame([
    {"task": "volatility", "model": "RandomForest", "rmse": vol_rmse, "mae": vol_mae},
    {"task": "spike", "model": "RandomForest", "precision": spike_precision, "recall": spike_recall, "f1": spike_f1, "auc": spike_auc},
])

OUT_PATH = r"C:\Users\okeae\PycharmProjects\Dissertation Coding\results\tables\ml_results.csv"
out_metrics.to_csv(OUT_PATH, index=False)
print("\nSaved ML results to:", OUT_PATH)

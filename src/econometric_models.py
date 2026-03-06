import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

FILE_PATH = r"C:\Users\okeae\PycharmProjects\Dissertation Coding\.venv\data\processed\processed_data.csv"

df = pd.read_csv(FILE_PATH)

print("Loaded rows:", len(df))
print("Columns:", df.columns.tolist())

# ==========================================================
# ARIMA(1,0,1) MEAN MODEL
# ==========================================================

model = ARIMA(df["log_return"].dropna(), order=(1, 0, 1))
arma_res = model.fit()
print(arma_res.summary())

# Use ARIMA residuals for volatility models
r = arma_res.resid.dropna() * 100

# ==========================================================
# GARCH FAMILY MODELS
# ==========================================================

# GARCH(1,1) Normal
garch = arch_model(r, vol="GARCH", p=1, q=1, dist="normal")
garch_res = garch.fit()
print(garch_res.summary())

# GARCH(1,1) Student-t
garch_t = arch_model(r, vol="GARCH", p=1, q=1, dist="t")
garch_t_res = garch_t.fit()
print(garch_t_res.summary())

# EGARCH(1,1) with asymmetry
egarch = arch_model(r, vol="EGARCH", p=1, o=1, q=1, dist="normal")
egarch_res = egarch.fit()
print(egarch_res.summary())

# GJR-GARCH(1,1)
gjr = arch_model(r, vol="GARCH", p=1, o=1, q=1, dist="normal")
gjr_res = gjr.fit()
print(gjr_res.summary())

# ==========================================================
# CLEAN MODEL COMPARISON TABLE
# ==========================================================

results_rows = []

def extract_model_row(name, res):
    params = res.params
    return {
        "model": name,
        "omega": params.get("omega", np.nan),
        "alpha1": params.get("alpha[1]", np.nan),
        "beta1": params.get("beta[1]", np.nan),
        "gamma1": params.get("gamma[1]", params.get("o[1]", np.nan)),
        "alpha_plus_beta": (
            params.get("alpha[1]", np.nan) + params.get("beta[1]", np.nan)
            if np.isfinite(params.get("alpha[1]", np.nan)) and np.isfinite(params.get("beta[1]", np.nan))
            else np.nan
        ),
        "nu": params.get("nu", np.nan),
        "loglik": res.loglikelihood,
        "aic": res.aic,
        "bic": res.bic
    }

results_rows.append({
    "model": "ARIMA(1,0,1)",
    "omega": np.nan,
    "alpha1": np.nan,
    "beta1": np.nan,
    "gamma1": np.nan,
    "alpha_plus_beta": np.nan,
    "nu": np.nan,
    "loglik": arma_res.llf,
    "aic": arma_res.aic,
    "bic": arma_res.bic
})

results_rows.append(extract_model_row("GARCH(1,1)-Normal", garch_res))
results_rows.append(extract_model_row("GARCH(1,1)-t", garch_t_res))
results_rows.append(extract_model_row("EGARCH(1,1)-Normal", egarch_res))
results_rows.append(extract_model_row("GJR-GARCH(1,1)-Normal", gjr_res))

econometrics_results = pd.DataFrame(results_rows)

OUTPUT_PATH = r"C:\Users\okeae\PycharmProjects\Dissertation Coding\econometrics_results.csv"
econometrics_results.to_csv(OUTPUT_PATH, index=False)

print("Saved econometrics results to:")
print(OUTPUT_PATH)

# ==========================================================
# PARAMETER INFERENCE TABLE
# ==========================================================

inference_rows = []

def extract_inference(name, res):
    for param in res.params.index:
        inference_rows.append({
            "model": name,
            "parameter": param,
            "estimate": res.params[param],
            "std_error": res.std_err[param],
            "t_stat": res.tvalues[param],
            "p_value": res.pvalues[param]
        })

extract_inference("GARCH(1,1)-Normal", garch_res)
extract_inference("GARCH(1,1)-t", garch_t_res)
extract_inference("EGARCH(1,1)-Normal", egarch_res)
extract_inference("GJR-GARCH(1,1)-Normal", gjr_res)

inference_df = pd.DataFrame(inference_rows)

INFERENCE_PATH = r"C:\Users\okeae\PycharmProjects\Dissertation Coding\econometrics_inference.csv"
inference_df.to_csv(INFERENCE_PATH, index=False)

print("Saved econometrics inference table to:")
print(INFERENCE_PATH)

print("Saved econometrics results to:")
print(OUTPUT_PATH)


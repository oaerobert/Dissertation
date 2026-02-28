import pandas as pd

FILE_PATH = r"C:\Users\okeae\PycharmProjects\Dissertation Coding\.venv\data\processed\processed_data.csv"

df = pd.read_csv(FILE_PATH)

print("Loaded rows:", len(df))
print("Columns:", df.columns.tolist())


from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(df["log_return"].dropna(), order=(1,0,1))
arma_res = model.fit()
print(arma_res.summary())

#ARIMA modelling is the one above (econometrics)

from arch import arch_model

r = df["log_return"].dropna() * 100  # scale improves convergence

garch = arch_model(r, vol="GARCH", p=1, q=1, dist="normal")
garch_res = garch.fit()
print(garch_res.summary())

#ARCH modelling is the one above (econometrics)

egarch = arch_model(r, vol="EGARCH", p=1, q=1, dist="normal")
egarch_res = egarch.fit()
print(egarch_res.summary())

#EGARCH modelling

garch_t = arch_model(r, vol="GARCH", p=1, q=1, dist="t")
garch_t_res = garch_t.fit()

#t-errors above

# ==========================================================
# CREATE CLEAN ECONOMETRICS RESULTS TABLE
# ==========================================================

import numpy as np

results_rows = []

# ---- ARIMA ----
results_rows.append({
    "model": "ARIMA(1,0,1)",
    "loglik": arma_res.llf,
    "aic": arma_res.aic,
    "bic": arma_res.bic,
    "alpha1": np.nan,
    "beta1": np.nan,
    "alpha_plus_beta": np.nan,
    "asymmetry_param": np.nan
})

# ---- GARCH(1,1) Normal ----
garch_params = garch_res.params.to_dict()

alpha = garch_params.get("alpha[1]", np.nan)
beta = garch_params.get("beta[1]", np.nan)

results_rows.append({
    "model": "GARCH(1,1)-Normal",
    "loglik": garch_res.loglikelihood,
    "aic": garch_res.aic,
    "bic": garch_res.bic,
    "alpha1": alpha,
    "beta1": beta,
    "alpha_plus_beta": alpha + beta,
    "asymmetry_param": np.nan
})

# ---- EGARCH(1,1) Normal ----
egarch_params = egarch_res.params.to_dict()

alpha_e = egarch_params.get("alpha[1]", np.nan)
beta_e = egarch_params.get("beta[1]", np.nan)
gamma_e = egarch_params.get("gamma[1]", np.nan)

results_rows.append({
    "model": "EGARCH(1,1)-Normal",
    "loglik": egarch_res.loglikelihood,
    "aic": egarch_res.aic,
    "bic": egarch_res.bic,
    "alpha1": alpha_e,
    "beta1": beta_e,
    "alpha_plus_beta": np.nan,  # not defined same way for EGARCH
    "asymmetry_param": gamma_e
})

# ==========================================================
# CONVERT TO DATAFRAME
# ==========================================================

econometrics_results = pd.DataFrame(results_rows)

# ==========================================================
# SAVE CSV
# ==========================================================

OUTPUT_PATH = r"C:\Users\okeae\PycharmProjects\Dissertation Coding\econometrics_results.csv"

econometrics_results.to_csv(OUTPUT_PATH, index=False)

print("Saved econometrics results to:")
print(OUTPUT_PATH)


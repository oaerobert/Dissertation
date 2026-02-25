# Dissertation
Quantifying the Effect of Renewable Generation on UK Electricity Price Volatility: an Econometric and Machine Learning Approach
# Quantifying the Effect of Renewable Generation on UK Electricity Price Volatility  
## An Econometric and Machine Learning Approach  

---

## 📘 Project Overview  

Over the past decade, the UK electricity market has undergone a structural transformation driven by legally binding net-zero targets and rapid renewable expansion. In 2024, renewables generated over 50% of UK electricity, with wind contributing roughly 29% of total generation :contentReference[oaicite:0]{index=0}.

While the merit-order effect implies that renewables reduce average wholesale prices, their impact on **price volatility** remains theoretically ambiguous and empirically mixed. Wind generation is inherently stochastic and forecast-dependent, raising concerns about short-run supply uncertainty and volatility clustering in day-ahead markets.

This project investigates:

> **To what extent does renewable generation — particularly wind output — affect the volatility of UK day-ahead electricity prices between 2015 and 2024?**

The study integrates **econometric volatility modelling (ARX–GARCH)** with **machine learning techniques (Random Forest)** to assess whether volatility effects are linear and stable, or nonlinear and regime-dependent.

---

## 🎯 Research Objectives  

1. Quantify the impact of wind output on **mean day-ahead prices** (merit-order effect).  
2. Estimate the effect of wind penetration on **conditional price volatility**.  
3. Compare parametric volatility models with flexible ML approaches.  
4. Assess whether volatility responses exhibit **nonlinear thresholds or regime dependence**.  

---

## 📚 Theoretical Background  

### Merit-Order Effect  

Wind and solar generation have near-zero marginal costs. Increased renewable output shifts the supply curve outward, displacing thermal generation and reducing equilibrium prices (Sensfuß et al., 2008; Gelabert et al., 2011; Ketterer, 2014).

This relationship is robust for **mean price levels**, but volatility dynamics remain contested.

### Competing Volatility Mechanisms  

| Mechanism | Predicted Effect |
|------------|------------------|
| Intermittency & forecast error | ↑ Short-run volatility |
| Reduced exposure to fuel price shocks | ↓ Volatility |
| System flexibility & storage | Conditional / regime-dependent |

Empirical findings across European markets are mixed, and high-frequency UK evidence remains comparatively limited :contentReference[oaicite:1]{index=1}.

---

## 🧠 Methodology  

### 1️⃣ Econometric Framework  

#### Baseline Mean Equation (ARX)

\[
P_t = \alpha + \beta_1 WindShare_t + \beta_2 GasPrice_t + \beta_3 Demand_t + \varepsilon_t
\]

#### Volatility Equation (GARCH-type)

\[
Var(P_t) = \gamma_0 + \gamma_1 WindShare_t + \gamma_2 Controls_t + u_t
\]

- ARX–GARCH(1,1) and EGARCH specifications  
- Wind share enters both mean and variance equations  
- Clustered standard errors  
- Rolling-window robustness checks  
- Sub-sample analysis (pre/post-2021)  

This allows direct testing of:

- **β₁ < 0** (merit-order effect)  
- **γ₁ > 0** (wind increases conditional variance)

---

### 2️⃣ Machine Learning Benchmark  

A **Random Forest Regressor** is implemented to capture:

- Nonlinear interactions  
- Regime-dependent volatility  
- Threshold effects in wind penetration  

Performance is evaluated using:

- MAE  
- RMSE  
- R²  
- Cross-validation  

The objective is not purely predictive accuracy, but structural comparison:

> Do nonlinear models reveal patterns obscured by linear GARCH specifications?

---

## 📊 Data  

**Period:** 2015–2024  
**Frequency:** High-frequency (day-ahead / hourly aggregated)

### Core Variables  

- UK Day-Ahead Electricity Price  
- Wind Generation / Wind Share  
- Gas Prices  
- Electricity Demand  
- Additional control variables  

COVID-era distortions (2020–2021) are excluded in baseline estimations and tested separately in robustness checks.

---

## 🧪 Comparative Design  

| Stage | Model | Purpose |
|-------|-------|----------|
| 1 | OLS / Fixed Effects | Mean price impact |
| 2 | ARX–GARCH | Conditional volatility modelling |
| 3 | Random Forest | Nonlinear benchmarking |

This sequential framework enables interpretation-first modelling before flexibility is introduced.

---

## 🔍 Robustness Checks  

- Lagged wind instruments  
- Rolling-window stability tests  
- Alternative volatility measures  
- Pre- vs post-2021 sub-samples  
- Clustered standard errors  
- EGARCH specification comparison  

---

## 📈 Expected Contributions  

1. Provides updated high-frequency evidence for the UK day-ahead market.  
2. Isolates wind-driven intermittency rather than aggregate renewable penetration.  
3. Integrates econometric volatility modelling with ML methods.  
4. Clarifies whether renewable expansion alters both **mean price levels** and **risk characteristics** of wholesale markets.  

---

## 🏛 Policy Relevance  

Understanding volatility effects is central to:

- Contract for Difference (CfD) scheme design  
- Capacity market reform  
- Risk pricing and balancing costs  
- Industrial electricity competitiveness  
- Net-zero market transition strategy  

If renewables reduce mean prices but increase variance, welfare implications depend on market design and hedging structures.

---

## 🛠 Technical Stack  

- Python  
  - statsmodels  
  - arch  
  - scikit-learn  
  - pandas  
  - numpy  
  - matplotlib  

---

## 📂 Repository Structure  

# Dissertation

# Quantifying the Effect of Renewable Generation on UK Electricity Price Volatility  
## An Econometric and Machine Learning Approach  

---

## 📘 Project Overview  

Over the past decade, the UK electricity market has undergone a structural transformation driven by legally binding net-zero targets and rapid renewable expansion. In 2024, renewables generated over 50% of UK electricity, with wind contributing roughly 29% of total generation.

While the merit-order effect implies that renewables reduce average wholesale prices, their impact on **price volatility** remains theoretically ambiguous and empirically mixed. Wind generation is inherently stochastic and forecast-dependent, raising concerns about short-run supply uncertainty and volatility clustering in day-ahead markets.

This project investigates:

> **To what extent does renewable generation — particularly wind output — affect the volatility of UK day-ahead electricity prices between 2015 and 2020?**

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

Wind and solar generation have near-zero marginal costs. Increased renewable output shifts the supply curve outward, displacing thermal generation and reducing equilibrium prices.

This relationship is robust for **mean price levels**, but volatility dynamics remain contested.

### Competing Volatility Mechanisms  

| Mechanism                              | Predicted Effect                     |
|-----------------------------------------|--------------------------------------|
| Intermittency & forecast error          | ↑ Short-run volatility               |
| Reduced exposure to fuel price shocks   | ↓ Volatility                        |
| System flexibility & storage expansion  | Conditional / regime-dependent       |

Empirical findings across European markets are mixed, and high-frequency UK evidence remains comparatively limited.

---

## 🧠 Methodology  

### 1️⃣ Econometric Framework  

#### Baseline Mean Equation (ARX)

```math
P_t = \alpha + \beta_1 WindShare_t + \beta_2 GasPrice_t + \beta_3 Demand_t + \varepsilon_t
```

#### Volatility Equation (GARCH-type)

```math
Var(P_t) = \gamma_0 + \gamma_1 WindShare_t + \gamma_2 Controls_t + u_t
```

**Specifications**

- ARX–GARCH(1,1) and EGARCH models  
- Wind share enters both mean and variance equations  
- Clustered standard errors  
- Rolling-window robustness checks  
- Sub-sample analysis (pre- and post-2021)  

**Testable Hypotheses**

- β₁ < 0  → Merit-order effect (renewables lower mean prices)  
- γ₁ > 0  → Wind increases conditional variance  

---

### 2️⃣ Machine Learning Benchmark  

A **Random Forest Regressor** is implemented to capture:

- Nonlinear interactions  
- Regime-dependent volatility  
- Threshold effects in wind penetration  

**Evaluation Metrics**

- MAE  
- RMSE  
- R²  
- Cross-validation  

The objective is not purely predictive accuracy, but structural comparison:

> Do nonlinear models reveal patterns obscured by linear GARCH specifications?

---

## Data Source

Day-ahead electricity price data were obtained from the ENTSO-E Transparency Platform:

ENTSO-E Transparency Platform  
https://transparency.entsoe.eu/

### Dataset Details

- Market Area (Bidding Zone): BZN|GB (Great Britain)
- Data Type: Day-Ahead Prices
- Resolution: Hourly
- Currency: GBP/MWh
- Time Standard: UTC (as provided by ENTSO-E)
- Date Range Used: 2015–2020

### How to Download the Data

1. Go to: https://transparency.entsoe.eu/
2. Navigate to:  
   `Market → Energy Prices`
3. Select:
   - Bidding Zone: **GB**
   - Product: **Day-Ahead Prices**
   - Time resolution: **Hourly**
4. Choose the desired year or date range.
5. Export the data as CSV.
6. Repeat for each year required.

Place all downloaded CSV files in:

data/raw/entsoe/

The raw data files are not included in this repository.

---

## 🧪 Comparative Design  

| Stage | Model              | Purpose                              |
|-------|-------------------|--------------------------------------|
| 1     | OLS / Fixed Effects | Estimate mean price impact           |
| 2     | ARX–GARCH         | Model conditional volatility         |
| 3     | Random Forest     | Capture nonlinear structure          |

This sequential framework prioritises interpretability before introducing flexible machine learning methods.

---

## 🔍 Robustness Checks  

- Lagged wind instruments  
- Rolling-window stability tests  
- Alternative volatility measures   
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

If renewables reduce mean prices but increase variance, welfare implications depend on market design, hedging structures, and system flexibility.

---

## 🛠 Technical Stack  

- **Python**
  - statsmodels  
  - arch  
  - scikit-learn  
  - pandas  
  - numpy  
  - matplotlib  

---

## 📂 Repository Structure  

```
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_descriptive_analysis.ipynb
│   ├── 03_garch_models.ipynb
│   └── 04_random_forest.ipynb
│
├── src/
│   ├── data_processing.py
│   ├── econometric_models.py
│   ├── ml_models.py
│   └── evaluation.py
│
├── results/
│   ├── tables/
│   └── figures/
│
└── README.md
```

---

## 🚀 Reproducibility  

1. Clone the repository  

```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies  

```bash
pip install -r requirements.txt
```

3. Run notebooks sequentially  

All model specifications, diagnostics, and robustness checks are documented within the repository.

---

## 📌 Key Hypotheses  

- **H1:** Renewable penetration reduces mean day-ahead prices.  
- **H2:** Wind intermittency increases conditional volatility.  
- **H3:** Machine learning models capture nonlinear volatility regimes better than linear GARCH frameworks.  

---

## ⚡ Summary  

This project rigorously evaluates whether renewable expansion in the UK reduces not only average electricity prices but also reshapes the volatility structure of wholesale markets. By integrating structural econometrics with flexible machine learning, it bridges causal interpretation and predictive performance in modern electricity market analysis.

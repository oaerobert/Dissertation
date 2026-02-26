# Are GARCH Models Enough?  
## Forecasting UK Day-Ahead Electricity Price Volatility (2015–2019)  
### Econometrics vs Machine Learning

---

## 📘 Dissertation Overview

This repository contains the full codebase and documentation for the undergraduate economics dissertation:

**“Are GARCH Models Enough? Forecasting UK Day-Ahead Electricity Price Volatility with Econometrics vs Machine Learning (2015–2019)”**

The project evaluates whether classical econometric volatility models (GARCH-family) remain sufficient in forecasting UK day-ahead electricity price volatility, or whether nonlinear machine-learning methods provide measurable improvements — particularly in spike prediction and tail behaviour.

The study focuses on **volatility structure and forecasting performance**, not causal drivers of price formation.

---

## 🎯 Research Question

> How do volatility dynamics in UK day-ahead electricity prices behave over 2015–2019, and do machine-learning models outperform GARCH-type econometric models in forecasting price volatility and spikes?

---

## 📊 Data

### Source

UK day-ahead hourly electricity prices were obtained from:

**ENTSO-E Transparency Platform**  
https://transparency.entsoe.eu/

### Data Characteristics

- Market: Great Britain (BZN|GB)  
- Frequency: Hourly  
- Period: 2015–2019  
- Unit: GBP/MWh  
- Type: Day-ahead market prices  

---

## ⬇️ How to Download the Data (CSV Format)

1. Go to the ENTSO-E Transparency Platform.  
2. Navigate to:  

   `Market → Energy Prices → Bidding Zone → Great Britain (GB)`

3. Select:
   - Data type: Day-ahead  
   - Time period (e.g. 01/01/2015 – 31/12/2019)  
4. Switch to table view.  
5. Use the download/export option to export as **CSV**.  

Repeat per year if required.

---

## 🗂 Repository Structure

```
.
├── data/
│   ├── raw/                 # Raw ENTSO-E CSV exports
│   ├── processed/           # Cleaned & transformed datasets
│
├── notebooks/               # Jupyter notebooks (exploratory + modelling)
│
├── src/
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── econometric_models.py
│   ├── ml_models.py
│   ├── evaluation.py
│
├── results/
│   ├── figures/
│   ├── tables/
│
├── dissertation.pdf
└── README.md
```

---

## ⚙️ Methodology Overview

### 1️⃣ Data Preparation

- Merge yearly hourly price CSV files  
- Handle missing values  
- Construct:
  - Log returns  
  - Rolling realised volatility (24h, 7-day)  
  - Seasonal dummies (hour-of-day, day-of-week, month)  
  - Spike indicator (extreme return threshold)  

---

### 2️⃣ Econometric Models

**Mean Equation**
- ARMA(p,q)

**Variance Equations**
- GARCH(1,1)  
- EGARCH  
- GJR-GARCH  
- Student-t innovations  

**Diagnostics**
- ADF stationarity test  
- ARCH LM test  
- Residual autocorrelation tests  
- Volatility persistence (α + β)

---

### 3️⃣ Machine Learning Models

- Random Forest (nonlinear benchmark)  
- Optional: XGBoost  

**Features**
- Lagged returns (1–24 hours, 168 hours)  
- Rolling volatility  
- Seasonal indicators  
- Spike labels  

**Validation**
- Rolling-window out-of-sample evaluation  

---

### 4️⃣ Evaluation Metrics

**Volatility Forecasting**
- RMSE  
- MAE  

**Spike Prediction**
- Precision  
- Recall  
- AUC  

Comparative assessment:
- Econometric vs ML performance  
- Seasonal sub-samples  
- Tail-event behaviour  

---

## 📈 Expected Findings

- Strong volatility clustering  
- Significant ARCH effects  
- High persistence in conditional variance  
- Asymmetry (EGARCH/GJR outperform symmetric GARCH)  
- Machine learning improves spike classification and extreme-event forecasting  
- Seasonal differences in model performance  

---

## 🔬 Robustness Checks

- Alternative spike definitions (top 1% price vs |returns| threshold)  
- Squared returns vs realised volatility  
- Sub-sample analysis (winter vs summer)  
- Alternative rolling window lengths  
- Price levels vs returns modelling  

---

## 🧰 Requirements

Python 3.10+

Suggested packages:

```
pandas
numpy
matplotlib
seaborn
statsmodels
arch
scikit-learn
xgboost
scipy
tqdm
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

1. Download raw CSV data into:

   ```
   data/raw/
   ```

2. Run data processing:

   ```bash
   python src/data_processing.py
   ```

3. Estimate econometric models:

   ```bash
   python src/econometric_models.py
   ```

4. Train machine learning models:

   ```bash
   python src/ml_models.py
   ```

5. Generate evaluation tables and figures:

   ```bash
   python src/evaluation.py
   ```

---

## 📌 Contribution

This project contributes by:

- Providing UK-specific high-frequency volatility evidence  
- Replicating established GARCH frameworks  
- Systematically benchmarking them against nonlinear machine-learning models  
- Evaluating volatility persistence and spike predictability  
- Assessing implications for market risk and hedging costs  

---

## ⚠️ Limitations

- Price-only analysis (no structural drivers included)  
- Not a causal study  
- Market events within 2015–2019 not explicitly modelled  
- Machine-learning interpretability vs accuracy trade-off  

---

## 📜 Academic Integrity

AI tools were used only for minor phrasing refinement. All modelling design, empirical analysis, coding, interpretation, and literature synthesis were conducted independently.

---

## 📄 License

For academic use only.

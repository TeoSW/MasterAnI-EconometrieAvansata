# 🔍 Economic Analysis of Fraud Risk in Financial Transactions

> Academic project for the course **Advanced Econometrics**  
> Bucharest University of Economic Studies (ASE) | Faculty of Cybernetics, Statistics and Economic Informatics  
> Authors: **Constantin Teodor-Vasile**, **Nițu Vlad-Cristian**

---

## 📌 Description

This project applies **advanced econometrics** and **machine learning** techniques to detect fraud risk in retail financial transactions. Three model specifications are built and compared — Logistic Regression (clean model), Logistic Regression (reduced model), and Random Forest — on a dataset of ~100,000 transactions, with a focus on econometric inference correctness and avoiding data leakage.

📦 **Data source:** [Kaggle – Retail Intelligence Fraud Detection Dataset](https://www.kaggle.com/datasets/noopurbhatt/retail-intelligence-fraud-detection-dataset/data)

---

## 🎯 Objectives

- Exploring and preprocessing a financial transactions dataset
- Building a correctly specified Logistic Regression model without data leakage
- Selecting statistically significant variables (p < 0.1) for a parsimonious model
- Evaluating an alternative model including risk flags (for comparison purposes)
- Building a Random Forest model and comparing predictive performance
- Diagnosing models (heteroskedasticity, autocorrelation, normality, VIF, calibration)

---

## 🗂️ Project Structure

```
📁 fraud-detection-econometrics/
├── 📓 main.ipynb     # Main notebook with the full analysis
├── 📊 db.csv         # Dataset (downloaded from Kaggle, link below)
└── 📄 README.md
```

---

## 🧮 Analysis Pipeline

### 1. Data loading & inspection
CSV reading with `parse_dates`, shape inspection, data types and missing values.

### 2. Preprocessing
- Drop irrelevant columns (IDs, timestamp)
- Binary variable encoding as `int`
- Descriptive statistics: continuous, binary and categorical variables

### 3. Exploratory Data Analysis (EDA)
Four charts generated and saved as `plot_relevante.png`:
- Boxplot: transaction amount vs. fraud
- Fraud rate per behavioral flag
- KDE: 24h transaction frequency by class
- Heatmap: correlation matrix

### 4–5. Logistic Regression — Clean Model
Exclusion of leakage-prone variables (`fraud_risk`, risk flags), one-hot encoding of categorical variables, estimation with `statsmodels.Logit`, coefficient table with significance stars.

### 6. Reduced Model (variables with p < 0.1)
Re-estimation with a subset of significant predictors — a more parsimonious specification with similar performance.

### 7. Error Analysis
Classification by type (TP, TN, FP, FN) at threshold 0.50, mean analysis per error type, most "confident" false positives and false negatives.

### 8. Alternative Model with Risk Flags
Logistic Regression retaining behavioral flags, presented as a comparative analysis. Shows quasi-separation and very high pseudo R² — confirmed as leakage.

### 9. Random Forest
300 trees, `class_weight="balanced"`, stratified 80/20 split, evaluation with ROC-AUC, Average Precision, Classification Report and feature importance.

### 10. Econometric Diagnostics
- **Breusch-Pagan** – heteroskedasticity on residuals
- **Durbin-Watson** + **Ljung-Box** – autocorrelation
- **Jarque-Bera** + **Shapiro-Wilk** – residual normality
- **VIF** – multicollinearity
- **Condition Number** – numerical stability
- **Hosmer-Lemeshow** – calibration of estimated probabilities

---

## 📊 Main Results

| Model | ROC-AUC | Avg. Precision | Notes |
|---|---|---|---|
| Clean Logit (all vars.) | ~0.76 | — | McFadden R² ≈ 0.18 |
| Reduced Logit (p < 0.1) | ~0.76 | — | Parsimonious specification |
| Logit with risk flags | ≫ 0.90 | — | ⚠️ Quasi-separation / leakage |
| **Random Forest** | **~0.812** | **~0.803** | Fraud recall ≈ 72.3% |

**Top predictors (Random Forest):** `transaction_frequency_24h`, `failed_transaction_count_24h`, `transaction_amount`, `avg_transaction_amount_7d`

---

## ⚙️ Installation & Usage

### Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy statsmodels
```

### Run

```bash
jupyter notebook main.ipynb
```

> ⚠️ Make sure `db.csv` is in the same directory as `main.ipynb`.  
> Dataset available on [Kaggle](https://www.kaggle.com/datasets/noopurbhatt/retail-intelligence-fraud-detection-dataset/data).

---

## 🛠️ Technologies

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-yellow?logo=scikit-learn)
![statsmodels](https://img.shields.io/badge/statsmodels-Econometrics-green)
![pandas](https://img.shields.io/badge/pandas-Data-purple?logo=pandas)

| Library | Usage |
|---|---|
| `pandas` / `numpy` | Data manipulation and preprocessing |
| `matplotlib` / `seaborn` | EDA visualizations |
| `statsmodels` | Logistic Regression, econometric diagnostics |
| `scikit-learn` | Random Forest, evaluation metrics, data splitting |
| `scipy` | KDE, statistical tests (Jarque-Bera, Shapiro-Wilk) |

---

## 📝 Conclusions

The clean Logit model is recommended for **econometric inference**, as it avoids leakage and perfect separation, and its coefficients are stable and interpretable. The model with risk flags demonstrates that these are quasi-deterministic with respect to `fraud_flag` and should not be used as independent predictors.

Random Forest delivers the best **predictive performance** (ROC-AUC ≈ 0.812) and confirms that behavioral transaction variables are more informative than categorical ones (payment method, location, merchant category).

Econometric diagnostics reveal no serious autocorrelation or multicollinearity issues. Breusch-Pagan and normality tests are significant, but these are OLS-specific tests and do not represent fundamental assumptions of logistic regression.

---

*Academic project | Bucharest University of Economic Studies (ASE), 2026*

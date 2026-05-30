# 🔍 Analiza Economică a Riscului de Fraudă în Tranzacții Financiare
 
> Proiect academic realizat în cadrul cursului **Econometrie Avansată**  
> Academia de Studii Economice din București | Facultatea de Cibernetică, Statistică și Informatică Economică  
> Studenți: **Constantin Teodor-Vasile**, **Nițu Vlad-Cristian**
 
---
 
## 📌 Descriere
 
Acest proiect aplică tehnici de **econometrie avansată** și **machine learning** pentru detectarea riscului de fraudă în tranzacții financiare de retail. Sunt construite și comparate trei specificații de model — Regresie Logistică (model curat), Regresie Logistică (model redus) și Random Forest — pe un dataset de ~100.000 de tranzacții, cu accent pe corectitudinea inferenței econometrice și evitarea data leakage-ului.
 
📦 **Sursă date:** [Kaggle – Retail Intelligence Fraud Detection Dataset](https://www.kaggle.com/datasets/noopurbhatt/retail-intelligence-fraud-detection-dataset/data)
 
---
 
## 🎯 Obiective
 
- Explorarea și preprocesarea datelor dintr-un dataset de tranzacții financiare
- Construirea unui model de Regresie Logistică corect specificat, fără data leakage
- Selectarea variabilelor semnificative statistic (p < 0.1) pentru un model parsimonios
- Evaluarea unui model alternativ care include flag-uri de risc (pentru comparație)
- Construirea unui model Random Forest și compararea performanței predictive
- Diagnosticarea modelelor (heteroskedasticitate, autocorelație, normalitate, VIF, calibrare)
---
 
## 🗂️ Structura Proiectului
 
```
📁 fraud-detection-econometrie/
├── 📓 main.ipynb     # Notebook principal cu toată analiza
├── 📊 db.csv         # Dataset (descărcat de pe Kaggle, link mai jos)
└── 📄 README.md
```
 
---
 
## 🧮 Pipeline de Analiză
 
### 1. Încărcarea și inspecția datelor
Citire CSV cu `parse_dates`, inspecție de formă, tipuri de date și valori lipsă.
 
### 2. Preprocesare
- Drop coloane irelevante (ID-uri, timestamp)
- Encodare variabile binare ca `int`
- Statistici descriptive: variabile continue, binare și categoriale
### 3. Analiză exploratorie (EDA)
Patru grafice generate și salvate în `plot_relevante.png`:
- Boxplot: suma tranzacției vs. fraud
- Rata de fraudă per flag comportamental
- KDE: frecvența tranzacțiilor 24h pe clase
- Heatmap: matrice de corelații
### 4–5. Regresie Logistică — Model Curat
Excludere variabile cu potențial de leakage (`fraud_risk`, flag-uri de risc), one-hot encoding variabile categoriale, estimare cu `statsmodels.Logit`, tabel de coeficienți cu stele de semnificație.
 
### 6. Model Redus (variabile cu p < 0.1)
Re-estimare cu subset de predictori semnificativi — specificație mai parsimonioasă cu performanță similară.
 
### 7. Error Analysis
Clasificare pe tipuri (TP, TN, FP, FN) la threshold 0.50, analiza mediilor per tip de eroare, cele mai „sigure" false positives și false negatives.
 
### 8. Model Alternativ cu Flag-uri de Risc
Regresie Logistică care păstrează flag-urile comportamentale, prezentată ca analiză comparativă. Prezintă quasi-separation și pseudo R² foarte ridicat — confirmat ca leakage.
 
### 9. Random Forest
300 de arbori, `class_weight="balanced"`, split 80/20 stratificat, evaluare cu ROC-AUC, Average Precision, Classification Report și importanța variabilelor.
 
### 10. Diagnostice Econometrice
- **Breusch-Pagan** – heteroskedasticitate pe reziduuri
- **Durbin-Watson** + **Ljung-Box** – autocorelație
- **Jarque-Bera** + **Shapiro-Wilk** – normalitatea reziduurilor
- **VIF** – multicoliniaritate
- **Condition Number** – stabilitate numerică
- **Hosmer-Lemeshow** – calibrarea probabilităților estimate
---
 
## 📊 Rezultate Principale
 
| Model | ROC-AUC | Avg. Precision | Note |
|---|---|---|---|
| Logit curat (toate var.) | ~0.76 | — | McFadden R² ≈ 0.18 |
| Logit redus (p < 0.1) | ~0.76 | — | Specificație parsimonioasă |
| Logit cu flag-uri | ≫ 0.90 | — | ⚠️ Quasi-separation / leakage |
| **Random Forest** | **~0.812** | **~0.803** | Recall fraudă ≈ 72.3% |
 
**Top predictori (Random Forest):** `transaction_frequency_24h`, `failed_transaction_count_24h`, `transaction_amount`, `avg_transaction_amount_7d`
 
---
 
## ⚙️ Instalare & Rulare
 
### Cerințe
 
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy statsmodels
```
 
### Rulare
 
```bash
jupyter notebook main.ipynb
```
 
> ⚠️ Asigurați-vă că `db.csv` se află în același director cu `main.ipynb`.  
> Dataset disponibil pe [Kaggle](https://www.kaggle.com/datasets/noopurbhatt/retail-intelligence-fraud-detection-dataset/data).
 
---
 
## 🛠️ Tehnologii utilizate
 
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-yellow?logo=scikit-learn)
![statsmodels](https://img.shields.io/badge/statsmodels-Econometrics-green)
![pandas](https://img.shields.io/badge/pandas-Data-purple?logo=pandas)
 
| Librărie | Utilizare |
|---|---|
| `pandas` / `numpy` | Manipulare și preprocesare date |
| `matplotlib` / `seaborn` | Vizualizări EDA |
| `statsmodels` | Regresie Logistică, diagnostice econometrice |
| `scikit-learn` | Random Forest, metrici de evaluare, split date |
| `scipy` | KDE, teste statistice (Jarque-Bera, Shapiro-Wilk) |
 
---
 
## 📝 Concluzii
 
Modelul Logit curat este recomandat pentru **inferență econometrică**, deoarece evită leakage-ul și separarea perfectă, iar coeficienții sunt stabili și interpretabili. Modelul cu flag-uri de risc demonstrează că acestea sunt cvasi-deterministe față de `fraud_flag` și nu trebuie utilizate ca predictori independenți.
 
Random Forest oferă cea mai bună **performanță predictivă** (ROC-AUC ≈ 0.812) și confirmă că variabilele comportamentale ale tranzacției sunt mai informative decât cele categoriale (metodă de plată, locație, categorie comerciant).
 
Diagnosticele econometrice nu relevă probleme de autocorelație sau multicoliniaritate serioasă. Testele Breusch-Pagan și de normalitate sunt semnificative, dar acestea sunt teste specifice OLS și nu reprezintă ipoteze fundamentale ale regresiei logistice.
 
---
 
*Proiect academic | ASE București, 2026*

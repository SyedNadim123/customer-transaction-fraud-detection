# Detecting Fraudulent Customer Transactions

> **MSc Data Science Dissertation — University of Sussex (2024–2025)**  
> 🏆 Highest Mark in Cohort — 83%

---

## 📌 Project Overview

This project presents an end-to-end machine learning pipeline for detecting fraudulent credit card transactions. The dataset contains **284,807 real-world anonymised transactions** from European cardholders, of which only **492 (0.172%)** are fraudulent — making class imbalance one of the core challenges addressed in this work.

The pipeline covers everything from exploratory data analysis and class rebalancing to model training, hyperparameter tuning, threshold optimisation, SHAP-based interpretability, and probability calibration.

---

## 🎯 Problem Statement

How can a machine learning pipeline be designed to achieve high predictive accuracy, maintain interpretability, handle severe class imbalance, and adapt to evolving fraud patterns in real-world financial environments?

---

## 📊 Dataset

| Attribute | Detail |

| Source | European credit card transactions (September 2013) |
| Total Records | 284,807 transactions |
| Features | 28 PCA-transformed (V1–V28) + Time + Amount |
| Target Variable | Class (0 = Legitimate, 1 = Fraudulent) |
| Fraud Count | 492 cases (~0.172%) |
| Imbalance Ratio | 1 Fraud : 577 Non-Fraud |

---

## 🛠️ Technologies & Libraries

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-green)
![LightGBM](https://img.shields.io/badge/LightGBM-Boosting-yellow)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-red)

**Languages:** Python
**ML Libraries:** Scikit-learn, XGBoost, LightGBM, Imbalanced-learn
**Explainability:** SHAP (SHapley Additive Explanations)
**Benchmarking:** LazyPredict
**Data Processing:** Pandas, NumPy
**Visualisation:** Matplotlib, Seaborn



##  Methodology

### 1. Exploratory Data Analysis (EDA)
Feature distribution analysis across fraud vs non-fraud classes
Class imbalance visualisation
Time-based and amount-based fraud pattern analysis
Pearson correlation heatmap for feature selection

### 2. Data Preprocessing
Stratified 80/20 train-test split
Z-score normalisation of the `Amount` feature
Removal of `Time` feature (low predictive value)

### 3. Class Rebalancing
| Method | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Random Undersampling | 0.94 | 0.75 | 0.84 | 0.79 | 0.91 |
| **SMOTE** | **0.97** | **0.91** | **0.93** | **0.92** | **0.97** |

✅ **SMOTE selected** as the final resampling method.

### 4. Models Evaluated
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM ✅ *(Final Model)*

### 5. Optimisation
- **GridSearchCV** hyperparameter tuning with 5-fold stratified cross-validation
- **Threshold tuning** to optimise F1-score (optimal threshold: 0.89)
- **Probability calibration** using Isotonic Regression

### 6. Interpretability
- **SHAP Global Summary Plot** — identifies top features: `V4`, `V14`, `V10`, `V12`, `V16`
- **SHAP Local Analysis** — transaction-level prediction explanations
- **Permutation Feature Importance** — model-agnostic feature validation

---

## 📈 Final Model Results (LightGBM + SMOTE)

| Metric | Score |
|---|---|
| Accuracy | 99.93% |
| Precision | 0.92 |
| Recall | 0.81 |
| F1-Score | 0.86 |
| ROC-AUC | 0.98 |

**Confusion Matrix (Threshold = 0.89):**
True Positives (Fraud Detected): 77
True Negatives (Legitimate Identified): 55,000
False Positives: 362
False Negatives (Missed Fraud): 8   


## 🧠 Key Findings

**SMOTE** significantly outperformed Random Undersampling across all metrics, especially recall
**LightGBM** delivered the best balance of accuracy, recall, interpretability, and efficiency
**Top predictive features:** `V4`, `V14`, `V10`, `V12` (confirmed by both SHAP and Permutation Importance)
**Threshold tuning** from 0.5 → 0.89 dramatically improved fraud capture rate
**Probability calibration** ensured predicted risk scores align with real-world outcomes

---

## 📂 Project Structure

```
customer-transaction-fraud-detection/
│
├── FINAL DISSERTATION 1 .ipynb   # Full end-to-end ML pipeline
└── README.md
```

---

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/SyedNadim123/customer-transaction-fraud-detection.git

# Install dependencies
pip install pandas numpy scikit-learn xgboost lightgbm imbalanced-learn shap lazypredict matplotlib seaborn

# Open the notebook
jupyter notebook "FINAL DISSERTATION 1 .ipynb"
```

> **Dataset:** Available on [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## 📚 Academic Context

- **Institution:** University of Sussex
- **Department:** Mathematics and Physical Sciences
- **Degree:** MSc Data Science
- **Academic Year:** 2024–2025
- **Supervisor:** Dr. Miroslav Chlebik
- **Grade:** 83% — Highest in Cohort



## 👤 Author

**Syed Nadimul Haque**  
MSc Data Science | ML & AI Engineer | Software Engineer  
🔗 [LinkedIn](https://www.linkedin.com/in/syed-nadimul-haque-691311307/)  
📧 nadimulhaq001@gmail.com



## Disclaimer

This project was developed for academic research purposes. The dataset used is publicly available and fully anonymised. No sensitive personal data was used or exposed.

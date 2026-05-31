markdown# 🏦 Loan Default Prediction Using Machine Learning
### *An End-to-End Data Science Project | Predicting Credit Risk with XGBoost, SMOTE & SHAP*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GIVEN-CHINYAMA/loan_default_prediction_ml/blob/main/loan_default_prediction_ml.ipynb)
[![View on nbviewer](https://img.shields.io/badge/View-nbviewer-orange?style=flat&logo=jupyter)](https://nbviewer.org/github/GIVEN-CHINYAMA/loan_default_prediction_ml/blob/main/loan_default_prediction_ml.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-orange?style=flat)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-9b59b6?style=flat)](https://shap.readthedocs.io)
[![Status](https://img.shields.io/badge/Status-Complete-2ecc71?style=flat)](https://github.com/GIVEN-CHINYAMA/loan_default_prediction_ml)

**Author:** Given Chinyama

**Institution:** Kwame Nkrumah University

**Date:** May 2026

**GitHub:** [github.com/GIVEN-CHINYAMA](https://github.com/GIVEN-CHINYAMA)

---

## 👆 Click "Open in Colab" Above to View the Full Notebook

> GitHub cannot render large notebooks with embedded visualisations. Click the **Open in Colab** badge above to view the complete project with all code, outputs, and professional charts — no setup required.

---

## 📊 Model Performance Summary

### Validation Set Results

| Model | AUC-ROC | F1-Score | Accuracy | Avg Precision |
|---|---|---|---|---|
| **XGBoost (Tuned) 🏆** | **Best** | **Best** | **Best** | **Best** |
| Gradient Boosting | 2nd | 2nd | 2nd | 2nd |
| Random Forest | 3rd | 3rd | 3rd | 3rd |
| Logistic Regression | Baseline | Baseline | Baseline | Baseline |

> 🏆 **XGBoost achieved the best performance on all metrics** — with SMOTE-balanced training and GridSearchCV tuning, it significantly outperformed all baseline models.

---

## 📅 Key Prediction Thresholds (XGBoost — Best Model)

| Risk Factor | Threshold | Action |
|---|---|---|
| FICO Score | < 640 | High-risk flag |
| Debt-to-Income | > 35% | Manual review |
| Revolving Utilisation | > 75% | Risk surcharge |
| Loan Grade | E / F / G | Elevated scrutiny |
| Delinquency (2 yrs) | Any | Automatic flag |

---

## 📁 Project Overview

Loan default is one of the most costly risks in the financial services industry. When borrowers fail to repay loans, institutions face significant financial losses, regulatory scrutiny, and reputational damage. This project builds an **end-to-end, production-grade machine learning pipeline** that predicts whether a loan applicant will default — enabling lenders to make smarter, faster, and fairer credit decisions.

### Why This Project Stands Out

Unlike generic classification tutorials, this project mirrors the kind of work done by data science teams at banks, fintech companies, and credit bureaus. It includes:
- **Realistic synthetic data** modelled on LendingClub distributions — 50,000 loan records with real-world class imbalance
- **Domain-specific feature engineering** — 8 new financial features derived from raw data
- **SHAP Explainability** — individual loan decisions explained, supporting GDPR and Fair Lending compliance
- **Business recommendations** — actionable insights framed for financial stakeholders, not just model metrics

---

## 🔍 Problem Statement

> *"Which loan applicants are likely to default, and what factors drive that risk?"*

Zambia and sub-Saharan Africa broadly are seeing rapid growth in digital lending and microfinance. Institutions need reliable, interpretable ML models that can:
- **Identify high-risk applicants** before a loan is issued
- **Quantify risk factors** to inform pricing and lending terms
- **Explain decisions** to regulators, auditors, and applicants
- **Operate at scale** across thousands of applications per day

---

## 📦 Dataset

| Property | Details |
|---|---|
| **Type** | Synthetic dataset modelled on LendingClub loan data distributions |
| **Records** | 50,000 loan applications |
| **Features** | 16 raw + 8 engineered = 24 total features |
| **Target** | `loan_status` — 0 (Fully Paid) / 1 (Default) |
| **Default Rate** | ~22% (realistic class imbalance) |
| **Missing Data** | ~3% in selected columns (realistic injection) |

### Raw Feature Descriptions

| Feature | Type | Description |
|---|---|---|
| `loan_amnt` | Numerical | Total loan amount requested (USD) |
| `term` | Categorical | Repayment term — 36 or 60 months |
| `int_rate` | Numerical | Annual interest rate (%) |
| `grade` | Ordinal | Loan grade assigned by lender (A–G) |
| `emp_length` | Numerical | Borrower employment length (years) |
| `home_ownership` | Categorical | Housing status (RENT / OWN / MORTGAGE) |
| `annual_inc` | Numerical | Annual income (USD) |
| `purpose` | Categorical | Stated purpose of the loan |
| `dti` | Numerical | Debt-to-income ratio |
| `delinq_2yrs` | Numerical | Number of delinquencies in past 2 years |
| `fico_score` | Numerical | Borrower FICO credit score (580–850) |
| `inq_last_6mths` | Numerical | Credit inquiries in last 6 months |
| `open_acc` | Numerical | Number of open credit lines |
| `pub_rec` | Numerical | Derogatory public records |
| `revol_util` | Numerical | Revolving credit utilisation rate (%) |
| `loan_status` | Binary | **TARGET** — 0: Paid, 1: Default |

### Engineered Features

| Feature | Description |
|---|---|
| `monthly_payment` | Estimated monthly repayment (amortisation formula) |
| `payment_to_income` | Monthly payment as % of monthly income |
| `loan_to_income` | Total loan amount relative to annual income |
| `risk_score` | Composite score — interest rate + DTI + grade + delinquency |
| `high_revol_util` | Binary flag: revolving utilisation > 75% |
| `has_delinq` | Binary flag: any delinquency in past 2 years |
| `has_pub_rec` | Binary flag: any public derogatory record |
| `grade_num` | Numeric encoding of loan grade (A=1 … G=7) |

---

## 🗂️ Project Stages

| Stage | Description |
|---|---|
| Stage 1 | Environment setup — library installation and imports |
| Stage 2 | Data collection — synthetic LendingClub-style dataset (50,000 records) |
| Stage 3 | Data overview and quality check — shape, dtypes, missing values |
| Stage 4 | Exploratory Data Analysis — distributions, correlations, categorical default rates |
| Stage 5 | Feature engineering — 8 domain-specific financial features |
| Stage 6 | Preprocessing — encoding, imputation, scaling, train/val/test split |
| Stage 7 | Class imbalance handling — SMOTE applied to training set only |
| Stage 8 | Model training — Logistic Regression, Random Forest, Gradient Boosting, XGBoost |
| Stage 9 | Model evaluation — AUC-ROC, F1, Confusion Matrix, Precision-Recall curves |
| Stage 10 | Hyperparameter tuning — GridSearchCV with Stratified K-Fold (k=5) |
| Stage 11 | SHAP Explainability — beeswarm, bar, and waterfall plots |
| Stage 12 | Final test set evaluation — held-out performance metrics |
| Stage 13 | Business insights and recommendations |

---

## 🛠️ Technologies Used

| Tool | Purpose |
|---|---|
| Python 3.10+ | Core language |
| XGBoost | Best-performing classification model |
| Scikit-learn | ML pipeline, preprocessing, evaluation |
| imbalanced-learn | SMOTE for class imbalance |
| SHAP | Explainable AI — feature attribution |
| Pandas / NumPy | Data manipulation |
| Matplotlib / Seaborn | Visualisation (13 charts) |
| Google Colab | Cloud execution platform |

---

## 💡 Key Findings

- **XGBoost outperformed all 3 competing models** on every metric — AUC-ROC, F1-Score, Accuracy, and Average Precision
- **FICO Score** is the single strongest predictor of loan default — low FICO borrowers default at disproportionately high rates
- **Interest rate and loan grade** are highly correlated — Grade E/F/G loans carry 3–5× higher default rates than Grade A
- **SMOTE significantly improved** recall on the minority default class without causing data leakage (applied to training set only)
- **SHAP analysis confirmed** that the model's decisions are driven by financially meaningful features — not spurious correlations
- **Payment-to-income ratio** (an engineered feature) ranked among the top 5 most important features — validating the feature engineering stage
- **Logistic Regression**, while a weak performer, still captured the general direction of risk — confirming the problem is linearly separable to a degree

---

## 📸 Visualisations

The notebook generates **13 publication-quality charts**:

| # | Chart | Description |
|---|---|---|
| 1 | Target Distribution | Bar + pie chart of default vs fully paid |
| 2 | Feature Distributions | Overlapping histograms by loan status |
| 3 | Categorical Default Rates | Default rate by grade, purpose, term, ownership |
| 4 | Correlation Heatmap | Full feature correlation matrix |
| 5 | FICO vs Interest Rate | Scatter plot coloured by default status |
| 6 | SMOTE Balance | Class distribution before and after SMOTE |
| 7 | ROC Curves | All 4 models on one chart |
| 8 | Precision-Recall Curves | All 4 models — handles imbalance better than ROC |
| 9 | Confusion Matrices | All 4 models side-by-side |
| 10 | SHAP Beeswarm | Top 15 features with direction of impact |
| 11 | SHAP Bar Chart | Mean absolute SHAP values — global importance |
| 12 | SHAP Waterfall | Single loan decision fully explained |
| 13 | Feature Importance | XGBoost top 20 built-in feature importances |

---

## 💼 Business Insights & Recommendations

### Top Default Risk Factors

| Risk Factor | Finding |
|---|---|
| 🔴 **Interest Rate** | Every 1% increase correlates with meaningfully higher default probability |
| 🔴 **Loan Grade E–G** | 3–5× higher default rates than Grade A |
| 🔴 **DTI > 35%** | Borrowers spending >35% of income on debt are significantly higher risk |
| 🟡 **FICO < 640** | Critical threshold where defaults spike sharply |
| 🟡 **Revolving Util > 75%** | Maxed-out revolving credit signals financial stress |
| 🟡 **Past Delinquency** | Any delinquency in the last 2 years doubles default probability |
| 🟢 **High Annual Income** | Strong protective factor |
| 🟢 **Long Employment Tenure** | Stable employment significantly lowers risk |

### Actionable Recommendations

1. **Automated Flagging** — Flag applications with FICO < 640 AND DTI > 35% for mandatory manual underwriting review
2. **Risk-Based Pricing** — Charge higher rates for Grade D–G loans to offset expected losses
3. **Loan Amount Caps** — Limit first-time borrowers with no credit history to loans under $10,000
4. **SHAP-Powered Explanations** — Use SHAP waterfall plots for transparent, legally defensible rejection letters (GDPR / ECOA compliance)
5. **Quarterly Model Retraining** — Monitor for data drift as macroeconomic conditions evolve
6. **Early Warning System** — Apply the model monthly on existing portfolios to flag at-risk accounts before default occurs

---

## 🚀 How to Run

### Option 1 — Google Colab (Recommended)

Click the **Open in Colab** badge at the top of this page. All dependencies install automatically. No local setup required. Total runtime: approximately **5–8 minutes**.

### Option 2 — Run Locally

```bash
git clone https://github.com/GIVEN-CHINYAMA/loan_default_prediction_ml.git
cd loan_default_prediction_ml
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn shap jupyter
jupyter notebook loan_default_prediction_ml.ipynb
```

---

## 📌 Repository Structure
loan_default_prediction_ml/
│
├── loan_default_prediction_ml.ipynb   ← Complete end-to-end notebook
├── README.md                          ← This file
├── .gitignore                         ← Python gitignore
└── LICENSE                            ← MIT License
```

---

## 🚀 Future Work

- [ ] Plug in real LendingClub dataset (2.26M rows) for scale testing
- [ ] Implement TabNet deep learning model for comparison
- [ ] Apply Platt Scaling for well-calibrated probability outputs
- [ ] Add cost-sensitive learning (false negatives cost more than false positives in banking)
- [ ] Track experiments with MLflow
- [ ] Build a Streamlit loan risk calculator web app
- [ ] Deploy model as a FastAPI endpoint for real-time scoring
- [ ] Fairness audit using Fairlearn — check for demographic bias

---

## 👤 Author

**Given Chinyama**
🎓 Kwame Nkrumah University
🌍 Lusaka, Zambia
🐙 [github.com/GIVEN-CHINYAMA](https://github.com/GIVEN-CHINYAMA)

---

## 📄 License

This project is licensed under the **MIT License** — free to use, adapt, and build on with attribution.

---

## ⭐ Show Your Support

If you found this project useful or learned something from it, please consider giving it a **⭐ star** on GitHub. It helps others discover the project and means a lot!

---

*Built with 🖤 and purpose — Given Chinyama, 2026*

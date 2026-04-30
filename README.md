# 💳 Credit Risk Analysis — Probability of Default Modelling

## 📌 Project Overview
This project builds a full credit risk analysis pipeline on a financial dataset using both MySQL and Python. It covers data preprocessing, SQL-based feature engineering, advanced EDA, custom probability-of-default (PD) modelling, and borrower risk tier segmentation — closely mirroring real-world credit underwriting workflows used by banks and NBFCs.

---

## 🎯 Business Problem
Financial institutions need to assess the likelihood that a borrower will default on a loan before approving credit. Manual rule-based systems miss complex risk signals. This project builds a data-driven risk scoring system that segments borrowers into Prime, Near-Prime, and Subprime tiers — enabling smarter lending decisions.

**Key Questions Answered:**
- Which financial behaviours are the strongest predictors of default?
- How can we build a credit scorecard without a pre-labelled target variable?
- What is the default probability distribution across different borrower segments?
- Which features should be retained for a production-grade risk model?

---

## 📂 Project Structure
```
credit-risk-analysis/
│
├── datasets/
│   ├── risk_analysis_dataset.csv
│   ├── kingametric_credit_risk.csv
│   ├── kingametric_lean_v2.csv
│   └── final_risk_analysis_dataset.csv
│
├── visualizations/
│   └── GB_selected_feature_importances.png
│
├── sql/
│   ├── sql_analysis.ipynb            ← Core SQL analysis queries
│   ├── adv_sql_analysis_review.ipynb ← Advanced SQL + window functions
│   └── sql_feature_engineering.ipynb ← SQL feature engineering in MySQL
│
├── Preprocess.ipynb    ← Data cleaning & preparation
├── EDA.ipynb           ← Feature selection & encoding
├── Model.ipynb         ← PD model & borrower tier segmentation
└── README.md
```

---

## 📊 Dataset
- **Source:** Kaggle — Credit Risk / King-A-Metrics Dataset
- **Raw Features:** 45 columns including income, debt, payment behaviour, credit history
- **Final Features After Selection:** 20 (selected via Gradient Boosting feature importance)
- **Target Variable:** `Default_Flag` (1 = Default, 0 = No Default) — engineered via custom PD formula

---

## 🔧 Tools & Technologies
| Category | Tools |
|---|---|
| Language | Python 3.14, SQL (MySQL) |
| Data Analysis | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, Gradient Boosting |
| Encoding | Category Encoders (Target Encoding) |
| Database | MySQL (via ipython-sql + PyMySQL) |
| IDE | VS Code + Jupyter Notebook |

---

## 🗄️ SQL Analysis Highlights
40+ queries executed across 3 SQL notebooks:

| Query Type | What It Analysed |
|---|---|
| CTEs + CASE | Debt-to-Income buckets (Low / Medium / High) |
| Window Functions | RANK, LAG, PERCENT_RANK, Rolling Average on delayed payments |
| Aggregations | EMI burden ratio by Credit Mix, payment behaviour frequency |
| ALTER + UPDATE | 12 normalized features added directly to MySQL table |
| Multi-condition CASE | Credit risk categorization (High Risk / Low-Medium Risk) |
| Composite Scoring | Credit Score Proxy using weighted income, history, behaviour |

---

## ⚙️ Feature Engineering
**12 normalized features engineered in MySQL:**

| Feature | Formula |
|---|---|
| `normalized_dti` | Outstanding Debt / Annual Income (capped 0-1) |
| `normalized_emi` | Total EMI / Monthly Salary (capped 0-1) |
| `normalized_delinquency` | Delayed Payments / Num Loans (capped 0-1) |
| `normalized_credit_history` | Credit History Age / 120 (capped 0-1) |
| `normalized_savings` | Monthly Balance / Monthly Salary (capped 0-1) |
| `normalized_utilization` | Credit Utilization Ratio (capped 0-1) |
| `behavioral_risk_indicator` | Binary flag — pays minimum amount only |
| `credit_mix_quality` | Ordinal encoding: Good=2, Standard=1, Bad=0 |

**7 composite features engineered in Python:**

| Feature | Formula |
|---|---|
| `Debt_Stress` | normalized_dti × normalized_utilization |
| `Repayment_Stress` | normalized_emi × normalized_delinquency |
| `Credit_Exposure` | Num Credit Cards × Utilization Ratio |
| `Financial_Stress_Index` | Weighted sum of DTI, utilization, delinquency |
| `Behavioral_Risk_Composite` | Combined payment behaviour indicators |
| `Net_Cash_Flow` | Income − EMI − Investment |
| `Obligation_Ratio` | Total obligations / Monthly income |

---

## 🤖 Probability of Default Model

**Custom 8-factor weighted risk score formula:**
```
Risk Score =
  0.22 × (Outstanding_Debt / Annual_Income)
+ 0.18 × (Total_EMI / Monthly_Salary)
+ 0.18 × (Delayed_Payments / Num_Loans)
+ 0.12 × (Credit_Utilization / 100)
+ 0.10 × (Num_Credit_Inquiries / 10)
+ 0.10 × (1 − Credit_History_Age / 120)
+ 0.05 × (Delay_from_due_date / 30)
+ 0.05 × (1 − Monthly_Balance / Monthly_Salary)
```

Probability of default derived via sigmoid transformation:
```python
prob_default = 1 / (1 + exp(−z))
```

**Borrower Tier Segmentation:**
| Tier | Default Rate | Description |
|---|---|---|
| Prime | ~10% | Low risk — strong financials |
| Near-Prime | ~25% | Moderate risk — needs monitoring |
| Subprime | ~55% | High risk — credit intervention needed |

---

## 🔍 Feature Selection
- Used **Gradient Boosting Classifier** with `SelectFromModel`
- **Target Encoding** applied to categorical variables (smoothing=10)
- Reduced 51 features → **top 20 most predictive features**
- Bottom 20 noise features removed before final model training

---

## 💡 Business Recommendations
1. **Decline or price higher** for Subprime borrowers (55% default rate)
2. **Monitor Near-Prime borrowers** monthly — small changes push them to Subprime
3. **Flag accounts** where `normalized_delinquency > 0.5` AND `normalized_dti > 0.4` for early intervention
4. **Use Credit Score Proxy** as a lightweight pre-screening tool before full underwriting
5. **Prioritize customers** with `Payment_of_Min_Amount = Yes` — strong default predictor

---

## 🚀 How to Run
```bash
# Clone the repo
git clone https://github.com/YOUR-USERNAME/credit-risk-analysis.git

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn category-encoders ipython-sql pymysql jupyter

# Open Jupyter
python -m notebook
```
Run notebooks in this order:
1. `Preprocess.ipynb` — data cleaning
2. SQL notebooks — feature engineering in MySQL
3. `EDA.ipynb` — feature selection
4. `Model.ipynb` — PD model and tier segmentation

---

## 👤 Author
**Shivakumar V K**
- 📧 shivakumarkadagad25@gmail.com
- 🔗 [LinkedIn](#)
- 💻 [GitHub](#)

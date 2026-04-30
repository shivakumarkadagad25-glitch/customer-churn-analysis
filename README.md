# 🛒 E-Commerce Customer Churn Analysis & Prediction

## 📌 Project Overview
This project performs end-to-end customer churn analysis on an e-commerce dataset of 2,000 customers. The goal is to identify customers at risk of churning, understand the key behavioural drivers behind churn, and build a predictive model to support targeted retention strategies.

---

## 🎯 Business Problem
E-commerce businesses lose significant revenue when customers stop purchasing. Early identification of at-risk customers allows the business to intervene with personalized offers, re-engagement campaigns, and loyalty incentives — reducing churn before it happens.

**Key Questions Answered:**
- What is the overall churn rate and how are customers distributed across subscription statuses?
- Which customer segments (age, category, country, gender) churn the most?
- What behavioural features (recency, frequency, inactivity) most strongly predict churn?
- Can we build a model to predict which customers will churn next?

---

## 📂 Project Structure
```
customer-churn-analysis/
│
├── data/
│   └── cleaned_ecommerce_churn_dataset.csv
│
├── charts/
│   ├── churn_distribution.png
│   ├── churn_by_age.png
│   ├── churn_by_category.png
│   ├── churn_by_country.png
│   ├── churn_by_gender.png
│   ├── recency_vs_churn.png
│   ├── correlation_heatmap.png
│   ├── rfm_segment_analysis.png
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── roc_curve.png
│
├── Customer_churn_analysis.ipynb   ← Data prep & feature engineering
├── EDA.ipynb                       ← Exploratory data analysis
├── Model.ipynb                     ← ML model training & evaluation
└── README.md
```

---

## 📊 Dataset
- **Source:** Kaggle — E-Commerce Customer Churn Dataset
- **Size:** 2,000 records × 17 features
- **Target Variable:** `churn` (1 = Churned, 0 = Active)
- **Missing Values:** 0

---

## 🔧 Tools & Technologies
| Category | Tools |
|---|---|
| Language | Python 3.14 |
| Data Analysis | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Database | MySQL (via SQLAlchemy) |
| IDE | VS Code + Jupyter Notebook |

---

## ⚙️ Feature Engineering
8 new features engineered from raw data:

| Feature | Description |
|---|---|
| `recency_days` | Days since last purchase |
| `frequency` | Total number of purchases |
| `monetary` | Unit price × quantity |
| `tenure_days` | Days since customer signup |
| `avg_order_value` | Monetary / Frequency |
| `purchase_intensity` | Orders per tenure day |
| `inactive_ratio` | Recency / Tenure |
| `engagement_score` | Weighted combination of frequency, recency, intensity |

**RFM Segmentation** — Customers segmented into 5 tiers:
| Segment | Description |
|---|---|
| Champions | Highest RFM score — loyal, active, high value |
| Loyal Customers | Regular buyers with strong engagement |
| Potential Loyalists | Moderate engagement — growth opportunity |
| At Risk | Declining activity — intervention needed |
| Lost | Inactive, high churn probability |

---

## 📈 Key EDA Findings
- Overall churn rate: **24.6%**
- 15.2% of customers are **Paused** — a soft-churn at-risk segment
- **Inactive ratio** and **recency days** showed the strongest correlation with churn
- Lost and At-Risk RFM segments had the highest churn rates
- Champions segment had near-zero churn despite being the smallest group

---

## 🤖 Model Results

Two models trained and compared:

| Model | ROC-AUC | Notes |
|---|---|---|
| Logistic Regression | > 0.75 | Interpretable, good for stakeholder communication |
| Random Forest | > 0.75 | Better recall on churned customers |

- **Class imbalance** handled using `class_weight='balanced'`
- **Stratified K-Fold** cross validation used for reliable evaluation
- **ROC-AUC** used as primary metric (preferred over accuracy for imbalanced data)

---

## 💡 Business Recommendations
1. **Target Lost + At-Risk segments** with re-engagement campaigns (discounts, loyalty rewards)
2. **Monitor Paused customers** — 303 customers (15.2%) are soft-churning and need immediate outreach
3. **Focus retention spend on high inactive_ratio customers** — strongest churn predictor
4. **Protect Champions** — small segment but highest lifetime value, prioritize VIP treatment
5. **Use model scores weekly** to flag newly at-risk customers before they fully churn

---

## 🚀 How to Run
```bash
# Clone the repo
git clone https://github.com/YOUR-USERNAME/customer-churn-analysis.git

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn jupyter notebook

# Open Jupyter
python -m notebook
```
Run notebooks in this order:
1. `Customer_churn_analysis.ipynb` — data prep
2. `EDA.ipynb` — analysis and charts
3. `Model.ipynb` — model training

---

## 👤 Author
**Shivakumar V K**
- 📧 shivakumarkadagad25@gmail.com
- 🔗 [LinkedIn](#)
- 💻 [GitHub](#)

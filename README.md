# Lead Conversion Prediction

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Status](https://img.shields.io/badge/Status-Complete-green)

> Predicting which leads will convert into paying customers using machine learning  and translating model performance into measurable business value.

---

## Project Overview

Aurnect Creativity's sales team was treating all leads equally spending the same time on cold leads as hot ones, with no way to prioritise who to contact first. This project builds a supervised machine learning model that predicts the probability of a lead converting into a paying customer, assigns every lead an actionable priority tier, and translates model output into measurable business value.

**The key finding:** The ML-optimal threshold (0.61, best F1 score) is not the same as the business-optimal threshold (0.35, best net profit). Lowering the decision threshold from 0.61 to 0.35 generates an additional **$14,275 in net profit** by catching 22 more real converters at an outreach cost far outweighed by the revenue gained.

---

## Live Demo

| Service | URL |
|---|---|
| API (Swagger Docs) | [https://lead-scoring-api-eig5.onrender.com/docs](https://lead-scoring-api-eig5.onrender.com/docs) |
| Streamlit Frontend | [https://lead-scoring-streamlit.onrender.com](https://lead-scoring-streamlit.onrender.com) |

> **Note:** Services are hosted on Render's free tier and may take 30–60 seconds to wake up after inactivity.

---

## Installation and Setup

### Codes and Resources Used

- **Editor:** Databricks (model training and evaluation), VS Code (API development)
- **Python Version:** 3.10

### Python Packages Used

- **General Purpose:** `os`, `joblib`
- **Data Manipulation:** `pandas`, `numpy`
- **Data Visualization:** `matplotlib`, `seaborn`
- **Machine Learning:** `scikit-learn`, `xgboost`

Install all dependencies with:

```bash
pip install pandas numpy scikit-learn xgboost joblib matplotlib seaborn
```

---

## Data

### Source Data

The dataset contains **2,000 leads** from Aurnect Creativity's CRM with the following feature categories:

| Category | Features |
|---|---|
| Numerical | `lead_age_days`, `email_opens`, `website_visits`, `content_downloads`, `ad_clicks`, `num_contacts`, `interactions_last_30_days` |
| Numerical (with nulls) | `days_since_last_contact`, `email_reply_rate`, `response_time_days` |
| Ordered Categorical | `company_size`, `funnel_stage`, `preferred_contact_time` |
| Unordered Categorical | `business_type`, `industry`, `lead_source`, `ad_platform` |
| Binary | `budget_indicated`, `pricing_page_visited` |
| Target | `converted` (1 = converted, 0 = did not convert) |

**Class distribution:** 75.5% non-converters vs 24.5% converters — an imbalanced dataset handled explicitly through model configuration.

### Data Acquisition

Data was sourced directly from Aurnect Creativity's internal CRM system.

### Data Preprocessing

All preprocessing was done **after** the train/test split to prevent data leakage — fit only on training data and applied to both sets:

| Step | Strategy | Reason |
|---|---|---|
| Numerical missing values | Median from X_train only | Robust to outliers, no leakage |
| Binary missing values | Fill with 0 | Missing likely means event did not occur |
| Categorical missing values | Fill with 'Unknown' | Preserves missing as a meaningful signal |
| Numerical scaling | RobustScaler | Resistant to outliers |
| Ordered categorical encoding | OrdinalEncoder (custom order) | Preserves meaningful order |
| Unordered categorical encoding | OneHotEncoder | No inherent order between categories |

All transformations were packaged into a `ColumnTransformer` inside a sklearn `Pipeline` to guarantee consistent preprocessing across training and scoring.

---

## Code Structure

```
lead_conversion_project/
│
├── notebooks/
│   ├── 01_eda.ipynb                  ← Exploratory data analysis
│   ├── 02_preprocessing.ipynb        ← Cleaning, imputation, encoding
│   ├── 03_model_training.ipynb       ← Training, evaluation, tuning
│   └── 04_scoring_output.ipynb       ← Scoring new leads, saving output
│
├── models/
│   └── lead_scoring_model.pkl        ← Saved model package
│
├── data/
│   ├── raw/                          ← Original unmodified dataset
│   └── processed/                    ← Test set after imputation
│
├── outputs/
│   └── scored_leads.csv              ← Final prioritised lead list
│
└── README.md
```

---

## Results and Evaluation

### Model Comparison

Three models were trained and evaluated on the held-out test set (400 leads):

| Model | Precision (Class 1) | Recall (Class 1) | F1 (Class 1) | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.57 | 0.63 | 0.60 | 0.8188 |
| Random Forest | 0.85 | 0.41 | 0.55 | 0.8166 |
| XGBoost | 0.62 | 0.47 | 0.53 | 0.7612 |

**Logistic Regression was selected** — highest recall on converters (0.63), highest ROC-AUC (0.8188), and best F1 on the minority class. Random Forest had the highest precision (0.85) but missed 59% of actual converters — too conservative for a lead conversion use case.

### Hyperparameter Tuning

`RandomizedSearchCV` with `StratifiedKFold(n_splits=5)` — 50 combinations tested, 250 total model fits.

Best parameters found:

```python
penalty='elasticnet', C=0.1, l1_ratio=0.3, solver='saga'
```

ROC-AUC improved from **0.8188 → 0.8295** on the test set after tuning.

### Threshold Optimisation

| | Default (0.50) | ML Optimal (0.61) | Business Optimal (0.35) |
|---|---|---|---|
| Precision | 0.58 | 0.74 | 0.44 |
| Recall | 0.66 | 0.59 | 0.82 |
| F1 | 0.62 | 0.66 | 0.57 |
| Converters caught | 65 / 98 | 58 / 98 | 80 / 98 |
| Net Profit | — | $30,140 | **$39,370** |

### Lead Prioritisation Tiers

| Tier | Probability | Conversion Rate | Leads | Action |
|---|---|---|---|---|
| High Priority | p ≥ 0.70 | 74.2% | 62 | Call immediately |
| Medium Priority | 0.50 ≤ p < 0.70 | 38.0% | 50 | Email nurture campaign |
| Low Priority | p < 0.50 | 11.5% | 288 | Deprioritize |

### Business Impact (at threshold 0.35)

| Metric | Value |
|---|---|
| Revenue captured | $44,800 |
| Net profit generated | $39,370 |
| ROI on outreach | 5.9× |
| Converters caught | 80 out of 98 (82%) |

*Based on estimated $560 average revenue per conversion and $30 cost per outreach for Aurnect Creativity.*

**Key insight:**

> *"The optimal threshold is not the one with the best F1 score, but the one that maximises business value."*

---

## Author

**Eniola Ajiboye**
- GitHub: [@Ajiboyeniola](https://github.com/Ajiboyeniola)
- Email: ajiboyeniola@gmail.com

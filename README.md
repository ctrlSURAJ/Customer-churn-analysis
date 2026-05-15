# Customer Churn Analysis

**Resume line:** Analyzed 7,000+ customer records using Python & SQL to identify churn drivers; built Logistic Regression model achieving 78% accuracy and 0.83 AUC score

## Overview
End-to-end analysis of telecom customer churn using SQL for exploration and Python for cleaning, visualization, and predictive modeling.

## Tools Used
- Python (pandas, seaborn, matplotlib, scikit-learn)
- MySQL (data loading, segmentation queries)
- Jupyter Notebook

## Key Findings
- 26.58% overall churn rate — 1 in 4 customers left
- Month-to-month customers churn at 43% vs only 3% for 2-year contract customers
- Churned customers pay $13 more per month on average ($74.44 vs $61.27)
- Fiber optic users churn at 42% vs 19% for DSL users
- Electronic check users churn at 45% — highest of any payment method
- Paperless billing customers churn at double the rate of non-paperless customers

## Model Performance
- Algorithm: Logistic Regression
- Accuracy: 78.75%
- AUC Score: 0.83
- Key churn drivers: Contract type, Phone Service, Paperless Billing, Senior Citizen status

## Business Recommendations
1. Target month-to-month customers with loyalty discounts at the 3-month mark
2. Promote long-term contracts — 2-year contract customers churn 14x less
3. Encourage auto-pay setup — reduces churn from 45% to 15-17%
4. Flag new customers (0-12 months) for proactive outreach
5. Review fiber optic pricing — 42% churn despite being a premium service

## Files
| File | Description |
|------|-------------|
| `churn_eda.py` | Data cleaning + 6 visualizations |
| `churn_model.py` | Logistic Regression + feature importance + ROC curve |
| `Churn_Analysis.ipynb` | Full Jupyter notebook with narrative |
| `sql_findings.txt` | Key SQL query results and observations |
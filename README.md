# customer_churn_prediction_model
This project predicts customer churn for a telecommunications provider using an XGBoost model on the Telco Customer Churn dataset. The goal is to demonstrate a full professional ML workflow: feature engineering, model tuning, and threshold calibration to optimize F1 score.

## Problem
Customer churn is expensive. If a company can identify which customers are likely to leave ahead of time, they can perform targeted retention interventions (discounts, phone calls, better plans, etc.).

This project trains a model that predicts whether a customer will churn in the next month.

## Dataset

Kaggle: Telco Customer Churn
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Rows: ~7,000 customers
Target: Churn (1 = customer leaves)

Most features are boolean or categorical (contract, phone service, paperless billing, etc.)

## Approach
- Train/Test Split, 75% train / 25% test
- Preprocessing, convert Yes/No to bools, log transform TotalCharges, one-hot encode categorical features, fill blank values
- Feature engineering, boolean combination features, aggregation features
- Feature selection, RFE Wrapper method
- Model, Gradient Boosted trees
- Hyperparameter tuning, 5-fold CV Grid search on F1 score
- Threshold tuning, improve recall

  ## Results
  Final model F1 ~ .64

  ## Tech Stack
  - Python
  - pandas / numpy / scikit-learn
  - XGBoost
  - KaggleHub
 
  ## Future improvements
  - Try CatBoost or LightGBM to compare lift vs XGBoost
  - Add feature reduction (PCA) on dummy-encoded high-cardinality categories
  - Use SHAP feature attribution to see which features really drive churn
  

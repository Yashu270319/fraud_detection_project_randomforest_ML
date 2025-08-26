# Fraud Detection System (Finance / Banking)

This project builds a Fraud Detection System using Random Forest on the famous Credit Card Fraud Detection dataset, with EDA, proper preprocessing, scaling, class imbalance dealing using SMOTE, with visualization using matplotlib

## Dataset
- Source: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- ~284,807 transactions
- Features: Time, Amount, PCA-transformed features (V1–V28)
- Target: Fraud (1) vs Legitimate (0)

## Project Workflow
1. Exploratory Data Analysis (EDA)
2. Data Preprocessing & Feature Engineering
3. Model Training (Random Forest with class weights)
4. Evaluation (Precision, Recall, F1, ROC-AUC)
5. Feature Importance (SHAP values)
6. Deployment (Streamlit App)

##  Folder Structure
See project structure in repo.

## Tech Stack
- Python (Pandas, NumPy, Scikit-learn, Imbalanced-learn)
- Random Forest Classifier
- Streamlit (for deployment)
- SHAP (explainability)

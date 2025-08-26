import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# ----------------------------
# Load Model and Data
# ----------------------------
MODEL_PATH = "data/processed/random_forest_model.pkl"
DATA_PATH = "data/raw/creditcard.csv"  # update if dataset is in different location

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        model = joblib.load(file)
    return model

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

# ----------------------------
# Streamlit App
# ----------------------------
def main():
    st.title("💳 Fraud Detection System (Random Forest)")
    st.markdown("This app loads a trained Random Forest model and applies it to the dataset directly (no manual input).")

    # Load
    model = load_model()
    df = load_data()

    st.subheader("Dataset Overview")
    st.write(df.head())
    st.write("Shape of data:", df.shape)

    # Features and target
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Predictions
    y_pred = model.predict(X)

    # ----------------------------
    # Results Summary
    # ----------------------------
    st.subheader("Fraud vs Non-Fraud Counts (Predicted)")
    pred_counts = pd.Series(y_pred).value_counts().rename({0: "Non-Fraud", 1: "Fraud"})
    st.bar_chart(pred_counts)

    # ----------------------------
    # Classification Report
    # ----------------------------
    st.subheader("Classification Report")
    report = classification_report(y, y_pred, target_names=["Non-Fraud", "Fraud"], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

    # ----------------------------
    # Confusion Matrix
    # ----------------------------
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Fraud", "Fraud"], yticklabels=["Non-Fraud", "Fraud"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.success("✅ Fraud detection completed using Random Forest on full dataset!")

if __name__ == "__main__":
    main()
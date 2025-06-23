import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st
import pandas as pd
import joblib
from drift_detector import detect_drift
import mlflow
from pipeline.mlflow_tracker import log_inference_metrics


st.title("Loan Default Prediction")

# Load trained model
model = joblib.load("model.pkl")

# Define expected column types
NUMERIC_COLS = ['Client_Income', 'Credit_Amount', 'Loan_Annuity', 'Age_Days', 'Employed_Days', 'Registration_Days', 'ID_Days', 'Score_Source_3']
CATEGORICAL_COLS = ['Client_Education', 'Client_Housing_Type', 'Client_Income_Type', 'Client_Marital_Status', 'Loan_Contract_Type', 'Client_Gender']


# Load reference data for drift detection
REFERENCE_DATA_PATH = "Data/Dataset.csv"
reference_df = pd.read_csv(REFERENCE_DATA_PATH)

def preprocess_input(input_df):
    input_df = input_df.copy()

    # Ensure numeric columns are numeric and imputed
    for col in NUMERIC_COLS:
        if col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
            median_value = input_df[col].median()
            input_df[col] = input_df[col].fillna(median_value)
        else:
            input_df[col] = 0  # Default value if column missing

    # Ensure categorical columns are string and fill missing values
    for col in CATEGORICAL_COLS:
        if col in input_df.columns:
            input_df[col] = input_df[col].astype(str).fillna("missing")
        else:
            input_df[col] = "missing"

    # Keep only expected columns
    input_df = input_df[NUMERIC_COLS + CATEGORICAL_COLS]

    return input_df

def predict(input_df):
    input_df = preprocess_input(input_df)
    probabilities = model.predict_proba(input_df)[:, 1]
    return probabilities

def log_to_mlflow(y_true, y_pred, drift_report):
    with mlflow.start_run(run_name="Streamlit_Inference"):
        report = classification_report(y_true, y_pred, output_dict=True)
        mlflow.log_metric("precision", report["1"]["precision"])
        mlflow.log_metric("recall", report["1"]["recall"])
        mlflow.log_metric("f1-score", report["1"]["f1-score"])
        mlflow.log_dict(drift_report, "drift_report.json")

# THRESHOLD = 0.3  # Lowering threshold increases recall

# def predict(input_df):
#     input_df = preprocess_input(input_df)
#     probabilities = model.predict_proba(input_df)[:, 1]
#     predictions = (probabilities >= THRESHOLD).astype(int)
#     return probabilities, predictions

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV File for Prediction", type="csv")

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    # try:
    #     # Preprocess both datasets before drift detection
    #     ref_processed = preprocess_input(reference_df)
    #     input_processed = preprocess_input(input_df)

    #     drift_report = detect_drift(ref_processed, input_processed, log_to_mlflow=False)
    #     st.subheader("Drift Detection Report")
    #     st.json(drift_report)

    #     # Run prediction
    #     probs = predict(input_df)
    #     input_df['Default_Probability'] = probs
    #     st.success("Predictions generated successfully!")
    #     st.write(input_df)
    #     if "Default" in input_df.columns:
    #         y_true = input_df["Default"]
    #         log_inference_metrics(y_true, preds, drift_report)
    # except Exception as e:
    #     st.error(f"Error in prediction: {str(e)}")
    try:
        probs = predict(input_df)
        input_df['Default_Probability'] = probs
        predictions = (probs >= 0.3).astype(int)

        y_true = input_df['Default'] if 'Default' in input_df.columns else None

        # Log inference metrics to MLflow
        drift_report = detect_drift(reference_df,input_df)
        log_inference_metrics(y_true=y_true, y_pred=predictions, drift_report=drift_report)

        if drift_report and drift_report.get("drift_detected"):
            st.warning("⚠️ Potential Data Drift Detected")
            st.json(drift_report)

        st.success("✅ Predictions generated and logged successfully!")
        st.write(input_df)

    except Exception as e:
        st.error(f"❌ Error in prediction: {str(e)}")


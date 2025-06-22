import streamlit as st
import pandas as pd
import joblib

st.title("Loan Default Prediction")

# Load trained model
model = joblib.load("model.pkl")

# Define expected column types
NUMERIC_COLS = ['Client_Income', 'Credit_Amount', 'Loan_Annuity', 'Age_Days', 'Employed_Days', 'Registration_Days', 'ID_Days', 'Score_Source_3']
CATEGORICAL_COLS = ['Client_Education', 'Client_Housing_Type', 'Client_Income_Type', 'Client_Marital_Status', 'Loan_Contract_Type', 'Client_Gender']

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

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV File for Prediction", type="csv")

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    try:
        probs = predict(input_df)
        input_df['Default_Probability'] = probs
        st.success(" Predictions generated successfully!")
        st.write(input_df)
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")

import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.pkl")
st.title("Loan Default Prediction")

def predict(input_df):
    return model.predict_proba(input_df)[:, 1]

uploaded_file = st.file_uploader("Upload CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    probs = predict(df)
    df['Default_Probability'] = probs
    st.write(df)
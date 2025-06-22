from pipeline.data_loader import load_data
from pipeline.model import build_model
from pipeline.evaluate import evaluate_model
from pipeline.feature_selection import chi2_feature_selection, anova_feature_selection
from pipeline.mlflow_tracker import log_experiment
from sklearn.model_selection import train_test_split
import pandas as pd

CATEGORICAL_COLS = [
    "Client_Gender", "Client_Marital_Status", "Client_Housing_Type",
    "Client_Income_Type", "Client_Education", "Loan_Contract_Type"
]

NUMERIC_COLS = [
    'Client_Income', 'Credit_Amount', 'Loan_Annuity', 'Age_Days',
    'Employed_Days', 'Registration_Days', 'ID_Days', 'Score_Source_3'
]
def retrain():
    df = load_data()
    X = df.drop(columns=["Default"])
    y = df["Default"]

    cat_features = chi2_feature_selection(X[CATEGORICAL_COLS], y, k=5)
    num_features = anova_feature_selection(X[NUMERIC_COLS], y, k=5)

    X = X[cat_features + num_features]
    model = build_model(num_features, cat_features)
    model.fit(X, y)
    return model
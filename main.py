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

def main():
    df = load_data()
    X = df.drop(columns=["Default"])
    y = df["Default"]

    cat_features = chi2_feature_selection(X[CATEGORICAL_COLS], y, k=5)
    X_numeric = X[NUMERIC_COLS]
    num_features = anova_feature_selection(X_numeric, y, k=5)
    selected_features = cat_features + num_features

    X = X[selected_features].copy()
    X[cat_features] = X[cat_features].astype(str).fillna("missing")

    for col in num_features:
        X[col] = pd.to_numeric(X[col], errors='coerce')
        X[col].fillna(X[col].median(), inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)

    model = build_model(num_features, cat_features)
    model.fit(X_train, y_train)
    report, auc = evaluate_model(model, X_test, y_test)
    log_experiment(model, report, auc)

if __name__ == "__main__":
    main()

from dotenv import load_dotenv
from pipeline.data_loader import load_data
from pipeline.model import build_model
from pipeline.evaluate import evaluate_model
from pipeline.feature_selection import chi2_feature_selection, anova_feature_selection,preprocess_and_select_numeric_features
# from pipeline.model2 import build_model2
from pipeline.mlflow_tracker import log_experiment, compare_and_promote_model
from sklearn.model_selection import train_test_split
from pipeline.hyperparameter_tuning import tune_hyperparameters
import pandas as pd
import argparse
import os
import numpy as np
import mlflow

# CATEGORICAL_COLS = [
#     "Client_Gender", "Client_Marital_Status", "Client_Housing_Type",
#     "Client_Income_Type", "Client_Education", "Loan_Contract_Type"
# ]

# NUMERIC_COLS = [
#     'Client_Income', 'Credit_Amount', 'Loan_Annuity', 'Age_Days',
#     'Employed_Days', 'Registration_Days', 'ID_Days', 'Score_Source_3'
# ]
load_dotenv()
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")
# mlflow.set_tracking_uri("https://dagshub.com/utkarsh.shelke03/loan_deafult_prediction.mlflow")
# def main():
#     df = load_data()
#     X = df.drop(columns=["Default"])
#     y = df["Default"]

#     cat_features = chi2_feature_selection(X[CATEGORICAL_COLS], y, k=5)
#     X_numeric = X[NUMERIC_COLS]
#     num_features = anova_feature_selection(X_numeric, y, k=5)
#     selected_features = cat_features + num_features

#     X = X[selected_features].copy()
#     X[cat_features] = X[cat_features].astype(str).fillna("missing")

#     for col in num_features:
#         X[col] = pd.to_numeric(X[col], errors='coerce')
#         X[col].fillna(X[col].median(), inplace=True)

#     X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)

#     model = build_model(num_features, cat_features)
#     model.fit(X_train, y_train)
#     report, auc = evaluate_model(model, X_test, y_test)
#     log_experiment(model, report, auc)

# if __name__ == "__main__":
#     main()

# def main():
#     df = load_data()
#     X = df.drop(columns=["Default"])
#     y = df["Default"]

#     cat_features = chi2_feature_selection(X[CATEGORICAL_COLS], y, k=5)
#     X_numeric = X[NUMERIC_COLS]
#     num_features = anova_feature_selection(X_numeric, y, k=5)
#     selected_features = cat_features + num_features

#     X = X[selected_features].copy()
#     X[cat_features] = X[cat_features].astype(str).fillna("missing")
#     for col in num_features:
#         X[col] = pd.to_numeric(X[col], errors='coerce')
#         X[col].fillna(X[col].median(), inplace=True)

#     X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)
#     model = build_model(num_features, cat_features)
#     model.fit(X_train, y_train)
#     report, auc = evaluate_model(model, X_test, y_test)
#     log_experiment(model, report, auc)

# if __name__ == "__main__":
#     main()
#  model 1 with hyperparameters


def main(run_name=None):
    df = load_data()
    X = df.drop(columns=["Default"])
    y = df["Default"]
    
    CATEGORICAL_COLS = X.select_dtypes(include="object").columns.tolist()
    NUMERIC_COLS = X.select_dtypes(include=["number"]).columns.tolist()

    # NUMERIC_COLS_1 = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = chi2_feature_selection(X[CATEGORICAL_COLS], y, k=5)
    # num_features = anova_feature_selection(X[NUMERIC_COLS], y, k=5)
    num_features = preprocess_and_select_numeric_features(df, NUMERIC_COLS)

    selected_features = cat_features + num_features

    X = X[selected_features].copy()
    X[cat_features] = X[cat_features].astype(str).fillna("missing")
    for col in num_features:
        X[col] = pd.to_numeric(X[col], errors='coerce')
        X[col].fillna(X[col].median(), inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)

    model = tune_hyperparameters(num_features, cat_features, X_train, y_train)

    report, auc = evaluate_model(model, X_test, y_test)
    log_experiment(model, report, auc)
    compare_and_promote_model(report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()
    main(run_name=args.run_name)

# XGBOOST model
# def main():
#     df = load_data()
#     X = df.drop(columns=["Default"])
#     y = df["Default"]

#     cat_features = chi2_feature_selection(X[CATEGORICAL_COLS], y, k=5)
#     num_features = anova_feature_selection(X[NUMERIC_COLS], y, k=5)

#     selected_features = cat_features + num_features

#     X = X[selected_features].copy()
#     X[cat_features] = X[cat_features].astype(str).fillna("missing")
#     for col in num_features:
#         X[col] = pd.to_numeric(X[col], errors='coerce')
#         X[col].fillna(X[col].median(), inplace=True)

#     X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)

#     model = build_model2(num_features, cat_features)
#     model.fit(X_train, y_train)

#     report, auc = evaluate_model(model, X_test, y_test)
#     log_experiment(model.best_estimator_, report, auc)

# if __name__ == "__main__":
#     main()
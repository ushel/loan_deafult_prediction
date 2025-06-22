import mlflow
import mlflow.sklearn
from pipeline.config import EXPERIMENT_NAME
import dagshub
import os
import joblib
os.environ["MLFLOW_TRACKING_USERNAME"] = "utkarshshelke"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "bdef59aa61d087c002100e05ccb84c2f94083565"


dagshub.init(repo_owner='utkarsh.shelke03', repo_name='loan_deafult_prediction', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/utkarsh.shelke03/loan_deafult_prediction.mlflow")
mlflow.set_experiment(EXPERIMENT_NAME)

def log_experiment(model, report, roc_auc):
    with mlflow.start_run():
        mlflow.log_param("classifier", type(model.named_steps['classifier']).__name__)
        mlflow.log_param("smote_strategy", "auto")
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("f1_score", report['1']['f1-score'])
        joblib.dump(model, "model.pkl")
        mlflow.log_artifact("model.pkl")
import mlflow
from dotenv import load_dotenv
import os
import dagshub

load_dotenv()

os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")
dagshub.init(repo_owner='utkarsh.shelke03', repo_name='loan_deafult_prediction', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/utkarsh.shelke03/loan_deafult_prediction.mlflow")
mlflow.set_experiment("TestAuthExperiment")

with mlflow.start_run():
    mlflow.log_param("test_param", 1)
    mlflow.log_metric("test_metric", 0.99)
    print("✅ Successfully logged!")
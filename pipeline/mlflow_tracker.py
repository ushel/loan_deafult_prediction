import mlflow
import mlflow.sklearn
from pipeline.config import EXPERIMENT_NAME
import dagshub
import os
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from drift_detector import detect_drift
from mlflow.tracking import MlflowClient
# os.environ["MLFLOW_TRACKING_USERNAME"] = "utkarsh.shelke03"
# os.environ["MLFLOW_TRACKING_PASSWORD"] = "bdef59aa61d087c002100e05ccb84c2f94083565"

# username = os.getenv("MLFLOW_TRACKING_USERNAME")
# password = os.getenv("MLFLOW_TRACKING_PASSWORD")

# os.environ["MLFLOW_TRACKING_USERNAME"] = username
# os.environ["MLFLOW_TRACKING_PASSWORD"] = password


# dagshub.init(repo_owner='utkarsh.shelke03', repo_name='loan_deafult_prediction', mlflow=True)
mlflow.set_tracking_uri("https://utkarsh.shelke03:55bbb7475062084353130c086369f4602ba5785e@dagshub.com/utkarsh.shelke03/loan_deafult_prediction.mlflow")
mlflow.set_experiment(EXPERIMENT_NAME)

# def log_experiment(model, report, roc_auc):
#     with mlflow.start_run():
#         mlflow.log_param("classifier", type(model.named_steps['classifier']).__name__)
#         mlflow.log_param("smote_strategy", "auto")
#         mlflow.log_metric("roc_auc", roc_auc)
#         mlflow.log_metric("f1_score", report['1']['f1-score'])
#         joblib.dump(model, "model.pkl")
#         mlflow.log_artifact("model.pkl")

def log_experiment(model, report, roc_auc):
    with mlflow.start_run():
        clf_name = type(model.named_steps['classifier']).__name__
        mlflow.log_param("classifier", clf_name)
        best_params = model.get_params()
        for param_name, param_val in best_params.items():
            if isinstance(param_val, (int, float, str, bool)):
                mlflow.log_param(param_name, param_val)

        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("f1_score", report['1']['f1-score'])
        mlflow.log_metric("precision", report['1']['precision'])
        mlflow.log_metric("recall", report['1']['recall'])
        # if drift_report:
        #     mlflow.log_dict(drift_report, "drift_report.json")

        joblib.dump(model, "model.pkl")
        mlflow.log_artifact("model.pkl")
        
def log_inference_metrics(y_true, y_pred, drift_report=None, model_name="LoanDefaultModel"):
    with mlflow.start_run():
        if y_true is not None:
            mlflow.log_metric("inference_precision", precision_score(y_true, y_pred))
            mlflow.log_metric("inference_recall", recall_score(y_true, y_pred))
            mlflow.log_metric("inference_f1", f1_score(y_true, y_pred))

        if drift_report:
            mlflow.log_dict(drift_report, "inference_drift_report.json")

        joblib.dump(y_pred, "predictions.pkl")
        mlflow.log_artifact("predictions.pkl")

def compare_and_promote_model(new_report):
    client = MlflowClient()
    model_name = "LoanDefaultModel"

    try:
        prod_versions = [mv for mv in client.get_latest_versions(model_name, stages=["Production"])]
        if not prod_versions:
            print("No production model found. Promoting new model.")
            _register_and_promote_latest_run(model_name)
            return

        prod_run_id = prod_versions[0].run_id
        prod_run = client.get_run(prod_run_id)
        old_recall = float(prod_run.data.metrics.get("recall", 0))

        new_recall = new_report["recall"]
        if new_recall > old_recall:
            print(f"New model recall ({new_recall}) is better than production ({old_recall}). Promoting new model.")
            _register_and_promote_latest_run(model_name)
        else:
            print(f"New model recall ({new_recall}) did not improve over production ({old_recall}).")

    except Exception as e:
        print(f"Error during model promotion check: {str(e)}")

def _register_and_promote_latest_run(model_name="LoanDefaultModel"):
    client = MlflowClient()
    
    # Ensure active run exists
    run = mlflow.active_run()
    if not run:
        raise Exception("No active MLflow run found. Please start a run using `mlflow.start_run()`.")

    # Log and register the model
    model_uri = f"runs:/{run.info.run_id}/model"
    
    print(f"Registering model from URI: {model_uri}")
    result = mlflow.register_model(model_uri, model_name)
    
    # Optional: Wait until model is registered (depends on backend, avoid premature transition)
    import time
    for _ in range(10):  # wait up to 10 seconds
        model_version_details = client.get_model_version(name=model_name, version=result.version)
        if model_version_details.status == "READY":
            break
        time.sleep(1)
    
    # Promote the newly registered model to Production
    client.transition_model_version_stage(
        name=model_name,
        version=result.version,
        stage="Production",
        archive_existing_versions=True
    )
    
    print(f" Model '{model_name}' version {result.version} promoted to Production.")
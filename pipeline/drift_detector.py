import pandas as pd
from scipy.stats import ks_2samp
import mlflow

def detect_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame, threshold: float = 0.05):
    """
    Compare the reference (training) and current datasets to detect data drift using the Kolmogorov-Smirnov test.

    Parameters:
    - reference_df: pd.DataFrame, reference dataset (e.g., training data)
    - current_df: pd.DataFrame, current dataset (e.g., new inference data)
    - threshold: float, p-value threshold to flag drift (default = 0.05)

    Returns:
    - drift_report: dict, mapping feature -> drift_detected (True/False) and p_value
    """
    drift_report = {}
    for col in reference_df.columns:
        try:
            ref_col = pd.to_numeric(reference_df[col], errors='coerce').dropna()
            cur_col = pd.to_numeric(current_df[col], errors='coerce').dropna()

            if ref_col.empty or cur_col.empty:
                continue

            stat, p_value = ks_2samp(ref_col, cur_col)
            drift_report[col] = {
                "drift_detected": p_value < threshold,
                "p_value": round(p_value, 4)
            }
        except Exception as e:
            drift_report[col] = {
                "error": str(e)
            }

    
    if log_to_mlflow:
        mlflow.log_dict(drift_report, "drift_report.json")

    return drift_report

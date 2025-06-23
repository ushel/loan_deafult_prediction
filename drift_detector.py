import pandas as pd
from scipy.stats import ks_2samp
from sklearn.preprocessing import LabelEncoder

def detect_drift(reference_df, current_df):
    drift_report = {}

    common_cols = [col for col in reference_df.columns if col in current_df.columns]
    for col in common_cols:
        ref_col = reference_df[col]
        curr_col = current_df[col]

        if pd.api.types.is_numeric_dtype(ref_col):
            stat, p_value = ks_2samp(ref_col.dropna(), curr_col.dropna())
            drift_report[col] = {
                "type": "numerical",
                "p_value": p_value,
                "drift_detected": p_value < 0.05
            }
        else:
            le = LabelEncoder()
            try:
                ref_enc = le.fit_transform(ref_col.astype(str))
                curr_enc = le.transform(curr_col.astype(str))
                stat, p_value = ks_2samp(ref_enc, curr_enc)
                drift_report[col] = {
                    "type": "categorical",
                    "p_value": p_value,
                    "drift_detected": p_value < 0.05
                }
            except:
                drift_report[col] = {
                    "type": "categorical",
                    "p_value": None,
                    "drift_detected": False,
                    "error": "Label mismatch"
                }

    return drift_report
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from pipeline.preprocessing import get_preprocessor
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek

def build_model(numeric_cols, categorical_cols,sampler_strategy="SMOTETomek"):
    preprocessor = get_preprocessor(numeric_cols, categorical_cols)
    classifier = RandomForestClassifier(random_state=42,class_weight="balanced")
    
    if sampler_strategy == "SMOTEENN":
        sampler = SMOTEENN(random_state=42)
    elif sampler_strategy == "SMOTETomek":
        sampler = SMOTETomek(random_state=42)
    elif sampler_strategy == "TomekLinks":
        sampler = TomekLinks()
    else:
        raise ValueError("Unsupported sampler strategy. Choose from 'SMOTEENN', 'SMOTETomek', or 'TomekLinks'.")

    model = ImbPipeline(steps=[
        ("preprocessing", preprocessor),
        ("sampler", sampler),
        ("classifier", classifier)
    ])
    return model

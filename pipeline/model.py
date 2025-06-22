from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from pipeline.preprocessing import get_preprocessor

def build_model(numeric_cols, categorical_cols):
    preprocessor = get_preprocessor(numeric_cols, categorical_cols)
    classifier = RandomForestClassifier(random_state=42)

    model = ImbPipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])
    return model

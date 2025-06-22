from xgboost import XGBClassifier
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from pipeline.preprocessing import get_preprocessor

def build_model2(numeric_cols, categorical_cols):
    preprocessor = get_preprocessor(numeric_cols, categorical_cols)

    base_pipeline = ImbPipeline([
        ("preprocessor", preprocessor),
        ("smoteenn", SMOTEENN()),
        ("classifier", XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
    ])

    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [3, 5, 7],
        'classifier__learning_rate': [0.01, 0.1, 0.2]
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid = GridSearchCV(base_pipeline, param_grid, scoring='roc_auc', cv=cv, n_jobs=-1, verbose=1)
    return grid
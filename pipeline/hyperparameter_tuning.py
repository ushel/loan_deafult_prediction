from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.pipeline import Pipeline as ImbPipeline
from pipeline.preprocessing import get_preprocessor
from imblearn.combine import SMOTEENN, SMOTETomek

def tune_hyperparameters(numeric_cols, categorical_cols, X_train, y_train):
    preprocessor = get_preprocessor(numeric_cols, categorical_cols)
    clf = RandomForestClassifier(random_state=42, class_weight='balanced')
    sampler = SMOTETomek(random_state=42)

    pipeline = ImbPipeline([
        ("preprocessor", preprocessor),
        ("sampler", sampler),
        ("classifier", clf)
    ])

    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5]
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='recall', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

# from sklearn.model_selection import GridSearchCV
# from sklearn.ensemble import RandomForestClassifier

# def tune_hyperparameters(numeric_cols, categorical_cols, X_train, y_train):
#     classifier = RandomForestClassifier(random_state=42, class_weight="balanced")
#     model = build_model(numeric_cols, categorical_cols, classifier=classifier, sampler_strategy="SMOTEENN")

#     param_grid = {
#         'classifier__n_estimators': [100, 200],
#         'classifier__max_depth': [None, 10, 20],
#         'classifier__min_samples_split': [2, 5]
#     }

#     grid_search = GridSearchCV(model, param_grid, cv=3, scoring='recall', n_jobs=-1)
#     grid_search.fit(X_train, y_train)
#     return grid_search.best_estimator_

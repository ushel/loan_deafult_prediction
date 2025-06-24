import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
def chi2_feature_selection(X, y, k=10):
    X = X.copy()
    cat_cols = X.select_dtypes(include='object').columns
    X[cat_cols] = X[cat_cols].fillna("missing")
    X[cat_cols] = X[cat_cols].astype(str)
    for col in cat_cols:
        X[col] = LabelEncoder().fit_transform(X[col])

    selector = SelectKBest(chi2, k=k)
    selector.fit(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    return selected_features

def anova_feature_selection(X, y, k=10):
    X = X.copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
        X[col].fillna(X[col].median(), inplace=True)

    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    return selected_features


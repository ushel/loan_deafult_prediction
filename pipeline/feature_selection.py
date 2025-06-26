import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import winsorize

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

def preprocess_and_select_numeric_features(df, numerical, winsor_limits=[0.01, 0.01], corr_threshold=0.9):
    df = df.copy()

    for col in numerical:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert strings to NaN
        df[col].fillna(df[col].median(), inplace=True)     # Fill NaNs
        df[col] = winsorize(df[col], limits=winsor_limits) # Winsorize
        
        # Correlation matrix
        corr_matrix = df[numerical].corr()

        # plt.figure(figsize=(12, 10))
        # sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', annot_kws={"size": 8})
        # plt.title("Numerical Feature Correlation", fontsize=14)
        # plt.xticks(rotation=45, ha='right')
        # plt.yticks(rotation=0)
        # plt.tight_layout()
        # plt.show()

        # Remove highly correlated features above the threshold
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col].abs() > corr_threshold)]
    NUMERIC_COLS = [col for col in numerical if col not in to_drop]

    print("Remaining numeric features:", NUMERIC_COLS)
    return NUMERIC_COLS
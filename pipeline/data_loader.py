import pandas as pd
from config import DATA_FILE

def load_dataset(path="Data/Dataset.csv"):
    df = pd.read_csv(path, low_memory=False)
    df["Default"] = pd.to_numeric(df["Default"], errors="coerce")
    df = df.dropna(subset=["Default"])
    return df
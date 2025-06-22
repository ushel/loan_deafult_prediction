import pandas as pd
from pipeline.config import DATA_PATH

def load_data():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = df[df["Default"].notnull()]
    df["Default"] = df["Default"].astype(int)
    return df
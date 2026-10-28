from pathlib import Path
import pandas as pd

RAW = Path("data/raw")
PROCESSED = Path("data/processed")

TARGET = "Activity"
GROUP = "subject"

def load_row():
    train=pd.read_csv(RAW/"train.csv")
    test=pd.read_csv(RAW/"test.csv")
    return train,test


def load_clean():
    train=pd.read_csv(PROCESSED/"train_clean.csv")
    test=pd.read_csv(PROCESSED/"test_clean.csv")
    print(train.shape)
    return train,test
def split_xy(df):
    X=df.drop(columns=[TARGET,GROUP])
    Y=df[TARGET]
    subjects=df[GROUP]
    return X,Y,subjects
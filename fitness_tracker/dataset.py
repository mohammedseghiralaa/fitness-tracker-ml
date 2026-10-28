import pandas as pd
from fitness_tracker.src.load_data import GROUP,PROCESSED,TARGET,load_row


def audit(df,name):
    print(name)
    print("shape",df.shape)
    print("missing values",df.isna().sum())

    print("duplicate",df.duplicated().sum())
    print("non numeric",df.select_dtypes(exclude="number"))
    print(df[TARGET].value_counts())


def main():
    train,test=load_row()
    audit(train,"train")
    audit(test,"test")


if __name__ == "__main__":
    main()

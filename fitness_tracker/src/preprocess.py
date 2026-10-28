import re
from fitness_tracker.src.load_data import load_row, PROCESSED, TARGET, GROUP


def clean_columns(df):
    """Make column names safe: no (), -, commas; ensure unique."""
    new_cols = []
    seen = {}
    for c in df.columns:
        name = re.sub(r"[^0-9a-zA-Z]+", "_", c).strip("_")
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        new_cols.append(name)
    df.columns = new_cols
    return df


def clean(df):
    df = df.copy()
    # keep the names of target and group columns unchanged
    meta = df[[TARGET, GROUP]]
    features = clean_columns(df.drop(columns=[TARGET, GROUP]))
    df = features.join(meta)

    df = df.drop_duplicates()
    df = df.dropna()
    df[TARGET] = df[TARGET].str.strip().str.upper()
    return df


if __name__ == "__main__":
    PROCESSED.mkdir(parents=True, exist_ok=True)
    train, test = load_row()

    train_c = clean(train)
    test_c = clean(test)

    print("train:", train.shape, "->", train_c.shape)
    print("test: ", test.shape, "->", test_c.shape)
    print(train_c[TARGET].value_counts())

    train_c.to_csv(PROCESSED / "train_clean.csv", index=False)
    test_c.to_csv(PROCESSED / "test_clean.csv", index=False)
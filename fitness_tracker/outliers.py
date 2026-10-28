from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from fitness_tracker.load_data import GROUP, TARGET, load_clean, split_xy

FIG = Path("reports/figures")
FIG.mkdir(parents=True, exist_ok=True)


def iqr_outliers(X, k=3.0):
    """Per-feature: values outside [Q1 - k*IQR, Q3 + k*IQR]. Returns bool DataFrame."""
    q1, q3 = X.quantile(0.25), X.quantile(0.75)
    iqr = q3 - q1
    return (X < q1 - k * iqr) | (X > q3 + k * iqr)


def zscore_outliers(X, thr=4.0):
    z = (X - X.mean()) / (X.std() + 1e-9)
    return z.abs() > thr


def isolation_outliers(X, contamination=0.02):
    """Multivariate: whole-row outliers. Returns bool array (True = outlier)."""
    Xs = StandardScaler().fit_transform(X)
    iso = IsolationForest(
        n_estimators=300, contamination=contamination, random_state=42, n_jobs=-1
    )
    return iso.fit_predict(Xs) == -1

def iqr_within_class(X, y, k=3.0):
    mask = pd.DataFrame(False, index=X.index, columns=X.columns)
    for act in y.unique():
        idx = y == act
        q1, q3 = X[idx].quantile(0.25), X[idx].quantile(0.75)
        iqr = q3 - q1
        mask.loc[idx] = (X[idx] < q1 - k * iqr) | (X[idx] > q3 + k * iqr)
    return mask

def main():
    train, _ = load_clean()
    X, y, s = split_xy(train)

    # 1) Univariate: IQR and z-score
    iqr_mask = iqr_outliers(X)
    z_mask = zscore_outliers(X)
    print(f"IQR (k=3): {iqr_mask.values.mean():.2%} of all cells are outliers")
    print(f"Z-score (>4): {z_mask.values.mean():.2%} of all cells are outliers")

    row_rate = iqr_mask.mean(axis=1)  # share of features that are outliers per row
    print("Rows with >5% outlier features:", int((row_rate > 0.05).sum()))

    # 2) Multivariate: Isolation Forest
    iso_mask = isolation_outliers(X)
    print(f"Isolation Forest: {iso_mask.sum()} rows flagged ({iso_mask.mean():.2%})")

    # 3) Where are they? By activity and by subject
    info = pd.DataFrame({TARGET: y.values, GROUP: s.values, "outlier": iso_mask})
    by_act = info.groupby(TARGET)["outlier"].mean().sort_values(ascending=False)
    by_sub = info.groupby(GROUP)["outlier"].mean().sort_values(ascending=False)
    print("\nOutlier rate by activity:\n", by_act.round(3))
    print("\nTop subjects by outlier rate:\n", by_sub.head(5).round(3))

    # 4) Plots
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    sns.barplot(x=by_act.values, y=by_act.index, ax=ax[0])
    ax[0].set_title("Isolation Forest outlier rate by activity")
    sns.barplot(x=by_sub.index.astype(str), y=by_sub.values, ax=ax[1])
    ax[1].set_title("Outlier rate by subject")
    ax[1].tick_params(axis="x", rotation=90)
    plt.tight_layout()
    plt.savefig(FIG / "07_outliers.png", dpi=150)
    plt.close()

    top_feats = iqr_mask.mean().nlargest(8).index
    plt.figure(figsize=(12, 5))
    sns.boxplot(data=X[top_feats], orient="h")
    plt.title("Features with the most IQR outliers")
    plt.tight_layout()
    plt.savefig(FIG / "08_outlier_features.png", dpi=150)
    plt.close()

    # 5) Save flagged rows for inspection (do NOT delete yet)
    flagged = train.loc[iso_mask, [GROUP, TARGET]]
    flagged.to_csv("reports/flagged_outliers.csv", index=True)
    print("\nSaved reports/flagged_outliers.csv")

    mask = iqr_within_class(X, y)
    print(f"Within-class outliers: {mask.values.mean():.2%}")


   


if __name__ == "__main__":
    main()
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from fitness_tracker.load_data import GROUP, TARGET, load_clean, split_xy

FIG = Path("reports/figures")
FIG.mkdir(parents=True, exist_ok=True)


def class_balance(train, test):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for a, df, t in zip(ax, (train, test), ("Train", "Test")):
        order = df[TARGET].value_counts().index
        sns.countplot(data=df, y=TARGET, order=order, ax=a)
        a.set_title(f"{t} class distribution")
    plt.tight_layout()
    plt.savefig(FIG / "01_class_balance.png", dpi=150)
    plt.close()


def samples_per_subject(train):
    counts = train.groupby([GROUP, TARGET]).size().unstack(fill_value=0)
    counts.plot(kind="bar", stacked=True, figsize=(12, 5))
    plt.title("Samples per subject and activity (train)")
    plt.tight_layout()
    plt.savefig(FIG / "02_samples_per_subject.png", dpi=150)
    plt.close()


def top_features_by_activity(X, y, n=6):
    """Pick features that vary most between activities (ANOVA F-score)."""
    from sklearn.feature_selection import f_classif

    f, _ = f_classif(X, y)
    top = pd.Series(f, index=X.columns).nlargest(n).index
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for a, col in zip(axes.ravel(), top):
        sns.boxplot(data=pd.concat([X[col], y], axis=1), x=TARGET, y=col, ax=a)
        a.tick_params(axis="x", rotation=60)
        a.set_title(col[:35])
    plt.tight_layout()
    plt.savefig(FIG / "03_top_features.png", dpi=150)
    plt.close()
    return list(top)


def correlation_report(X):
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    high = [c for c in upper.columns if (upper[c] > 0.95).any()]
    print(f"Features with |corr| > 0.95 to another feature: {len(high)} / {X.shape[1]}")
    sample = X.iloc[:, :40].corr()
    plt.figure(figsize=(9, 7))
    sns.heatmap(sample, cmap="coolwarm", center=0)
    plt.title("Correlation (first 40 features)")
    plt.tight_layout()
    plt.savefig(FIG / "04_correlation.png", dpi=150)
    plt.close()
    return high


def pca_plot(X, y):
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(Xs)
    print("PCA explained variance (2 comps):", pca.explained_variance_ratio_.round(3))
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=Z[:, 0], y=Z[:, 1], hue=y.values, s=12, alpha=0.7)
    plt.title("PCA of train features")
    plt.tight_layout()
    plt.savefig(FIG / "05_pca.png", dpi=150)
    plt.close()


def subject_variability(X, y, subjects, feature):
    """Does the same activity look different across people?"""
    df = pd.DataFrame({feature: X[feature], "subject": subjects, TARGET: y})
    plt.figure(figsize=(12, 5))
    sns.boxplot(data=df[df[TARGET] == "WALKING"], x="subject", y=feature)
    plt.title(f"WALKING: {feature[:40]} by subject")
    plt.tight_layout()
    plt.savefig(FIG / "06_subject_variability.png", dpi=150)
    plt.close()


def main():
    train, test = load_clean()
    X, y, s = split_xy(train)

    class_balance(train, test)
    samples_per_subject(train)
    top = top_features_by_activity(X, y)
    print("Top discriminative features:", top)
    high = correlation_report(X)
    pca_plot(X, y)
    subject_variability(X, y, s, top[0])

    # distribution shift between train and test
    Xt, _, _ = split_xy(test)
    shift = (Xt.mean() - X.mean()).abs() / (X.std() + 1e-9)
    print("Largest train/test mean shifts (in std units):")
    print(shift.nlargest(5).round(2))

    pd.Series(high, name="redundant_features").to_csv(
        "reports/redundant_features.csv", index=False
    )
    print("Figures saved to reports/figures/")


if __name__ == "__main__":
    main()
    